"""CuPy fused trajectory metadata kernel for GFN sampling.

The CT backend already applies Clifford gates in batched CUDA kernels. This
module fuses the remaining GFN bookkeeping into one raw CUDA kernel — avoiding a
chain of PyTorch advanced-indexing kernels with dynamic-shape ``nonzero`` steps —
and keeps a safe Python fallback for CPU / missing-CuPy environments.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Optional

import torch

try:
    from .cp_stream import current_external_stream
except ImportError:  # pragma: no cover - direct-execution mode
    from cp_stream import current_external_stream


# Recoverable (transient) OOM exception types, mirroring
# ``fused_apply_adapter._RECOVERABLE_OOM_TYPES``. ``torch.cuda.OutOfMemoryError``
# is a ``RuntimeError`` subclass on modern torch; resolve it via ``getattr`` so
# the ``except`` tuple in ``apply_metadata_kernel`` can never itself raise
# ``AttributeError`` on a torch build that lacks the class (which would mask the
# real launch failure instead of latching it).
_TORCH_CUDA_OOM = getattr(getattr(torch, "cuda", None), "OutOfMemoryError", None)
_RECOVERABLE_OOM_TYPES = (
    (MemoryError,) if _TORCH_CUDA_OOM is None else (MemoryError, _TORCH_CUDA_OOM)
)


def _is_recoverable_oom(exc: BaseException) -> bool:
    """True for a transient allocation/OOM failure (soft fallback, NOT a latch).

    Mirrors ``fused_apply_adapter._is_recoverable_oom``: covers ``MemoryError``
    (host-side), ``torch.cuda.OutOfMemoryError`` when present (modern torch's
    ``RuntimeError`` subclass), and the legacy ``RuntimeError("... out of
    memory...")`` form some older / minimal PyTorch builds raise. Keeping the
    same predicate as the sibling fused kernels keeps the soft-vs-persistent
    failure boundary aligned across the family (``torch`` is unpinned, so the
    legacy form still matters).
    """
    if isinstance(exc, _RECOVERABLE_OOM_TYPES):
        return True
    if isinstance(exc, RuntimeError):
        msg = str(exc).lower()
        return "out of memory" in msg or "cuda error: out of memory" in msg
    return False


_KERNEL_SRC = r'''
extern "C" __global__
void gfn_update_metadata(
    const long long* __restrict__ actions,
    bool* __restrict__ active,
    bool* __restrict__ tableau_active,
    bool* __restrict__ terminated,
    long long* __restrict__ circuit_depths,
    bool* __restrict__ current_layer_qubits,
    long long* __restrict__ qubit_last_layer,
    long long* __restrict__ last_single_qubit_gates,
    long long* __restrict__ qubit_last_use_step,
    long long* __restrict__ action_qubits,
    const long long* __restrict__ action_gate_types,
    const long long* __restrict__ action_qubit1,
    const long long* __restrict__ action_qubit2,
    const bool* __restrict__ action_is_single,
    const bool* __restrict__ action_is_two,
    const int total,
    const int n_qubits,
    const int max_length,
    const int terminal_index,
    const int step
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }

    terminated[idx] = false;
    if (!active[idx]) {
        return;
    }

    const long long action = actions[idx];
    if (action == terminal_index) {
        terminated[idx] = true;
        active[idx] = false;
        tableau_active[idx] = false;
        return;
    }

    const long long gate_type = action_gate_types[action];
    const bool is_single = action_is_single[action];
    const bool is_two = action_is_two[action];
    const int q1 = (int) action_qubit1[action];
    const int q2 = (int) action_qubit2[action];
    const int q_base = idx * n_qubits;

    // strict 2q-only depth: 1q gates never consume depth budget and
    // never touch `current_layer_qubits` (which now tracks only the qubits
    // occupied by 2q gates in the current 2q layer). A 2q gate opens a new
    // layer when either of its qubits is already busy in the current 2q
    // layer OR when no 2q layer has been opened yet (circuit_depths == 0).
    // The depth==0 case bumps depth 0 -> 1 on the first 2q gate so that
    // depth k means "k 2q layers have been opened".
    bool needs_new_layer = false;
    if (is_two) {
        needs_new_layer =
            current_layer_qubits[q_base + q1] ||
            current_layer_qubits[q_base + q2] ||
            circuit_depths[idx] == 0;
    }

    if (needs_new_layer) {
        circuit_depths[idx] += 1;
        for (int q = 0; q < n_qubits; ++q) {
            current_layer_qubits[q_base + q] = false;
        }
    }
    const long long current_depth = circuit_depths[idx];

    if (is_single) {
        if (step >= 0) {
            qubit_last_use_step[q_base + q1] = step;
            action_qubits[(idx * max_length + step) * 2] = q1;
        }
        last_single_qubit_gates[q_base + q1] = gate_type;
        qubit_last_layer[q_base + q1] = current_depth;
    } else if (is_two) {
        if (step >= 0) {
            qubit_last_use_step[q_base + q1] = step;
            qubit_last_use_step[q_base + q2] = step;
            action_qubits[(idx * max_length + step) * 2] = q1;
            action_qubits[(idx * max_length + step) * 2 + 1] = q2;
        }
        last_single_qubit_gates[q_base + q1] = -1;
        last_single_qubit_gates[q_base + q2] = -1;
        current_layer_qubits[q_base + q1] = true;
        current_layer_qubits[q_base + q2] = true;
        qubit_last_layer[q_base + q1] = current_depth;
        qubit_last_layer[q_base + q2] = current_depth;
    }
}
'''


@lru_cache(maxsize=8)
def _metadata_kernel(cuda_index: int):
    import cupy as cp

    with cp.cuda.Device(cuda_index):
        return cp.RawKernel(_KERNEL_SRC, "gfn_update_metadata")


# Shared DLPack torch->cupy view; centralized in _kernel_runtime.py.
try:
    from ._kernel_runtime import cp_from_torch as _cp_from_torch
except ImportError:  # pragma: no cover - direct-execution mode
    from _kernel_runtime import cp_from_torch as _cp_from_torch


# Fail-once latch, mirroring the sibling fused kernels (sampling_kernel,
# mask_counts_kernel,...). Set ONLY on a host-state failure that will not
# resolve mid-process — a CuPy import / shared-library load failure
# (ImportError / OSError) or an NVRTC compile / RawKernel launch failure. Once
# latched, ``apply_metadata_kernel`` short-circuits without re-attempting
# ``import cupy`` every per-step call (the bug this fixes: on a CUDA host that
# lacks CuPy, the metadata path re-ran ``import cupy`` on every step). Per-call
# shape bails (CPU tensors, empty batch, shape mismatch) do NOT latch.
_persistent_failure: bool = False


def fused_kernel_persistently_unavailable() -> bool:
    """``True`` once a CuPy import / NVRTC compile / launch failure has been
    observed for this process, so callers can stop paying retry overhead.

    The latch is process-global (one GPU per process in FlowMeas training).
    """
    return _persistent_failure


def reset_persistent_failure() -> None:
    """Clear the persistent-failure latch (remediated host state, or test isolation)."""
    global _persistent_failure
    _persistent_failure = False


def apply_metadata_kernel(
    actions: torch.Tensor,
    trajectory_batch,
    batched_tableau,
    action_gate_types: torch.Tensor,
    action_qubit1: torch.Tensor,
    action_qubit2: torch.Tensor,
    action_is_single: torch.Tensor,
    action_is_two: torch.Tensor,
    terminal_index: int,
    step: Optional[int],
) -> Optional[torch.Tensor]:
    """Try the fused CUDA metadata update.

    Returns the ``terminated`` tensor on success. Returns ``None`` when the
    fused path is unavailable so callers can fall back to the PyTorch path.
    """
    global _persistent_failure
    if actions.device.type != "cuda":
        return None
    # Already-latched host-state failure: short-circuit without re-importing
    # CuPy on every per-step call.
    if _persistent_failure:
        return None

    try:
        import cupy as cp
    except (ImportError, OSError):
        # CuPy absent (ImportError) or its shared libs unloadable (OSError,
        # e.g. ``libnvrtc.so.x not found``) — neither resolves mid-process, so
        # latch. Narrow except: a genuine bug *inside* cupy must propagate, not
        # be silently swallowed as "cupy unavailable".
        _persistent_failure = True
        return None

    cuda_index = actions.device.index
    if cuda_index is None:
        cuda_index = torch.cuda.current_device()

    batch_size, n_measurements = actions.shape
    total = batch_size * n_measurements
    if total == 0:
        # ``empty_like`` returns uninitialized memory — for a bool dtype
        # that means arbitrary True/False entries leak into ``terminated``
        # and the caller silently believes random trajectories ended.
        # ``zeros_like`` preserves the documented "no trajectories
        # terminated on an empty batch" semantics.
        return torch.zeros_like(actions, dtype=torch.bool)

    if (
        hasattr(trajectory_batch, "_terminated_buffers")
        and trajectory_batch._terminated_buffers[0].shape == (batch_size, n_measurements)
    ):
        idx = trajectory_batch._terminated_buffer_idx
        trajectory_batch._terminated_buffer_idx = 1 - idx
        terminated = trajectory_batch._terminated_buffers[idx]
    else:
        terminated = torch.empty((batch_size, n_measurements), dtype=torch.bool, device=actions.device)

    tableau_active = getattr(batched_tableau, "active", trajectory_batch.active)
    if tableau_active.shape != trajectory_batch.active.shape:
        return None

    # Prepare and validate all caller-owned tensors before entering the
    # host-failure latch boundary. Contract/programming errors here must
    # propagate to the caller rather than being misclassified as a permanent
    # CuPy/NVRTC failure and disabling the kernel process-wide. Allocation OOM
    # remains a soft per-call fallback.
    try:
        actions_flat = (
            actions.to(device=actions.device, dtype=torch.long)
            .contiguous()
            .view(-1)
        )
        active_flat = trajectory_batch.active.view(-1)
        tableau_active_flat = tableau_active.view(-1)
        terminated_flat = terminated.view(-1)
        circuit_depths_flat = trajectory_batch.circuit_depths.view(-1)
        current_layer_qubits_flat = (
            trajectory_batch.current_layer_qubits.view(-1)
        )
        qubit_last_layer_flat = trajectory_batch.qubit_last_layer.view(-1)
        last_single_qubit_gates_flat = (
            trajectory_batch.last_single_qubit_gates.view(-1)
        )
        qubit_last_use_step_flat = (
            trajectory_batch.qubit_last_use_step.view(-1)
        )
        action_qubits_flat = trajectory_batch.action_qubits.view(-1)
        action_gate_types_cuda = action_gate_types.to(
            device=actions.device, dtype=torch.long
        ).contiguous()
        action_qubit1_cuda = action_qubit1.to(
            device=actions.device, dtype=torch.long
        ).contiguous()
        action_qubit2_cuda = action_qubit2.to(
            device=actions.device, dtype=torch.long
        ).contiguous()
        action_is_single_cuda = action_is_single.to(
            device=actions.device, dtype=torch.bool
        ).contiguous()
        action_is_two_cuda = action_is_two.to(
            device=actions.device, dtype=torch.bool
        ).contiguous()
    except Exception as exc:
        if _is_recoverable_oom(exc):
            return None
        raise

    # PyTorch <-> CuPy stream handoff: the kernel below mutates tensors PyTorch
    # reads immediately on return, so launching on the CuPy default stream while
    # PyTorch uses its own would race. Wrapping PyTorch's current stream as a CuPy
    # ExternalStream issues the launch onto exactly that stream, making ordering
    # implicit with no cross-stream sync. ``ExternalStream(0)`` is the legacy
    # null stream; ``cp_stream.current_external_stream`` carries the real
    # current-stream pointer.
    try:
        with cp.cuda.Device(cuda_index):
            with current_external_stream(cuda_index):
                kernel = _metadata_kernel(cuda_index)
                block = 256
                grid = ((total + block - 1) // block,)
                # Keep references to the CuPy views (via _cp_from_torch) alive
                # in this scope until the launch returns. The kernel records
                # its dependencies onto the stream; the underlying torch
                # tensors stay alive through the caller's references.
                kernel_args = (
                    _cp_from_torch(actions_flat),
                    _cp_from_torch(active_flat),
                    _cp_from_torch(tableau_active_flat),
                    _cp_from_torch(terminated_flat),
                    _cp_from_torch(circuit_depths_flat),
                    _cp_from_torch(current_layer_qubits_flat),
                    _cp_from_torch(qubit_last_layer_flat),
                    _cp_from_torch(last_single_qubit_gates_flat),
                    _cp_from_torch(qubit_last_use_step_flat),
                    _cp_from_torch(action_qubits_flat),
                    _cp_from_torch(action_gate_types_cuda),
                    _cp_from_torch(action_qubit1_cuda),
                    _cp_from_torch(action_qubit2_cuda),
                    _cp_from_torch(action_is_single_cuda),
                    _cp_from_torch(action_is_two_cuda),
                    total,
                    int(trajectory_batch.n_qubits),
                    int(trajectory_batch.max_length),
                    int(terminal_index),
                    -1 if step is None else int(step),
                )
                kernel(grid, (block,), kernel_args)
                # ``kernel_args`` lives until the with-block exits — the CuPy
                # launch is queued on the torch stream, so PyTorch's
                # subsequent reads on the same stream see the writes in order.
    except Exception as exc:
        if _is_recoverable_oom(exc):
            # Transient allocation failure while creating DLPack views or
            # launching. Recoverable -- memory may free up -- so fall back
            # softly for THIS call WITHOUT tripping the process-global latch.
            return None
        if isinstance(
            exc,
            (
                TypeError,
                ValueError,
                AttributeError,
                IndexError,
                BufferError,
                AssertionError,
            ),
        ):
            # These indicate an invalid caller contract or a programming/API
            # mismatch, not a persistent host capability failure. Surface the
            # defect and keep the process latch clear so a later valid call is
            # not silently forced onto the fallback.
            raise
        # Genuine NVRTC compile / RawKernel launch failure: a host-state failure
        # that won't resolve mid-process (and CuPy/NVRTC errors are not a clean
        # exception subclass). Latch so the next call short-circuits instead of
        # re-attempting the compile/launch.
        _persistent_failure = True
        return None

    return terminated
