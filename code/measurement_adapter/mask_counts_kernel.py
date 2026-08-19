"""CuPy fused active-mask + forward/backward valid-count kernel for GFN.

The dynamic-active sampler's per-step bookkeeping calls three
``MaskingEngine`` helpers in sequence — ``compute_action_masks_active_gpu``,
``compute_forward_valid_counts_gpu``, ``compute_backward_valid_counts_gpu`` —
which together account for ~32% of ``sample_trajectories`` time on the H2O
envelope and issue ~15-20 CUDA launches per sampled step. The predicates read
exactly the same trajectory state and most of the same per-action LUTs; only
the output shape differs.

This module fuses all three into one CuPy ``RawKernel`` that walks each
``(batch, meas)`` row once and emits, for each row:

* one row of the ``(n_active, num_actions)`` active mask, but only when this
  ``(batch, meas)`` is in the ``indices`` tensor (otherwise the mask row is
  left untouched).
* one entry of the ``(B, M)`` forward valid-count.
* one entry of the ``(B, M)`` backward valid-count, with the existing
  ``exposed > 0 ? exposed: forward_count`` fallback baked in.

The PyTorch fallback keeps the original three-call path so CPU / missing-CuPy
environments behave identically. The fused and fallback paths are bit-for-bit
identical.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Optional, Tuple

import torch

try:
    from .cp_stream import current_external_stream
except ImportError:  # pragma: no cover - direct-execution mode
    from cp_stream import current_external_stream


# One CUDA thread = one ``(batch, meas)`` row. Each thread:
#   1. Loops over single- and two-qubit actions applying the forward predicate to
#      count valid forward actions, writing per-action validity into the active
#      mask when the row appears in ``indices_lookup``.
#   2. Sets the terminal entry of the active-mask row.
#   3. Loops again to count backward-exposed gates.
#   4. Stores ``backward = exposed > 0 ? exposed : forward``.
#
# Index conventions (host-side; the kernel just reads pointers + ints):
#   * ``mask`` is ``(n_active, num_actions)`` row-major; the row for ``(b, m)`` is
#     ``active_lookup[b * n_meas + m]``, and -1 means do NOT write its mask.
#   * ``forward_counts`` / ``backward_counts`` are ``(B, M)`` row-major.
#   * ``actions_time_major`` is ``(max_length, B, M)`` row-major.
#   * ``last_single_qubit_gates``, ``qubit_last_use_step``,
#     ``current_layer_qubits`` are ``(B, M, n_q)`` row-major.
#
# Per-thread state is tiny and global reads are coalesced across a warp, since
# adjacent rows have adjacent ``(b, m)`` ids and every thread reads the same
# action index in lock-step. Nothing is staged into shared memory — the per-action
# LUTs live in global memory and the L1/texture cache services them well.

_KERNEL_SRC = r'''
extern "C" __global__
void gfn_mask_counts_fused(
    // Per-row trajectory state (B, M, n_q)
    const long long* __restrict__ last_single_qubit_gates,
    const long long* __restrict__ qubit_last_use_step,
    const bool*      __restrict__ current_layer_qubits,
    // Per-row scalars (B, M)
    const long long* __restrict__ circuit_depths,
    const long long* __restrict__ lengths,
    const bool*      __restrict__ active,
    // Time-major action history (max_length, B, M)
    const long long* __restrict__ actions_time_major,
    // Action LUTs (num_actions or fewer)
    const long long* __restrict__ action_gate_types,    // (num_actions,)
    const long long* __restrict__ action_qubit1,        // (num_actions,)
    const long long* __restrict__ action_qubit2,        // (num_actions,)
    const long long* __restrict__ single_qubit_indices, // (n_single,)
    const long long* __restrict__ single_qubit_qubits,  // (n_single,)
    const long long* __restrict__ two_qubit_indices,    // (n_two,)
    const long long* __restrict__ two_qubit_q1,         // (n_two,)
    const long long* __restrict__ two_qubit_q2,         // (n_two,)
    // Per-row → active-mask-row mapping. Length B*M, value in [0, n_active)
    // for rows present in ``indices``, -1 for absent rows.
    const long long* __restrict__ active_lookup,
    // Outputs
    bool*            __restrict__ mask,                 // (n_active, num_actions)
    long long*       __restrict__ forward_counts,       // (B, M)
    long long*       __restrict__ backward_counts,      // (B, M)
    // Kernel hyperparams
    const int B,
    const int M,
    const int n_q,
    const int num_actions,
    const int max_length,
    const int n_single,
    const int n_two,
    const int terminal_index,
    const int max_depth,            // INT_MAX sentinel = "no cap"
    const int max_depth_active,     // 0 or 1; whether max_depth applies
    const int current_step,         // backward "current_step" param; 0 ⇒ no backward pass
    const long long* __restrict__ current_step_dev,  // optional device scalar
    const int current_step_use_dev, // 0 = use current_step, 1 = read current_step_dev[0]
    const int backward_compute      // 0 or 1; whether to fill backward_counts
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = B * M;
    if (idx >= total) {
        return;
    }
    const int eff_current_step =
        current_step_use_dev ? (int)(current_step_dev[0]): current_step;

    const int b = idx / M;
    const int m = idx - b * M;
    const long long active_row = active_lookup[idx];
    const bool write_mask = (active_row >= 0);
    const long long mask_base = active_row * (long long) num_actions;

    const bool row_active = active[idx];

    // Depth-cap precompute (host fills max_depth=INT_MAX when unused).
    const long long depth = circuit_depths[idx];
    const bool depth_limit  = max_depth_active && (depth >  (long long) max_depth);
    const bool at_max_depth = max_depth_active && (depth == (long long) max_depth);

    // Walk single-qubit actions: forward predicate + count + active mask.
    // ``fwd_count`` deliberately ignores the depth cap because the legacy
    // ``compute_forward_valid_counts_gpu`` is invoked with
    // ``max_depth=None`` throughout the codebase (see callers in GFNs.py;
    // the asymmetry is intentional — the backward-mask fallback path in
    // ``compute_backward_masks_gpu`` uses an unfiltered forward mask, so
    // backward counts must mirror that). The active mask honours the
    // depth cap because the on-policy sampler must not select actions
    // that would exceed max_depth.
    long long fwd_count = 0;
    const long long state_base = (long long) idx * (long long) n_q;
    // strict 2q-only depth: 1q gates never consume budget; only the
    // defensive `depth > max_depth` guard remains for them.
    for (int s = 0; s < n_single; ++s) {
        const long long q = single_qubit_qubits[s];
        const bool no_existing_single = (last_single_qubit_gates[state_base + q] < 0);
        const bool depth_block = max_depth_active && depth_limit;
        const bool count_valid = row_active && no_existing_single;
        const bool mask_valid = count_valid && !depth_block;
        if (count_valid) {
            fwd_count += 1;
        }
        if (write_mask) {
            mask[mask_base + single_qubit_indices[s]] = mask_valid;
        }
    }

    // Walk two-qubit actions: same structure, paired-qubit predicate.
    for (int t = 0; t < n_two; ++t) {
        const long long q1 = two_qubit_q1[t];
        const long long q2 = two_qubit_q2[t];
        const long long last1 = qubit_last_use_step[state_base + q1];
        const long long last2 = qubit_last_use_step[state_base + q2];

        const bool both_used = (last1 >= 0) && (last2 >= 0);
        const bool same_step = (last1 == last2);
        bool blocked_by_history = false;
        if (row_active && both_used && same_step) {
            // Last action at this shared step — look it up and compare.
            // ``clamped_steps`` mirrors the PyTorch path: clamp to
            // ``max_length - 1`` so we never index past the buffer.
            long long step = last1;
            if (step < 0) step = 0;
            if (step > (long long) (max_length - 1)) step = (long long) (max_length - 1);
            const long long last_action_id =
                actions_time_major[step * (long long) total + (long long) idx];
            const long long last_gate = action_gate_types[last_action_id];
            const long long curr_gate = action_gate_types[two_qubit_indices[t]];
            if (last_gate == curr_gate) {
                const long long last_q1 = action_qubit1[last_action_id];
                const long long last_q2 = action_qubit2[last_action_id];
                const bool same_pair_ordered  = (last_q1 == q1) && (last_q2 == q2);
                const bool same_pair_swapped  = (last_q1 == q2) && (last_q2 == q1);
                if (same_pair_ordered || same_pair_swapped) {
                    blocked_by_history = true;
                }
            }
        }

        // strict 2q-only depth: a 2q gate opens a new layer when its
        // qubits conflict with the current 2q layer OR when no 2q layer has
        // been opened yet (depth == 0 bumps to 1 on the first 2q gate).
        bool depth_block = false;
        if (max_depth_active) {
            const bool requires_new_layer =
                current_layer_qubits[state_base + q1]
                || current_layer_qubits[state_base + q2]
                || depth == 0;
            depth_block = depth_limit || (at_max_depth && requires_new_layer);
        }
        // Same count/mask asymmetry as the single-qubit branch:
        // count ignores the depth cap, mask honours it.
        const bool count_valid = row_active && !blocked_by_history;
        const bool mask_valid = count_valid && !depth_block;
        if (count_valid) {
            fwd_count += 1;
        }
        if (write_mask) {
            mask[mask_base + two_qubit_indices[t]] = mask_valid;
        }
    }

    forward_counts[idx] = fwd_count;

    // Active-mask terminal column: always True. The legacy path force-
    // wrote ``mask[..., terminal] = True`` after all other writes; we
    // do the same. Single-/two-qubit indices never collide with the
    // terminal index by construction in ``MaskingEngine.__init__``.
    if (write_mask) {
        mask[mask_base + (long long) terminal_index] = true;
    }

    // Backward valid-counts.
    if (!backward_compute) {
        return;
    }
    if (eff_current_step == 0) {
        backward_counts[idx] = fwd_count;
        return;
    }

    const long long len_b = lengths[idx];
    long long exposed = 0;

    // Single-qubit exposed: last_use matches a recorded single-qubit
    // action id and last_use is in [0, eff_current_step) ∩ [0, length).
    for (int s = 0; s < n_single; ++s) {
        const long long q = single_qubit_qubits[s];
        const long long last_use = qubit_last_use_step[state_base + q];
        const bool valid_last = (last_use >= 0)
            && (last_use < (long long) eff_current_step)
            && (last_use < len_b);
        if (!valid_last) continue;
        long long step = last_use;
        if (step > (long long) (max_length - 1)) step = (long long) (max_length - 1);
        const long long last_action_id =
            actions_time_major[step * (long long) total + (long long) idx];
        if (last_action_id == single_qubit_indices[s]) {
            exposed += 1;
        }
    }

    // Two-qubit exposed: both qubits last used at the same step, that
    // step in [0, eff_current_step) ∩ [0, length), and that step's recorded
    // action id matches the two-qubit action id.
    for (int t = 0; t < n_two; ++t) {
        const long long q1 = two_qubit_q1[t];
        const long long q2 = two_qubit_q2[t];
        const long long last1 = qubit_last_use_step[state_base + q1];
        const long long last2 = qubit_last_use_step[state_base + q2];
        if (last1 != last2) continue;
        const long long last_use = last1;  // == last2
        const bool valid_last = (last_use >= 0)
            && (last_use < (long long) eff_current_step)
            && (last_use < len_b);
        if (!valid_last) continue;
        long long step = last_use;
        if (step > (long long) (max_length - 1)) step = (long long) (max_length - 1);
        const long long last_action_id =
            actions_time_major[step * (long long) total + (long long) idx];
        if (last_action_id == two_qubit_indices[t]) {
            exposed += 1;
        }
    }

    backward_counts[idx] = (exposed > 0) ? exposed: fwd_count;
}
'''


@lru_cache(maxsize=8)
def _kernel(cuda_index: int):
    import cupy as cp

    with cp.cuda.Device(cuda_index):
        return cp.RawKernel(_KERNEL_SRC, "gfn_mask_counts_fused")


# Shared DLPack torch->cupy view; centralized in _kernel_runtime.py.
try:
    from ._kernel_runtime import cp_from_torch as _cp_from_torch
except ImportError:  # pragma: no cover - direct-execution mode
    from _kernel_runtime import cp_from_torch as _cp_from_torch


# Persistent-failure latch. Set only on host-state failures that WILL
# recur (CuPy import error, NVRTC compile error, RawKernel launch
# exception). Per-call bails — CPU tensors, missing indices, non-contiguous
# inputs — do NOT set this; those decisions are call-shape dependent.
# Mirrors the same pattern used by ``measurement_adapter.sampling_kernel``.
_persistent_failure: bool = False

# Per-device scratch tensors used when optional outputs/args are unused.
# These avoid per-call allocations in the hot path (and reduce the risk of
# CUDA-graph capture pool misses when this fused wrapper is used near capture).
_scratch_bool: dict[int, torch.Tensor] = {}
_scratch_long: dict[int, torch.Tensor] = {}


def _get_scratch_bool(cuda_index: int) -> torch.Tensor:
    scratch = _scratch_bool.get(cuda_index)
    if (
        scratch is None
        or scratch.device.type != "cuda"
        or scratch.device.index != cuda_index
    ):
        scratch = torch.empty(
            (1,), dtype=torch.bool, device=torch.device("cuda", cuda_index)
        )
        _scratch_bool[cuda_index] = scratch
    return scratch


def _get_scratch_long(cuda_index: int) -> torch.Tensor:
    scratch = _scratch_long.get(cuda_index)
    if (
        scratch is None
        or scratch.device.type != "cuda"
        or scratch.device.index != cuda_index
    ):
        scratch = torch.empty(
            (1,), dtype=torch.long, device=torch.device("cuda", cuda_index)
        )
        _scratch_long[cuda_index] = scratch
    return scratch


def fused_kernel_persistently_unavailable() -> bool:
    """Return ``True`` once a CuPy / NVRTC / launch failure has been
    observed in this process. Callers should AND this into their per-
    instance latch so the hot path stops paying repeated import /
    compile-attempt overhead.
    """
    return _persistent_failure


def reset_persistent_failure() -> None:
    """Clear the persistent-failure latch — public hook for tests and
    for callers that have remediated the underlying host state.
    """
    global _persistent_failure
    _persistent_failure = False


def compute_mask_counts_fused(
    trajectory_batch,
    indices: torch.Tensor,
    *,
    masking_engine,
    current_step: int | torch.Tensor,
    max_depth: Optional[int],
    compute_backward: bool,
    use_fused_kernel: bool = True,
) -> Optional[Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]]:
    """Try the fused CUDA mask + counts kernel.

    Returns a triple ``(active_mask, forward_counts, backward_counts)``
    on success, or ``None`` if the fused path is unavailable so the
    caller can fall back to the legacy three-call PyTorch path.

    When ``compute_backward=False`` both ``forward_counts`` and
    ``backward_counts`` are ``None`` in the returned tuple — matching
    the ``MaskingEngine.compute_masks_and_counts_fused`` fallback
    contract. The kernel internally still computes forward counts
    (they live in the same row walk as the mask); the buffer is just
    not exposed to the caller.

    Bit-for-bit parity with the three legacy functions holds because
    the kernel mirrors the same
    predicates and the same clamp/lookup behavior; the only difference
    is launch count.
    """
    global _persistent_failure
    if not use_fused_kernel:
        return None
    if _persistent_failure:
        return None

    device = trajectory_batch.device
    if not isinstance(device, torch.device):
        device = torch.device(device)
    if device.type != "cuda":
        return None

    cuda_index = device.index
    if cuda_index is None:
        cuda_index = torch.cuda.current_device()

    current_step_tensor: Optional[torch.Tensor] = None
    if torch.is_tensor(current_step):
        if current_step.dim() != 0:
            raise ValueError("current_step tensor must be 0-dimensional")
        if current_step.dtype != torch.long:
            raise ValueError("current_step tensor must have dtype torch.long")
        step_device_index = current_step.device.index
        if step_device_index is None:
            step_device_index = torch.cuda.current_device()
        if current_step.device.type != "cuda" or step_device_index != cuda_index:
            raise ValueError("current_step tensor must be on the trajectory CUDA device")
        current_step_tensor = current_step.reshape(1)
        current_step_int = 0
        current_step_use_dev = 1
    else:
        current_step_int = int(current_step)
        current_step_use_dev = 0

    try:
        import cupy as cp
    except (ImportError, OSError):
        # Narrow except (matches sampling_kernel): CuPy absent (ImportError) or
        # its shared libs unloadable (OSError) won't resolve mid-process, so
        # latch. A genuine bug *inside* cupy must propagate, not be mislatched
        # as "cupy unavailable". The launch-site broad except below stays (it is
        # the intentional NVRTC/launch latch).
        _persistent_failure = True
        return None

    B = int(trajectory_batch.batch_size)
    M = int(trajectory_batch.n_measurements)
    n_q = int(trajectory_batch.n_qubits)
    num_actions = int(masking_engine.num_actions)
    max_length = int(trajectory_batch.max_length)
    n_single = int(masking_engine.single_qubit_indices.numel())
    n_two = int(masking_engine.two_qubit_indices.numel())
    terminal_index = int(masking_engine.terminal_index)
    total = B * M

    # Build the active-mask row lookup. ``indices`` is ``(n_active, 2)`` int64
    # [batch_idx, meas_idx] — the same tensor the legacy
    # ``compute_action_masks_active_gpu`` consumes. The lookup maps a flat (b, m) id
    # to its output row, or -1 when the row is absent. n_active==0 is handled below.
    n_active = int(indices.shape[0]) if indices.numel() > 0 else 0

    # Allocate outputs. ``forward_counts`` / ``backward_counts`` use
    # ``torch.empty``: every thread writes its row exactly once (grid sizing covers
    # every ``idx`` in ``[0, total)``, threads past ``total`` early-return), and a
    # launch failure returns ``None`` before the caller sees the buffer.
    # ``active_mask`` does need a zero init — the kernel only writes the
    # single/two-qubit and terminal columns of rows present in ``indices``.
    # ``forward_scratch`` exists because the kernel writes ``forward_counts[idx]``
    # unconditionally even when ``compute_backward=False``; it is then discarded.
    active_mask = torch.empty(
        (n_active, num_actions), dtype=torch.bool, device=device
    )
    forward_buffer = torch.empty((B, M), dtype=torch.long, device=device)
    backward_counts: Optional[torch.Tensor]
    if compute_backward:
        backward_counts = torch.empty((B, M), dtype=torch.long, device=device)
    else:
        backward_counts = None

    if total == 0:
        # Empty-batch shortcut. ``forward_buffer`` is uninitialised but
        # also empty, so we expose ``None`` to the caller — and the
        # caller would see the same shape semantics either way.
        forward_return = forward_buffer if compute_backward else None
        return active_mask, forward_return, backward_counts

    # ``active_lookup``: -1 everywhere, then scatter in the row indices present in
    # ``indices``. A single PyTorch ``scatter_`` is far cheaper than another kernel
    # launch for an at-most ``B*M`` int64 tensor.
    # IMPORTANT precondition: ``indices`` must contain each ``(batch, meas)`` pair at
    # most once. ``scatter_`` is not idempotent for duplicate destinations — colliding
    # entries leave only one row written, so the others stay zero-initialised and
    # silently diverge from the legacy path, which handles duplicates via advanced
    # indexing. The sole production producer returns unique rows; new callers that
    # synthesize ``indices`` must dedup themselves. A runtime ``unique()`` check is
    # deliberately skipped: it would add a host sync per sampled step.
    active_lookup = torch.full((total,), -1, dtype=torch.long, device=device)
    if n_active > 0:
        batch_idx = indices[:, 0].to(dtype=torch.long)
        meas_idx = indices[:, 1].to(dtype=torch.long)
        flat_idx = batch_idx * M + meas_idx
        row_arange = torch.arange(n_active, device=device, dtype=torch.long)
        active_lookup.scatter_(0, flat_idx, row_arange)

    # Match the fallback's ``compute_action_masks_active_gpu``: rows where
    # ``trajectory_batch.active[b,m]`` is False must end up with only the terminal
    # column True. The kernel does this because ``valid`` includes ``row_active`` and
    # the terminal column is set unconditionally. Initialising the mask to False keeps
    # untouched columns invalid — one extra fill on a tiny buffer, worth it for
    # parity safety.
    if n_active > 0:
        active_mask.zero_()

    # INT_MAX sentinel for "no depth cap" — keeps the kernel signature
    # uniform regardless of whether the caller passed ``max_depth=None``.
    INT_MAX = (1 << 31) - 1
    if max_depth is None:
        max_depth_value = INT_MAX
        max_depth_active = 0
    else:
        max_depth_value = int(max_depth)
        max_depth_active = 1

    # Stream handoff identical to ``metadata_kernel.py`` and
    # ``sampling_kernel.py``: launch the CuPy kernel onto PyTorch's
    # current stream so the writes are visible without explicit sync.
    # (Cached wrapper — see ``cp_stream.current_external_stream``.)
    try:
        # Pre-build the per-arg CuPy views so the ``kernel_args`` tuple below reads
        # cleanly. The views are reference-counted from this scope and only need to
        # outlive the launch; the underlying torch tensors are held by the caller.
        # ``mask_arg`` needs a 1-element scratch tensor when the output is logically
        # absent (``n_active==0``), because the kernel signature has a fixed slot and
        # guards writes with ``write_mask``.
        with cp.cuda.Device(cuda_index):
            with current_external_stream(cuda_index):
                if n_active > 0:
                    mask_arg = _cp_from_torch(active_mask.view(-1))
                else:
                    mask_arg = _cp_from_torch(_get_scratch_bool(cuda_index))
                if backward_counts is not None:
                    backward_arg = _cp_from_torch(backward_counts.view(-1))
                else:
                    # ``backward_compute==0`` returns before touching this
                    # pointer, so reuse the already-required forward buffer
                    # instead of allocating throwaway scratch in mask-only calls.
                    backward_arg = _cp_from_torch(forward_buffer.view(-1))
                if current_step_tensor is not None:
                    current_step_dev_arg = _cp_from_torch(current_step_tensor)
                else:
                    # ``current_step_use_dev==0`` makes the kernel use the
                    # by-value ``current_step`` parameter; this pointer is not
                    # read. Reuse the forward buffer to avoid an otherwise-dead
                    # CUDA allocation.
                    current_step_dev_arg = _cp_from_torch(forward_buffer.view(-1))
                kernel = _kernel(cuda_index)
                block = 256
                grid = ((total + block - 1) // block,)
                kernel_args = (
                    _cp_from_torch(trajectory_batch.last_single_qubit_gates.view(-1)),
                    _cp_from_torch(trajectory_batch.qubit_last_use_step.view(-1)),
                    _cp_from_torch(trajectory_batch.current_layer_qubits.view(-1)),
                    _cp_from_torch(trajectory_batch.circuit_depths.view(-1)),
                    _cp_from_torch(trajectory_batch.lengths.view(-1)),
                    _cp_from_torch(trajectory_batch.active.view(-1)),
                    _cp_from_torch(trajectory_batch.actions_time_major.contiguous().view(-1)),
                    _cp_from_torch(masking_engine.action_gate_types.contiguous()),
                    _cp_from_torch(masking_engine.action_qubit1.contiguous()),
                    _cp_from_torch(masking_engine.action_qubit2.contiguous()),
                    _cp_from_torch(masking_engine.single_qubit_indices.contiguous()),
                    _cp_from_torch(masking_engine.single_qubit_qubits.contiguous()),
                    _cp_from_torch(masking_engine.two_qubit_indices.contiguous()),
                    _cp_from_torch(masking_engine.two_qubit_q1.contiguous()),
                    _cp_from_torch(masking_engine.two_qubit_q2.contiguous()),
                    _cp_from_torch(active_lookup),
                    mask_arg,
                    _cp_from_torch(forward_buffer.view(-1)),
                    backward_arg,
                    int(B),
                    int(M),
                    int(n_q),
                    int(num_actions),
                    int(max_length),
                    int(n_single),
                    int(n_two),
                    int(terminal_index),
                    int(max_depth_value),
                    int(max_depth_active),
                    int(current_step_int),
                    current_step_dev_arg,
                    int(current_step_use_dev),
                    int(1 if backward_counts is not None else 0),
                )
                kernel(grid, (block,), kernel_args)
    except Exception:
        _persistent_failure = True
        return None

    # API consistency with the fallback: callers passing ``compute_backward=False``
    # get ``(active_mask, None, None)``. The kernel still writes ``forward_buffer``
    # because it always stores forward_counts; we simply don't expose it. Callers
    # needing a forward-only count should call ``compute_forward_valid_counts_gpu``.
    forward_return = forward_buffer if compute_backward else None
    return active_mask, forward_return, backward_counts
