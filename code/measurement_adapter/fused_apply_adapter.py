"""FlowMeas adapter for CT fused apply+metadata API.

The CT `apply_action_layer_batched_with_metadata` entry point collapses
FlowMeas's per-layer chain of

    ActionAdapter.translate_step   (1 LUT gather)
  + BatchedCliffordSim.apply_layer_batched  (1-3 substep launches)
  + apply_metadata_kernel          (1 launch)

into **two CUDA kernels per layer**: one fused substep-apply that walks
each trajectory's primitive substep sequence from a device-resident
``ActionLoweringTable``, then a metadata writer with byte-equivalent
ABI to FlowMeas's existing ``gfn_update_metadata`` kernel.

This module owns:

* ``build_lowering_table()``: bakes FlowMeas's ``action_map`` and gate-type
  tensors into an ``ActionLoweringTable``. Called once per training run
  (typically lazily in ``GFlowNet.__init__``) and validated against
  ``n_qubits`` so the per-step wrapper call skips its own sync.

* ``apply_action_layer_fused()``: the per-step entry point. Wraps the
  CT call with the FlowMeas stream-handoff convention, builds the
  ``FusedApplyMetadataView`` from ``trajectory_batch.*`` torch tensors
  via DLPack, returns the ``terminated`` tensor or ``None`` to signal
  fallback. Uses the same fail-once latch pattern as
  ``mask_counts_kernel.py``: CuPy import / NVRTC compile / RawKernel
  launch failures set a process-wide latch so subsequent calls skip
  the fused attempt.

The legacy chain (``TableauBatchAdapter.apply_actions_step`` +
``apply_metadata_kernel``) remains in place as the fallback for
CPU / missing-CuPy / ``use_fused_apply_kernel=False`` / latched
failures.
"""
from __future__ import annotations

import logging
import weakref
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch

try:
    from .cp_stream import current_external_stream
except ImportError:  # pragma: no cover - direct-execution mode
    from cp_stream import current_external_stream


_ct_meta_needs_last_two: Optional[bool] = None  # cached CT-version probe result


def _ct_fused_metadata_requires_last_two() -> bool:
    """True iff the installed clifford-tableau's ``FusedApplyMetadataView`` still
    declares ``last_two_qubit_gates``.

    FlowMeas removed that field in lockstep with the CT packed-hit-features
    release. If the installed CT still requires it, constructing the view without
    it raises a cryptic ``TypeError`` on the DEFAULT fused-apply path. Probe the
    dataclass fields once so the caller can soft-disable the fused path instead of
    crashing.
    """
    global _ct_meta_needs_last_two
    if _ct_meta_needs_last_two is None:
        import dataclasses
        try:
            from clifford_tableau.sim import FusedApplyMetadataView
            field_names = {f.name for f in dataclasses.fields(FusedApplyMetadataView)}
            _ct_meta_needs_last_two = "last_two_qubit_gates" in field_names
        except (ImportError, TypeError):
            # Can't introspect (not a dataclass / import issue): assume compatible
            # and let any genuine error surface at the construction site.
            _ct_meta_needs_last_two = False
    return _ct_meta_needs_last_two


# A recoverable OOM can arrive as a built-in ``MemoryError``, a
# ``torch.cuda.OutOfMemoryError`` (a ``RuntimeError`` subclass, so a plain
# ``except MemoryError`` misses it), or a legacy ``RuntimeError`` with
# "out of memory" in the message. Centralizing the predicate keeps the per-call
# ``except`` blocks and ``build_lowering_table`` aligned.
_TORCH_CUDA_OOM = getattr(
    getattr(torch, "cuda", None), "OutOfMemoryError", None
)
if _TORCH_CUDA_OOM is None:
    _RECOVERABLE_OOM_TYPES: Tuple[type, ...] = (MemoryError,)
else:
    _RECOVERABLE_OOM_TYPES = (MemoryError, _TORCH_CUDA_OOM)


def _is_recoverable_oom(exc: BaseException) -> bool:
    """True for any allocation/OOM failure considered recoverable by the
    fused-apply preflight policy.

    Covers ``MemoryError`` (CPU/CuPy host-side), ``torch.cuda.OutOfMemoryError``
    when present (the modern PyTorch subclass of ``RuntimeError``), and
    the legacy ``RuntimeError("... out of memory...")`` form some older
    PyTorch builds raise. Callers should classify these as soft
    preflight failures: fall back this call without latching the fused
    path off.
    """
    if isinstance(exc, _RECOVERABLE_OOM_TYPES):
        return True
    # Legacy fallback: pre-2.0 PyTorch builds raised plain ``RuntimeError``
    # with "out of memory" in the message for CUDA OOM. The match is
    # case-insensitive and tolerates surrounding text.
    if isinstance(exc, RuntimeError):
        msg = str(exc).lower()
        return "out of memory" in msg or "cuda error: out of memory" in msg
    return False


# Module-level fail-once latch. Set only on host-state failures that WILL recur
# (CuPy import error, NVRTC compile, CT API call exception). Per-call bails — CPU
# tensors, ``use_fused_kernel=False`` — do NOT touch this.
_persistent_failure: bool = False
_trusted_device_step_tensor_refs: Dict[int, weakref.ReferenceType[torch.Tensor]] = {}


def mark_trusted_device_step_tensor(step: torch.Tensor) -> None:
    """Allow an internal graph-owned device step scalar during CUDA capture.

    Public callers using a tensor step are still host-range-validated before CT
    entry. The graph sampler owns its scalar, resets it before capture/replay,
    and advances it in the captured graph; validating that live value with
    ``item()`` during capture would break graph capture, so it registers the
    exact tensor object here.
    """
    if not torch.is_tensor(step):
        raise TypeError("trusted fused-apply step must be a torch.Tensor")
    if step.dim() != 0 or step.dtype != torch.long or step.device.type != "cuda":
        raise ValueError(
            "trusted fused-apply step must be a 0-dimensional CUDA torch.long tensor"
        )
    _trusted_device_step_tensor_refs[id(step)] = weakref.ref(step)


def _is_trusted_device_step_tensor(step: torch.Tensor) -> bool:
    ref = _trusted_device_step_tensor_refs.get(id(step))
    if ref is None:
        return False
    trusted = ref()
    if trusted is step:
        return True
    if trusted is None:
        _trusted_device_step_tensor_refs.pop(id(step), None)
    return False


def _current_stream_is_capturing() -> bool:
    try:
        return bool(torch.cuda.is_current_stream_capturing())
    except Exception:
        return False


def fused_kernel_persistently_unavailable() -> bool:
    """Return ``True`` once a CuPy / NVRTC / CT API failure has been
    observed in this process. Callers should AND this into their per-
    instance latch so the hot path stops paying repeated import /
    compile-attempt overhead.
    """
    return _persistent_failure


def reset_persistent_failure() -> None:
    """Clear the **module-level** persistent-failure latch only.

    Public hook for tests and for callers that have remediated the
    underlying host state (re-installed CuPy, rotated to a healthy
    GPU, etc.).

    **Scope warning**: this clears ONLY
    the module-level ``_persistent_failure`` flag in this file. Each
    ``GFlowNet`` instance maintains its OWN per-instance latch
    (``_fused_apply_kernel_failed``) which the
    ``_effective_fused_apply_kernel`` helper checks before consulting
    the module-level state. After this reset, existing live
    ``GFlowNet`` instances whose instance latch tripped will still
    short-circuit to ``False`` on every call regardless of the
    module-level state. Use ``GFlowNet.reset_fused_apply_latches()``
    on each instance OR reconstruct the trainer to restore the fast
    path. Notebooks / long-running benchmark harnesses that intend
    to retry the fused path after a remediated host should use the
    instance method; tests that have just stubbed/monkey-patched the
    underlying module state should call BOTH.
    """
    global _persistent_failure
    _persistent_failure = False


def _build_lowering_lut_numpy(
    action_map: Dict[int, Tuple],
    n_qubits: int,
) -> Tuple[np.ndarray, int]:
    """Bake FlowMeas's ``action_map`` into a ``(num_actions, max_depth, 3) int32``
    NumPy LUT of primitive substep triples.

    This mirrors ``ActionAdapter._lower_one`` exactly so the fused path
    consumes the same substep sequences the legacy ``translate_step``
    produces. Re-implementing the lowering here (rather than reaching
    into ``ActionAdapter``'s private ``_lut_np``) keeps the adapter
    boundary clean and lets the fused module construct its own
    ``ActionLoweringTable`` without holding an extra ``ActionAdapter``
    instance.

    Returns
    -------
    (lut, max_depth)
        ``lut`` is the per-action substep array (CPU). ``max_depth`` is
        the longest substep chain across all actions (used to size the
        ``ActionLoweringTable``'s middle dimension).
    """
    from clifford_tableau.interfaces import GateType

    H = int(GateType.H)
    S = int(GateType.S)
    CX = int(GateType.CX)
    composite_lowering: Dict[str, list] = {
        "H":   [H],
        "S":   [S],
        # FlowMeas apply_HS = apply_S; apply_H  (right-to-left composition)
        "HS":  [S, H],
        # FlowMeas apply_SH = apply_H; apply_S
        "SH":  [H, S],
        # FlowMeas apply_HSH = apply_H; apply_S; apply_H
        "HSH": [H, S, H],
    }

    if not action_map:
        raise ValueError("action_map is empty")
    for aid in action_map:
        if not isinstance(aid, (int, np.integer)):
            raise TypeError(
                f"action_map keys must be ints; got {type(aid).__name__}"
            )
        if int(aid) < 0:
            raise ValueError(
                f"action_map keys must be non-negative; got {int(aid)}"
            )
    max_id = max(int(k) for k in action_map)
    num_actions = max_id + 1

    sequences: list = [[] for _ in range(num_actions)]
    for aid, tup in action_map.items():
        if tup is None or len(tup) == 0 or tup[0] == "terminal":
            sequences[int(aid)] = []
            continue
        gname = tup[0]
        if gname in composite_lowering:
            q = int(tup[1])
            if not (0 <= q < n_qubits):
                raise ValueError(
                    f"action {int(aid)} qubit {q} out of range for "
                    f"n_qubits={n_qubits}"
                )
            sequences[int(aid)] = [(gcode, q, -1) for gcode in composite_lowering[gname]]
        elif gname == "CNOT":
            if len(tup) < 3:
                raise ValueError(f"CNOT requires (control, target); got {tup}")
            c, t = int(tup[1]), int(tup[2])
            if c == t:
                raise ValueError(f"CNOT control == target ({c}) for action {int(aid)}")
            for q in (c, t):
                if not (0 <= q < n_qubits):
                    raise ValueError(
                        f"action {int(aid)} CNOT qubit {q} out of range for "
                        f"n_qubits={n_qubits}"
                    )
            sequences[int(aid)] = [(CX, c, t)]
        else:
            raise ValueError(f"Unknown FlowMeas gate name: {gname!r}")

    max_depth = max((len(seq) for seq in sequences), default=1) or 1
    lut = np.full((num_actions, max_depth, 3), -1, dtype=np.int32)
    for aid, seq in enumerate(sequences):
        for s, (g, q1, q2) in enumerate(seq):
            lut[aid, s, 0] = g
            lut[aid, s, 1] = q1
            lut[aid, s, 2] = q2
    return lut, max_depth


def build_lowering_table(
    action_map: Dict[int, Tuple],
    n_qubits: int,
    action_gate_types: torch.Tensor,
    action_qubit1: torch.Tensor,
    action_qubit2: torch.Tensor,
    single_qubit_mask: torch.Tensor,
    two_qubit_mask: torch.Tensor,
    *,
    device: torch.device,
):
    """Construct an ``ActionLoweringTable`` from FlowMeas's per-run state.

    The returned table holds both the primitive substep LUT (consumed
    by the fused apply kernel) and the per-action metadata columns
    (consumed by the metadata kernel). All arrays live on ``device``.

    The table's ``validate_qubit_ranges(n_qubits)`` is invoked once
    here so subsequent per-step wrapper calls short-circuit the
    validation host-sync (per the CT API's pre-warm contract).

    Returns ``None`` if CuPy or the CT API is unavailable on this host.
    Caller must check the return value and route to fallback if None.
    """
    global _persistent_failure
    if _persistent_failure:
        return None
    if device.type != "cuda":
        return None
    # Narrow import handler — only ``ImportError`` should set the
    # process-wide latch (CuPy missing / CT not installed). A broader
    # ``Exception`` catch here would also latch on transient host
    # issues that are not "fused path is unavailable".
    try:
        import cupy as cp
        from clifford_tableau.sim import ActionLoweringTable
    except ImportError:
        _persistent_failure = True
        return None

    cuda_index = device.index
    if cuda_index is None:
        cuda_index = torch.cuda.current_device()

    # Build the primitive LUT on CPU (cheap; runs once per training run).
    # We re-bake instead of pulling from ``ActionAdapter._lut_np`` so the
    # fused module doesn't depend on ``ActionAdapter`` having been
    # constructed first.
    lut_np, _max_depth = _build_lowering_lut_numpy(action_map, n_qubits)

    # Metadata-facing columns: must be int64 (CT contract), 1-D, len num_actions.
    # FlowMeas's ``GFlowNet`` stores these as torch tensors; we DLPack-bridge to CuPy
    # to keep them on the same device the simulator uses.
    if not (
        action_gate_types.dtype == torch.long
        and action_qubit1.dtype == torch.long
        and action_qubit2.dtype == torch.long
    ):
        raise TypeError(
            "action_gate_types / action_qubit1 / action_qubit2 must be torch.long"
        )
    if not (
        single_qubit_mask.dtype == torch.bool
        and two_qubit_mask.dtype == torch.bool
    ):
        raise TypeError("single_qubit_mask / two_qubit_mask must be torch.bool")

    # ``gate_types_for_meta`` is encoded in CT's ``GateType`` namespace, NOT FlowMeas's
    # ``gate_name_to_idx`` (where "SH"=3 collides with CT's CX=3 and "HSH"=4 with
    # CZ=4). CT validates at table construction that single-qubit rows carry a
    # single-qubit arity code, which FlowMeas's encoding would fail for the composites.
    # Every downstream consumer reads ``last_single_qubit_gates`` only via
    # ``>= 0`` / ``< 0`` presence checks, never for identity matching, so mapping
    # all single-qubit actions to ``GateType.H`` and all two-qubit ones to
    # ``GateType.CX`` preserves semantics while passing CT's arity validation.
    # Inactive metadata rows get -1, which CT ignores.
    from clifford_tableau.interfaces import GateType
    single_code = int(GateType.H)
    two_code = int(GateType.CX)
    is_single_cpu = single_qubit_mask.cpu().numpy()
    is_two_cpu = two_qubit_mask.cpu().numpy()
    gate_types_for_meta_np = np.where(
        is_single_cpu,
        np.int64(single_code),
        np.where(is_two_cpu, np.int64(two_code), np.int64(-1)),
    ).astype(np.int64)

    try:
        with cp.cuda.Device(cuda_index):
            lut_cp = cp.asarray(lut_np)
            gate_types_cp = cp.asarray(gate_types_for_meta_np)
            q1_cp = cp.from_dlpack(action_qubit1.to(device=device).contiguous())
            q2_cp = cp.from_dlpack(action_qubit2.to(device=device).contiguous())
            is_single_cp = cp.from_dlpack(
                single_qubit_mask.to(device=device).contiguous()
            )
            is_two_cp = cp.from_dlpack(
                two_qubit_mask.to(device=device).contiguous()
            )

            table = ActionLoweringTable(
                lut=lut_cp,
                gate_types_for_meta=gate_types_cp,
                qubit1_for_meta=q1_cp,
                qubit2_for_meta=q2_cp,
                is_single_for_meta=is_single_cp,
                is_two_for_meta=is_two_cp,
            )
            # Pre-warm the n_qubits validation cache. Folds the validation
            # host-sync into one-time construction so per-step calls are
            # sync-free (the CT API's "pre-warm" contract).
            table.validate_qubit_ranges(int(n_qubits))
    except (TypeError, ValueError):
        # Caller-side data error (FlowMeas action map / gate-type tensors malformed for
        # CT). Do NOT latch the fused path globally — a different ``GFlowNet`` with
        # a different action map may still build a valid table. Re-raise so the
        # caller sees the actual failure instead of a silent fallback.
        raise
    except Exception as build_exc:
        # CUDA OOM during lowering-table construction is recoverable: it means THIS
        # trainer can't afford the table now, not that the fused path is permanently
        # unavailable. Return ``None`` WITHOUT setting the module-level latch so a
        # smaller or later ``GFlowNet`` can still try; the caller's per-instance
        # latch trips on the ``None``, which is the right granularity.
        if _is_recoverable_oom(build_exc):
            return None
        # Genuine host-state failure (CuPy import crash, NVRTC compile failure, CT
        # ABI mismatch). These indicate process-wide unavailability, so set the latch
        # and let future constructions short-circuit at the preflight check.
        _persistent_failure = True
        return None

    return table


def _build_metadata_view(
    trajectory_batch,
    batched_tableau,
    *,
    terminated: torch.Tensor,
    cuda_index: int,
):
    """Wrap ``trajectory_batch.*`` and ``batched_tableau.active`` into a
    ``FusedApplyMetadataView`` via DLPack.

    Cheap (CuPy view construction, no device-side work) but unavoidable
    per call because CT's CuPy views can become stale if the underlying
    torch tensors are reallocated. For FlowMeas's per-sample-call
    ``TrajectoryBatch`` lifetime, "per-call" is the right granularity.

    ``terminated`` is the caller's already-selected buffer (see
    ``apply_action_layer_fused`` docstring on ``terminated`` for the
    rationale). We DLPack-wrap it as-is; the CT
    metadata kernel writes through to the same storage.
    """
    import cupy as cp
    from clifford_tableau.sim import FusedApplyMetadataView

    n_qubits = int(trajectory_batch.n_qubits)
    max_length = int(trajectory_batch.max_length)

    # Each FlowMeas tensor must be contiguous + 1-D (CT validates both). The
    # underlying torch tensors are always C-contiguous by construction in
    # ``TrajectoryBatch.__init__``, so ``.view(-1)`` is a metadata-only
    # no-copy reshape.
    with cp.cuda.Device(cuda_index):
        view = FusedApplyMetadataView(
            active=cp.from_dlpack(trajectory_batch.active.contiguous().view(-1)),
            tableau_active=cp.from_dlpack(batched_tableau.active.contiguous().view(-1)),
            terminated=cp.from_dlpack(terminated.contiguous().view(-1)),
            circuit_depths=cp.from_dlpack(
                trajectory_batch.circuit_depths.contiguous().view(-1)
            ),
            current_layer_qubits=cp.from_dlpack(
                trajectory_batch.current_layer_qubits.contiguous().view(-1)
            ),
            qubit_last_layer=cp.from_dlpack(
                trajectory_batch.qubit_last_layer.contiguous().view(-1)
            ),
            last_single_qubit_gates=cp.from_dlpack(
                trajectory_batch.last_single_qubit_gates.contiguous().view(-1)
            ),
            qubit_last_use_step=cp.from_dlpack(
                trajectory_batch.qubit_last_use_step.contiguous().view(-1)
            ),
            action_qubits=cp.from_dlpack(
                trajectory_batch.action_qubits.contiguous().view(-1)
            ),
            n_qubits=n_qubits,
            max_length=max_length,
        )
    return view


class FusedApplyMidCallError(RuntimeError):
    """Raised when the CT fused apply API enters and then fails partway.

    Distinct from the silent-``None`` fallback used for preflight
    failures (missing CuPy, lowering table absent, etc.). Once
    ``apply_action_layer_batched_with_metadata`` has been entered, the
    CT fused-substep-apply kernel may have already been queued onto the
    caller's stream — silently retrying the legacy chain on the same
    actions would *double-apply* the layer's Clifford gates to the
    tableau, corrupting trajectory state non-recoverably.

    Callers must NOT swallow this exception and route to the legacy
    fallback. The fused-apply per-process latch is set as a side
    effect so subsequent calls take the preflight ``None`` path, but
    the current call must propagate (or restore tableau state from a
    checkpoint, which FlowMeas does not currently support).
    """


class FusedApplyRecoverableError(RuntimeError):
    """Raised on **soft** preflight failures that should fall back this
    call without disabling the fused path on this ``GFlowNet`` instance.

    Cases like a transient CuPy/DLPack
    ``MemoryError`` or a one-off CUDA resource hiccup happen BEFORE any
    CT-side mutation; the legacy chain is safe to take for this call,
    but the failure is not evidence that the fused path is permanently
    unavailable on this host. Returning a plain ``None`` to the GFN
    caller would incorrectly trip ``_fused_apply_kernel_failed``, the
    per-instance latch, and disable fused apply for the rest of the
    trainer's lifetime.

    The GFN caller catches this exception, takes the legacy fallback
    for the current call, and leaves both the per-instance and the
    module-level latches clear so the next call retries the fused path.

    Hard preflight failures (lowering table missing, ``_persistent_failure``
    set, non-CUDA tensors, ``use_fused_kernel=False``) keep returning
    ``None`` and DO set the per-instance latch — those signal the
    fused path is genuinely not available for this trainer and there is
    no point retrying every layer.
    """


def apply_action_layer_fused(
    *,
    lowering,
    trajectory_batch,
    batched_tableau,
    terminated: torch.Tensor,
    actions: torch.Tensor,
    terminal_index: int,
    step: Optional[Union[int, torch.Tensor]] = None,
    validate_action_ids: bool = False,
    use_fused_kernel: bool = True,
) -> Optional[torch.Tensor]:
    """Try the CT fused apply+metadata API; return ``terminated`` or
    ``None`` for fallback.

    Parameters
    ----------
    lowering: ActionLoweringTable
        Pre-baked LUT from ``build_lowering_table``; the caller keeps it alive
        for the run (typically one per ``GFlowNet`` instance).
    trajectory_batch: TrajectoryBatch
        Consumes ``.active``, ``.circuit_depths``, ``.current_layer_qubits``,
        ``.qubit_last_layer``, ``.last_single_qubit_gates``,
        ``.qubit_last_use_step``, ``.action_qubits`` and the
        ``_terminated_buffers`` double-buffer.
    batched_tableau: TableauBatchAdapter
        ``.active`` becomes ``tableau_active`` in the CT metadata view;
        ``._sim`` is the ``BatchedCliffordSim`` the fused kernel mutates.
    actions: torch.Tensor
        ``(B, M)`` int64 on the simulator's device. Inactive rows must carry
        ``terminal_index``.
    terminal_index: int
        Must fit in int32.
    step: int or torch.Tensor, optional
        ``-1`` skips the step-indexed ``action_qubits[step, :]`` and
        ``qubit_last_use_step[..., step]`` writes; otherwise an integer in
        ``[0, trajectory_batch.max_length)``. CUDA-graph callers may pass a
        0-D CUDA scalar, DLPack-bridged so CT reads the live step from device
        memory.
    validate_action_ids: bool
        ``False`` (default) matches the sampling hot path, where ids come from
        this run's own sampler. Pass ``True`` for replay or external streams.
    terminated: torch.Tensor
        ``(B, M)`` bool buffer the caller selects, written in place by the CT
        metadata kernel. Passing it in keeps the double-buffer alternation
        owned by a single layer.
    use_fused_kernel: bool
        ``False`` returns ``None`` immediately so the caller takes the legacy
        path.

    Returns
    -------
    ``terminated`` when the fused path succeeded (the caller's buffer is now
    populated), or ``None`` when the caller should fall back to the legacy
    chain AND treat the fused path as hard-unavailable for this instance
    (sets ``_fused_apply_kernel_failed`` on the GFN). ``None`` is returned
    only for preflight conditions that recur for the trainer's lifetime, so
    no CT-side mutation has occurred.

    Raises
    ------
    FusedApplyRecoverableError
        Preflight failure, before the CT entry point, so no CT-side mutation
        is possible — currently any recoverable OOM from the DLPack bridge or
        the actions-tensor normalization. The caller takes the legacy chain
        for this call and leaves every latch clear so the next call retries.
    FusedApplyMidCallError
        The CT API was entered and raised something other than the documented
        prelaunch types ``(TypeError, ValueError)``. The substep-apply kernel
        may already be queued and the tableau partially mutated, so replaying
        the legacy chain would double-apply the layer. Sets the module-level
        latch; callers must propagate.
    TypeError, ValueError
        Re-raised from CT prelaunch validation or the preflight tensor prep;
        no CT-side mutation occurred. A caller-data error for this batch, not
        fused-path unavailability, so no latch is set.
    """
    global _persistent_failure
    # ------------------------------------------------------------------
    # Preflight (safe to fall back via None: no CT mutation possible yet)
    # ------------------------------------------------------------------
    if not use_fused_kernel:
        return None
    if _persistent_failure:
        return None
    if lowering is None:
        return None
    if actions.device.type != "cuda":
        return None

    # Narrow import handler: only set the
    # process latch on a real "fused path is unavailable on this host"
    # signal. Broader ``Exception`` would also catch transient CUDA/
    # OOM errors which are not feature unavailability.
    try:
        import cupy as cp
        from clifford_tableau.sim import apply_action_layer_batched_with_metadata
    except ImportError:
        _persistent_failure = True
        return None

    # Cross-repo lockstep guard: this FlowMeas build removed last_two_qubit_gates
    # from the fused-apply metadata in lockstep with the CT packed-hit-features
    # release. If the installed CT still requires that field,
    # ``FusedApplyMetadataView`` would raise a cryptic TypeError on the DEFAULT path,
    # which the GFN caller does not catch. Detect once and latch the fused path off.
    if _ct_fused_metadata_requires_last_two():
        if not _persistent_failure:
            logging.getLogger(__name__).warning(
                "clifford-tableau's FusedApplyMetadataView still requires "
                "last_two_qubit_gates; this FlowMeas build targets the CT "
                "packed-hit-features release. Disabling the fused-apply kernel and "
                "using the legacy apply path. Update clifford-tableau to enable it."
            )
        _persistent_failure = True
        return None

    device = actions.device
    cuda_index = device.index
    if cuda_index is None:
        cuda_index = torch.cuda.current_device()

    # CT expects a (total,) int64 CuPy view, and the torch tensor must be contiguous
    # before DLPack hand-off, so ``.contiguous()`` is called defensively (a no-op for
    # the sampler's contiguous leading-dim slice).
    # The dtype-normalize and contiguous-view operations are deferred into the
    # preflight ``try`` below so that a CUDA OOM from either — both are real device
    # allocations when the input mismatches — is classified as a recoverable
    # preflight failure rather than propagating raw. ``actions_flat`` is computed
    # inside the same ``Device`` + ``ExternalStream`` context as the rest of
    # preflight, which is where ``cp.from_dlpack`` runs.

    # The DLPack bridge AND the CT call must run inside the same CuPy ``Device`` +
    # ``ExternalStream`` context. Splitting them puts the producer/consumer boundary
    # on different streams: pending PyTorch writes to ``actions`` /
    # ``trajectory_batch.*`` / ``batched_tableau.active`` could then be read stale by
    # CT kernels. The legacy ``apply_actions_step`` avoids this the same way.
    #
    # CT runs all shape/dtype/range validation synchronously BEFORE queuing, so
    # ``TypeError`` / ``ValueError`` from CT always means "caller-data error, no
    # mutation happened" — re-raise without latching. Everything else must be
    # treated as a post-launch hazard, since a CT-side ``MemoryError`` cannot be
    # assumed to precede the queue:
    #   * Preflight (DLPack bridge, before CT entry): recoverable OOM raises
    #     ``FusedApplyRecoverableError``; no CT mutation is possible.
    #   * After CT entry: anything outside the prelaunch contract re-raises as
    #     ``FusedApplyMidCallError`` with the module latch set, OOM included.
    # The tracker flips inside the ``with`` block immediately before the CT call,
    # so a prelaunch error is still classified correctly.
    crossed_ct_boundary = False
    try:
        with cp.cuda.Device(cuda_index), current_external_stream(cuda_index):
            # ---- Preflight: DLPack bridge (no CT-side mutation yet) ----
            try:
                # Dtype-normalize and
                # contiguous-view inside the preflight envelope so a
                # CUDA OOM from these tensor operations classifies as
                # recoverable. See comment block above for rationale.
                if actions.dtype != torch.long:
                    actions = actions.to(dtype=torch.long)
                actions_flat = actions.contiguous().view(-1)
                if step is None:
                    step_for_ct = -1
                elif torch.is_tensor(step):
                    if step.dim() != 0:
                        raise ValueError("fused apply tensor step must be 0-dimensional")
                    if step.dtype != torch.long:
                        raise ValueError(
                            "fused apply tensor step must have dtype torch.long"
                        )
                    step_device_index = step.device.index
                    if step_device_index is None:
                        step_device_index = torch.cuda.current_device()
                    if step.device.type != "cuda" or step_device_index != cuda_index:
                        raise ValueError(
                            "fused apply tensor step must be on the actions CUDA device"
                        )
                    if _current_stream_is_capturing():
                        if not _is_trusted_device_step_tensor(step):
                            raise ValueError(
                                "fused apply tensor step cannot be value-validated "
                                "during CUDA graph capture; use a graph-owned "
                                "registered scalar"
                            )
                        if validate_action_ids:
                            raise ValueError(
                                "fused apply action-id validation is not CUDA-graph "
                                "capturable with a device step; disable validation "
                                "before capture"
                            )
                    else:
                        step_value = int(step.item())
                        if step_value < 0 or step_value >= int(trajectory_batch.max_length):
                            raise ValueError(
                                "fused apply tensor step must be in "
                                f"[0, {int(trajectory_batch.max_length)})"
                            )
                    step_for_ct = cp.from_dlpack(step.contiguous().reshape(-1))
                else:
                    step_for_ct = int(step)

                metadata_view = _build_metadata_view(
                    trajectory_batch,
                    batched_tableau,
                    terminated=terminated,
                    cuda_index=cuda_index,
                )
                actions_cp = cp.from_dlpack(actions_flat)
            except (TypeError, ValueError):
                # Caller-data error (shape/dtype). No CT mutation
                # happened. Re-raise without latching so future valid
                # batches can still take the fused path.
                raise
            except Exception as preflight_exc:
                # Broaden OOM detection to
                # cover ``torch.cuda.OutOfMemoryError`` (subclass of
                # ``RuntimeError``, NOT ``MemoryError``) and the legacy
                # ``RuntimeError("...out of memory...")`` form.
                if _is_recoverable_oom(preflight_exc):
                    raise FusedApplyRecoverableError(
                        "CuPy/DLPack/CUDA OOM during preflight; falling "
                        "back for this call without latching the fused "
                        "path off."
                    ) from preflight_exc
                raise
            # ---- Enter CT API ----
            # CT runs prelaunch validation synchronously and raises
            # (TypeError, ValueError) BEFORE any kernel queues if inputs are
            # malformed. Any OTHER exception — including OOM — may have queued the
            # substep-apply kernel first, so it signals a post-launch hazard.
            crossed_ct_boundary = True
            apply_action_layer_batched_with_metadata(
                sim=batched_tableau._sim,
                actions=actions_cp,
                lowering=lowering,
                metadata=metadata_view,
                terminal_index=int(terminal_index),
                step=step_for_ct,
                validate_action_ids=bool(validate_action_ids),
            )
    except (FusedApplyRecoverableError, FusedApplyMidCallError):
        # Already classified — propagate as-is.
        raise
    except (TypeError, ValueError):
        # CT prelaunch validation OR our own DLPack-bridge validation: either way no
        # CT-side mutation happened. Re-raise without setting the process-wide latch
        # so future valid batches can still use the fused path. The GFN caller does
        # NOT catch these, so the calling code sees the actual error. The legacy
        # chain is deliberately not retried — if CT rejected the action ids as
        # out-of-range, the legacy chain would silently execute the wrong gates.
        raise
    except Exception as exc:
        # Anything else from inside the ``with`` block.
        # PRE-CT-ENTRY (``crossed_ct_boundary`` False): non-validation,
        # non-recoverable-OOM preflight failures from the DLPack bridge,
        # ``_build_metadata_view``, or the actions prep — e.g. a corrupted CuPy
        # install or a torch DLPack ABI mismatch. No CT mutation occurred, so return
        # ``None`` and let the GFN latch this instance off: these are "this trainer
        # can't use the fused path" signals, not transient pressure.
        # POST-CT-ENTRY (True): the substep-apply kernel may already be queued, so the
        # tableau may be partially mutated and the legacy fallback would double-apply.
        # This branch INTENTIONALLY catches CT-side OOM, because we cannot prove
        # without an explicit CT contract that all allocation precedes queuing.
        if not crossed_ct_boundary:
            return None
        _persistent_failure = True
        raise FusedApplyMidCallError(
            "CT fused apply+metadata raised after kernel entry; the "
            "tableau state may have been partially mutated. Re-applying "
            "the legacy chain on the same actions would double-apply "
            "the layer. The fused path is latched off for this process."
        ) from exc

    # Caller's terminated buffer was written by the CT metadata kernel.
    return terminated
