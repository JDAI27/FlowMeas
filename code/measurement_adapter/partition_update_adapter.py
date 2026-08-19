"""FlowMeas bridge for the CT Phase-4 fused bucketed partition/lengths update.

Follows the fused-apply bridge pattern: FlowMeas owns the torch
surface and the fallback; the reusable primitive lives in ``clifford-tableau``
(``clifford_tableau.sim.partition_update_bucketed``) and is consumed here via
a public API. One launch replaces the bucketed sampler's per-layer tail —
survivor detection, the newly-terminated ``lengths`` update, the STABLE
ordered queue partition, and the count/entering/device_step scalar updates
(~14 torch kernels inside every captured bucketed graph).

Graph capture: every operand is passed by device address and the CT call
launches on the wrapped torch stream — capture-safe, replay reads live values.

The caller keeps the torch chain as the fallback; this module only reports
availability (None return) the way the sibling kernel adapters do.
"""
from __future__ import annotations

from typing import Optional

import torch

try:
    from .cp_stream import current_external_stream
except ImportError:  # pragma: no cover - direct-execution mode
    from cp_stream import current_external_stream

try:
    from clifford_tableau.sim import partition_update_bucketed as _ct_partition_update
except ImportError:  # pragma: no cover - CT too old / absent
    _ct_partition_update = None


# Persistent-failure latch: same contract as the sibling kernel modules — set
# only on host-state failures that WILL recur (CT import/NVRTC/launch errors).
_persistent_failure: bool = False


def fused_kernel_persistently_unavailable() -> bool:
    """``True`` once a CT import / NVRTC compile / launch failure latched."""
    return _persistent_failure


def reset_persistent_failure() -> None:
    """Clear the latch (host remediated / test isolation)."""
    global _persistent_failure
    _persistent_failure = False


def partition_update_bucketed_torch(
    active_flat: torch.Tensor,
    idx_in: torch.Tensor,
    idx_out: torch.Tensor,
    count: torch.Tensor,
    entering: torch.Tensor,
    lengths_flat: torch.Tensor,
    step_plus1: torch.Tensor,
    device_step: torch.Tensor,
    *,
    use_fused_kernel: bool = True,
) -> Optional[bool]:
    """Try the CT fused partition/update; ``None`` means "caller falls back".

    All tensors are the caller's stable device buffers (the captured graph
    binds their addresses): ``active_flat`` bool ``[B*M]``, ``idx_in``/
    ``idx_out`` int64 ``[K]`` (distinct), ``count``/``entering``/
    ``step_plus1``/``device_step`` int64 scalars (0- or 1-dim), and
    ``lengths_flat`` int64 ``[B*M]``. Semantics documented on the CT API.
    """
    global _persistent_failure
    if not use_fused_kernel:
        return None
    if _ct_partition_update is None:
        return None
    if _persistent_failure:
        return None
    if active_flat.device.type != "cuda":
        return None

    try:
        import cupy as cp
    except (ImportError, OSError):
        _persistent_failure = True
        return None

    cuda_index = active_flat.device.index
    if cuda_index is None:
        cuda_index = torch.cuda.current_device()

    try:
        with cp.cuda.Device(cuda_index):
            with current_external_stream(cuda_index):
                _ct_partition_update(
                    cp.from_dlpack(active_flat),
                    cp.from_dlpack(idx_in),
                    cp.from_dlpack(idx_out),
                    cp.from_dlpack(count.reshape(1)),
                    cp.from_dlpack(entering.reshape(1)),
                    cp.from_dlpack(lengths_flat),
                    cp.from_dlpack(step_plus1.reshape(1)),
                    cp.from_dlpack(device_step.reshape(1)),
                )
    except ValueError:
        # CT contract violation (shape/dtype/aliasing) — a caller bug, not
        # host state. Surface it instead of silently latching the fast path
        # off (mirrors the sampling-kernel device-mismatch policy).
        raise
    except Exception:
        # NVRTC compile / launch / CT-version failures — host state; latch.
        _persistent_failure = True
        return None

    return True
