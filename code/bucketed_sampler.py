"""Torch reference primitives for bucketed sampling.

These helpers preserve the FlowMeas flat row contract ``flat = batch * M + meas``
and the row ORDER of ``TableauBatchAdapter.to_flat_tensors_active_only`` (ascending flat
index, i.e. ``active.view(-1).nonzero()`` order) — load-bearing for the cache/loss contract.
``ActiveQueue``, ``build_fixed_k_features``, ``flat_to_bm``, and ``ordered_compact`` are
wired into ``GFlowNet._sample_trajectories_bucketed``. ``ordered_compact_scatter`` and
``counter_uniforms`` remain reference pieces for a device-driven loop and
graph-capture work.

Sync discipline (the whole point of bucketed sampling is to remove the per-layer host-sync):
  * ``build_fixed_k_features`` is **device-count-native** — it accepts ``active_count`` as a
    0-dim device tensor and derives ``row_valid`` on-device (``arange(K) < active_count``) with
    **no** ``.item()``.
  * ``ordered_compact`` is a **torch REFERENCE** whose boolean-index + ``.numel()`` materializes
    the count (a host-sync on CUDA). It is correctness-equivalent to a CUB
    ``DeviceSelect::Flagged`` with preallocated temp storage and a device-side next-count, and
    syncs no more than the dynamic path. Do NOT treat it as sync-free.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch

# Fused counter-RNG kernel (one launch vs the ~160-launch torch SplitMix64
# chain below). Package-relative with absolute fallback so both
# ``python3 code/run_config.py`` and ``python3 -m code.run_config`` modes work;
# narrow ``except ImportError`` so real NVRTC/syntax errors surface.
try:
    from .measurement_adapter import counter_rng_kernel as _counter_rng_kernel
except ImportError:
    try:
        from measurement_adapter import counter_rng_kernel as _counter_rng_kernel
    except ImportError:  # pragma: no cover - kernel module truly absent
        _counter_rng_kernel = None


CountLike = Union[int, torch.Tensor]


_SPLITMIX64_MUL1 = -4658895280553007687  # uint64 0xbf58476d1ce4e5b9 as int64
_SPLITMIX64_MUL2 = -7723592293110705685  # uint64 0x94d049bb133111eb as int64
_COUNTER_FIELD_SALTS = (
    2623536861626805475,   # seed domain
    1376283091369227076,   # train_step domain
    -6254258256340208487,  # sample_invocation_id domain
    -4517693140606591791,  # ar_step domain
    7640891576956012809,   # rank domain
    -4942790177534073029,  # flat_idx domain
    4186410302734601927,   # action_id domain
)
_FLOAT24_SCALE = 1 << 24


def _logical_right_shift_u64(x: torch.Tensor, bits: int) -> torch.Tensor:
    """Logical right shift for int64 tensors interpreted as uint64 lanes."""

    return torch.bitwise_and(torch.bitwise_right_shift(x, bits), (1 << (64 - bits)) - 1)


def _splitmix64_finalizer(x: torch.Tensor) -> torch.Tensor:
    """SplitMix64 avalanche finalizer in two's-complement int64 torch arithmetic."""

    z = torch.bitwise_xor(x, _logical_right_shift_u64(x, 30))
    z = z * _SPLITMIX64_MUL1
    z = torch.bitwise_xor(z, _logical_right_shift_u64(z, 27))
    z = z * _SPLITMIX64_MUL2
    return torch.bitwise_xor(z, _logical_right_shift_u64(z, 31))


def _as_long_scalar(value: Union[int, torch.Tensor], *, device: torch.device, name: str) -> torch.Tensor:
    if isinstance(value, int):
        return torch.tensor(value, dtype=torch.long, device=device)
    if isinstance(value, torch.Tensor):
        if value.ndim != 0:
            raise ValueError(f"{name} tensor must be 0-dimensional")
        if value.dtype != torch.long:
            raise TypeError(f"{name} tensor must have dtype torch.long")
        # Normalize device rather than raising on a label mismatch ('cuda' vs 'cuda:0' compare
        # unequal but are the same physical device). .to() is a no-op (identity) when already on
        # `device`, so this is graph-capture-safe (no copy node when matched).
        return value.to(device=device)
    raise TypeError(f"{name} must be an int or 0-dimensional torch.Tensor")


def _mix_counter_field(acc: torch.Tensor, value: torch.Tensor, salt: int) -> torch.Tensor:
    field = _splitmix64_finalizer(value + salt)
    return _splitmix64_finalizer(torch.bitwise_xor(acc, field))


def counter_uniforms(
    seed: Union[int, torch.Tensor],
    train_step: Union[int, torch.Tensor],
    sample_invocation_id: Union[int, torch.Tensor],
    ar_step: Union[int, torch.Tensor],
    rank: Union[int, torch.Tensor],
    flat_idx: torch.Tensor,
    n_actions: int,
    *,
    use_fused_kernel: bool = True,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Counter-based uniforms in ``(0, 1]`` with shape ``[K, n_actions]``.

    Values are keyed by ``(seed, train_step, sample_invocation_id, ar_step, flat_idx,
    action_id, rank)``. The bucket lane and padded capacity are deliberately absent from the key,
    so a logical ``(flat_idx, action_id)`` pair receives the same variate in any bucket layout.

    Scalar key fields may be Python ints or same-device 0-dim ``torch.long`` tensors. The tensor
    path is graph-capture oriented: the scalar buffers participate directly in tensor arithmetic,
    so replay reads their current contents by address instead of baking capture-time values.

    ``use_fused_kernel=True`` (default) routes CUDA inputs through the single-launch
    ``counter_rng_kernel`` (bit-identical to the torch chain below; falls back silently when
    CuPy / NVRTC is unavailable). ``out``, when given, must be a contiguous float32
    ``[K, n_actions]`` tensor on ``flat_idx.device``; both paths write it in place and return
    it — graph capture keeps a stable output address that way.
    """
    if flat_idx.ndim != 1:
        raise ValueError("flat_idx must be a 1-dimensional tensor")
    if flat_idx.dtype != torch.long:
        raise TypeError("flat_idx must have dtype torch.long")
    if n_actions < 0:
        raise ValueError("n_actions must be non-negative")

    device = flat_idx.device
    seed_t = _as_long_scalar(seed, device=device, name="seed")
    train_step_t = _as_long_scalar(train_step, device=device, name="train_step")
    sample_invocation_id_t = _as_long_scalar(
        sample_invocation_id,
        device=device,
        name="sample_invocation_id",
    )
    ar_step_t = _as_long_scalar(ar_step, device=device, name="ar_step")
    rank_t = _as_long_scalar(rank, device=device, name="rank")

    if (
        use_fused_kernel
        and _counter_rng_kernel is not None
        and device.type == "cuda"
    ):
        fused = _counter_rng_kernel.counter_uniforms_fused(
            seed_t,
            train_step_t,
            sample_invocation_id_t,
            ar_step_t,
            rank_t,
            flat_idx,
            n_actions,
            out=out,
        )
        if fused is not None:
            return fused

    h = _splitmix64_finalizer(seed_t + _COUNTER_FIELD_SALTS[0])
    h = _mix_counter_field(h, train_step_t, _COUNTER_FIELD_SALTS[1])
    h = _mix_counter_field(h, sample_invocation_id_t, _COUNTER_FIELD_SALTS[2])
    h = _mix_counter_field(h, ar_step_t, _COUNTER_FIELD_SALTS[3])
    h = _mix_counter_field(h, rank_t, _COUNTER_FIELD_SALTS[4])

    flat_field = flat_idx[:, None]
    action_field = torch.arange(n_actions, dtype=torch.long, device=device)[None, :]
    h = _mix_counter_field(h, flat_field, _COUNTER_FIELD_SALTS[5])
    h = _mix_counter_field(h, action_field, _COUNTER_FIELD_SALTS[6])

    # Use the high 24 bits to build float32 uniforms exactly in {1/2^24,..., 1}.
    mantissa = _logical_right_shift_u64(h, 40) + 1
    uniforms = mantissa.to(torch.float32) * (1.0 / _FLOAT24_SCALE)
    if out is not None:
        out.copy_(uniforms)
        return out
    return uniforms



def _count_to_int(active_count: CountLike, *, name: str = "active_count") -> int:
    """Materialize a count to a host int. NOTE: on CUDA tensors this is a HOST-SYNC; only call it
    in reference/validation paths, never in the per-layer hot loop (use the device-count path)."""
    if isinstance(active_count, int):
        count = active_count
    elif isinstance(active_count, torch.Tensor):
        if active_count.ndim != 0:
            raise ValueError(f"{name} tensor must be 0-dimensional")
        count = int(active_count.item())
    else:
        raise TypeError(f"{name} must be an int or 0-dimensional torch.Tensor")
    if count < 0:
        raise ValueError(f"{name} must be non-negative")
    return count


def _validate_capacity(length: int, count: int, *, name: str = "active_count") -> None:
    if count > length:
        raise ValueError(f"{name}={count} exceeds buffer length {length}")


@dataclass
class ActiveQueue:
    """Preallocated ordered active flat-id queue plus a ping-pong partner."""

    active_idx: torch.Tensor
    active_count: CountLike
    next_active_idx: torch.Tensor

    def __post_init__(self) -> None:
        if self.active_idx.ndim != 1:
            raise ValueError("active_idx must be a 1-dimensional tensor")
        if self.next_active_idx.shape != self.active_idx.shape:
            raise ValueError("next_active_idx must have the same shape as active_idx")
        if self.active_idx.dtype != torch.long:
            raise TypeError("active_idx must have dtype torch.long")
        if self.next_active_idx.dtype != torch.long:
            raise TypeError("next_active_idx must have dtype torch.long")
        if self.next_active_idx.device != self.active_idx.device:
            raise ValueError("next_active_idx must be on the same device as active_idx")
        _validate_capacity(self.active_idx.numel(), _count_to_int(self.active_count))

    @classmethod
    def from_mask(cls, mask: torch.Tensor, max_K: int) -> "ActiveQueue":
        """Build an ordered active queue from a ``[B, M]`` bool mask (flat nonzero order)."""

        if mask.ndim != 2:
            raise ValueError("mask must have shape [B, M]")
        if mask.dtype != torch.bool:
            raise TypeError("mask must have dtype torch.bool")
        if max_K < 0:
            raise ValueError("max_K must be non-negative")

        flat_active = mask.reshape(-1).nonzero(as_tuple=True)[0]
        active_count = flat_active.numel()
        if active_count > max_K:
            raise ValueError(
                f"active rows ({active_count}) exceed max_K capacity ({max_K})"
            )

        active_idx = torch.zeros(max_K, dtype=torch.long, device=mask.device)
        next_active_idx = torch.zeros_like(active_idx)
        if active_count:
            active_idx[:active_count].copy_(flat_active)

        return cls(
            active_idx=active_idx,
            active_count=active_count,
            next_active_idx=next_active_idx,
        )

    @property
    def active_slice(self) -> torch.Tensor:
        """Valid active flat-id prefix as a view. NOTE: resolving the count via _count_to_int is a
        host-sync on a CUDA device-count; for the 1.2b hot loop prefer the full buffer + row_valid
        gate (build_fixed_k_features), not this slice."""

        return self.active_idx[: _count_to_int(self.active_count)]


def build_fixed_k_features(
    full_rows: torch.Tensor,
    active_idx: torch.Tensor,
    active_count: CountLike,
    K: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Gather active rows into a fixed ``[K, feat]`` buffer plus a ``[K]`` ``row_valid`` mask.

    DEVICE-COUNT-NATIVE: ``active_count`` may be a 0-dim device tensor; ``row_valid`` is derived
    on-device as ``arange(K) < active_count`` with NO ``.item()``. Padded/invalid lanes are zeroed
    and ``row_valid=False``. Invalid lanes gather a safe in-range id (0) so a stale ``active_idx``
    tail can never cause an out-of-bounds gather.

    Row ORDER is identical to ``TableauBatchAdapter.to_flat_tensors_active_only`` (rows appear in
    ``flat = batch*M + meas`` ascending order), so the flat-index cache/loss contract is preserved.
    ``full_rows`` should already be float (the W-matrix view must be cast before this call); the
    output dtype follows ``full_rows``.

    PRECONDITION: ``active_count <= K``. Enforced when ``active_count`` is an int
    (free, host-side); for a device tensor it is DOCUMENTED, not checked, to avoid a host-sync
    (the boundary-driven K selection guarantees it). If violated with a tensor count, rows
    ``[K, active_count)`` are silently dropped.
    """
    if full_rows.ndim != 2:
        raise ValueError("full_rows must have shape [total_rows, feat]")
    if active_idx.ndim != 1:
        raise ValueError("active_idx must be a 1-dimensional tensor")
    if active_idx.dtype != torch.long:
        raise TypeError("active_idx must have dtype torch.long")
    if active_idx.device != full_rows.device:
        raise ValueError("active_idx must be on the same device as full_rows")
    if K < 0:
        raise ValueError("K must be non-negative")
    if K > active_idx.numel():
        raise ValueError(f"K={K} exceeds active_idx capacity {active_idx.numel()}")

    device = full_rows.device
    if isinstance(active_count, torch.Tensor):
        if active_count.ndim != 0:
            raise ValueError("active_count tensor must be 0-dimensional")
        if active_count.dtype != torch.long:
            raise TypeError("active_count tensor must have dtype torch.long")
        count = active_count.to(device=device)
        # Precondition active_count <= K is documented (not checked) on the device path: a host
        # check here would be exactly the per-layer sync this design removes.
    else:
        host_count = _count_to_int(active_count)
        if host_count > K:
            raise ValueError(
                f"active_count={host_count} > K={K}; capacity K must be >= active_count "
                "(capacity overflow invariant)"
            )
        count = torch.as_tensor(host_count, device=device, dtype=torch.long)

    row_valid = torch.arange(K, device=device) < count            # [K] bool, device-side, no sync
    if full_rows.shape[0] == 0:
        return full_rows.new_zeros((K, full_rows.shape[1])), torch.zeros(
            K,
            dtype=torch.bool,
            device=device,
        )
    idx_k = active_idx[:K]
    safe_idx = torch.where(row_valid, idx_k, torch.zeros_like(idx_k))  # invalid lanes -> row 0
    gathered = full_rows.index_select(0, safe_idx)                # [K, feat]
    out = gathered.masked_fill(~row_valid.unsqueeze(1), 0)        # zero padded/invalid lanes

    # Wired into GFNs._sample_trajectories_bucketed; padded lanes (row_valid=False)
    # must stay filtered from cache/loss-visible rows.
    return out, row_valid


def ordered_compact(
    active_idx: torch.Tensor,
    active_count: CountLike,
    survivor_mask: torch.Tensor,
    out_buf: torch.Tensor,
) -> Tuple[torch.Tensor, int]:
    """STABLE-select surviving active flat IDs into ``out_buf`` (preserving original order).

    REFERENCE IMPL — boolean indexing + ``.numel()`` materializes the count, a HOST-SYNC on CUDA.
    It is the correctness reference for a CUB ``DeviceSelect::Flagged`` replacement that
    would keep the next-count device-side. ``out_buf`` is fully zeroed first so lanes
    ``[next_count, max_K)`` never retain stale flat-ids (a stale tail would cause wrong-row gathers
    downstream). Invariant: ``next_count <= active_count``.
    """
    if active_idx.ndim != 1:
        raise ValueError("active_idx must be a 1-dimensional tensor")
    if out_buf.shape != active_idx.shape:
        raise ValueError("out_buf must have the same shape as active_idx")
    if active_idx.dtype != torch.long:
        raise TypeError("active_idx must have dtype torch.long")
    if out_buf.dtype != torch.long:
        raise TypeError("out_buf must have dtype torch.long")
    if out_buf.device != active_idx.device:
        raise ValueError("out_buf must be on the same device as active_idx")
    if survivor_mask.dtype != torch.bool:
        raise TypeError("survivor_mask must have dtype torch.bool")
    if survivor_mask.device != active_idx.device:
        raise ValueError("survivor_mask must be on the same device as active_idx")

    active_count_int = _count_to_int(active_count)
    _validate_capacity(active_idx.numel(), active_count_int)
    if survivor_mask.ndim != 1 or survivor_mask.numel() != active_count_int:
        raise ValueError("survivor_mask must have shape [active_count]")

    selected = active_idx[:active_count_int][survivor_mask]  # stable: preserves original order
    next_count = selected.numel()
    out_buf.zero_()                                          # clear stale tail (P0-2)
    if next_count:
        out_buf[:next_count].copy_(selected)

    return out_buf, next_count


def ordered_compact_scatter(
    active_idx: torch.Tensor,
    survivor_full: torch.Tensor,
    out_buf: torch.Tensor,
    out_count: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Stable, ORDER-PRESERVING device-side compaction over a fixed ``[max_K]`` buffer.

    ``active_idx`` is ``long[max_K]``; ``survivor_full`` is ``bool[max_K]`` and must already be
    gated with ``row_valid`` so lanes ``>= active_count`` are ``False``; ``out_buf`` is
    ``long[max_K]`` and is written in full; ``out_count`` is a long 0-dim device scalar written
    with the survivor count.

    Sync-free + graph-capturable stand-in for CUB ``DeviceSelect::Flagged``: NO ``.item()``, NO
    boolean indexing, and NO data-dependent output shapes. The bucketed loop will switch to this
    in a later sub-step once the row-valid-aware fixed-K kernel path lands.

    PRECONDITION: surviving ``active_idx`` values are unique. The active queue normally enforces
    this because each logical ``flat = batch * M + meas`` row appears at most once.
    """
    if active_idx.ndim != 1:
        raise ValueError("active_idx must be a 1-dimensional tensor")
    if survivor_full.shape != active_idx.shape:
        raise ValueError("survivor_full must have the same shape as active_idx")
    if out_buf.shape != active_idx.shape:
        raise ValueError("out_buf must have the same shape as active_idx")
    if active_idx.dtype != torch.long:
        raise TypeError("active_idx must have dtype torch.long")
    if survivor_full.dtype != torch.bool:
        raise TypeError("survivor_full must have dtype torch.bool")
    if out_buf.dtype != torch.long:
        raise TypeError("out_buf must have dtype torch.long")
    if out_count.ndim != 0:
        raise ValueError("out_count must be a 0-dimensional tensor")
    if out_count.dtype != torch.long:
        raise TypeError("out_count must have dtype torch.long")
    if survivor_full.device != active_idx.device:
        raise ValueError("survivor_full must be on the same device as active_idx")
    if out_buf.device != active_idx.device:
        raise ValueError("out_buf must be on the same device as active_idx")
    if out_count.device != active_idx.device:
        raise ValueError("out_count must be on the same device as active_idx")

    s = survivor_full.to(torch.long)
    positions = torch.cumsum(s, 0) - 1
    out_count.copy_(s.sum())
    out_buf.zero_()
    contrib = active_idx * s
    # If this scatter_add_ pattern is adapted to scatter by active_idx keys, duplicate survivor
    # keys would be summed instead of preserved as separate compacted rows.
    out_buf.scatter_add_(0, positions.clamp_min(0), contrib)
    return out_buf, out_count


def ordered_partition_scatter(
    active_idx: torch.Tensor,
    survivor_full: torch.Tensor,
    out_buf: torch.Tensor,
    out_count: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Stable, ORDER-PRESERVING device-side partition over a fixed ``[max_K]`` buffer.

    PARTITION variant of ``ordered_compact_scatter``: survivors are written to the front
    ``[0:count]`` and losers to the back ``[count:max_K]``, with both sides preserving original
    lane order. This preserves the queue-permutation invariant: ``active_idx`` is and remains a
    permutation of ``0..max_K-1`` across calls, so every fixed-``K`` prefix contains unique flats
    for any later ``K``.

    ``active_idx`` is ``long[max_K]``; ``survivor_full`` is ``bool[max_K]`` and must already be
    gated with ``row_valid`` so lanes ``>= active_count`` are ``False``; ``out_buf`` is
    ``long[max_K]`` and is written in full; ``out_count`` is a long 0-dim device scalar written
    with the survivor count.

    Sync-free + graph-capturable stand-in for CUB ``DevicePartition::Flagged``: NO ``.item()``,
    NO boolean indexing, and NO data-dependent output shapes. ``positions`` is a bijection of
    ``[0:max_K]`` (survivors fill ``[0:count]``; losers fill ``[count:max_K]``), so the scatter is
    a plain assignment. PRECONDITION: lanes ``>= active_count`` in ``survivor_full`` are already
    ``False`` so terminated/padding lanes flow to the back.
    """
    if active_idx.ndim != 1:
        raise ValueError("active_idx must be a 1-dimensional tensor")
    if survivor_full.shape != active_idx.shape:
        raise ValueError("survivor_full must have the same shape as active_idx")
    if out_buf.shape != active_idx.shape:
        raise ValueError("out_buf must have the same shape as active_idx")
    if active_idx.dtype != torch.long:
        raise TypeError("active_idx must have dtype torch.long")
    if survivor_full.dtype != torch.bool:
        raise TypeError("survivor_full must have dtype torch.bool")
    if out_buf.dtype != torch.long:
        raise TypeError("out_buf must have dtype torch.long")
    if out_count.ndim != 0:
        raise ValueError("out_count must be a 0-dimensional tensor")
    if out_count.dtype != torch.long:
        raise TypeError("out_count must have dtype torch.long")
    if survivor_full.device != active_idx.device:
        raise ValueError("survivor_full must be on the same device as active_idx")
    if out_buf.device != active_idx.device:
        raise ValueError("out_buf must be on the same device as active_idx")
    if out_count.device != active_idx.device:
        raise ValueError("out_count must be on the same device as active_idx")

    s = survivor_full.to(torch.long)
    ns = 1 - s
    pos_s = torch.cumsum(s, 0) - 1
    count = s.sum()
    pos_ns = count + torch.cumsum(ns, 0) - 1
    positions = torch.where(survivor_full, pos_s, pos_ns)
    out_count.copy_(count)
    out_buf.scatter_(0, positions, active_idx)
    return out_buf, out_count


def flat_to_bm(flat: torch.Tensor, M: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert flat row IDs to vectorized ``(batch_idx, meas_idx)`` tensors. Valid for M >= 1."""

    if M <= 0:
        raise ValueError("M must be positive")
    return torch.div(flat, M, rounding_mode="floor"), torch.remainder(flat, M)


def bm_to_flat(b: torch.Tensor, m: torch.Tensor, M: int) -> torch.Tensor:
    """Convert vectorized ``(batch_idx, meas_idx)`` tensors to flat row IDs. Valid for M >= 1."""

    if M <= 0:
        raise ValueError("M must be positive")
    return b * M + m
