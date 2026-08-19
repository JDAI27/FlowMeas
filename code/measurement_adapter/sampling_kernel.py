"""CuPy fused masked Gumbel-max sampling kernel for GFN on-policy sampling.

The previous sampling path issued five to six CUDA launches per sampled layer
(``rand_like``, ``clamp_min``, two ``log`` ops, an add, and ``argmax``). The CT
math itself is cheap; this overhead dominates the baseline. This module
fuses the Gumbel transform plus masked argmax into a single CuPy kernel so the
sampling step issues one ``rand_like`` plus one fused launch instead.

The fallback executes the original PyTorch chain so CPU and missing-CuPy
environments still work bit-for-bit identically to the legacy path when given
the same uniforms.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Optional

import torch

try:
    from .cp_stream import current_external_stream
except ImportError:  # pragma: no cover - direct-execution mode
    from cp_stream import current_external_stream


_KERNEL_SRC = r'''
extern "C" __global__
void masked_gumbel_argmax(
    const float* __restrict__ logits,    // (N, A) row-major
    const bool*  __restrict__ mask,      // (N, A) row-major; true = valid
    const float* __restrict__ uniforms,  // (N, A) row-major; values in (0, 1]
    long long*   __restrict__ out,       // (N,)
    const int N,
    const int A,
    const int terminal_index
) {
    // WARP-PER-ROW (audit): one warp cooperatively reduces one row, vs the
    // legacy one-thread-per-row serial scan. The 32 lanes scan the action axis
    // strided (lane, lane+32,...), so consecutive lanes read consecutive
    // addresses => the logits/mask/uniforms reads are COALESCED across the warp
    // (the serial kernel's fixed-a reads were strided by A => uncoalesced). A
    // warp-shuffle reduction then picks the winner. Bit-identical to the serial
    // kernel for a given ``uniforms``: the per-(row,a) Gumbel values are the
    // same, and the reduction is max-by-value with a LOWEST-index tie-break
    // (associative), matching the serial ``v > best`` first-wins rule.
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;  // global warp id == row
    const int lane = threadIdx.x & 31;
    if (warp >= N) {
        return;
    }
    const int row = warp * A;

    // Initialize the fallback index to ``terminal_index`` so an all-masked row
    // (caller is supposed to avoid this, but mask bugs happen) returns terminal
    // rather than a stale gate id — same defensive posture as the PyTorch path.
    // NVRTC does not expose INFINITY without extra includes, so bit-cast -inf.
    float best = __int_as_float(0xff800000);
    long long best_idx = (long long) terminal_index;

    for (int a = lane; a < A; a += 32) {
        if (!mask[row + a]) {
            continue;
        }
        // Clamp uniforms away from zero before the double-log to avoid
        // ``log(0) = -inf`` propagating into the Gumbel sample (matches the
        // PyTorch ``u.clamp_min(1e-20)`` constant for parity).
        float u = uniforms[row + a];
        if (u < 1.0e-20f) {
            u = 1.0e-20f;
        }
        const float g = -logf(-logf(u));
        const float v = logits[row + a] + g;
        // Strict ``>`` keeps the LOWEST action index on ties within a lane (a
        // increases by 32 each step), matching the serial kernel + fallback.
        if (v > best) {
            best = v;
            best_idx = (long long) a;
        }
    }

    // Warp reduction into lane 0: maximize value, tie-break to the lowest index.
    // The comparator (max value, min-index tiebreak) is a strict total order on
    // (value, index) pairs, so the shuffle-down tree produces the correct global
    // winner: highest value wins, and on equal values the lowest action index
    // wins -- matching the serial kernel's first-wins rule.
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        const float    other = __shfl_down_sync(0xffffffff, best, offset);
        const long long oidx = __shfl_down_sync(0xffffffff, best_idx, offset);
        if (other > best || (other == best && oidx < best_idx)) {
            best = other;
            best_idx = oidx;
        }
    }
    if (lane == 0) {
        out[warp] = best_idx;
    }
}
'''


@lru_cache(maxsize=8)
def _kernel(cuda_index: int):
    import cupy as cp

    with cp.cuda.Device(cuda_index):
        return cp.RawKernel(_KERNEL_SRC, "masked_gumbel_argmax")


# Shared DLPack torch->cupy view; centralized in _kernel_runtime.py.
try:
    from ._kernel_runtime import cp_from_torch as _cp_from_torch
except ImportError:  # pragma: no cover - direct-execution mode
    from _kernel_runtime import cp_from_torch as _cp_from_torch


# Persistent-failure latch. Set only on host-state failures that WILL recur (CuPy
# import error, NVRTC compile error, RawKernel launch exception). Per-call bails —
# ``use_fused_kernel=False``, CPU tensors, non-contiguous inputs — do NOT set this,
# since those are call-shape dependent and may resolve next call. Callers should
# query ``fused_kernel_persistently_unavailable()`` and AND it into their own latch.
_persistent_failure: bool = False


def fused_kernel_persistently_unavailable() -> bool:
    """Return ``True`` once a CuPy import / NVRTC compile / launch failure
    has been observed for this process. Callers should latch this so the
    hot path stops paying repeated import/compile-attempt overhead.

    Note: the latch is process-global. A failure on one CUDA device
    disables the fused path for *all* devices in this process. Today's
    FlowMeas training uses one GPU per process so this is academic.
    Multi-GPU-per-process callers should treat the flag as advisory and
    test individual devices themselves if they need per-device behavior.
    """
    return _persistent_failure


def reset_persistent_failure() -> None:
    """Clear the persistent-failure latch.

    Public hook for callers that have remediated the underlying host
    state (re-installed CuPy, fixed CUDA visibility, rotated to a new
    process worker) and want to re-attempt the fused path without a
    full process restart. Also used by the test suite for isolation
    between latch-related cases.
    """
    global _persistent_failure
    _persistent_failure = False


def masked_gumbel_argmax(
    logits: torch.Tensor,
    mask: torch.Tensor,
    *,
    terminal_index: int,
    uniforms: Optional[torch.Tensor] = None,
    use_fused_kernel: bool = True,
) -> torch.Tensor:
    """Sample ``argmax(logits + Gumbel)`` over valid mask entries.

    Both the fused and fallback paths agree exactly when given the same
    ``uniforms`` tensor — the only difference is launch count.

    Args:
        logits: ``(N, A)`` float tensor of policy logits. Invalid slots may
            already be set to ``-inf``; this function does not re-mask them
            because the fused kernel sees the mask directly.
        mask: ``(N, A)`` boolean tensor; ``True`` = action is valid.
        terminal_index: Returned for rows whose mask is entirely ``False``.
            Defensive only — the policy contract is that terminal is always
            valid, so this fallback should not be reachable in production.
        uniforms: Optional ``(N, A)`` float tensor of pre-sampled uniforms in
            ``(0, 1]``. When ``None`` we draw with ``torch.rand_like`` on the
            same device. Pass an explicit tensor for tests that need a
            reproducible RNG stream.
        use_fused_kernel: When ``False``, force the PyTorch fallback path.
            The fused path will also fall back silently if CuPy or the kernel
            launch is unavailable.

    Returns:
        ``(N,)`` int64 tensor of sampled action indices.
    """
    if logits.dim() != 2 or mask.shape != logits.shape:
        raise ValueError(
            f"logits and mask must be 2-D with matching shape; "
            f"got logits={tuple(logits.shape)}, mask={tuple(mask.shape)}"
        )
    if mask.dtype != torch.bool:
        raise TypeError(f"mask must be torch.bool; got {mask.dtype}")
    # Validate devices up front. Catching the device-mismatch inside ``_try_fused``'s
    # broad ``except Exception`` would raise from the fallback with a worse traceback
    # AND latch the process-wide fused-failure flag for what is a caller bug. Raising
    # here keeps the latch reserved for genuine host-state failures.
    if mask.device != logits.device:
        raise ValueError(
            f"mask must be on the same device as logits; "
            f"got mask.device={mask.device}, logits.device={logits.device}"
        )

    n, a = logits.shape
    if n == 0:
        return torch.empty((0,), dtype=torch.long, device=logits.device)

    # Generate uniforms (if not supplied) in the caller's logit dtype so
    # the fallback path stays bit-identical to the legacy chain regardless
    # of input precision. The fused CuPy kernel requires float32 and gets
    # its own promoted views below.
    if uniforms is None:
        uniforms = torch.rand_like(logits)
    else:
        if uniforms.shape != logits.shape:
            raise ValueError(
                f"uniforms must match logits shape; "
                f"got uniforms={tuple(uniforms.shape)}, logits={tuple(logits.shape)}"
            )
        if uniforms.device != logits.device:
            raise ValueError(
                f"uniforms must be on the same device as logits; "
                f"got uniforms.device={uniforms.device}, "
                f"logits.device={logits.device}"
            )

    # The fused CuPy kernel signature takes ``const float*`` for both ``logits`` and
    # ``uniforms``, so both must be float32; otherwise route to the fallback so the
    # default ``use_fused_kernel=True`` still produces the caller's-dtype indices. A
    # float32-only ``logits`` check is not enough: a float64 ``uniforms`` passed
    # through DLPack is reinterpreted as ``float*`` inside the kernel, silently
    # corrupting the Gumbel transform. In production uniforms come from
    # ``torch.rand_like(logits)`` and already share the logit dtype.
    if logits.dtype != torch.float32 or uniforms.dtype != torch.float32:
        return _fallback(logits, mask, uniforms, terminal_index)

    out = _try_fused(logits, mask, uniforms, terminal_index, use_fused_kernel)
    if out is not None:
        return out

    return _fallback(logits, mask, uniforms, terminal_index)


def _try_fused(
    logits: torch.Tensor,
    mask: torch.Tensor,
    uniforms: torch.Tensor,
    terminal_index: int,
    use_fused_kernel: bool,
) -> Optional[torch.Tensor]:
    global _persistent_failure
    # Caller-driven opt-out: do NOT latch — the caller may flip the flag
    # back on for the next call.
    if not use_fused_kernel:
        return None
    # CPU tensors will never run on the CUDA kernel. This is a per-call
    # shape decision, not a host-state failure, so don't latch.
    if logits.device.type != "cuda":
        return None
    # Already-latched host-state failure: short-circuit without retrying.
    if _persistent_failure:
        return None

    try:
        import cupy as cp
    except (ImportError, OSError):
        # CuPy is absent from the environment (ImportError) or its shared
        # libraries cannot be loaded (OSError, e.g. ``libnvrtc.so.x not
        # found``); neither changes mid-process. A narrow except is used so a
        # genuine bug inside cupy (not its absence) propagates rather than
        # being silently latched as "cupy unavailable".
        _persistent_failure = True
        return None

    cuda_index = logits.device.index
    if cuda_index is None:
        # Tensor was allocated with ``torch.device('cuda')`` (no index), so PyTorch's
        # default device at allocation time determines where the storage lives. Use
        # that same default here so CuPy's ``Device(cuda_index)`` context binds to the
        # matching device.
        cuda_index = torch.cuda.current_device()

    n, a = logits.shape
    out = torch.empty((n,), dtype=torch.long, device=logits.device)

    # The kernel reads row-major as ``base + n*A + a``. Upstream producers
    # (``torch.rand_like``, model output, mask construction) already return contiguous
    # tensors, so the defensive ``.contiguous()`` calls are skipped in the hot path.
    # Contiguity is per-call shape, not host state — bail without latching so a later
    # contiguous call can still take the fast path.
    if not (logits.is_contiguous() and mask.is_contiguous() and uniforms.is_contiguous()):
        return None

    try:
        # CuPy was already successfully imported at the availability check
        # above; module-level caching makes this a no-op lookup, but we
        # rebind the alias here so the with-blocks read clearly.
        with cp.cuda.Device(cuda_index):
            with current_external_stream(cuda_index):
                kernel = _kernel(cuda_index)
                # Warp-per-row: launch one warp (32 threads) per row. block=256
                # => 8 warps/block, so grid covers ceil(N / 8) blocks.
                block = 256
                warps_per_block = block // 32
                grid = ((n + warps_per_block - 1) // warps_per_block,)
                kernel_args = (
                    _cp_from_torch(logits),
                    _cp_from_torch(mask),
                    _cp_from_torch(uniforms),
                    _cp_from_torch(out),
                    int(n),
                    int(a),
                    int(terminal_index),
                )
                kernel(grid, (block,), kernel_args)
    except Exception:
        # NVRTC compile failure or RawKernel launch error — both are host
        # state, not call shape. Latch so the next call skips the
        # import / compile / launch-attempt overhead and goes straight to
        # the PyTorch fallback.
        _persistent_failure = True
        return None

    return out


def _fallback(
    logits: torch.Tensor,
    mask: torch.Tensor,
    uniforms: torch.Tensor,
    terminal_index: int,
) -> torch.Tensor:
    # Bit-identical to the fused CuPy kernel when given the same ``uniforms`` — the
    # parity contract the public docstring advertises. Both this fallback and the
    # fused kernel implement one explicit contract for the edge cases the legacy
    # chain left to ``argmax`` quirks: masked rows and all-non-finite rows return
    # ``terminal_index``; otherwise pick the highest finite score.
    gumbel = -torch.log(-torch.log(uniforms.clamp_min(1e-20)))
    scores = logits.masked_fill(~mask, float("-inf")) + gumbel
    # NaN-safe argmax: PyTorch's CPU ``argmax`` treats NaN as greater than
    # any real, which would let a NaN-tainted logit poison the action
    # stream. The fused CUDA kernel naturally skips NaN via ``v > best``
    # (IEEE-754: NaN-vs-anything is False). Replace NaN with -inf in the
    # fallback to match.
    scores = torch.where(torch.isnan(scores), torch.full_like(scores, float("-inf")), scores)
    sampled = scores.argmax(dim=-1)
    # Rows whose post-mask, post-NaN-sanitize scores have no entry that would beat the
    # fused kernel's initial ``best = -inf`` return ``terminal_index``. This covers
    # three cases a bare ``argmax`` over all ``-inf`` would silently turn into index 0:
    #   1. Fully-masked rows (all entries ``-inf`` via ``masked_fill``).
    #   2. Rows with a valid entry but all valid logits ``-inf`` from upstream.
    #   3. Rows with a valid entry but all valid logits NaN (now ``-inf``).
    # The predicate must match the kernel's ``v > best`` rule exactly: ``scores > -inf``.
    # ``isfinite`` is wrong here — the kernel accepts ``+inf`` as a winning score
    # (e.g. ``uniforms==1.0`` makes Gumbel ``-log(-log(1)) = +inf``).
    has_winnable_score = (scores > float("-inf")).any(dim=-1)
    no_score = ~has_winnable_score
    return torch.where(
        no_score,
        torch.full_like(sampled, terminal_index),
        sampled,
    )
