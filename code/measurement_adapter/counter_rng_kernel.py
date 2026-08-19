"""CuPy fused counter-based uniform RNG kernel for the bucketed sampler.

The torch reference ``bucketed_sampler.counter_uniforms`` builds its SplitMix64
hash chain out of ~160 elementwise int64 kernels per call (each
``_splitmix64_finalizer`` is ~11 launches and the chain runs seven mixed
fields). That is the dominant per-layer launch count of the bucketed sampler —
more kernels per sampled layer than the entire dynamic-active path — and it
inflates every captured bucketed CUDA graph with ~160 interior nodes. This
module fuses the whole chain into ONE kernel: each thread computes the full
per-(row, action) hash from the five device-scalar key fields plus
``flat_idx[row]`` and the action id.

Bit-exactness contract: identical output to the torch reference for the same
inputs (``torch.equal``). SplitMix64 is exact unsigned-64 integer math; the
torch reference implements it in two's-complement int64 (same bit patterns),
and the final ``(mantissa in [1, 2^24]) -> float32 * 2^-24`` conversion is
exact in both.

Graph capture: the five scalar key fields are passed as device POINTERS, so a
captured launch re-reads their current contents on every replay (the same
freeze-hazard discipline as the torch reference's 0-dim-tensor path).

The fallback is the torch reference chain itself, so CPU and missing-CuPy
environments keep working bit-for-bit identically.
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
typedef unsigned long long u64;

__device__ __forceinline__ u64 splitmix64_fin(u64 x) {
    // SplitMix64 avalanche finalizer; mirrors bucketed_sampler._splitmix64_finalizer.
    u64 z = x ^ (x >> 30);
    z = z * 0xbf58476d1ce4e5b9ULL;
    z = z ^ (z >> 27);
    z = z * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

__device__ __forceinline__ u64 mix_field(u64 acc, u64 value, u64 salt) {
    // bucketed_sampler._mix_counter_field
    return splitmix64_fin(acc ^ splitmix64_fin(value + salt));
}

extern "C" __global__
void counter_uniforms_fused(
    const long long* __restrict__ seed,         // [1] device scalar
    const long long* __restrict__ train_step,   // [1] device scalar
    const long long* __restrict__ invocation,   // [1] device scalar
    const long long* __restrict__ ar_step,      // [1] device scalar
    const long long* __restrict__ rank,         // [1] device scalar
    const long long* __restrict__ flat_idx,     // [K]
    float* __restrict__ out,                    // [K * A] row-major
    const int K,
    const int A
) {
    const long long i = (long long) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long long) K * A) {
        return;
    }
    const int row = (int)(i / A);
    const int a = (int)(i % A);

    // _COUNTER_FIELD_SALTS, as uint64 bit patterns of the int64 constants.
    u64 h = splitmix64_fin((u64) seed[0] + 0x2468ace02468ace3ULL);
    h = mix_field(h, (u64) train_step[0], 0x13198a2e03707344ULL);
    h = mix_field(h, (u64) invocation[0], 0xa93469409949b499ULL);
    h = mix_field(h, (u64) ar_step[0],    0xc14dee063decdcd1ULL);
    h = mix_field(h, (u64) rank[0],       0x6a09e667f3bcc909ULL);
    h = mix_field(h, (u64) flat_idx[row], 0xbb67ae8584caa73bULL);
    h = mix_field(h, (u64) a,             0x3a191dfd62880ec7ULL);

    // High 24 bits -> float32 uniforms exactly in {1/2^24,..., 1}. The
    // mantissa is <= 2^24 so the u64 -> float conversion is exact, and the
    // 2^-24 scale is a power of two — both match the torch reference exactly.
    const u64 mantissa = (h >> 40) + 1ULL;
    out[i] = (float) mantissa * (1.0f / 16777216.0f);
}
'''


@lru_cache(maxsize=8)
def _kernel(cuda_index: int):
    import cupy as cp

    with cp.cuda.Device(cuda_index):
        return cp.RawKernel(_KERNEL_SRC, "counter_uniforms_fused")


# Shared DLPack torch->cupy view; centralized in _kernel_runtime.py.
try:
    from ._kernel_runtime import cp_from_torch as _cp_from_torch
except ImportError:  # pragma: no cover - direct-execution mode
    from _kernel_runtime import cp_from_torch as _cp_from_torch


# Persistent-failure latch: same contract as ``sampling_kernel`` — set only on
# host-state failures that WILL recur (CuPy import error, NVRTC compile error,
# RawKernel launch exception). Per-call bails (CPU tensors, opt-out) do not latch.
_persistent_failure: bool = False


def fused_kernel_persistently_unavailable() -> bool:
    """``True`` once a CuPy import / NVRTC compile / launch failure has been
    observed in this process (process-global, advisory for multi-GPU)."""
    return _persistent_failure


def reset_persistent_failure() -> None:
    """Clear the persistent-failure latch (host remediated / test isolation)."""
    global _persistent_failure
    _persistent_failure = False


def counter_uniforms_fused(
    seed: torch.Tensor,
    train_step: torch.Tensor,
    sample_invocation_id: torch.Tensor,
    ar_step: torch.Tensor,
    rank: torch.Tensor,
    flat_idx: torch.Tensor,
    n_actions: int,
    *,
    out: Optional[torch.Tensor] = None,
) -> Optional[torch.Tensor]:
    """Try the fused CUDA counter-uniforms kernel.

    All five scalar key fields must already be 0-dim ``torch.long`` CUDA
    tensors on ``flat_idx.device`` (the ``bucketed_sampler.counter_uniforms``
    wrapper normalizes them before calling here). Returns the ``[K, n_actions]``
    float32 uniforms on success, or ``None`` so the caller can fall back to the
    torch reference chain.

    ``out``, when given, must be a contiguous float32 ``[K, n_actions]`` CUDA
    tensor; the kernel writes it in place (graph-capture friendly: the captured
    launch keeps writing the same address) and it is also the return value.
    """
    global _persistent_failure
    if flat_idx.device.type != "cuda":
        return None
    if _persistent_failure:
        return None
    if not flat_idx.is_contiguous():
        return None

    try:
        import cupy as cp
    except (ImportError, OSError):
        _persistent_failure = True
        return None

    cuda_index = flat_idx.device.index
    if cuda_index is None:
        cuda_index = torch.cuda.current_device()

    k = int(flat_idx.shape[0])
    a = int(n_actions)
    if out is None:
        out = torch.empty((k, a), dtype=torch.float32, device=flat_idx.device)
    else:
        if out.shape != (k, a):
            raise ValueError(
                f"out must have shape {(k, a)}; got {tuple(out.shape)}"
            )
        if out.dtype != torch.float32 or not out.is_contiguous():
            raise ValueError("out must be a contiguous float32 tensor")
        if out.device != flat_idx.device:
            raise ValueError(
                f"out must be on {flat_idx.device}; got {out.device}"
            )
    total = k * a
    if total == 0:
        return out

    try:
        with cp.cuda.Device(cuda_index):
            with current_external_stream(cuda_index):
                kernel = _kernel(cuda_index)
                block = 256
                grid = ((total + block - 1) // block,)
                kernel_args = (
                    _cp_from_torch(seed.reshape(1)),
                    _cp_from_torch(train_step.reshape(1)),
                    _cp_from_torch(sample_invocation_id.reshape(1)),
                    _cp_from_torch(ar_step.reshape(1)),
                    _cp_from_torch(rank.reshape(1)),
                    _cp_from_torch(flat_idx),
                    _cp_from_torch(out.view(-1)),
                    int(k),
                    int(a),
                )
                kernel(grid, (block,), kernel_args)
    except Exception:
        # NVRTC compile failure or RawKernel launch error — host state, not
        # call shape. Latch so later calls skip the attempt overhead.
        _persistent_failure = True
        return None

    return out
