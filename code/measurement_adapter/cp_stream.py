# -*- coding: utf-8 -*-
"""Cached torch-current-stream -> CuPy ``ExternalStream`` lookup.

Every fused CuPy kernel wrapper needs its launch enqueued on PyTorch's
CURRENT stream (so it orders correctly with surrounding torch kernels).
The historical per-call pattern

    torch_stream = torch.cuda.current_stream(device).cuda_stream
    with cp.cuda.ExternalStream(torch_stream):
...

allocates a fresh torch ``Stream`` wrapper AND a fresh CuPy
``ExternalStream`` wrapper on every call, which is a measurable share of a
training step. The raw stream pointer for a given
(device, stream) pair is stable for the stream's lifetime, so the wrapper
can be cached by pointer.

``ExternalStream`` does NOT own the underlying stream (no destruction on
GC), so caching is safe: even if torch ever tears a stream down and a new
one reuses the pointer value, the cached wrapper still denotes exactly
that pointer.

The pointer lookup uses ``torch._C._cuda_getCurrentRawStream`` — the same
private accessor torch.compile's generated wrappers call on every graph
launch, so it is de-facto stable — with the public
``torch.cuda.current_stream()`` as fallback.
"""

import torch

try:
    _get_raw_stream = torch._C._cuda_getCurrentRawStream
except AttributeError:  # pragma: no cover - very old torch
    _get_raw_stream = None

_STREAM_CACHE = {}


def current_external_stream(device_index: int):
    """Return a (cached) CuPy ``ExternalStream`` wrapping torch's current
    CUDA stream on ``device_index``.

    Caller must hold the GIL between this call and entering the returned
    context (always true for the synchronous kernel wrappers that use it).
    CuPy is imported lazily so importing this module never requires it.
    """
    import cupy as cp

    if _get_raw_stream is not None:
        ptr = _get_raw_stream(device_index)
    else:
        ptr = torch.cuda.current_stream(device_index).cuda_stream
    key = (device_index, ptr)
    stream = _STREAM_CACHE.get(key)
    if stream is None:
        stream = cp.cuda.ExternalStream(ptr)
        _STREAM_CACHE[key] = stream
    return stream
