# -*- coding: utf-8 -*-
"""Shared runtime helpers for the measurement_adapter CuPy fused kernels.

Centralizes ``cp_from_torch`` (the DLPack torch->cupy zero-copy view) used
identically by the counter_rng / mask_counts / metadata / sampling kernels.

 scope note: the per-module fail-once latch triple (``_persistent_failure`` +
``fused_kernel_persistently_unavailable`` + ``reset_persistent_failure``) and the
per-module ``@lru_cache`` RawKernel factories are deliberately LEFT in their
modules. Each carries module-specific state — the latch is process-global PER
kernel and is imported BY NAME from gfn_runtime (``_fused_<X>_persistently_
unavailable``), and each factory holds a distinct kernel SOURCE/launch.
Centralizing them rewires the hot kernel functions' inline state mechanics, so it
is a separate, GPU-parity-gated follow-up rather than part of this pure-helper move.
"""
from __future__ import annotations

import torch


def cp_from_torch(tensor: torch.Tensor):
    """Zero-copy CuPy view of a CUDA torch tensor via DLPack."""
    import cupy as cp

    return cp.from_dlpack(tensor)
