# -*- coding: utf-8 -*-
"""Fused-kernel eligibility gates + policy-forward / GIPTE-capture methods.

These methods are variant-agnostic and call NO monkeypatched module global,
so they live in their own module; GFlowNetSamplingMixin inherits this mixin. GFlowNet state is resolved via self/MRO at call time (no gfn_core
import), and warning sites retain the public gfn_sampling logger name."""

import gc
import logging
import torch
from collections import OrderedDict
from typing import List, Tuple, Dict, Optional, Any

_LOGGER = logging.getLogger(__package__ or "gfn_sampling")

try:
    from ..gfn_runtime import (
        _SAMPLING_MODE_BUCKETED,
        BucketedGraphPreflightError,
        FlowMeasTableau,
        SamplingMode,
        masked_gumbel_argmax,
        _fused_sampling_persistently_unavailable,
        _compute_mask_counts_fused,
        _fused_mask_counts_persistently_unavailable,
        _fused_counter_rng_persistently_unavailable,
        _partition_update_bucketed_torch,
        _fused_partition_update_persistently_unavailable,
        _fused_apply_adapter,
        counter_uniforms,
        ordered_partition_scatter,
    )
    from ..gfn_trajectory import (
        TrajectoryBatch,
    )
except ImportError:  # pragma: no cover - direct-execution mode
    from gfn_runtime import (
        _SAMPLING_MODE_BUCKETED,
        BucketedGraphPreflightError,
        FlowMeasTableau,
        SamplingMode,
        masked_gumbel_argmax,
        _fused_sampling_persistently_unavailable,
        _compute_mask_counts_fused,
        _fused_mask_counts_persistently_unavailable,
        _fused_counter_rng_persistently_unavailable,
        _partition_update_bucketed_torch,
        _fused_partition_update_persistently_unavailable,
        _fused_apply_adapter,
        counter_uniforms,
        ordered_partition_scatter,
    )
    from gfn_trajectory import (
        TrajectoryBatch,
    )


class GatesPolicyMixin:
    """Gate, policy-forward, and GIPTE-capture methods split from gfn_sampling."""

    def _effective_fused_sampling_kernel(self) -> bool:
        """``use_fused_sampling_kernel`` ANDed with the per-process latch.

        Once a CuPy import / NVRTC compile / kernel launch fails, future
        sampling layers should skip the import + compile + launch-attempt
        overhead and go straight to the PyTorch fallback. We re-check the
        module-level latch at each call site so a hard failure in any
        prior call (e.g. inside a different ``GFlowNet`` instance sharing
        this process) is respected here too. Mirrors the
        ``_fused_metadata_kernel_failed`` pattern in
        ``apply_actions_to_batch``.
        """
        if not self.use_fused_sampling_kernel:
            return False
        if self._fused_sampling_kernel_failed:
            return False
        if _fused_sampling_persistently_unavailable():
            self._fused_sampling_kernel_failed = True
            if self.debug:
                logging.debug(
                    "fused sampling kernel latched off after persistent CuPy/NVRTC failure"
                )
            return False
        return True

    def _effective_fused_mask_counts_kernel(self) -> bool:
        """``use_fused_mask_counts_kernel`` ANDed with the per-process latch.

        Same fail-once pattern as ``_effective_fused_sampling_kernel``: once
        the CuPy import / NVRTC compile / launch fails, subsequent sampled
        layers should skip the fused-attempt overhead and use the PyTorch
        fallback directly.
        """
        if not self.use_fused_mask_counts_kernel:
            return False
        if self._fused_mask_counts_kernel_failed:
            return False
        if _fused_mask_counts_persistently_unavailable():
            self._fused_mask_counts_kernel_failed = True
            if self.debug:
                logging.debug(
                    "fused mask+counts kernel latched off after persistent CuPy/NVRTC failure"
                )
            return False
        return True

    def _effective_uint8_state_cache(self) -> bool:
        """``use_uint8_state_cache`` gated to the flattened-W feature mode.

        Only the default flattened-W policy input is exactly 0/1-valued, so
        the uint8 round-trip (``cache_step_data`` downcast +
        ``_forward_selected`` upcast) is bit-exact there. GIPTE feature
        extractors emit float features and packed-W caches packed integer
        tokens — both keep their existing cache dtype.
        """
        return (
            self.use_uint8_state_cache
            and self.feature_extractor is None
            and not self.packed_w_input
        )

    def _effective_fused_partition_update_kernel(self) -> bool:
        """``use_fused_partition_update_kernel`` ANDed with the per-process latch.

        Same fail-once pattern as the sibling fused kernels: once the CT
        import / NVRTC compile / launch fails, later sampled layers skip the
        attempt overhead and use the torch partition chain directly.
        """
        if not self.use_fused_partition_update_kernel:
            return False
        if self._fused_partition_update_kernel_failed:
            return False
        if _fused_partition_update_persistently_unavailable():
            self._fused_partition_update_kernel_failed = True
            if self.debug:
                logging.debug(
                    "fused partition-update kernel latched off after persistent failure"
                )
            return False
        return True

    def _effective_fused_counter_rng_kernel(self) -> bool:
        """``use_fused_counter_rng_kernel`` ANDed with the per-process latch.

        Same fail-once pattern as ``_effective_fused_sampling_kernel``: once
        the CuPy import / NVRTC compile / launch fails, subsequent sampled
        layers should skip the fused-attempt overhead and use the torch
        SplitMix64 chain directly.
        """
        if not self.use_fused_counter_rng_kernel:
            return False
        if self._fused_counter_rng_kernel_failed:
            return False
        if _fused_counter_rng_persistently_unavailable():
            self._fused_counter_rng_kernel_failed = True
            if self.debug:
                logging.debug(
                    "fused counter-RNG kernel latched off after persistent CuPy/NVRTC failure"
                )
            return False
        return True

    def _effective_fused_apply_kernel(self) -> bool:
        """``use_fused_apply_kernel`` ANDed with the per-process latch + the
        lowering-table availability.

        hot-path opt-in. Returns True only when:
          * the user-facing knob is on,
          * no per-instance latch has tripped,
          * the module-level latch is clear,
          * we're on CUDA (the CT API is CUDA-only),
          * the lowering table can be built (lazy, once).

        Same fail-once pattern as the sibling fused-kernel paths.
        """
        if not self.use_fused_apply_kernel:
            return False
        if self._fused_apply_kernel_failed:
            return False
        # CPU short-circuit MUST run before the
        # ``_fused_apply_adapter is None`` check. The fused path is
        # CUDA-only, so a CPU GFlowNet was never eligible — if the
        # adapter import failed in a CPU-only environment we don't
        # want to record a fused failure (``_fused_apply_kernel_failed
        # = True``) on the instance, which would persist on the GFN
        # and look like a tripped latch in telemetry.
        if self.device.type != "cuda":
            return False
        if _fused_apply_adapter is None:
            self._fused_apply_kernel_failed = True
            return False
        if _fused_apply_adapter.fused_kernel_persistently_unavailable():
            self._fused_apply_kernel_failed = True
            if self.debug:
                logging.debug(
                    "fused apply kernel latched off after persistent CuPy/NVRTC failure"
                )
            return False
        # Lazy lowering-table build: depends on action map + per-action
        # tensors + n_qubits, all immutable after ``GFlowNet.__init__``,
        # so one build is enough for the lifetime of the instance. If
        # build_lowering_table returns None (CT unavailable, etc.) latch
        # off so subsequent calls don't retry.
        if self._fused_apply_lowering is None:
            self._fused_apply_lowering = _fused_apply_adapter.build_lowering_table(
                action_map=self.action_mapping,
                n_qubits=self.n_qubits,
                action_gate_types=self.action_gate_types,
                action_qubit1=self.action_qubit1,
                action_qubit2=self.action_qubit2,
                single_qubit_mask=self.single_qubit_mask,
                two_qubit_mask=self.two_qubit_mask,
                device=self.device,
            )
            if self._fused_apply_lowering is None:
                self._fused_apply_kernel_failed = True
                if self.debug:
                    logging.debug(
                        "fused apply lowering-table build failed; latching off"
                    )
                return False
        return True

    def _effective_bucketed_graph(self) -> bool:
        # This gate MUST be a superset of every config/capability precondition that
        # _sample_trajectories_bucketed_graph() hard-requires (and would RAISE on),
        # so an opt-in bucketed run that is ineligible for graph capture degrades
        # gracefully to the eager bucketed sampler instead of crashing at runtime.
        # The graph path raises if the CT fused apply (~L4089), the CT fused
        # mask/counts kernel (~L4099, which also latches _fused_mask_counts_kernel_failed
        # so _effective_fused_mask_counts_kernel() returns False thereafter), or
        # batched_tableau.reset_inplace_with_mask (~L4168) are unavailable -- mirror
        # all three here. (CuPy import ~L4155 and the CT _sim surface ~L4175 are
        # covered transitively: no CuPy => fused apply unavailable; reset_inplace_with_mask
        # is CT-adapter-only and ships alongside _sim.)
        return (
            self._bucketed_use_graph
            and self._effective_sampling_mode == _SAMPLING_MODE_BUCKETED
            and self.device.type == 'cuda'
            and self.feature_extractor is None
            and not self.packed_w_input
            and hasattr(self._tableau_cls, 'reset_inplace_with_mask')
            and self._effective_fused_apply_kernel()
            and self._effective_fused_mask_counts_kernel()
        )

    def _policy_features(self, batched_tableau: FlowMeasTableau,
                         total_rows: int) -> torch.Tensor:
        """Full (static-shape) per-row policy input for ALL ``B*M`` rows.

        Default (no feature extractor): the flattened-W float32 tensor,
        ``(total_rows, (2n)^2)``. GIPTE mode: the
        gauge-invariant packed hit-feature set ``(total_rows, K, feature_dim)``
        from the injected extractor (no float32 (2n)^2 materialization, no
        canonicalization; fixed shape -> CUDA-graph friendly).
        """
        if self.feature_extractor is not None:
            feats, _ = self.feature_extractor.extract(batched_tableau, active_only=False)
            return feats
        if self.packed_w_input:
            # Bit-packed W read straight from CT (N, 2n, ceil(2n/32)) int32 — the
            # model unpacks on-GPU; the float (2n)^2 tensor is never built.
            return batched_tableau.policy_packed_w()
        states_full, _ = batched_tableau.to_flat_tensors()
        states = states_full.reshape(total_rows, -1).contiguous()
        states._source_ref = states_full
        return states

    def _policy_features_active(
        self, batched_tableau: FlowMeasTableau
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Active-row policy input + ``(n_active, 2)`` indices.

        Default: ``to_flat_tensors_active_only()`` (byte-identical). GIPTE mode:
        the extractor's active-only hit-feature set ``(n_active, K, feature_dim)``
        with the same ``(batch_idx, meas_idx)`` index map.
        """
        if self.feature_extractor is not None:
            return self.feature_extractor.extract(batched_tableau, active_only=True)
        if self.packed_w_input:
            # Packed W for the active rows + their (batch_idx, meas_idx) map.
            # CT returns all B*M rows; gather the active ones (cheap index).
            packed = batched_tableau.policy_packed_w()      # (B*M, 2n, ceil(2n/32))
            active = batched_tableau.active                  # (B, M) bool
            m = active.shape[1]
            flat_active = active.reshape(-1)
            idx = flat_active.nonzero(as_tuple=True)[0]
            indices = torch.stack([idx // m, idx % m], dim=1)
            return packed[idx], indices
        return batched_tableau.to_flat_tensors_active_only()

    def _policy_forward_static(self, states_tensor: torch.Tensor) -> torch.Tensor:
        """Run the policy on a fixed-shape tensor, using CUDA Graph replay when possible.

        NOTE: ``use_bf16_sampling`` is intentionally NOT honored here. The CUDA
        Graph capture/replay path records the model at fp32/TF32 and replays with
        the same dtype; wrapping the forward in ``torch.autocast`` would force a
        re-capture (or silently run fp32 on replay). bf16 sampling is applied
        only in the dynamic-active path (see ``sample_trajectories`` ~L2846), so
        ``use_bf16_sampling=True`` is a no-op when ``static_shape_sampling=True``.
        """
        if (
            not self.cuda_graph_sampling
            or self._policy_graph_failed
            or self.device.type != 'cuda'
            or torch.is_grad_enabled()
        ):
            return self.pf_model(states_tensor)

        cuda_index = (
            self.device.index
            if self.device.index is not None
            else torch.cuda.current_device()
        )
        key = (cuda_index, tuple(states_tensor.shape), states_tensor.dtype)
        entry = self._policy_graph_cache.get(key)
        # Pin every capture/replay path to ``self.device``. Without this,
        # capture on a process whose ambient ``current_device`` differs
        # from ``self.device`` (multi-GPU run, or any code that did
        # ``torch.cuda.set_device`` elsewhere) either fails or — worse —
        # records the graph against the wrong device, silently disabling
        # graph sampling for the rest of the run via ``_policy_graph_failed``.
        with torch.cuda.device(self.device):
            if entry is None:
                try:
                    static_input = torch.empty_like(states_tensor)
                    static_input.copy_(states_tensor)

                    warmup_stream = torch.cuda.Stream(device=self.device)
                    current_stream = torch.cuda.current_stream(self.device)
                    warmup_stream.wait_stream(current_stream)
                    with torch.cuda.stream(warmup_stream):
                        for _ in range(3):
                            self.pf_model(static_input)
                    current_stream.wait_stream(warmup_stream)

                    # Capture on the explicit warmup stream so the graph is
                    # bound to a stream we control on the target device.
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=warmup_stream):
                        static_output = self.pf_model(static_input)
                    entry = {
                        'graph': graph,
                        'input': static_input,
                        'output': static_output,
                    }
                    self._policy_graph_cache[key] = entry
                except Exception as e:
                    self._policy_graph_failed = True
                    if self.debug:
                        logging.debug(f"CUDA graph policy capture disabled: {e}")
                    return self.pf_model(states_tensor)

            entry['input'].copy_(states_tensor)
            entry['graph'].replay()
            return entry['output']

    def _policy_forward_dynamic(self, states_tensor: torch.Tensor) -> torch.Tensor:
        """Policy forward for the DYNAMIC-active sampling path → fp32 logits.

        Always honors ``use_bf16_sampling`` (the policy GEMMs run in bf16, logits
        cast back to fp32 for the fused gumbel kernel). When ``use_cuda_graph_policy``
        is on, n_active>0, and n_active <= ``_cuda_graph_policy_max_rows``, the input
        is padded up to the next power-of-two bucket and a per-bucket CUDA graph is
        replayed; the padded rows are sliced off the output. Rows are independent in
        the policy (LayerNorm normalizes within a row), so the bucketed result is
        numerically identical to
        the eager forward on the first n_active rows. Falls back to the eager bf16
        forward on capture failure (latched via ``_policy_graph_failed``), on CPU,
        under grad, when n_active==0, or above the row cap (where eager is faster).
        """
        bf16 = self.use_bf16_sampling
        n_active = int(states_tensor.shape[0])
        graph_requested = bool(self.use_cuda_graph_policy)

        def _eager(use_graph_impl: bool = False) -> torch.Tensor:
            fwd = self._dynamic_policy_forward_impl(use_graph_impl)
            if bf16:
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    out = fwd(states_tensor)
                return out.float()
            return fwd(states_tensor)

        if (
            not graph_requested
            or self._policy_graph_failed
            or self.device.type != 'cuda'
            or torch.is_grad_enabled()
            or n_active == 0
            or n_active > self._cuda_graph_policy_max_rows
        ):
            return _eager(False)

        # Next power of two >= n_active, capped at max_rows (>= n_active here, so the
        # bucket always covers n_active). Few distinct buckets => bounded graph cache.
        bucket = 1 << (n_active - 1).bit_length() if n_active > 1 else 1
        bucket = min(bucket, self._cuda_graph_policy_max_rows)
        tail_shape = tuple(int(s) for s in states_tensor.shape[1:])

        cuda_index = (
            self.device.index if self.device.index is not None
            else torch.cuda.current_device()
        )
        key = self._dynamic_policy_graph_cache_key(
            cuda_index,
            bucket,
            states_tensor,
            bf16,
        )
        fwd = self._dynamic_policy_forward_impl(True)

        with torch.cuda.device(self.device):
            entry = self._policy_graph_dyn_cache.get(key)
            if entry is None:
                try:
                    static_in = torch.empty((bucket, *tail_shape), dtype=states_tensor.dtype,
                                            device=self.device)
                    static_in[:n_active].copy_(states_tensor)
                    warmup = torch.cuda.Stream(device=self.device)
                    cur = torch.cuda.current_stream(self.device)
                    warmup.wait_stream(cur)
                    with torch.cuda.stream(warmup):
                        for _ in range(3):
                            # cache_enabled=False keeps the bf16 weight casts INSIDE
                            # the captured region so replay re-reads the LIVE fp32
                            # weights (they change every optimizer step) — otherwise a
                            # cached bf16 copy would freeze the policy at capture time.
                            with torch.autocast(device_type='cuda', dtype=torch.bfloat16,
                                                enabled=bf16, cache_enabled=False):
                                fwd(static_in)
                    cur.wait_stream(warmup)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=warmup):
                        with torch.autocast(device_type='cuda', dtype=torch.bfloat16,
                                            enabled=bf16, cache_enabled=False):
                            static_out = fwd(static_in)
                    entry = {'graph': graph, 'input': static_in, 'output': static_out}
                    self._policy_graph_dyn_cache[key] = entry
                except Exception as e:
                    self._policy_graph_failed = True
                    if self.debug:
                        logging.debug(f"CUDA graph policy (dynamic) capture disabled: {e}")
                    return _eager(False)

            entry['input'][:n_active].copy_(states_tensor)
            entry['graph'].replay()
            out = entry['output'][:n_active]
            # bf16 output: ``.float()`` returns a fresh fp32 tensor (a copy), so it
            # survives the next replay that overwrites the static output buffer.
            return out.float() if bf16 else out

    def _dynamic_policy_forward_impl(self, graph_forward_requested: bool):
        """Return the module used by the dynamic sampling policy forward.

        Capturing/replaying a ``torch.compile``'d reduce-overhead wrapper inside
        a manual CUDA graph can double-graph. Use the eager inner module only
        for actual capture/replay; fallbacks use the normal model path.
        """
        if graph_forward_requested:
            return getattr(self.pf_model, "_orig_mod", self.pf_model)
        return self.pf_model

    def _bucketed_policy_forward_impl(self):
        """Return the policy module used by the bucketed sampler.

        The per-K bucketed CUDA graph captures the eager inner module because
        capturing a ``torch.compile(..., mode='reduce-overhead')`` wrapper inside
        a manual CUDA graph can nest graph machinery. Use the same module for
        the eager compacted bucketed reference so bucketed graph capture remains
        a routing/perf knob rather than a compiled-vs-eager numerical knob.
        """
        return getattr(self.pf_model, "_orig_mod", self.pf_model)

    @staticmethod
    def _dynamic_policy_graph_cache_key(
        cuda_index: int,
        bucket: int,
        states_tensor: torch.Tensor,
        bf16: bool,
    ) -> Tuple[int, int, Tuple[int, ...], torch.dtype, bool]:
        """Cache key for dynamic policy graphs, preserving the full input tail."""
        tail_shape = tuple(int(s) for s in states_tensor.shape[1:])
        return (int(cuda_index), int(bucket), tail_shape, states_tensor.dtype, bool(bf16))

    def _gipte_capture_eligible(self) -> bool:
        """Whether the fused-step GIPTE CUDA graph (extraction+forward) can run.

        Requires the injected extractor, CUDA-graph sampling on a CUDA device, no
        prior graph failure, and inference (no grad — the captured forward is the
        sampling forward; training recompute uses the eager cached-flow path).
        """
        return (
            self.feature_extractor is not None
            and self.cuda_graph_sampling
            and self.device.type == 'cuda'
            and not self._policy_graph_failed
            and not torch.is_grad_enabled()
        )

    def _gipte_capture_logits(self, batched_tableau: FlowMeasTableau,
                              total_rows: int):
        """Replay (or capture) the fused GIPTE step graph; return ``(logits, H)``.

        Captures ``conjugate_into(static_w, dict) -> assemble H -> pf_model`` into
        one CUDA graph the first time per shape. Each call refreshes ``static_w``
        from the current sim via the adapter's ``policy_packed_w_into(out=...)``
        (which wraps CT ``get_W_bits_packed_u32(out=...)``) (eager, on the torch
        stream) then replays. ``logits`` and ``H`` are the graph's STATIC
        output buffers — the caller must clone ``H`` before the next step if it
        caches it. Returns ``None`` on any capture failure (caller falls back to
        the eager path); the failure latches off future attempts.
        """
        ext = self.feature_extractor
        n = self.n_qubits
        width = 2 * n
        row_words = (width + 31) // 32  # uint32 word-packed W rows
        K = ext.K
        cuda_index = (
            self.device.index if self.device.index is not None
            else torch.cuda.current_device()
        )
        key = (cuda_index, total_rows, K, ext.feature_dim)
        entry = self._gipte_graph_cache.get(key)

        def _refresh_static_w(static_w):
            # Refresh the static packed-W buffer the captured graph reads. Routed
            # through the adapter's public ``policy_packed_w_into`` (which wraps
            # the CT sim getter on the torch stream) so the write is ordered
            # before the replay and no caller reaches into ``_sim`` directly.
            batched_tableau.policy_packed_w_into(static_w)

        with torch.cuda.device(self.device):
            if entry is None:
                try:
                    # Narrow ImportError catch (fused-kernel discipline):
                    # a missing CuPy or a CT public-API break that drops this symbol
                    # latches the documented eager fallback and is logged at WARNING
                    # so it is visible in non-debug runs — not silently swallowed.
                    # CT is a pip-installed package, so the absolute import works in
                    # both run modes. Both names are used only within this capture
                    # block (and its _extract_forward closure).
                    import cupy as cp
                    from clifford_tableau.measurement import conjugate_dictionary_packed_into
                except ImportError as e:
                    self._policy_graph_failed = True
                    _LOGGER.warning(
                        "GIPTE fused-step capture deps unavailable (cupy/CT); "
                        "using eager path: %r", e)
                    return None
                try:
                    forward_fn = getattr(self.pf_model, "_orig_mod", self.pf_model)
                    dict_cp = ext.dict_packed_cupy()
                    with cp.cuda.Device(cuda_index):
                        static_w = cp.empty((total_rows, width, row_words), dtype=cp.uint32)
                        static_hit = cp.empty((total_rows, K), dtype=cp.uint8)
                        static_xw = cp.empty((total_rows, K), dtype=cp.int32)
                    t_hit = torch.from_dlpack(static_hit)
                    t_xw = torch.from_dlpack(static_xw)

                    def _extract_forward():
                        s = torch.cuda.current_stream(self.device)
                        with cp.cuda.Device(cuda_index), cp.cuda.ExternalStream(s.cuda_stream):
                            conjugate_dictionary_packed_into(
                                static_w, dict_cp, static_hit, static_xw, n
                            )
                        H = ext.assemble_static(t_hit, t_xw, total_rows)
                        return forward_fn(H), H

                    _refresh_static_w(static_w)
                    warm = torch.cuda.Stream(device=self.device)
                    cur = torch.cuda.current_stream(self.device)
                    warm.wait_stream(cur)
                    with torch.cuda.stream(warm):
                        for _ in range(3):
                            _extract_forward()
                    cur.wait_stream(warm)

                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=warm):
                        static_logits, static_H = _extract_forward()
                    # Retain EVERY static buffer the captured graph reads/writes.
                    # The graph replays against fixed addresses, so any buffer that
                    # is garbage-collected here would be freed and replayed into ->
                    # illegal memory access. dict_cp lives on the extractor; the
                    # rest are owned here.
                    entry = {
                        'graph': graph, 'static_w': static_w,
                        'static_hit': static_hit, 'static_xw': static_xw,
                        't_hit': t_hit, 't_xw': t_xw, 'dict_cp': dict_cp,
                        'logits': static_logits, 'H': static_H,
                    }
                    self._gipte_graph_cache[key] = entry
                except Exception as e:
                    # Broad catch is intentional here (allocations, from_dlpack,
                    # warmup, CUDAGraph capture can fail in many ways); all latch
                    # off to the parity-tested eager path. Log at WARNING (not
                    # debug-gated) so a real NVRTC/capture failure is visible.
                    self._policy_graph_failed = True
                    _LOGGER.warning(
                        "GIPTE fused-step CUDA graph capture failed; "
                        "falling back to eager path: %r", e)
                    return None

            _refresh_static_w(entry['static_w'])
            entry['graph'].replay()
            return entry['logits'], entry['H']
