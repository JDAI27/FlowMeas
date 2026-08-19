# -*- coding: utf-8 -*-
"""``GFlowNet`` core: construction, action application, update, replay,
checkpointing.

Split out of ``GFNs.py``. The class composes ``GFlowNetFlowsMixin`` and
``GFlowNetSamplingMixin``; all method bodies are verbatim from the
original module.
"""

import logging
import os
import time
import torch
from collections import OrderedDict
from collections import defaultdict
from typing import List, Tuple, Dict, Optional, Union, Callable, Any

try:
    from .gfn_runtime import (
        _SAMPLING_MODE_DYNAMIC_ACTIVE,
        _SAMPLING_MODE_STATIC_FULL,
        _SAMPLING_MODE_BUCKETED,
        _coerce_sampling_mode,
        _resolve_cuda_graph_policy_enabled,
        get_device,
        default_reward_fn,
        FlowMeasTableau,
        DiscreteUniform,
        create_clifford_model,
        create_gfn_objective,
        build_action_mapping,
        MaskingEngine,
        resolve_tableau_backend,
        apply_metadata_kernel,
        _fused_apply_adapter,
    )
    from .gfn_trajectory import (
        AdaptiveBufferTracker,
        TrajectoryBatch,
    )
    from .gfn_flows import GFlowNetFlowsMixin
    from .gfn_sampling import GFlowNetSamplingMixin
except ImportError:  # pragma: no cover - direct-execution mode
    from gfn_runtime import (
        _SAMPLING_MODE_DYNAMIC_ACTIVE,
        _SAMPLING_MODE_STATIC_FULL,
        _SAMPLING_MODE_BUCKETED,
        _coerce_sampling_mode,
        _resolve_cuda_graph_policy_enabled,
        get_device,
        default_reward_fn,
        FlowMeasTableau,
        DiscreteUniform,
        create_clifford_model,
        create_gfn_objective,
        build_action_mapping,
        MaskingEngine,
        resolve_tableau_backend,
        apply_metadata_kernel,
        _fused_apply_adapter,
    )
    from gfn_trajectory import (
        AdaptiveBufferTracker,
        TrajectoryBatch,
    )
    from gfn_flows import GFlowNetFlowsMixin
    from gfn_sampling import GFlowNetSamplingMixin


class GFlowNet(GFlowNetFlowsMixin, GFlowNetSamplingMixin):
    """
    GFlowNet with minimal CPU-GPU data transfer and depth-based sampling.

    batch_size = update_freq (number of batch elements)
    n_measurements = number of trajectories per batch element
    Total trajectories = batch_size × n_measurements
    """

    # OOM safeguard threshold: at/above this qubit count, a model that materializes
    # the dense (N, 2n, d) per-row token tensor is forced to checkpoint even if
    # ``use_activation_checkpointing`` resolved False. Light models are unaffected.
    _HEAVY_TOKEN_CHECKPOINT_QUBITS = 18

    # ---- Adaptive cached-flow ``chunk_size`` (L2-size-classified default) ----
    # Pure launch-count vs per-chunk-working-set knob, numerically equivalent at any
    # value. Large-L2 devices prefer the small chunk (a PyTorch caching-allocator
    # interaction), so L2 size is an empirical CLASSIFIER, not a mechanism claim.
    # FLOWMEAS_FLOW_CHUNK_SIZE overrides every branch.
    _CHUNK_L2_SENSITIVE_BYTES = 64 * 1024 * 1024  # L2 >=64 MB -> small-chunk regime
    _CHUNK_CACHE_FIT = 5000          # never-worst safe value; global fallback
    _CHUNK_BANDWIDTH_AMPLE = 200000  # big chunks: small-L2 devices
    _CHUNK_AMPLE_INPUT_BUDGET = 2_000_000_000  # cap so chunk*(2n)^2*4B stays bounded

    def __init__(self,
                n_qubits: int,
                hidden_dim: int,
                num_hidden_layers: int,
                lr: float = 1e-3,
                weight_decay: float = 1e-4,
                reward_fn: Optional[Callable] = None,
                device: Optional[torch.device] = None,
                model_type: str = 'clifford_mlp',
                model_kwargs: Optional[Dict] = None,
                objective_type: str = 'tb',
                objective_kwargs: Optional[Dict] = None,
                debug: bool = False,
                device_preference: Optional[str] = None,
                K: int = 5,
                buffer_strategy: str = 'conservative',  # Default to conservative
                adaptive_warmup: int = 100,
                measurement_backend: Optional[str] = None,
                static_shape_sampling: Optional[bool] = None,
                cuda_graph_sampling: Optional[bool] = None,
                sampling_mode: Optional[str] = None,
                use_fused_metadata_kernel: bool = True,
                use_fused_sampling_kernel: bool = True,
                use_fused_mask_counts_kernel: bool = True,
                use_fused_counter_rng_kernel: bool = True,
                use_fused_partition_update_kernel: bool = True,
                use_fused_apply_kernel: bool = True,
                use_activation_checkpointing: bool = True,
                use_uint8_state_cache: bool = True,
                use_bf16_sampling: bool = True,
                use_cuda_graph_policy: bool = False,
                cuda_graph_policy_max_rows: int = 2048,
                use_bf16_backward: bool = False,
                feature_extractor: Optional[Any] = None,
                packed_w_input: bool = False):

        # GIPTE opt-in: an injected TableauFeatureExtractor turning the batched
        # tableau into gauge-invariant, packed, static-shape hit features. When set,
        # pf_model MUST be a hit-feature model; None keeps the legacy flattened-W
        # float32 path. The boundary lives in ``_policy_features``.
        self.feature_extractor = feature_extractor

        # packed_w_rowtoken opt-in: the policy consumes the bit-packed W
        # ``(N, 2n, ceil(2n/32))`` straight from CT instead of flattened-W floats.
        # Mutually exclusive with ``feature_extractor``. CUDA-only; NOT
        # gauge-invariant.
        self.packed_w_input = packed_w_input
        if packed_w_input and feature_extractor is not None:
            raise ValueError(
                "packed_w_input and feature_extractor are mutually exclusive "
                "(packed raw W vs hit-features); set at most one."
            )
        _PACKED_W_MODELS = ("packed_w_rowtoken", "packed_w_split")
        if model_type in _PACKED_W_MODELS and not packed_w_input:
            raise ValueError(
                f"model_type={model_type!r} requires packed_w_input=True so "
                "the policy receives bit-packed W instead of the legacy flat-W tensor."
            )
        if packed_w_input and model_type not in _PACKED_W_MODELS:
            raise ValueError(
                "packed_w_input=True is only valid with "
                "model_type in ('packed_w_rowtoken', 'packed_w_split')."
            )
        if model_type in ("hit_mlp", "hit_deepsets") and feature_extractor is None:
            raise ValueError(
                f"model_type={model_type!r} requires a TableauFeatureExtractor "
                "via feature_extractor so the policy receives hit features."
            )
        if feature_extractor is not None and model_type not in ("hit_mlp", "hit_deepsets"):
            raise ValueError(
                "feature_extractor is only valid with model_type in "
                "{'hit_mlp', 'hit_deepsets'}."
            )

        # Activation checkpointing in the cached-flow gradient path. True (default)
        # drops per-step MLP activations and re-runs the forward during backward to
        # cap memory; False trades memory for one fewer policy forward per update.
        # Numerically identical either way — a perf knob, not a hyperparameter.
        self.use_activation_checkpointing = use_activation_checkpointing
        # uint8 cached-state compression (4x smaller cached flows — the
        # per-batch memory term that scales with trajectory row-steps).
        # Bit-exact for the flattened-W feature mode (values are exactly 0/1);
        # _effective_uint8_state_cache gates off GIPTE / packed-W modes.
        self.use_uint8_state_cache = bool(use_uint8_state_cache)

        # Store buffer strategy
        self.buffer_strategy = buffer_strategy
        self.adaptive_warmup = adaptive_warmup
        
        # Initialize conservative bound calculator
        self.conservative_multiplier = 1.2  # 20% safety margin
        
        # Initialize adaptive tracker if needed
        self.adaptive_tracker = None
        if self.buffer_strategy == 'adaptive':
            # Will be initialized when we know max_depth
            pass
        
        self.n_qubits = n_qubits
        self.device = device or get_device(device_preference)
        bucketed_seed = int(torch.initial_seed() & 0x7FFFFFFF)
        self._bucketed_seed_buf = torch.tensor(
            bucketed_seed, dtype=torch.long, device=self.device
        )
        self._bucketed_train_step_buf = torch.tensor(
            0, dtype=torch.long, device=self.device
        )
        self._bucketed_sample_invocation_buf = torch.tensor(
            0, dtype=torch.long, device=self.device
        )
        self._bucketed_rank_buf = torch.tensor(
            0, dtype=torch.long, device=self.device
        )
        # Phase-2 boundary-driven K: re-evaluate bucket capacity every
        # ``_bucketed_k_window`` layers (bucketed sampler only). Perf knob with no
        # training-output effect; not plumbed to ExperimentConfig.
        self._bucketed_k_window = 4
        # Bucketed CUDA graph capture is opt-in via cuda_graph_sampling.
        # Resolved after backend selection because it requires the CT tableau
        # backend plus default flattened-W features.
        self._bucketed_use_graph = False
        self._bucketed_graph_cache_enabled = True
        self._bucketed_graph_cache = OrderedDict()
        self._bucketed_graph_cache_max = 4
        self._bucketed_graph_capture_count = 0
        self.model_type = model_type
        self.debug = debug
        # The static CUDA sampler can help policy graph capture, but cached
        # training envelopes with rapidly decaying active rows spend far more
        # time on wasted full-row work than they save. Keep the path available
        # as an explicit opt-in; default to the active-row sampler so
        # cache_for_flows=True avoids the static full-shape regression.
        requested_sampling_mode = _coerce_sampling_mode(sampling_mode)
        legacy_static_requested = (
            False
            if static_shape_sampling is None
            else bool(static_shape_sampling)
        )
        if requested_sampling_mode is None:
            selected_sampling_mode = (
                _SAMPLING_MODE_STATIC_FULL
                if legacy_static_requested
                else _SAMPLING_MODE_DYNAMIC_ACTIVE
            )
        else:
            selected_sampling_mode = requested_sampling_mode
            if static_shape_sampling is not None:
                legacy_mode = (
                    _SAMPLING_MODE_STATIC_FULL
                    if legacy_static_requested
                    else _SAMPLING_MODE_DYNAMIC_ACTIVE
                )
                if legacy_mode != selected_sampling_mode:
                    logging.warning(
                        "sampling_mode=%r conflicts with "
                        "static_shape_sampling=%r (legacy mode %r); using "
                        "sampling_mode=%r",
                        sampling_mode,
                        static_shape_sampling,
                        legacy_mode,
                        selected_sampling_mode,
                    )

        self._requested_sampling_mode = (
            sampling_mode
            if sampling_mode is not None
            else selected_sampling_mode
        )
        self.static_shape_sampling = (
            selected_sampling_mode == _SAMPLING_MODE_STATIC_FULL
            if requested_sampling_mode is not None
            else legacy_static_requested
        )
        if selected_sampling_mode == _SAMPLING_MODE_BUCKETED:
            self._effective_sampling_mode = _SAMPLING_MODE_BUCKETED
        elif self.static_shape_sampling and self.device.type == 'cuda':
            self._effective_sampling_mode = _SAMPLING_MODE_STATIC_FULL
        else:
            self._effective_sampling_mode = _SAMPLING_MODE_DYNAMIC_ACTIVE
        self.sampling_mode = self._effective_sampling_mode

        graph_default = (
            self._effective_sampling_mode == _SAMPLING_MODE_STATIC_FULL
            and self.device.type == 'cuda'
        )
        resolved_cuda_graph_sampling = (
            graph_default
            if cuda_graph_sampling is None
            else bool(cuda_graph_sampling)
        )
        # bucketed is graph-eligible only as an explicit opt-in. NOTE: for bucketed,
        # self.cuda_graph_sampling is NON-FINAL here — the CT-backend/default-W
        # precondition below can demote it. Read self._bucketed_use_graph for the
        # authoritative runtime gate.
        graph_eligible = (
            self._effective_sampling_mode in (
                _SAMPLING_MODE_STATIC_FULL,
                _SAMPLING_MODE_BUCKETED,
            )
            and self.device.type == 'cuda'
        )
        self.cuda_graph_sampling = (
            resolved_cuda_graph_sampling
            and graph_eligible
        )
        if static_shape_sampling is True and self.device.type != 'cuda':
            logging.warning(
                "static_shape_sampling=True ignored because the static-shape "
                "sampler is CUDA-only"
            )
        if cuda_graph_sampling is True and not self.cuda_graph_sampling:
            if (
                selected_sampling_mode == _SAMPLING_MODE_BUCKETED
                and self.device.type != 'cuda'
            ):
                logging.warning(
                    "cuda_graph_sampling=True ignored: bucketed graph capture is "
                    "CUDA-only (device=%s)",
                    self.device.type,
                )
            else:
                logging.warning(
                    "cuda_graph_sampling=True ignored because policy graph replay "
                    "requires static_shape_sampling=True on CUDA"
                )
        # Surface sampling_mode='static_full' explicitly requested but not in effect
        # (CPU). The kwarg-only check above only inspects static_shape_sampling, so a mode-only
        # request is otherwise silent. Gated on requested_sampling_mode to avoid double-warning the
        # legacy-bool path.
        if (requested_sampling_mode == _SAMPLING_MODE_STATIC_FULL
                and self._effective_sampling_mode != _SAMPLING_MODE_STATIC_FULL):
            logging.warning(
                "sampling_mode='static_full' ignored because the static-shape sampler is "
                "CUDA-only (device=%s); using dynamic_active", self.device.type,
            )
        self._init_optimization_flags(
            use_fused_metadata_kernel=use_fused_metadata_kernel,
            use_fused_sampling_kernel=use_fused_sampling_kernel,
            use_fused_mask_counts_kernel=use_fused_mask_counts_kernel,
            use_fused_counter_rng_kernel=use_fused_counter_rng_kernel,
            use_fused_partition_update_kernel=use_fused_partition_update_kernel,
            use_bf16_sampling=use_bf16_sampling,
            use_bf16_backward=use_bf16_backward,
            use_fused_apply_kernel=use_fused_apply_kernel,
            use_cuda_graph_policy=use_cuda_graph_policy,
            cuda_graph_policy_max_rows=cuda_graph_policy_max_rows,
        )

        # Measurement backend selection is centralized in
        # ``measurement_adapter.resolve_tableau_backend`` so training and
        # evaluation cannot drift on the legacy-vs-CT validation rules.
        backend_selection = resolve_tableau_backend(measurement_backend, self.device)
        self.measurement_backend = backend_selection.name
        self._tableau_cls = backend_selection.tableau_cls

        # packed_w_rowtoken / GIPTE need the CT adapter surface. The CUDA-only guard
        # checks the device but not the resolved backend, so a non-CT backend on a
        # CUDA device would fail cryptically mid-training. Fail fast instead.
        if self.packed_w_input and not hasattr(self._tableau_cls, "policy_packed_w"):
            raise ValueError(
                "packed_w_rowtoken / packed_w_split (packed_w_input=True) require a measurement "
                "backend exposing policy_packed_w (the clifford-tableau adapter), "
                f"but the resolved backend {self.measurement_backend!r} "
                f"({self._tableau_cls.__name__}) does not. Use measurement_backend="
                "None or 'tableau_batch_adapter' on CUDA."
            )
        if self.feature_extractor is not None and not hasattr(self._tableau_cls, "hit_features"):
            raise ValueError(
                "GIPTE hit-feature models require a measurement backend exposing "
                "hit_features (the clifford-tableau adapter), but the resolved "
                f"backend {self.measurement_backend!r} ({self._tableau_cls.__name__}) "
                "does not. Use measurement_backend=None or 'tableau_batch_adapter' on CUDA."
            )
        if (
            self._effective_sampling_mode == _SAMPLING_MODE_BUCKETED
            and self.cuda_graph_sampling
        ):
            # Align the CT-backend capability signal with _effective_bucketed_graph's
            # gate so opt-in resolution and the call-time gate agree on the
            # reset/backend axis. Fused-kernel availability is call-time-only and can
            # latch off mid-run, so it is deliberately NOT probed here: the gate, not
            # this probe, is the safety boundary.
            ct_ok = hasattr(self._tableau_cls, 'reset_inplace_with_mask')
            feat_ok = self.feature_extractor is None and not self.packed_w_input
            if ct_ok and feat_ok:
                self._bucketed_use_graph = True
            else:
                self.cuda_graph_sampling = False
                self._bucketed_use_graph = False
                logging.warning(
                    "cuda_graph_sampling=True for bucketed requires the CT tableau "
                    "backend (tableau_batch_adapter) + default flattened-W features "
                    "(no feature_extractor / packed_w_input); falling back to the "
                    "eager bucketed sampler"
                )
        if backend_selection.auto_selected:
            logging.info(
                f"Measurement backend auto-selected: {self.measurement_backend} "
                f"(device={self.device})"
            )
        else:
            logging.info(f"Measurement backend: {self.measurement_backend}")
        
        # Set logging level based on debug flag
        if self.debug:
            logging.getLogger().setLevel(logging.DEBUG)
        
        logging.info(f"Using device: {self.device}")
        logging.info(f"Buffer strategy: {self.buffer_strategy}")
        if self.device.type == 'cuda':
            # ``FLOWMEAS_FLOW_DEDUP`` is an env-only emergency off-switch (no config
            # field); this is an init-time snapshot only -- the real gate is
            # re-read per call at the ``dedup_on`` site in compute_flows_cached.
            _dedup_env = os.environ.get('FLOWMEAS_FLOW_DEDUP', '1')
            logging.info(
                "CUDA sampling optimizations: "
                f"sampler={self._effective_sampling_mode}, "
                f"static_shape={self.static_shape_sampling}, "
                f"cuda_graph_sampling={self.cuda_graph_sampling}, "
                f"bucketed_graph={self._bucketed_use_graph}, "
                f"cuda_graph_policy={self.use_cuda_graph_policy}"
                f"{f' (max_rows={self._cuda_graph_policy_max_rows})' if self.use_cuda_graph_policy else ''}, "
                f"fused_metadata={self.use_fused_metadata_kernel}, "
                f"fused_sampling={self.use_fused_sampling_kernel}, "
                f"fused_mask_counts={self.use_fused_mask_counts_kernel}, "
                f"fused_counter_rng={self.use_fused_counter_rng_kernel}, "
                f"fused_partition_update={self.use_fused_partition_update_kernel}, "
                f"fused_apply={self.use_fused_apply_kernel}, "
                f"bf16_sampling={self.use_bf16_sampling}, "
                f"bf16_backward={self.use_bf16_backward}, "
                f"dedup_circuit_flows={_dedup_env != '0'} "
                f"(env FLOWMEAS_FLOW_DEDUP={_dedup_env!r}; set =0 to disable), "
                f"flow_chunk_size={self._default_chunk_size} (L2-adaptive; "
                f"env FLOWMEAS_FLOW_CHUNK_SIZE to override)"
            )
        if self.debug:
            logging.debug("Debug mode enabled")
        
        self.reward_fn = reward_fn or default_reward_fn
        
        objective_kwargs = objective_kwargs or {}
        self.objective = create_gfn_objective(objective_type, **objective_kwargs)
        self.objective_type = objective_type
        
        self.action_mapping = self._build_action_mapping()
        self.num_actions = len(self.action_mapping)
        
        # Pre-compute gate info first (needed by _precompute_gate_indices)
        self._precompute_gate_info()
        
        # Pre-compute gate type indices for GPU operations
        self._precompute_gate_indices()
        
        self.state_dim = (2 * n_qubits) ** 2  # Only W matrix (2nx2n Clifford tableau), no phase vector
        
        model_kwargs = model_kwargs or {}
        
        self.pf_model = create_clifford_model(
            model_type=model_type,
            n_qubits=n_qubits,
            hidden_dim=hidden_dim,
            num_hidden_layers=num_hidden_layers,
            output_dim=self.num_actions,
            **model_kwargs
        ).to(self.device)

        self.pb_model = DiscreteUniform(self.num_actions).to(self.device)

        self._init_model_weights()

        # Setup optimizer based on model type
        if hasattr(self.pf_model, 'logZ'):
            self.optimizer = torch.optim.Adam([
                {'params': self.pf_model.logZ, 'lr': 100*lr},
                {'params': [p for n, p in self.pf_model.named_parameters() if n != 'logZ'], 
                 'lr': lr , 'weight_decay': weight_decay}
            ])
        else:
            # For models without logZ (like DiscreteUniform)
            # Check if model has any parameters
            params = list(self.pf_model.parameters())
            if params:
                self.optimizer = torch.optim.Adam(
                    params, 
                    lr=lr, 
                    weight_decay=weight_decay
                )
            else:
                # No parameters to optimize (e.g., DiscreteUniform)
                # Create a dummy optimizer with no parameters
                self.optimizer = None
        
        self.grad_clip_value = 1e3  # Gradient clipping value

        # Apply torch.compile for GPU optimization (PyTorch 2.0+) AFTER optimizer setup
        if torch.__version__ >= '2.0.0' and self.device.type in ['cuda', 'mps']:
            try:
                # Unconditional ``reduce-overhead``. Switching the dynamic path to
                # ``default`` mode was measured to TRIPLE the per-step ATen-dispatch
                # count, so cudagraph_trees is collapsing real dispatches here.
                # Mode never changes numerics.
                compile_options = {
                    'mode': 'reduce-overhead',
                    'fullgraph': False  # Allow better handling of dynamic shapes
                }
                if self.device.type == 'cuda':
                    # Set CUDAGraph options to handle dynamic shapes better
                    import torch._inductor.config as config
                    config.triton.cudagraph_trees = True  # Use tree-based CUDAGraphs
                    # Skip dynamic graphs to avoid overhead (recommended for many shapes)
                    config.triton.cudagraph_skip_dynamic_graphs = True

                self.pf_model = torch.compile(self.pf_model, **compile_options)
                logging.info("Applied torch.compile optimization to forward model")
            except Exception as e:
                logging.warning(f"torch.compile not applied: {e}")

        # OOM safeguard. Models that materialize the dense (N, 2n, d) per-row token
        # tensor hold those activations across all cached steps when checkpointing is
        # off. The decision is made per-call in ``compute_flows_cached`` via
        # ``_effective_checkpoint``, which compares estimated retained-token bytes for
        # the actual cached row count against free GPU memory.

        # Store top trajectories as tensors to avoid CPU transfer
        self.top_trajectories_actions = []
        self.top_trajectories_lengths = []
        self.top_trajectories_rewards = []
        self.top_trajectories_costs = []
        self.K = K

    def _init_optimization_flags(
        self,
        *,
        use_fused_metadata_kernel,
        use_fused_sampling_kernel,
        use_fused_mask_counts_kernel,
        use_fused_counter_rng_kernel,
        use_fused_partition_update_kernel,
        use_bf16_sampling,
        use_bf16_backward,
        use_fused_apply_kernel,
        use_cuda_graph_policy,
        cuda_graph_policy_max_rows,
    ):
        """Initialize the per-step optimization flags, fail-once latches, bf16/
        chunk-size knobs, and CUDA-graph policy caches.

        Verbatim extract from __init__ (cold; runs once at construction). The
        _CHUNK_*/_HEAVY_TOKEN class constants intentionally stay on GFlowNet and
        are NOT moved (the memory-bound chunk heuristic resolves them via MRO)."""
        self.use_fused_metadata_kernel = bool(use_fused_metadata_kernel)
        self._fused_metadata_kernel_failed = False
        # Per-step fused Gumbel-max sampling: collapses the legacy rand→clamp
        # →log→log→add→argmax chain (5–6 launches) into one CuPy kernel.
        # Toggle off to force the PyTorch fallback (useful for parity tests
        # and CPU-only environments where the kernel can't run).
        self.use_fused_sampling_kernel = bool(use_fused_sampling_kernel)
        # Latch host-state failures (CuPy import / NVRTC compile / kernel launch) so
        # a broken CuPy stack stops paying repeated attempt cost on every sampled
        # layer. Re-checked at each sampling site so the module-level latch can flip
        # this on after the first hard failure.
        self._fused_sampling_kernel_failed = False
        # Fused active-mask + forward/backward valid-count CuPy kernel. Collapses the
        # legacy three-call chain in the cached on-policy sampler into a single
        # launch. Same fail-once latch as the other fused kernels.
        self.use_fused_mask_counts_kernel = bool(use_fused_mask_counts_kernel)
        self._fused_mask_counts_kernel_failed = False
        # Fused counter-RNG (bucketed sampler SplitMix64 uniforms): collapses
        # the ~160-launch torch hash chain into one CuPy kernel per sampled
        # layer. Bit-identical to the torch reference; same fail-once latch
        # discipline as the other fused kernels.
        self.use_fused_counter_rng_kernel = bool(use_fused_counter_rng_kernel)
        self._fused_counter_rng_kernel_failed = False
        # Phase-4 CT fused bucketed queue partition + lengths update: one CT
        # launch replaces the bucketed per-layer torch tail (~14 kernels in
        # every captured graph). Bit-compatible; torch chain stays as the
        # fallback. Same fail-once latch discipline as the other kernels.
        self.use_fused_partition_update_kernel = bool(use_fused_partition_update_kernel)
        self._fused_partition_update_kernel_failed = False
        # bf16 autocast on the DYNAMIC sampling policy forward only. Sampling runs
        # under no_grad and only feeds the gumbel-argmax action choice, so this shifts
        # the exploration distribution but NOT the gradients. The fused gumbel kernel
        # needs fp32, so the output is cast back. The OFF path is bit-identical.
        self.use_bf16_sampling = bool(use_bf16_sampling) and self.device.type == 'cuda'
        # (EXPERIMENTAL, default OFF): bf16 autocast on the GRADIENT-path
        # policy-MLP forward in ``_forward_selected`` -> bf16 backward GEMMs. This
        # perturbs the gradient (unlike no_grad sampling bf16), so it is opt-in and
        # must clear a convergence + gradient-parity study before any default flip.
        self.use_bf16_backward = bool(use_bf16_backward) and self.device.type == 'cuda'
        # One-shot latch for the legacy non-cached-flow bf16 warning (see compute_flows).
        self._warned_bf16_legacy_flow = False
        # Adaptive cached-flow chunk_size, resolved once from the GPU's L2 cache size
        # (see the class constants above + ``_resolve_default_chunk_size``). Used as the
        # default whenever ``compute_flows_cached`` is called without an explicit
        # ``chunk_size`` (i.e. every production caller). Bit-identical perf knob.
        self._default_chunk_size = self._resolve_default_chunk_size()
        # CT-side fused action lowering + primitive apply + metadata kernel, replacing
        # the legacy per-layer chain with two CT-side launches. Same fail-once latch;
        # the ``ActionLoweringTable`` depends only on construction-time state, so it
        # is built once on first use and cached across calls.
        self.use_fused_apply_kernel = bool(use_fused_apply_kernel)
        self._fused_apply_kernel_failed = False
        self._fused_apply_lowering = None
        # Count of successful fused-apply calls this process. Increments only when the
        # CT fused path actually ran AND mutated the tableau, so observers can tell
        # "fused apply was used" from "fused apply was available".
        self._fused_apply_call_count = 0
        self._policy_graph_cache = {}
        self._policy_graph_failed = False
        # Opt-in bucketed CUDA-graph replay for the DYNAMIC-active policy forward.
        # n_active is padded to the next power-of-two bucket with one graph per
        # bucket; rows are independent in the policy, so padding + output-slicing is
        # exact. Only launch-bound small layers are graphed — above ``max_rows`` the
        # forward is GPU-bound and replay is neutral-to-negative. Default OFF;
        # CUDA-only; bit-identical to the eager bf16 forward.
        self.use_cuda_graph_policy = _resolve_cuda_graph_policy_enabled(
            use_cuda_graph_policy,
            self.device.type,
            self.static_shape_sampling,
            self._effective_sampling_mode,
        )
        self._cuda_graph_policy_max_rows = max(1, int(cuda_graph_policy_max_rows))
        self._policy_graph_dyn_cache = {}
        # GIPTE fused-step graph cache: captures (packed conjugation -> assemble
        # hit-feature set -> policy forward) into one CUDA graph reading a STATIC
        # packed-W buffer refreshed each step. Keyed by (cuda_index, total_rows,
        # K, feature_dim). Shares the _policy_graph_failed latch.
        self._gipte_graph_cache = {}
        self._full_indices_cache = {}

        
    def _init_model_weights(self):
        """Initialize model weights for stability"""
        with torch.no_grad():
            for name, param in self.pf_model.named_parameters():
                if 'weight' in name and param.dim() > 1:
                    torch.nn.init.xavier_uniform_(param, gain=0.01)
                elif 'bias' in name:
                    torch.nn.init.zeros_(param)
            
            if hasattr(self.pf_model, 'logZ'):
                self.pf_model.logZ.data.fill_(0.0)
    
    def _build_action_mapping(self) -> Dict[int, Tuple]:
        """Build action mapping for quantum gates."""
        actions, terminal_index = build_action_mapping(self.n_qubits)
        self.terminal_index = terminal_index
        return actions

    def _precompute_gate_info(self):
        """Pre-compute gate type information"""
        self.single_qubit_gates = {"H", "S", "HS", "SH", "HSH"}
        self.two_qubit_gates = {"CNOT"} #"SWAP"}
        
    def _precompute_gate_indices(self):
        """Pre-compute gate indices for GPU operations"""
        # Create mappings from gate names to indices
        self.gate_name_to_idx = {
            "H": 0, "S": 1, "HS": 2, "SH": 3, "HSH": 4,
            "CNOT": 5, "terminal": 6
        }
        
        # Create tensors for fast GPU lookups
        self.action_gate_types = torch.zeros(self.num_actions, dtype=torch.long)
        self.action_qubit1 = torch.zeros(self.num_actions, dtype=torch.long)
        self.action_qubit2 = torch.zeros(self.num_actions, dtype=torch.long) - 1  # -1 for single qubit gates
        
        for idx, action in self.action_mapping.items():
            gate_name = action[0]
            self.action_gate_types[idx] = self.gate_name_to_idx[gate_name]
            
            if gate_name != "terminal":
                self.action_qubit1[idx] = action[1]
                if len(action) > 2:  # Two-qubit gate
                    self.action_qubit2[idx] = action[2]
        
        # Move to device
        self.action_gate_types = self.action_gate_types.to(self.device)
        self.action_qubit1 = self.action_qubit1.to(self.device)
        self.action_qubit2 = self.action_qubit2.to(self.device)
        
        # Create masks for gate types
        self.single_qubit_mask = torch.zeros(self.num_actions, dtype=torch.bool)
        self.two_qubit_mask = torch.zeros(self.num_actions, dtype=torch.bool)
        
        for idx, action in self.action_mapping.items():
            if action[0] in self.single_qubit_gates:
                self.single_qubit_mask[idx] = True
            elif action[0] in self.two_qubit_gates:
                self.two_qubit_mask[idx] = True
        
        self.single_qubit_mask = self.single_qubit_mask.to(self.device)
        self.two_qubit_mask = self.two_qubit_mask.to(self.device)

        # Initialize MaskingEngine for efficient mask computation
        self.masking_engine = MaskingEngine(
            n_qubits=self.n_qubits,
            num_actions=self.num_actions,
            action_gate_types=self.action_gate_types,
            action_qubit1=self.action_qubit1,
            action_qubit2=self.action_qubit2,
            single_qubit_mask=self.single_qubit_mask,
            two_qubit_mask=self.two_qubit_mask,
            terminal_index=self.terminal_index,
            device=self.device,
            debug=self.debug
        )

    def calculate_conservative_buffer_size(self, max_depth: int) -> int:
        """Conservative upper bound on the number of actions per trajectory.

        Under the strict 2q-only depth semantics, ``max_depth`` is the
        budget of 2q-gate layers; 1q gates are free with respect to depth. The
        ``last_single_qubit_gates[q] >= 0`` mask still bounds how many 1q gates
        a single qubit can fire: each qubit q can fire at most one 1q gate
        before requiring an intervening 2q gate on q. So in the worst case, q
        contributes (K_q + 1) one-qubit gates where K_q is the number of 2q
        gates involving q. Total 2q gates are bounded by
        ``n_qubits * max_depth // 2`` (n_qubits/2 per 2q layer), and total 1q
        gates are bounded by ``n_qubits + 2 * (#2q gates)``. The terminal
        sentinel adds one more slot.

        For ``max_depth == 0`` every 2q action is masked but the mask still
        allows the initial 1q gates plus terminal, so the buffer must hold at
        least ``n_qubits + 1`` actions.
        """
        two_qubit_bound = (self.n_qubits * max_depth) // 2
        one_qubit_bound = self.n_qubits + 2 * two_qubit_bound
        terminal_slot = 1
        worst_case = one_qubit_bound + two_qubit_bound + terminal_slot

        return int(self.conservative_multiplier * worst_case)
    
    def determine_buffer_size(self, max_depth: int) -> int:
        """Determine buffer size based on strategy."""
        if self.buffer_strategy == 'conservative':
            return self.calculate_conservative_buffer_size(max_depth)
        
        elif self.buffer_strategy == 'adaptive':
            # Initialize adaptive tracker on first use
            if self.adaptive_tracker is None:
                initial_size = self.calculate_conservative_buffer_size(max_depth)
                self.adaptive_tracker = AdaptiveBufferTracker(
                    initial_buffer_size=initial_size,
                    device=self.device,
                    warmup_updates=self.adaptive_warmup
                )
                logging.info(f"Initialized adaptive tracker with conservative bound: {initial_size}")
            
            return self.adaptive_tracker.get_recommended_buffer_size(max_depth)
        
        else:
            raise ValueError(f"Unknown buffer strategy: {self.buffer_strategy}")
    
    def apply_actions_to_batch(self, 
                                    batched_tableau: FlowMeasTableau,
                                    actions: torch.Tensor,
                                    trajectory_batch: TrajectoryBatch,
                                    step: Optional[Union[int, torch.Tensor]] = None) -> torch.Tensor:
        """Fully vectorized action application with depth tracking.

        ``step`` may be a CUDA torch scalar for the fused CT apply path.
        If a tensor step reaches the legacy metadata path, it is coerced
        with ``int(step)`` there, which performs a host sync and is not
        suitable for CUDA graph capture.
        """
        batch_size, n_measurements = actions.shape
        
        # GPU OPTIMIZATION: Double-buffered pre-allocation for safety + performance
        # Alternating buffers ensures previous results survive at least one more call
        if (hasattr(trajectory_batch, '_terminated_buffers') and 
            trajectory_batch._terminated_buffers[0].shape == (batch_size, n_measurements)):
            idx = trajectory_batch._terminated_buffer_idx
            trajectory_batch._terminated_buffer_idx = 1 - idx  # Alternate for next call
            terminated = trajectory_batch._terminated_buffers[idx].zero_()
        else:
            terminated = torch.zeros((batch_size, n_measurements), dtype=torch.bool, device=actions.device)
        
        # Get active trajectories. Don't sync-check ``active_mask.any`` here
        # — TBA's ``apply_actions_step`` already no-ops on an all-False mask
        # (the kernel reads ``active_mask`` per-row), and the downstream
        # ``len(active_indices) == 0`` provides the only return-early sync
        # we actually need.
        active_mask = trajectory_batch.active

        # Collapses the legacy
        # ``TableauBatchAdapter.apply_actions_step`` (LUT gather + 1-3
        # substep launches) + ``apply_metadata_kernel`` (1 launch) chain
        # into two CT-side CUDA kernels per layer. Falls back to the
        # legacy chain on CPU / missing-CuPy / latched failure / opt-out.
        if (
            self._effective_fused_apply_kernel()
            and hasattr(batched_tableau, "_sim")
        ):
            # ``validate_action_ids`` follows the tableau-adapter's backend knob. The
            # sampling hot path disables it (ids are built from this run's
            # ``action_map``, so the host-sync range-check is redundant); replay and
            # flow-reconstruction keep it on because they consume checkpoint-derived
            # action streams whose ids may no longer fit the current map. ``getattr``
            # defaults True for backends that lack the knob.
            tableau_validate = bool(
                getattr(batched_tableau, "validate_action_ids", True)
            )
            try:
                terminated_fused = _fused_apply_adapter.apply_action_layer_fused(
                    lowering=self._fused_apply_lowering,
                    trajectory_batch=trajectory_batch,
                    batched_tableau=batched_tableau,
                    terminated=terminated,
                    actions=actions,
                    terminal_index=self.terminal_index,
                    step=step,
                    validate_action_ids=tableau_validate,
                )
            except _fused_apply_adapter.FusedApplyRecoverableError:
                # Soft preflight failure (e.g. CuPy/DLPack OOM). NO CT-side
                # mutation occurred, so falling back to the legacy chain for this
                # call is safe. Do NOT set the per-instance latch — the failure is
                # transient and the next call should retry.
                if self.debug:
                    logging.debug(
                        "fused apply soft preflight failure; falling back "
                        "for this call without latching"
                    )
                terminated_fused = None
                # Fall through to the legacy chain WITHOUT setting
                # ``_fused_apply_kernel_failed``.
            else:
                if terminated_fused is not None:
                    # Bump the tableau-adapter's version counter so cache
                    # consumers don't miss the state mutation (the legacy
                    # chain bumps it inside ``apply_actions_step``).
                    if hasattr(batched_tableau, "version"):
                        batched_tableau.version += 1
                    # Record actual fused-apply usage.
                    self._fused_apply_call_count += 1
                    return terminated_fused
                # Hard preflight failure (CT unavailable / process latch / non-CUDA
                # / missing lowering table): no CT-side mutation occurred, so the
                # legacy chain is safe, and these conditions persist for this
                # trainer's lifetime, so latch this instance. Soft failures raise
                # ``FusedApplyRecoverableError`` (handled above, no latch); mid-call
                # failures raise ``FusedApplyMidCallError`` and propagate, since the
                # tableau may be partially mutated and re-running would double-apply.
                self._fused_apply_kernel_failed = True
                if self.debug:
                    logging.debug(
                        "fused apply kernel returned None (hard preflight); "
                        "latching off this instance"
                    )

        legacy_step = int(step) if torch.is_tensor(step) else step

        # Apply all gates in a single batched call
        batched_tableau.apply_actions_step(actions, self.action_mapping, active_mask)

        if (
            self._effective_fused_metadata_kernel()
            and apply_metadata_kernel is not None
            and actions.device.type == 'cuda'
        ):
            fused_terminated = apply_metadata_kernel(
                actions=actions,
                trajectory_batch=trajectory_batch,
                batched_tableau=batched_tableau,
                action_gate_types=self.action_gate_types,
                action_qubit1=self.action_qubit1,
                action_qubit2=self.action_qubit2,
                action_is_single=self.single_qubit_mask,
                action_is_two=self.two_qubit_mask,
                terminal_index=self.terminal_index,
                step=legacy_step,
            )
            if fused_terminated is not None:
                return fused_terminated
            # ``apply_metadata_kernel`` returns None for a host-state failure, a rare
            # soft bail (shape mismatch), or a transient OOM. We latch THIS INSTANCE
            # on any None: a shape mismatch is structural and persistent, and a
            # transient OOM only costs this instance the fused path. Other instances
            # still attempt it (no process-global latch is set on OOM).
            self._fused_metadata_kernel_failed = True
            if self.debug:
                logging.debug("Fused metadata kernel unavailable; falling back to PyTorch metadata updates")

        # Create a 2D view for easier indexing
        flat_active = active_mask.view(-1)
        flat_actions = actions.view(-1)
        active_indices = flat_active.nonzero(as_tuple=True)[0]

        if len(active_indices) == 0:
            return terminated
        
        # Get active actions
        active_actions = flat_actions[active_indices]
        
        # Convert flat indices back to 2D indices
        batch_indices = active_indices // n_measurements
        meas_indices = active_indices % n_measurements
        
        # Check for terminal actions. No ``.any`` guard around the scatter
        # writes: PyTorch's indexed assignment is a no-op on an empty index
        # tensor, so a guard would only cost a host sync per layer.
        is_terminal = active_actions == self.terminal_index
        term_batch = batch_indices[is_terminal]
        term_meas = meas_indices[is_terminal]
        terminated[term_batch, term_meas] = True
        trajectory_batch.active[term_batch, term_meas] = False
        batched_tableau.active[term_batch, term_meas] = False

        # Process non-terminal actions. Don't ``.any``-short-circuit the
        # tail: the indexed reads / writes below are no-ops on empty inputs
        # (kernels launch but do no per-element work), which is cheaper than
        # the host sync the guard would force.
        non_terminal_mask = ~is_terminal
        nt_actions = active_actions[non_terminal_mask]
        nt_batch = batch_indices[non_terminal_mask]
        nt_meas = meas_indices[non_terminal_mask]
        
        # Get gate types and qubits for all non-terminal actions
        gate_types = self.action_gate_types[nt_actions]
        qubit1 = self.action_qubit1[nt_actions]
        qubit2 = self.action_qubit2[nt_actions]
        
        # Determine which actions are single vs two-qubit
        is_single = self.single_qubit_mask[nt_actions]
        is_two = self.two_qubit_mask[nt_actions]
        
        # 1q gates never trigger needs_new_layer and never mark current_layer_qubits;
        # only 2q gates bump depth. A 2q gate opens a new layer when its qubits
        # conflict with the current 2q layer, or when no 2q layer is open yet. Index
        # splits are still needed for the per-gate-type metadata writes below.
        single_idx = is_single.nonzero(as_tuple=True)[0]
        single_batch = nt_batch[single_idx]
        single_meas = nt_meas[single_idx]
        single_q = qubit1[single_idx]

        two_idx = is_two.nonzero(as_tuple=True)[0]
        two_batch = nt_batch[two_idx]
        two_meas = nt_meas[two_idx]
        two_q1 = qubit1[two_idx]
        two_q2 = qubit2[two_idx]
        q1_used = trajectory_batch.current_layer_qubits[two_batch, two_meas, two_q1]
        q2_used = trajectory_batch.current_layer_qubits[two_batch, two_meas, two_q2]
        two_depths_pre = trajectory_batch.circuit_depths[two_batch, two_meas]
        needs_new_two = q1_used | q2_used | (two_depths_pre == 0)

        # Update depths for trajectories needing a new layer (no.any guard — empty
        # index tensors are scatter no-ops). ``needs_new_layer`` is only ever True at
        # ``two_idx[needs_new_two]``, so use that index directly and skip the
        # (B,M)-bool alloc, the scatter-write, and the ``.nonzero`` host-sync.
        new_layer_idx = two_idx[needs_new_two]
        new_layer_batch = nt_batch[new_layer_idx]
        new_layer_meas = nt_meas[new_layer_idx]
        trajectory_batch.circuit_depths[new_layer_batch, new_layer_meas] += 1
        trajectory_batch.current_layer_qubits[new_layer_batch, new_layer_meas] = False
        
        # Get current depths for all non-terminal trajectories
        current_depths = trajectory_batch.circuit_depths[nt_batch, nt_meas]
        
        # Reuse the single/two indices and corresponding qubits computed
        # above; redundant ``.nonzero`` calls were each a host sync and
        # also wasted work re-computing the same masks.
        single_gate_types = gate_types[single_idx]
        single_depths = current_depths[single_idx]
        two_depths = current_depths[two_idx]

        # Update qubit last use step if step is provided. Empty index
        # tensors make these scatter writes no-ops automatically — no
        # ``.any`` guard needed.
        if legacy_step is not None:
            trajectory_batch.qubit_last_use_step[single_batch, single_meas, single_q] = legacy_step
            trajectory_batch.action_qubits[single_batch, single_meas, legacy_step, 0] = single_q

            trajectory_batch.qubit_last_use_step[two_batch, two_meas, two_q1] = legacy_step
            trajectory_batch.qubit_last_use_step[two_batch, two_meas, two_q2] = legacy_step
            trajectory_batch.action_qubits[two_batch, two_meas, legacy_step, 0] = two_q1
            trajectory_batch.action_qubits[two_batch, two_meas, legacy_step, 1] = two_q2

        # Update gate tracking for single-qubit gates.
        # 1q gates do NOT mark current_layer_qubits — that field
        # now tracks 2q-layer occupancy only.
        trajectory_batch.last_single_qubit_gates[single_batch, single_meas, single_q] = single_gate_types
        trajectory_batch.qubit_last_layer[single_batch, single_meas, single_q] = single_depths

        # Clear last-single tracking on the affected qubits so single-qubit
        # gates immediately after a two-qubit gate are allowed.
        trajectory_batch.last_single_qubit_gates[two_batch, two_meas, two_q1] = -1
        trajectory_batch.last_single_qubit_gates[two_batch, two_meas, two_q2] = -1
        trajectory_batch.current_layer_qubits[two_batch, two_meas, two_q1] = True
        trajectory_batch.current_layer_qubits[two_batch, two_meas, two_q2] = True
        trajectory_batch.qubit_last_layer[two_batch, two_meas, two_q1] = two_depths
        trajectory_batch.qubit_last_layer[two_batch, two_meas, two_q2] = two_depths

        return terminated
    
    def update_step(self, accumulated_loss: torch.Tensor,
                    return_tensor: bool = False) -> Union[float, torch.Tensor]:
        """Perform a single gradient update step."""
        # Counter-RNG train_step is a call-count key for bucketed sampling. Increment once per
        # update_step invocation, even if the optimizer later skips a non-finite/no-op batch.
        with torch.no_grad():
            self._bucketed_train_step_buf.add_(1)

        # Skip update if no optimizer (e.g., DiscreteUniform model)
        if self.optimizer is None:
            zero = torch.zeros((), device=self.device)
            return zero if return_tensor else 0.0

        accumulated_loss.backward()

        # Single fused finiteness guard, synced ONCE per update, covering BOTH the
        # loss and every gradient — an additive non-finite constant can leave
        # parameter grads finite, so the loss check alone is not enough.
        # ``torch._foreach_norm`` reduces all grads in one fused multi-tensor kernel
        # and a grad is all-finite iff its L2 norm is finite. It covers ``logZ``
        # while ``clip_grad_norm_`` below excludes it; the more-inclusive check is
        # the safe side. Falls back to a per-parameter reduction if unavailable.
        loss_finite = torch.isfinite(accumulated_loss.detach()).all()
        grads = [p.grad for p in self.pf_model.parameters() if p.grad is not None]
        grad_nonzero = None
        if not grads:
            update_finite = loss_finite
        elif hasattr(torch, "_foreach_norm"):
            norms = torch.stack(torch._foreach_norm(grads))
            grad_finite = torch.isfinite(norms).all()
            update_finite = loss_finite & grad_finite
            grad_nonzero = norms.sum() > 0
        else:
            norms = torch.stack([g.norm() for g in grads])
            grad_finite = torch.isfinite(norms).all()
            update_finite = loss_finite & grad_finite
            grad_nonzero = norms.sum() > 0
        # All-zero gradient => no-op batch: skip the optimizer step. A fully masked
        # batch gives a graph-CONNECTED 0 loss, so backward fills every ``.grad`` with
        # zeros rather than None; stepping would still apply Adam momentum and
        # decoupled weight decay on a batch that must be a no-op. Folded into the same
        # host sync, reusing ``norms``. ``grad_nonzero is None`` already no-ops inside
        # ``optimizer.step`` because every param has ``grad is None``.
        should_step = update_finite if grad_nonzero is None else (update_finite & grad_nonzero)
        if not bool(should_step.item()):  # the only finiteness/no-op host sync
            if not bool(update_finite):
                logging.warning("NaN/Inf detected in loss/gradients, skipping update")
            self.optimizer.zero_grad()
            zero = torch.zeros((), device=self.device)
            return zero if return_tensor else 0.0

        # After ``torch.compile`` the parameter name becomes ``_orig_mod.logZ``, so
        # match ``n.endswith('logZ')`` to cover any prefix wrapper. The filter is
        # static after construction, so cache the filtered Parameter references once
        # (compile wraps rather than copies, so they stay valid) and re-evaluate only
        # the dynamic ``p.grad is not None`` predicate per step. Lazy so the cache
        # captures the final pf_model after any compile/load_checkpoint wrapping.
        if getattr(self, "_clip_param_refs", None) is None:
            self._clip_param_refs = [
                p for n, p in self.pf_model.named_parameters()
                if not n.endswith('logZ')
            ]
        params_to_clip = [p for p in self._clip_param_refs if p.grad is not None]
        torch.nn.utils.clip_grad_norm_(params_to_clip, self.grad_clip_value)
        
        self.optimizer.step()
        self.optimizer.zero_grad()
        
        loss_detached = accumulated_loss.detach()
        return loss_detached if return_tensor else loss_detached.item()
    
    def _update_top_trajectories(self, trajectory_batch: TrajectoryBatch, costs: torch.Tensor):
        """Update top-K replay buffer with one tiny CPU transfer for selected indices.

        Costs stay on-device; no per-batch-element ``.item`` / ``.any`` sync.
        """
        batch_size = trajectory_batch.batch_size

        candidate_actions = list(self.top_trajectories_actions)
        candidate_lengths = list(self.top_trajectories_lengths)
        for b_idx in range(batch_size):
            candidate_actions.append(trajectory_batch.actions[b_idx].clone())
            candidate_lengths.append(trajectory_batch.lengths[b_idx].clone())

        existing_costs = []
        for cost in self.top_trajectories_costs:
            if torch.is_tensor(cost):
                existing_costs.append(cost.detach().to(self.device).reshape(()))
            else:
                existing_costs.append(torch.tensor(float(cost), device=self.device))
        existing_costs_tensor = (
            torch.stack(existing_costs)
            if existing_costs
            else torch.empty(0, device=self.device, dtype=costs.dtype)
        )

        valid_batch = (trajectory_batch.lengths > 0).any(dim=1)
        inf_costs = torch.full_like(costs.detach(), float('inf'))
        new_costs = torch.where(valid_batch, costs.detach(), inf_costs)
        all_costs = torch.cat([existing_costs_tensor.to(costs.dtype), new_costs])
        if all_costs.numel() == 0:
            return

        k = min(self.K, all_costs.numel())
        top_vals, top_idx = torch.topk(all_costs, k=k, largest=False)
        # Fuse the two device->host transfers (selected indices + their finite mask)
        # into ONE copy, shipped as a single (k, 2) int64 tensor. Identical result.
        sel_finite = torch.stack(
            [top_idx.to(torch.int64), torch.isfinite(top_vals).to(torch.int64)],
            dim=1,
        ).cpu().tolist()  # [[idx, keep],...] — single sync

        self.top_trajectories_actions = [
            candidate_actions[i] for i, keep in sel_finite if keep
        ]
        self.top_trajectories_lengths = [
            candidate_lengths[i] for i, keep in sel_finite if keep
        ]
        self.top_trajectories_costs = [
            top_vals[pos].detach() for pos, (_i, keep) in enumerate(sel_finite) if keep
        ]

    def _replay_trajectories(self, batch_data_list: Optional[List[Dict]], 
                                     max_length: int) -> TrajectoryBatch:
        """Replay trajectories with minimal CPU-GPU transfer."""
        # Use stored top trajectories
        if not hasattr(self, 'top_trajectories_actions') or not self.top_trajectories_actions:
            # No stored trajectories, return empty batch
            return TrajectoryBatch(
                batch_size=0,
                n_measurements=1,
                max_length=max_length,
                n_qubits=self.n_qubits,
                device=self.device
            )
        
        n_batches = len(self.top_trajectories_actions)
        batch_size = n_batches
        
        # Get n_measurements from stored data
        n_measurements = self.top_trajectories_actions[0].shape[0]
        
        # Ensure max_length is sufficient for stored action tensors without a
        # GPU→CPU length reduction. Stored actions are shaped ``(M, L)``.
        actual_max_length = max(
            [max_length] + [actions.shape[1] for actions in self.top_trajectories_actions]
        )
        
        if actual_max_length > max_length:
            if self.debug:
                logging.info(f"Replay: Extending max_length from {max_length} to {actual_max_length} to avoid truncation")
            max_length = actual_max_length
        
        # Create tableau on the same device as the model
        batched_tableau = self._tableau_cls(
            n_qubits=self.n_qubits,
            batch_size=batch_size,
            n_measurements=n_measurements,
            device=str(self.device)
        )
        
        trajectory_batch = TrajectoryBatch(
            batch_size=batch_size,
            n_measurements=n_measurements,
            max_length=max_length,
            n_qubits=self.n_qubits,
            device=self.device
        )
        trajectory_batch.batched_tableau = batched_tableau
        
        # Enable caching for flow computation
        trajectory_batch.enable_caching(
            states_uint8=self._effective_uint8_state_cache()
        )
        
        # Initialize qubit_last_use_step to -1 (no qubit used yet)
        # This is CRITICAL for correct mask computation during replay
        trajectory_batch.qubit_last_use_step.fill_(-1)
        
        # Fill trajectories from stored GPU tensors
        for b_idx in range(batch_size):
            stored_actions = self.top_trajectories_actions[b_idx]
            stored_lengths = self.top_trajectories_lengths[b_idx]
            
            # Copy to trajectory batch
            actual_max_length = min(max_length, stored_actions.shape[1])
            trajectory_batch.actions_time_major[:actual_max_length, b_idx] = (
                stored_actions[:, :actual_max_length].transpose(0, 1)
            )
            # Clamp(max=scalar) is
            # numerically identical to minimum(x, max_length) for int64 and
            # avoids the per-b_idx scalar-tensor alloc + H2D copy in the loop.
            # ``stored_lengths`` is untouched (clamp returns a new tensor), so
            # the ``stored_lengths > 0`` active check below is unaffected.
            trajectory_batch.lengths[b_idx] = stored_lengths.clamp(max=max_length)
            
            # Update active status
            trajectory_batch.active[b_idx] = stored_lengths > 0
        
        # Apply all actions to reconstruct final states and compute depths, with caching
        with torch.no_grad():
            for step in range(max_length):
                # Drive replay activity from stored lengths and let
                # ``to_flat_tensors_active_only`` provide the only empty-set
                # break. This removes the redundant per-step ``.any`` sync.
                step_active = step < trajectory_batch.lengths
                trajectory_batch.active = step_active
                batched_tableau.active = step_active
                
                # Get states for active trajectories
                states_tensor, indices = self._policy_features_active(batched_tableau)
                if states_tensor.shape[0] == 0:
                    break

                if isinstance(indices, torch.Tensor):
                    indices_tensor = indices.to(self.device)
                else:
                    indices_tensor = torch.as_tensor(indices, dtype=torch.long, device=self.device)

                # Compute masks + counts via the fused kernel.
                # Replay always uses ``max_depth=None`` (no depth cap on
                # historical trajectories), so the fused method runs in
                # its simplest configuration here.
                need_counts = step < max_length - 1
                active_masks, _fwd_counts, backward_valid_counts = (
                    self.masking_engine.compute_masks_and_counts_fused(
                        trajectory_batch,
                        indices_tensor,
                        current_step=step + 1,
                        max_depth=None,
                        compute_backward=need_counts,
                        use_fused_kernel=self._effective_fused_mask_counts_kernel(),
                    )
                )
                
                # Cache step data for flow computation
                trajectory_batch.cache_step_data(
                    step, states_tensor, indices_tensor, active_masks, backward_valid_counts
                )
                
                # Get actions for this step
                actions = trajectory_batch.actions_time_major[step]
                
                # Apply actions using the same function with depth tracking
                terminated = self.apply_actions_to_batch(
                    batched_tableau, actions, trajectory_batch, step=step
                )
                
                # Update active status
                trajectory_batch.active &= ~terminated
        
        return trajectory_batch
    
    def save_checkpoint(self, path: str, update: int, metrics: Dict):
        """Save model checkpoint including adaptive tracker state and async evaluation data."""
        # Convert GPU tensors to CPU for saving in a format suitable for async evaluation
        top_trajectories_cpu = []
        if hasattr(self, 'top_trajectories_actions'):
            for i in range(len(self.top_trajectories_actions)):
                cost = self.top_trajectories_costs[i]
                if torch.is_tensor(cost):
                    cost = cost.detach().cpu().item()
                top_trajectories_cpu.append({
                    'actions': self.top_trajectories_actions[i].cpu(),
                    'lengths': self.top_trajectories_lengths[i].cpu(),
                    'cost': cost,
                    'n_measurements': self.top_trajectories_actions[i].shape[0]
                })
        
        # Unwrap ``torch.compile``'d modules before serializing. Compiled
        # modules state-dict with ``_orig_mod.`` prefixed keys, which CPU
        # / uncompiled / async-sampler loaders can't consume cleanly. The
        # load path also strips the prefix defensively in case an older
        # checkpoint was saved without this unwrap.
        pf_inner = getattr(self.pf_model, "_orig_mod", self.pf_model)
        pb_inner = getattr(self.pb_model, "_orig_mod", self.pb_model)
        checkpoint = {
            'pf_model_state_dict': pf_inner.state_dict(),
            'pb_model_state_dict': pb_inner.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer is not None else None,
            'update': update,
            'checkpoint_id': time.time(),  # Add unique checkpoint ID for async coordination
            'top_trajectories': top_trajectories_cpu,
            'metrics': metrics,
            'model_type': self.model_type,
            'n_qubits': self.n_qubits,
            'num_actions': self.num_actions,
            'objective_type': self.objective_type,
            'checkpoint_version': 'gpu_with_depth_async',  # Updated version
            'buffer_strategy': self.buffer_strategy,
            # Add essential data for evaluation without needing full GFN instance
            'action_mapping': self.action_mapping,
            'terminal_index': self.terminal_index,
            'measurement_backend': self.measurement_backend,
        }
        
        # Save adaptive tracker state if using adaptive strategy
        if self.adaptive_tracker is not None:
            checkpoint['adaptive_tracker_state'] = {
                'gates_per_depth': dict(self.adaptive_tracker.gates_per_depth),
                'buffer_utilization': self.adaptive_tracker.buffer_utilization,
                'depth_distribution': dict(self.adaptive_tracker.depth_distribution),
                'max_gates_seen': self.adaptive_tracker.max_gates_seen.cpu().item(),
                'total_trajectories': self.adaptive_tracker.total_trajectories,
                'update_count': self.adaptive_tracker.update_count,
                'gate_counts': self.adaptive_tracker.gate_counts[:1000],  # Save last 1000
            }
        
        # Use atomic write to prevent corruption during concurrent access
        temp_path = path + '.tmp'
        torch.save(checkpoint, temp_path)
        os.rename(temp_path, path)  # Atomic on most filesystems
    
    def load_checkpoint(self, path: str) -> Tuple[int, Dict]:
        """Load model checkpoint and restore adaptive tracker state."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        
        if checkpoint.get('n_qubits') != self.n_qubits:
            logging.info(f"Qubit mismatch: checkpoint has {checkpoint.get('n_qubits')}, "
                  f"model has {self.n_qubits}")
            return 0, {}
        
        try:
            # Defensive ``_orig_mod.`` strip for checkpoints saved before
            # the unwrap-on-save fix, and unwrap our own compiled wrapper
            # so load_state_dict sees the same key namespace either way.
            def _strip_orig_mod(sd):
                if any(k.startswith("_orig_mod.") for k in sd.keys()):
                    return {k.removeprefix("_orig_mod."): v for k, v in sd.items()}
                return sd

            pf_target = getattr(self.pf_model, "_orig_mod", self.pf_model)
            pb_target = getattr(self.pb_model, "_orig_mod", self.pb_model)
            pf_target.load_state_dict(_strip_orig_mod(checkpoint['pf_model_state_dict']))
            pb_target.load_state_dict(_strip_orig_mod(checkpoint['pb_model_state_dict']))
            if self.optimizer is not None and checkpoint.get('optimizer_state_dict') is not None:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # Load top trajectories back to GPU
            self.top_trajectories_actions = []
            self.top_trajectories_lengths = []
            self.top_trajectories_costs = []
            
            if checkpoint.get('checkpoint_version') in ['gpu', 'gpu_with_depth', 'gpu_with_depth_async']:
                for traj_data in checkpoint.get('top_trajectories', []):
                    self.top_trajectories_actions.append(traj_data['actions'].to(self.device))
                    self.top_trajectories_lengths.append(traj_data['lengths'].to(self.device))
                    self.top_trajectories_costs.append(traj_data['cost'])
            
            # Restore adaptive tracker state if present
            if 'adaptive_tracker_state' in checkpoint and self.buffer_strategy == 'adaptive':
                tracker_state = checkpoint['adaptive_tracker_state']
                
                # Initialize tracker if not already done
                if self.adaptive_tracker is None:
                    # Use a dummy max_depth for initialization
                    initial_size = self.calculate_conservative_buffer_size(10)
                    self.adaptive_tracker = AdaptiveBufferTracker(
                        initial_buffer_size=initial_size,
                        device=self.device,
                        warmup_updates=self.adaptive_warmup
                    )
                
                # Restore state
                self.adaptive_tracker.gates_per_depth = defaultdict(list, tracker_state['gates_per_depth'])
                self.adaptive_tracker.buffer_utilization = tracker_state['buffer_utilization']
                self.adaptive_tracker.depth_distribution = defaultdict(int, tracker_state['depth_distribution'])
                self.adaptive_tracker.max_gates_seen = torch.tensor(
                    tracker_state['max_gates_seen'], device=self.device
                )
                self.adaptive_tracker.total_trajectories = tracker_state['total_trajectories']
                self.adaptive_tracker.update_count = tracker_state['update_count']
                self.adaptive_tracker.gate_counts = tracker_state['gate_counts']
                
                logging.info(f"Restored adaptive tracker state: {self.adaptive_tracker.update_count} updates")

            return checkpoint['update'], checkpoint.get('metrics', {})
        except RuntimeError as e:
            logging.info(f"Cannot load checkpoint: {e}")
            return 0, {}


def build_gfn_kwargs(model_config: Dict[str, Any]) -> Dict[str, Any]:
    """Build the shared config-derived ``GFlowNet`` construction kwargs.

    Consumed by BOTH ``EfficientGFNTrainer.__init__`` and ``gfn_async.async_learner``
    so the drift-prone buffer/performance/sampling/backend controls are threaded
    once. The async CPU sampler consumes the subset that affects the batches it
    creates: buffer sizing and cached-state dtype.

    Site-specific kwargs (n_qubits/device/reward_fn/model_type/model_kwargs/
    feature_extractor/packed_w_input/objective/K) stay at their construction sites.
    """
    return dict(
        buffer_strategy=model_config.get("buffer_strategy", "conservative"),
        adaptive_warmup=model_config.get("adaptive_warmup", 100),
        measurement_backend=model_config.get("measurement_backend", None),
        static_shape_sampling=model_config.get("static_shape_sampling", None),
        cuda_graph_sampling=model_config.get("cuda_graph_sampling", None),
        sampling_mode=model_config.get("sampling_mode", None),
        use_cuda_graph_policy=model_config.get("use_cuda_graph_policy", False),
        cuda_graph_policy_max_rows=model_config.get("cuda_graph_policy_max_rows", 2048),
        use_fused_metadata_kernel=model_config.get("use_fused_metadata_kernel", True),
        use_fused_sampling_kernel=model_config.get("use_fused_sampling_kernel", True),
        use_fused_mask_counts_kernel=model_config.get("use_fused_mask_counts_kernel", True),
        use_fused_counter_rng_kernel=model_config.get("use_fused_counter_rng_kernel", True),
        use_fused_partition_update_kernel=model_config.get("use_fused_partition_update_kernel", True),
        use_fused_apply_kernel=model_config.get("use_fused_apply_kernel", True),
        use_bf16_sampling=model_config.get("use_bf16_sampling", True),
        use_bf16_backward=model_config.get("use_bf16_backward", False),
        use_activation_checkpointing=model_config.get("use_activation_checkpointing", True),
        use_uint8_state_cache=model_config.get("use_uint8_state_cache", True),
    )
