# -*- coding: utf-8 -*-
"""``EfficientGFNTrainer``: high-level training loop.

Split out of ``GFNs.py``. Owns the synchronous ``train`` loop, the async
``train_async`` wrapper, cost computation with cached Pauli vectors,
metrics ingestion/trimming.
"""

import json
import logging
import os
import time
import torch
from collections import defaultdict
from typing import List, Dict, Optional, Callable, Any

try:
    from .gfn_runtime import (
        record_function,
        _null_record_function,
        _METRICS_TRIM_SLACK,
        convert_metrics_history_to_cpu,
        get_device,
        FlowMeasTableau,
        SamplingMode,
        CostComputer,
        ThresholdCost,
    )
    from .gfn_core import GFlowNet, build_gfn_kwargs
except ImportError:  # pragma: no cover - direct-execution mode
    from gfn_runtime import (
        record_function,
        _null_record_function,
        _METRICS_TRIM_SLACK,
        convert_metrics_history_to_cpu,
        get_device,
        FlowMeasTableau,
        SamplingMode,
        CostComputer,
        ThresholdCost,
    )
    from gfn_core import GFlowNet, build_gfn_kwargs


class EfficientGFNTrainer:
    """High-level trainer with minimal CPU-GPU transfer and configurable cost functions."""

    # How often to emit the per-update cost statistics. Each emission triggers
    # a host sync to read the (mean, min, max) from GPU; emitting every update
    # forces one sync per training step just for log output. The first / last /
    # debug-mode updates always log regardless. Subclass / instance-override
    # if you want different logging cadence.
    _COST_LOG_EVERY_N = 10

    def __init__(self, config: Dict,
                 reward_fn: Optional[Callable] = None,
                 device: Optional[torch.device] = None,
                 device_preference: Optional[str] = None,
                 metric_store: Optional[Any] = None,
                 metrics_window: int = 512):
        self.config = config
        self.device = device or get_device(device_preference)

        # Persist the resolved device into the config so async mode picks the same
        # GPU. Ambiguous values (``None``, ``"auto"``, plain ``"cuda"``) are
        # overwritten with the resolved ``str(self.device)`` so async always agrees
        # with the parent; explicit indexed values are preserved when they match and
        # raise on mismatch.
        config.setdefault("model", {})
        model_cfg = config["model"]
        existing_pref = model_cfg.get("device_preference")
        resolved_pref = str(self.device)  # e.g., "cuda:0"
        # Set of "looks unspecified" sentinels we always overwrite.
        _AMBIGUOUS = {None, "", "auto", "cuda", "gpu", "default"}
        if existing_pref in _AMBIGUOUS:
            model_cfg["device_preference"] = resolved_pref
        elif existing_pref != resolved_pref:
            # Explicit indexed value disagrees with the resolved device.
            # Fail loud rather than silently picking one or the other —
            # multi-GPU drift is the bug we're trying to prevent.
            raise ValueError(
                f"config['model']['device_preference']={existing_pref!r} "
                f"disagrees with the trainer's resolved device "
                f"{resolved_pref!r}. Pass the matching device_preference "
                f"to EfficientGFNTrainer or omit the config value."
            )

        self.model_config = config["model"]
        self.training_config = config["training"]
        self.quantum_config = config["quantum"]
        
        self.pauli_str_list = self.quantum_config["pauli_str_list"]
        self.w_list = self.quantum_config["w_list"]
        self.n_qubits = len(self.pauli_str_list[0])
        # Pre-compute the hashable cache key fragment once: ``pauli_str_list`` is a
        # list of strings (unhashable), and ``compute_costs_with_probabilities``
        # otherwise rebuilt and re-hashed a fresh tuple on every training step.
        self._pauli_str_list_tuple = tuple(self.pauli_str_list)
        
        self.n_measurements = self.training_config["n_measurements"]
        self.update_freq = self.training_config["update_freq"]
        
        # Initialize cost computer
        cost_config = self.training_config.get("cost", {})
        cost_type = cost_config.get("type", "exponential")
        normalization_type = cost_config.get("normalization_type", "sum")  # 'sum' or 'max'
        
        # Extract cost_kwargs - these are passed to compute_batch_cost
        self.cost_kwargs = {k: v for k, v in cost_config.items() 
                           if k not in ("type", "custom_costs", "normalization_type")}
        
        # Check for legacy epsilon parameter
        if "epsilon" in self.training_config and "epsilon" not in self.cost_kwargs:
            self.cost_kwargs["epsilon"] = self.training_config["epsilon"]
        
        self.cost_computer = CostComputer(
            cost_type=cost_type,
            n_measurements=self.n_measurements,
            device=self.device,
            normalize_weights=True,  # Normalize weights for cost computation (excludes identity)
            normalization_type=normalization_type,  # 'sum' (default) or 'max'
            pauli_strings=self.pauli_str_list,
            n_qubits=self.n_qubits,
            # zero_stabilizer_cost_weights: main.py resolves the flag to a per-term bool
            # mask (cost_computer.detect_stabilizer_terms, fail-fast) and ships it in the
            # quantum config; None when the flag is off (bit-identical legacy weights).
            zero_weight_mask=self.quantum_config.get("stabilizer_zero_mask"),
        )
        
        # Add custom cost functions if specified
        if "custom_costs" in cost_config:
            for name, custom_cost_config in cost_config["custom_costs"].items():
                if name == "threshold":
                    threshold = custom_cost_config.get("threshold", 0.5)
                    self.cost_computer.add_custom_cost_function(
                        name, ThresholdCost(threshold)
                    )
        
        # GIPTE opt-in: when a hit-feature model is requested, build the
        # gauge-invariant packed feature extractor from this run's Hamiltonian
        # Pauli dictionary (+ coefficients) and inject it into the GFlowNet so
        # the sampling/flow/replay paths feed the policy hit features instead of
        # the flattened-W float32 tensor. CUDA-only (the packed kernel is GPU).
        model_type = self.model_config.get("model_type", "clifford_mlp")
        model_kwargs = dict(self.model_config.get("model_kwargs", {}))
        feature_extractor = None
        packed_w_input = False
        if model_type in ("packed_w_rowtoken", "packed_w_split"):
            # Compact, W-based policy that reads the bit-packed W straight from CT
            # (no float (2n)^2 unpack). CUDA-only: the CT packed getter is GPU.
            if self.device.type != "cuda":
                raise ValueError(
                    f"model_type={model_type!r} (packed-W policy) is CUDA-only; "
                    f"got device {self.device}"
                )
            packed_w_input = True
            logging.info(
                f"{model_type} enabled: policy reads CT bit-packed W "
                f"(row_embed_dim={model_kwargs.get('row_embed_dim', 128)}, "
                f"pool={model_kwargs.get('pool', 'mean')})"
            )
        if model_type in ("hit_mlp", "hit_deepsets"):
            if self.device.type != "cuda":
                raise ValueError(
                    f"model_type={model_type!r} (GIPTE encoder) is CUDA-only; "
                    f"got device {self.device}"
                )
            try:
                from .measurement_adapter import TableauFeatureExtractor
            except ImportError:
                from measurement_adapter import TableauFeatureExtractor
            feature_extractor = TableauFeatureExtractor(
                self.pauli_str_list,
                self.n_qubits,
                coeffs=self.w_list,
                device=self.device,
                covariant_shaping=bool(model_kwargs.get("covariant_shaping", False)),
            )
            model_kwargs["feature_dim"] = feature_extractor.feature_dim
            if model_type == "hit_mlp":
                model_kwargs["n_dict"] = feature_extractor.K
            logging.info(
                f"GIPTE encoder enabled: model_type={model_type}, "
                f"K={feature_extractor.K}, feature_dim={feature_extractor.feature_dim}"
            )

        self.gfn = GFlowNet(
            n_qubits=self.n_qubits,
            hidden_dim=self.model_config["hidden_dim"],
            num_hidden_layers=self.model_config["num_hidden_layers"],
            lr=self.model_config["lr"],
            weight_decay=self.model_config["weight_decay"],
            reward_fn=reward_fn,
            device=self.device,
            model_type=model_type,
            model_kwargs=model_kwargs,
            objective_type=self.model_config.get("objective_type", "tb"),
            objective_kwargs=self.model_config.get("objective_kwargs", {}),
            debug=self.model_config.get("debug", False),
            device_preference=device_preference,
            K=self.training_config["K"],
            **build_gfn_kwargs(self.model_config),
            feature_extractor=feature_extractor,
            packed_w_input=packed_w_input,
        )
        
        self.beta = self.training_config["beta"]
        self.max_depth = self.training_config.get("max_depth", self.training_config.get("max_layer", 6))
        self.K = self.training_config["K"]
        
        self.reward_kwargs = self.training_config.get("reward_kwargs", {})
        
        self.metrics_history = defaultdict(list)
        self.timing_history = defaultdict(list)
        self.metric_store = metric_store
        self.metrics_window = max(metrics_window, 0)
        # Explicit current-update tracker for the preempt-checkpoint path.
        # ``metrics_history["loss"]`` is bounded to ``metrics_window``, so deriving
        # the update number from its length capped out and rewrote
        # ``checkpoint_update.pth`` to the wrong offset on long runs.
        # ``current_update`` is the in-progress loop index; ``completed_updates`` is
        # bumped only AFTER an iteration finishes. The signal handler uses the latter
        # so a mid-flight emergency save doesn't tag advanced weights with the
        # previous update number.
        self.current_update = 0
        self.completed_updates = 0
        self.profiler = None
        self._pauli_vecs_cache: Optional[torch.Tensor] = None
        self._pauli_vecs_cache_device: Optional[torch.device] = None

    def attach_profiler(self, profiler: Any) -> None:
        """Register an optional torch.profiler instance."""
        self.profiler = profiler

    def _get_cached_pauli_vecs(self, batched_tableau: FlowMeasTableau) -> torch.Tensor:
        """Return static Hamiltonian Pauli vectors on the tableau device."""
        tableau_device = torch.device(getattr(batched_tableau, 'device', self.device))
        if (
            self._pauli_vecs_cache is None
            or self._pauli_vecs_cache_device != tableau_device
        ):
            pauli_vecs, _ = batched_tableau._pauli_string_to_symplectic(self.pauli_str_list)
            self._pauli_vecs_cache = pauli_vecs
            self._pauli_vecs_cache_device = tableau_device
        return self._pauli_vecs_cache

    def compute_costs_with_probabilities(self, batched_tableau: FlowMeasTableau,
                                       silence: bool = True, **override_kwargs) -> torch.Tensor:
        """Compute costs using the CostComputer with probabilities from the tableau.

        Args:
            batched_tableau: The batched Clifford tableau
            silence: Whether to suppress debug output
            **override_kwargs: Additional kwargs to override self.cost_kwargs
        """
        # Get probabilities for all Pauli strings. The Hamiltonian is static across
        # updates, so avoid reparsing Pauli strings every cost call. Prefer the packed
        # conjugation kernel: ``prob_P_multi`` issues one O(B*K) GF(2) XOR/popcount
        # launch plus a uint8 hit tensor when the CT backend exposes
        # ``conjugate_dictionary_packed``, falling back to the float
        # ``transform_paulis`` path (which materializes a large (B,M,K,2n) bool
        # intermediate) only for non-CT backends or K==0. The packed result is
        # bit-identical to the float path.
        cache = getattr(batched_tableau, '_prob_cache', None)
        use_prob_cache = bool(getattr(batched_tableau, 'enable_prob_cache', False)) and cache is not None
        cache_key = (getattr(batched_tableau, 'version', 0), self._pauli_str_list_tuple)
        probs = cache.get(cache_key) if use_prob_cache else None
        if probs is None:
            if hasattr(batched_tableau, 'prob_P_multi'):
                probs = batched_tableau.prob_P_multi(self.pauli_str_list)
            elif (
                hasattr(batched_tableau, 'transform_paulis')
                and hasattr(batched_tableau, '_pauli_string_to_symplectic')
            ):
                # Very-legacy backends that predate prob_P_multi: float path.
                pauli_vecs = self._get_cached_pauli_vecs(batched_tableau)
                p_out = batched_tableau.transform_paulis(pauli_vecs)
                has_x = p_out[..., :self.n_qubits].any(dim=3)
                probs = (~has_x).float()
            else:
                raise AttributeError(
                    "batched_tableau exposes neither prob_P_multi nor transform_paulis"
                )
            if use_prob_cache:
                cache.put(cache_key, probs.clone())
        
        if self.gfn.debug and not silence:
            logging.debug("\nDEBUG compute_costs_with_probabilities:")
            logging.debug(f"  Probabilities shape: {probs.shape}, dtype: {probs.dtype}")
            logging.debug(f"  Probabilities sum (first 3): {probs.sum(dim=1)[:3].cpu().numpy()}")
        
        # Merge kwargs with overrides
        kwargs = {**self.cost_kwargs, **override_kwargs}
        
        # Use CostComputer to compute batch costs
        costs = self.cost_computer.compute_batch_cost(probs, self.w_list, **kwargs)
        
        if not silence:
            logging.info(f"Probabilities shape: {probs.shape}, dtype: {probs.dtype}")
            logging.info(probs.sum(dim=1))
            logging.info(f"Computed costs shape: {costs.shape}, dtype: {costs.dtype}")
            logging.info(f"Costs (first 4): {costs.flatten()[:4].detach().cpu().numpy()}")
        
        if self.gfn.debug and not silence:
            logging.debug(f"  Costs shape: {costs.shape}, device: {costs.device}")
            logging.debug(f"  Costs (all): {costs.cpu().numpy()}")

        return costs

    def ingest_metrics(self, metrics: Optional[Dict[str, List]], timing: Optional[Dict[str, List]] = None) -> None:
        """Load existing metrics into bounded in-memory history and persist to disk."""
        if not metrics:
            return

        if self.metric_store is not None:
            self.metric_store.replace(metrics, timing)

        window = self.metrics_window

        trimmed_metrics = defaultdict(list)
        for key, values in metrics.items():
            if window and len(values) > window:
                trimmed_metrics[key].extend(values[-window:])
            else:
                trimmed_metrics[key].extend(values)
        self.metrics_history = trimmed_metrics

        if timing:
            trimmed_timing = defaultdict(list)
            for key, values in timing.items():
                if window and len(values) > window:
                    trimmed_timing[key].extend(values[-window:])
                else:
                    trimmed_timing[key].extend(values)
            self.timing_history = trimmed_timing

    def train(self, num_updates: int, 
              replay_every: Optional[int] = None,
              offpolicy_every: Optional[int] = None,
              checkpoint_every: int = 100,
              profile: bool = True,
              cost_schedule: Optional[Dict[int, str]] = None):
        """Main training loop with minimal CPU-GPU transfer and configurable cost functions.

        Args:
            num_updates: Number of training updates
            replay_every: Frequency of replay training
            offpolicy_every: Frequency of off-policy training
            checkpoint_every: Frequency of checkpointing
            profile: Whether to profile timing
            cost_schedule: Optional dictionary mapping update numbers to cost types
                          e.g., {0: "exponential", 1000: "logarithmic", 2000: "l1"}
        """
        start_update = 0
        
        checkpoint_path = os.path.join(self.model_config["model_dir"], "checkpoint_latest.pth")
        if os.path.exists(checkpoint_path):
            try:
                start_update, metrics = self.gfn.load_checkpoint(checkpoint_path)
                if start_update > 0:
                    # Ensure loaded metrics are scalars, not tensors
                    cleaned_metrics = {}
                    for k, v_list in metrics.items():
                        cleaned_list = []
                        for v in v_list:
                            if torch.is_tensor(v):
                                v = v.item() if v.numel() == 1 else v.cpu().numpy()
                            cleaned_list.append(v)
                        cleaned_metrics[k] = cleaned_list
                    self.ingest_metrics(cleaned_metrics)
                    logging.info(f"Resumed from update {start_update}")
            except Exception as e:
                logging.info(f"Could not load checkpoint: {e}")
                start_update = 0
        
        # Pre-allocate timing tensors on GPU to avoid repeated transfers
        if self.device.type != 'cpu':
            torch.cuda.synchronize() if self.device.type == 'cuda' else None

        # Seed ``completed_updates`` from the resume offset so a preempt
        # before the first new iteration doesn't tag the emergency save
        # with update=0 (which downstream resume treats as "no progress")
        # — a window-of-overwriting-a-valid-resume window that
        # MED 11 flagged. Mirrors what we do in ``main.run_experiment``.
        self.completed_updates = max(int(start_update or 0), int(self.completed_updates or 0))

        if self.gfn.debug:
            logging.debug("\nStarting training with debug mode enabled")
            logging.debug(f"Config: {json.dumps(self.config, indent=2)}")

        # Only pay for ``record_function`` range push/pops when the caller
        # actually wants timing/annotations. Outside a profiler they are pure
        # host overhead between the kernel-issuing phases of an issue-bound step.
        rf = record_function if profile else _null_record_function

        for update in range(start_update, num_updates):
            # Mirror the loop counter into ``self.current_update`` so the
            # preemption handler in run_config.py can read the true
            # in-progress update number instead of falling back to the
            # ``metrics_window``-bounded ``len(metrics_history['loss'])``.
            self.current_update = update

            # Check if we need to change cost function
            if cost_schedule and update in cost_schedule:
                new_cost_type = cost_schedule[update]
                logging.info(f"\nSwitching to {new_cost_type} cost function at update {update}")
                self.cost_computer.set_cost_type(new_cost_type, self.n_measurements)

            update_start = time.time() if profile else None

            if self.gfn.debug:
                logging.debug(f"\n=== DEBUG Update {update+1}/{num_updates} ===")

            logging.info(f"\n=== Update {update+1}/{num_updates} ===")
            logging.info(f"Cost function: {self.cost_computer.cost_type}")
            if self.cost_kwargs:
                logging.info(f"Cost kwargs: {self.cost_kwargs}")

            sample_start = time.time() if profile else None

            # Sample on-policy trajectories with depth limit and caching
            with rf("train.sample_trajectories"):
                trajectory_batch = self.gfn.sample_trajectories(
                    batch_size=self.update_freq,
                    n_measurements=self.n_measurements,
                    max_depth=self.max_depth,
                    mode=SamplingMode.ON_POLICY,
                    cache_for_flows=True  # Enable caching
                )
            
            # Compute costs using CostComputer with kwargs
            batched_tableau = trajectory_batch.batched_tableau
            with rf("train.compute_costs"):
                costs = self.compute_costs_with_probabilities(batched_tableau)
            
            # Device-aware logging. We rate-limit cost-stat emission to every
            # ``_COST_LOG_EVERY_N`` updates (plus the first one and the final
            # one). The previous unconditional 3x.item per update forced
            # one host sync per training step just for log lines.
            logger = logging.getLogger()
            debug_active = bool(getattr(self.gfn, 'debug', False))
            is_last_update = (update == num_updates - 1)
            cost_log_step = (
                debug_active
                or update == start_update
                or is_last_update
                or ((update + 1) % self._COST_LOG_EVERY_N == 0)
            )
            if self.device.type in ['cuda', 'mps']:
                info_on = logger.isEnabledFor(logging.INFO)
                debug_on = logger.isEnabledFor(logging.DEBUG)
                if (info_on and cost_log_step) or debug_on:
                    # Single (3,) tensor: one sync covers all three stats.
                    stats = torch.stack([costs.mean(), costs.min(), costs.max()]).cpu().tolist()
                    mean_cost, min_cost, max_cost = stats
                    if debug_on:
                        costs_cpu = costs.cpu().numpy()
                        logging.debug(f"  Batch costs (one per batch element): {costs_cpu}")
                    if info_on and cost_log_step:
                        logging.info(f"  Mean batch cost: {mean_cost:.4f}, "
                              f"Min: {min_cost:.4f}, Max: {max_cost:.4f}")
            elif cost_log_step:
                costs_cpu = costs.cpu().numpy()
                logging.info(f"  Batch costs (one per batch element): {costs_cpu}")
                logging.info(f"  Mean batch cost: {costs_cpu.mean():.4f}, "
                      f"Min: {costs_cpu.min():.4f}, Max: {costs_cpu.max():.4f}")
            
            # Compute loss using the standard method
            with rf("train.compute_loss"):
                loss, metrics = self.gfn.compute_loss(
                    trajectory_batch,
                    costs,
                    self.beta,
                    max_depth=self.max_depth,
                    metrics_to_cpu=False,
                    **self.reward_kwargs
                )

            sample_time = (time.time() - sample_start) if profile else 0.0

            # Update model using the standard method
            with rf("train.optimizer_step"):
                loss_value = self.gfn.update_step(loss, return_tensor=True)

            # Update top trajectories for replay
            with rf("train.update_top_trajectories"):
                self.gfn._update_top_trajectories(trajectory_batch, costs)
            metrics['loss'] = loss_value
            metrics['cost_type'] = self.cost_computer.cost_type

            # update time after sampling and loss computation
            update_time = (time.time() - update_start) if profile else 0.0
            
            # Replay training
            if replay_every and (update + 1) % replay_every == 0 and self.gfn.top_trajectories_actions:
                self.gfn.optimizer.zero_grad()
                
                # Sample replay trajectories WITH caching (needed for flow computation)
                with rf("train.replay.sample"):
                    replay_batch = self.gfn.sample_trajectories(
                        batch_size=min(len(self.gfn.top_trajectories_actions), self.update_freq),
                        n_measurements=self.n_measurements,
                        max_depth=self.max_depth,
                        mode=SamplingMode.REPLAY,
                        batch_data_list=None,
                        cache_for_flows=True  # CRITICAL: Need caching for flow computation
                    )
                
                # Compute costs using CostComputer
                replay_tableau = replay_batch.batched_tableau
                logging.info(f" At Step {update + 1}, Replay batch size: {replay_batch.batch_size}, with probabilities:")
                with rf("train.replay.compute_costs"):
                    replay_costs = self.compute_costs_with_probabilities(replay_tableau, silence=False)
                
                self.gfn._debug_replay = True
                
                # Compute loss using the exact same method as on-policy
                with rf("train.replay.compute_loss"):
                    replay_loss, replay_metrics = self.gfn.compute_loss(
                        replay_batch,
                        replay_costs,
                        self.beta,
                        max_depth=self.max_depth,
                        metrics_to_cpu=False,
                        **self.reward_kwargs
                    )
                
                self.gfn._debug_replay = False
                
                # Update step using the same method
                with rf("train.replay.optimizer_step"):
                    replay_loss_value = self.gfn.update_step(replay_loss, return_tensor=True)
                replay_metrics['loss'] = replay_loss_value
                
                # Store metrics with replay prefix
                for k, v in replay_metrics.items():
                    metrics[f'replay_{k}'] = v
                
                replay_log_vals = torch.stack([
                    replay_metrics['loss'].detach().float(),
                    replay_metrics['reward'].detach().float(),
                ]).cpu().tolist()
                logging.info(f"  Replay on {len(self.gfn.top_trajectories_actions)} batches: "
                      f"loss={replay_log_vals[0]:.6f}, batch_reward={replay_log_vals[1]:.4f}")
            
            # Off-policy training
            if offpolicy_every and (update + 1) % offpolicy_every == 0:
                self.gfn.optimizer.zero_grad()
                
                # Sample off-policy trajectories with caching
                with rf("train.offpolicy.sample"):
                    offpolicy_batch = self.gfn.sample_trajectories(
                        batch_size=self.update_freq,
                        n_measurements=self.n_measurements,
                        max_depth=self.max_depth,
                        mode=SamplingMode.OFF_POLICY,
                        cache_for_flows=True  # Enable caching
                    )
                
                # Compute costs using CostComputer
                offpolicy_tableau = offpolicy_batch.batched_tableau
                with rf("train.offpolicy.compute_costs"):
                    offpolicy_costs = self.compute_costs_with_probabilities(offpolicy_tableau)
                
                # Compute loss using the exact same method as on-policy
                with rf("train.offpolicy.compute_loss"):
                    offpolicy_loss, offpolicy_metrics = self.gfn.compute_loss(
                        offpolicy_batch,
                        offpolicy_costs,
                        self.beta,
                        max_depth=self.max_depth,
                        metrics_to_cpu=False,
                        **self.reward_kwargs
                    )
                
                # Update step using the same method
                with rf("train.offpolicy.optimizer_step"):
                    offpolicy_loss_value = self.gfn.update_step(offpolicy_loss, return_tensor=True)
                offpolicy_metrics['loss'] = offpolicy_loss_value
                
                # Store metrics with offpolicy prefix
                for k, v in offpolicy_metrics.items():
                    metrics[f'offpolicy_{k}'] = v
                
                offpolicy_log_vals = torch.stack([
                    offpolicy_metrics['loss'].detach().float(),
                    offpolicy_metrics['reward'].detach().float(),
                ]).cpu().tolist()
                logging.info(f"  Off-policy: loss={offpolicy_log_vals[0]:.6f}, "
                      f"batch_reward={offpolicy_log_vals[1]:.4f}")
            
            # Update metrics with bounded history
            # Keep GPU tensors detached in history, convert only for logging/serialization
            for k, v in metrics.items():
                if torch.is_tensor(v):
                    v_detached = v.detach()
                    metrics[k] = v_detached
                    series = self.metrics_history[k]
                    series.append(v_detached)
                else:
                    series = self.metrics_history[k]
                    series.append(v)

                # Amortize the O(window) list compaction. Trimming every
                # update slices the whole list each step; only compact once the
                # overflow exceeds a slack margin (still always retaining at
                # least ``metrics_window`` most-recent entries).
                if self.metrics_window and len(series) > self.metrics_window + _METRICS_TRIM_SLACK:
                    del series[:-self.metrics_window]

            total_trajectories = self.update_freq * self.n_measurements

            timing_payload = None
            if profile:
                safe_update_time = update_time if update_time > 0 else 1e-9
                throughput = total_trajectories / safe_update_time
                timing_payload = {
                    'total': update_time,
                    'sample': sample_time,
                    'throughput': throughput,
                }

                for key, value in timing_payload.items():
                    series = self.timing_history[key]
                    series.append(value)
                    if self.metrics_window and len(series) > self.metrics_window + _METRICS_TRIM_SLACK:
                        del series[:-self.metrics_window]

            # Convert only metrics needed for logging (single batched GPU sync)
            # Batch extract the 4-5 scalars needed for logging in one transfer
            log_keys = ['loss', 'reward', 'cost', 'logZ', 'avg_trajectories_per_batch']
            log_tensors = [metrics[k] for k in log_keys if torch.is_tensor(metrics[k])]

            if log_tensors:
                # Single GPU→CPU transfer for all logging metrics
                # Ensure dtype compatibility for torch.stack (convert to float32 if needed)
                if len(log_tensors) > 1 and not all(t.dtype == log_tensors[0].dtype for t in log_tensors):
                    log_tensors = [t.float() for t in log_tensors]
                log_tensors_stacked = torch.stack(log_tensors).cpu()
                idx = 0
                log_vals = {}
                for k in log_keys:
                    if torch.is_tensor(metrics[k]):
                        log_vals[k] = log_tensors_stacked[idx].item()
                        idx += 1
                    else:
                        log_vals[k] = metrics[k]
            else:
                log_vals = {k: metrics[k] for k in log_keys}

            avg_trajs = log_vals.get('avg_trajectories_per_batch', 0.0)

            # Only convert all metrics if metric_store needs them
            if self.metric_store is not None:
                metrics_cpu = {}
                for k, v in metrics.items():
                    if torch.is_tensor(v):
                        metrics_cpu[k] = v.item() if v.numel() == 1 else v.cpu().numpy()
                    else:
                        metrics_cpu[k] = v
                self.metric_store.append(update + 1, metrics_cpu, timing_payload)

            logging.info(f"\nSummary - Loss: {log_vals['loss']:.6f}, "
                  f"Batch Reward: {log_vals['reward']:.4f}, "
                  f"Batch Cost: {log_vals['cost']:.4f}, "
                  f"LogZ: {log_vals['logZ']:.3f}, "
                  f"Avg Trajs/Batch: {avg_trajs:.1f}")
            
            if profile:
                logging.info(f"Time: {update_time:.2f}s, "
                      f"Throughput: {total_trajectories/update_time:.1f} traj/s "
                      f"({total_trajectories} total trajectories)")
            
            if (update + 1) % checkpoint_every == 0:
                os.makedirs(self.model_config["model_dir"], exist_ok=True)
                # Convert GPU tensors in metrics history to CPU for checkpoint
                metrics_history_cpu = convert_metrics_history_to_cpu(self.metrics_history)
                self.gfn.save_checkpoint(checkpoint_path, update + 1, metrics_history_cpu)
                logging.info(f"Checkpoint saved at update {update + 1}")

            if self.profiler is not None:
                self.profiler.step()

            # Bump the *completed* count only after the entire iteration
            # (including any periodic checkpoint save above) has finished
            # successfully. The signal handler uses this for the emergency
            # save so an interrupted mid-iteration doesn't tag advanced
            # weights with the previous update number.
            self.completed_updates = update + 1

    def train_async(self, num_updates: int, **kwargs):
        """
        Asynchronous training with separate sampler and learner processes.

        Args:
            num_updates: Number of training updates
            **kwargs: Ignored (for compatibility with train method)
        """
        # GIPTE / packed-W are sync-only: gfn_async's sampler/learner processes
        # construct their OWN GFlowNet without the trainer-built feature extractor
        # / packed_w_input flag (and the CT packed getter is CUDA-only, while CPU
        # sampler workers cannot run it). Fail with an explicit message instead of
        # the confusing downstream shape error. (Async-sibling rule.)
        if getattr(self.gfn, "feature_extractor", None) is not None:
            raise NotImplementedError(
                "GIPTE hit-feature models (model_type in {'hit_mlp','hit_deepsets'}) "
                "are not supported in async training: the async sampler/learner "
                "processes build their own GFlowNet without the trainer-built "
                "feature extractor. Use synchronous train."
            )
        if getattr(self.gfn, "packed_w_input", False):
            raise NotImplementedError(
                "packed_w_rowtoken / packed_w_split (packed_w_input=True) are not supported in async "
                "training: the async sampler/learner processes build their own "
                "GFlowNet without the packed-W policy-input flag. Use synchronous "
                "train."
            )

        logging.info("\n" + "="*60)
        logging.info("Starting ASYNCHRONOUS training mode")
        logging.info(f"Samplers: {self.config['training'].get('num_samplers', 2)}")
        logging.info(f"Pipeline depth: {self.config['training'].get('pipeline_depth', 4)}")
        logging.info(f"Broadcast every: {self.config['training'].get('broadcast_every', 10)} updates")
        logging.info("="*60 + "\n")
        
        # Import here to avoid issues when not using async mode, using the
        # module-vs-package try/except idiom: ``gfn_async.py`` itself uses relative
        # imports, so the plain form only works when this module is loaded as a
        # top-level script.
        try:
            from .gfn_async import async_learner
        except ImportError:
            from gfn_async import async_learner

        # Snapshot the caller's resumed/warm-started state and pass it plus the
        # configured reward callable to ``async_learner``. Without ``initial_state``,
        # async resume builds a fresh ``GFlowNet`` and discards loaded weights,
        # optimizer state and top-K replay; without ``reward_fn`` it silently falls
        # back to ``default_reward_fn``.
        inner = getattr(self.gfn.pf_model, "_orig_mod", self.gfn.pf_model)
        initial_state: Dict[str, Any] = {
            "pf_model": {k: v.detach().cpu().clone() for k, v in inner.state_dict().items()},
        }
        try:
            initial_state["optimizer"] = self.gfn.optimizer.state_dict()
        except Exception:
            pass
        if getattr(self.gfn, "top_trajectories_actions", None):
            initial_state["top_trajectories"] = {
                "actions": [a.detach().cpu().clone() for a in self.gfn.top_trajectories_actions],
                "lengths": [l.detach().cpu().clone() for l in self.gfn.top_trajectories_lengths],
                "costs": [
                    (c.detach().cpu().item() if torch.is_tensor(c) else c)
                    for c in self.gfn.top_trajectories_costs
                ],
            }

        # Run async training. ``async_learner`` returns the relative update count from
        # this call, translated into an absolute count below using
        # ``self.completed_updates`` as the resume baseline. Pass REMAINING updates
        # rather than the absolute target so a resumed run doesn't over-advance:
        # resuming from N and asking for T must stop at absolute T, not N + T.
        resume_offset = int(getattr(self, "completed_updates", 0) or 0)
        remaining_updates = max(int(num_updates) - resume_offset, 0)
        if remaining_updates == 0:
            logging.info(
                f"train_async: resume_offset={resume_offset} already at/past "
                f"num_updates={num_updates}; nothing to do"
            )
            return
        if resume_offset > 0:
            logging.info(
                f"train_async: resuming from update {resume_offset}; "
                f"running {remaining_updates} fresh updates to reach "
                f"absolute target {num_updates}"
            )
        (metrics_history, timing_history,
         trained_state, completed_updates_rel, setup_error) = async_learner(
            self.config, remaining_updates,
            initial_state=initial_state,
            reward_fn=getattr(self.gfn, "reward_fn", None),
        )
        # Translate to absolute progress for checkpoint naming + payload.
        completed_updates = resume_offset + completed_updates_rel

        # Reconcile the learner state back into the caller's trainer so
        # downstream code (checkpoint save, plotting, resume) sees the
        # trained model. The async loop ran on its own ``GFlowNet`` (CPU
        # samplers + a separate GPU learner); copy weights / optimizer +
        # top-K replay back into ``self.gfn`` here.
        if trained_state is not None:
            inner = getattr(self.gfn.pf_model, "_orig_mod", self.gfn.pf_model)
            inner.load_state_dict(trained_state["pf_model"])
            inner.to(self.device)
            opt_state = trained_state.get("optimizer")
            if opt_state is not None:
                try:
                    self.gfn.optimizer.load_state_dict(opt_state)
                except Exception as e:
                    logging.warning(f"Could not restore async optimizer state: {e}")
            # Top trajectories were updated on the learner-local gfn; without
            # this restore, the saved checkpoint would always carry the
            # caller's empty top-K lists rather than the trained ones.
            top = trained_state.get("top_trajectories")
            if top is not None:
                self.gfn.top_trajectories_actions = [
                    a.to(self.device) for a in top["actions"]
                ]
                self.gfn.top_trajectories_lengths = [
                    l.to(self.device) for l in top["lengths"]
                ]
                self.gfn.top_trajectories_costs = list(top["costs"])
        else:
            logging.warning(
                "Async learner returned no trained state — self.gfn remains at "
                "initial weights. Checkpoint will not reflect async training."
            )

        # Update trainer's history, ensuring metrics are scalars
        cleaned_metrics = {}
        for k, v_list in metrics_history.items():
            cleaned_list = []
            for v in v_list:
                if torch.is_tensor(v):
                    v = v.item() if v.numel() == 1 else v.cpu().numpy()
                cleaned_list.append(v)
            cleaned_metrics[k] = cleaned_list

        cleaned_timing = {}
        for k, v_list in timing_history.items():
            cleaned_list = []
            for v in v_list:
                if torch.is_tensor(v):
                    v = v.item() if v.numel() == 1 else v.cpu().numpy()
                cleaned_list.append(v)
            cleaned_timing[k] = cleaned_list

        self.ingest_metrics(cleaned_metrics, cleaned_timing)

        # Persist a checkpoint of the reconciled trainer state so async runs are
        # restartable. Always use the canonical ``checkpoint_update_<N>.pth`` name so
        # main.py's glob discovers partial async runs — ``checkpoint_async_partial.pth``
        # was off-glob and silently bypassed resume. The update number in the filename
        # and payload is the real ``completed_updates``, so a crashed-at-0 run writes
        # ``checkpoint_update_0.pth`` rather than overwriting a valid later one.
        if trained_state is not None:
            try:
                os.makedirs(self.model_config["model_dir"], exist_ok=True)
                metrics_history_cpu = convert_metrics_history_to_cpu(self.metrics_history)
                # Canonical name: ``checkpoint_update.pth`` (no suffix) is the
                # latest-clean-completion alias the rest of the stack writes, mirrored
                # exactly on full completion so async is a drop-in. Partial saves use
                # the numbered form so they don't shadow an older clean file. "Full
                # completion" means the relative progress hit ``remaining_updates``;
                # the absolute count is used only for filename/payload tagging.
                if completed_updates_rel >= remaining_updates:
                    checkpoint_path = os.path.join(
                        self.model_config["model_dir"], "checkpoint_update.pth"
                    )
                    self.gfn.save_checkpoint(checkpoint_path, completed_updates, metrics_history_cpu)
                    logging.info(
                        f"Final async checkpoint saved at {checkpoint_path} "
                        f"(absolute update={completed_updates}, "
                        f"resumed from {resume_offset})"
                    )
                else:
                    checkpoint_path = os.path.join(
                        self.model_config["model_dir"],
                        f"checkpoint_update_{completed_updates}.pth",
                    )
                    self.gfn.save_checkpoint(checkpoint_path, completed_updates, metrics_history_cpu)
                    logging.warning(
                        f"Partial async checkpoint saved at {checkpoint_path} "
                        f"(absolute completed {completed_updates}, "
                        f"+{completed_updates_rel}/{remaining_updates} fresh, "
                        f"resumed from {resume_offset}, target {num_updates}) — "
                        f"discoverable by main.py's checkpoint_update*.pth glob"
                    )
            except Exception as e:
                logging.warning(f"Async checkpoint save failed: {e}")

        # Surface mid-run failures so orchestration sees the run as failed.
        # The partial checkpoint above is already on disk before this raise.
        if setup_error is not None:
            raise setup_error

