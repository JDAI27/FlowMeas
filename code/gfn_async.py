# -*- coding: utf-8 -*-
# gfn_async.py

import torch
import torch.multiprocessing as mp
from multiprocessing.synchronize import Event as MPEvent
from queue import Empty, Full
import logging
import time
import traceback
from typing import Dict, Tuple, Optional, Any
from collections import defaultdict

from .GFNs import GFlowNet, TrajectoryBatch, SamplingMode, get_device
from .gfn_core import build_gfn_kwargs
from .cost_computer import CostComputer, ThresholdCost
from .measurement_adapter import AUTO_BACKEND, TABLEAU_BATCH_BACKEND


def _cpu_sampler_measurement_backend(model_config: Dict[str, Any]) -> Optional[str]:
    """Return the backend policy for CPU-only async sampler workers.

    Async samplers intentionally build CPU ``GFlowNet`` instances. The
    CT-backed backend is CUDA-only, so an explicit learner-side CT request
    must not be forwarded directly to these sampler workers. Passing
    ``auto`` keeps the decision routed through the shared measurement adapter
    resolver while selecting the CPU-safe legacy backend.
    """
    measurement_backend = model_config.get("measurement_backend", None)
    if measurement_backend == TABLEAU_BATCH_BACKEND:
        return AUTO_BACKEND
    return measurement_backend


def _broadcast_state_dict(pf_model) -> Dict[str, torch.Tensor]:
    """Return a CPU state_dict that the eager sampler model can load.

    ``torch.compile`` wraps the module in ``OptimizedModule`` whose
    ``state_dict()`` keys carry an ``_orig_mod.`` prefix; the CPU sampler
    is constructed as an uncompiled ``GFlowNet`` and calls
    ``load_state_dict(..., strict=True)``, so the prefixed keys would
    fail before the first batch in any run where compile succeeds.
    Pull the inner module explicitly and detach/clone its tensors so we
    never mutate the learner's parameter device.
    """
    inner = getattr(pf_model, "_orig_mod", pf_model)
    return {k: v.detach().cpu().clone() for k, v in inner.state_dict().items()}


def _load_sampler_state_dict(model, state_dict: Dict[str, torch.Tensor]) -> None:
    """Defensive load: strip an ``_orig_mod.`` prefix and unwrap the target.

    Belt-and-suspenders against the ``torch.compile`` prefix mismatch in
    *both* directions:
      * ``state_dict`` may carry ``_orig_mod.`` keys if the sender hasn't
        unwrapped (older checkpoints, hand-built dicts). Strip on receive.
      * ``model`` may itself be an ``OptimizedModule`` if the receiver is
        compiled (the CUDA learner path), in which case ``load_state_dict``
        would expect ``_orig_mod.*`` keys — the opposite mismatch. Always
        load into the inner module.
    Without the target unwrap, async resume seeding silently fails on a
    compiled learner: ``load_state_dict`` with unprefixed keys against an
    ``OptimizedModule`` either errors or no-ops in strict-False paths.
    """
    if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
        state_dict = {k.removeprefix("_orig_mod."): v for k, v in state_dict.items()}
    target = getattr(model, "_orig_mod", model)
    target.load_state_dict(state_dict)


def _build_cost_computer(training_config: Dict, quantum_config: Dict,
                         device: torch.device) -> Tuple[CostComputer, Dict[str, Any]]:
    """Mirror ``EfficientGFNTrainer``'s cost setup so async/sync agree.

    The previous async build dropped ``normalization_type``, ``pauli_strings``,
    ``n_qubits``, custom-cost wiring, and read ``training_config['epsilon']``
    directly — which is missing in current configs (epsilon lives under
    ``training.cost.epsilon`` / ``cost_kwargs``). Result: async could
    ``KeyError`` on startup or, even when it ran, train on a different cost
    surface than sync mode. Returns the configured computer plus
    ``cost_kwargs`` (epsilon + anything else the cost-fn expects) so the
    sampler can call ``compute_batch_cost`` with the same kwargs the sync
    trainer uses.
    """
    cost_config = training_config.get("cost", {})
    cost_type = cost_config.get("type", "exponential")
    normalization_type = cost_config.get("normalization_type", "sum")
    cost_kwargs = {k: v for k, v in cost_config.items()
                   if k not in ("type", "custom_costs", "normalization_type")}
    # Legacy compatibility — older configs put epsilon at the top level.
    if "epsilon" in training_config and "epsilon" not in cost_kwargs:
        cost_kwargs["epsilon"] = training_config["epsilon"]

    pauli_strings = quantum_config["pauli_str_list"]
    n_qubits = len(pauli_strings[0]) if pauli_strings else 0

    cost_computer = CostComputer(
        cost_type=cost_type,
        n_measurements=training_config["n_measurements"],
        device=device,
        normalize_weights=True,
        normalization_type=normalization_type,
        pauli_strings=pauli_strings,
        n_qubits=n_qubits,
        # zero_stabilizer_cost_weights: same mask the sync trainer consumes (resolved
        # once in main.py, shipped in the pickled quantum config), so async sampler
        # workers optimize the SAME cost surface. The async LEARNER never constructs a
        # CostComputer (costs arrive in sampler batches), so this is the only async site.
        zero_weight_mask=quantum_config.get("stabilizer_zero_mask"),
    )
    if "custom_costs" in cost_config:
        for name, ccfg in cost_config["custom_costs"].items():
            if name == "threshold":
                cost_computer.add_custom_cost_function(
                    name, ThresholdCost(ccfg.get("threshold", 0.5))
                )
    return cost_computer, cost_kwargs


def move_batch_to_device(batch: TrajectoryBatch, device: torch.device) -> TrajectoryBatch:
    """Move *all* TrajectoryBatch tensors and cached flow data to ``device``.

    Async samplers produce batches with ``cache_for_flows=True``; the learner
    immediately takes ``compute_flows_cached`` which indexes GPU lengths with
    CPU cached indices and feeds CPU states/masks into the CUDA policy. Any
    field still on the source device causes a silent device-mismatch crash
    or, worse, computes flows on the wrong device.

    Covers, beyond the original four:
    - ``actions_time_major`` (source of truth for the time-major view) +
      ``actions`` rebound as a permuted view of the new tensor.
    - The per-trajectory metadata: ``circuit_depths``, ``current_layer_qubits``,
      ``qubit_last_layer``, ``qubit_last_use_step``, ``action_qubits``.
    - The cached-flow lists: ``cached_states`` (each entry is a
      ``(states_tensor, indices_tensor)`` tuple), ``cached_masks``,
      ``cached_backward_valid_counts``. Each list entry may be ``None`` for
      steps that weren't reached, hence the per-entry None check.
    - ``batch.device`` so any consumer that reads it sees the new target.
    """
    # Source-of-truth time-major tensor (preserves view aliasing).
    if hasattr(batch, "actions_time_major"):
        batch.actions_time_major = batch.actions_time_major.to(device)
        batch.actions = batch.actions_time_major.permute(1, 2, 0)
    else:
        batch.actions = batch.actions.to(device)

    # Core per-trajectory tensors.
    for attr in (
        "lengths", "active", "masks",
        "last_single_qubit_gates",
        "circuit_depths", "current_layer_qubits",
        "qubit_last_layer", "qubit_last_use_step", "action_qubits",
    ):
        t = getattr(batch, attr, None)
        if isinstance(t, torch.Tensor):
            setattr(batch, attr, t.to(device))

    # Cached flow data: parallel lists with possible None placeholders.
    for list_name in ("cached_states", "cached_masks", "cached_backward_valid_counts"):
        cached = getattr(batch, list_name, None)
        if not cached:
            continue
        new_list = []
        for entry in cached:
            if entry is None:
                new_list.append(None)
            elif isinstance(entry, tuple):
                # cached_states[step] = (states_tensor, indices_tensor)
                new_list.append(tuple(t.to(device) if isinstance(t, torch.Tensor) else t for t in entry))
            elif isinstance(entry, torch.Tensor):
                new_list.append(entry.to(device))
            else:
                new_list.append(entry)
        setattr(batch, list_name, new_list)

    # Pre-allocated internal flow / termination buffers, with two reuse paths:
    #   * ``_forward_flows_buffers`` / ``_backward_flows_buffers`` are reused via
    #     ``GFlowNet._fresh_flow_tensors``, which checks shape AND device and falls
    #     back to fresh allocation — so moving them is a perf optimization.
    #   * ``_terminated_buffers`` is reused inline in ``apply_actions_step`` with a
    #     shape-only check, so leaving a CPU buffer after the batch moves to GPU
    #     would scatter CUDA writes into CPU storage.
    for buf_name in ("_forward_flows_buffers", "_backward_flows_buffers", "_terminated_buffers"):
        bufs = getattr(batch, buf_name, None)
        if not bufs:
            continue
        setattr(batch, buf_name, [
            b.to(device) if isinstance(b, torch.Tensor) else b for b in bufs
        ])

    # Inform any device-aware consumer.
    if hasattr(batch, "device"):
        batch.device = device
    return batch


def sampler_worker(worker_id: int, 
                  param_queue: mp.Queue,
                  batch_queue: mp.Queue,
                  config: Dict,
                  stop_event: MPEvent):  # Fixed type annotation
    """
    Sampler process that generates trajectories on CPU.

    Args:
        worker_id: Unique identifier for this sampler
        param_queue: Queue to receive parameter updates
        batch_queue: Queue to send completed batches
        config: Full configuration dict
        stop_event: Signal to terminate
    """
    torch.set_num_threads(1)  # Prevent CPU oversubscription
    
    try:
        # Initialize components on CPU
        model_config = config["model"]
        training_config = config["training"]
        quantum_config = config["quantum"]
        shared_gfn_kwargs = build_gfn_kwargs(model_config)
        
        n_qubits = len(quantum_config["pauli_str_list"][0])

        # Deliberately omits sampling_mode and the use_fused_* / use_bf16_* knobs:
        # this CPU worker performs all async on-policy sampling and stays on
        # dynamic_active. bucketed/static_full are GPU optimizations whose fast paths
        # gate off when device.type != 'cuda', and the async master GFlowNet does not
        # sample (it only learns from the batch queue). async_learner warns when a
        # non-dynamic_active mode is requested so the no-op is visible.
        gfn = GFlowNet(
            n_qubits=n_qubits,
            hidden_dim=model_config["hidden_dim"],
            num_hidden_layers=model_config["num_hidden_layers"],
            lr=model_config["lr"],  # Not used for sampling
            weight_decay=model_config["weight_decay"],  # Not used
            device=torch.device('cpu'),
            model_type=model_config.get("model_type", "clifford_mlp"),
            model_kwargs=model_config.get("model_kwargs", {}),
            objective_type=model_config.get("objective_type", "tb"),
            objective_kwargs=model_config.get("objective_kwargs", {}),
            K=training_config["K"],
            measurement_backend=_cpu_sampler_measurement_backend(model_config),
            buffer_strategy=shared_gfn_kwargs["buffer_strategy"],
            adaptive_warmup=shared_gfn_kwargs["adaptive_warmup"],
            use_uint8_state_cache=shared_gfn_kwargs["use_uint8_state_cache"],
        )
        
        # Set to eval mode - no gradients needed
        gfn.pf_model.eval()

        # Initialize cost computer using the same full setup the sync trainer
        # uses — normalization, pauli_strings, custom costs, kwargs (incl.
        # epsilon-from-cost-config) — so async and sync optimize the same
        # objective surface and async no longer KeyError's on configs whose
        # epsilon lives under ``training.cost.epsilon``.
        cost_computer, cost_kwargs = _build_cost_computer(
            training_config, quantum_config, torch.device('cpu')
        )
        # Pre-compute normalized weights once (same as sync trainer's fast path).
        cost_computer.precompute_weights(quantum_config["w_list"])

        # Wait for initial parameters
        logging.info(f"[Sampler {worker_id}] Waiting for initial parameters...")
        initial_params = param_queue.get()
        _load_sampler_state_dict(gfn.pf_model, initial_params)
        logging.info(f"[Sampler {worker_id}] Received initial parameters, starting sampling")

        sample_count = 0

        while not stop_event.is_set():
            # Check for parameter updates (non-blocking)
            try:
                new_params = param_queue.get_nowait()
                _load_sampler_state_dict(gfn.pf_model, new_params)
                logging.info(f"[Sampler {worker_id}] Updated parameters")
            except Empty:
                pass
            
            # Sample trajectories. ``sample_trajectories`` takes ``max_depth`` (not
            # the legacy ``max_length``), and current configs use ``max_depth`` (not
            # ``max_layer``). Accept either to keep old checkpoints loadable, but
            # raise a clean error if neither is set rather than dying inside the call.
            max_depth = training_config.get("max_depth", training_config.get("max_layer"))
            if max_depth is None:
                raise KeyError(
                    "training_config missing both 'max_depth' (current) and "
                    "'max_layer' (legacy) — cannot determine trajectory depth."
                )
            with torch.no_grad():
                batch = gfn.sample_trajectories(
                    batch_size=training_config["update_freq"],
                    n_measurements=training_config["n_measurements"],
                    max_depth=max_depth,
                    mode=SamplingMode.ON_POLICY
                )
                
                # Compute costs on CPU. Pass through every kwarg the sync
                # trainer would (epsilon + any cost-fn-specific extras), so
                # async never silently runs a different objective surface.
                tableau = batch.batched_tableau
                probs = tableau.prob_P_multi(quantum_config["pauli_str_list"])
                costs = cost_computer.compute_batch_cost(
                    probs,
                    quantum_config["w_list"],
                    cost_kwargs.get("epsilon", 0.0),
                )
                
            # Send to learner (blocks if queue is full)
            try:
                batch_queue.put((batch, costs, worker_id), timeout=1.0)
                sample_count += 1
                
                if sample_count % 100 == 0:
                    logging.info(f"[Sampler {worker_id}] Generated {sample_count} batches")
                    
            except Full:
                continue  # Queue full, try again
                
    except Exception as e:
        logging.error(f"[Sampler {worker_id}] Error: {e}")
        traceback.print_exc()
        # Send error to learner
        batch_queue.put(("ERROR", str(e), worker_id))
    finally:
        logging.info(f"[Sampler {worker_id}] Shutting down")


def async_learner(config: Dict, num_updates: int,
                  initial_state: Optional[Dict[str, Any]] = None,
                  reward_fn: Optional[Any] = None):
    """
    Main learner process that trains on GPU while samplers work on CPU.

    Args:
        config: Full configuration dict
        num_updates: Number of training updates
        initial_state: Optional dict ``{"pf_model": state_dict, "optimizer":
            state_dict, "top_trajectories": {...}}`` to seed the learner from.
            When ``train_async`` calls this after resume/warm-start, the
            caller's ``self.gfn`` already holds the resumed weights — passing
            it here keeps async from broadcasting freshly-initialised
            parameters back to the samplers and discarding the resume.
        reward_fn: Optional reward callable (e.g. ``log_reward_fn``). The
            ``GFlowNet`` constructor previously fell back to
            ``default_reward_fn`` here even when the sync trainer was using
            ``log_reward_fn``, so async could optimise a different reward
            surface than sync. Pass the same callable the caller built.
    """
    # Setup multiprocessing
    mp.set_start_method('spawn', force=True)

    # Get async-specific config
    num_samplers = config["training"].get("num_samplers", 2)
    pipeline_depth = config["training"].get("pipeline_depth", 4)
    broadcast_every = config["training"].get("broadcast_every", 10)

    # Visibility: async on-policy sampling runs in the CPU sampler_worker(s) (dynamic_active); the
    # master GFlowNet does not sample. A requested bucketed/static_full sampling_mode is therefore a
    # no-op in async mode — warn rather than silently ignore it.
    _req_mode = config["model"].get("sampling_mode")
    if _req_mode is not None and _req_mode not in ("dynamic_active", "dynamic"):
        logging.warning(
            "sampling_mode=%r is not applied in async training: on-policy sampling runs on CPU "
            "sampler workers (dynamic_active) and the master does not sample; bucketed/static_full "
            "are synchronous-GPU-sampling optimizations. Use synchronous training to exercise them.",
            _req_mode,
        )

    # Pre-initialize the return values BEFORE the try-block so a setup
    # failure (queue/process construction, GFlowNet init, etc.) doesn't
    # leave them ``UnboundLocalError`` at the bottom return. The default
    # ``completed_updates=0`` also doubles as the canonical "training
    # didn't finish" signal for the caller's checkpoint-name logic.
    metrics_history: "defaultdict[str, list]" = defaultdict(list)
    timing_history: "defaultdict[str, list]" = defaultdict(list)
    trained_state: Optional[Dict[str, Any]] = None
    completed_updates = 0
    setup_error: Optional[BaseException] = None

    # Create queues and events. One *param_queue per sampler* so the periodic
    # broadcast actually reaches every sampler — the previous shared
    # ``mp.Queue(maxsize=1)`` meant a single ``put_nowait`` was consumed by
    # only one sampler and the rest kept running on stale parameters.
    param_queues = [mp.Queue(maxsize=1) for _ in range(num_samplers)]
    batch_queue = mp.Queue(maxsize=pipeline_depth)
    stop_event = mp.Event()

    # Start sampler processes; each sampler reads from its own queue.
    samplers = []
    for i in range(num_samplers):
        p = mp.Process(
            target=sampler_worker,
            args=(i, param_queues[i], batch_queue, config, stop_event)
        )
        p.start()
        samplers.append(p)

    try:
        # Initialize master GFlowNet on GPU
        device = get_device(config["model"].get("device_preference"))
        quantum_config = config["quantum"]
        n_qubits = len(quantum_config["pauli_str_list"][0])

        # If the caller passed a reward callable, hand it to ``GFlowNet``. Otherwise
        # the constructor defaults to ``default_reward_fn``, which silently diverges
        # from a sync run using ``log_reward_fn`` — same config and kwargs, different
        # callable, and the loss numbers still look reasonable.
        gfn = GFlowNet(
            n_qubits=n_qubits,
            hidden_dim=config["model"]["hidden_dim"],
            num_hidden_layers=config["model"]["num_hidden_layers"],
            lr=config["model"]["lr"],
            weight_decay=config["model"]["weight_decay"],
            device=device,
            model_type=config["model"].get("model_type", "clifford_mlp"),
            model_kwargs=config["model"].get("model_kwargs", {}),
            objective_type=config["model"].get("objective_type", "tb"),
            objective_kwargs=config["model"].get("objective_kwargs", {}),
            K=config["training"]["K"],
            reward_fn=reward_fn,
            **build_gfn_kwargs(config["model"]),
        )

        # Seed from the caller's trained / resumed state if provided. Without
        # this, async resume silently starts from a fresh learner: the caller
        # loads a checkpoint into ``self.gfn``, calls ``train_async``, and
        # ``async_learner`` builds its own ``GFlowNet`` and broadcasts those
        # fresh weights — discarding the resume.
        if initial_state is not None:
            pf_seed = initial_state.get("pf_model")
            if pf_seed is not None:
                _load_sampler_state_dict(gfn.pf_model, pf_seed)
                logging.info("[Learner] Seeded pf_model from caller's initial_state")
            opt_seed = initial_state.get("optimizer")
            if opt_seed is not None and gfn.optimizer is not None:
                try:
                    gfn.optimizer.load_state_dict(opt_seed)
                    logging.info("[Learner] Seeded optimizer from caller's initial_state")
                except Exception as e:
                    logging.warning(f"[Learner] Could not seed optimizer: {e}")
            top_seed = initial_state.get("top_trajectories")
            if top_seed is not None:
                gfn.top_trajectories_actions = [
                    a.to(device) for a in top_seed.get("actions", [])
                ]
                gfn.top_trajectories_lengths = [
                    l.to(device) for l in top_seed.get("lengths", [])
                ]
                gfn.top_trajectories_costs = list(top_seed.get("costs", []))
                logging.info(
                    f"[Learner] Seeded top-K with {len(gfn.top_trajectories_actions)} "
                    f"trajectories from caller's initial_state"
                )

        # Broadcast initial parameters — one put per sampler queue.
        # ``_broadcast_state_dict`` unwraps a ``torch.compile``'d learner so
        # the eager CPU sampler doesn't see ``_orig_mod.`` prefixed keys,
        # AND it detach-clones tensors to CPU so we no longer have to
        # round-trip the live learner module through ``.cpu()`` / ``.to(device)``.
        initial_params = _broadcast_state_dict(gfn.pf_model)
        for q in param_queues:
            q.put(initial_params)

        logging.info(f"\n[Learner] Starting async training with {num_samplers} samplers")
        logging.info(f"[Learner] Pipeline depth: {pipeline_depth}, broadcast every: {broadcast_every}")

        for update in range(num_updates):
            update_start = time.time()
            
            # Get a batch from the queue with a periodic timeout so a sampler killed
            # by OOM/SIGKILL (no Python error sentinel) doesn't block the learner
            # forever — hard deaths skip past try/except entirely.
            # ``batch_queue_max_wait_s`` bounds the TOTAL wait for a live-but-stuck
            # sampler; without it the inner loop would wake, see all samplers alive,
            # log, and re-enter the wait indefinitely. Default 600s, config-tunable
            # because the right value depends on update rate.
            batch_queue_max_wait_s = float(
                config["training"].get("batch_queue_max_wait_s", 600.0)
            )
            queue_wait_start = time.time()
            batch_data = None
            while batch_data is None:
                try:
                    batch_data = batch_queue.get(timeout=30.0)
                except Empty:
                    # Liveness probe. If any sampler has exited (exitcode
                    # is not None), the queue won't be refilled — turn
                    # that into a surfaced learner error instead of
                    # waiting indefinitely.
                    dead = [
                        (idx, p.exitcode)
                        for idx, p in enumerate(samplers)
                        if not p.is_alive()
                    ]
                    if dead:
                        msg = (
                            f"async sampler(s) died without sending an "
                            f"ERROR sentinel: "
                            + ", ".join(f"sampler {i} exitcode={ec}" for i, ec in dead)
                        )
                        logging.error(f"[Learner] {msg}")
                        setup_error = RuntimeError(msg)
                        break
                    # All samplers alive; check whether we've exceeded the
                    # hard deadline. An alive-but-wedged sampler (deadlock,
                    # tight Python loop with no Q.put, etc.) would have
                    # otherwise blocked the learner forever.
                    elapsed = time.time() - queue_wait_start
                    if elapsed >= batch_queue_max_wait_s:
                        msg = (
                            f"async batch_queue empty for {elapsed:.0f}s "
                            f">= batch_queue_max_wait_s={batch_queue_max_wait_s:.0f}s "
                            f"at update {update}; all {num_samplers} samplers "
                            f"alive but producing nothing — declaring stuck"
                        )
                        logging.error(f"[Learner] {msg}")
                        setup_error = RuntimeError(msg)
                        break
                    logging.warning(
                        f"[Learner] batch_queue empty after {elapsed:.0f}s wait at update {update}; "
                        f"all {num_samplers} samplers still alive — continuing "
                        f"(deadline at {batch_queue_max_wait_s:.0f}s)"
                    )
            if setup_error is not None:
                break
            queue_wait_time = time.time() - queue_wait_start

            # Check for errors. ``completed_updates`` is *not* incremented for
            # this iteration, so the caller's checkpoint-name logic can tell
            # the run died here rather than finishing num_updates cleanly.
            if batch_data[0] == "ERROR":
                logging.error(f"[Learner] Received error from sampler {batch_data[2]}: {batch_data[1]}")
                setup_error = RuntimeError(
                    f"async sampler {batch_data[2]} failed: {batch_data[1]}"
                )
                break
            
            batch, costs, sampler_id = batch_data
            
            # Move batch to GPU
            transfer_start = time.time()
            batch = move_batch_to_device(batch, device)
            costs = costs.to(device)
            transfer_time = time.time() - transfer_start
            
            # Compute loss and update. ``metrics_to_cpu=False`` + ``return_tensor=True``
            # keep the hot path off the CPU sync boundary; the periodic logging
            # conversion is the one sync per ``broadcast_every`` iterations. The
            # defaults would force an ``.item()`` per update on the learner GPU,
            # undercutting the async pipeline's overlap with sampler work.
            train_start = time.time()
            loss, metrics = gfn.compute_loss(
                batch, costs,
                config["training"]["beta"],
                max_depth=config["training"].get("max_depth"),
                metrics_to_cpu=False,
                **config["training"].get("reward_kwargs", {})
            )

            loss_value = gfn.update_step(loss, return_tensor=True)
            train_time = time.time() - train_start
            
            # Update top trajectories. ``_update_top_trajectories`` is COST-ordered
            # (``largest=False``), so it must be fed ``costs`` directly — passing the
            # reward-transformed values would keep the smallest rewards, i.e. the
            # worst trajectories. Synchronous training passes ``costs`` too.
            gfn._update_top_trajectories(batch, costs)

            # Broadcast parameters periodically. ``_broadcast_state_dict``
            # detach-clones the (unwrapped) inner module's tensors to CPU,
            # so the learner stays on GPU throughout — we no longer need
            # the round-trip ``.cpu()`` / ``.to(device)`` cycle that made
            # the queue-full path tricky.
            if (update + 1) % broadcast_every == 0:
                new_params = _broadcast_state_dict(gfn.pf_model)
                # Try to push to each sampler's queue independently. A
                # sampler that hasn't drained its previous snapshot keeps
                # the old one (its put_nowait raises Full); the others
                # update on this round. Aggregate count for the log line.
                delivered = 0
                for q in param_queues:
                    try:
                        q.put_nowait(new_params)
                        delivered += 1
                    except Full:
                        pass
                if delivered:
                    logging.info(
                        f"[Learner] Broadcast parameters at update "
                        f"{update + 1} (delivered to {delivered}/{len(param_queues)} samplers)"
                    )
            
            # Log metrics
            metrics['loss'] = loss_value
            metrics['sampler_id'] = sampler_id
            metrics['queue_wait_ms'] = queue_wait_time * 1000
            metrics['transfer_ms'] = transfer_time * 1000
            metrics['train_ms'] = train_time * 1000
            
            for k, v in metrics.items():
                metrics_history[k].append(v)
            
            # Progress logging — convert the GPU tensors to host floats
            # here (the rate-limited sync boundary) rather than on every
            # iteration. ``.item()`` on the cached tensors triggers exactly
            # one host sync per ``broadcast_every`` (default 10) updates.
            if (update + 1) % 10 == 0:
                update_time = time.time() - update_start
                throughput = config["training"]["update_freq"] * config["training"]["n_measurements"] / update_time

                _loss_f = loss_value.item() if torch.is_tensor(loss_value) else float(loss_value)
                _reward_f = metrics['reward'].item() if torch.is_tensor(metrics.get('reward')) else float(metrics.get('reward', 0.0))
                _cost_f = metrics['cost'].item() if torch.is_tensor(metrics.get('cost')) else float(metrics.get('cost', 0.0))

                logging.info(f"\n[Update {update + 1}/{num_updates}]")
                logging.info(f"  Loss: {_loss_f:.6f}, Reward: {_reward_f:.4f}, Cost: {_cost_f:.4f}")
                logging.info(f"  Timing - Queue wait: {queue_wait_time*1000:.1f}ms, "
                      f"Transfer: {transfer_time*1000:.1f}ms, Train: {train_time*1000:.1f}ms")
                logging.info(f"  Throughput: {throughput:.1f} traj/s (from sampler {sampler_id})")
                # ``Queue.qsize`` raises ``NotImplementedError`` on macOS;
                # treat it as best-effort progress logging only.
                try:
                    qsize_str = f"{batch_queue.qsize()}/{pipeline_depth}"
                except (NotImplementedError, OSError):
                    qsize_str = f"unknown/{pipeline_depth}"
                logging.info(f"  Queue depth: {qsize_str}")

            # Only count an update as "completed" *after* the loss step + top-K
            # update finish without exception. The caller uses this to decide
            # whether to write a clean ``checkpoint_update.pth`` (final) or a
            # ``checkpoint_partial.pth``; an off-by-one here means a crashed
            # run can be silently mistaken for a finished one.
            completed_updates = update + 1

        logging.info(f"\n[Learner] Training complete ({completed_updates}/{num_updates} updates)")

    except Exception as e:
        logging.error(f"[Learner] Error: {e}")
        traceback.print_exc()
        setup_error = e
        
    finally:
        # Signal samplers to stop
        stop_event.set()

        # Clear queues to unblock samplers
        while not batch_queue.empty():
            try:
                batch_queue.get_nowait()
            except Empty:
                break

        # Wait for samplers to finish
        for p in samplers:
            p.join(timeout=5.0)
            if p.is_alive():
                p.terminate()

        logging.info("[Learner] All processes terminated")

        # Snapshot the trained learner state so the caller's trainer can adopt it.
        # Without this, ``train_async`` would ingest metrics from training the LOCAL
        # ``gfn`` but leave ``self.gfn`` at its initial weights, silently discarding
        # everything async learned. ``gfn`` may not exist if the try-block raised
        # before construction, which is why this lives in its own try block.
        try:
            captured: Dict[str, Any] = {
                "pf_model": _broadcast_state_dict(gfn.pf_model),
            }
            # Top trajectories live on ``gfn`` and are updated via
            # ``_update_top_trajectories``; without explicit capture they never reach
            # ``self.gfn`` in ``train_async`` and the final checkpoint carries empty
            # top-K lists. CPU-clone each tensor so the caller can adopt them.
            top_actions = getattr(gfn, "top_trajectories_actions", None)
            if top_actions:
                captured["top_trajectories"] = {
                    "actions": [a.detach().cpu().clone() for a in top_actions],
                    "lengths": [l.detach().cpu().clone() for l in gfn.top_trajectories_lengths],
                    "costs": [
                        (c.detach().cpu().item() if torch.is_tensor(c) else c)
                        for c in gfn.top_trajectories_costs
                    ],
                }
            # Optimizer state is opportunistic — tensors may carry CUDA
            # device, which load_state_dict on the caller side handles.
            # Wrap in its own try because some optimizers (e.g. when
            # objective is exotic) may not have a meaningful state.
            try:
                captured["optimizer"] = gfn.optimizer.state_dict()
            except Exception as opt_err:  # pragma: no cover
                logging.warning(f"[Learner] Optimizer state capture failed: {opt_err}")
            trained_state = captured
        except NameError:
            logging.warning("[Learner] Trained state unavailable (gfn never constructed)")

    # Return ``setup_error`` alongside the partial state so ``train_async`` can save
    # a partial checkpoint BEFORE surfacing the failure. Previously the function
    # either re-raised (losing the partial state) or returned a successful-looking
    # tuple when ``completed_updates > 0``, so orchestration saw a clean completion
    # even though training died mid-run.
    return metrics_history, timing_history, trained_state, completed_updates, setup_error
