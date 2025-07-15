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

from GFNs import GFlowNet, TrajectoryBatch, SamplingMode, get_device
from clifford_map import BatchedCliffordMap
from cost_computer import CostComputer


def move_batch_to_device(batch: TrajectoryBatch, device: torch.device) -> TrajectoryBatch:
    """Move TrajectoryBatch tensors to specified device."""
    batch.actions = batch.actions.to(device)
    batch.lengths = batch.lengths.to(device)
    batch.active = batch.active.to(device)
    batch.masks = batch.masks.to(device)
    batch.last_single_qubit_gates = batch.last_single_qubit_gates.to(device)
    batch.last_two_qubit_gates = batch.last_two_qubit_gates.to(device)
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
        
        n_qubits = len(quantum_config["pauli_str_list"][0])
        
        # Create GFlowNet instance on CPU
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
            K=training_config["K"]
        )
        
        # Set to eval mode - no gradients needed
        gfn.pf_model.eval()
        
        # Initialize cost computer
        cost_config = training_config.get("cost", {})
        cost_computer = CostComputer(
            cost_type=cost_config.get("type", "exponential"),
            n_measurements=training_config["n_measurements"],
            device=torch.device('cpu')
        )
        
        # Wait for initial parameters
        logging.info(f"[Sampler {worker_id}] Waiting for initial parameters...")
        initial_params = param_queue.get()
        gfn.pf_model.load_state_dict(initial_params)
        logging.info(f"[Sampler {worker_id}] Received initial parameters, starting sampling")
        
        sample_count = 0
        
        while not stop_event.is_set():
            # Check for parameter updates (non-blocking)
            try:
                new_params = param_queue.get_nowait()
                gfn.pf_model.load_state_dict(new_params)
                logging.info(f"[Sampler {worker_id}] Updated parameters")
            except Empty:
                pass
            
            # Sample trajectories
            with torch.no_grad():
                batch = gfn.sample_trajectories(
                    batch_size=training_config["update_freq"],
                    n_measurements=training_config["n_measurements"],
                    max_length=training_config["max_layer"],
                    mode=SamplingMode.ON_POLICY
                )
                
                # Compute costs on CPU
                tableau = batch.batched_tableau
                probs = tableau.prob_P_multi(quantum_config["pauli_str_list"])
                costs = cost_computer.compute_batch_cost(
                    probs, 
                    quantum_config["w_list"], 
                    training_config["epsilon"]
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


def async_learner(config: Dict, num_updates: int):
    """
    Main learner process that trains on GPU while samplers work on CPU.
    
    Args:
        config: Full configuration dict
        num_updates: Number of training updates
    """
    # Setup multiprocessing
    mp.set_start_method('spawn', force=True)
    
    # Get async-specific config
    num_samplers = config["training"].get("num_samplers", 2)
    pipeline_depth = config["training"].get("pipeline_depth", 4)
    broadcast_every = config["training"].get("broadcast_every", 10)
    
    # Create queues and events
    param_queue = mp.Queue(maxsize=1)
    batch_queue = mp.Queue(maxsize=pipeline_depth)
    stop_event = mp.Event()
    
    # Start sampler processes
    samplers = []
    for i in range(num_samplers):
        p = mp.Process(
            target=sampler_worker,
            args=(i, param_queue, batch_queue, config, stop_event)
        )
        p.start()
        samplers.append(p)
    
    try:
        # Initialize master GFlowNet on GPU
        device = get_device(config["model"].get("device_preference"))
        quantum_config = config["quantum"]
        n_qubits = len(quantum_config["pauli_str_list"][0])
        
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
            K=config["training"]["K"]
        )
        
        # Broadcast initial parameters
        initial_params = gfn.pf_model.cpu().state_dict()
        for _ in range(num_samplers):
            param_queue.put(initial_params)
        gfn.pf_model.to(device)
        
        # Training metrics
        metrics_history = defaultdict(list)
        timing_history = defaultdict(list)
        
        logging.info(f"\n[Learner] Starting async training with {num_samplers} samplers")
        logging.info(f"[Learner] Pipeline depth: {pipeline_depth}, broadcast every: {broadcast_every}")
        
        for update in range(num_updates):
            update_start = time.time()
            
            # Get batch from queue (blocks until available)
            queue_wait_start = time.time()
            batch_data = batch_queue.get()
            queue_wait_time = time.time() - queue_wait_start
            
            # Check for errors
            if batch_data[0] == "ERROR":
                logging.error(f"[Learner] Received error from sampler {batch_data[2]}: {batch_data[1]}")
                break
            
            batch, costs, sampler_id = batch_data
            
            # Move batch to GPU
            transfer_start = time.time()
            batch = move_batch_to_device(batch, device)
            costs = costs.to(device)
            transfer_time = time.time() - transfer_start
            
            # Compute loss and update
            train_start = time.time()
            loss, metrics = gfn.compute_loss(
                batch, costs,
                config["training"]["beta"],
                max_depth=config["training"].get("max_depth"),
                **config["training"].get("reward_kwargs", {})
            )
            
            loss_value = gfn.update_step(loss)
            train_time = time.time() - train_start
            
            # Update top trajectories
            rewards = gfn.reward_fn(
                costs, 
                beta=config["training"]["beta"], 
                **config["training"].get("reward_kwargs", {})
            )
            gfn._update_top_trajectories_optimized(batch, rewards)
            
            # Broadcast parameters periodically
            if (update + 1) % broadcast_every == 0:
                try:
                    # Try to put new params (non-blocking)
                    new_params = gfn.pf_model.cpu().state_dict()
                    param_queue.put_nowait(new_params)
                    gfn.pf_model.to(device)
                    logging.info(f"[Learner] Broadcast parameters at update {update + 1}")
                except Full:
                    # Old params haven't been consumed yet
                    pass
            
            # Log metrics
            metrics['loss'] = loss_value
            metrics['sampler_id'] = sampler_id
            metrics['queue_wait_ms'] = queue_wait_time * 1000
            metrics['transfer_ms'] = transfer_time * 1000
            metrics['train_ms'] = train_time * 1000
            
            for k, v in metrics.items():
                metrics_history[k].append(v)
            
            # Progress logging
            if (update + 1) % 10 == 0:
                update_time = time.time() - update_start
                throughput = config["training"]["update_freq"] * config["training"]["n_measurements"] / update_time
                
                logging.info(f"\n[Update {update + 1}/{num_updates}]")
                logging.info(f"  Loss: {loss_value:.6f}, Reward: {metrics['reward']:.4f}, Cost: {metrics['cost']:.4f}")
                logging.info(f"  Timing - Queue wait: {queue_wait_time*1000:.1f}ms, "
                      f"Transfer: {transfer_time*1000:.1f}ms, Train: {train_time*1000:.1f}ms")
                logging.info(f"  Throughput: {throughput:.1f} traj/s (from sampler {sampler_id})")
                logging.info(f"  Queue depth: {batch_queue.qsize()}/{pipeline_depth}")
        
        logging.info("\n[Learner] Training complete")
        
    except Exception as e:
        logging.error(f"[Learner] Error: {e}")
        traceback.print_exc()
        
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
        
        return metrics_history, timing_history
