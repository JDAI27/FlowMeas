# -*- coding: utf-8 -*-
# GFNs.py

import os
import json
import heapq
import torch
import torch.nn as nn
import logging
from math import exp
from torch.distributions import Categorical
import numpy as np
from collections import defaultdict
import time
from enum import Enum
from typing import List, Tuple, Dict, Optional, Union, Callable
import matplotlib.pyplot as plt

from clifford_map import CliffordMap
from models import DiscreteUniform, CliffordMLP, QuantumAwareMLP, AttentionMLP, create_clifford_model, CliffordTableauProcessor
from gfn_objectives import GFlowNetObjective, create_gfn_objective, OBJECTIVE_CONFIGS
from cost_computer import CostComputer, CostFunction, ThresholdCost

from quantum_action_mapping import build_action_mapping


def get_device(device_preference: Optional[str] = None) -> torch.device:
    """Get the best available device, with preference for CUDA > MPS > CPU"""
    if device_preference:
        if device_preference == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        elif device_preference == "mps" and torch.backends.mps.is_available():
            return torch.device("mps")
        elif device_preference == "cpu":
            return torch.device("cpu")
    
    # Auto-detect best device
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def default_reward_fn(costs: torch.Tensor, beta: float = 1.0, alpha: float = 5e-3 , **kwargs) -> torch.Tensor:
    """Default reward function: linear transformation of costs."""
    return beta * (alpha - costs)


def exponential_reward_fn(costs: torch.Tensor, beta: float = 1.0, alpha: float = 2.0, **kwargs) -> torch.Tensor:
    """Exponential reward function for stronger differentiation."""
    return beta * torch.exp(-alpha * costs)


def threshold_reward_fn(costs: torch.Tensor, beta: float = 1.0, threshold: float = 0.5, **kwargs) -> torch.Tensor:
    """Threshold-based reward function with bonus for achieving low cost."""
    base_reward = beta * (1.0 - costs)
    bonus = beta * (costs < threshold).float()
    return base_reward + bonus


def polynomial_reward_fn(costs: torch.Tensor, beta: float = 1.0, power: float = 2.0, **kwargs) -> torch.Tensor:
    """Polynomial reward function."""
    return beta * torch.pow(1.0 - costs, power)


class SamplingMode(Enum):
    """Enum for different sampling strategies"""
    ON_POLICY = "on_policy"
    OFF_POLICY = "off_policy"
    REPLAY = "replay"


class AdaptiveBufferTracker:
    """Track statistics to adaptively determine buffer sizes."""
    
    def __init__(self, initial_buffer_size: int, device: torch.device, 
                 warmup_updates: int = 100):
        self.device = device
        self.initial_buffer_size = initial_buffer_size
        self.warmup_updates = warmup_updates
        
        # Track statistics across trajectories
        self.gates_per_depth = defaultdict(list)
        self.buffer_utilization = []
        self.depth_distribution = defaultdict(int)
        
        # Running statistics (kept on GPU for efficiency)
        self.max_gates_seen = torch.tensor(0, device=device)
        self.total_trajectories = 0
        self.update_count = 0
        
        # Percentile tracking for robust estimation
        self.gate_counts = []  # All observed gate counts
        
    def update_statistics(self, trajectory_batch):
        """Update statistics from a batch of trajectories."""
        batch_size = trajectory_batch.batch_size
        n_measurements = trajectory_batch.n_measurements
        
        # Process each trajectory
        for b_idx in range(batch_size):
            for m_idx in range(n_measurements):
                if trajectory_batch.lengths[b_idx, m_idx] > 0:
                    depth = trajectory_batch.circuit_depths[b_idx, m_idx].item()
                    gates = trajectory_batch.lengths[b_idx, m_idx].item()
                    
                    self.gates_per_depth[depth].append(gates)
                    self.gate_counts.append(gates)
                    self.depth_distribution[depth] += 1
                    self.total_trajectories += 1
                    
                    # Update max gates seen
                    self.max_gates_seen = torch.max(
                        self.max_gates_seen, 
                        trajectory_batch.lengths[b_idx, m_idx]
                    )
        
        # Update buffer utilization
        max_used = trajectory_batch.lengths.max()
        utilization = max_used.float() / trajectory_batch.max_length
        self.buffer_utilization.append(utilization.item())
        
        self.update_count += 1
        
    def get_recommended_buffer_size(self, max_depth: int, percentile: float = 95.0) -> int:
        """Get recommended buffer size based on statistics."""
        # During warmup, return initial size
        if self.update_count < self.warmup_updates:
            return self.initial_buffer_size
        
        # If we have data for this specific depth
        if max_depth in self.gates_per_depth and len(self.gates_per_depth[max_depth]) >= 20:
            gates_at_depth = self.gates_per_depth[max_depth]
            gates_tensor = torch.tensor(gates_at_depth, device=self.device, dtype=torch.float32)
            recommended = torch.quantile(gates_tensor, percentile / 100.0).item()
        
        # Otherwise, use all data with scaling
        elif len(self.gate_counts) >= 100:
            all_gates = torch.tensor(self.gate_counts, device=self.device, dtype=torch.float32)
            base_percentile = torch.quantile(all_gates, percentile / 100.0).item()
            
            # Scale based on depth ratio
            avg_depth = sum(d * count for d, count in self.depth_distribution.items()) / self.total_trajectories
            depth_ratio = max_depth / max(avg_depth, 1.0)
            recommended = base_percentile * depth_ratio
        
        else:
            # Not enough data, use conservative estimate with slight reduction
            recommended = self.initial_buffer_size * 0.8
        
        # Ensure we have some headroom (10% safety margin)
        recommended = int(recommended * 1.1)
        
        # Never go below 50% of initial conservative estimate
        min_size = int(self.initial_buffer_size * 0.5)
        
        # Never exceed initial conservative estimate
        return max(min_size, min(recommended, self.initial_buffer_size))
    
    def get_statistics_summary(self) -> Dict:
        """Get summary of buffer statistics."""
        if not self.buffer_utilization:
            return {}
        
        recent_utilization = self.buffer_utilization[-100:] if len(self.buffer_utilization) > 100 else self.buffer_utilization
        
        return {
            'avg_utilization': np.mean(recent_utilization),
            'max_utilization': np.max(recent_utilization),
            'max_gates_seen': self.max_gates_seen.item(),
            'total_trajectories': self.total_trajectories,
            'updates': self.update_count
        }


class TrajectoryBatch:
    """Container for batch trajectory data with circuit depth tracking."""
    
    def __init__(self, batch_size: int, n_measurements: int, max_length: int, 
                n_qubits: int, device: torch.device):
        self.batch_size = batch_size
        self.n_measurements = n_measurements
        self.max_length = max_length
        self.n_qubits = n_qubits
        self.device = device
        
        # Keep all tensors on GPU
        self.actions = torch.zeros((batch_size, n_measurements, max_length), 
                                    dtype=torch.long, device=device)
        self.lengths = torch.zeros((batch_size, n_measurements), 
                                    dtype=torch.long, device=device)
        self.active = torch.ones((batch_size, n_measurements), 
                                dtype=torch.bool, device=device)
        self.masks = torch.ones((batch_size, n_measurements, max_length), 
                                dtype=torch.bool, device=device)
        
        # Circuit depth tracking
        self.circuit_depths = torch.zeros((batch_size, n_measurements), 
                                         dtype=torch.long, device=device)
        
        # Track which qubits are occupied in the current layer
        self.current_layer_qubits = torch.zeros((batch_size, n_measurements, n_qubits), 
                                               dtype=torch.bool, device=device)
        
        # Track the last layer where each qubit was used
        self.qubit_last_layer = torch.zeros((batch_size, n_measurements, n_qubits), 
                                           dtype=torch.long, device=device) - 1
        
        # Gate tracking
        self.last_single_qubit_gates = torch.zeros((batch_size, n_measurements, n_qubits), 
                                                    dtype=torch.long, device=device) - 1
        self.last_two_qubit_gates = torch.zeros((batch_size, n_measurements, n_qubits, n_qubits), 
                                                dtype=torch.long, device=device) - 1
        
        # Add: Track which step each qubit was last used (for backward policy)
        self.qubit_last_use_step = torch.full((batch_size, n_measurements, n_qubits), 
                                              -1, dtype=torch.long, device=device)
        
        # Add: Track action history with qubit info for exposed gate computation
        self.action_qubits = torch.full((batch_size, n_measurements, max_length, 2), 
                                        -1, dtype=torch.long, device=device)  # -1 for unused
        
        self.batched_tableau = None
        
    def set_action(self, batch_idx: int, meas_idx: int, step: int, action: int):
        """Set action for a specific trajectory at a specific step"""
        self.actions[batch_idx, meas_idx, step] = action
        
    def set_length(self, batch_idx: int, meas_idx: int, length: int):
        """Set the length of a specific trajectory"""
        self.lengths[batch_idx, meas_idx] = length
        self.masks[batch_idx, meas_idx, length:] = False
        
    def deactivate(self, batch_idx: int, meas_idx: int):
        """Mark a trajectory as inactive"""
        self.active[batch_idx, meas_idx] = False
        
    def batch_set_actions(self, indices: torch.Tensor, step: int, actions: torch.Tensor):
        """Batch set actions for multiple trajectories"""
        self.actions[indices[:, 0], indices[:, 1], step] = actions
        
    def batch_set_lengths(self, indices: torch.Tensor, lengths: torch.Tensor):
        """Batch set lengths for multiple trajectories"""
        self.lengths[indices[:, 0], indices[:, 1]] = lengths
        # Update masks
        for i, (b_idx, m_idx) in enumerate(indices):
            self.masks[b_idx, m_idx, lengths[i]:] = False


class GFlowNet:
    """
    GFlowNet with minimal CPU-GPU data transfer and depth-based sampling.
    
    batch_size = update_freq (number of batch elements)
    n_measurements = number of trajectories per batch element
    Total trajectories = batch_size × n_measurements
    """
    
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
                adaptive_warmup: int = 100):
        
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
        self.model_type = model_type
        self.debug = debug
        
        logging.info(f"Using device: {self.device}")
        logging.info(f"Buffer strategy: {self.buffer_strategy}")
        
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
        
        self.state_dim = (2 * n_qubits) ** 2 + (2 * n_qubits)
        
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
        
        self.optimizer = torch.optim.Adam([
            {'params': self.pf_model.logZ, 'lr': 100*lr},
            {'params': [p for n, p in self.pf_model.named_parameters() if n != 'logZ'], 
             'lr': lr , 'weight_decay': weight_decay}
        ])
        
        self.grad_clip_value = 10.0  # Gradient clipping value
        
        # Store top trajectories as tensors to avoid CPU transfer
        self.top_trajectories_actions = []
        self.top_trajectories_lengths = []
        self.top_trajectories_rewards = []
        self.top_trajectories_costs = []
        self.K = K
        
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
        self.two_qubit_gates = {"CNOT"} #, "SWAP"}
        
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
    
    def calculate_conservative_buffer_size(self, max_depth: int) -> int:
        """Calculate conservative upper bound for buffer size."""
        # Worst case: all single-qubit gates (each qubit gets a gate at each layer)
        single_qubit_bound = self.n_qubits * max_depth
        
        # Add safety margin
        return int(self.conservative_multiplier * single_qubit_bound)
    
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
    
    def apply_actions_to_batch_optimized(self, 
                                       batched_tableau: CliffordMap,
                                       actions: torch.Tensor,
                                       trajectory_batch: TrajectoryBatch,
                                       step: Optional[int] = None) -> torch.Tensor:
        """Apply actions to batched tableau with depth tracking."""
        batch_size, n_measurements = actions.shape
        terminated = torch.zeros((batch_size, n_measurements), dtype=torch.bool, device=self.device)
        
        # Flatten indices for batch processing
        active_mask = trajectory_batch.active
        active_indices = torch.nonzero(active_mask, as_tuple=False)

        if active_indices.shape[0] == 0:
            return terminated

        # Get actions for active trajectories
        active_actions = actions[active_indices[:, 0], active_indices[:, 1]]

        # Apply all gates in a single batched call
        batched_tableau.apply_actions_step(actions, self.action_mapping, active_mask)
        
        # Group by action type using GPU operations
        for action_idx in range(self.num_actions):
            action_mask = (active_actions == action_idx)
            if not action_mask.any():
                continue
            
            # Get indices where this action is applied
            action_indices = active_indices[action_mask]
            
            action = self.action_mapping[action_idx]
            gate_name = action[0]
            
            if gate_name == "terminal":
                # Mark as terminated
                terminated[action_indices[:, 0], action_indices[:, 1]] = True
                trajectory_batch.active[action_indices[:, 0], action_indices[:, 1]] = False
                
                # Update batched tableau active status
                for idx in action_indices:
                    batched_tableau.active[idx[0].item(), idx[1].item()] = False
            else:
                # Check if we need to start a new layer for any trajectory
                needs_new_layer = torch.zeros(len(action_indices), dtype=torch.bool, device=self.device)
                
                if gate_name in self.single_qubit_gates:
                    q = action[1]
                    # Check if this qubit is already used in current layer
                    for i, (b_idx, m_idx) in enumerate(action_indices):
                        if trajectory_batch.current_layer_qubits[b_idx, m_idx, q]:
                            needs_new_layer[i] = True
                            
                elif gate_name in self.two_qubit_gates:
                    q1, q2 = action[1], action[2]
                    # Check if either qubit is already used in current layer
                    for i, (b_idx, m_idx) in enumerate(action_indices):
                        if (trajectory_batch.current_layer_qubits[b_idx, m_idx, q1] or 
                            trajectory_batch.current_layer_qubits[b_idx, m_idx, q2]):
                            needs_new_layer[i] = True
                
                # Update depth and reset current layer for trajectories that need it
                if needs_new_layer.any():
                    new_layer_indices = action_indices[needs_new_layer]
                    trajectory_batch.circuit_depths[new_layer_indices[:, 0], new_layer_indices[:, 1]] += 1
                    # Reset current layer qubits for these trajectories
                    trajectory_batch.current_layer_qubits[new_layer_indices[:, 0], new_layer_indices[:, 1]] = False
                
                # Update gate tracking and layer information
                gate_idx = self.gate_name_to_idx[gate_name]
                current_depth = trajectory_batch.circuit_depths[action_indices[:, 0], action_indices[:, 1]]
                
                # Add: Update qubit last use step
                if step is not None:
                    if gate_name in self.single_qubit_gates:
                        q = action[1]
                        trajectory_batch.qubit_last_use_step[
                            action_indices[:, 0], action_indices[:, 1], q
                        ] = step
                        # Store action qubit info
                        trajectory_batch.action_qubits[
                            action_indices[:, 0], action_indices[:, 1], step, 0
                        ] = q
                        
                    elif gate_name in self.two_qubit_gates:
                        q1, q2 = action[1], action[2]
                        trajectory_batch.qubit_last_use_step[
                            action_indices[:, 0], action_indices[:, 1], q1
                        ] = step
                        trajectory_batch.qubit_last_use_step[
                            action_indices[:, 0], action_indices[:, 1], q2
                        ] = step
                        # Store action qubit info
                        trajectory_batch.action_qubits[
                            action_indices[:, 0], action_indices[:, 1], step, 0
                        ] = q1
                        trajectory_batch.action_qubits[
                            action_indices[:, 0], action_indices[:, 1], step, 1
                        ] = q2
                
                if gate_name in self.single_qubit_gates:
                    q = action[1]
                    trajectory_batch.last_single_qubit_gates[
                        action_indices[:, 0], action_indices[:, 1], q
                    ] = gate_idx
                    # Mark qubit as used in current layer
                    trajectory_batch.current_layer_qubits[action_indices[:, 0], action_indices[:, 1], q] = True
                    # Update last layer for this qubit
                    trajectory_batch.qubit_last_layer[action_indices[:, 0], action_indices[:, 1], q] = current_depth
                    
                elif gate_name in self.two_qubit_gates:
                    q1, q2 = action[1], action[2]
                    trajectory_batch.last_two_qubit_gates[
                        action_indices[:, 0], action_indices[:, 1], q1, q2
                    ] = gate_idx
                    trajectory_batch.last_two_qubit_gates[
                        action_indices[:, 0], action_indices[:, 1], q2, q1
                    ] = gate_idx
                    # Mark both qubits as used in current layer
                    trajectory_batch.current_layer_qubits[action_indices[:, 0], action_indices[:, 1], q1] = True
                    trajectory_batch.current_layer_qubits[action_indices[:, 0], action_indices[:, 1], q2] = True
                    # Update last layer for both qubits
                    trajectory_batch.qubit_last_layer[action_indices[:, 0], action_indices[:, 1], q1] = current_depth
                    trajectory_batch.qubit_last_layer[action_indices[:, 0], action_indices[:, 1], q2] = current_depth
        
        return terminated
    
    def compute_action_masks_gpu(self, trajectory_batch: TrajectoryBatch,
                                 max_depth: Optional[int] = None) -> torch.Tensor:
        """Compute valid action masks entirely on GPU.

        If ``max_depth`` is provided, actions that would require starting a new
        layer when the trajectory is already at ``max_depth`` are masked out.
        """
        batch_size = trajectory_batch.batch_size
        n_measurements = trajectory_batch.n_measurements
        
        # Initialize all masks to True
        masks = torch.ones((batch_size, n_measurements, self.num_actions), 
                         dtype=torch.bool, device=self.device)
        
        # Inactive trajectories can only take terminal action
        inactive_mask = ~trajectory_batch.active
        if inactive_mask.any():
            # Get indices of inactive trajectories
            inactive_indices = torch.nonzero(inactive_mask, as_tuple=True)
            # Set all actions to False for inactive trajectories
            masks[inactive_indices[0], inactive_indices[1], :] = False
            # Set only terminal action to True for inactive trajectories
            masks[inactive_indices[0], inactive_indices[1], self.terminal_index] = True
        
        # For active trajectories, compute constraints
        active_mask = trajectory_batch.active
        
        if active_mask.any():
            depth_limit_mask = None
            if max_depth is not None:
                depth_limit_mask = trajectory_batch.circuit_depths >= max_depth

            # Check if any gates have been applied
            has_gates = (trajectory_batch.last_single_qubit_gates >= 0).any(dim=2) | \
                       (trajectory_batch.last_two_qubit_gates >= 0).any(dim=(2, 3))
            
            # For each action, check if it's valid
            for action_idx in range(self.num_actions):
                gate_type = self.action_gate_types[action_idx]
                gate_name = self.action_mapping[action_idx][0]
                
                if gate_name == "terminal":
                    continue
                
                if self.single_qubit_mask[action_idx]:
                    # Single qubit gate
                    q = self.action_qubit1[action_idx]

                    # Check if same gate already applied to this qubit
                    same_gate_mask = (trajectory_batch.last_single_qubit_gates[:, :, q] == gate_type)
                    masks[:, :, action_idx] = masks[:, :, action_idx] & ~same_gate_mask

                    if depth_limit_mask is not None:
                        requires_new_layer = trajectory_batch.current_layer_qubits[:, :, q]
                        masks[:, :, action_idx] = masks[:, :, action_idx] & ~(depth_limit_mask & requires_new_layer)

                elif self.two_qubit_mask[action_idx]:
                    # Two qubit gate
                    q1 = self.action_qubit1[action_idx]
                    q2 = self.action_qubit2[action_idx]
                    
                    # Check connectivity constraint
                    q1_has_gate = (trajectory_batch.last_single_qubit_gates[:, :, q1] >= 0) | \
                                  (trajectory_batch.last_two_qubit_gates[:, :, q1, :].max(dim=2)[0] >= 0)
                    q2_has_gate = (trajectory_batch.last_single_qubit_gates[:, :, q2] >= 0) | \
                                  (trajectory_batch.last_two_qubit_gates[:, :, q2, :].max(dim=2)[0] >= 0)
                    
                    connectivity_violated = has_gates & ~q1_has_gate & ~q2_has_gate

                    # Check if same gate already applied to this pair
                    same_gate_mask = (trajectory_batch.last_two_qubit_gates[:, :, q1, q2] == gate_type)
                    masks[:, :, action_idx] = masks[:, :, action_idx] & ~connectivity_violated & ~same_gate_mask

                    if depth_limit_mask is not None:
                        requires_new_layer = trajectory_batch.current_layer_qubits[:, :, q1] | \
                                             trajectory_batch.current_layer_qubits[:, :, q2]
                        masks[:, :, action_idx] = masks[:, :, action_idx] & ~(depth_limit_mask & requires_new_layer)
        
        # Terminal action is always valid
        masks[:, :, self.terminal_index] = True
        
        return masks
    
    def compute_backward_masks_gpu_vectorized(self, trajectory_batch: TrajectoryBatch,
                                             current_step: int) -> torch.Tensor:
        """Vectorized computation of backward masks based on exposed gates."""
        batch_size = trajectory_batch.batch_size
        n_measurements = trajectory_batch.n_measurements
        
        # Initialize masks
        masks = torch.zeros((batch_size, n_measurements, self.num_actions), 
                           dtype=torch.bool, device=self.device)
        
        # Create a tensor to track max last use for each action's qubits
        # Shape: (batch_size, n_measurements, num_actions)
        action_max_last_use = torch.full((batch_size, n_measurements, self.num_actions),
                                        -1, dtype=torch.long, device=self.device)
        
        # For single qubit actions
        single_qubit_actions = torch.where(self.single_qubit_mask)[0]
        for action_idx in single_qubit_actions:
            q = self.action_qubit1[action_idx]
            action_max_last_use[:, :, action_idx] = trajectory_batch.qubit_last_use_step[:, :, q]
        
        # For two qubit actions  
        two_qubit_actions = torch.where(self.two_qubit_mask)[0]
        for action_idx in two_qubit_actions:
            q1 = self.action_qubit1[action_idx]
            q2 = self.action_qubit2[action_idx]
            max_last_use = torch.maximum(
                trajectory_batch.qubit_last_use_step[:, :, q1],
                trajectory_batch.qubit_last_use_step[:, :, q2]
            )
            action_max_last_use[:, :, action_idx] = max_last_use
        
        # Check each step up to current_step
        for step in range(current_step):
            # Get actions at this step
            step_actions = trajectory_batch.actions[:, :, step]
            
            # Check if this step is within trajectory length
            valid_step = step < trajectory_batch.lengths
            
            # Vectorized check: for each (b,m), check if action at step is exposed
            for b in range(batch_size):
                for m in range(n_measurements):
                    if valid_step[b, m]:
                        action = step_actions[b, m]
                        if action != self.terminal_index:
                            # Check if this action's max last use equals current step
                            if action_max_last_use[b, m, action] == step:
                                masks[b, m, action] = True
        
        # Fallback for trajectories with no exposed gates
        no_exposed = ~masks.any(dim=2)
        if no_exposed.any():
            # Use standard masks excluding terminal
            forward_masks = self.compute_action_masks_gpu(trajectory_batch)
            forward_masks[..., self.terminal_index] = False
            masks[no_exposed] = forward_masks[no_exposed]
        
        return masks
    
    def sample_trajectories(self, 
                          batch_size: int,
                          n_measurements: int,
                          max_depth: int,  # Changed from max_length
                          mode: SamplingMode = SamplingMode.ON_POLICY,
                          batch_data_list: Optional[List[Dict]] = None) -> TrajectoryBatch:
        """Sample trajectories with depth limit and adaptive buffer sizing."""

        # Store max_depth for loss computation
        self.last_max_depth = max_depth

        # Determine buffer size
        max_length = self.determine_buffer_size(max_depth)
        
        if self.debug or (self.adaptive_tracker and self.adaptive_tracker.update_count % 50 == 0):
            logging.info(f"Buffer size for depth {max_depth}: {max_length} "
                  f"(strategy: {self.buffer_strategy})")
            if self.adaptive_tracker:
                stats = self.adaptive_tracker.get_statistics_summary()
                if stats:
                    logging.info(f"  Avg utilization: {stats['avg_utilization']:.1%}, "
                          f"Max gates seen: {stats['max_gates_seen']}")
        
        if mode == SamplingMode.REPLAY:
            return self._replay_trajectories_optimized(batch_data_list, max_length)
        
        # Create tableau on the same device as the model
        batched_tableau = CliffordMap(
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
        
        with torch.no_grad():
            for step in range(max_length):
                if not trajectory_batch.active.any():
                    break
                
                states_tensor, indices = batched_tableau.to_flat_tensors_active_only()
                if states_tensor.shape[0] == 0:
                    break
                
                # Initialize actions with terminal
                actions = torch.full((batch_size, n_measurements), self.terminal_index,
                                   dtype=torch.long, device=self.device)
                
                if mode == SamplingMode.ON_POLICY:
                    logits = self.pf_model(states_tensor)
                    masks = self.compute_action_masks_gpu(trajectory_batch, max_depth)

                    # Ensure indices tensor on device
                    if isinstance(indices, torch.Tensor):
                        indices_tensor = indices.to(self.device)
                    else:
                        indices_tensor = torch.as_tensor(indices, dtype=torch.long, device=self.device)

                    # Gather masks for active trajectories
                    active_masks = masks[indices_tensor[:, 0], indices_tensor[:, 1]]

                    # Mask invalid actions with -inf
                    masked_logits = logits.clone()
                    masked_logits[~active_masks] = float('-inf')

                    # Sample actions in batch
                    dist = Categorical(logits=masked_logits)
                    sampled_actions = dist.sample()

                    # Handle degenerate cases with no valid actions
                    valid_any = torch.isfinite(masked_logits).any(dim=1)
                    sampled_actions = torch.where(
                        valid_any,
                        sampled_actions,
                        torch.full_like(sampled_actions, self.terminal_index),
                    )

                    # Write back sampled actions
                    actions[indices_tensor[:, 0], indices_tensor[:, 1]] = sampled_actions
                    trajectory_batch.actions[indices_tensor[:, 0], indices_tensor[:, 1], step] = sampled_actions

                elif mode == SamplingMode.OFF_POLICY:
                    masks = self.compute_action_masks_gpu(trajectory_batch, max_depth)

                    # Ensure indices tensor on device
                    if isinstance(indices, torch.Tensor):
                        indices_tensor = indices.to(self.device)
                    else:
                        indices_tensor = torch.as_tensor(indices, dtype=torch.long, device=self.device)

                    active_masks = masks[indices_tensor[:, 0], indices_tensor[:, 1]]

                    # Uniform logits over valid actions
                    off_logits = torch.zeros_like(active_masks, dtype=torch.float32)
                    off_logits[~active_masks] = float('-inf')

                    dist = Categorical(logits=off_logits)
                    sampled_actions = dist.sample()

                    actions[indices_tensor[:, 0], indices_tensor[:, 1]] = sampled_actions
                    trajectory_batch.actions[indices_tensor[:, 0], indices_tensor[:, 1], step] = sampled_actions
                
                # Apply actions using depth-aware function with step tracking
                terminated = self.apply_actions_to_batch_optimized(
                    batched_tableau, actions, trajectory_batch, step=step
                )
                
                # Update lengths for terminated trajectories
                newly_terminated = terminated & (trajectory_batch.lengths == 0)
                if newly_terminated.any():
                    terminated_indices = torch.nonzero(newly_terminated, as_tuple=False)
                    trajectory_batch.lengths[terminated_indices[:, 0], terminated_indices[:, 1]] = step + 1
                
                # Handle max length reached
                if step == max_length - 1:
                    still_active = trajectory_batch.active & (trajectory_batch.lengths == 0)
                    if still_active.any():
                        active_indices = torch.nonzero(still_active, as_tuple=False)
                        trajectory_batch.lengths[active_indices[:, 0], active_indices[:, 1]] = max_length
                        trajectory_batch.active[active_indices[:, 0], active_indices[:, 1]] = False
        
        # Update adaptive statistics if using adaptive strategy
        if self.adaptive_tracker is not None:
            self.adaptive_tracker.update_statistics(trajectory_batch)
        
        return trajectory_batch
    
    def compute_flows(self, trajectory_batch: TrajectoryBatch,
                     max_depth: Optional[int] = None,
                     compute_gradients: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute forward and backward flows with minimal CPU-GPU transfer."""
        batch_size = trajectory_batch.batch_size
        n_measurements = trajectory_batch.n_measurements
        max_length = trajectory_batch.max_length
        
        forward_flows = torch.zeros((batch_size, n_measurements), device=self.device)
        backward_flows = torch.zeros((batch_size, n_measurements), device=self.device)
        
        # Create a new tableau for flow computation
        batched_tableau = CliffordMap(
            n_qubits=self.n_qubits,
            batch_size=batch_size,
            n_measurements=n_measurements,
            device=str(self.device)
        )
        
        # Create temporary batch for tracking state
        temp_batch = TrajectoryBatch(
            batch_size=batch_size,
            n_measurements=n_measurements,
            max_length=max_length,
            n_qubits=self.n_qubits,
            device=self.device
        )
        
        temp_batch.lengths = trajectory_batch.lengths.clone()
        temp_batch.active = trajectory_batch.lengths > 0
        batched_tableau.active = temp_batch.active.clone()
        
        # Copy trajectory info for backward mask computation
        temp_batch.actions = trajectory_batch.actions.clone()
        temp_batch.qubit_last_use_step = trajectory_batch.qubit_last_use_step.clone()
        
        for step in range(max_length):
            # Update active mask for this step
            step_active = step < trajectory_batch.lengths
            temp_batch.active = step_active
            batched_tableau.active = step_active
            
            states_tensor, indices = batched_tableau.to_flat_tensors_active_only()
            if states_tensor.shape[0] == 0:
                break
            
            # Compute logits
            if compute_gradients:
                logits_f = self.pf_model(states_tensor)
            else:
                with torch.no_grad():
                    logits_f = self.pf_model(states_tensor)
            
            
            # Compute masks
            masks = self.compute_action_masks_gpu(temp_batch, max_depth)
            
            # Convert indices to tensor for batch operations
            if isinstance(indices, list):
                indices_tensor = torch.tensor(indices, dtype=torch.long, device=self.device)
            elif isinstance(indices, torch.Tensor):
                indices_tensor = indices.to(self.device)
            else:
                # indices might be a list of tensors
                indices_tensor = torch.stack([torch.as_tensor(idx) for idx in indices]).to(self.device)

            # Get actions for this step and filter out trajectories that have ended
            step_actions = trajectory_batch.actions[indices_tensor[:, 0], indices_tensor[:, 1], step]
            valid_length_mask = step < trajectory_batch.lengths[indices_tensor[:, 0], indices_tensor[:, 1]]
            if not valid_length_mask.any():
                continue
            indices_tensor = indices_tensor[valid_length_mask]
            step_actions = step_actions[valid_length_mask]

            # Vectorized forward flows with action validity checks
            traj_masks = masks[indices_tensor[:, 0], indices_tensor[:, 1]]
            masked_logits_f = logits_f[valid_length_mask].masked_fill(~traj_masks, float('-inf'))

            valid_any = torch.isfinite(masked_logits_f).any(dim=1)
            log_probs_f = torch.zeros_like(masked_logits_f)
            if valid_any.any():
                log_probs_f[valid_any] = torch.nn.functional.log_softmax(
                    masked_logits_f[valid_any], dim=-1
                )

            action_valid = traj_masks.gather(1, step_actions.unsqueeze(1)).squeeze(1)
            selected_f = log_probs_f.gather(1, step_actions.unsqueeze(1)).squeeze(1)
            # Use float() to avoid dtype issues when multiplying boolean mask
            selected_f = selected_f * action_valid.float()
            forward_flows[indices_tensor[:, 0], indices_tensor[:, 1]] += selected_f

            # Determine which trajectories will contribute to backward flows
            lengths_selected = trajectory_batch.lengths[indices_tensor[:, 0], indices_tensor[:, 1]]
            non_terminal = self.action_gate_types[step_actions] != self.gate_name_to_idx["terminal"]
            valid_backward = non_terminal & (step < lengths_selected - 1)
            if valid_backward.any():
                b_indices = indices_tensor[valid_backward]
                b_actions = step_actions[valid_backward]
            else:
                b_indices = None
                b_actions = None
            
            # Apply actions for next step
            if step < max_length - 1:
                # Prepare action tensor for applying current step actions
                actions = torch.full((batch_size, n_measurements), self.terminal_index,
                                   dtype=torch.long, device=self.device)

                active_mask = (step < trajectory_batch.lengths - 1)
                if active_mask.any():
                    next_actions = trajectory_batch.actions[:, :, step]
                    terminal_mask = self.action_gate_types[next_actions] == self.gate_name_to_idx["terminal"]

                    actions[active_mask & ~terminal_mask] = next_actions[active_mask & ~terminal_mask]

                    temp_batch.active = active_mask & ~terminal_mask
                    batched_tableau.active = temp_batch.active.clone()

                    if temp_batch.active.any():
                        self.apply_actions_to_batch_optimized(
                            batched_tableau, actions, temp_batch, step=step
                        )

                # After applying actions, compute backward probabilities
                if b_indices is not None and b_indices.shape[0] > 0:
                    states_next, indices_next = batched_tableau.to_flat_tensors_active_only()
                    with torch.no_grad():
                        logits_b = self.pb_model(states_next)
                        if logits_b.dim() == 1:
                            logits_b = logits_b.unsqueeze(0).expand(states_next.shape[0], -1)

                    # Use exposed-based backward masks instead of forward masks
                    masks_next = self.compute_backward_masks_gpu_vectorized(
                        temp_batch, current_step=step + 1
                    )

                    if isinstance(indices_next, list):
                        indices_next_tensor = torch.tensor(indices_next, dtype=torch.long, device=self.device)
                    elif isinstance(indices_next, torch.Tensor):
                        indices_next_tensor = indices_next.to(self.device)
                    else:
                        indices_next_tensor = torch.stack([torch.as_tensor(idx) for idx in indices_next]).to(self.device)

                    mapping = { (int(indices_next_tensor[i,0].item()), int(indices_next_tensor[i,1].item())): i
                               for i in range(indices_next_tensor.shape[0]) }

                    mapped_list = []
                    valid_b_mask = torch.zeros(b_indices.shape[0], dtype=torch.bool, device=self.device)
                    for idx, b in enumerate(b_indices):
                        key = (int(b[0].item()), int(b[1].item()))
                        if key in mapping:
                            mapped_list.append(mapping[key])
                            valid_b_mask[idx] = True

                    if torch.any(valid_b_mask):
                        b_indices = b_indices[valid_b_mask]
                        b_actions = b_actions[valid_b_mask]
                        mapped = torch.tensor(mapped_list, dtype=torch.long, device=self.device)

                        b_masks = masks_next[b_indices[:,0], b_indices[:,1]].clone()
                        b_masks[:, self.terminal_index] = False
                        masked_logits_b = logits_b[mapped].masked_fill(~b_masks, float('-inf'))
                        valid_any_b = torch.isfinite(masked_logits_b).any(dim=1)
                        log_probs_b = torch.zeros_like(masked_logits_b)
                        if valid_any_b.any():
                            log_probs_b[valid_any_b] = torch.nn.functional.log_softmax(
                                masked_logits_b[valid_any_b], dim=-1
                            )
                        
                        # Handle NaN/inf in backward log probs
                        if not torch.isfinite(log_probs_b).all():
                            log_probs_b = torch.nan_to_num(log_probs_b, nan=0.0, posinf=0.0, neginf=-20.0)
                        
                        selected_b = log_probs_b.gather(1, b_actions.unsqueeze(1)).squeeze(1)
                        backward_flows[b_indices[:,0], b_indices[:,1]] += selected_b
        
        if self.debug:
            logging.debug("\nDEBUG compute_flows final:")
            logging.debug(f"  Forward flows shape: {forward_flows.shape}")
            logging.debug(f"  Forward flows (first 4): {forward_flows.flatten()[:4].detach().cpu().numpy()}")
            logging.debug(f"  Backward flows (first 4): {backward_flows.flatten()[:4].detach().cpu().numpy()}")
            logging.debug(f"  Non-zero forward flows: {(forward_flows != 0).sum().item()}/{forward_flows.numel()}")
            logging.debug(f"  Non-zero backward flows: {(backward_flows != 0).sum().item()}/{backward_flows.numel()}")
        
        return forward_flows, backward_flows
    
    def compute_loss(self, trajectory_batch: TrajectoryBatch, costs: torch.Tensor,
                    beta: float = 1.0, max_depth: Optional[int] = None,
                    **reward_kwargs) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute loss for a batch of trajectories.
        
        This is the unified loss computation method used by all training modes:
        on-policy, off-policy, and replay. It ensures consistent:
        - Reward computation using the same reward function
        - Flow computation (forward and backward)
        - Loss calculation using the specified objective
        
        Args:
            trajectory_batch: Batch of trajectories with actions and states
            costs: Batch-level costs from CliffordMap
            beta: Temperature parameter for reward function
            **reward_kwargs: Additional arguments for reward function
            
        Returns:
            loss: Computed loss tensor
            metrics: Dictionary of metrics for logging
        """
        assert costs.shape[0] == trajectory_batch.batch_size, \
            f"Costs shape {costs.shape} doesn't match batch size {trajectory_batch.batch_size}"
        
        # Ensure costs are on the correct device
        costs = costs.to(self.device)
        
        if max_depth is None:
            max_depth = getattr(self, "last_max_depth", None)

        forward_flows, backward_flows = self.compute_flows(trajectory_batch, max_depth=max_depth, compute_gradients=True)
        
        rewards = self.reward_fn(costs, beta=beta, **reward_kwargs)
        
        valid_mask = trajectory_batch.lengths > 0
        valid_counts = valid_mask.sum(dim=1)
        
        # Vectorized averaging across valid trajectories
        valid_counts_clamped = valid_counts.clamp(min=1)
        mask_float = valid_mask.float()
        forward_flows_sum = (forward_flows * mask_float).sum(dim=1)
        backward_flows_sum = (backward_flows * mask_float).sum(dim=1)
        forward_flows_avg = forward_flows_sum / valid_counts_clamped
        backward_flows_avg = backward_flows_sum / valid_counts_clamped
        
        batch_valid = valid_counts > 0
        
        if batch_valid.any():
            forward_flows_avg = forward_flows_avg[batch_valid]
            backward_flows_avg = backward_flows_avg[batch_valid]
            rewards_filtered = rewards[batch_valid]
            
            if self.debug:
                logging.debug("\nDEBUG compute_loss:")
                logging.debug(f"  Valid batches: {batch_valid.sum().item()}/{trajectory_batch.batch_size}")
                logging.debug(f"  Forward flows avg: {forward_flows_avg.detach().cpu().numpy()}")
                logging.debug(f"  Backward flows avg: {backward_flows_avg.detach().cpu().numpy()}")
                logging.debug(f"  Rewards: {rewards_filtered.detach().cpu().numpy()}")
        else:
            return torch.tensor(0.0, device=self.device, requires_grad=True), {
                'loss': 0.0,
                'reward': 0.0,
                'cost': costs.mean().item() if costs.numel() > 0 else 0.0,
                'logZ': self.pf_model.logZ.item(),
                'avg_trajectories_per_batch': 0.0
            }
        
        loss, objective_metrics = self.objective.compute_loss(
            forward_flows=forward_flows_avg,
            backward_flows=backward_flows_avg,
            rewards=rewards_filtered,
            logZ=self.pf_model.logZ
        )
        
        # Use .item() only once at the end for metrics
        metrics = {
            'loss': loss.item(),
            'reward': rewards.mean().item(),
            'cost': costs.mean().item(),
            'logZ': self.pf_model.logZ.item(),
            'avg_trajectories_per_batch': valid_counts.float().mean().item(),
            **objective_metrics
        }
        
        return loss, metrics
    
    def update_step(self, accumulated_loss: torch.Tensor) -> float:
        """Perform a single gradient update step."""
        if torch.isnan(accumulated_loss) or torch.isinf(accumulated_loss):
            logging.warning("NaN or Inf detected in accumulated loss, skipping update")
            self.optimizer.zero_grad()
            return 0.0
        
        accumulated_loss.backward()
        
        # Check for NaN gradients without transferring to CPU
        has_nan_grad = False
        for p in self.pf_model.parameters():
            if p.grad is not None and torch.isnan(p.grad).any():
                has_nan_grad = True
                break
        
        if has_nan_grad:
            logging.warning("NaN detected in gradients, skipping update")
            self.optimizer.zero_grad()
            return accumulated_loss.item()
        
        params_to_clip = [p for n, p in self.pf_model.named_parameters() if n != 'logZ' and p.grad is not None]
        torch.nn.utils.clip_grad_norm_(params_to_clip, self.grad_clip_value)
        
        self.optimizer.step()
        self.optimizer.zero_grad()
        
        return accumulated_loss.item()
    
    def _update_top_trajectories(self, trajectory_batch: TrajectoryBatch, costs: torch.Tensor):
        """Update top K batches buffer with minimal CPU transfer - based on lowest costs."""
        batch_size = trajectory_batch.batch_size
        
        # Process each batch element
        for b_idx in range(batch_size):
            batch_cost = costs[b_idx].item()  # Single CPU transfer for cost
            
            # Check if this batch has valid trajectories
            valid_mask = trajectory_batch.lengths[b_idx] > 0
            if not valid_mask.any():
                continue
            
            # Extract batch data (keep on GPU)
            batch_actions = trajectory_batch.actions[b_idx].clone()
            batch_lengths = trajectory_batch.lengths[b_idx].clone()
            
            # Update top K list (keeping lowest costs)
            if len(self.top_trajectories_costs) < self.K:
                self.top_trajectories_actions.append(batch_actions)
                self.top_trajectories_lengths.append(batch_lengths)
                self.top_trajectories_costs.append(batch_cost)
            else:
                # Find maximum cost (worst trajectory)
                max_cost = max(self.top_trajectories_costs)
                if batch_cost < max_cost:  # If new cost is lower, replace the worst
                    max_idx = self.top_trajectories_costs.index(max_cost)
                    self.top_trajectories_actions[max_idx] = batch_actions
                    self.top_trajectories_lengths[max_idx] = batch_lengths
                    self.top_trajectories_costs[max_idx] = batch_cost

    def _replay_trajectories_optimized(self, batch_data_list: Optional[List[Dict]], 
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
        
        # Create tableau on the same device as the model
        batched_tableau = CliffordMap(
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
        
        # Fill trajectories from stored GPU tensors
        for b_idx in range(batch_size):
            stored_actions = self.top_trajectories_actions[b_idx]
            stored_lengths = self.top_trajectories_lengths[b_idx]
            
            # Copy to trajectory batch
            actual_max_length = min(max_length, stored_actions.shape[1])
            trajectory_batch.actions[b_idx, :, :actual_max_length] = stored_actions[:, :actual_max_length]
            trajectory_batch.lengths[b_idx] = torch.minimum(stored_lengths, 
                                                           torch.tensor(max_length, device=self.device))
            
            # Update active status
            trajectory_batch.active[b_idx] = stored_lengths > 0
        
        # Apply all actions to reconstruct final states and compute depths
        with torch.no_grad():
            for step in range(max_length):
                # Check if any trajectory is active at this step
                step_active = step < trajectory_batch.lengths
                if not step_active.any():
                    break
                
                # Get states for active trajectories
                states_tensor, indices = batched_tableau.to_flat_tensors_active_only()
                if states_tensor.shape[0] == 0:
                    break
                
                # Get actions for this step
                actions = trajectory_batch.actions[:, :, step]
                
                # Apply actions using the same optimized function with depth tracking
                terminated = self.apply_actions_to_batch_optimized(
                    batched_tableau, actions, trajectory_batch, step=step
                )
                
                # Update active status
                trajectory_batch.active &= ~terminated
        
        return trajectory_batch
    
    def save_checkpoint(self, path: str, update: int, metrics: Dict):
        """Save model checkpoint including adaptive tracker state."""
        # Convert GPU tensors to CPU for saving
        top_trajectories_cpu = []
        if hasattr(self, 'top_trajectories_actions'):
            for i in range(len(self.top_trajectories_actions)):
                top_trajectories_cpu.append({
                    'actions': self.top_trajectories_actions[i].cpu(),
                    'lengths': self.top_trajectories_lengths[i].cpu(),
                    'cost': self.top_trajectories_costs[i],
                    'n_measurements': self.top_trajectories_actions[i].shape[0]
                })
        
        checkpoint = {
            'pf_model_state_dict': self.pf_model.state_dict(),
            'pb_model_state_dict': self.pb_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'update': update,
            'top_trajectories': top_trajectories_cpu,
            'metrics': metrics,
            'model_type': self.model_type,
            'n_qubits': self.n_qubits,
            'num_actions': self.num_actions,
            'objective_type': self.objective_type,
            'checkpoint_version': 'gpu_optimized_with_depth',
            'buffer_strategy': self.buffer_strategy,
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
        
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: str) -> Tuple[int, Dict]:
        """Load model checkpoint and restore adaptive tracker state."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        
        if checkpoint.get('n_qubits') != self.n_qubits:
            logging.info(f"Qubit mismatch: checkpoint has {checkpoint.get('n_qubits')}, "
                  f"model has {self.n_qubits}")
            return 0, {}
        
        try:
            self.pf_model.load_state_dict(checkpoint['pf_model_state_dict'])
            self.pb_model.load_state_dict(checkpoint['pb_model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # Load top trajectories back to GPU
            self.top_trajectories_actions = []
            self.top_trajectories_lengths = []
            self.top_trajectories_costs = []
            
            if checkpoint.get('checkpoint_version') in ['gpu_optimized', 'gpu_optimized_with_depth']:
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


class EfficientGFNTrainer:
    """High-level trainer with minimal CPU-GPU transfer and configurable cost functions."""
    
    def __init__(self, config: Dict, 
                 reward_fn: Optional[Callable] = None,
                 device: Optional[torch.device] = None,
                 device_preference: Optional[str] = None):
        self.config = config
        self.device = device or get_device(device_preference)
        
        self.model_config = config["model"]
        self.training_config = config["training"]
        self.quantum_config = config["quantum"]
        
        self.pauli_str_list = self.quantum_config["pauli_str_list"]
        self.w_list = self.quantum_config["w_list"]
        self.n_qubits = len(self.pauli_str_list[0])
        
        self.n_measurements = self.training_config["n_measurements"]
        self.update_freq = self.training_config["update_freq"]
        
        # Initialize cost computer
        cost_config = self.training_config.get("cost", {})
        cost_type = cost_config.get("type", "exponential")
        
        # Extract cost_kwargs - these are passed to compute_batch_cost
        self.cost_kwargs = {k: v for k, v in cost_config.items() if k != "type" and k != "custom_costs"}
        
        # Check for legacy epsilon parameter
        if "epsilon" in self.training_config and "epsilon" not in self.cost_kwargs:
            self.cost_kwargs["epsilon"] = self.training_config["epsilon"]
        
        self.cost_computer = CostComputer(
            cost_type=cost_type,
            n_measurements=self.n_measurements,
            device=self.device
        )
        
        # Add custom cost functions if specified
        if "custom_costs" in cost_config:
            for name, custom_cost_config in cost_config["custom_costs"].items():
                if name == "threshold":
                    threshold = custom_cost_config.get("threshold", 0.5)
                    self.cost_computer.add_custom_cost_function(
                        name, ThresholdCost(threshold)
                    )
        
        self.gfn = GFlowNet(
            n_qubits=self.n_qubits,
            hidden_dim=self.model_config["hidden_dim"],
            num_hidden_layers=self.model_config["num_hidden_layers"],
            lr=self.model_config["lr"],
            weight_decay=self.model_config["weight_decay"],
            reward_fn=reward_fn,
            device=self.device,
            model_type=self.model_config.get("model_type", "clifford_mlp"),
            model_kwargs=self.model_config.get("model_kwargs", {}),
            objective_type=self.model_config.get("objective_type", "tb"),
            objective_kwargs=self.model_config.get("objective_kwargs", {}),
            debug=self.model_config.get("debug", False),
            device_preference=device_preference,
            K=self.training_config["K"],
            buffer_strategy=self.model_config.get("buffer_strategy", "conservative"),
            adaptive_warmup=self.model_config.get("adaptive_warmup", 100)
        )
        
        self.beta = self.training_config["beta"]
        self.max_depth = self.training_config.get("max_depth", self.training_config.get("max_layer", 6))
        self.K = self.training_config["K"]
        
        self.reward_kwargs = self.training_config.get("reward_kwargs", {})
        
        self.metrics_history = defaultdict(list)
        self.timing_history = defaultdict(list)

    def compute_costs_with_probabilities(self, batched_tableau: CliffordMap, 
                                       silence: bool = True, **override_kwargs) -> torch.Tensor:
        """Compute costs using the CostComputer with probabilities from the tableau.
        
        Args:
            batched_tableau: The batched Clifford tableau
            silence: Whether to suppress debug output
            **override_kwargs: Additional kwargs to override self.cost_kwargs
        """
        # Get probabilities for all Pauli strings
        probs = batched_tableau.prob_P_multi(self.pauli_str_list)
        
        # Merge kwargs with overrides
        kwargs = {**self.cost_kwargs, **override_kwargs}
        
        # Use CostComputer to compute batch costs
        costs = self.cost_computer.compute_batch_cost(probs, self.w_list, **kwargs)
        
        if not silence:
            logging.info(f"Probabilities shape: {probs.shape}, dtype: {probs.dtype}")
            logging.info(probs.sum(dim=1))
            logging.info(f"Computed costs shape: {costs.shape}, dtype: {costs.dtype}")
            logging.info(f"Costs (first 4): {costs.flatten()[:4].detach().cpu().numpy()}")

        return costs
    
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
                          e.g., {0: "exponential", 1000: "quadratic", 2000: "linear"}
        """
        start_update = 0
        
        checkpoint_path = os.path.join(self.model_config["model_dir"], "checkpoint_latest.pth")
        if os.path.exists(checkpoint_path):
            try:
                start_update, metrics = self.gfn.load_checkpoint(checkpoint_path)
                if start_update > 0:
                    self.metrics_history = defaultdict(list, metrics)
                    logging.info(f"Resumed from update {start_update}")
            except Exception as e:
                logging.info(f"Could not load checkpoint: {e}")
                start_update = 0
        
        # Pre-allocate timing tensors on GPU to avoid repeated transfers
        if self.device.type != 'cpu':
            torch.cuda.synchronize() if self.device.type == 'cuda' else None
        
        for update in range(start_update, num_updates):
            # Check if we need to change cost function
            if cost_schedule and update in cost_schedule:
                new_cost_type = cost_schedule[update]
                logging.info(f"\nSwitching to {new_cost_type} cost function at update {update}")
                self.cost_computer.set_cost_type(new_cost_type, self.n_measurements)
            
            update_start = time.time()
            logging.info(f"\n=== Update {update+1}/{num_updates} ===")
            logging.info(f"Cost function: {self.cost_computer.cost_type}")
            if self.cost_kwargs:
                logging.info(f"Cost kwargs: {self.cost_kwargs}")
            
            sample_start = time.time()
            
            # Sample on-policy trajectories with depth limit
            trajectory_batch = self.gfn.sample_trajectories(
                batch_size=self.update_freq,
                n_measurements=self.n_measurements,
                max_depth=self.max_depth,
                mode=SamplingMode.ON_POLICY
            )
            
            # Compute costs using CostComputer with kwargs
            batched_tableau = trajectory_batch.batched_tableau
            costs = self.compute_costs_with_probabilities(batched_tableau)
            
            # Single CPU transfer for logging
            costs_cpu = costs.cpu().numpy()
            logging.info(f"  Batch costs (one per batch element): {costs_cpu}")
            logging.info(f"  Mean batch cost: {costs_cpu.mean():.4f}, "
                  f"Min: {costs_cpu.min():.4f}, Max: {costs_cpu.max():.4f}")
            
            # Compute loss using the standard method
            loss, metrics = self.gfn.compute_loss(
                trajectory_batch, costs, self.beta, max_depth=self.max_depth, **self.reward_kwargs
            )
            
            sample_time = time.time() - sample_start
            
            # Update model using the standard method
            loss_value = self.gfn.update_step(loss)
            
            # Update top trajectories for replay
            self.gfn._update_top_trajectories(trajectory_batch, costs)
            metrics['loss'] = loss_value
            metrics['cost_type'] = self.cost_computer.cost_type
            
            # update time after sampling and loss computation
            update_time = time.time() - update_start
            
            # Replay training
            if replay_every and (update + 1) % replay_every == 0 and self.gfn.top_trajectories_actions:
                replay_start = time.time()
                
                self.gfn.optimizer.zero_grad()
                
                # Sample replay trajectories using the same method as on/off-policy
                replay_batch = self.gfn.sample_trajectories(
                    batch_size=min(len(self.gfn.top_trajectories_actions), self.update_freq),
                    n_measurements=self.n_measurements,
                    max_depth=self.max_depth,
                    mode=SamplingMode.REPLAY,
                    batch_data_list=None  # Will use stored top trajectories
                )
                
                # Compute costs using CostComputer
                replay_tableau = replay_batch.batched_tableau
                logging.info(f" At Step {update + 1}, Replay batch size: {replay_batch.batch_size}, with probabilities:")
                replay_costs = self.compute_costs_with_probabilities(replay_tableau, silence=False)
                
                # Compute loss using the exact same method as on-policy
                replay_loss, replay_metrics = self.gfn.compute_loss(
                    replay_batch, replay_costs, self.beta, max_depth=self.max_depth, **self.reward_kwargs
                )
                
                # Update step using the same method
                replay_loss_value = self.gfn.update_step(replay_loss)
                replay_metrics['loss'] = replay_loss_value
                
                # Store metrics with replay prefix
                for k, v in replay_metrics.items():
                    metrics[f'replay_{k}'] = v
                
                replay_time = time.time() - replay_start
                logging.info(f"  Replay on {len(self.gfn.top_trajectories_actions)} batches: "
                      f"loss={replay_metrics['loss']:.6f}, batch_reward={replay_metrics['reward']:.4f}")
            
            # Off-policy training
            if offpolicy_every and (update + 1) % offpolicy_every == 0:
                offpolicy_start = time.time()
                
                self.gfn.optimizer.zero_grad()
                
                # Sample off-policy trajectories using the same method structure
                offpolicy_batch = self.gfn.sample_trajectories(
                    batch_size=self.update_freq,
                    n_measurements=self.n_measurements,
                    max_depth=self.max_depth,
                    mode=SamplingMode.OFF_POLICY
                )
                
                # Compute costs using CostComputer
                offpolicy_tableau = offpolicy_batch.batched_tableau
                offpolicy_costs = self.compute_costs_with_probabilities(offpolicy_tableau)
                
                # Compute loss using the exact same method as on-policy
                offpolicy_loss, offpolicy_metrics = self.gfn.compute_loss(
                    offpolicy_batch, offpolicy_costs, self.beta, max_depth=self.max_depth, **self.reward_kwargs
                )
                
                # Update step using the same method
                offpolicy_loss_value = self.gfn.update_step(offpolicy_loss)
                offpolicy_metrics['loss'] = offpolicy_loss_value
                
                # Store metrics with offpolicy prefix
                for k, v in offpolicy_metrics.items():
                    metrics[f'offpolicy_{k}'] = v
                
                offpolicy_time = time.time() - offpolicy_start
                logging.info(f"  Off-policy: loss={offpolicy_metrics['loss']:.6f}, "
                      f"batch_reward={offpolicy_metrics['reward']:.4f}")
            
            # Update metrics
            for k, v in metrics.items():
                self.metrics_history[k].append(v)

            total_trajectories = self.update_freq * self.n_measurements
            
            if profile:
                self.timing_history['total'].append(update_time)
                self.timing_history['sample'].append(sample_time)
                self.timing_history['throughput'].append(total_trajectories / update_time)
            
            logging.info(f"\nSummary - Loss: {metrics['loss']:.6f}, "
                  f"Batch Reward: {metrics['reward']:.4f}, "
                  f"Batch Cost: {metrics['cost']:.4f}, "
                  f"LogZ: {metrics['logZ']:.3f}, "
                  f"Avg Trajs/Batch: {metrics.get('avg_trajectories_per_batch', 0):.1f}")
            
            if profile:
                logging.info(f"Time: {update_time:.2f}s, "
                      f"Throughput: {total_trajectories/update_time:.1f} traj/s "
                      f"({total_trajectories} total trajectories)")
            
            if (update + 1) % checkpoint_every == 0:
                os.makedirs(self.model_config["model_dir"], exist_ok=True)
                self.gfn.save_checkpoint(checkpoint_path, update + 1, dict(self.metrics_history))
                logging.info(f"Checkpoint saved at update {update + 1}")
                self.plot_metrics(update + 1)
                
    def train_async(self, num_updates: int, **kwargs):
        """
        Asynchronous training with separate sampler and learner processes.
        
        Args:
            num_updates: Number of training updates
            **kwargs: Ignored (for compatibility with train method)
        """
        logging.info("\n" + "="*60)
        logging.info("Starting ASYNCHRONOUS training mode")
        logging.info(f"Samplers: {self.config['training'].get('num_samplers', 2)}")
        logging.info(f"Pipeline depth: {self.config['training'].get('pipeline_depth', 4)}")
        logging.info(f"Broadcast every: {self.config['training'].get('broadcast_every', 10)} updates")
        logging.info("="*60 + "\n")
        
        # Import here to avoid issues when not using async mode
        from gfn_async import async_learner
        
        # Run async training
        metrics_history, timing_history = async_learner(self.config, num_updates)
        
        # Update trainer's history
        self.metrics_history = defaultdict(list, metrics_history)
        self.timing_history = defaultdict(list, timing_history)
        
        # Plot final metrics
        self.plot_metrics(num_updates)
    
    def plot_metrics(self, update: int):
        """Plot training metrics."""
        plots_dir = os.path.join(self.model_config["model_dir"], "figures")
        os.makedirs(plots_dir, exist_ok=True)
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        if 'loss' in self.metrics_history:
            axes[0].plot(self.metrics_history['loss'], label='On-policy')
            if 'replay_loss' in self.metrics_history:
                axes[0].plot(self.metrics_history['replay_loss'], 
                           label='Replay', alpha=0.7, linestyle='--')
            if 'offpolicy_loss' in self.metrics_history:
                axes[0].plot(self.metrics_history['offpolicy_loss'], 
                           label='Off-policy', alpha=0.7, linestyle=':')
            axes[0].set_title('Training Loss')
            axes[0].set_xlabel('Update')
            axes[0].set_ylabel('Loss')
            axes[0].set_yscale('log')
            axes[0].grid(True, alpha=0.3)
            axes[0].legend()
        
        if 'reward' in self.metrics_history:
            axes[1].plot(self.metrics_history['reward'], label='On-policy')
            if 'replay_reward' in self.metrics_history:
                axes[1].plot(self.metrics_history['replay_reward'], 
                           label='Replay', alpha=0.7, linestyle='--')
            if 'offpolicy_reward' in self.metrics_history:
                axes[1].plot(self.metrics_history['offpolicy_reward'], 
                           label='Off-policy', alpha=0.7, linestyle=':')
            axes[1].set_title('Average Batch Reward')
            axes[1].set_xlabel('Update')
            axes[1].set_ylabel('Batch Reward')
            axes[1].set_ylim(0,)
            axes[1].grid(True, alpha=0.3)
            axes[1].legend()
        
        if 'cost' in self.metrics_history:
            axes[2].plot(self.metrics_history['cost'])
            axes[2].set_title('Average Cost')
            axes[2].set_xlabel('Update')
            axes[2].set_ylabel('Cost')
            axes[2].set_yscale('log')
            axes[2].grid(True, alpha=0.3)
        
        if 'throughput' in self.timing_history:
            axes[3].plot(self.timing_history['throughput'])
            axes[3].set_title('Training Throughput')
            axes[3].set_xlabel('Update')
            axes[3].set_ylabel('Trajectories/Second')
            axes[3].grid(True, alpha=0.3)
        elif 'logZ' in self.metrics_history:
            axes[3].plot(self.metrics_history['logZ'], color='green')
            axes[3].set_xlabel('Update')
            axes[3].set_ylabel('logZ')
            axes[3].set_title('logZ Evolution')
            axes[3].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f'metrics_update.png'), 
                    dpi=300, bbox_inches='tight')
        plt.close()


if __name__ == "__main__":
    config = {
        "model": {
            "model_type": "clifford_mlp",
            "hidden_dim": 1024,
            "num_hidden_layers": 3,
            "lr": 5e-3,
            "weight_decay": 1e-5,
            "model_dir": "../results/",
            "model_kwargs": {},
            "objective_type": "tb",
            "objective_kwargs": {"loss_type": "squared"},
            "debug": False,
            "buffer_strategy": "conservative",  # Change to "adaptive" for adaptive strategy
            "adaptive_warmup": 100
        },
        "training": {
            "beta": 2e3,
            "n_measurements": 1000,  # Number of trajectories per batch element
            "update_freq": 10,     # Number of batch elements (this is batch_size)
            "max_depth": 3,        # Now using depth instead of max_layer
            "K": 5,  # Number of top batches to keep for replay
            "reward_kwargs": {"alpha" : 5e-2},
            "cost": {
                "type": "linear_bias",
                "epsilon": 0.9  # This is now part of cost kwargs
            }
        },
        "quantum": {
            "pauli_str_list": ["ZIII", "IZII", "IIZI", "IIIZ", "ZZII", "ZIZI", "ZIIZ", 
                             "IZZI", "IZIZ", "IIZZ", "YYYY", "XXXX", "YYXX", "XXYY"],
            "w_list": [0.172, -0.226, 0.172, -0.226, 0.121, 0.169, 0.166,
                      0.166, 0.175, 0.121, 0.045, 0.045, 0.045, 0.045]
        }
    }
    
    # Update model_dir based on config
    cost_kwargs_str = "_".join([f"{k}_{v}" for k, v in config['training']['cost'].items() if k != 'type'])
    config["model"]["model_dir"] = os.path.join(
        config["model"]["model_dir"], 
        f"beta_{int(config['training']['beta'])}_"
        f"n_measurements_{config['training']['n_measurements']}_"
        f"depth_{config['training']['max_depth']}_"
        f"{config['model']['buffer_strategy']}_"
        f"{config['training']['cost']['type']}"
        f"_{cost_kwargs_str}" if cost_kwargs_str else ""
    )
    
    logging.info("directory: %s", config["model"]["model_dir"])
    
    # You can specify device preference: "cuda", "mps", "cpu", or None for auto-detect
    trainer = EfficientGFNTrainer(config, device_preference="cpu")
    
    trainer.train(
        num_updates=100000,
        replay_every=25,
        offpolicy_every=20,
        checkpoint_every=50,
    )
