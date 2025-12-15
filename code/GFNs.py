# -*- coding: utf-8 -*-
import os
import json
import heapq
import torch
import logging

if torch.cuda.is_available():
    torch.set_float32_matmul_precision('high')
from torch.distributions import Categorical
import numpy as np
from collections import defaultdict
import time
from enum import Enum
from typing import List, Tuple, Dict, Optional, Union, Callable, Any
from contextlib import contextmanager
import matplotlib.pyplot as plt
import torch.nn.functional as F

try:
    from torch.profiler import record_function as torch_record_function
except ImportError:
    def torch_record_function(_name):
        @contextmanager
        def _noop():
            yield
        return _noop()

record_function = torch_record_function

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

try:
    from .clifford_map import CliffordMap
    from .models import DiscreteUniform, CliffordMLP, QuantumAwareMLP, AttentionMLP, create_clifford_model, CliffordTableauProcessor
    from .gfn_objectives import GFlowNetObjective, create_gfn_objective
    from .cost_computer import CostComputer, CostFunction, ThresholdCost
    from .quantum_action_mapping import build_action_mapping
    from .masking_engine import MaskingEngine
except ImportError:
    from clifford_map import CliffordMap
    from models import DiscreteUniform, CliffordMLP, QuantumAwareMLP, AttentionMLP, create_clifford_model, CliffordTableauProcessor
    from gfn_objectives import GFlowNetObjective, create_gfn_objective
    from cost_computer import CostComputer, CostFunction, ThresholdCost
    from quantum_action_mapping import build_action_mapping
    from masking_engine import MaskingEngine


def convert_metrics_history_to_cpu(metrics_history: Dict[str, List[Any]]) -> Dict[str, List[float]]:
    """Convert metrics history (may contain lists of GPU tensors) to CPU."""
    converted_history = {}
    for metric_name, value_list in metrics_history.items():
        converted_list = []
        for v in value_list:
            if torch.is_tensor(v):
                converted_list.append(v.item() if v.numel() == 1 else v.cpu().numpy())
            else:
                converted_list.append(v)
        converted_history[metric_name] = converted_list
    return converted_history


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
    else:
        return torch.device("cpu")


def default_reward_fn(costs: torch.Tensor, beta: float = 1.0, alpha: float = 5e-3 , **kwargs) -> torch.Tensor:
    """Default reward function: linear transformation of costs."""
    return beta * (alpha - costs)

def log_reward_fn(costs: torch.Tensor, beta: float = 1.0, alpha: float = 1.0, **kwargs) -> torch.Tensor:
    """Logarithmic reward function for stronger differentiation."""
    epsilon = 1e-8
    return -beta * torch.log(alpha * costs + epsilon)


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
        
        self.gates_per_depth = defaultdict(list)
        self.buffer_utilization = []
        self.depth_distribution = defaultdict(int)
        
        self.max_gates_seen = torch.tensor(0, device=device)
        self.total_trajectories = 0
        self.update_count = 0
        
        self.gate_counts = []
        
    def update_statistics(self, trajectory_batch):
        """Update statistics from a batch of trajectories."""
        valid_mask = trajectory_batch.lengths > 0
        valid_depths = trajectory_batch.circuit_depths[valid_mask]
        valid_gates = trajectory_batch.lengths[valid_mask]
        
        if valid_gates.numel() > 0:
            depths_cpu = valid_depths.cpu().tolist()
            gates_cpu = valid_gates.cpu().tolist()
            
            for depth, gates in zip(depths_cpu, gates_cpu):
                self.gates_per_depth[depth].append(gates)
                self.gate_counts.append(gates)
                self.depth_distribution[depth] += 1
            
            self.total_trajectories += len(gates_cpu)
            self.max_gates_seen = torch.max(self.max_gates_seen, valid_gates.max())
        
        max_used = trajectory_batch.lengths.max()
        utilization = max_used.float() / trajectory_batch.max_length
        self.buffer_utilization.append(utilization.item())
        
        self.update_count += 1
        
    def get_recommended_buffer_size(self, max_depth: int, percentile: float = 95.0) -> int:
        """Get recommended buffer size based on statistics."""
        if self.update_count < self.warmup_updates:
            return self.initial_buffer_size
        
        if max_depth in self.gates_per_depth and len(self.gates_per_depth[max_depth]) >= 20:
            gates_at_depth = self.gates_per_depth[max_depth]
            gates_tensor = torch.tensor(gates_at_depth, device=self.device, dtype=torch.float32)
            recommended = torch.quantile(gates_tensor, percentile / 100.0).item()
        elif len(self.gate_counts) >= 100:
            all_gates = torch.tensor(self.gate_counts, device=self.device, dtype=torch.float32)
            base_percentile = torch.quantile(all_gates, percentile / 100.0).item()
            avg_depth = sum(d * count for d, count in self.depth_distribution.items()) / self.total_trajectories
            depth_ratio = max_depth / max(avg_depth, 1.0)
            recommended = base_percentile * depth_ratio
        else:
            recommended = self.initial_buffer_size * 0.8
        
        recommended = int(recommended * 1.1)
        min_size = int(self.initial_buffer_size * 0.5)
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
    """Container for batch trajectory data with circuit depth tracking and state caching."""
    
    def __init__(self, batch_size: int, n_measurements: int, max_length: int, 
                n_qubits: int, device: torch.device):
        self.batch_size = batch_size
        self.n_measurements = n_measurements
        self.max_length = max_length
        self.n_qubits = n_qubits
        self.device = device
        
        self.actions = torch.zeros((batch_size, n_measurements, max_length), 
                                    dtype=torch.long, device=device)
        self.lengths = torch.zeros((batch_size, n_measurements), 
                                    dtype=torch.long, device=device)
        self.active = torch.ones((batch_size, n_measurements), 
                                dtype=torch.bool, device=device)
        self.masks = torch.ones((batch_size, n_measurements, max_length), 
                                dtype=torch.bool, device=device)
        
        self.circuit_depths = torch.zeros((batch_size, n_measurements), 
                                         dtype=torch.long, device=device)
        self.current_layer_qubits = torch.zeros((batch_size, n_measurements, n_qubits), 
                                               dtype=torch.bool, device=device)
        self.qubit_last_layer = torch.zeros((batch_size, n_measurements, n_qubits), 
                                           dtype=torch.long, device=device) - 1
        
        self.last_single_qubit_gates = torch.zeros((batch_size, n_measurements, n_qubits), 
                                                    dtype=torch.long, device=device) - 1
        self.last_two_qubit_gates = torch.zeros((batch_size, n_measurements, n_qubits, n_qubits), 
                                                dtype=torch.long, device=device) - 1
        self.qubit_last_use_step = torch.full((batch_size, n_measurements, n_qubits), 
                                              -1, dtype=torch.long, device=device)
        self.action_qubits = torch.full((batch_size, n_measurements, max_length, 2), 
                                        -1, dtype=torch.long, device=device)
        
        self.batched_tableau = None
        
        self.cached_states = []
        self.cached_masks = []
        self.cached_backward_valid_counts = []
        self.cache_enabled = False
        
        # Double-buffered pre-allocation for flow computation
        self._flow_buffer_idx = 0
        self._forward_flows_buffers = [
            torch.zeros((batch_size, n_measurements), device=device),
            torch.zeros((batch_size, n_measurements), device=device)
        ]
        self._backward_flows_buffers = [
            torch.zeros((batch_size, n_measurements), device=device),
            torch.zeros((batch_size, n_measurements), device=device)
        ]
        self._terminated_buffer_idx = 0
        self._terminated_buffers = [
            torch.zeros((batch_size, n_measurements), dtype=torch.bool, device=device),
            torch.zeros((batch_size, n_measurements), dtype=torch.bool, device=device)
        ]
        
    def enable_caching(self):
        """Enable state caching for flow computation."""
        self.cache_enabled = True
        self.cached_states = []
        self.cached_masks = []
        self.cached_backward_valid_counts = []
        
    def cache_step_data(self, step: int, states_tensor: torch.Tensor, 
                       indices: Union[List[Tuple], torch.Tensor], masks: torch.Tensor,
                       backward_valid_counts: Optional[torch.Tensor] = None):
        """Cache state data for a specific step during sampling."""
        if not self.cache_enabled:
            return
        
        while len(self.cached_states) <= step:
            self.cached_states.append(None)
            self.cached_masks.append(None)
            self.cached_backward_valid_counts.append(None)
        
        if isinstance(indices, list):
            indices_tensor = torch.tensor(indices, dtype=torch.long, device=self.device)
        else:
            indices_tensor = indices.clone()
            
        self.cached_states[step] = (states_tensor.clone(), indices_tensor)
        self.cached_masks[step] = masks.clone()
        
        if backward_valid_counts is not None:
            self.cached_backward_valid_counts[step] = backward_valid_counts.clone()
        
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
                buffer_strategy: str = 'conservative',
                adaptive_warmup: int = 100):
        
        self.buffer_strategy = buffer_strategy
        self.adaptive_warmup = adaptive_warmup
        self.conservative_multiplier = 1.2
        self.adaptive_tracker = None
        
        self.n_qubits = n_qubits
        self.device = device or get_device(device_preference)
        self.model_type = model_type
        self.debug = debug
        
        if self.debug:
            logging.getLogger().setLevel(logging.DEBUG)
        
        logging.info(f"Using device: {self.device}")
        logging.info(f"Buffer strategy: {self.buffer_strategy}")
        
        self.reward_fn = reward_fn or default_reward_fn
        
        objective_kwargs = objective_kwargs or {}
        self.objective = create_gfn_objective(objective_type, **objective_kwargs)
        self.objective_type = objective_type
        
        self.action_mapping = self._build_action_mapping()
        self.num_actions = len(self.action_mapping)
        
        self._precompute_gate_info()
        self._precompute_gate_indices()
        
        self.state_dim = (2 * n_qubits) ** 2
        
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

        if hasattr(self.pf_model, 'logZ'):
            self.optimizer = torch.optim.Adam([
                {'params': self.pf_model.logZ, 'lr': 100*lr},
                {'params': [p for n, p in self.pf_model.named_parameters() if n != 'logZ'], 
                 'lr': lr , 'weight_decay': weight_decay}
            ])
        else:
            params = list(self.pf_model.parameters())
            if params:
                self.optimizer = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
            else:
                self.optimizer = None
        
        self.grad_clip_value = 1e3

        if torch.__version__ >= '2.0.0' and self.device.type in ['cuda', 'mps']:
            try:
                compile_options = {'mode': 'reduce-overhead', 'fullgraph': False}
                if self.device.type == 'cuda':
                    import torch._inductor.config as config
                    config.triton.cudagraph_trees = True
                    config.triton.cudagraph_skip_dynamic_graphs = True
                self.pf_model = torch.compile(self.pf_model, **compile_options)
                logging.info(f"Applied torch.compile optimization to forward model")
            except Exception as e:
                logging.warning(f"torch.compile not applied: {e}")

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
        """Pre-compute gate type information."""
        self.single_qubit_gates = {"H", "S", "HS", "SH", "HSH"}
        self.two_qubit_gates = {"CNOT"}
        
    def _precompute_gate_indices(self):
        """Pre-compute gate indices for GPU operations."""
        self.gate_name_to_idx = {
            "H": 0, "S": 1, "HS": 2, "SH": 3, "HSH": 4,
            "CNOT": 5, "terminal": 6
        }
        
        self.action_gate_types = torch.zeros(self.num_actions, dtype=torch.long)
        self.action_qubit1 = torch.zeros(self.num_actions, dtype=torch.long)
        self.action_qubit2 = torch.zeros(self.num_actions, dtype=torch.long) - 1
        
        for idx, action in self.action_mapping.items():
            gate_name = action[0]
            self.action_gate_types[idx] = self.gate_name_to_idx[gate_name]
            if gate_name != "terminal":
                self.action_qubit1[idx] = action[1]
                if len(action) > 2:
                    self.action_qubit2[idx] = action[2]
        
        self.action_gate_types = self.action_gate_types.to(self.device)
        self.action_qubit1 = self.action_qubit1.to(self.device)
        self.action_qubit2 = self.action_qubit2.to(self.device)
        
        self.single_qubit_mask = torch.zeros(self.num_actions, dtype=torch.bool)
        self.two_qubit_mask = torch.zeros(self.num_actions, dtype=torch.bool)
        
        for idx, action in self.action_mapping.items():
            if action[0] in self.single_qubit_gates:
                self.single_qubit_mask[idx] = True
            elif action[0] in self.two_qubit_gates:
                self.two_qubit_mask[idx] = True
        
        self.single_qubit_mask = self.single_qubit_mask.to(self.device)
        self.two_qubit_mask = self.two_qubit_mask.to(self.device)

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
        """Calculate conservative upper bound for buffer size."""
        single_qubit_bound = self.n_qubits * max_depth
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
    
    def apply_actions_to_batch(self, 
                                    batched_tableau: CliffordMap,
                                    actions: torch.Tensor,
                                    trajectory_batch: TrajectoryBatch,
                                    step: Optional[int] = None) -> torch.Tensor:
        """Fully vectorized action application with depth tracking."""
        batch_size, n_measurements = actions.shape
        
        if (hasattr(trajectory_batch, '_terminated_buffers') and 
            trajectory_batch._terminated_buffers[0].shape == (batch_size, n_measurements)):
            idx = trajectory_batch._terminated_buffer_idx
            trajectory_batch._terminated_buffer_idx = 1 - idx
            terminated = trajectory_batch._terminated_buffers[idx].zero_()
        else:
            terminated = torch.zeros((batch_size, n_measurements), dtype=torch.bool, device=actions.device)
        
        active_mask = trajectory_batch.active
        if not active_mask.any():
            return terminated
        
        batched_tableau.apply_actions_step(actions, self.action_mapping, active_mask)
        
        flat_active = active_mask.view(-1)
        flat_actions = actions.view(-1)
        active_indices = flat_active.nonzero(as_tuple=True)[0]
        
        if len(active_indices) == 0:
            return terminated
        
        active_actions = flat_actions[active_indices]
        batch_indices = active_indices // n_measurements
        meas_indices = active_indices % n_measurements
        
        is_terminal = active_actions == self.terminal_index
        if is_terminal.any():
            term_batch = batch_indices[is_terminal]
            term_meas = meas_indices[is_terminal]
            terminated[term_batch, term_meas] = True
            trajectory_batch.active[term_batch, term_meas] = False
            batched_tableau.active[term_batch, term_meas] = False
        
        non_terminal_mask = ~is_terminal
        if not non_terminal_mask.any():
            return terminated
        
        nt_actions = active_actions[non_terminal_mask]
        nt_batch = batch_indices[non_terminal_mask]
        nt_meas = meas_indices[non_terminal_mask]
        
        gate_types = self.action_gate_types[nt_actions]
        qubit1 = self.action_qubit1[nt_actions]
        qubit2 = self.action_qubit2[nt_actions]
        
        is_single = self.single_qubit_mask[nt_actions]
        is_two = self.two_qubit_mask[nt_actions]
        needs_new_layer = torch.zeros_like(non_terminal_mask)
        
        if is_single.any():
            single_idx = is_single.nonzero(as_tuple=True)[0]
            single_batch = nt_batch[single_idx]
            single_meas = nt_meas[single_idx]
            single_q = qubit1[single_idx]
            already_used = trajectory_batch.current_layer_qubits[single_batch, single_meas, single_q]
            needs_new_layer[single_idx[already_used]] = True
        
        if is_two.any():
            two_idx = is_two.nonzero(as_tuple=True)[0]
            two_batch = nt_batch[two_idx]
            two_meas = nt_meas[two_idx]
            two_q1 = qubit1[two_idx]
            two_q2 = qubit2[two_idx]
            q1_used = trajectory_batch.current_layer_qubits[two_batch, two_meas, two_q1]
            q2_used = trajectory_batch.current_layer_qubits[two_batch, two_meas, two_q2]
            already_used = q1_used | q2_used
            needs_new_layer[two_idx[already_used]] = True
        
        if needs_new_layer.any():
            new_layer_idx = needs_new_layer.nonzero(as_tuple=True)[0]
            new_layer_batch = nt_batch[new_layer_idx]
            new_layer_meas = nt_meas[new_layer_idx]
            trajectory_batch.circuit_depths[new_layer_batch, new_layer_meas] += 1
            trajectory_batch.current_layer_qubits[new_layer_batch, new_layer_meas] = False
        
        current_depths = trajectory_batch.circuit_depths[nt_batch, nt_meas]
        
        if step is not None:
            if is_single.any():
                single_idx = is_single.nonzero(as_tuple=True)[0]
                single_batch = nt_batch[single_idx]
                single_meas = nt_meas[single_idx]
                single_q = qubit1[single_idx]
                trajectory_batch.qubit_last_use_step[single_batch, single_meas, single_q] = step
                trajectory_batch.action_qubits[single_batch, single_meas, step, 0] = single_q
            
            if is_two.any():
                two_idx = is_two.nonzero(as_tuple=True)[0]
                two_batch = nt_batch[two_idx]
                two_meas = nt_meas[two_idx]
                two_q1 = qubit1[two_idx]
                two_q2 = qubit2[two_idx]
                trajectory_batch.qubit_last_use_step[two_batch, two_meas, two_q1] = step
                trajectory_batch.qubit_last_use_step[two_batch, two_meas, two_q2] = step
                trajectory_batch.action_qubits[two_batch, two_meas, step, 0] = two_q1
                trajectory_batch.action_qubits[two_batch, two_meas, step, 1] = two_q2
        
        if is_single.any():
            single_idx = is_single.nonzero(as_tuple=True)[0]
            single_batch = nt_batch[single_idx]
            single_meas = nt_meas[single_idx]
            single_q = qubit1[single_idx]
            single_gate_types = gate_types[single_idx]
            single_depths = current_depths[single_idx]
            trajectory_batch.last_single_qubit_gates[single_batch, single_meas, single_q] = single_gate_types
            trajectory_batch.current_layer_qubits[single_batch, single_meas, single_q] = True
            trajectory_batch.qubit_last_layer[single_batch, single_meas, single_q] = single_depths
        
        if is_two.any():
            two_idx = is_two.nonzero(as_tuple=True)[0]
            two_batch = nt_batch[two_idx]
            two_meas = nt_meas[two_idx]
            two_q1 = qubit1[two_idx]
            two_q2 = qubit2[two_idx]
            two_gate_types = gate_types[two_idx]
            two_depths = current_depths[two_idx]
            trajectory_batch.last_two_qubit_gates[two_batch, two_meas, two_q1, two_q2] = two_gate_types
            trajectory_batch.last_two_qubit_gates[two_batch, two_meas, two_q2, two_q1] = two_gate_types
            trajectory_batch.last_single_qubit_gates[two_batch, two_meas, two_q1] = -1
            trajectory_batch.last_single_qubit_gates[two_batch, two_meas, two_q2] = -1
            trajectory_batch.current_layer_qubits[two_batch, two_meas, two_q1] = True
            trajectory_batch.current_layer_qubits[two_batch, two_meas, two_q2] = True
            trajectory_batch.qubit_last_layer[two_batch, two_meas, two_q1] = two_depths
            trajectory_batch.qubit_last_layer[two_batch, two_meas, two_q2] = two_depths
        
        return terminated
    
    def compute_action_masks_gpu(self, trajectory_batch: TrajectoryBatch,
                                 max_depth: Optional[int] = None) -> torch.Tensor:
        """Compute valid action masks entirely on GPU using MaskingEngine.

        If ``max_depth`` is provided, actions that would require starting a new
        layer when the trajectory is already at ``max_depth`` are masked out.
        """
        return self.masking_engine.compute_action_masks_gpu(trajectory_batch, max_depth)
    
    def compute_backward_masks_gpu(self, trajectory_batch: TrajectoryBatch,
                                             current_step: int,
                                             forward_masks: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Fully vectorized computation of backward masks using MaskingEngine."""
        return self.masking_engine.compute_backward_masks_gpu(
            trajectory_batch, current_step, forward_masks)
    
    def compute_flows(self, trajectory_batch: TrajectoryBatch,
                     max_depth: Optional[int] = None,
                     compute_gradients: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute forward and backward flows with vectorized operations."""
        if trajectory_batch.cache_enabled and trajectory_batch.cached_states:
            return self.compute_flows_cached(
                trajectory_batch, 
                max_depth=max_depth, 
                compute_gradients=compute_gradients
            )
        
        batch_size = trajectory_batch.batch_size
        n_measurements = trajectory_batch.n_measurements
        max_length = trajectory_batch.max_length
        device = trajectory_batch.device
        
        if (hasattr(trajectory_batch, '_forward_flows_buffers') and 
            trajectory_batch._forward_flows_buffers[0].shape == (batch_size, n_measurements)):
            idx = trajectory_batch._flow_buffer_idx
            trajectory_batch._flow_buffer_idx = 1 - idx
            forward_flows = trajectory_batch._forward_flows_buffers[idx].zero_()
            backward_flows = trajectory_batch._backward_flows_buffers[idx].zero_()
        else:
            forward_flows = torch.zeros((batch_size, n_measurements), device=device)
            backward_flows = torch.zeros((batch_size, n_measurements), device=device)
        
        batched_tableau = CliffordMap(
            n_qubits=self.n_qubits,
            batch_size=batch_size,
            n_measurements=n_measurements,
            device=str(device)
        )
        
        temp_batch = TrajectoryBatch(
            batch_size=batch_size,
            n_measurements=n_measurements,
            max_length=max_length,
            n_qubits=self.n_qubits,
            device=device
        )
        
        temp_batch.lengths = trajectory_batch.lengths.clone()
        temp_batch.active = trajectory_batch.lengths > 0
        batched_tableau.active = temp_batch.active.clone()
        temp_batch.actions = trajectory_batch.actions.clone()
        temp_batch.qubit_last_use_step = trajectory_batch.qubit_last_use_step.clone()
        
        for step in range(max_length):
            step_active = step < trajectory_batch.lengths
            temp_batch.active = step_active
            batched_tableau.active = step_active
            
            states_tensor, indices = batched_tableau.to_flat_tensors_active_only()
            if states_tensor.shape[0] == 0:
                break
            
            if compute_gradients:
                logits_f = self.pf_model(states_tensor)
            else:
                with torch.no_grad():
                    logits_f = self.pf_model(states_tensor)
            
            masks = self.compute_action_masks_gpu(temp_batch, max_depth)
            
            if isinstance(indices, list):
                indices_tensor = torch.tensor(indices, dtype=torch.long, device=device)
            elif isinstance(indices, torch.Tensor):
                indices_tensor = indices.to(device)
            else:
                indices_tensor = torch.stack([torch.as_tensor(idx, device=device) for idx in indices])
            
            step_actions = trajectory_batch.actions[indices_tensor[:, 0], indices_tensor[:, 1], step]
            valid_length_mask = step < trajectory_batch.lengths[indices_tensor[:, 0], indices_tensor[:, 1]]
            if not valid_length_mask.any():
                continue
            indices_tensor = indices_tensor[valid_length_mask]
            step_actions = step_actions[valid_length_mask]

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
            selected_f = selected_f * action_valid.float()
            forward_flows[indices_tensor[:, 0], indices_tensor[:, 1]] += selected_f

            lengths_selected = trajectory_batch.lengths[indices_tensor[:, 0], indices_tensor[:, 1]]
            non_terminal = self.action_gate_types[step_actions] != self.gate_name_to_idx["terminal"]
            valid_backward = non_terminal & (step < lengths_selected - 1)
            if valid_backward.any():
                b_indices = indices_tensor[valid_backward]
                b_actions = step_actions[valid_backward]
            else:
                b_indices = None
                b_actions = None
            
            if step < max_length - 1:
                actions = torch.full((batch_size, n_measurements), self.terminal_index,
                                   dtype=torch.long, device=device)

                active_mask = (step < trajectory_batch.lengths - 1)
                if active_mask.any():
                    next_actions = trajectory_batch.actions[:, :, step]
                    terminal_mask = self.action_gate_types[next_actions] == self.gate_name_to_idx["terminal"]

                    actions[active_mask & ~terminal_mask] = next_actions[active_mask & ~terminal_mask]

                    temp_batch.active = active_mask & ~terminal_mask
                    batched_tableau.active = temp_batch.active.clone()

                    if temp_batch.active.any():
                        self.apply_actions_to_batch(
                            batched_tableau, actions, temp_batch, step=step
                        )

                if b_indices is not None and b_indices.shape[0] > 0:
                    states_next, indices_next = batched_tableau.to_flat_tensors_active_only()
                    with torch.no_grad():
                        logits_b = self.pb_model(states_next)
                        if logits_b.dim() == 1:
                            logits_b = logits_b.unsqueeze(0).expand(states_next.shape[0], -1)

                    masks_next = self.compute_backward_masks_gpu(
                        temp_batch, current_step=step + 1, forward_masks=None
                    )

                    if isinstance(indices_next, list):
                        indices_next_tensor = torch.tensor(indices_next, dtype=torch.long, device=device)
                    elif isinstance(indices_next, torch.Tensor):
                        indices_next_tensor = indices_next.to(device)
                    else:
                        indices_next_tensor = torch.stack([torch.as_tensor(idx, device=device) for idx in indices_next])

                    b_exp = b_indices.unsqueeze(1)
                    next_exp = indices_next_tensor.unsqueeze(0)
                    matches = torch.all(b_exp == next_exp, dim=-1)
                    valid_b_mask = torch.any(matches, dim=1)
                    if valid_b_mask.any():
                        mapped = torch.argmax(matches.float(), dim=1)[valid_b_mask]
                        b_indices = b_indices[valid_b_mask]
                        b_actions = b_actions[valid_b_mask]

                        b_masks = masks_next[b_indices[:,0], b_indices[:,1]].clone()
                        b_masks[:, self.terminal_index] = False
                        masked_logits_b = logits_b[mapped].masked_fill(~b_masks, float('-inf'))
                        valid_any_b = torch.isfinite(masked_logits_b).any(dim=1)
                        log_probs_b = torch.zeros_like(masked_logits_b)
                        if valid_any_b.any():
                            log_probs_b[valid_any_b] = torch.nn.functional.log_softmax(
                                masked_logits_b[valid_any_b], dim=-1
                            )
                        
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
    
    def compute_flows_cached(self, trajectory_batch: TrajectoryBatch,
                           max_depth: Optional[int] = None,
                           compute_gradients: bool = True,
                           chunk_size: int = 5000) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute flows using cached states with batched operations and precomputed backward counts."""
        batch_size = trajectory_batch.batch_size
        n_measurements = trajectory_batch.n_measurements
        device = trajectory_batch.device
        
        if (hasattr(trajectory_batch, '_forward_flows_buffers') and 
            trajectory_batch._forward_flows_buffers[0].shape == (batch_size, n_measurements)):
            idx = trajectory_batch._flow_buffer_idx
            trajectory_batch._flow_buffer_idx = 1 - idx
            forward_flows = trajectory_batch._forward_flows_buffers[idx].zero_()
            backward_flows = trajectory_batch._backward_flows_buffers[idx].zero_()
        else:
            forward_flows = torch.zeros((batch_size, n_measurements), device=device)
            backward_flows = torch.zeros((batch_size, n_measurements), device=device)
        
        if not trajectory_batch.cached_states:
            logging.warning("No cached states found in trajectory batch!")
            return forward_flows, backward_flows
        
        all_states = []
        all_indices = []
        all_actions = []
        all_masks = []
        step_mapping = []
        
        for step, cached_data in enumerate(trajectory_batch.cached_states):
            if cached_data is None:
                continue
                
            states_tensor, indices = cached_data
            if states_tensor.shape[0] == 0:
                continue
                
            valid_mask = step < trajectory_batch.lengths[indices[:, 0], indices[:, 1]]
            
            if valid_mask.any():
                valid_indices = indices[valid_mask]
                valid_states = states_tensor[valid_mask]
                valid_actions = trajectory_batch.actions[
                    valid_indices[:, 0], valid_indices[:, 1], step
                ]
                if (hasattr(trajectory_batch, 'cached_masks') and 
                    trajectory_batch.cached_masks and 
                    step < len(trajectory_batch.cached_masks) and 
                    trajectory_batch.cached_masks[step] is not None):
                    valid_masks = trajectory_batch.cached_masks[step][
                        valid_indices[:, 0], valid_indices[:, 1]
                    ]
                else:
                    valid_masks = torch.ones(
                        (valid_states.shape[0], self.num_actions), 
                        dtype=torch.bool, 
                        device=valid_states.device
                    )
                    if self.debug:
                        logging.warning(f"Step {step}: Masks not cached, using all-true masks")
                
                all_states.append(valid_states)
                all_indices.append(valid_indices)
                all_actions.append(valid_actions)
                all_masks.append(valid_masks)
                step_mapping.extend([step] * valid_states.shape[0])
        
        if not all_states:
            return forward_flows, backward_flows
        
        concat_states = torch.cat(all_states, dim=0)
        concat_indices = torch.cat(all_indices, dim=0)
        concat_actions = torch.cat(all_actions, dim=0)
        concat_masks = torch.cat(all_masks, dim=0)
        
        num_states = concat_states.shape[0]
        
        all_log_probs_f = []
        for chunk_idx, i in enumerate(range(0, num_states, chunk_size)):
            chunk_end = min(i + chunk_size, num_states)
            chunk_states = concat_states[i:chunk_end]
            chunk_masks = concat_masks[i:chunk_end]
            
            if compute_gradients:
                logits_f = self.pf_model(chunk_states)
            else:
                with torch.no_grad():
                    logits_f = self.pf_model(chunk_states)
            
            masked_logits_f = logits_f.masked_fill(~chunk_masks, float('-inf'))
            valid_any = torch.isfinite(masked_logits_f).any(dim=1)
            
            log_probs_f = torch.zeros_like(masked_logits_f)
            if valid_any.any():
                log_probs_f[valid_any] = F.log_softmax(masked_logits_f[valid_any], dim=-1)
            
            all_log_probs_f.append(log_probs_f)
        
        all_log_probs_f = torch.cat(all_log_probs_f, dim=0)
        selected_log_probs = all_log_probs_f.gather(1, concat_actions.unsqueeze(1)).squeeze(1)
        
        valid_probs = torch.isfinite(selected_log_probs)
        if not valid_probs.all():
            valid_indices = valid_probs.nonzero(as_tuple=True)[0]
            selected_log_probs = selected_log_probs[valid_indices]
            concat_indices = concat_indices[valid_indices]
        
        flat_indices = concat_indices[:, 0] * n_measurements + concat_indices[:, 1]
        forward_flows_flat = torch.zeros(batch_size * n_measurements, device=device)
        if len(selected_log_probs) > 0:
            forward_flows_flat.scatter_add_(0, flat_indices, selected_log_probs)
        forward_flows = forward_flows_flat.view(batch_size, n_measurements)
        
        if trajectory_batch.cached_backward_valid_counts:
            for step in range(len(trajectory_batch.cached_states) - 1):
                if trajectory_batch.cached_states[step] is None:
                    continue
                
                states_tensor, indices = trajectory_batch.cached_states[step]
                if states_tensor.shape[0] == 0:
                    continue
                
                if trajectory_batch.cached_backward_valid_counts[step + 1] is None:
                    continue
                    
                next_valid_counts = trajectory_batch.cached_backward_valid_counts[step + 1]
                
                valid_mask = step < trajectory_batch.lengths[indices[:, 0], indices[:, 1]]
                if not valid_mask.any():
                    continue
                
                valid_indices = indices[valid_mask]
                step_actions = trajectory_batch.actions[valid_indices[:, 0], valid_indices[:, 1], step]
                
                lengths_selected = trajectory_batch.lengths[valid_indices[:, 0], valid_indices[:, 1]]
                non_terminal = self.action_gate_types[step_actions] != self.gate_name_to_idx["terminal"]
                valid_backward = non_terminal & (step < lengths_selected - 1)
                
                if not valid_backward.any():
                    continue
                
                b_indices = valid_indices[valid_backward]
                b_valid_counts = next_valid_counts[b_indices[:, 0], b_indices[:, 1]]
                b_valid_counts = b_valid_counts.clamp(min=1)
                log_probs_uniform = -torch.log(b_valid_counts.float())
                backward_flows[b_indices[:, 0], b_indices[:, 1]] += log_probs_uniform
        
        if self.debug:
            logging.debug("\nDEBUG compute_flows_cached final:")
            logging.debug(f"  Forward flows shape: {forward_flows.shape}")
            logging.debug(f"  Backward flows shape: {backward_flows.shape}")
            logging.debug(f"  Total cached states processed: {num_states}")
            logging.debug(f"  Used precomputed backward counts: {bool(trajectory_batch.cached_backward_valid_counts)}")
        
        return forward_flows, backward_flows
    
    def compute_loss(self, trajectory_batch: TrajectoryBatch, costs: torch.Tensor,
                    beta: float = 1.0, max_depth: Optional[int] = None,
                    **reward_kwargs) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute loss for a batch of trajectories with fully vectorized operations."""
        assert costs.shape[0] == trajectory_batch.batch_size, \
            f"Costs shape {costs.shape} doesn't match batch size {trajectory_batch.batch_size}"
        
        costs = costs.to(self.device)
        if max_depth is None:
            max_depth = getattr(self, "last_max_depth", None)
        
        if self.debug:
            logging.debug(f"\nDEBUG compute_loss start:")
            logging.debug(f"  Batch size: {trajectory_batch.batch_size}")
            logging.debug(f"  N measurements: {trajectory_batch.n_measurements}")
            logging.debug(f"  Costs shape: {costs.shape}, device: {costs.device}")
            logging.debug(f"  Max depth: {max_depth}")
            logging.debug(f"  Cache enabled: {trajectory_batch.cache_enabled}")
            logging.debug(f"  Cached states: {len(trajectory_batch.cached_states) if trajectory_batch.cached_states else 0}")
        
        forward_flows, backward_flows = self.compute_flows(trajectory_batch, max_depth=max_depth, compute_gradients=True)
        
        rewards = self.reward_fn(costs, beta=beta, **reward_kwargs)
        
        if torch.isnan(rewards).any() or torch.isinf(rewards).any():
            logging.warning(f"NaN/Inf detected in rewards! Costs stats: min={costs.min().item():.6f}, "
                          f"max={costs.max().item():.6f}, mean={costs.mean().item():.6f}")
        
        if self.debug:
            logging.debug(f"  Forward flows shape: {forward_flows.shape}, non-zero: {(forward_flows != 0).sum().item()}")
            logging.debug(f"  Backward flows shape: {backward_flows.shape}, non-zero: {(backward_flows != 0).sum().item()}")
            logging.debug(f"  Rewards shape: {rewards.shape}, mean: {rewards.mean().item():.4f}")
        
        valid_mask = trajectory_batch.lengths > 0
        valid_counts = valid_mask.sum(dim=1)
        
        if self.debug:
            logging.debug(f"  Valid counts shape: {valid_counts.shape}, non-zero: {(valid_counts > 0).sum().item()}")
        
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
            
            logZ = self.pf_model.logZ if hasattr(self.pf_model, 'logZ') else torch.tensor(0.0, device=self.device)
            
            loss, objective_metrics = self.objective.compute_loss(
                forward_flows=forward_flows_avg,
                backward_flows=backward_flows_avg,
                rewards=rewards_filtered,
                logZ=logZ
            )
            
            if self.debug:
                logging.debug(f"  Loss value: {loss.item()}, logZ value: {self.pf_model.logZ.item()}")
            
            metrics_tensors = {
                'loss': loss.mean() if loss.dim() > 0 else loss,
                'reward': rewards.mean(),
                'cost': costs.mean(),
                'logZ': logZ.mean() if logZ.dim() > 0 else logZ,
                'avg_trajectories_per_batch': valid_counts.float().mean()
            }
            for k, v in objective_metrics.items():
                metrics_tensors[k] = v.mean() if torch.is_tensor(v) and v.dim() > 0 else v
            
            metrics = {k: v.item() if torch.is_tensor(v) else v for k, v in metrics_tensors.items()}
        else:
            zero_loss = torch.zeros(1, device=self.device, requires_grad=True).squeeze()
            return zero_loss, {
                'loss': 0.0,
                'reward': 0.0,
                'cost': costs.mean().item() if costs.numel() > 0 else 0.0,
                'logZ': logZ.item() if hasattr(self.pf_model, 'logZ') and self.pf_model.logZ.numel() == 1 else 0.0,
                'avg_trajectories_per_batch': 0.0
            }
        
        return loss, metrics
    
    def sample_trajectories(self, 
                          batch_size: int,
                          n_measurements: int,
                          max_depth: int,
                          mode: SamplingMode = SamplingMode.ON_POLICY,
                          batch_data_list: Optional[List[Dict]] = None,
                          cache_for_flows: bool = True) -> TrajectoryBatch:
        """Sample trajectories with depth limit and adaptive buffer sizing."""
        self.last_max_depth = max_depth
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
            return self._replay_trajectories(batch_data_list, max_length)
        
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
        
        if cache_for_flows:
            trajectory_batch.enable_caching()
        
        with torch.no_grad():
            for step in range(max_length):
                if not trajectory_batch.active.any():
                    break

                states_tensor, indices = batched_tableau.to_flat_tensors_active_only()
                if states_tensor.shape[0] == 0:
                    break

                if self.device.type in ['cuda', 'mps']:
                    states_tensor = states_tensor.contiguous()

                masks = self.compute_action_masks_gpu(trajectory_batch, max_depth)
                
                backward_valid_counts = None
                if cache_for_flows and step < max_length - 1:
                    backward_masks = self.compute_backward_masks_gpu(
                        trajectory_batch, current_step=step + 1, forward_masks=None
                    )
                    backward_masks[:, :, self.terminal_index] = False
                    backward_valid_counts = backward_masks.sum(dim=2)
                
                if cache_for_flows:
                    trajectory_batch.cache_step_data(
                        step, states_tensor, indices, masks, backward_valid_counts
                    )
                
                actions = torch.full((batch_size, n_measurements), self.terminal_index,
                                   dtype=torch.long, device=self.device)
                
                if mode == SamplingMode.ON_POLICY:
                    if self.device.type == 'cuda' and states_tensor.shape[0] > 0:
                        bucket_sizes = [64, 128, 256, 512, 1024, 2048, 4096, 8192]
                        current_size = states_tensor.shape[0]
                        padded_size = current_size
                        for bucket in bucket_sizes:
                            if current_size <= bucket:
                                padded_size = bucket
                                break

                        if padded_size > current_size:
                            padding = padded_size - current_size
                            padding_tensor = torch.zeros(padding, states_tensor.shape[1],
                                                        dtype=states_tensor.dtype,
                                                        device=states_tensor.device)
                            padded_states = torch.cat([states_tensor, padding_tensor], dim=0)
                            padded_logits = self.pf_model(padded_states)
                            logits = padded_logits[:current_size]
                        else:
                            logits = self.pf_model(states_tensor)
                    else:
                        logits = self.pf_model(states_tensor)

                    if isinstance(indices, torch.Tensor):
                        indices_tensor = indices.to(self.device)
                    else:
                        indices_tensor = torch.as_tensor(indices, dtype=torch.long, device=self.device)

                    active_masks = masks[indices_tensor[:, 0], indices_tensor[:, 1]]
                    masked_logits = logits.clone()
                    masked_logits[~active_masks] = float('-inf')

                    dist = Categorical(logits=masked_logits)
                    sampled_actions = dist.sample()

                    valid_any = torch.isfinite(masked_logits).any(dim=1)
                    sampled_actions = torch.where(
                        valid_any,
                        sampled_actions,
                        torch.full_like(sampled_actions, self.terminal_index),
                    )

                    actions[indices_tensor[:, 0], indices_tensor[:, 1]] = sampled_actions
                    trajectory_batch.actions[indices_tensor[:, 0], indices_tensor[:, 1], step] = sampled_actions

                elif mode == SamplingMode.OFF_POLICY:
                    if isinstance(indices, torch.Tensor):
                        indices_tensor = indices.to(self.device)
                    else:
                        indices_tensor = torch.as_tensor(indices, dtype=torch.long, device=self.device)

                    active_masks = masks[indices_tensor[:, 0], indices_tensor[:, 1]]
                    off_logits = torch.zeros_like(active_masks, dtype=torch.float32)
                    off_logits[~active_masks] = float('-inf')

                    dist = Categorical(logits=off_logits)
                    sampled_actions = dist.sample()

                    actions[indices_tensor[:, 0], indices_tensor[:, 1]] = sampled_actions
                    trajectory_batch.actions[indices_tensor[:, 0], indices_tensor[:, 1], step] = sampled_actions
                
                terminated = self.apply_actions_to_batch(
                    batched_tableau, actions, trajectory_batch, step=step
                )
                
                newly_terminated = terminated & (trajectory_batch.lengths == 0)
                if newly_terminated.any():
                    terminated_indices = torch.nonzero(newly_terminated, as_tuple=False)
                    trajectory_batch.lengths[terminated_indices[:, 0], terminated_indices[:, 1]] = step + 1
                
                if step == max_length - 1:
                    still_active = trajectory_batch.active & (trajectory_batch.lengths == 0)
                    if still_active.any():
                        active_indices = torch.nonzero(still_active, as_tuple=False)
                        trajectory_batch.lengths[active_indices[:, 0], active_indices[:, 1]] = max_length
                        trajectory_batch.active[active_indices[:, 0], active_indices[:, 1]] = False
        
        if self.adaptive_tracker is not None:
            self.adaptive_tracker.update_statistics(trajectory_batch)
        
        return trajectory_batch
    
    def update_step(self, accumulated_loss: torch.Tensor) -> float:
        """Perform a single gradient update step."""
        # Skip update if no optimizer (e.g., DiscreteUniform model)
        if self.optimizer is None:
            return 0.0
            
        if torch.isnan(accumulated_loss) or torch.isinf(accumulated_loss):
            logging.warning(f"NaN or Inf detected in accumulated loss (value: {accumulated_loss.item() if accumulated_loss.numel() == 1 else 'tensor'}), skipping update")
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
        
        for b_idx in range(batch_size):
            batch_cost = costs[b_idx].item()
            
            valid_mask = trajectory_batch.lengths[b_idx] > 0
            if not valid_mask.any():
                continue
            
            batch_actions = trajectory_batch.actions[b_idx].clone()
            batch_lengths = trajectory_batch.lengths[b_idx].clone()
            
            if len(self.top_trajectories_costs) < self.K:
                self.top_trajectories_actions.append(batch_actions)
                self.top_trajectories_lengths.append(batch_lengths)
                self.top_trajectories_costs.append(batch_cost)
            else:
                max_cost = max(self.top_trajectories_costs)
                if batch_cost < max_cost:
                    max_idx = self.top_trajectories_costs.index(max_cost)
                    self.top_trajectories_actions[max_idx] = batch_actions
                    self.top_trajectories_lengths[max_idx] = batch_lengths
                    self.top_trajectories_costs[max_idx] = batch_cost

    def _replay_trajectories(self, batch_data_list: Optional[List[Dict]], 
                                     max_length: int) -> TrajectoryBatch:
        """Replay trajectories with minimal CPU-GPU transfer."""
        if not hasattr(self, 'top_trajectories_actions') or not self.top_trajectories_actions:
            return TrajectoryBatch(
                batch_size=0,
                n_measurements=1,
                max_length=max_length,
                n_qubits=self.n_qubits,
                device=self.device
            )
        
        n_batches = len(self.top_trajectories_actions)
        batch_size = n_batches
        n_measurements = self.top_trajectories_actions[0].shape[0]
        
        if self.top_trajectories_lengths:
            all_lengths_max = torch.stack([l.max() for l in self.top_trajectories_lengths]).max().item()
            actual_max_length = max(max_length, int(all_lengths_max))
        else:
            actual_max_length = max_length
        
        if actual_max_length > max_length:
            if self.debug:
                logging.info(f"Replay: Extending max_length from {max_length} to {actual_max_length}")
            max_length = actual_max_length
        
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
        trajectory_batch.enable_caching()
        trajectory_batch.qubit_last_use_step.fill_(-1)
        
        for b_idx in range(batch_size):
            stored_actions = self.top_trajectories_actions[b_idx]
            stored_lengths = self.top_trajectories_lengths[b_idx]
            
            actual_max_length = min(max_length, stored_actions.shape[1])
            trajectory_batch.actions[b_idx, :, :actual_max_length] = stored_actions[:, :actual_max_length]
            trajectory_batch.lengths[b_idx] = torch.minimum(stored_lengths, 
                                                           torch.tensor(max_length, device=self.device))
            trajectory_batch.active[b_idx] = stored_lengths > 0
        
        with torch.no_grad():
            for step in range(max_length):
                step_active = step < trajectory_batch.lengths
                if not step_active.any():
                    break
                
                states_tensor, indices = batched_tableau.to_flat_tensors_active_only()
                if states_tensor.shape[0] == 0:
                    break
                
                masks = self.compute_action_masks_gpu(trajectory_batch, max_depth=None)
                
                backward_valid_counts = None
                if step < max_length - 1:
                    backward_masks = self.compute_backward_masks_gpu(
                        trajectory_batch, current_step=step + 1, forward_masks=None
                    )
                    backward_masks[:, :, self.terminal_index] = False
                    backward_valid_counts = backward_masks.sum(dim=2)
                
                trajectory_batch.cache_step_data(
                    step, states_tensor, indices, masks, backward_valid_counts
                )
                
                actions = trajectory_batch.actions[:, :, step]
                terminated = self.apply_actions_to_batch(
                    batched_tableau, actions, trajectory_batch, step=step
                )
                trajectory_batch.active &= ~terminated
        
        return trajectory_batch
    
    def save_checkpoint(self, path: str, update: int, metrics: Dict):
        """Save model checkpoint including adaptive tracker state and async evaluation data."""
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
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer is not None else None,
            'update': update,
            'checkpoint_id': time.time(),
            'top_trajectories': top_trajectories_cpu,
            'metrics': metrics,
            'model_type': self.model_type,
            'n_qubits': self.n_qubits,
            'num_actions': self.num_actions,
            'objective_type': self.objective_type,
            'checkpoint_version': 'gpu_with_depth_async',
            'buffer_strategy': self.buffer_strategy,
            'action_mapping': self.action_mapping,
            'terminal_index': self.terminal_index,
        }
        
        if self.adaptive_tracker is not None:
            checkpoint['adaptive_tracker_state'] = {
                'gates_per_depth': dict(self.adaptive_tracker.gates_per_depth),
                'buffer_utilization': self.adaptive_tracker.buffer_utilization,
                'depth_distribution': dict(self.adaptive_tracker.depth_distribution),
                'max_gates_seen': self.adaptive_tracker.max_gates_seen.cpu().item(),
                'total_trajectories': self.adaptive_tracker.total_trajectories,
                'update_count': self.adaptive_tracker.update_count,
                'gate_counts': self.adaptive_tracker.gate_counts[:1000],
            }
        
        temp_path = path + '.tmp'
        torch.save(checkpoint, temp_path)
        os.rename(temp_path, path)
    
    def load_checkpoint(self, path: str) -> Tuple[int, Dict]:
        """Load model checkpoint and restore adaptive tracker state."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        
        if checkpoint.get('n_qubits') != self.n_qubits:
            logging.info(f"Qubit mismatch: checkpoint has {checkpoint.get('n_qubits')}, model has {self.n_qubits}")
            return 0, {}
        
        try:
            self.pf_model.load_state_dict(checkpoint['pf_model_state_dict'])
            self.pb_model.load_state_dict(checkpoint['pb_model_state_dict'])
            if self.optimizer is not None and checkpoint.get('optimizer_state_dict') is not None:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            self.top_trajectories_actions = []
            self.top_trajectories_lengths = []
            self.top_trajectories_costs = []
            
            if checkpoint.get('checkpoint_version') in ['gpu', 'gpu_with_depth', 'gpu_with_depth_async']:
                for traj_data in checkpoint.get('top_trajectories', []):
                    self.top_trajectories_actions.append(traj_data['actions'].to(self.device))
                    self.top_trajectories_lengths.append(traj_data['lengths'].to(self.device))
                    self.top_trajectories_costs.append(traj_data['cost'])
            
            if 'adaptive_tracker_state' in checkpoint and self.buffer_strategy == 'adaptive':
                tracker_state = checkpoint['adaptive_tracker_state']
                
                if self.adaptive_tracker is None:
                    initial_size = self.calculate_conservative_buffer_size(10)
                    self.adaptive_tracker = AdaptiveBufferTracker(
                        initial_buffer_size=initial_size,
                        device=self.device,
                        warmup_updates=self.adaptive_warmup
                    )
                
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
                 device_preference: Optional[str] = None,
                 metric_store: Optional[Any] = None,
                 metrics_window: int = 512):
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
        
        cost_config = self.training_config.get("cost", {})
        cost_type = cost_config.get("type", "exponential")
        normalization_type = cost_config.get("normalization_type", "sum")
        
        self.cost_kwargs = {k: v for k, v in cost_config.items() 
                           if k not in ("type", "custom_costs", "normalization_type")}
        
        if "epsilon" in self.training_config and "epsilon" not in self.cost_kwargs:
            self.cost_kwargs["epsilon"] = self.training_config["epsilon"]
        
        self.cost_computer = CostComputer(
            cost_type=cost_type,
            n_measurements=self.n_measurements,
            device=self.device,
            normalize_weights=True,
            normalization_type=normalization_type,
            pauli_strings=self.pauli_str_list,
            n_qubits=self.n_qubits
        )
        
        if "custom_costs" in cost_config:
            for name, custom_cost_config in cost_config["custom_costs"].items():
                if name == "threshold":
                    threshold = custom_cost_config.get("threshold", 0.5)
                    self.cost_computer.add_custom_cost_function(name, ThresholdCost(threshold))
        
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
        self.metric_store = metric_store
        self.metrics_window = max(metrics_window, 0)
        self.profiler = None

    def attach_profiler(self, profiler: Any) -> None:
        """Register an optional torch.profiler instance."""
        self.profiler = profiler

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
        
        if self.gfn.debug and not silence:
            logging.debug(f"\nDEBUG compute_costs_with_probabilities:")
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
                          e.g., {0: "exponential", 1000: "quadratic", 2000: "linear"}
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
        
        if self.gfn.debug:
            logging.debug(f"\nStarting training with debug mode enabled")
            logging.debug(f"Config: {json.dumps(self.config, indent=2)}")
        
        for update in range(start_update, num_updates):
            # Check if we need to change cost function
            if cost_schedule and update in cost_schedule:
                new_cost_type = cost_schedule[update]
                logging.info(f"\nSwitching to {new_cost_type} cost function at update {update}")
                self.cost_computer.set_cost_type(new_cost_type, self.n_measurements)
            
            update_start = time.time()
            
            if self.gfn.debug:
                logging.debug(f"\n=== DEBUG Update {update+1}/{num_updates} ===")
            
            logging.info(f"\n=== Update {update+1}/{num_updates} ===")
            logging.info(f"Cost function: {self.cost_computer.cost_type}")
            if self.cost_kwargs:
                logging.info(f"Cost kwargs: {self.cost_kwargs}")
            
            sample_start = time.time()
            
            # Sample on-policy trajectories with depth limit and caching
            with record_function("train.sample_trajectories"):
                trajectory_batch = self.gfn.sample_trajectories(
                    batch_size=self.update_freq,
                    n_measurements=self.n_measurements,
                    max_depth=self.max_depth,
                    mode=SamplingMode.ON_POLICY,
                    cache_for_flows=True  # Enable caching
                )
            
            # Compute costs using CostComputer with kwargs
            batched_tableau = trajectory_batch.batched_tableau
            with record_function("train.compute_costs"):
                costs = self.compute_costs_with_probabilities(batched_tableau)
            
            # Device-aware logging to minimize CPU transfers
            if self.device.type in ['cuda', 'mps']:
                mean_cost = costs.mean().item()
                min_cost = costs.min().item()
                max_cost = costs.max().item()
                if logging.getLogger().isEnabledFor(logging.DEBUG):
                    costs_cpu = costs.cpu().numpy()
                    logging.debug(f"  Batch costs (one per batch element): {costs_cpu}")
                logging.info(f"  Mean batch cost: {mean_cost:.4f}, "
                      f"Min: {min_cost:.4f}, Max: {max_cost:.4f}")
            else:
                costs_cpu = costs.cpu().numpy()
                logging.info(f"  Batch costs (one per batch element): {costs_cpu}")
                logging.info(f"  Mean batch cost: {costs_cpu.mean():.4f}, "
                      f"Min: {costs_cpu.min():.4f}, Max: {costs_cpu.max():.4f}")
            
            # Compute loss using the standard method
            with record_function("train.compute_loss"):
                loss, metrics = self.gfn.compute_loss(
                    trajectory_batch, costs, self.beta, max_depth=self.max_depth, **self.reward_kwargs
                )
            
            sample_time = time.time() - sample_start
            
            # Update model using the standard method
            with record_function("train.optimizer_step"):
                loss_value = self.gfn.update_step(loss)
            
            # Update top trajectories for replay
            with record_function("train.update_top_trajectories"):
                self.gfn._update_top_trajectories(trajectory_batch, costs)
            metrics['loss'] = loss_value
            metrics['cost_type'] = self.cost_computer.cost_type
            
            # update time after sampling and loss computation
            update_time = time.time() - update_start
            
            # Replay training
            if replay_every and (update + 1) % replay_every == 0 and self.gfn.top_trajectories_actions:
                replay_start = time.time()
                
                self.gfn.optimizer.zero_grad()
                
                # Sample replay trajectories WITH caching (needed for flow computation)
                with record_function("train.replay.sample"):
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
                with record_function("train.replay.compute_costs"):
                    replay_costs = self.compute_costs_with_probabilities(replay_tableau, silence=False)
                
                self.gfn._debug_replay = True
                
                # Compute loss using the exact same method as on-policy
                with record_function("train.replay.compute_loss"):
                    replay_loss, replay_metrics = self.gfn.compute_loss(
                        replay_batch, replay_costs, self.beta, max_depth=self.max_depth, **self.reward_kwargs
                    )
                
                self.gfn._debug_replay = False
                
                # Update step using the same method
                with record_function("train.replay.optimizer_step"):
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
                
                # Sample off-policy trajectories with caching
                with record_function("train.offpolicy.sample"):
                    offpolicy_batch = self.gfn.sample_trajectories(
                        batch_size=self.update_freq,
                        n_measurements=self.n_measurements,
                        max_depth=self.max_depth,
                        mode=SamplingMode.OFF_POLICY,
                        cache_for_flows=True  # Enable caching
                    )
                
                # Compute costs using CostComputer
                offpolicy_tableau = offpolicy_batch.batched_tableau
                with record_function("train.offpolicy.compute_costs"):
                    offpolicy_costs = self.compute_costs_with_probabilities(offpolicy_tableau)
                
                # Compute loss using the exact same method as on-policy
                with record_function("train.offpolicy.compute_loss"):
                    offpolicy_loss, offpolicy_metrics = self.gfn.compute_loss(
                        offpolicy_batch, offpolicy_costs, self.beta, max_depth=self.max_depth, **self.reward_kwargs
                    )
                
                # Update step using the same method
                with record_function("train.offpolicy.optimizer_step"):
                    offpolicy_loss_value = self.gfn.update_step(offpolicy_loss)
                offpolicy_metrics['loss'] = offpolicy_loss_value
                
                # Store metrics with offpolicy prefix
                for k, v in offpolicy_metrics.items():
                    metrics[f'offpolicy_{k}'] = v
                
                offpolicy_time = time.time() - offpolicy_start
                logging.info(f"  Off-policy: loss={offpolicy_metrics['loss']:.6f}, "
                      f"batch_reward={offpolicy_metrics['reward']:.4f}")
            
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

                if self.metrics_window and len(series) > self.metrics_window:
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
                    if self.metrics_window and len(series) > self.metrics_window:
                        del series[:-self.metrics_window]

            # Convert only metrics needed for logging (single batched GPU sync)
            # Batch extract the 4-5 scalars needed for logging in one transfer
            log_keys = ['loss', 'reward', 'cost', 'logZ']
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

            avg_trajs = metrics.get('avg_trajectories_per_batch', 0)
            if torch.is_tensor(avg_trajs):
                avg_trajs = avg_trajs.item()

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
                self.plot_metrics(update + 1)

            if self.profiler is not None:
                self.profiler.step()

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
        
        # Plot final metrics
        self.plot_metrics(num_updates)
    
    def plot_metrics(self, update: int):
        """Plot training metrics."""
        plots_dir = os.path.join(self.model_config["model_dir"], "figures")
        os.makedirs(plots_dir, exist_ok=True)

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()

        metrics_source = self.metrics_history
        timing_source = self.timing_history

        if self.metric_store is not None:
            _, stored_metrics, stored_timing = self.metric_store.load_series()
            if stored_metrics:
                metrics_source = stored_metrics
            if stored_timing:
                timing_source = stored_timing

        # Helper function to convert GPU tensors in lists to CPU for plotting
        def convert_list_for_plot(values):
            """Convert list of potential GPU tensors to numpy/floats for matplotlib."""
            converted = []
            for v in values:
                if torch.is_tensor(v):
                    converted.append(v.cpu().item() if v.numel() == 1 else v.cpu().numpy())
                else:
                    converted.append(v)
            return converted

        if 'loss' in metrics_source:
            axes[0].plot(convert_list_for_plot(metrics_source['loss']), label='On-policy')
            if 'replay_loss' in metrics_source:
                axes[0].plot(convert_list_for_plot(metrics_source['replay_loss']),
                           label='Replay', alpha=0.7, linestyle='--')
            if 'offpolicy_loss' in metrics_source:
                axes[0].plot(convert_list_for_plot(metrics_source['offpolicy_loss']),
                           label='Off-policy', alpha=0.7, linestyle=':')
            axes[0].set_title('Training Loss')
            axes[0].set_xlabel('Update')
            axes[0].set_ylabel('Loss')
            axes[0].set_yscale('log')
            axes[0].grid(True, alpha=0.3)
            axes[0].legend()

        if 'reward' in metrics_source:
            reward_cpu = convert_list_for_plot(metrics_source['reward'])
            axes[1].plot(reward_cpu, label='On-policy')
            if 'replay_reward' in metrics_source:
                axes[1].plot(convert_list_for_plot(metrics_source['replay_reward']),
                           label='Replay', alpha=0.7, linestyle='--')
            if 'offpolicy_reward' in metrics_source:
                axes[1].plot(convert_list_for_plot(metrics_source['offpolicy_reward']),
                           label='Off-policy', alpha=0.7, linestyle=':')
            axes[1].set_title('Average Batch Reward')
            axes[1].set_xlabel('Update')
            axes[1].set_ylabel('Batch Reward')
            # find the maximum y value for better visibility
            max_y = max(max(reward_cpu),
                        max(convert_list_for_plot(metrics_source.get('replay_reward', [0]))),
                        max(convert_list_for_plot(metrics_source.get('offpolicy_reward', [0]))))
            axes[1].set_ylim(0, max_y * 1.1)
            axes[1].grid(True, alpha=0.3)
            axes[1].legend()

        if 'cost' in metrics_source:
            axes[2].plot(convert_list_for_plot(metrics_source['cost']))
            axes[2].set_title('Average Cost')
            axes[2].set_xlabel('Update')
            axes[2].set_ylabel('Cost')
            axes[2].set_yscale('log')
            axes[2].grid(True, alpha=0.3)

        if 'throughput' in timing_source:
            axes[3].plot(convert_list_for_plot(timing_source['throughput']))
            axes[3].set_title('Training Throughput')
            axes[3].set_xlabel('Update')
            axes[3].set_ylabel('Trajectories/Second')
            axes[3].grid(True, alpha=0.3)
        elif 'logZ' in metrics_source:
            axes[3].plot(convert_list_for_plot(metrics_source['logZ']), color='green')
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
            "hidden_dim": 512,
            "num_hidden_layers": 3,
            "lr": 5e-3,
            "weight_decay": 1e-5,
            "model_dir": "results/",
            "model_kwargs": {},
            "objective_type": "tb",
            "objective_kwargs": {"loss_type": "squared"},
            "debug": False,  # Enable debug mode
            "buffer_strategy": "conservative",  # Change to "adaptive" for adaptive strategy
            "adaptive_warmup": 100
        },
        "training": {
            "beta": 100,
            "n_measurements": 1000,  # Number of trajectories per batch element
            "update_freq": 20,     # Number of batch elements (this is batch_size)
            "max_depth": 2,        # Now using depth instead of max_layer
            "K": 5,  # Number of top batches to keep for replay
            "reward_kwargs": {"alpha" : 10.0},
            "cost": {
                "type": "linear_bias",
                "epsilon": 0.9  # This is now part of cost kwargs
            }
        },
        "quantum": {
        "pauli_str_list": [
        "YZYIIIII",
        "XZXIIIII",
        "ZIIIIIII",
        "IIZIIIII",
        "IZIIIIII",
        "IYZYIIII",
        "IXZXIIII",
        "IIIZIIII",
        "IIIIZIII",
        "IIIIYZYI",
        "IIIIXZXI",
        "IIIIIZII",
        "IIIIIYZY",
        "IIIIIXZX",
        "IIIIIIZI",
        "IIIIIIIZ",
        "IYIYIIII",
        "IXIXIIII",
        "XIXIIIII",
        "YIYIIIII",
        "YZYIIZII",
        "XZXIIZII",
        "XXXXIIII",
        "YXXYIIII",
        "XXYYIIII",
        "YYXXIIII",
        "XYYXIIII",
        "YYYYIIII",
        "YYIIIXXI",
        "YYIIIYYI",
        "XXIIIXXI",
        "XXIIIYYI",
        "ZIIZIIII",
        "YZZYIXXI",
        "YZZYIYYI",
        "XZZXIXXI",
        "XZZXIYYI",
        "ZYZYIIII",
        "ZXZXIIII",
        "ZIIIIXZX",
        "ZIIIIYZY",
        "YZYIIXZX",
        "YZYIIYZY",
        "XZXIIXZX",
        "XZXIIYZY",
        "ZZIIIIII",
        "ZIIIYZYI",
        "ZIIIXZXI",
        "YZYIYZYI",
        "YZYIXZXI",
        "XZXIYZYI",
        "XZXIXZXI",
        "ZIIIIIZI",
        "YZYIIIZI",
        "XZXIIIZI",
        "YYIIIIXX",
        "YYIIIIYY",
        "XXIIIIXX",
        "XXIIIIYY",
        "IZZIIIII",
        "YZZYIIXX",
        "YZZYIIYY",
        "XZZXIIXX",
        "XZZXIIYY",
        "YYIIYZZY",
        "YYIIXZZX",
        "XXIIYZZY",
        "XXIIXZZX",
        "YZZYYZZY",
        "YZZYXZZX",
        "XZZXYZZY",
        "XZZXXZZX",
        "ZIIIIIIZ",
        "YZYIIIIZ",
        "XZXIIIIZ",
        "YZYIZIII",
        "XZXIZIII",
        "YZYZIIII",
        "XZXZIIII",
        "YYIIXXII",
        "YYIIYYII",
        "XXIIXXII",
        "XXIIYYII",
        "IYZYYZYI",
        "IYZYXZXI",
        "IXZXYZYI",
        "IXZXXZXI",
        "IYYIYZZY",
        "IYYIXZZX",
        "IXXIYZZY",
        "IXXIXZZX",
        "IZIIIYZY",
        "IZIIIXZX",
        "IYZYIYZY",
        "IYZYIXZX",
        "IXZXIYZY",
        "IXZXIXZX",
        "IYYIIYYI",
        "IYYIIXXI",
        "IXXIIYYI",
        "IXXIIXXI",
        "IYYIIIYY",
        "IYYIIIXX",
        "IXXIIIYY",
        "IXXIIIXX",
        "ZIZIIIII",
        "IZIIIIZI",
        "IZIIIIIZ",
        "IZIZIIII",
        "IYYIYYII",
        "IYYIXXII",
        "IXXIYYII",
        "IXXIXXII",
        "ZIIIZIII",
        "IZIIIZII",
        "IYZYIZII",
        "IXZXIZII",
        "IZIIYZYI",
        "IZIIXZXI",
        "IZIIZIII",
        "IYZYZIII",
        "IXZXZIII",
        "IYZYIIZI",
        "IXZXIIZI",
        "YZZYXXII",
        "YZZYYYII",
        "XZZXXXII",
        "XZZXYYII",
        "ZIIIIZII",
        "IYZYIIIZ",
        "IXZXIIIZ",
        "IIZZIIII",
        "IIZIZIII",
        "IIYYXXII",
        "IIYYYYII",
        "IIXXXXII",
        "IIXXYYII",
        "IIZIXZXI",
        "IIZIYZYI",
        "IIYYXZZX",
        "IIYYYZZY",
        "IIXXXZZX",
        "IIXXYZZY",
        "IIZIIZII",
        "IIYYIXXI",
        "IIYYIYYI",
        "IIXXIXXI",
        "IIXXIYYI",
        "IIZIIXZX",
        "IIZIIYZY",
        "IIZIIIZI",
        "IIYYIIXX",
        "IIYYIIYY",
        "IIXXIIXX",
        "IIXXIIYY",
        "IIZIIIIZ",
        "IIIZZIII",
        "IIIZXZXI",
        "IIIZYZYI",
        "IIIZIZII",
        "IIIZIXZX",
        "IIIZIYZY",
        "IIIZIIZI",
        "IIIZIIIZ",
        "IIIIZZII",
        "IIIIZYZY",
        "IIIIZXZX",
        "IIIIZIZI",
        "IIIIZIIZ",
        "IIIIYIYI",
        "IIIIXIXI",
        "IIIIYYYY",
        "IIIIYYXX",
        "IIIIYXXY",
        "IIIIXYYX",
        "IIIIXXYY",
        "IIIIXXXX",
        "IIIIYZYZ",
        "IIIIXZXZ",
        "IIIIIZZI",
        "IIIIIZIZ",
        "IIIIIYIY",
        "IIIIIXIX",
        "IIIIIIZZ"
        ],
        "w_list": [
        0.08794122934165582,
        0.08794122934165582,
        -0.2724180193998683,
        -0.6921496164450819,
        -0.4043851244705001,
        -0.05255329774132767,
        -0.05255329774132767,
        -1.0346287559322498,
        -0.2724180193998682,
        0.08794122934165581,
        0.08794122934165581,
        -0.4043851244705001,
        -0.05255329774132766,
        -0.05255329774132766,
        -0.6921496164450818,
        -1.0346287559322498,
        0.023907160243420156,
        0.023907160243420156,
        -0.017318726572294618,
        -0.017318726572294618,
        -0.012608046401606652,
        -0.012608046401606652,
        -0.015412105039717084,
        -0.012794256152536574,
        -0.0026178488871805078,
        -0.0026178488871805078,
        -0.012794256152536574,
        -0.015412105039717084,
        -0.004710680170687968,
        -0.004710680170687968,
        -0.004710680170687968,
        -0.004710680170687968,
        0.13095738454560127,
        -0.005591850052488086,
        -0.005591850052488086,
        -0.005591850052488086,
        -0.005591850052488086,
        0.015961155441576884,
        0.015961155441576884,
        0.03583078300215678,
        0.03583078300215678,
        -0.01838610620502466,
        -0.01838610620502466,
        -0.01838610620502466,
        -0.01838610620502466,
        0.08831604919069061,
        -0.041611907907888844,
        -0.041611907907888844,
        0.027263069036128847,
        0.027263069036128847,
        0.027263069036128847,
        0.027263069036128847,
        0.1328282413043635,
        -0.02996880154502259,
        -0.02996880154502259,
        -0.021003955092205172,
        -0.021003955092205172,
        -0.021003955092205172,
        -0.021003955092205172,
        0.08647780044339264,
        0.030819246688616207,
        0.030819246688616207,
        0.030819246688616207,
        0.030819246688616207,
        -0.0198696275605799,
        -0.0198696275605799,
        -0.0198696275605799,
        -0.0198696275605799,
        0.0342610794129385,
        0.0342610794129385,
        0.0342610794129385,
        0.0342610794129385,
        0.16521846395853979,
        -0.050238442633643486,
        -0.050238442633643486,
        -0.04161190790788884,
        -0.04161190790788884,
        -0.01941919594502728,
        -0.01941919594502728,
        0.02016748720380559,
        0.02016748720380559,
        0.02016748720380559,
        0.02016748720380559,
        -0.01838610620502466,
        -0.01838610620502466,
        -0.01838610620502466,
        -0.01838610620502466,
        -0.005591850052488085,
        -0.005591850052488085,
        -0.005591850052488085,
        -0.005591850052488085,
        0.013785633495284501,
        0.013785633495284501,
        0.016976725669297192,
        0.016976725669297192,
        0.016976725669297192,
        0.016976725669297192,
        0.008949858543082407,
        0.008949858543082407,
        0.008949858543082407,
        0.008949858543082407,
        -0.0007987348107616162,
        -0.0007987348107616162,
        -0.0007987348107616162,
        -0.0007987348107616162,
        0.10556517226823464,
        0.09542765898647504,
        0.11066162842599216,
        0.09368490275669497,
        -0.004710680170687968,
        -0.004710680170687968,
        -0.004710680170687968,
        -0.004710680170687968,
        0.1619798167438987,
        0.09652650762551396,
        0.013785633495284501,
        0.013785633495284501,
        -0.012608046401606652,
        -0.012608046401606652,
        0.1084835363944962,
        0.03583078300215678,
        0.03583078300215678,
        0.02470589505418177,
        0.02470589505418177,
        -0.0198696275605799,
        -0.0198696275605799,
        -0.0198696275605799,
        -0.0198696275605799,
        0.1084835363944962,
        0.041946477622790114,
        0.041946477622790114,
        0.10585039887546002,
        0.1328282413043635,
        -0.021003955092205172,
        -0.021003955092205172,
        -0.021003955092205172,
        -0.021003955092205172,
        -0.02996880154502259,
        -0.02996880154502259,
        0.030819246688616207,
        0.030819246688616207,
        0.030819246688616207,
        0.030819246688616207,
        0.09542765898647504,
        -0.0007987348107616162,
        -0.0007987348107616162,
        -0.0007987348107616162,
        -0.0007987348107616162,
        0.02470589505418177,
        0.02470589505418177,
        0.1159589087829886,
        0.03211770000941987,
        0.03211770000941987,
        0.03211770000941987,
        0.03211770000941987,
        0.1379680988848799,
        0.16521846395853979,
        -0.050238442633643486,
        -0.050238442633643486,
        0.11066162842599216,
        0.04194647762279012,
        0.04194647762279012,
        0.1379680988848799,
        0.18449350486670163,
        0.08831604919069061,
        0.015961155441576884,
        0.015961155441576884,
        0.10556517226823464,
        0.13095738454560127,
        -0.017318726572294618,
        -0.017318726572294618,
        -0.01541210503971709,
        -0.002617848887180506,
        -0.012794256152536574,
        -0.012794256152536574,
        -0.002617848887180506,
        -0.01541210503971709,
        -0.01941919594502728,
        -0.01941919594502728,
        0.08647780044339264,
        0.09368490275669497,
        0.023907160243420156,
        0.023907160243420156,
        0.10585039887546002
        ]
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
    trainer = EfficientGFNTrainer(config,reward_fn=log_reward_fn,)# device_preference="mps")
    
    trainer.train(
        num_updates=100000,
        replay_every=25,
        offpolicy_every=20,
        checkpoint_every=50,
    )
