# -*- coding: utf-8 -*-
"""Trajectory containers for GFlowNet sampling/training.

Split out of ``GFNs.py``. Owns ``AdaptiveBufferTracker`` and
``TrajectoryBatch`` (GPU-resident trajectory storage with per-step state
caching used by the cached-flow path).
"""

import numpy as np
import torch
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union

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
        # GPU OPTIMIZATION: Vectorized processing instead of nested Python loops
        # Get valid trajectories (lengths > 0)
        valid_mask = trajectory_batch.lengths > 0
        valid_depths = trajectory_batch.circuit_depths[valid_mask]
        valid_gates = trajectory_batch.lengths[valid_mask]
        
        if valid_gates.numel() > 0:
            # Single CPU transfer for all valid data
            depths_cpu = valid_depths.cpu().tolist()
            gates_cpu = valid_gates.cpu().tolist()
            
            # Process transferred data
            for depth, gates in zip(depths_cpu, gates_cpu):
                self.gates_per_depth[depth].append(gates)
                self.gate_counts.append(gates)
                self.depth_distribution[depth] += 1
            
            self.total_trajectories += len(gates_cpu)
            
            # Update max gates seen (keep on GPU)
            self.max_gates_seen = torch.max(self.max_gates_seen, valid_gates.max())
        
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
    """Container for batch trajectory data with circuit depth tracking and state caching."""
    
    def __init__(self, batch_size: int, n_measurements: int, max_length: int, 
                n_qubits: int, device: torch.device):
        self.batch_size = batch_size
        self.n_measurements = n_measurements
        self.max_length = max_length
        self.n_qubits = n_qubits
        self.device = device
        
        # Keep all tensors on GPU. Action history is stored time-major so the
        # hot per-step plane is contiguous; ``self.actions`` is the legacy
        # batch-major view expected by existing callers/tests.
        self.actions_time_major = torch.zeros((max_length, batch_size, n_measurements),
                                              dtype=torch.long, device=device)
        self.actions = self.actions_time_major.permute(1, 2, 0)
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
        
        # Gate tracking. NOTE: a former (B, M, n, n) ``last_two_qubit_gates``
        # buffer was removed — it was the only O(B*M*n^2) tensor and was
        # write-only (moved 2q-layer validity onto ``current_layer_qubits``,
        # O(B*M*n)); nothing read it. Removing it drops the next memory ceiling
        # above the GIPTE cache and the per-step symmetric scatter writes.
        self.last_single_qubit_gates = torch.zeros((batch_size, n_measurements, n_qubits),
                                                    dtype=torch.long, device=device) - 1
        
        # Track which step each qubit was last used (for backward policy)
        self.qubit_last_use_step = torch.full((batch_size, n_measurements, n_qubits), 
                                              -1, dtype=torch.long, device=device)
        
        # Track action history with qubit info for exposed gate computation
        self.action_qubits = torch.full((batch_size, n_measurements, max_length, 2), 
                                        -1, dtype=torch.long, device=device)  # -1 for unused
        
        self.batched_tableau = None
        
        # State caching for flow computation
        self.cached_states = []  # List of (states_tensor, indices) for each step
        self.cached_masks = []  # Action masks for each step
        self.cached_backward_valid_counts = []  # Number of valid backward actions for each step
        self.cache_enabled = False  # Flag to control caching
        # Store cached state features as uint8 instead of fp32 (4x smaller —
        # the cached-state term scales with row-steps and dominates the
        # per-batch memory at 20q+). VALID ONLY when the features are exactly
        # 0/1 (the default flattened-W policy input): the uint8->fp32 upcast
        # in ``_forward_selected`` is then bit-exact. The sampler sets this
        # from ``GFlowNet._effective_uint8_state_cache()``, which gates off
        # feature_extractor (GIPTE floats) and packed_w_input modes.
        self.cache_states_uint8 = False
        
        # GPU OPTIMIZATION: Double-buffered pre-allocation to avoid repeated allocations while
        # maintaining safety (previous results survive at least one more call).
        # This is a classic GPU optimization pattern that eliminates buffer aliasing issues.
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
        
    def enable_caching(self, states_uint8: bool = False):
        """Enable state caching for flow computation.

        ``states_uint8=True`` opts the cached state features into uint8
        storage (callers pass ``GFlowNet._effective_uint8_state_cache()``,
        which gates on the flattened-W feature mode — see ``__init__``).
        """
        self.cache_enabled = True
        self.cache_states_uint8 = bool(states_uint8)
        self.cached_states = []
        self.cached_masks = []
        self.cached_backward_valid_counts = []
        
    def cache_step_data(self, step: int, states_tensor: torch.Tensor,
                       indices: Union[List[Tuple], torch.Tensor], masks: torch.Tensor,
                       backward_valid_counts: Optional[torch.Tensor] = None):
        """Cache state data for a specific step during sampling.

        ``masks`` is the full ``(B, M, num_actions)`` mask tensor; this method
        slices it down to only the rows in ``indices`` (active at this step)
        and stores the resulting ``(n_active, num_actions)`` slice. Consumers
        in ``compute_flows`` then index POSITIONALLY into the cache, parallel
        to how they already index ``cached_states[step]`` (which has always
        been active-only). Only active-only data is cached, never full masks.
        """
        if not self.cache_enabled:
            return

        # Ensure we have enough slots
        while len(self.cached_states) <= step:
            self.cached_states.append(None)
            self.cached_masks.append(None)
            self.cached_backward_valid_counts.append(None)

        # Convert indices to tensor if needed
        if isinstance(indices, list):
            indices_tensor = torch.tensor(indices, dtype=torch.long, device=self.device)
        else:
            indices_tensor = indices

        # uint8 state compression (see the __init__ comment): single choke
        # point — every sampler funnels its per-step cache through here.
        # PRECONDITION (documented, not runtime-checked — a .max() probe would
        # add a host-sync per sampled step): values must be exactly 0/1, which
        # the eligibility gate in ``_effective_uint8_state_cache`` guarantees
        # by restricting to the flattened-W feature mode.
        if self.cache_states_uint8 and states_tensor.dtype == torch.float32:
            states_tensor = states_tensor.to(torch.uint8)

        self.cached_states[step] = (states_tensor, indices_tensor)
        # Active-only mask cache: (n_active, num_actions) instead of full
        # (B, M, num_actions). Saves memory and avoids the full-shape clone
        # that ran every sampling step.
        if masks.dim() == 3 and indices_tensor.numel() > 0:
            self.cached_masks[step] = masks[indices_tensor[:, 0], indices_tensor[:, 1]]
        elif masks.dim() == 2:
            # Already active-only. The bucketed natural-order path pre-slices
            # full-K masks before caching so this method does not re-index by
            # global (batch, measurement) coordinates.
            self.cached_masks[step] = masks
        else:
            # Empty active set — store an empty placeholder shaped correctly.
            num_actions = masks.shape[-1] if masks.dim() >= 1 else 0
            self.cached_masks[step] = torch.empty(
                (0, num_actions), dtype=masks.dtype, device=masks.device
            )

        # Cache backward valid counts if provided
        if backward_valid_counts is not None:
            self.cached_backward_valid_counts[step] = backward_valid_counts

    # NOTE: the former per-trajectory setters (set_action / set_length /
    # deactivate / batch_set_actions / batch_set_lengths) were removed —
    # nothing in the repo called them, and batch_set_lengths hid a
    # per-row Python loop over a GPU tensor (one host sync per row).
    # All samplers write the underlying tensors directly in batch form.

