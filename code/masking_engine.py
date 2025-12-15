# -*- coding: utf-8 -*-
"""Masking Engine component for GFlowNet.

Handles computation of forward and backward action masks for valid actions.
"""

import torch
from typing import Optional, Tuple
import logging


class MaskingEngine:
    """Computes action masks for forward and backward sampling in GFlowNet."""

    def __init__(self, n_qubits: int, num_actions: int,
                 action_gate_types: torch.Tensor,
                 action_qubit1: torch.Tensor,
                 action_qubit2: torch.Tensor,
                 single_qubit_mask: torch.Tensor,
                 two_qubit_mask: torch.Tensor,
                 terminal_index: int,
                 device: torch.device,
                 debug: bool = False):
        """Initialize masking engine.

        Args:
            n_qubits: Number of qubits
            num_actions: Total number of possible actions
            action_gate_types: Tensor mapping action indices to gate type indices
            action_qubit1: Tensor mapping action indices to first qubit
            action_qubit2: Tensor mapping action indices to second qubit (-1 for single qubit)
            single_qubit_mask: Boolean mask for single qubit actions
            two_qubit_mask: Boolean mask for two qubit actions
            terminal_index: Index of the terminal action
            device: Torch device
            debug: Enable debug logging
        """
        self.n_qubits = n_qubits
        self.num_actions = num_actions
        self.action_gate_types = action_gate_types
        self.action_qubit1 = action_qubit1
        self.action_qubit2 = action_qubit2
        self.single_qubit_mask = single_qubit_mask
        self.two_qubit_mask = two_qubit_mask
        self.terminal_index = terminal_index
        self.device = device
        self.debug = debug

        self._mask_buffer: Optional[torch.Tensor] = None
        self._zero_mask_buffer: Optional[torch.Tensor] = None
        self._inactive_indices: Optional[torch.Tensor] = None
        self._batch_arange: Optional[torch.Tensor] = None
        self._meas_arange: Optional[torch.Tensor] = None
        self._step_arange: Optional[torch.Tensor] = None

        self.single_qubit_indices = self.single_qubit_mask.nonzero(as_tuple=True)[0]
        self.two_qubit_indices = self.two_qubit_mask.nonzero(as_tuple=True)[0]
        self.single_qubit_qubits = self.action_qubit1[self.single_qubit_indices]
        self.two_qubit_q1 = self.action_qubit1[self.two_qubit_indices]
        self.two_qubit_q2 = self.action_qubit2[self.two_qubit_indices]

    def _get_mask_buffer(self, batch_size: int, n_measurements: int) -> torch.Tensor:
        """Return a mask tensor filled with True, reusing allocated storage."""
        shape = (batch_size, n_measurements, self.num_actions)
        if self._mask_buffer is None or self._mask_buffer.shape != shape:
            self._mask_buffer = torch.ones(shape, dtype=torch.bool, device=self.device)
        else:
            self._mask_buffer.fill_(True)
        return self._mask_buffer

    def _get_zero_mask_buffer(self, batch_size: int, n_measurements: int) -> torch.Tensor:
        shape = (batch_size, n_measurements, self.num_actions)
        if self._zero_mask_buffer is None or self._zero_mask_buffer.shape != shape:
            self._zero_mask_buffer = torch.zeros(shape, dtype=torch.bool, device=self.device)
        else:
            self._zero_mask_buffer.zero_()
        return self._zero_mask_buffer

    def _get_arange(self, size: int, cache_attr: str) -> torch.Tensor:
        """Get cached arange tensor on device for broadcasting indices."""
        cached = getattr(self, cache_attr)
        if cached is None or cached.numel() != size:
            cached = torch.arange(size, device=self.device)
            setattr(self, cache_attr, cached)
        return cached

    def compute_action_masks_gpu(self, trajectory_batch,
                                max_depth: Optional[int] = None) -> torch.Tensor:
        """Compute valid action masks entirely on GPU.

        If max_depth is provided, actions that would require starting a new
        layer when the trajectory is already at max_depth are masked out.

        Args:
            trajectory_batch: TrajectoryBatch object with current state
            max_depth: Optional maximum circuit depth

        Returns:
            Boolean tensor of shape (batch_size, n_measurements, num_actions)
        """
        batch_size = trajectory_batch.batch_size
        n_measurements = trajectory_batch.n_measurements

        masks = self._get_mask_buffer(batch_size, n_measurements)

        inactive_mask = ~trajectory_batch.active
        inactive_expanded = inactive_mask.unsqueeze(-1)
        masks = masks & ~inactive_expanded

        terminal_mask = torch.zeros_like(masks)
        terminal_mask[:, :, self.terminal_index] = inactive_mask
        masks = masks | terminal_mask

        depth_limit_mask = None
        at_max_depth_mask = None
        if max_depth is not None:
            depth_limit_mask = trajectory_batch.circuit_depths > max_depth
            at_max_depth_mask = trajectory_batch.circuit_depths == max_depth

        if self.single_qubit_indices.numel() > 0:
            single_qubits = self.single_qubit_qubits

            has_single_gate = trajectory_batch.last_single_qubit_gates[:, :, single_qubits] >= 0

            masks[:, :, self.single_qubit_indices] &= ~has_single_gate

            if depth_limit_mask is not None:
                requires_new_layer = trajectory_batch.current_layer_qubits[:, :, single_qubits]

                depth_limit_expanded = depth_limit_mask.unsqueeze(2).expand(-1, -1, self.single_qubit_indices.size(0))
                at_max_depth_expanded = at_max_depth_mask.unsqueeze(2).expand(-1, -1, self.single_qubit_indices.size(0))

                depth_block = depth_limit_expanded | (at_max_depth_expanded & requires_new_layer)
                masks[:, :, self.single_qubit_indices] &= ~depth_block

        if self.two_qubit_indices.numel() > 0:
            two_qubit_q1 = self.two_qubit_q1
            two_qubit_q2 = self.two_qubit_q2

            q1_last_steps = trajectory_batch.qubit_last_use_step[:, :, two_qubit_q1]
            q2_last_steps = trajectory_batch.qubit_last_use_step[:, :, two_qubit_q2]

            both_used = (q1_last_steps >= 0) & (q2_last_steps >= 0)
            same_step = q1_last_steps == q2_last_steps
            valid_check = both_used & same_step & trajectory_batch.active.unsqueeze(-1)

            batch_indices = self._get_arange(batch_size, '_batch_arange').view(-1, 1, 1)
            meas_indices = self._get_arange(n_measurements, '_meas_arange').view(1, -1, 1)
            clamped_steps = q1_last_steps.clamp(min=0, max=trajectory_batch.max_length - 1)

            last_actions = trajectory_batch.actions[batch_indices, meas_indices, clamped_steps]
            last_gate_types = self.action_gate_types[last_actions]
            current_gate_types = self.action_gate_types[self.two_qubit_indices].view(1, 1, -1)
            gate_type_matches = last_gate_types == current_gate_types

            last_q1 = self.action_qubit1[last_actions]
            last_q2 = self.action_qubit2[last_actions]
            curr_q1 = two_qubit_q1.view(1, 1, -1).expand(batch_size, n_measurements, -1)
            curr_q2 = two_qubit_q2.view(1, 1, -1).expand(batch_size, n_measurements, -1)
            qubit_matches = ((last_q1 == curr_q1) & (last_q2 == curr_q2)) | ((last_q1 == curr_q2) & (last_q2 == curr_q1))

            block_mask = valid_check & gate_type_matches & qubit_matches
            masks[:, :, self.two_qubit_indices] &= ~block_mask

            if depth_limit_mask is not None:
                requires_new_layer = trajectory_batch.current_layer_qubits[:, :, two_qubit_q1] | trajectory_batch.current_layer_qubits[:, :, two_qubit_q2]
                depth_limit_expanded = depth_limit_mask.unsqueeze(2).expand(-1, -1, self.two_qubit_indices.size(0))
                at_max_depth_expanded = at_max_depth_mask.unsqueeze(2).expand(-1, -1, self.two_qubit_indices.size(0))
                depth_block = depth_limit_expanded | (at_max_depth_expanded & requires_new_layer)
                masks[:, :, self.two_qubit_indices] &= ~depth_block

        masks[:, :, self.terminal_index] = True

        return masks

    def compute_backward_masks_gpu(self, trajectory_batch,
                                  current_step: int,
                                  forward_masks: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Fully vectorized computation of backward masks based on exposed gates.

        Args:
            trajectory_batch: TrajectoryBatch object with current state
            current_step: Current step in backward traversal
            forward_masks: Optional pre-computed forward masks to use as fallback

        Returns:
            Boolean tensor of shape (batch_size, n_measurements, num_actions)
        """
        batch_size = trajectory_batch.batch_size
        n_measurements = trajectory_batch.n_measurements
        device = trajectory_batch.device

        if self.debug:
            logging.debug(f"\nDEBUG compute_backward_masks_gpu:")
            logging.debug(f"  Current step: {current_step}")
            logging.debug(f"  Batch size: {batch_size}, n_measurements: {n_measurements}")

        masks = self._get_zero_mask_buffer(batch_size, n_measurements)

        if current_step == 0:
            if forward_masks is None:
                forward_masks = self.compute_action_masks_gpu(trajectory_batch)
            forward_masks_copy = forward_masks.clone()
            forward_masks_copy[..., self.terminal_index] = False
            return forward_masks_copy

        single_qubits = self.single_qubit_qubits
        last_use_single = trajectory_batch.qubit_last_use_step[:, :, single_qubits]

        masks[:, :, self.single_qubit_indices] = self._check_exposed(
            last_use_single, trajectory_batch, current_step, self.single_qubit_indices
        )

        two_qubits1 = self.two_qubit_q1
        two_qubits2 = self.two_qubit_q2

        last_use1 = trajectory_batch.qubit_last_use_step[:, :, two_qubits1]
        last_use2 = trajectory_batch.qubit_last_use_step[:, :, two_qubits2]
        max_last_use = torch.maximum(last_use1, last_use2)

        masks[:, :, self.two_qubit_indices] = self._check_exposed(
            max_last_use, trajectory_batch, current_step, self.two_qubit_indices
        )

        masks[:, :, self.terminal_index] = False

        exposed_count = masks.sum(dim=-1, keepdim=True)
        has_exposed = exposed_count > 0

        needs_fallback = ~has_exposed
        needs_fallback_expanded = needs_fallback.expand_as(masks)

        fallback_source = (
            forward_masks
            if forward_masks is not None
            else self.compute_action_masks_gpu(trajectory_batch)
        )

        masks = torch.where(needs_fallback_expanded, fallback_source, masks)
        masks[..., self.terminal_index] = False

        return masks

    def _check_exposed(self, last_use_steps, trajectory_batch, current_step, action_indices):
        """Check if actions are exposed - fully vectorized using broadcasting."""
        batch_size, n_measurements, n_actions = last_use_steps.shape
        device = last_use_steps.device

        max_steps = trajectory_batch.actions.shape[2]
        actual_steps = min(current_step, max_steps)
        all_actions = trajectory_batch.actions[:, :, :actual_steps]

        steps = self._get_arange(actual_steps, '_step_arange') if actual_steps > 0 else torch.empty(0, device=device, dtype=torch.long)

        action_indices_expanded = action_indices.view(1, 1, 1, -1)
        all_actions_expanded = all_actions.unsqueeze(-1)

        action_matches = (all_actions_expanded == action_indices_expanded)

        steps_expanded = steps.view(1, 1, -1, 1)
        last_use_expanded = last_use_steps.unsqueeze(2)

        step_matches = (steps_expanded == last_use_expanded)

        step_indices = torch.arange(actual_steps, device=device).view(1, 1, -1, 1)
        lengths_expanded = trajectory_batch.lengths.unsqueeze(-1).unsqueeze(-1)
        valid_steps = step_indices < lengths_expanded

        exposed = action_matches & step_matches & valid_steps.expand_as(action_matches)

        return exposed.any(dim=2)

    def compute_forward_masks(self, trajectory_batch, batched_tableau=None, indices=None, step=None, max_depth: Optional[int] = None) -> torch.Tensor:
        """Alias for compute_action_masks_gpu for backward compatibility.

        Args:
            trajectory_batch: TrajectoryBatch object with current state
            batched_tableau: Ignored for compatibility
            indices: Ignored for compatibility
            step: Ignored for compatibility
            max_depth: Optional maximum circuit depth

        Returns:
            Boolean tensor of shape (batch_size, n_measurements, num_actions)
        """
        return self.compute_action_masks_gpu(trajectory_batch, max_depth)