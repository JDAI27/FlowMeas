# -*- coding: utf-8 -*-
"""
Masking Engine for GFlowNet action validity.

This module computes forward and backward action masks to ensure:
1. Forward masks: Valid actions from current state (circuit constraints)
2. Backward masks: Valid parent actions (for trajectory balance)

Masking Rules:
==============
Forward (P_F):
    - Always allow terminal action (end circuit)
    - Respect max_depth constraint
    - Optional: enforce gate connectivity or other physical constraints

Backward (P_B):
    - Mask to parent state that led to current state
    - Required for trajectory balance objective

Tensor Shapes:
    forward_mask: (B, C, n_actions) - 1 for valid actions, 0 for invalid
    backward_mask: (B, C, n_actions-1) - excludes terminal (can't undo terminal)
"""
from __future__ import annotations

import torch
from typing import Optional, Tuple
import logging

# Fused active-mask + forward/backward valid-count CuPy kernel.
# Importing the module is safe on CPU-only hosts (it only ``import torch``
# at module level; the ``cupy`` import is deferred until the kernel is
# actually used).
#
# The import must work in BOTH execution modes the repo supports:
#   * ``python3 code/run_config.py...``  — ``code/`` is on ``sys.path``, so
#     ``masking_engine`` is a top-level module and ``measurement_adapter``
#     resolves as a sibling top-level package.
#   * ``python3 -m code.run_config...`` — ``masking_engine`` is
#     ``code.masking_engine``; the sibling import needs to be relative.
#
# Both spellings must be tried, and the ``except`` must stay narrow: a broad
# ``except Exception`` swallows the ``ImportError`` in package mode and silently
# disables the fused path.
try:
    from .measurement_adapter import mask_counts_kernel as _mask_counts_kernel
except ImportError:
    try:
        from measurement_adapter import mask_counts_kernel as _mask_counts_kernel
    except ImportError:  # pragma: no cover - kernel module truly absent
        _mask_counts_kernel = None


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

        # Cache frequently reused tensors to avoid per-call allocations
        self._mask_buffer: Optional[torch.Tensor] = None
        self._zero_mask_buffer: Optional[torch.Tensor] = None
        self._inactive_indices: Optional[torch.Tensor] = None
        self._batch_arange: Optional[torch.Tensor] = None
        self._meas_arange: Optional[torch.Tensor] = None
        self._step_arange: Optional[torch.Tensor] = None

        # Pre-compute static action index lists
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

        # Initialize all masks to True
        masks = self._get_mask_buffer(batch_size, n_measurements)

        # PHASE B2a FIX: Handle inactive trajectories without boolean indexing (zero-sync)
        inactive_mask = ~trajectory_batch.active

        # Clear all actions for inactive trajectories
        inactive_expanded = inactive_mask.unsqueeze(-1)
        masks = masks & ~inactive_expanded

        # Set terminal action for inactive using torch.where (avoids IndexError)
        terminal_mask = torch.zeros_like(masks)
        terminal_mask[:, :, self.terminal_index] = inactive_mask
        masks = masks | terminal_mask

        # For active trajectories, compute constraints
        depth_limit_mask = None
        at_max_depth_mask = None
        if max_depth is not None:
            depth_limit_mask = trajectory_batch.circuit_depths > max_depth
            at_max_depth_mask = trajectory_batch.circuit_depths == max_depth

        # PHASE B2a: FULLY VECTORIZED SINGLE-QUBIT GATE PROCESSING (no sync)
        # Process all masks unconditionally for true zero-sync
        if self.single_qubit_indices.numel() > 0:  # This is OK - computed at init time
            single_qubits = self.single_qubit_qubits

            has_single_gate = trajectory_batch.last_single_qubit_gates[:, :, single_qubits] >= 0

            masks[:, :, self.single_qubit_indices] &= ~has_single_gate

            # 1q gates never consume the depth
            # budget, so no depth-driven blocking applies to them. Only the
            # 1q-after-1q `has_single_gate` rule remains.
            if depth_limit_mask is not None:
                depth_limit_expanded = depth_limit_mask.unsqueeze(2).expand(
                    -1, -1, self.single_qubit_indices.size(0)
                )
                masks[:, :, self.single_qubit_indices] &= ~depth_limit_expanded

        # PHASE B2a: FULLY VECTORIZED TWO-QUBIT GATE PROCESSING (no sync)
        # Process all masks unconditionally for true zero-sync
        if self.two_qubit_indices.numel() > 0:  # This is OK - computed at init time
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

            last_actions = trajectory_batch.actions_time_major[
                clamped_steps, batch_indices, meas_indices
            ]
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
                # A 2q gate opens a new layer
                # when its qubits conflict with the current 2q layer OR when
                # no 2q layer has been opened yet (depth==0 bumps to depth==1
                # on the first 2q gate). At the depth budget we block any 2q
                # gate that would open a new layer.
                depth_zero_expanded = (
                    trajectory_batch.circuit_depths == 0
                ).unsqueeze(2).expand(-1, -1, self.two_qubit_indices.size(0))
                requires_new_layer = (
                    trajectory_batch.current_layer_qubits[:, :, two_qubit_q1]
                    | trajectory_batch.current_layer_qubits[:, :, two_qubit_q2]
                    | depth_zero_expanded
                )
                depth_limit_expanded = depth_limit_mask.unsqueeze(2).expand(-1, -1, self.two_qubit_indices.size(0))
                at_max_depth_expanded = at_max_depth_mask.unsqueeze(2).expand(-1, -1, self.two_qubit_indices.size(0))
                depth_block = depth_limit_expanded | (at_max_depth_expanded & requires_new_layer)
                masks[:, :, self.two_qubit_indices] &= ~depth_block

        # Terminal action is always valid
        masks[:, :, self.terminal_index] = True

        return masks

    def compute_action_masks_active_gpu(self, trajectory_batch,
                                        indices: torch.Tensor,
                                        max_depth: Optional[int] = None) -> torch.Tensor:
        """Compute forward masks only for the active rows in ``indices``.

        This avoids materializing the full ``(B, M, A)`` mask when the caller
        immediately slices it down to active rows for policy sampling.

        note — INTENTIONAL duplication with ``compute_action_masks_gpu``:
        the single-/two-qubit validity predicates below mirror that method's
        logic, but the two operate on different layouts (this one on a gathered
        ``(n_active, A)`` active-row subset via ``indices``; the other on the full
        ``(B, M, A)`` grid). This active path is the mask-producing PyTorch
        fallback for the fused CuPy mask-counts kernel and is bit-for-bit
        parity with it, and with the full-grid layout. Both
        paths are hand-vectorized for zero host synchronization, so a shared
        helper was deliberately not extracted: it could not be made
        simultaneously zero-overhead, operation-order-identical, and
        layout-agnostic without risking that vectorization. Keep them in sync
        by hand.
        """
        if indices.numel() == 0:
            return torch.empty(
                (0, self.num_actions), dtype=torch.bool, device=self.device
            )

        batch_idx = indices[:, 0]
        meas_idx = indices[:, 1]
        n_active = indices.shape[0]
        masks = torch.ones(
            (n_active, self.num_actions), dtype=torch.bool, device=self.device
        )

        active = trajectory_batch.active[batch_idx, meas_idx]
        masks &= active.unsqueeze(-1)
        masks[:, self.terminal_index] = True

        at_max_depth = None
        depth_limit = None
        if max_depth is not None:
            depths = trajectory_batch.circuit_depths[batch_idx, meas_idx]
            depth_limit = depths > max_depth
            at_max_depth = depths == max_depth

        if self.single_qubit_indices.numel() > 0:
            single_qubits = self.single_qubit_qubits
            has_single_gate = (
                trajectory_batch.last_single_qubit_gates[
                    batch_idx.unsqueeze(1),
                    meas_idx.unsqueeze(1),
                    single_qubits.view(1, -1),
                ] >= 0
            )
            single_valid = ~has_single_gate

            # 1q gates never consume the depth
            # budget; only the defensive `depth > max_depth` guard remains.
            if depth_limit is not None:
                single_valid &= ~depth_limit.unsqueeze(1)

            masks[:, self.single_qubit_indices] = single_valid & active.unsqueeze(1)

        if self.two_qubit_indices.numel() > 0:
            two_q1 = self.two_qubit_q1
            two_q2 = self.two_qubit_q2
            q1_last_steps = trajectory_batch.qubit_last_use_step[
                batch_idx.unsqueeze(1), meas_idx.unsqueeze(1), two_q1.view(1, -1)
            ]
            q2_last_steps = trajectory_batch.qubit_last_use_step[
                batch_idx.unsqueeze(1), meas_idx.unsqueeze(1), two_q2.view(1, -1)
            ]

            both_used = (q1_last_steps >= 0) & (q2_last_steps >= 0)
            same_step = q1_last_steps == q2_last_steps
            valid_check = both_used & same_step & active.unsqueeze(1)

            clamped_steps = q1_last_steps.clamp(
                min=0, max=trajectory_batch.max_length - 1
            )
            last_actions = trajectory_batch.actions_time_major[
                clamped_steps,
                batch_idx.unsqueeze(1),
                meas_idx.unsqueeze(1),
            ]
            last_gate_types = self.action_gate_types[last_actions]
            current_gate_types = self.action_gate_types[self.two_qubit_indices].view(1, -1)
            gate_type_matches = last_gate_types == current_gate_types

            last_q1 = self.action_qubit1[last_actions]
            last_q2 = self.action_qubit2[last_actions]
            curr_q1 = two_q1.view(1, -1).expand(n_active, -1)
            curr_q2 = two_q2.view(1, -1).expand(n_active, -1)
            qubit_matches = (
                ((last_q1 == curr_q1) & (last_q2 == curr_q2))
                | ((last_q1 == curr_q2) & (last_q2 == curr_q1))
            )

            two_valid = ~(valid_check & gate_type_matches & qubit_matches)

            if depth_limit is not None:
                # The first 2q gate (depth==0)
                # always opens a new layer; subsequent 2q gates open a new
                # layer only when their qubits conflict with the current
                # layer. Block at the budget when a new layer would be needed.
                depths_active = trajectory_batch.circuit_depths[batch_idx, meas_idx]
                requires_new_layer = (
                    trajectory_batch.current_layer_qubits[
                        batch_idx.unsqueeze(1),
                        meas_idx.unsqueeze(1),
                        two_q1.view(1, -1),
                    ]
                    | trajectory_batch.current_layer_qubits[
                        batch_idx.unsqueeze(1),
                        meas_idx.unsqueeze(1),
                        two_q2.view(1, -1),
                    ]
                    | (depths_active == 0).unsqueeze(1)
                )
                depth_block = depth_limit.unsqueeze(1) | (
                    at_max_depth.unsqueeze(1) & requires_new_layer
                )
                two_valid &= ~depth_block

            masks[:, self.two_qubit_indices] = two_valid & active.unsqueeze(1)

        return masks

    def compute_forward_valid_counts_gpu(self, trajectory_batch,
                                         max_depth: Optional[int] = None,
                                         include_terminal: bool = False) -> torch.Tensor:
        """Count valid forward actions without materializing a full mask."""
        batch_size = trajectory_batch.batch_size
        n_measurements = trajectory_batch.n_measurements
        counts = torch.zeros(
            (batch_size, n_measurements), dtype=torch.long, device=self.device
        )
        active = trajectory_batch.active

        depth_limit = None
        at_max_depth = None
        if max_depth is not None:
            depth_limit = trajectory_batch.circuit_depths > max_depth
            at_max_depth = trajectory_batch.circuit_depths == max_depth

        if self.single_qubit_indices.numel() > 0:
            single_qubits = self.single_qubit_qubits
            single_valid = trajectory_batch.last_single_qubit_gates[:, :, single_qubits] < 0

            # 1q gates never consume the depth
            # budget; only the defensive `depth > max_depth` guard remains.
            if depth_limit is not None:
                single_valid &= ~depth_limit.unsqueeze(2)

            counts += (single_valid & active.unsqueeze(-1)).sum(dim=-1)

        if self.two_qubit_indices.numel() > 0:
            two_q1 = self.two_qubit_q1
            two_q2 = self.two_qubit_q2
            q1_last_steps = trajectory_batch.qubit_last_use_step[:, :, two_q1]
            q2_last_steps = trajectory_batch.qubit_last_use_step[:, :, two_q2]

            both_used = (q1_last_steps >= 0) & (q2_last_steps >= 0)
            same_step = q1_last_steps == q2_last_steps
            valid_check = both_used & same_step & active.unsqueeze(-1)

            batch_indices = self._get_arange(batch_size, '_batch_arange').view(-1, 1, 1)
            meas_indices = self._get_arange(n_measurements, '_meas_arange').view(1, -1, 1)
            clamped_steps = q1_last_steps.clamp(min=0, max=trajectory_batch.max_length - 1)

            last_actions = trajectory_batch.actions_time_major[
                clamped_steps, batch_indices, meas_indices
            ]
            last_gate_types = self.action_gate_types[last_actions]
            current_gate_types = self.action_gate_types[self.two_qubit_indices].view(1, 1, -1)
            gate_type_matches = last_gate_types == current_gate_types

            last_q1 = self.action_qubit1[last_actions]
            last_q2 = self.action_qubit2[last_actions]
            curr_q1 = two_q1.view(1, 1, -1).expand(batch_size, n_measurements, -1)
            curr_q2 = two_q2.view(1, 1, -1).expand(batch_size, n_measurements, -1)
            qubit_matches = (
                ((last_q1 == curr_q1) & (last_q2 == curr_q2))
                | ((last_q1 == curr_q2) & (last_q2 == curr_q1))
            )

            two_valid = ~(valid_check & gate_type_matches & qubit_matches)

            if depth_limit is not None:
                # Include the depth==0 case in
                # "would open a new layer" so the first 2q gate is blocked
                # at max_depth==0.
                depth_zero_expanded = (
                    trajectory_batch.circuit_depths == 0
                ).unsqueeze(2).expand(-1, -1, self.two_qubit_indices.size(0))
                requires_new_layer = (
                    trajectory_batch.current_layer_qubits[:, :, two_q1]
                    | trajectory_batch.current_layer_qubits[:, :, two_q2]
                    | depth_zero_expanded
                )
                depth_block = depth_limit.unsqueeze(2) | (
                    at_max_depth.unsqueeze(2) & requires_new_layer
                )
                two_valid &= ~depth_block

            counts += (two_valid & active.unsqueeze(-1)).sum(dim=-1)

        if include_terminal:
            counts += 1

        return counts

    def _normalize_current_step(self, current_step: int | torch.Tensor) -> int | torch.Tensor:
        """Validate backward-count step inputs consistently across fused/fallback paths."""
        if not torch.is_tensor(current_step):
            return int(current_step)
        if current_step.dim() != 0:
            raise ValueError("current_step tensor must be 0-dimensional")
        if current_step.dtype != torch.long:
            raise ValueError("current_step tensor must have dtype torch.long")

        engine_device = self.device
        if not isinstance(engine_device, torch.device):
            engine_device = torch.device(engine_device)
        if current_step.device.type != engine_device.type:
            raise ValueError("current_step tensor must be on the trajectory device")
        if current_step.device.type == "cuda":
            step_device_index = current_step.device.index
            engine_device_index = engine_device.index
            if step_device_index is None:
                step_device_index = torch.cuda.current_device()
            if engine_device_index is None:
                engine_device_index = torch.cuda.current_device()
            if step_device_index != engine_device_index:
                raise ValueError("current_step tensor must be on the trajectory device")
        return current_step

    def compute_backward_valid_counts_gpu(self, trajectory_batch,
                                          current_step: int | torch.Tensor,
                                          forward_valid_counts: Optional[torch.Tensor] = None,
                                          max_depth: Optional[int] = None) -> torch.Tensor:
        """Count valid backward actions without constructing backward masks.

        This matches ``compute_backward_masks_gpu.sum(-1)`` with terminal
        excluded, but avoids the expensive history-broadcast mask tensor.
        If ``forward_valid_counts`` is provided by the caller, it must also
        exclude terminal actions.
        """
        if forward_valid_counts is None:
            forward_valid_counts = self.compute_forward_valid_counts_gpu(
                trajectory_batch, max_depth=max_depth, include_terminal=False
            )

        current_step = self._normalize_current_step(current_step)
        # For tensor ``current_step`` keep the value on device: the vectorized
        # predicate below naturally yields no exposed actions when step == 0.
        if not torch.is_tensor(current_step) and current_step == 0:
            return forward_valid_counts

        batch_size = trajectory_batch.batch_size
        n_measurements = trajectory_batch.n_measurements
        lengths = trajectory_batch.lengths
        exposed_count = torch.zeros(
            (batch_size, n_measurements), dtype=torch.long, device=self.device
        )

        batch_indices = self._get_arange(batch_size, '_batch_arange').view(-1, 1, 1)
        meas_indices = self._get_arange(n_measurements, '_meas_arange').view(1, -1, 1)

        if self.single_qubit_indices.numel() > 0:
            last_use = trajectory_batch.qubit_last_use_step[:, :, self.single_qubit_qubits]
            valid_last = (
                (last_use >= 0)
                & (last_use < current_step)
                & (last_use < lengths.unsqueeze(-1))
            )
            clamped_steps = last_use.clamp(min=0, max=trajectory_batch.max_length - 1)
            last_actions = trajectory_batch.actions_time_major[
                clamped_steps, batch_indices, meas_indices
            ]
            action_matches = last_actions == self.single_qubit_indices.view(1, 1, -1)
            exposed_count += (valid_last & action_matches).sum(dim=-1)

        if self.two_qubit_indices.numel() > 0:
            last1 = trajectory_batch.qubit_last_use_step[:, :, self.two_qubit_q1]
            last2 = trajectory_batch.qubit_last_use_step[:, :, self.two_qubit_q2]
            last_use = torch.maximum(last1, last2)
            valid_last = (
                (last1 == last2)
                & (last_use >= 0)
                & (last_use < current_step)
                & (last_use < lengths.unsqueeze(-1))
            )
            clamped_steps = last_use.clamp(min=0, max=trajectory_batch.max_length - 1)
            last_actions = trajectory_batch.actions_time_major[
                clamped_steps, batch_indices, meas_indices
            ]
            action_matches = last_actions == self.two_qubit_indices.view(1, 1, -1)
            exposed_count += (valid_last & action_matches).sum(dim=-1)

        return torch.where(exposed_count > 0, exposed_count, forward_valid_counts)

    def compute_masks_and_counts_fused(
        self,
        trajectory_batch,
        indices: torch.Tensor,
        *,
        current_step: int | torch.Tensor,
        max_depth: Optional[int] = None,
        compute_backward: bool = True,
        use_fused_kernel: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Compute active mask + forward & backward valid counts in one pass.

        Combines ``compute_action_masks_active_gpu``,
        ``compute_forward_valid_counts_gpu``, and
        ``compute_backward_valid_counts_gpu`` into a single fused
        CuPy kernel launch when available, with a PyTorch fallback that
        calls the existing three functions in sequence.

        Args:
            trajectory_batch: TrajectoryBatch with current state.
            indices: ``(n_active, 2)`` int64 [batch_idx, meas_idx] rows for
                which the active mask is materialized. **Each ``(batch_idx,
                meas_idx)`` must be unique**: the fused path scatters one
                output-row id per flat ``(b, m)`` slot, so a duplicate leaves
                the earlier row all-False. The only production producer
                (``to_flat_tensors_active_only``) returns unique rows; callers
                that synthesize ``indices`` must pre-dedup or pass
                ``use_fused_kernel=False``.
            current_step: Backward "current_step". At 0 the backward count
                equals the forward count. The fused path also accepts a 0-dim
                ``torch.long`` CUDA tensor, read from device memory at
                launch/replay time.
            max_depth: Optional depth cap, applied to the active mask ONLY —
                the counts deliberately ignore it, matching the legacy
                invocation. So ``forward_valid_counts`` is the count of
                structurally valid non-terminal entries (what the
                backward-mask fallback needs), NOT the policy-visible count;
                for that, count ``True`` entries in the returned mask.
            compute_backward: ``False`` returns ``(mask, None, None)`` and
                routes to ``compute_action_masks_active_gpu``, so work scales
                as ``n_active`` rather than ``B*M``. For a forward-only count,
                call ``compute_forward_valid_counts_gpu`` directly.
            use_fused_kernel: ``False`` forces the PyTorch fallback; the fused
                path also falls back silently without CUDA / CuPy.

        Returns:
            ``(active_mask, forward_counts, backward_counts)``.
            ``active_mask`` is ``(n_active, num_actions)`` bool; the counts are
            ``(B, M)`` int64, or ``None`` when ``compute_backward=False``.
        """
        # Mask-only shortcut. With ``compute_backward=False`` the counts are
        # discarded, and the fused kernel would still run one thread per
        # ``(B, M)`` row plus the host-side ``active_lookup`` scatter just to
        # fill a buffer that is dropped. Late dynamic-active layers can have
        # ``n_active << B*M``, so route straight to
        # ``compute_action_masks_active_gpu``, which touches only the rows in
        # ``indices``. Bit-identical on the mask by construction.
        if not compute_backward:
            active_mask = self.compute_action_masks_active_gpu(
                trajectory_batch, indices, max_depth=max_depth
            )
            return active_mask, None, None

        current_step = self._normalize_current_step(current_step)
        if (
            use_fused_kernel
            and _mask_counts_kernel is not None
            and trajectory_batch.device.type == "cuda"
        ):
            result = _mask_counts_kernel.compute_mask_counts_fused(
                trajectory_batch,
                indices,
                masking_engine=self,
                current_step=current_step,
                max_depth=max_depth,
                compute_backward=compute_backward,
                use_fused_kernel=True,
            )
            if result is not None:
                return result

        # PyTorch fallback. Runs the legacy functions with the same
        # max_depth asymmetry the fused kernel implements: mask honours
        # ``max_depth``; counts use ``max_depth=None``. Semantics identical
        # to the fused path by construction.
        active_mask = self.compute_action_masks_active_gpu(
            trajectory_batch, indices, max_depth=max_depth
        )
        forward_counts = self.compute_forward_valid_counts_gpu(
            trajectory_batch, max_depth=None, include_terminal=False
        )
        backward_counts = self.compute_backward_valid_counts_gpu(
            trajectory_batch,
            current_step=current_step,
            forward_valid_counts=forward_counts,
            max_depth=None,
        )
        return active_mask, forward_counts, backward_counts

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

        # Initialize masks
        masks = self._get_zero_mask_buffer(batch_size, n_measurements)

        # Early exit for step 0
        if current_step == 0:
            # Fallback to forward masks
            if forward_masks is None:
                # Only compute if not provided
                forward_masks = self.compute_action_masks_gpu(trajectory_batch)
            forward_masks_copy = forward_masks.clone()
            forward_masks_copy[..., self.terminal_index] = False
            return forward_masks_copy

        # Vectorized computation of max last use for all actions
        # PHASE B2b: Process single qubit actions unconditionally (zero-sync)
        # Note: Empty indices will naturally produce empty results
        single_qubits = self.single_qubit_qubits
        # Advanced indexing to gather last use for single qubit actions
        # Shape: (batch, meas, n_single_actions)
        last_use_single = trajectory_batch.qubit_last_use_step[:, :, single_qubits]

        # Check exposed using vectorized operations
        masks[:, :, self.single_qubit_indices] = self._check_exposed(
            last_use_single, trajectory_batch, current_step, self.single_qubit_indices
        )

        # PHASE B2b: Process two qubit actions unconditionally (zero-sync)
        # Note: Empty indices will naturally produce empty results
        two_qubits1 = self.two_qubit_q1
        two_qubits2 = self.two_qubit_q2

        # Gather and take maximum
        last_use1 = trajectory_batch.qubit_last_use_step[:, :, two_qubits1]
        last_use2 = trajectory_batch.qubit_last_use_step[:, :, two_qubits2]
        max_last_use = torch.maximum(last_use1, last_use2)

        masks[:, :, self.two_qubit_indices] = self._check_exposed(
            max_last_use, trajectory_batch, current_step, self.two_qubit_indices
        )

        # Ensure terminal is never exposed
        masks[:, :, self.terminal_index] = False

        # CRITICAL FIX: Zero-sync fallback for trajectories with no exposed gates
        # When nothing is exposed (common at start), fall back to forward masks

        # Check which trajectories have ANY exposed gates (without .any() sync)
        # Sum across action dimension and check if > 0
        # This is mathematically equivalent but avoids the .any() reduction
        exposed_count = masks.sum(dim=-1, keepdim=True)  # (batch, meas, 1)
        has_exposed = exposed_count > 0  # (batch, meas, 1)

        # ZERO-SYNC: determine fallback contribution without introducing sync points
        needs_fallback = ~has_exposed  # (batch, meas, 1)
        needs_fallback_expanded = needs_fallback.expand_as(masks)

        # Use caller-provided forward masks when available; otherwise compute once here.
        # This keeps the hot path (callers that already have forward masks) zero-sync while
        # maintaining backwards compatibility for legacy callers.
        fallback_source = (
            forward_masks
            if forward_masks is not None
            else self.compute_action_masks_gpu(trajectory_batch)
        )

        # Blend exposed gates with fallback in a single tensor op. We set the terminal column
        # to False afterwards to ensure it is never exposed, avoiding an extra clone.
        masks = torch.where(needs_fallback_expanded, fallback_source, masks)
        masks[..., self.terminal_index] = False

        return masks

    def _check_exposed(self, last_use_steps, trajectory_batch, current_step, action_indices):
        """Helper to check if actions are exposed - fully vectorized (using broadcasting).

        Args:
            last_use_steps: Tensor of shape (batch, meas, n_actions) with last use steps
            trajectory_batch: TrajectoryBatch object
            current_step: Current step in backward traversal
            action_indices: Indices of actions being checked

        Returns:
            Boolean tensor indicating which actions are exposed
        """
        batch_size, n_measurements, n_actions = last_use_steps.shape
        device = last_use_steps.device

        # Get all actions up to current_step (clamp to actual tensor size)
        max_steps = trajectory_batch.actions.shape[2]
        actual_steps = min(current_step, max_steps)
        all_actions = trajectory_batch.actions_time_major[:actual_steps].permute(1, 2, 0)

        # Create tensors for steps that match actual_steps
        steps = self._get_arange(actual_steps, '_step_arange') if actual_steps > 0 else torch.empty(0, device=device, dtype=torch.long)

        # Expand dimensions for broadcasting (eliminates loops over b/m/step)
        # action_indices: (n_actions,) -> (1, 1, 1, n_actions)
        action_indices_expanded = action_indices.view(1, 1, 1, -1)
        # all_actions: (batch, meas, current_step) -> (batch, meas, current_step, 1)
        all_actions_expanded = all_actions.unsqueeze(-1)

        # Check where actions match
        action_matches = (all_actions_expanded == action_indices_expanded)  # (batch, meas, current_step, n_actions)

        # Check if step matches last use
        # steps: (current_step,) -> (1, 1, current_step, 1)
        steps_expanded = steps.view(1, 1, -1, 1)
        # last_use_steps: (batch, meas, n_actions) -> (batch, meas, 1, n_actions)
        last_use_expanded = last_use_steps.unsqueeze(2)

        step_matches = (steps_expanded == last_use_expanded)  # (batch, meas, current_step, n_actions)

        # Check valid steps
        step_indices = torch.arange(actual_steps, device=device).view(1, 1, -1, 1)
        lengths_expanded = trajectory_batch.lengths.unsqueeze(-1).unsqueeze(-1)  # (batch, meas, 1, 1)
        valid_steps = step_indices < lengths_expanded  # (batch, meas, current_step, 1)

        # Combine all conditions with broadcasting
        exposed = action_matches & step_matches & valid_steps.expand_as(action_matches)

        # Any step where the action is exposed (reduction eliminates step dim)
        return exposed.any(dim=2)  # (batch, meas, n_actions)

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
