# -*- coding: utf-8 -*-
"""Flow / loss computation mixin for ``GFlowNet``.

Split out of ``GFNs.py``. Owns mask delegation, ``compute_flows``,
``compute_flows_cached`` (chunked cached-flow path incl. activation
checkpointing and circuit dedup), and ``compute_loss``.

This is a mixin: methods are composed into ``GFlowNet`` (see
``gfn_core``) with identical MRO semantics — method bodies are verbatim
from the original module, so numerical behavior and kernel launch counts
are unchanged.
"""

import logging
import os
import torch
import torch.nn.functional as F
import torch.utils.checkpoint as _torch_checkpoint
from typing import Tuple, Dict, Optional

try:
    from .gfn_runtime import (
        _NULL_RECORD_CONTEXT,
        _resolve_device,
    )
    from .gfn_trajectory import (
        TrajectoryBatch,
    )
except ImportError:  # pragma: no cover - direct-execution mode
    from gfn_runtime import (
        _NULL_RECORD_CONTEXT,
        _resolve_device,
    )
    from gfn_trajectory import (
        TrajectoryBatch,
    )


class GFlowNetFlowsMixin:
    """Flow / loss methods of ``GFlowNet`` (split from ``GFNs.py``)."""

    def compute_action_masks_gpu(self, trajectory_batch: TrajectoryBatch,
                                 max_depth: Optional[int] = None) -> torch.Tensor:
        """Compute valid action masks entirely on GPU using MaskingEngine.

        If ``max_depth`` is provided, actions that would require starting a new
        layer when the trajectory is already at ``max_depth`` are masked out.
        """
        return self.masking_engine.compute_action_masks_gpu(trajectory_batch, max_depth)

    def compute_action_masks_active_gpu(self, trajectory_batch: TrajectoryBatch,
                                        indices: torch.Tensor,
                                        max_depth: Optional[int] = None) -> torch.Tensor:
        """Compute valid action masks only for the active rows in ``indices``."""
        return self.masking_engine.compute_action_masks_active_gpu(
            trajectory_batch, indices, max_depth
        )

    def compute_forward_valid_counts_gpu(self, trajectory_batch: TrajectoryBatch,
                                         max_depth: Optional[int] = None,
                                         include_terminal: bool = False) -> torch.Tensor:
        """Count valid forward actions without materializing full masks."""
        return self.masking_engine.compute_forward_valid_counts_gpu(
            trajectory_batch, max_depth=max_depth, include_terminal=include_terminal
        )
    
    def compute_backward_masks_gpu(self, trajectory_batch: TrajectoryBatch,
                                             current_step: int,
                                             forward_masks: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Fully vectorized computation of backward masks using MaskingEngine."""
        return self.masking_engine.compute_backward_masks_gpu(
            trajectory_batch, current_step, forward_masks)

    def compute_backward_valid_counts_gpu(self, trajectory_batch: TrajectoryBatch,
                                          current_step: int,
                                          forward_valid_counts: Optional[torch.Tensor] = None,
                                          max_depth: Optional[int] = None) -> torch.Tensor:
        """Count valid backward actions without materializing backward masks."""
        return self.masking_engine.compute_backward_valid_counts_gpu(
            trajectory_batch,
            current_step=current_step,
            forward_valid_counts=forward_valid_counts,
            max_depth=max_depth,
        )

    def _fresh_flow_tensors(
        self,
        trajectory_batch: TrajectoryBatch,
        batch_size: int,
        n_measurements: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return zeroed flow tensors, reusing compatible batch-owned storage."""
        shape = (batch_size, n_measurements)
        forward_buffers = getattr(trajectory_batch, "_forward_flows_buffers", None)
        backward_buffers = getattr(trajectory_batch, "_backward_flows_buffers", None)
        idx = getattr(trajectory_batch, "_flow_buffer_idx", 0)

        # Devices must compare with index resolution: ``torch.device('cuda')`` and
        # ``torch.device('cuda:0')`` are the same device but compare unequal.
        # Without normalization the can_reuse check fails on any GFlowNet whose
        # ``device`` was supplied as bare ``cuda``, defeating the double buffer.
        target_device = _resolve_device(device)
        can_reuse = (
            idx in (0, 1)
            and isinstance(forward_buffers, list)
            and isinstance(backward_buffers, list)
            and len(forward_buffers) > idx
            and len(backward_buffers) > idx
            and forward_buffers[idx].shape == shape
            and backward_buffers[idx].shape == shape
            and _resolve_device(forward_buffers[idx].device) == target_device
            and _resolve_device(backward_buffers[idx].device) == target_device
        )
        if not can_reuse:
            return torch.zeros(shape, device=device), torch.zeros(shape, device=device)

        trajectory_batch._flow_buffer_idx = 1 - idx
        # Reuse the storage, but start each call from detached tensors.
        # In-place flow accumulation attaches the forward-flow output to the
        # current autograd graph; reusing that same Tensor object after a
        # previous backward would make later calls try to traverse freed graph
        # state once the double buffer wraps around.
        return (
            forward_buffers[idx].detach().zero_(),
            backward_buffers[idx].detach().zero_(),
        )
    
    def compute_flows(self, trajectory_batch: TrajectoryBatch,
                     max_depth: Optional[int] = None,
                     compute_gradients: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute forward and backward flows with vectorized operations."""
        # Check if we have cached states
        if trajectory_batch.cache_enabled and trajectory_batch.cached_states:
            return self.compute_flows_cached(
                trajectory_batch, 
                max_depth=max_depth, 
                compute_gradients=compute_gradients
            )
        
        # GIPTE / packed-W guard: the fresh (non-cached) flow path runs the backward
        # ``DiscreteUniform`` model on the policy features, which for a 3D feature
        # tensor returns a wrong-shape result (DiscreteUniform reads dims [0:2] as
        # (batch, measurements)). Both therefore require cached flows. Fail loudly
        # rather than silently corrupting backward flows if caching is disabled.
        if self.feature_extractor is not None or self.packed_w_input:
            mode = "GIPTE (feature_extractor set)" if self.feature_extractor is not None \
                else "packed-W input (packed_w_input=True)"
            raise NotImplementedError(
                f"{mode} requires cached flows; the fresh compute_flows path "
                "feeds the backward DiscreteUniform model a 3D feature tensor. "
                "Sample with cache_for_flows=True (the default)."
            )

        # Original flow computation (unchanged)
        batch_size = trajectory_batch.batch_size
        n_measurements = trajectory_batch.n_measurements
        max_length = trajectory_batch.max_length
        device = trajectory_batch.device
        
        forward_flows, backward_flows = self._fresh_flow_tensors(
            trajectory_batch, batch_size, n_measurements, device
        )
        
        # Create a new tableau for flow computation
        batched_tableau = self._tableau_cls(
            n_qubits=self.n_qubits,
            batch_size=batch_size,
            n_measurements=n_measurements,
            device=str(device)
        )
        
        # Create temporary batch for tracking state
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
        
        # Copy trajectory info for backward mask computation
        temp_batch.actions_time_major = trajectory_batch.actions_time_major.clone()
        temp_batch.actions = temp_batch.actions_time_major.permute(1, 2, 0)
        temp_batch.qubit_last_use_step = trajectory_batch.qubit_last_use_step.clone()
        
        for step in range(max_length):
            # Update active mask for this step
            step_active = step < trajectory_batch.lengths
            temp_batch.active = step_active
            batched_tableau.active = step_active

            states_tensor, indices = self._policy_features_active(batched_tableau)
            if states_tensor.shape[0] == 0:
                break
            
            # Compute logits
            if compute_gradients:
                # Bf16-backward autocast is applied on the CACHED grad path
                # (_forward_selected); this legacy non-cached path (cache disabled /
                # empty cache) does NOT wrap it. No production caller reaches here
                # with grads (all pass cache_for_flows=True), so warn once rather
                # than silently giving the OFF-precision/perf on a knob the user set.
                if self.use_bf16_backward and not self._warned_bf16_legacy_flow:
                    logging.warning("use_bf16_backward is not applied on the legacy "
                                    "non-cached compute_flows gradient path (fp32/TF32 used here).")
                    self._warned_bf16_legacy_flow = True
                logits_f = self.pf_model(states_tensor)
            else:
                with torch.no_grad():
                    logits_f = self.pf_model(states_tensor)
            
            # Compute masks
            masks = self.compute_action_masks_gpu(temp_batch, max_depth)
            
            # Convert indices to tensor for batch operations
            if isinstance(indices, list):
                indices_tensor = torch.tensor(indices, dtype=torch.long, device=device)
            elif isinstance(indices, torch.Tensor):
                indices_tensor = indices.to(device)
            else:
                indices_tensor = torch.stack([torch.as_tensor(idx, device=device) for idx in indices])
            
            # Get actions for this step and filter out trajectories that have ended
            step_actions = trajectory_batch.actions_time_major[
                step, indices_tensor[:, 0], indices_tensor[:, 1]
            ]
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
                                   dtype=torch.long, device=device)

                active_mask = (step < trajectory_batch.lengths - 1)
                if active_mask.any():
                    next_actions = trajectory_batch.actions_time_major[step]
                    terminal_mask = self.action_gate_types[next_actions] == self.gate_name_to_idx["terminal"]

                    actions[active_mask & ~terminal_mask] = next_actions[active_mask & ~terminal_mask]

                    temp_batch.active = active_mask & ~terminal_mask
                    batched_tableau.active = temp_batch.active.clone()

                    if temp_batch.active.any():
                        self.apply_actions_to_batch(
                            batched_tableau, actions, temp_batch, step=step
                        )

                # After applying actions, compute backward probabilities
                if b_indices is not None and b_indices.shape[0] > 0:
                    states_next, indices_next = self._policy_features_active(batched_tableau)
                    with torch.no_grad():
                        logits_b = self.pb_model(states_next)
                        if logits_b.dim() == 1:
                            logits_b = logits_b.unsqueeze(0).expand(states_next.shape[0], -1)

                    # Use exposed-based backward masks
                    masks_next = self.compute_backward_masks_gpu(
                        temp_batch, current_step=step + 1, forward_masks=None
                    )

                    # Get indices_next_tensor
                    if isinstance(indices_next, list):
                        indices_next_tensor = torch.tensor(indices_next, dtype=torch.long, device=device)
                    elif isinstance(indices_next, torch.Tensor):
                        indices_next_tensor = indices_next.to(device)
                    else:
                        indices_next_tensor = torch.stack([torch.as_tensor(idx, device=device) for idx in indices_next])

                    # Vectorized mapping without dict or loops
                    b_exp = b_indices.unsqueeze(1)  # (num_b, 1, 2)
                    next_exp = indices_next_tensor.unsqueeze(0)  # (1, num_active, 2)
                    matches = torch.all(b_exp == next_exp, dim=-1)  # (num_b, num_active)
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
                        
                        # Handle NaN/inf in backward log probs
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

    def _resolve_default_chunk_size(self) -> int:
        """Pick the cached-flow ``chunk_size`` default once at init from the GPU's L2
        cache size (see the ``_CHUNK_*`` class constants). Large-L2 devices favour the
        small chunk; L2 size is an empirical CLASSIFIER, not a mechanism claim — the
        cause is a PyTorch caching-allocator interaction, and
        ``PYTORCH_NO_CUDA_MEMORY_CACHING=1`` reverses the effect.

        Resolution order:
          1. ``FLOWMEAS_FLOW_CHUNK_SIZE`` env override.
          2. Conservative ``_CHUNK_CACHE_FIT`` for: non-CUDA;
             ``use_activation_checkpointing`` (keeps per-step forwards small to cap
             memory, which a big chunk would fight); or an unknown L2 size.
          3. Large L2 (>= ``_CHUNK_L2_SENSITIVE_BYTES``) -> ``_CHUNK_CACHE_FIT``.
          4. Otherwise -> ``_CHUNK_BANDWIDTH_AMPLE``, capped so the per-chunk input
             ``chunk x (2n)^2 x 4B`` stays bounded on large systems.

        ``chunk_size`` barely moves peak memory, so ``FLOWMEAS_FLOW_CHUNK_SIZE`` is the
        escape hatch for memory-pressured configurations.
        """
        # Provenance flag: an env-forced value is the user's explicit escape hatch and
        # must survive the per-call checkpoint clamp in compute_flows_cached,
        # so it wins "ahead of every other branch" end-to-end, not just at resolve time.
        self._chunk_size_from_env = False
        env = os.environ.get("FLOWMEAS_FLOW_CHUNK_SIZE")
        if env:
            try:
                v = int(env)
                if v > 0:
                    self._chunk_size_from_env = True
                    return v
                logging.warning(f"Ignoring FLOWMEAS_FLOW_CHUNK_SIZE={env!r} (must be a positive int)")
            except ValueError:
                logging.warning(f"Ignoring FLOWMEAS_FLOW_CHUNK_SIZE={env!r} (not an int)")
        # large_hubbard / checkpoint regime keeps per-step forwards small on purpose;
        # CPU never hits the perf path. Both -> the safe never-worst value.
        if self.use_activation_checkpointing or self.device.type != "cuda":
            return self._CHUNK_CACHE_FIT
        try:
            l2 = getattr(torch.cuda.get_device_properties(self.device), "L2_cache_size", None)
        except Exception:
            l2 = None
        if l2 is None or l2 >= self._CHUNK_L2_SENSITIVE_BYTES:
            # unknown L2 -> never-worst 5000; large L2 -> cache-resident small chunks.
            return self._CHUNK_CACHE_FIT
        # ample-bandwidth regime: big chunks, bounded by input bytes for large systems.
        row_bytes = max(1, (2 * self.n_qubits) ** 2 * 4)
        return max(self._CHUNK_CACHE_FIT,
                   min(self._CHUNK_BANDWIDTH_AMPLE, self._CHUNK_AMPLE_INPUT_BUDGET // row_bytes))

    def _effective_checkpoint(self, n_total_rows: int) -> bool:
        """Decide whether to checkpoint the cached-flow forward for THIS call
        (memory-aware gate). Honors an explicit
        ``use_activation_checkpointing=True``. Otherwise, for heavy-token models
        (those exposing ``estimated_token_activation_bytes`` — packed_w_rowtoken,
        packed_w_split row_phi='nonlinear'), checkpoint only when the no-checkpoint
        backward-retained tokens for the ACTUAL concat size would not fit in free
        GPU memory. So a small-``row_embed_dim`` nonlinear model runs recompute-
        free (~500ms, matching linear), while a wide-``d`` / very-large-batch one
        is force-checkpointed to avoid OOM. Light models (clifford_mlp, linear
        packed_w_split) expose no estimator → never gated."""
        if self.use_activation_checkpointing:
            return True
        if self.device.type != 'cuda' or n_total_rows <= 0:
            return False
        inner = getattr(self.pf_model, "_orig_mod", self.pf_model)
        est = getattr(inner, "estimated_token_activation_bytes", None)
        if est is None:
            return False  # light model: never materializes the dense tokens
        try:
            need = int(est(n_total_rows))
            free, _total = torch.cuda.mem_get_info(self.device)
        except Exception:
            # mem_get_info unavailable → fall back to the coarse qubit-count guard.
            return self.n_qubits >= self._HEAVY_TOKEN_CHECKPOINT_QUBITS
        # ``need`` (retained tokens + bits) is the forward's additional allocation;
        # ``free`` already excludes the floor resident at this point. Checkpoint only
        # when the estimate would exceed the free budget, with small headroom for
        # grads/working set — calibrated to the measured d<=64 fits / d=128 OOMs
        # boundary.
        ckpt = need > free * 0.85
        if self.debug:
            logging.debug(
                f"_effective_checkpoint: n_total={n_total_rows} "
                f"need={need / 1e9:.2f}GB free={free / 1e9:.2f}GB -> ckpt={ckpt}"
            )
        return ckpt

    def _forward_selected(self, chunk_states, chunk_masks, chunk_actions,
                          forward_fn, compute_gradients, use_checkpoint=None):
        """Forward one chunk and RETURN the selected forward log-probs — the body
        of ``_scatter_flow_chunk`` up to (but not including) the ``scatter_add_``.

        Shared by the plain per-step / batched path (via ``_scatter_flow_chunk``)
        and the always-on circuit-dedup path, which forwards only the UNIQUE
        ``(circuit, step)`` rows and gathers the selected log-probs back to every
        cached row before a single ``scatter_add_``. The policy MLP is row-wise,
        so the result is independent of how rows are grouped/deduped into chunks.
        ``forward_fn`` is the eager ``_orig_mod`` (hoisted by the caller) so
        checkpoint's backward-time recompute does not fight CUDA-graph capture.
        ``use_checkpoint`` is the per-call ``_effective_checkpoint`` decision
        (defaults to the static ``use_activation_checkpointing`` flag).
        """
        if use_checkpoint is None:
            use_checkpoint = self.use_activation_checkpointing
        if chunk_states.dtype != torch.float32:
            # uint8 state cache: upcast per chunk at the last moment so the
            # 4x-smaller uint8 rows ride through the cache, the cross-step concat and
            # the dedup gather, leaving only one transient fp32 chunk alive. 0/1
            # uint8 -> fp32 is bit-exact and carries no autograd (states are
            # non-differentiable), so flows and gradients match the fp32-cache path.
            chunk_states = chunk_states.float()
        if compute_gradients:
            # Optionally run the policy-MLP forward (the (2n)^2-input Linears, most
            # of the step via their BACKWARD GEMMs) under bf16 autocast. autocast
            # makes both grad GEMMs use bf16 tensor-core OPERANDS while the stored
            # ``.grad`` tensors stay fp32 and grad_weight ACCUMULATES in fp32 — that
            # accumulation is why no GradScaler is needed. The lever is the
            # gradient-path GEMMs, not just the forward. Unlike use_bf16_sampling this
            # perturbs the GRADIENT, so it is OFF by default. Loss/log_softmax/scatter
            # stay fp32, and ``checkpoint(use_reentrant=False)`` preserves the autocast
            # state into the backward recompute.
            bf16 = self.use_bf16_backward
            # Only ENTER an autocast when bf16 is on: ``autocast(enabled=False)`` is
            # NOT a no-op under an OUTER autocast — it would suppress it for this
            # scope. Guarding with the shared ``_NULL_RECORD_CONTEXT`` keeps the OFF
            # path bit-identical. ``device_type`` comes from ``self.device.type`` so
            # the branch is correct on any device.
            ac = (torch.autocast(device_type=self.device.type, dtype=torch.bfloat16)
                  if bf16 else _NULL_RECORD_CONTEXT)
            with ac:
                if use_checkpoint:
                    # Checkpoint the forward so its hidden activations are
                    # not held across all cached steps until the single
                    # ``.backward`` (at 52q that stacked ~60 GiB). Backward
                    # re-runs the forward (~2x MLP).
                    logits_f = _torch_checkpoint.checkpoint(
                        forward_fn, chunk_states, use_reentrant=False
                    )
                else:
                    # Checkpointing disabled (memory fits): keep activations so
                    # backward does NOT re-run the forward. Same precision as the
                    # checkpoint arm above (both execute under the same ``ac``
                    # autocast context).
                    logits_f = forward_fn(chunk_states)
            if bf16:
                # Back to fp32 for the masked log_softmax (stability) and the fp32
                # forward_flows scatter buffer. This ``.float`` adds a
                # ToCopyBackward node whose backward casts the fp32 upstream gradient
                # back to bf16 before it reaches ``logits_f`` — that bf16 grad_output
                # is what makes the MLP grad_input GEMM use bf16 operands.
                logits_f = logits_f.float()
        else:
            # No-grad sampling/eval: deliberately use the compiled ``pf_model``
            # (NOT ``forward_fn``, which is the eager ``_orig_mod``). ``forward_fn``
            # is only needed on the checkpoint/autograd paths above, where
            # backward-time recompute must avoid fighting CUDA-graph capture.
            with torch.no_grad():
                logits_f = self.pf_model(chunk_states)

        # Forward masks always leave terminal valid, so rows are never all -inf
        # in normal cached sampling/replay.
        masked_logits_f = logits_f.masked_fill(~chunk_masks, float('-inf'))
        log_probs_f = F.log_softmax(masked_logits_f, dim=-1)
        selected = log_probs_f.gather(1, chunk_actions.unsqueeze(1)).squeeze(1)
        # Invalid replayed actions contribute nothing: map non-finite selected
        # log-probs to zero (covers all-masked NaN and masked-action -inf) in one
        # out-of-place op, no ``zeros_like`` allocation.
        return torch.nan_to_num(selected, nan=0.0, posinf=0.0, neginf=0.0)

    def _scatter_flow_chunk(self, chunk_states, chunk_masks, chunk_actions,
                            chunk_flat_idx, forward_flows_flat, forward_fn,
                            compute_gradients, use_checkpoint=None,
                            chunk_valid=None):
        """Forward one chunk of cached rows; scatter-add the selected forward
        log-probs into ``forward_flows_flat`` by global flat index. Thin wrapper
        over ``_forward_selected`` + ``scatter_add_`` (see that method).

        ``chunk_valid`` (optional ``(rows,)`` bool tensor) zeroes the
        contribution of rows past their trajectory length WITHOUT a boolean
        gather (whose data-dependent output shape forces a host sync); the
        zero weight also zeroes those rows' gradients exactly. ``None`` keeps
        the historical behavior (every row contributes)."""
        selected = self._forward_selected(
            chunk_states, chunk_masks, chunk_actions,
            forward_fn, compute_gradients, use_checkpoint,
        )
        if chunk_valid is not None:
            selected = selected * chunk_valid.to(selected.dtype)
        forward_flows_flat.scatter_add_(0, chunk_flat_idx, selected)

    def _circuit_ids(self, trajectory_batch) -> torch.Tensor:
        """Exact integer circuit id per ``(b, m)`` trajectory, collision-free.

        Two trajectories get the same id iff their action sequences over
        ``[0, length)`` AND their lengths are identical -> the same circuit.
        ``torch.unique(..., dim=0)`` compares full rows, so there is no hash
        collision risk (a wrong merge would silently corrupt flows). Returns a
        ``(B*M,)`` int64 tensor of ids in ``[0, U_circuits)``, indexed by the
        same flat ``b*M + m`` as ``cat_flat``. Used by the circuit dedup
        (batched grad path only; skipped when checkpointing is active) in
        ``compute_flows_cached`` to collapse identical-circuit rows.
        """
        atm = trajectory_batch.actions_time_major              # (T, B, M)
        T, B, M = atm.shape
        a = atm.permute(1, 2, 0).reshape(B * M, T).to(torch.int64)   # row = b*M + m
        L = trajectory_batch.lengths.reshape(B * M, 1).to(torch.int64)
        # Canonicalize padding: zero out actions past ``length`` and append the
        # length, so a short circuit never aliases a longer one sharing its
        # prefix (the prefix-collision pitfall). ``+1`` reserves 0 for padding.
        tidx = torch.arange(T, device=a.device).unsqueeze(0)        # (1, T)
        a_canon = torch.where(tidx < L, a + 1, torch.zeros_like(a))
        key_mat = torch.cat([a_canon, L], dim=1)                    # (B*M, T+1)
        _uniq, inverse = torch.unique(key_mat, dim=0, return_inverse=True)
        return inverse.reshape(B * M)

    def compute_flows_cached(self, trajectory_batch: TrajectoryBatch,
                           max_depth: Optional[int] = None,
                           compute_gradients: bool = True,
                           chunk_size: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute flows using cached ACTIVE-only states with row-wise MLP
        forwards + scatter_add.

        Branches on ``batch_forwards = compute_gradients and not eff_ckpt``,
        where ``eff_ckpt`` comes from ``_effective_checkpoint`` (True when
        ``use_activation_checkpointing=True``, or when the memory probe finds a
        heavy-token model would OOM without it):

          (a) Checkpointing ON: per-step path. Each step's cached active rows are
              forwarded in ``chunk_size`` blocks with each ``forward_fn`` call
              wrapped in ``torch.utils.checkpoint.checkpoint`` (see
              ``_scatter_flow_chunk``), so hidden activations are not held across
              every cached step until the single ``.backward``. Costs a forward
              recompute in backward; no cross-step concat is materialized, which
              is what caps memory at large-system scale.

          (b) Checkpointing OFF (the default for non-large_hubbard runs): batched
              path. The MLP is row-wise and ``scatter_add_`` is commutative, so
              every step's cached active rows are concatenated and forwarded in
              ``chunk_size`` blocks spanning step boundaries — numerically
              equivalent to (a) up to fp32 reorder noise, with fewer, fuller
              GEMMs and no recompute. It transiently holds both the per-step
              slices still referenced in ``pending`` and the concatenated copy,
              bounded by the SUM of active rows across steps (active-only
              caching) rather than ``max_length * B*M``.

        ``chunk_size`` bounds the rows fed to a single ``pf_model`` call within a
        chunk (per-step in (a), cross-step in (b)). ``None`` — every production
        caller — resolves to the L2-adaptive ``self._default_chunk_size``.
        Numerically equivalent at any value.
        """
        chunk_size_auto = chunk_size is None  # auto default -> safe to clamp under forced checkpointing
        if chunk_size is None:
            chunk_size = self._default_chunk_size
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")

        # Per-call circuit-dedup instrumentation (total_rows, unique_rows); set
        # by the batched path below, left None when that path isn't taken.
        self._last_flow_dedup_stats = None

        # Hoist the model unwrapping out of the hot loop. The trainer applies
        # ``torch.compile(..., mode='reduce-overhead')``, which uses CUDAGraphs, and
        # capture is invalidated by ``checkpoint``'s backward-time recompute — so the
        # gradient path calls the eager ``_orig_mod``. The no-grad path can keep the
        # compiled wrapper, since the recompute issue only triggers under autograd.
        forward_fn = getattr(self.pf_model, "_orig_mod", self.pf_model)

        batch_size = trajectory_batch.batch_size
        n_measurements = trajectory_batch.n_measurements
        device = trajectory_batch.device
        
        forward_flows, backward_flows = self._fresh_flow_tensors(
            trajectory_batch, batch_size, n_measurements, device
        )
        
        # Check if we have cached states
        if not trajectory_batch.cached_states:
            logging.warning("No cached states found in trajectory batch!")
            # Return zeros if no cached states
            return forward_flows, backward_flows
        
        # Forward flows: scatter-add into the preallocated/double-buffered
        # output storage, so neither a time-dim concat nor a fresh flat output
        # tensor is materialized per call.
        forward_flows_flat = forward_flows.view(-1)
        num_states = 0
        # With activations kept (no checkpointing), batch the per-(step, chunk)
        # policy forwards ACROSS steps into fewer, larger GEMMs — the MLP is row-wise
        # and ``scatter_add_`` is commutative, so this is numerically equivalent to
        # the per-step loop up to fp32 reorder noise. Under checkpointing keep the
        # per-step path: the cross-step concat is the time-dim materialization that
        # caps memory at large-system scale. ``eff_ckpt`` is a per-call decision, and
        # ``n_total_upper`` bounds the cross-step concat rows.
        n_total_upper = sum(
            cd[0].shape[0]
            for cd in trajectory_batch.cached_states
            if cd is not None and cd[0] is not None
        )
        eff_ckpt = self._effective_checkpoint(n_total_upper)
        # The adaptive default is resolved at INIT from the static
        # ``use_activation_checkpointing`` flag, but ``_effective_checkpoint`` can
        # force checkpointing PER CALL. In that case fall back to the conservative
        # ``_CHUNK_CACHE_FIT`` so the memory-pressured path is not run with a large
        # auto chunk, which would defeat the bound that forcing checkpointing exists
        # to provide. Only the AUTO default is clamped; an explicit caller
        # ``chunk_size`` and an ``FLOWMEAS_FLOW_CHUNK_SIZE`` override are both honored.
        # CAVEAT: an env-forced LARGE value bypasses this bound and can OOM a
        # forced-checkpointing run — unset it to restore the clamp.
        if (chunk_size_auto and not self._chunk_size_from_env
                and eff_ckpt and chunk_size > self._CHUNK_CACHE_FIT):
            chunk_size = self._CHUNK_CACHE_FIT
        batch_forwards = compute_gradients and not eff_ckpt
        pending = []  # [(states, masks, actions, flat_idx, step, valid),...] across steps

        for step, cached_data in enumerate(trajectory_batch.cached_states):
            if cached_data is None:
                continue  # Skip None entries instead of breaking

            states_tensor, indices = cached_data
            n_step = states_tensor.shape[0]
            if n_step == 0:
                continue

            # Filter by trajectory length as a WEIGHT, not a gather. Boolean gathers
            # have data-dependent output shapes, i.e. one forced device->host sync per
            # cached step per flow call. Instead every cached row takes the
            # fixed-shape forward and rows past their trajectory length contribute
            # with weight 0 in the scatter_add (exact in value; gradient exactly
            # zero). For every batch the current samplers produce the mask is all-true
            # anyway, so the weight form preserves the historical filter semantics
            # without the sync. ``num_states`` now counts ALL cached rows; its only
            # consumer is the ``== 0`` early return, whose result is unchanged.
            b_idx = indices[:, 0]
            m_idx = indices[:, 1]
            valid_mask = step < trajectory_batch.lengths[b_idx, m_idx]

            raw_step_actions = trajectory_batch.actions_time_major[
                step, b_idx, m_idx
            ]
            # Mask length-invalid rows rather than removing them, so dirty
            # padding sentinels (for example -1) cannot reach the gather while
            # the tensor shape stays fixed; valid rows still surface malformed
            # ids in the policy gather.
            step_actions = torch.where(valid_mask, raw_step_actions, 0)
            # Get masks - check if they're cached or need to be computed.
            # The cache stores active-only masks aligned positionally with
            # cached_states[step][1] (the ``indices`` rows above), so the full
            # cached tensor is used as-is.
            if (hasattr(trajectory_batch, 'cached_masks') and
                trajectory_batch.cached_masks and
                step < len(trajectory_batch.cached_masks) and
                trajectory_batch.cached_masks[step] is not None):
                step_masks = trajectory_batch.cached_masks[step]
            else:
                # Compute masks if not cached
                # For now, create all-true masks as a fallback
                # This ensures we don't crash but may not be ideal
                step_masks = torch.ones(
                    (n_step, self.num_actions),
                    dtype=torch.bool,
                    device=states_tensor.device
                )
                if self.debug:
                    logging.warning(f"Step {step}: Masks not cached, using all-true masks")

            num_states += n_step
            flat_idx_step = b_idx * n_measurements + m_idx

            if batch_forwards:
                # Defer the forward; concatenate across steps below. ``step`` is
                # carried so the dedup can key rows on (circuit, step);
                # ``valid_mask`` rides along as the per-row weight.
                pending.append((states_tensor, step_masks, step_actions,
                                flat_idx_step, step, valid_mask))
            else:
                # Checkpointing ON: per-step path. Circuit dedup is skipped here —
                # it requires the cross-step concat that checkpointing avoids in order
                # to cap memory at large-system scale. A within-step dedup is net
                # negative there: the duplicate fraction is ~0 while key construction
                # costs a few percent of wall time.
                for start in range(0, n_step, chunk_size):
                    end = min(start + chunk_size, n_step)
                    self._scatter_flow_chunk(
                        states_tensor[start:end], step_masks[start:end],
                        step_actions[start:end], flat_idx_step[start:end],
                        forward_flows_flat, forward_fn, compute_gradients,
                        use_checkpoint=eff_ckpt,
                        chunk_valid=valid_mask[start:end],
                    )

        if batch_forwards and pending:
            # One concatenated forward over all active rows, with circuit dedup
            # (batched grad path only; skipped when checkpointing). A cached row is a
            # ``(circuit, step)`` pair, and two rows give identical selected
            # log-probs iff they share BOTH the circuit and the step. Trained policies
            # re-sample a small circuit support many times across the whole B*M batch,
            # so the unique ``(circuit, step)`` rows can be a small fraction of the
            # total. We forward only those, gather the result back, and let the
            # existing ``cat_flat`` scatter_add fan out to every ``(b, m)`` — exact in
            # value, and the gather's backward sums every duplicate's gradient into
            # the one representative forward. With no duplicates this is bit-identical
            # to the plain chunked forward.
            # Dedup invariant: same (circuit, step) -> same tableau -> same
            # masking-engine output, so ``rep_masks`` is valid for all duplicates.
            # This breaks if masks ever depend on state outside the tableau.
            cat_states = torch.cat([p[0] for p in pending], dim=0)
            cat_masks = torch.cat([p[1] for p in pending], dim=0)
            cat_actions = torch.cat([p[2] for p in pending], dim=0)
            cat_flat = torch.cat([p[3] for p in pending], dim=0)
            # Per-row validity weight (see the step loop above). NOTE: with the
            # weight form, ``total`` (and the dedup monitoring stats) count ALL
            # cached rows, where they previously counted only length-valid rows
            # — identical for every sampler-produced batch (mask is all-true).
            cat_valid = torch.cat([p[5] for p in pending], dim=0)
            total = cat_states.shape[0]

            dev = cat_states.device
            circuit_id = self._circuit_ids(trajectory_batch)            # (B*M,)
            cat_step = torch.cat([
                torch.full((p[0].shape[0],), p[4], dtype=torch.long, device=dev)
                for p in pending
            ])
            # One int64 key per row: (circuit_id, step). circuit_id in
            # [0, U_circuits) with U_circuits <= B*M; step in [0, max_length-1]
            # (inclusive upper bound) < stride = max_length + 1. Distinct
            # (circuit_id, step) -> distinct key because |(s1-s2)| < stride
            # forces cid1==cid2 for equal keys.
            stride = trajectory_batch.max_length + 1
            row_key = circuit_id[cat_flat] * stride + cat_step
            uniq_key, inverse = torch.unique(row_key, return_inverse=True)
            n_uniq = int(uniq_key.shape[0])
            self._last_flow_dedup_stats = (int(total), n_uniq)

            # Dedup engages only when there ARE duplicate rows. ``FLOWMEAS_FLOW_DEDUP=0``
            # is an emergency off-switch (and the A/B-validation lever) that forces
            # the plain forward even with duplicates; default on (no config knob).
            dedup_on = n_uniq < total and os.environ.get("FLOWMEAS_FLOW_DEDUP", "1") != "0"
            if not dedup_on:
                # No duplicate rows (or dedup disabled) -> plain chunked forward,
                # bit-identical to the pre-dedup batched path (locks parity tests).
                for start in range(0, total, chunk_size):
                    end = min(start + chunk_size, total)
                    self._scatter_flow_chunk(
                        cat_states[start:end], cat_masks[start:end],
                        cat_actions[start:end], cat_flat[start:end],
                        forward_flows_flat, forward_fn, compute_gradients,
                        use_checkpoint=eff_ckpt,
                        chunk_valid=cat_valid[start:end],
                    )
            else:
                # First-occurrence representative row per unique (circuit, step).
                rep = torch.full((n_uniq,), total, dtype=torch.long, device=dev)
                rep.scatter_reduce_(
                    0, inverse, torch.arange(total, device=dev),
                    reduce='amin', include_self=True,
                )
                # rep_actions is also valid for all duplicates: same (circuit, step)
                # implies the same action sequence by ``_circuit_ids`` construction. A
                # future non-circuit action source would break this invariant; add a
                # guard then.
                rep_states, rep_masks, rep_actions = (
                    cat_states[rep], cat_masks[rep], cat_actions[rep]
                )
                # Forward ONLY the unique rows (chunked); collect selected log-probs.
                sel_parts = []
                for start in range(0, n_uniq, chunk_size):
                    end = min(start + chunk_size, n_uniq)
                    sel_parts.append(self._forward_selected(
                        rep_states[start:end], rep_masks[start:end],
                        rep_actions[start:end], forward_fn, compute_gradients,
                        use_checkpoint=eff_ckpt,
                    ))
                sel_uniq = sel_parts[0] if len(sel_parts) == 1 else torch.cat(sel_parts)
                # Expand to every cached row (differentiable gather), apply the
                # per-row validity weight, and fan out to (b, m) via the unchanged
                # cat_flat scatter_add. The weight is applied PER ROW because it is
                # row-aligned — validity is in fact constant within a key group, but
                # weighting per row needs no such reasoning to be exact.
                forward_flows_flat.scatter_add_(
                    0, cat_flat, sel_uniq[inverse] * cat_valid.to(sel_uniq.dtype)
                )

        if num_states == 0:
            return forward_flows, backward_flows

        # ``forward_flows_flat`` is a view of ``forward_flows`` (see line above
        # the loop), so the in-place scatter_add above already populated
        # ``forward_flows`` with the per-trajectory log-prob sums. No reshape
        # back to (B, M) is needed.

        # For backward flows, use precomputed valid counts, mirroring the forward
        # batching. The per-step loop previously issued ~7-10 small gather / indexed
        # ``+=`` launches PER cached step. The backward (uniform) flow is a pure
        # function of the cached integer valid-counts with NO autograd dependency on
        # model params, so we accumulate ``(flat_idx, -log count)`` pairs across ALL
        # steps and apply ONE commutative ``scatter_add_``. Within a step the cached
        # ``indices`` rows are unique and the concat is built in step order, so the
        # result is bit-equal to the sequential per-step ``+=``.
        if trajectory_batch.cached_backward_valid_counts:
            terminal_idx = self.gate_name_to_idx["terminal"]
            b_flat_parts = []
            b_logp_parts = []
            for step in range(len(trajectory_batch.cached_states) - 1):
                if trajectory_batch.cached_states[step] is None:
                    continue

                # Get cached data for this step
                states_tensor, indices = trajectory_batch.cached_states[step]
                if states_tensor.shape[0] == 0:
                    continue

                # Get valid counts for next step
                if trajectory_batch.cached_backward_valid_counts[step + 1] is None:
                    continue

                next_valid_counts = trajectory_batch.cached_backward_valid_counts[step + 1]

                # Contribution predicate as a WEIGHT, not a gather — a boolean gather
                # here would be a per-step host sync, like the forward loop above.
                # A row contributes iff it took a non-terminal action at
                # ``step`` AND step < length - 1.
                # Non-contributing rows are appended with weight 0; scatter_add_ of
                # exact zeros is bit-equal to not appending them.
                b_idx = indices[:, 0]
                m_idx = indices[:, 1]
                lengths_rows = trajectory_batch.lengths[b_idx, m_idx]
                row_valid = step < lengths_rows
                raw_step_actions = trajectory_batch.actions_time_major[
                    step, b_idx, m_idx
                ]
                # As in the forward path, sanitize only rows the historical
                # length gather removed. Invalid ids on valid rows still raise.
                step_actions = torch.where(row_valid, raw_step_actions, 0)
                non_terminal = self.action_gate_types[step_actions] != terminal_idx
                contributes = non_terminal & (step < lengths_rows - 1)

                # Uniform backward log-prob: log(1/n) = -log(n); clamp(min=1)
                # avoids log(0) (and keeps the weighted-zero rows finite). Defer
                # the indexed accumulation to a single cross-step scatter_add_
                # below.
                b_valid_counts = next_valid_counts[b_idx, m_idx].clamp(min=1)
                b_flat_parts.append(b_idx * n_measurements + m_idx)
                b_logp_parts.append(
                    -torch.log(b_valid_counts.float()) * contributes.to(torch.float32)
                )

            if b_flat_parts:
                backward_flows.view(-1).scatter_add_(
                    0, torch.cat(b_flat_parts), torch.cat(b_logp_parts)
                )

        if self.debug:
            logging.debug("\nDEBUG compute_flows_cached final:")
            logging.debug(f"  Forward flows shape: {forward_flows.shape}")
            logging.debug(f"  Backward flows shape: {backward_flows.shape}")
            logging.debug(f"  Total cached states processed: {num_states}")
            logging.debug(f"  Used precomputed backward counts: {bool(trajectory_batch.cached_backward_valid_counts)}")
        
        return forward_flows, backward_flows
    
    def compute_loss(self, trajectory_batch: TrajectoryBatch, costs: torch.Tensor,
                    beta: float = 1.0, max_depth: Optional[int] = None,
                    metrics_to_cpu: bool = True,
                    **reward_kwargs) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute loss for a batch of trajectories with fully vectorized operations."""
        assert costs.shape[0] == trajectory_batch.batch_size, \
            f"Costs shape {costs.shape} doesn't match batch size {trajectory_batch.batch_size}"
        
        # Ensure costs are on the correct device.: guard the move so the
        # steady-state CUDA path (compute_costs_with_probabilities already
        # returns on ``self.device``) skips an unconditional no-op aten
        # dispatch per loss call; the legacy CPU-cost path still moves.
        if _resolve_device(costs.device) != _resolve_device(self.device):
            costs = costs.to(self.device)
        
        if max_depth is None:
            max_depth = getattr(self, "last_max_depth", None)
        
        if self.debug:
            logging.debug("\nDEBUG compute_loss start:")
            logging.debug(f"  Batch size: {trajectory_batch.batch_size}")
            logging.debug(f"  N measurements: {trajectory_batch.n_measurements}")
            logging.debug(f"  Costs shape: {costs.shape}, device: {costs.device}")
            logging.debug(f"  Max depth: {max_depth}")
            logging.debug(f"  Cache enabled: {trajectory_batch.cache_enabled}")
            logging.debug(f"  Cached states: {len(trajectory_batch.cached_states) if trajectory_batch.cached_states else 0}")
        
        forward_flows, backward_flows = self.compute_flows(trajectory_batch, max_depth=max_depth, compute_gradients=True)
        
        rewards = self.reward_fn(costs, beta=beta, **reward_kwargs) # reward shape should match (batch_size, n_measurements)
        
        # Diagnostic NaN/Inf reward check — gated behind ``debug`` so it does not
        # force two ``.any`` host syncs on every training step. update_step's
        # fused finiteness guard already skips updates on non-finite loss, so the
        # production path loses no safety, only the per-step diagnostic.
        if self.debug and (torch.isnan(rewards).any() or torch.isinf(rewards).any()):
            logging.warning(f"NaN/Inf detected in rewards! Costs stats: min={costs.min().item():.6f}, "
                          f"max={costs.max().item():.6f}, mean={costs.mean().item():.6f}")
            logging.warning(f"Reward function: {self.reward_fn.__name__}, beta={beta}, reward_kwargs={reward_kwargs}")
        
        if self.debug:
            logging.debug(f"  Forward flows shape: {forward_flows.shape}, non-zero: {(forward_flows != 0).sum().item()}")
            logging.debug(f"  Backward flows shape: {backward_flows.shape}, non-zero: {(backward_flows != 0).sum().item()}")
            logging.debug(f"  Rewards shape: {rewards.shape}, mean: {rewards.mean().item():.4f}")
        
        valid_mask = trajectory_batch.lengths > 0
        valid_counts = valid_mask.sum(dim=1)
        
        #valid_counts should be as many as the number trajectories in each batch
        if self.debug:
            logging.debug(f"  Valid counts shape: {valid_counts.shape}, non-zero: {(valid_counts > 0).sum().item()}")
            logging.debug(f"  Valid mask: {valid_mask.sum(dim=1)}")
        
        # Vectorized averaging across valid trajectories
        valid_counts_clamped = valid_counts.clamp(min=1)
        mask_float = valid_mask.float()
        forward_flows_sum = (forward_flows * mask_float).sum(dim=1)
        backward_flows_sum = (backward_flows * mask_float).sum(dim=1)
        forward_flows_avg = forward_flows_sum / valid_counts_clamped
        backward_flows_avg = backward_flows_sum / valid_counts_clamped
        
        batch_valid = valid_counts > 0

        if getattr(self.objective, "supports_valid_mask", False):
            # Static-shape masked path. The legacy ``forward_flows_avg[batch_valid]``
            # boolean gather is a device->host sync (data-dependent output shape).
            # Instead pass the full (B,) rows plus the boolean ``valid_mask`` to the
            # objective, which averages its per-row error over the valid rows only —
            # exactly equal to the filtered mean, but with fixed shapes and no sync.
            # The all-invalid case yields loss 0 via the clamped denominator, and
            # update_step's finiteness guard still backstops a non-finite loss.
            pf_inner = getattr(self.pf_model, "_orig_mod", self.pf_model)
            logZ = pf_inner.logZ if hasattr(pf_inner, 'logZ') else torch.tensor(0.0, device=self.device)
            loss, objective_metrics = self.objective.compute_loss(
                forward_flows=forward_flows_avg,
                backward_flows=backward_flows_avg,
                rewards=rewards,
                logZ=logZ,
                valid_mask=batch_valid,
            )
            metrics_tensors = {
                'loss': loss.mean() if loss.dim() > 0 else loss,
                'reward': rewards.mean(),
                'cost': costs.mean(),
                'logZ': logZ.mean() if logZ.dim() > 0 else logZ,
                'avg_trajectories_per_batch': valid_counts.float().mean(),
            }
            for k, v in objective_metrics.items():
                metrics_tensors[k] = v.mean() if torch.is_tensor(v) and v.dim() > 0 else v
            metrics = (
                {k: v.item() if torch.is_tensor(v) else v for k, v in metrics_tensors.items()}
                if metrics_to_cpu
                else metrics_tensors
            )
            return loss, metrics

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
            
            # Get logZ via the inner module so a ``torch.compile``'d wrapper doesn't
            # shadow the trainable parameter: after compile, ``logZ`` lives on
            # ``_orig_mod``, so a direct read raises AttributeError (caught by
            # ``hasattr``) and the loss silently falls back to a constant zero.
            pf_inner = getattr(self.pf_model, "_orig_mod", self.pf_model)
            logZ = pf_inner.logZ if hasattr(pf_inner, 'logZ') else torch.tensor(0.0, device=self.device)

            loss, objective_metrics = self.objective.compute_loss(
                forward_flows=forward_flows_avg,
                backward_flows=backward_flows_avg,
                rewards=rewards_filtered,
                logZ=logZ
            )

            if self.debug:
                logging.debug(f"  Loss shape: {loss.shape if hasattr(loss, 'shape') else 'scalar'}")
                logging.debug(f"  Loss value: {loss.item()}")
                logging.debug(f"  logZ shape: {pf_inner.logZ.shape}")
                logging.debug(f"  logZ value: {pf_inner.logZ.item()}")
            
            # Compute metrics on GPU, transfer to CPU only once at the end
            metrics_tensors = {
                'loss': loss.mean() if loss.dim() > 0 else loss,
                'reward': rewards.mean(),
                'cost': costs.mean(),
                'logZ': logZ.mean() if logZ.dim() > 0 else logZ,
                'avg_trajectories_per_batch': valid_counts.float().mean()
            }
            # Add objective_metrics
            for k, v in objective_metrics.items():
                metrics_tensors[k] = v.mean() if torch.is_tensor(v) and v.dim() > 0 else v
            
            metrics = (
                {k: v.item() if torch.is_tensor(v) else v for k, v in metrics_tensors.items()}
                if metrics_to_cpu
                else metrics_tensors
            )
        else:
            # Return zero loss with gradient
            zero_loss = torch.zeros(1, device=self.device, requires_grad=True).squeeze()
            pf_inner_fb = getattr(self.pf_model, "_orig_mod", self.pf_model)
            logZ_fallback = (
                pf_inner_fb.logZ
                if hasattr(pf_inner_fb, 'logZ')
                else torch.zeros((), device=self.device)
            )
            if metrics_to_cpu:
                return zero_loss, {
                    'loss': 0.0,
                    'reward': 0.0,
                    'cost': costs.mean().item() if costs.numel() > 0 else 0.0,
                    'logZ': logZ_fallback.item() if torch.is_tensor(logZ_fallback) and logZ_fallback.numel() == 1 else 0.0,
                    'avg_trajectories_per_batch': 0.0
                }

            zero_metric = torch.zeros((), device=self.device)
            return zero_loss, {
                'loss': zero_metric,
                'reward': zero_metric,
                'cost': costs.mean() if costs.numel() > 0 else zero_metric,
                'logZ': logZ_fallback.mean() if torch.is_tensor(logZ_fallback) and logZ_fallback.dim() > 0 else logZ_fallback,
                'avg_trajectories_per_batch': zero_metric,
            }
        
        return loss, metrics

    def _full_batch_indices(self, batch_size: int, n_measurements: int) -> torch.Tensor:
        """Return a stable ``(B*M, 2)`` index tensor on the GFN device."""
        key = (batch_size, n_measurements, self.device)
        cached = self._full_indices_cache.get(key)
        if cached is None:
            b = torch.arange(batch_size, device=self.device).repeat_interleave(n_measurements)
            m = torch.arange(n_measurements, device=self.device).repeat(batch_size)
            cached = torch.stack([b, m], dim=1)
            self._full_indices_cache[key] = cached
        return cached

    def _offpolicy_zero_logits(self, n_rows: int, n_actions: int,
                               device: torch.device) -> torch.Tensor:
        """Return an ``(n_rows, n_actions)`` all-zero float32 logit buffer for
        OFF_POLICY (uniform-over-valid) sampling.

        The off-policy logits are
        conceptually a constant zero tensor that ``masked_gumbel_argmax`` only
        READS — the fused CuPy kernel takes ``const float* logits`` and the
        PyTorch fallback adds Gumbel into a fresh tensor (it does not re-mask in
        place). So a buffer that is never written stays all-zero and is safe to
        reuse/slice, removing the per-step ``torch.zeros_like`` allocation + the
        zero-fill kernel. The dynamic path's ``n_rows`` (active rows) decays
        within a sampling call, so the buffer is allocated to the largest
        request seen and sliced thereafter.
        """
        buf = getattr(self, "_offpolicy_zero_buf", None)
        if (buf is None or buf.device != device or buf.shape[1] != n_actions
                or buf.shape[0] < n_rows):
            cap = n_rows if buf is None else max(n_rows, buf.shape[0])
            buf = torch.zeros((cap, n_actions), dtype=torch.float32, device=device)
            self._offpolicy_zero_buf = buf
        return buf[:n_rows]
