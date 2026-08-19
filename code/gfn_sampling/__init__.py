# -*- coding: utf-8 -*-
"""Trajectory sampling mixin for ``GFlowNet``.

Split out of ``GFNs.py``. Owns the fused-kernel eligibility gates
(``_effective_*``), policy feature extraction / forward paths (static,
dynamic-active CUDA-graph, GIPTE capture), and the samplers
(static-shape, bucketed, bucketed-compacted, bucketed CUDA-graph) plus
the public ``sample_trajectories`` dispatch.

This is a mixin: methods are composed into ``GFlowNet`` (see
``gfn_core``) with identical MRO semantics. Numerical behavior and kernel
launch counts are unchanged; moved warning sites deliberately retain the
public ``code.gfn_sampling`` logger name.
"""

import gc
import logging
import torch
from collections import OrderedDict
from typing import List, Tuple, Dict, Optional, Any

try:
    from ..gfn_runtime import (
        _SAMPLING_MODE_BUCKETED,
        BucketedGraphPreflightError,
        FlowMeasTableau,
        SamplingMode,
        masked_gumbel_argmax,
        _fused_sampling_persistently_unavailable,
        _fused_metadata_persistently_unavailable,
        _compute_mask_counts_fused,
        _fused_mask_counts_persistently_unavailable,
        _fused_counter_rng_persistently_unavailable,
        _partition_update_bucketed_torch,
        _fused_partition_update_persistently_unavailable,
        _fused_apply_adapter,
        counter_uniforms,
        ordered_partition_scatter,
    )
    from ..gfn_trajectory import (
        TrajectoryBatch,
    )
except ImportError:  # pragma: no cover - direct-execution mode
    from gfn_runtime import (
        _SAMPLING_MODE_BUCKETED,
        BucketedGraphPreflightError,
        FlowMeasTableau,
        SamplingMode,
        masked_gumbel_argmax,
        _fused_sampling_persistently_unavailable,
        _fused_metadata_persistently_unavailable,
        _compute_mask_counts_fused,
        _fused_mask_counts_persistently_unavailable,
        _fused_counter_rng_persistently_unavailable,
        _partition_update_bucketed_torch,
        _fused_partition_update_persistently_unavailable,
        _fused_apply_adapter,
        counter_uniforms,
        ordered_partition_scatter,
    )
    from gfn_trajectory import (
        TrajectoryBatch,
    )

try:
    from .gates_policy import GatesPolicyMixin
except ImportError:  # pragma: no cover - direct-execution mode
    from gates_policy import GatesPolicyMixin


class GFlowNetSamplingMixin(GatesPolicyMixin):
    """Sampling / policy-forward methods of ``GFlowNet`` (split from ``GFNs.py``)."""

    def _configure_sampling_tableau(self, batched_tableau: FlowMeasTableau) -> None:
        """Apply sampler-only backend knobs to a newly-created tableau.

        This hot-path opt-out is only safe for sampling paths that construct
        action ids from this GFlowNet's action map and masks. Replay and
        checkpoint-derived action streams should keep backend validation on.
        """
        if hasattr(batched_tableau, "validate_action_ids"):
            # GFlowNet sampling constructs action ids from this run's own
            # action map and masks, so the CT ActionAdapter's per-step
            # out-of-range validation sync is redundant on this hot path.
            batched_tableau.validate_action_ids = False

    def _effective_fused_metadata_kernel(self) -> bool:
        """``use_fused_metadata_kernel`` ANDed with the per-process latch.

        Same fail-once pattern as ``_effective_fused_sampling_kernel``: once
        the CuPy import / NVRTC compile / launch fails, ``apply_actions_to_batch``
        skips the fused-attempt overhead and uses the PyTorch metadata path
        directly. Re-checks the module-level latch so a hard failure in any
        prior call (including a different ``GFlowNet`` instance sharing this
        process) latches this instance off too. Mirrors the metadata-kernel
        plumbing added in to match its five sibling fused kernels.
        """
        if not self.use_fused_metadata_kernel:
            return False
        if self._fused_metadata_kernel_failed:
            return False
        if _fused_metadata_persistently_unavailable():
            self._fused_metadata_kernel_failed = True
            if self.debug:
                logging.debug(
                    "fused metadata kernel latched off after persistent CuPy/NVRTC failure"
                )
            return False
        return True

    def reset_fused_apply_latches(self) -> None:
        """Clear this instance's per-instance fused-apply state.

        Pair with ``measurement_adapter.fused_apply_adapter.reset_persistent_failure``
        if you've remediated host-level state (re-installed CuPy,
        rotated GPUs) and want this live ``GFlowNet`` to retry the
        fused path on the next call instead of staying short-circuited.

        Resets:
          * ``_fused_apply_kernel_failed`` — the per-instance fail-once
            latch set when a preflight ``None`` return or a module-level
            unavailability is observed.
          * ``_fused_apply_lowering`` — forces the lazy lowering-table
            to be rebuilt on the next ``_effective_fused_apply_kernel``
            call. Useful if the build itself failed previously (e.g.,
            CuPy was reinstalled with a different toolkit version).
          * ``_fused_apply_call_count`` — telemetry counter; reset so
            post-reset observers see a clean slate.
          * ``_bucketed_graph_cache`` — captured bucketed graphs read the fused
            apply lowering buffers by fixed address, so they cannot survive a
            lowering rebuild.

        Does NOT clear ``use_fused_apply_kernel`` (the user-facing
        request flag) nor the module-level latch. Callers that need
        both should invoke
        ``fused_apply_adapter.reset_persistent_failure`` separately.
        Documented in the module-level reset hook's docstring under
        the scope-warning section.
        """
        self._fused_apply_kernel_failed = False
        self._fused_apply_lowering = None
        self._fused_apply_call_count = 0
        if self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(self.device)
        self._bucketed_graph_cache.clear()
        try:
            import torch._dynamo as _torch_dynamo

            _torch_dynamo.reset()
        except Exception:
            pass
        if self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize(self.device)

    def _sample_trajectories_static_shape(self,
                                          batch_size: int,
                                          n_measurements: int,
                                          max_depth: int,
                                          mode: SamplingMode,
                                          max_length: int,
                                          cache_for_flows: bool) -> TrajectoryBatch:
        """Static-shape CUDA sampling path.

        This path processes all ``B*M`` rows at every step and relies on the
        active mask to no-op completed rows. That removes dynamic active-row
        extraction from sampling and gives the policy forward a stable shape
        that can be captured with CUDA Graphs.
        """
        batched_tableau = self._tableau_cls(
            n_qubits=self.n_qubits,
            batch_size=batch_size,
            n_measurements=n_measurements,
            device=str(self.device)
        )
        self._configure_sampling_tableau(batched_tableau)
        trajectory_batch = TrajectoryBatch(
            batch_size=batch_size,
            n_measurements=n_measurements,
            max_length=max_length,
            n_qubits=self.n_qubits,
            device=self.device
        )
        trajectory_batch.batched_tableau = batched_tableau
        if cache_for_flows:
            trajectory_batch.enable_caching(
                states_uint8=self._effective_uint8_state_cache()
            )

        full_indices = self._full_batch_indices(batch_size, n_measurements)
        total_rows = batch_size * n_measurements

        with torch.no_grad():
            for step in range(max_length):
                # Fused-step GIPTE path: one CUDA graph captures packed conjugation
                # -> hit-feature assembly -> policy forward, producing both
                # ``logits`` and the feature set ``H``. ON_POLICY only. Returns
                # None on capture failure -> fall back to the eager path.
                captured = None
                if mode == SamplingMode.ON_POLICY and self._gipte_capture_eligible():
                    captured = self._gipte_capture_logits(batched_tableau, total_rows)

                gipte_logits = None
                if captured is not None:
                    gipte_logits, H_static = captured
                    # H_static is the graph's reused output buffer; clone before
                    # the next step's replay overwrites it if we cache it.
                    states_tensor = H_static.clone() if cache_for_flows else H_static
                else:
                    # No clone needed for the packed-W path: policy_packed_w
                    # routes to CT get_W_bits_packed_u32(out=None), which allocates
                    # a fresh buffer per call, so each cached step holds a distinct
                    # tensor. If this is ever routed through the fixed-buffer
                    # policy_packed_w_into(out=...), it MUST ``.clone`` here or
                    # every cached step aliases the final step's W.
                    states_tensor = self._policy_features(batched_tableau, total_rows)

                # Static-shape sampling intentionally materializes the full
                # (B, M, A) mask: the policy input/mask shape stays stable for
                # CUDA Graph replay even though this gives back the active-only
                # mask memory saving used by the dynamic path.
                full_masks = self.compute_action_masks_gpu(trajectory_batch, max_depth)
                active_masks = full_masks.reshape(total_rows, self.num_actions)

                backward_valid_counts = None
                if cache_for_flows and step < max_length - 1:
                    forward_valid_counts = self.compute_forward_valid_counts_gpu(
                        trajectory_batch, max_depth=None, include_terminal=False
                    )
                    # ``compute_backward_valid_counts_gpu`` expects the
                    # caller-provided forward counts to exclude terminal.
                    backward_valid_counts = self.compute_backward_valid_counts_gpu(
                        trajectory_batch,
                        current_step=step + 1,
                        forward_valid_counts=forward_valid_counts,
                        max_depth=None,
                    )

                if cache_for_flows:
                    # Cache only the ACTIVE rows. ``states_tensor`` stays FULL for
                    # the stable-shape policy forward (graph capture needs the fixed
                    # shape); only the cached copy is sliced, and a row is active here
                    # iff it is valid at this step in ``compute_flows_cached``.
                    # Index contracts: pass per-row GLOBAL ``(batch_idx, meas_idx)``
                    # pairs, not flat positions, and keep ``backward_valid_counts``
                    # FULL ``(B, M)`` — it is indexed by global ``(b, m)``.
                    active_flat = (
                        trajectory_batch.active.reshape(-1).nonzero(as_tuple=True)[0]
                    )
                    trajectory_batch.cache_step_data(
                        step,
                        states_tensor[active_flat],
                        full_indices[active_flat],
                        full_masks,
                        backward_valid_counts,
                    )

                if mode == SamplingMode.ON_POLICY:
                    logits = (
                        gipte_logits
                        if gipte_logits is not None
                        else self._policy_forward_static(states_tensor)
                    )
                    sampled_actions = masked_gumbel_argmax(
                        logits,
                        active_masks,
                        terminal_index=self.terminal_index,
                        use_fused_kernel=self._effective_fused_sampling_kernel(),
                    )
                elif mode == SamplingMode.OFF_POLICY:
                    # Reuse a cached zero buffer (read-only logits) instead
                    # of a fresh torch.zeros_like + zero-fill per step.
                    off_logits = self._offpolicy_zero_logits(
                        active_masks.shape[0], active_masks.shape[1], active_masks.device
                    )
                    sampled_actions = masked_gumbel_argmax(
                        off_logits,
                        active_masks,
                        terminal_index=self.terminal_index,
                        use_fused_kernel=self._effective_fused_sampling_kernel(),
                    )
                else:
                    raise ValueError(f"Static-shape sampling does not support mode={mode}")

                # ``sampled_actions`` covers every B*M row (all-masked inactive rows
                # resolve to ``terminal_index`` in the kernel), so it IS the full
                # action plane — view it directly instead of pre-filling a buffer and
                # copying over it. Both kernel and fallback return a fresh contiguous
                # tensor per call, so there is no cross-step aliasing.
                actions = sampled_actions.view(batch_size, n_measurements)
                trajectory_batch.actions_time_major[step].copy_(actions)

                terminated = self.apply_actions_to_batch(
                    batched_tableau, actions, trajectory_batch, step=step
                )

                newly_terminated = terminated & (trajectory_batch.lengths == 0)
                trajectory_batch.lengths = torch.where(
                    newly_terminated,
                    torch.full_like(trajectory_batch.lengths, step + 1),
                    trajectory_batch.lengths,
                )

                if step == max_length - 1:
                    still_active = trajectory_batch.active & (trajectory_batch.lengths == 0)
                    trajectory_batch.lengths = torch.where(
                        still_active,
                        torch.full_like(trajectory_batch.lengths, max_length),
                        trajectory_batch.lengths,
                    )
                    trajectory_batch.active = torch.where(
                        still_active,
                        torch.zeros_like(trajectory_batch.active),
                        trajectory_batch.active,
                    )

        if self.adaptive_tracker is not None:
            self.adaptive_tracker.update_statistics(trajectory_batch)

        return trajectory_batch

    def _sample_trajectories_bucketed(self,
                                      batch_size: int,
                                      n_measurements: int,
                                      max_depth: int,
                                      mode: SamplingMode,
                                      batch_data_list: Optional[List[Dict]] = None,
                                      cache_for_flows: bool = True,
                                      replay_actions: Optional[Dict[Tuple[int, int], int]] = None
                                      ) -> TrajectoryBatch:
        """Phase-1 bucketed sampler with fixed ``K = B*M``.

        This is the non-graph correctness path: it keeps a stable ordered active
        queue and runs the policy/mask/sampling scratch tensors at fixed
        capacity while caching only real active rows.
        """
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

        if mode not in (SamplingMode.ON_POLICY, SamplingMode.OFF_POLICY):
            raise ValueError(f"Bucketed sampling does not support mode={mode}")
        policy_forward = self._bucketed_policy_forward_impl()

        if self._effective_bucketed_graph():
            try:
                return self._sample_trajectories_bucketed_graph(
                    batch_size=batch_size,
                    n_measurements=n_measurements,
                    max_depth=max_depth,
                    mode=mode,
                    batch_data_list=batch_data_list,
                    cache_for_flows=cache_for_flows,
                    replay_actions=replay_actions,
                )
            except BucketedGraphPreflightError as exc:
                # First-use capability failure (e.g. the fused mask/counts
                # probe fell back before anything shared was mutated): degrade
                # THIS call to the eager bucketed path below; the latch set at
                # the probe site makes _effective_bucketed_graph refuse
                # future calls outright (adversarial gate #4).
                logging.warning(
                    "bucketed graph preflight failed (%s); falling back to "
                    "the eager bucketed sampler", exc
                )

        batched_tableau = self._tableau_cls(
            n_qubits=self.n_qubits,
            batch_size=batch_size,
            n_measurements=n_measurements,
            device=str(self.device)
        )
        self._configure_sampling_tableau(batched_tableau)

        trajectory_batch = TrajectoryBatch(
            batch_size=batch_size,
            n_measurements=n_measurements,
            max_length=max_length,
            n_qubits=self.n_qubits,
            device=self.device
        )
        trajectory_batch.batched_tableau = batched_tableau

        if cache_for_flows:
            trajectory_batch.enable_caching(
                states_uint8=self._effective_uint8_state_cache()
            )

        total_rows = batch_size * n_measurements
        full_bucket = total_rows
        full_indices = self._full_batch_indices(batch_size, n_measurements)
        all_flat = torch.arange(full_bucket, dtype=torch.long, device=self.device)

        # ---- Natural-order full-K loop -----------------------------------------
        # Mirrors the static_full sampler: every step processes ALL B*M rows in place
        # in natural flat order, no compaction, no active-count prefix. ``full_indices``
        # enumerates every (b, m) cell once, satisfying the
        # ``compute_masks_and_counts_fused`` UNIQUE-index precondition by construction.
        # The fused mask engine forces inactive rows to terminal-only, so a terminated
        # row's gumbel output is a no-op under apply's active gating, keeping
        # ``actions_time_major`` byte-identical to the dynamic sampler.
        trajectory_batch.bucketed_active_idx_history = []
        trajectory_batch.bucketed_active_count_history = []
        trajectory_batch.bucketed_row_valid_history = []
        trajectory_batch.bucketed_cached_flat_history = []
        trajectory_batch.bucketed_K = full_bucket
        trajectory_batch.bucketed_K_history = []

        with torch.no_grad():
            for step in range(max_length):
                # ``row_valid``: scattered device mask of still-active rows. The
                # nonzero that materializes the active positions is the only
                # remaining per-step host sync (device-driven termination is
                # piece 4); it runs after the policy forward in spirit, mirroring
                # static_full's active_flat selection.
                row_valid = trajectory_batch.active.reshape(-1)
                active_flat = row_valid.nonzero(as_tuple=True)[0]
                active_count_host = active_flat.numel()
                if active_count_host == 0:
                    break

                states_tensor = self._policy_features(batched_tableau, total_rows)
                if self.device.type in ['cuda', 'mps']:
                    states_tensor = states_tensor.contiguous()

                uniforms = counter_uniforms(
                    seed=self._bucketed_seed_buf,
                    train_step=self._bucketed_train_step_buf,
                    sample_invocation_id=self._bucketed_sample_invocation_buf,
                    ar_step=step,
                    rank=self._bucketed_rank_buf,
                    flat_idx=all_flat,
                    n_actions=self.num_actions,
                    use_fused_kernel=self._effective_fused_counter_rng_kernel(),
                )
                if uniforms.shape != (full_bucket, self.num_actions):
                    raise ValueError(
                        f"counter_uniforms returned shape {tuple(uniforms.shape)}, "
                        f"expected {(full_bucket, self.num_actions)}"
                    )
                if uniforms.dtype != torch.float32:
                    raise TypeError(
                        f"counter_uniforms returned dtype {uniforms.dtype}, "
                        "expected torch.float32"
                    )

                trajectory_batch.bucketed_active_idx_history.append(
                    active_flat.detach().clone()
                )
                trajectory_batch.bucketed_active_count_history.append(active_count_host)
                trajectory_batch.bucketed_K_history.append(full_bucket)
                need_counts = bool(cache_for_flows and step < max_length - 1)
                # Intentional cost: compute masks/counts for the full K=B*M shape
                # to preserve fixed-shape sampling. Boundary-driven K selection is
                # what would remove this full-K work.
                full_masks, _fwd_counts, backward_valid_counts = (
                    self.masking_engine.compute_masks_and_counts_fused(
                        trajectory_batch,
                        full_indices,
                        current_step=step + 1,
                        max_depth=max_depth,
                        compute_backward=need_counts,
                        use_fused_kernel=self._effective_fused_mask_counts_kernel(),
                    )
                )

                if cache_for_flows:
                    trajectory_batch.cache_step_data(
                        step,
                        states_tensor[active_flat],
                        full_indices[active_flat],
                        full_masks[active_flat],
                        backward_valid_counts,
                    )
                    trajectory_batch.bucketed_cached_flat_history.append(
                        active_flat.detach().clone()
                    )

                actions = trajectory_batch.actions_time_major[step]

                if mode == SamplingMode.ON_POLICY:
                    if self.use_bf16_sampling:
                        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                            logits = policy_forward(states_tensor)
                        logits = logits.float()
                    else:
                        logits = policy_forward(states_tensor)

                    # NOTE: pf_model runs on all K=B*M rows incl. terminated lanes
                    # (fixed-shape forward for Phase-3 graph capture). Terminated
                    # lanes carry the engine's terminal-only mask, so their gumbel
                    # action is always terminal and is a no-op under apply.
                    sampled_actions = masked_gumbel_argmax(
                        logits,
                        full_masks,
                        terminal_index=self.terminal_index,
                        uniforms=uniforms,
                        use_fused_kernel=self._effective_fused_sampling_kernel(),
                    )
                else:
                    off_logits = self._offpolicy_zero_logits(
                        full_bucket,
                        self.num_actions,
                        self.device,
                    )
                    sampled_actions = masked_gumbel_argmax(
                        off_logits,
                        full_masks,
                        terminal_index=self.terminal_index,
                        uniforms=uniforms,
                        use_fused_kernel=self._effective_fused_sampling_kernel(),
                    )

                # TEST-ONLY action-replay harness (not on the public
                # sample_trajectories API). The ``.cpu.tolist`` below is a
                # per-step host sync — fine for the parity test, never used in
                # production. In natural order lane == flat id, and only active rows
                # can carry a replay entry, so iterate the active rows.
                if replay_actions is not None:
                    replay_lanes = []
                    replay_values = []
                    for flat_id in active_flat.detach().cpu().tolist():
                        replay_action = replay_actions.get((int(flat_id), step))
                        if replay_action is not None:
                            replay_lanes.append(int(flat_id))
                            replay_values.append(int(replay_action))
                    if replay_lanes:
                        replay_lanes_tensor = torch.as_tensor(
                            replay_lanes, dtype=torch.long, device=self.device
                        )
                        replay_values_tensor = torch.as_tensor(
                            replay_values, dtype=torch.long, device=self.device
                        )
                        sampled_actions[replay_lanes_tensor] = replay_values_tensor

                # Exhaustive write of all B*M cells (terminated lanes already hold
                # terminal). Equivalent to the dynamic sampler's "fill terminal then
                # write active rows" since full_indices covers every cell once.
                actions.view(-1).copy_(sampled_actions)

                terminated = self.apply_actions_to_batch(
                    batched_tableau, actions, trajectory_batch, step=step
                )

                newly_terminated = terminated & (trajectory_batch.lengths == 0)
                trajectory_batch.lengths = torch.where(
                    newly_terminated,
                    step + 1,
                    trajectory_batch.lengths,
                )

                if step == max_length - 1:
                    still_active = trajectory_batch.active & (trajectory_batch.lengths == 0)
                    max_lengths = torch.full_like(trajectory_batch.lengths, max_length)
                    trajectory_batch.lengths = torch.where(
                        still_active,
                        max_lengths,
                        trajectory_batch.lengths,
                    )
                    trajectory_batch.active = torch.where(
                        still_active,
                        torch.zeros_like(trajectory_batch.active),
                        trajectory_batch.active,
                    )

        if self.adaptive_tracker is not None:
            self.adaptive_tracker.update_statistics(trajectory_batch)

        # Advance the per-call RNG stream AFTER sampling: the REPLAY early-return above never reaches
        # here (so a REPLAY call does not consume an invocation id), and the id USED during this call
        # equals the buffer value the caller set (no off-by-one). counter_uniforms keys on this id, so
        # each consuming sampler call gets a distinct, reproducible stream.
        with torch.no_grad():
            self._bucketed_sample_invocation_buf.add_(1)

        return trajectory_batch

    @staticmethod
    def _smallest_pow2_bucket(n: int, cap: int) -> int:
        """Smallest power-of-two bucket >= max(n, 1), capped at ``cap``.

        The bucket SET is powers of two so the eventual Phase-3 graph capture has
        a small, fixed family of capacities (one captured graph per K). Over-
        allocates by < 2x vs the live active count -- bounded GEMM waste that the
        Gate-0 histogram can later tune to a finer schedule.
        """
        n = max(int(n), 1)
        return min(1 << (n - 1).bit_length(), int(cap))

    def _sample_trajectories_bucketed_compacted(self,
                                                batch_size: int,
                                                n_measurements: int,
                                                max_depth: int,
                                                mode: SamplingMode,
                                                batch_data_list: Optional[List[Dict]] = None,
                                                cache_for_flows: bool = True,
                                                replay_actions: Optional[Dict[Tuple[int, int], int]] = None
                                                ) -> TrajectoryBatch:
        """Phase-2 compacted bucketed sampler: partition queue, fixed [K].

        The active queue is a PERMUTATION of all B*M flats (survivors front,
        losers back) maintained by the sync-free ordered_partition_scatter.
        Policy/mask/sample run on the FIXED [K] queue prefix; loser lanes
        [active_count:K] are terminated rows the mask engine forces to
        terminal-only, so they are no-ops under apply. K = full_bucket here
        (2a); boundary-driven K shrinking lands in 2b. counter_uniforms
        is keyed by the queue's flat ids, so it is invariant to queue position.
        The active-only cache (survivors = active_idx[:active_count],
        ascending) is byte-exact vs the dynamic sampler. This is the
        graph-eligible path; _sample_trajectories_bucketed (natural-order)
        stays the non-graph fallback.
        """
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

        if mode not in (SamplingMode.ON_POLICY, SamplingMode.OFF_POLICY):
            raise ValueError(f"Bucketed sampling does not support mode={mode}")

        batched_tableau = self._tableau_cls(
            n_qubits=self.n_qubits,
            batch_size=batch_size,
            n_measurements=n_measurements,
            device=str(self.device)
        )
        self._configure_sampling_tableau(batched_tableau)

        trajectory_batch = TrajectoryBatch(
            batch_size=batch_size,
            n_measurements=n_measurements,
            max_length=max_length,
            n_qubits=self.n_qubits,
            device=self.device
        )
        trajectory_batch.batched_tableau = batched_tableau

        if cache_for_flows:
            trajectory_batch.enable_caching(
                states_uint8=self._effective_uint8_state_cache()
            )

        total_rows = batch_size * n_measurements
        full_bucket = total_rows
        full_indices = self._full_batch_indices(batch_size, n_measurements)

        # Active queue = permutation of 0..full_bucket-1 (survivors front, losers
        # back). Maintained by ordered_partition_scatter; active_idx[:active_count]
        # is the active flats in ASCENDING order.
        active_idx = torch.arange(full_bucket, dtype=torch.long, device=self.device)
        next_active_idx = torch.zeros_like(active_idx)
        active_count_dev = torch.tensor(full_bucket, dtype=torch.long, device=self.device)
        next_count_scalar = torch.zeros((), dtype=torch.long, device=self.device)
        active_count_host = full_bucket
        arange_full = torch.arange(full_bucket, device=self.device)
        # Phase-3 freeze-hazard prep: ar_step is a DEVICE scalar (not the Python
        # ``step`` int) so a future captured graph reads it live and a per-replay
        # ``add_(1)`` advances the counter-RNG stream. Parity-neutral now: it holds
        # ``step`` each iteration (starts 0, +1 at loop end), so counter_uniforms
        # keys on the identical value the int path used.
        ar_step_dev = torch.zeros((), dtype=torch.long, device=self.device)
        # Phase-4 fused-tail scratch: the CT partition/update kernel reads a
        # step+1 device scalar and writes an entering-count snapshot (unused
        # by the eager path). Resolved once; a mid-run hard failure flips the
        # flag off and the torch chain takes over (latch discipline).
        use_fused_partition_update = self._effective_fused_partition_update_kernel()
        step_plus1_dev = torch.zeros((), dtype=torch.long, device=self.device)
        entering_scratch = torch.zeros((), dtype=torch.long, device=self.device)

        trajectory_batch.bucketed_active_idx_history = []
        trajectory_batch.bucketed_active_count_history = []
        trajectory_batch.bucketed_row_valid_history = []
        trajectory_batch.bucketed_cached_flat_history = []
        trajectory_batch.bucketed_K = full_bucket
        trajectory_batch.bucketed_K_history = []

        # ---- Phase-3 capture-ready loop ----------------------------------------
        # Host-read the active count ONLY at window boundaries, so a captured graph
        # can replay without the host stalling between replays; in between,
        # ``active_count_dev`` drives row_valid / survived_full with no host sync.
        # Cache slices and observability history are deferred to the window flush,
        # keyed on stored per-step DEVICE counts (one batched sync per window).
        # Three parity traps the boundary granularity introduces, handled at flush:
        #   (1) active_idx[:K] is a VIEW into the double-buffered queue -> CLONED.
        #   (2) the loop may run no-op steps past full termination -> steps whose
        #       entering active_count is 0 are DROPPED so cached_states matches.
        #   (3) those no-op steps fill actions_time_major with terminal, while the
        #       dynamic sampler leaves them zeroed -> zeroed post-loop.
        pending = []  # (step, idx_full, indices_full, states_full, masks_full, backward, count_dev, K)
        K = full_bucket
        window = self._bucketed_k_window
        kept = 0
        iterations_run = 0

        def flush_pending_cache() -> None:
            nonlocal kept, pending
            if not pending:
                return
            # ONE batched device->host sync resolves the whole window's
            # entering counts. INVARIANT: counts only decrease within a
            # window (termination is one-way), so count <= the [K] length
            # of every stored full-prefix tensor below.
            counts_host = torch.stack([p[6] for p in pending]).cpu().tolist()
            for (
                p_step,
                idx_full,
                indices_full,
                states_full,
                masks_full,
                p_backward,
                _count_dev,
                p_K,
            ), count in zip(pending, counts_host):
                count = int(count)
                if count == 0:
                    continue
                kept += 1
                # Slice-and-clone down to the entering count so the cache
                # retains count-sized tensors (the full-K pending bases are
                # freed when ``pending`` clears), matching the pre-deferral
                # cache memory shape.
                active_flat = idx_full[:count].clone()
                trajectory_batch.bucketed_active_idx_history.append(active_flat)
                trajectory_batch.bucketed_active_count_history.append(count)
                trajectory_batch.bucketed_K_history.append(p_K)
                if cache_for_flows:
                    trajectory_batch.cache_step_data(
                        p_step,
                        states_full[:count].clone(),
                        indices_full[:count].clone(),
                        masks_full[:count].clone(),
                        p_backward,
                    )
                    trajectory_batch.bucketed_cached_flat_history.append(
                        active_flat.clone()
                    )
            pending = []

        with torch.no_grad():
            for step in range(max_length):
                if step == 0 or (step % window == 0):
                    flush_pending_cache()
                    active_count_host = int(active_count_dev.item())
                    if active_count_host == 0:
                        break
                    if step > 0:
                        K = min(K, self._smallest_pow2_bucket(active_count_host, full_bucket))
                    assert active_count_host <= K, (
                        "bucketed-K overflow", active_count_host, K, step
                    )

                idx_k = active_idx[:K]
                indices_k = full_indices.index_select(0, idx_k)

                full_rows = self._policy_features(batched_tableau, total_rows)
                states_tensor = full_rows.index_select(0, idx_k)
                if self.device.type in ['cuda', 'mps']:
                    states_tensor = states_tensor.contiguous()

                uniforms = counter_uniforms(
                    seed=self._bucketed_seed_buf,
                    train_step=self._bucketed_train_step_buf,
                    sample_invocation_id=self._bucketed_sample_invocation_buf,
                    ar_step=ar_step_dev,
                    rank=self._bucketed_rank_buf,
                    flat_idx=idx_k,
                    n_actions=self.num_actions,
                    use_fused_kernel=self._effective_fused_counter_rng_kernel(),
                )
                if uniforms.shape != (K, self.num_actions):
                    raise ValueError(
                        f"counter_uniforms returned shape {tuple(uniforms.shape)}, "
                        f"expected {(K, self.num_actions)}"
                    )
                if uniforms.dtype != torch.float32:
                    raise TypeError(
                        f"counter_uniforms returned dtype {uniforms.dtype}, "
                        "expected torch.float32"
                    )

                need_counts = bool(cache_for_flows and step < max_length - 1)
                full_masks, _fwd_counts, backward_valid_counts = (
                    self.masking_engine.compute_masks_and_counts_fused(
                        trajectory_batch,
                        indices_k,
                        current_step=step + 1,
                        max_depth=max_depth,
                        compute_backward=need_counts,
                        use_fused_kernel=self._effective_fused_mask_counts_kernel(),
                    )
                )

                # Defer host-observable cache/history to the window flush.
                # states_tensor / indices_k / full_masks are FRESH per-step tensors,
                # so they defer with no copy. idx_k is a VIEW into the double-buffered
                # queue (trap 1) and is cloned; the entering count stays on DEVICE and
                # is resolved at the flush in one batched sync. Pending briefly holds
                # [K]-sized tensors for at most ``window`` steps.
                pending.append((
                    step,
                    idx_k.detach().clone(),
                    indices_k,
                    states_tensor if cache_for_flows else None,
                    full_masks if cache_for_flows else None,
                    backward_valid_counts,
                    active_count_dev.clone(),
                    K,
                ))
                iterations_run += 1

                actions = trajectory_batch.actions_time_major[step]
                actions.fill_(self.terminal_index)

                if mode == SamplingMode.ON_POLICY:
                    if self.use_bf16_sampling:
                        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                            logits = self.pf_model(states_tensor)
                        logits = logits.float()
                    else:
                        logits = self.pf_model(states_tensor)

                    sampled_actions = masked_gumbel_argmax(
                        logits,
                        full_masks,
                        terminal_index=self.terminal_index,
                        uniforms=uniforms,
                        use_fused_kernel=self._effective_fused_sampling_kernel(),
                    )
                else:
                    off_logits = self._offpolicy_zero_logits(
                        K,
                        self.num_actions,
                        self.device,
                    )
                    sampled_actions = masked_gumbel_argmax(
                        off_logits,
                        full_masks,
                        terminal_index=self.terminal_index,
                        uniforms=uniforms,
                        use_fused_kernel=self._effective_fused_sampling_kernel(),
                    )

                # TEST-ONLY action-replay harness. Iterate the FULL [K] queue prefix
                # (lane == queue position == flat idx_k[lane]); only active survivors
                # carry a replay entry, so no active-count read is needed. idx_k is
                # read NOW, before the partition swap, while the view is valid.
                if replay_actions is not None:
                    replay_lanes = []
                    replay_values = []
                    for lane, flat_id in enumerate(idx_k.detach().cpu().tolist()):
                        replay_action = replay_actions.get((int(flat_id), step))
                        if replay_action is not None:
                            replay_lanes.append(lane)
                            replay_values.append(int(replay_action))
                    if replay_lanes:
                        replay_lanes_tensor = torch.as_tensor(
                            replay_lanes, dtype=torch.long, device=self.device
                        )
                        replay_values_tensor = torch.as_tensor(
                            replay_values, dtype=torch.long, device=self.device
                        )
                        sampled_actions[replay_lanes_tensor] = replay_values_tensor

                actions[indices_k[:, 0], indices_k[:, 1]] = sampled_actions

                terminated = self.apply_actions_to_batch(
                    batched_tableau, actions, trajectory_batch, step=step
                )

                if not use_fused_partition_update:
                    # Torch chain: lengths update BEFORE the last-step fixup. The
                    # fused tail folds this into the kernel AFTER the fixup —
                    # equivalent, because newly-terminated rows are already inactive
                    # and fixed-up rows get nonzero lengths.
                    newly_terminated = terminated & (trajectory_batch.lengths == 0)
                    trajectory_batch.lengths = torch.where(
                        newly_terminated,
                        step + 1,
                        trajectory_batch.lengths,
                    )

                if step == max_length - 1:
                    still_active = trajectory_batch.active & (trajectory_batch.lengths == 0)
                    max_lengths = torch.full_like(trajectory_batch.lengths, max_length)
                    trajectory_batch.lengths = torch.where(
                        still_active,
                        max_lengths,
                        trajectory_batch.lengths,
                    )
                    trajectory_batch.active = torch.where(
                        still_active,
                        torch.zeros_like(trajectory_batch.active),
                        trajectory_batch.active,
                    )

                fused_done = False
                if use_fused_partition_update:
                    # Phase-4 CT fused tail (see the graph path for the full
                    # contract): one launch for survivor detection, the
                    # newly-terminated lengths update, the stable partition,
                    # and the count/step scalar updates.
                    step_plus1_dev.fill_(step + 1)
                    fused_done = _partition_update_bucketed_torch(
                        trajectory_batch.active.reshape(-1),
                        active_idx,
                        next_active_idx,
                        active_count_dev,
                        entering_scratch,
                        trajectory_batch.lengths.view(-1),
                        step_plus1_dev,
                        ar_step_dev,
                        use_fused_kernel=True,
                    ) is True
                    if fused_done:
                        active_idx, next_active_idx = next_active_idx, active_idx
                    else:
                        # Hard failure latched mid-run: redo the lengths
                        # update the fused branch deferred, then fall through
                        # to the torch chain for this and later steps.
                        use_fused_partition_update = False
                        newly_terminated = terminated & (trajectory_batch.lengths == 0)
                        trajectory_batch.lengths = torch.where(
                            newly_terminated,
                            step + 1,
                            trajectory_batch.lengths,
                        )
                if not fused_done:
                    survived_full = (
                        trajectory_batch.active.reshape(-1).index_select(0, active_idx)
                        & (arange_full < active_count_dev)
                    )
                    ordered_partition_scatter(
                        active_idx, survived_full, next_active_idx, next_count_scalar
                    )
                    active_idx, next_active_idx = next_active_idx, active_idx
                    active_count_dev.copy_(next_count_scalar)
                    ar_step_dev.add_(1)

        # ---- Post-loop: active-only cache + observability history ------------
        # Flush the final window. Steps with entering count 0 (no-op steps past
        # full termination) are DROPPED so cached/history lengths match the
        # dynamic sampler. Window flushing bounds transient full-K retention to
        # ``_bucketed_k_window`` steps instead of the whole trajectory.
        flush_pending_cache()
        # Trap (3): zero the actions of any no-op steps past termination (the
        # dynamic sampler leaves them at the zeros init; we filled them terminal).
        if kept < max_length:
            trajectory_batch.actions_time_major[kept:].zero_()
        # Test-only observability fields for the drop-empties path.
        trajectory_batch.bucketed_iterations_run = iterations_run
        trajectory_batch.bucketed_kept_steps = kept

        if self.adaptive_tracker is not None:
            self.adaptive_tracker.update_statistics(trajectory_batch)

        # Advance the per-call RNG stream AFTER sampling (REPLAY early-return never
        # reaches here; the id USED equals the caller-set buffer value).
        with torch.no_grad():
            self._bucketed_sample_invocation_buf.add_(1)

        return trajectory_batch

    def _sample_trajectories_bucketed_graph(self,
                                            batch_size: int,
                                            n_measurements: int,
                                            max_depth: int,
                                            mode: SamplingMode,
                                            batch_data_list: Optional[List[Dict]] = None,
                                            cache_for_flows: bool = True,
                                            replay_actions: Optional[Dict[Tuple[int, int], int]] = None
                                            ) -> TrajectoryBatch:
        """Bucketed sampler with per-K CUDA graphs and reusable LRU entries.

        The graph path is for production sampling only. The test-only
        ``replay_actions`` harness intentionally falls back to the eager
        compacted sampler, whose Python-side replay injection is not part of the
        captured production loop. Captured state is cached internally, but the
        returned batch is a snapshot so callers do not alias the next same-shape
        graph replay.
        """
        if replay_actions is not None:
            return self._sample_trajectories_bucketed_compacted(
                batch_size=batch_size,
                n_measurements=n_measurements,
                max_depth=max_depth,
                mode=mode,
                batch_data_list=batch_data_list,
                cache_for_flows=cache_for_flows,
                replay_actions=replay_actions,
            )

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

        if mode not in (SamplingMode.ON_POLICY, SamplingMode.OFF_POLICY):
            raise ValueError(f"Bucketed graph sampling does not support mode={mode}")

        total_rows = batch_size * n_measurements
        if total_rows == 0:
            return self._sample_trajectories_bucketed_compacted(
                batch_size=batch_size,
                n_measurements=n_measurements,
                max_depth=max_depth,
                mode=mode,
                batch_data_list=batch_data_list,
                cache_for_flows=cache_for_flows,
                replay_actions=None,
            )

        if self.device.type != "cuda" or not torch.cuda.is_available():
            return self._sample_trajectories_bucketed_compacted(
                batch_size=batch_size,
                n_measurements=n_measurements,
                max_depth=max_depth,
                mode=mode,
                batch_data_list=batch_data_list,
                cache_for_flows=cache_for_flows,
                replay_actions=None,
            )

        full_bucket = total_rows

        with torch.cuda.device(self.device):
            if not self._effective_fused_apply_kernel():
                raise RuntimeError(
                    "bucketed CUDA graph sampling requires the fused CT apply kernel"
                )
            if self.feature_extractor is not None or self.packed_w_input:
                raise RuntimeError(
                    "bucketed per-K graph capture supports only the default flattened-W "
                    "feature mode (feature_extractor is None and packed_w_input is False); "
                    "use the eager compacted or dynamic sampler for GIPTE/packed-W models"
                )
            use_fused_mask_counts = self._effective_fused_mask_counts_kernel()
            if not use_fused_mask_counts:
                raise RuntimeError(
                    "bucketed CUDA graph sampling requires the fused mask/counts kernel"
                )
            use_fused_sampling = self._effective_fused_sampling_kernel()
            if use_fused_sampling:
                probe_logits = torch.zeros(
                    (1, self.num_actions),
                    dtype=torch.float32,
                    device=self.device,
                )
                probe_masks = torch.ones(
                    (1, self.num_actions),
                    dtype=torch.bool,
                    device=self.device,
                )
                probe_uniforms = torch.full_like(probe_logits, 0.5)
                masked_gumbel_argmax(
                    probe_logits,
                    probe_masks,
                    terminal_index=self.terminal_index,
                    uniforms=probe_uniforms,
                    use_fused_kernel=True,
                )
                use_fused_sampling = self._effective_fused_sampling_kernel()

            # Counter-RNG first-use probe: latch a CuPy/NVRTC failure NOW so
            # capture below bakes the torch fallback knowingly (the wrapper
            # would fall back internally either way; probing keeps the
            # cache_key honest about which path the graph captured).
            use_fused_counter_rng = self._effective_fused_counter_rng_kernel()
            if use_fused_counter_rng:
                counter_uniforms(
                    seed=self._bucketed_seed_buf,
                    train_step=self._bucketed_train_step_buf,
                    sample_invocation_id=self._bucketed_sample_invocation_buf,
                    ar_step=0,
                    rank=self._bucketed_rank_buf,
                    flat_idx=torch.zeros(
                        (1,), dtype=torch.long, device=self.device
                    ),
                    n_actions=self.num_actions,
                    use_fused_kernel=True,
                )
                use_fused_counter_rng = self._effective_fused_counter_rng_kernel()

            # Phase-4 CT fused partition/update probe: force the CT import +
            # NVRTC compile + first launch NOW (eager), so capture never bakes
            # a half-initialized fast path and a broken host latches off
            # before the cache key is computed.
            use_fused_partition_update = (
                self._effective_fused_partition_update_kernel()
            )
            if use_fused_partition_update:
                probe_ok = _partition_update_bucketed_torch(
                    torch.ones(1, dtype=torch.bool, device=self.device),
                    torch.zeros(1, dtype=torch.long, device=self.device),
                    torch.zeros(1, dtype=torch.long, device=self.device),
                    torch.ones((), dtype=torch.long, device=self.device),
                    torch.zeros((), dtype=torch.long, device=self.device),
                    torch.zeros(1, dtype=torch.long, device=self.device),
                    torch.ones((), dtype=torch.long, device=self.device),
                    torch.zeros((), dtype=torch.long, device=self.device),
                    use_fused_kernel=True,
                )
                use_fused_partition_update = (
                    probe_ok is True
                    and self._effective_fused_partition_update_kernel()
                )

            cuda_index = (
                self.device.index
                if self.device.index is not None
                else torch.cuda.current_device()
            )
            current_policy_orig = self._bucketed_policy_forward_impl()
            cache_key = (
                cuda_index,
                batch_size,
                n_measurements,
                max_depth,
                max_length,
                mode.value,
                bool(cache_for_flows),
                bool(self.use_bf16_sampling),
                bool(use_fused_sampling),
                bool(use_fused_mask_counts),
                bool(use_fused_counter_rng),
                bool(use_fused_partition_update),
                int(self.num_actions),
                id(current_policy_orig),
                bool(current_policy_orig.training),
            )
            # max_length is in the key (not just max_depth):
            # buffer_strategy='adaptive' makes determine_buffer_size(max_depth)
            # drift, so each distinct static buffer shape gets its own entry --
            # adaptive growth shows up as a cache miss + re-capture, which is
            # why the hit path below needs no stale-shape check.

            def _bucketed_graph_entry() -> Dict[str, Any]:
                try:
                    import cupy as cp
                except ImportError as e:
                    # Pre-mutation capability failure -> preflight class, same
                    # rationale as the mask/counts probe below.
                    raise BucketedGraphPreflightError(
                        "bucketed CUDA graph sampling requires CuPy for the stable W "
                        "unpack and reset buffers"
                    ) from e

                batched_tableau = self._tableau_cls(
                    n_qubits=self.n_qubits,
                    batch_size=batch_size,
                    n_measurements=n_measurements,
                    device=str(self.device),
                )
                self._configure_sampling_tableau(batched_tableau)

                if not hasattr(batched_tableau, "reset_inplace_with_mask"):
                    raise RuntimeError(
                        "bucketed CUDA-graph cache requires "
                        "batched_tableau.reset_inplace_with_mask "
                        "(the CT TableauBatchAdapter)"
                    )
                if not hasattr(batched_tableau, "_sim"):
                    raise RuntimeError(
                        "bucketed CUDA graph sampling requires the CT tableau backend "
                        "with a fused-apply _sim surface"
                    )

                trajectory_batch = TrajectoryBatch(
                    batch_size=batch_size,
                    n_measurements=n_measurements,
                    max_length=max_length,
                    n_qubits=self.n_qubits,
                    device=self.device,
                )
                trajectory_batch.batched_tableau = batched_tableau

                full_indices = self._full_batch_indices(batch_size, n_measurements)

                active_idx = torch.arange(
                    full_bucket, dtype=torch.long, device=self.device
                )
                next_active_idx = torch.zeros_like(active_idx)
                active_count_dev = torch.tensor(
                    full_bucket, dtype=torch.long, device=self.device
                )
                next_count_scalar = torch.zeros(
                    (), dtype=torch.long, device=self.device
                )
                entering_count_dev = torch.zeros(
                    (), dtype=torch.long, device=self.device
                )
                arange_full = torch.arange(
                    full_bucket, dtype=torch.long, device=self.device
                )
                device_step = torch.zeros((), dtype=torch.long, device=self.device)
                length_step_dev = torch.zeros(
                    (), dtype=torch.long, device=self.device
                )
                _fused_apply_adapter.mark_trusted_device_step_tensor(device_step)

                setup_stream = torch.cuda.current_stream(self.device)
                with cp.cuda.Device(cuda_index), cp.cuda.ExternalStream(setup_stream.cuda_stream):
                    w_unpacked_buf = cp.empty(
                        (total_rows, 2 * self.n_qubits, 2 * self.n_qubits),
                        dtype=cp.uint8,
                    )
                    reset_mask = cp.ones(total_rows, dtype=cp.bool_)
                setup_stream.synchronize()

                probe_counts = _compute_mask_counts_fused(
                    trajectory_batch,
                    full_indices,
                    masking_engine=self.masking_engine,
                    current_step=1,
                    max_depth=max_depth,
                    compute_backward=cache_for_flows,
                    use_fused_kernel=use_fused_mask_counts,
                )
                if probe_counts is None:
                    # First-use failure: the knob was on and no latch had tripped,
                    # but the kernel's first real launch fell back. Latch it and raise
                    # the PREFLIGHT class so this call also degrades to the eager
                    # sampler — nothing shared has been mutated yet.
                    self._fused_mask_counts_kernel_failed = True
                    raise BucketedGraphPreflightError(
                        "bucketed CUDA graph sampling requires the fused "
                        "mask/counts kernel; the pre-capture count probe fell back"
                    )

                # CT fused-apply first-use preflight. Otherwise the kernel's first
                # real launch happens during warmup, where a hard failure silently
                # latches and completes via the legacy chain, then hits the
                # post-warmup hard raise mid-construction. Probe here instead: one
                # all-terminal layer through apply_action_layer_fused with no CT-side
                # mutation, on a throwaway TrajectoryBatch, tableau reset after. Soft
                # (non-latching) failures deliberately pass.
                probe_tb = TrajectoryBatch(
                    batch_size=batch_size,
                    n_measurements=n_measurements,
                    max_length=max_length,
                    n_qubits=self.n_qubits,
                    device=self.device,
                )
                probe_tb.batched_tableau = batched_tableau
                probe_actions = torch.full(
                    (batch_size, n_measurements),
                    self.terminal_index,
                    dtype=torch.long,
                    device=self.device,
                )
                self.apply_actions_to_batch(
                    batched_tableau, probe_actions, probe_tb, step=0
                )
                if (
                    self._fused_apply_kernel_failed
                    or not self._effective_fused_apply_kernel()
                ):
                    raise BucketedGraphPreflightError(
                        "bucketed CUDA graph sampling requires the CT fused "
                        "apply kernel; its first-use probe latched off"
                    )
                batched_tableau.reset_inplace_with_mask(reset_mask)
                del probe_tb, probe_actions

                feature_probe = batched_tableau.to_flat_tensors_into(w_unpacked_buf)
                feature_tail_shape = tuple(feature_probe.shape[1:])
                feature_dtype = feature_probe.dtype
                del feature_probe

                bucket_sizes = {full_bucket}
                power = 1
                while power < full_bucket:
                    bucket_sizes.add(power)
                    power <<= 1
                bucket_sizes = sorted(bucket_sizes, reverse=True)

                graph_buckets = OrderedDict()
                _graph = {}
                _graph_inputs = {}
                _graph_masks = {}
                for bucket_K in bucket_sizes:
                    actions_buf = torch.empty(
                        (batch_size, n_measurements),
                        dtype=torch.long,
                        device=self.device,
                    )
                    graph_inputs = torch.empty(
                        (bucket_K, *feature_tail_shape),
                        dtype=feature_dtype,
                        device=self.device,
                    )
                    graph_masks = torch.empty(
                        (bucket_K, self.num_actions),
                        dtype=torch.bool,
                        device=self.device,
                    )
                    bucket = {
                        "K": bucket_K,
                        "arange": torch.arange(
                            bucket_K, dtype=torch.long, device=self.device
                        ),
                        "_graph_inputs": graph_inputs,
                        "_graph_masks": graph_masks,
                        "states": graph_inputs,
                        "idx": torch.empty(
                            (bucket_K,), dtype=torch.long, device=self.device
                        ),
                        "indices": torch.empty(
                            (bucket_K, 2), dtype=torch.long, device=self.device
                        ),
                        "row_valid": torch.empty(
                            (bucket_K,), dtype=torch.bool, device=self.device
                        ),
                        "masks": graph_masks,
                        "backward": torch.empty(
                            (batch_size, n_measurements),
                            dtype=torch.long,
                            device=self.device,
                        ),
                        "uniforms": torch.empty(
                            (bucket_K, self.num_actions),
                            dtype=torch.float32,
                            device=self.device,
                        ),
                        "logits": torch.empty(
                            (bucket_K, self.num_actions),
                            dtype=torch.float32,
                            device=self.device,
                        ),
                        "sampled": torch.empty(
                            (bucket_K,), dtype=torch.long, device=self.device
                        ),
                        "survived": torch.empty(
                            (bucket_K,), dtype=torch.bool, device=self.device
                        ),
                        "actions": actions_buf,
                        "actions_flat": actions_buf.view(-1),
                        "_graph": None,
                    }
                    graph_buckets[bucket_K] = bucket
                    _graph_inputs[bucket_K] = graph_inputs
                    _graph_masks[bucket_K] = graph_masks

                policy_orig = current_policy_orig

                def write_logits(bucket) -> None:
                    graph_inputs = bucket["_graph_inputs"]
                    if mode == SamplingMode.ON_POLICY:
                        if self.use_bf16_sampling:
                            with torch.autocast(
                                device_type="cuda",
                                dtype=torch.bfloat16,
                                cache_enabled=False,
                            ):
                                logits_tmp = policy_orig(graph_inputs)
                            bucket["logits"].copy_(logits_tmp.float())
                        else:
                            bucket["logits"].copy_(policy_orig(graph_inputs))
                    else:
                        bucket["logits"].zero_()

                def reset_run_state(active_count_reset: int = full_bucket) -> None:
                    # Hot per-call reset (cache-hit replay path). Captured CT
                    # kernels bind _sim state addresses, and the in-place masked reset
                    # preserves them across all per-K graphs. Per-bucket scratch
                    # buffers are deliberately NOT touched: each is written inside
                    # compute_graph_step before any read, so zeroing them would only
                    # add dead launches.
                    batched_tableau.reset_inplace_with_mask(reset_mask)
                    active_idx.copy_(arange_full)
                    next_active_idx.zero_()
                    active_count_dev.fill_(active_count_reset)
                    next_count_scalar.zero_()
                    entering_count_dev.zero_()
                    device_step.zero_()
                    length_step_dev.zero_()
                    trajectory_batch.actions_time_major.zero_()
                    trajectory_batch.lengths.zero_()
                    trajectory_batch.active.fill_(True)
                    trajectory_batch.masks.fill_(True)
                    trajectory_batch.circuit_depths.zero_()
                    trajectory_batch.current_layer_qubits.zero_()
                    trajectory_batch.qubit_last_layer.fill_(-1)
                    trajectory_batch.last_single_qubit_gates.fill_(-1)
                    trajectory_batch.qubit_last_use_step.fill_(-1)
                    trajectory_batch.action_qubits.fill_(-1)
                    trajectory_batch._terminated_buffer_idx = 0
                    for terminated_buf in trajectory_batch._terminated_buffers:
                        terminated_buf.zero_()

                def reset_graph_state(active_count_reset: int = full_bucket) -> None:
                    # Full reset (warmup/capture hygiene): run state plus a
                    # deterministic zero of every per-bucket scratch buffer so
                    # capture never records reads of stale warmup garbage.
                    reset_run_state(active_count_reset)
                    for bucket in graph_buckets.values():
                        K_bucket = bucket["K"]
                        bucket["idx"].copy_(arange_full[:K_bucket])
                        bucket["indices"].copy_(full_indices[:K_bucket])
                        bucket["row_valid"].zero_()
                        bucket["_graph_inputs"].zero_()
                        bucket["_graph_masks"].zero_()
                        bucket["backward"].zero_()
                        bucket["uniforms"].zero_()
                        bucket["logits"].zero_()
                        bucket["sampled"].zero_()
                        bucket["survived"].zero_()
                        bucket["actions"].zero_()

                def compute_graph_step(bucket) -> None:
                    K_bucket = bucket["K"]
                    idx_k_buf = bucket["idx"]
                    indices_k_buf = bucket["indices"]
                    row_valid_buf = bucket["row_valid"]
                    full_masks_buf = bucket["_graph_masks"]
                    uniforms_buf = bucket["uniforms"]
                    logits_buf = bucket["logits"]
                    sampled_actions_buf = bucket["sampled"]
                    survived_full_buf = bucket["survived"]
                    actions_buf = bucket["actions"]
                    actions_flat_buf = bucket["actions_flat"]
                    active_idx_k = active_idx[:K_bucket]
                    next_active_idx_k = next_active_idx[:K_bucket]

                    if not use_fused_partition_update:
                        # The fused tail writes the entering snapshot itself
                        # (same value: the count is stable within a step).
                        entering_count_dev.copy_(active_count_dev)
                    idx_k_buf.copy_(active_idx_k)
                    torch.div(
                        idx_k_buf,
                        n_measurements,
                        rounding_mode="floor",
                        out=indices_k_buf[:, 0],
                    )
                    torch.remainder(idx_k_buf, n_measurements, out=indices_k_buf[:, 1])
                    torch.lt(bucket["arange"], active_count_dev, out=row_valid_buf)

                    # Step-entry features are computed INSIDE the captured region:
                    # the unpack reads the persistent _sim state the captured apply
                    # mutates in place, so replay N sees what replay N-1 wrote. After
                    # a replay the stable buffers still hold the step-entry values the
                    # policy consumed, which is where the eager cache clones read.
                    full_rows = batched_tableau.to_flat_tensors_into(w_unpacked_buf)
                    torch.index_select(
                        full_rows, 0, idx_k_buf, out=bucket["_graph_inputs"]
                    )

                    # ONE fused mask/counts launch per step, also captured. Call the
                    # kernel module directly: the engine wrapper routes
                    # compute_backward=False to the legacy multi-launch torch path,
                    # and a silent fallback inside capture would bake dead graph
                    # nodes. ``current_step`` is the step+1 DEVICE scalar, matching
                    # the eager compacted sampler.
                    torch.add(device_step, 1, out=length_step_dev)
                    mask_counts = _compute_mask_counts_fused(
                        trajectory_batch,
                        indices_k_buf,
                        masking_engine=self.masking_engine,
                        current_step=length_step_dev,
                        max_depth=max_depth,
                        compute_backward=bool(cache_for_flows),
                        use_fused_kernel=use_fused_mask_counts,
                    )
                    if mask_counts is None:
                        # Unreachable after the pre-capture probe; a raise
                        # here surfaces during warmup/capture, never replay.
                        raise RuntimeError(
                            "bucketed graph capture requires the fused "
                            "mask/counts kernel; it fell back during "
                            "warmup/capture"
                        )
                    masks_tmp = mask_counts[0]
                    backward_counts_tmp = mask_counts[2]
                    full_masks_buf.copy_(masks_tmp)
                    if backward_counts_tmp is not None:
                        bucket["backward"].copy_(backward_counts_tmp)

                    counter_uniforms(
                        seed=self._bucketed_seed_buf,
                        train_step=self._bucketed_train_step_buf,
                        sample_invocation_id=self._bucketed_sample_invocation_buf,
                        ar_step=device_step,
                        rank=self._bucketed_rank_buf,
                        flat_idx=idx_k_buf,
                        n_actions=self.num_actions,
                        use_fused_kernel=use_fused_counter_rng,
                        out=uniforms_buf,
                    )

                    write_logits(bucket)
                    sampled_actions_buf.copy_(
                        masked_gumbel_argmax(
                            logits_buf,
                            full_masks_buf,
                            terminal_index=self.terminal_index,
                            uniforms=uniforms_buf,
                            use_fused_kernel=use_fused_sampling,
                        )
                    )

                    actions_buf.fill_(self.terminal_index)
                    actions_flat_buf.scatter_(0, idx_k_buf, sampled_actions_buf)

                    terminated = self.apply_actions_to_batch(
                        batched_tableau,
                        actions_buf,
                        trajectory_batch,
                        step=device_step,
                    )

                    if use_fused_partition_update:
                        # Phase-4 CT fused tail: ONE launch covers survivor
                        # detection, the newly-terminated lengths update, the stable
                        # queue partition, and the count/entering/device_step scalar
                        # updates. ``terminated`` from the apply is deliberately
                        # unused — a lane is newly terminated iff its row went
                        # inactive with lengths==0, which the kernel derives from the
                        # post-apply active flags.
                        fused_ok = _partition_update_bucketed_torch(
                            trajectory_batch.active.reshape(-1),
                            active_idx_k,
                            next_active_idx_k,
                            active_count_dev,
                            entering_count_dev,
                            trajectory_batch.lengths.view(-1),
                            length_step_dev,
                            device_step,
                            use_fused_kernel=True,
                        )
                        if fused_ok is not True:
                            # Unreachable after the pre-capture probe; raising
                            # here surfaces during warmup/capture, never replay.
                            raise RuntimeError(
                                "bucketed graph capture: the CT fused "
                                "partition-update fell back during "
                                "warmup/capture"
                            )
                        active_idx_k.copy_(next_active_idx_k)
                    else:
                        # length_step_dev already holds step+1 (written before
                        # the mask/counts call above; device_step is unchanged
                        # since).
                        newly_terminated = terminated & (trajectory_batch.lengths == 0)
                        trajectory_batch.lengths.copy_(
                            torch.where(
                                newly_terminated,
                                length_step_dev,
                                trajectory_batch.lengths,
                            )
                        )

                        torch.index_select(
                            trajectory_batch.active.reshape(-1),
                            0,
                            active_idx_k,
                            out=survived_full_buf,
                        )
                        survived_full_buf.logical_and_(row_valid_buf)
                        ordered_partition_scatter(
                            active_idx_k,
                            survived_full_buf,
                            next_active_idx_k,
                            next_count_scalar,
                        )
                        active_idx_k.copy_(next_active_idx_k)
                        active_count_dev.copy_(next_count_scalar)
                        device_step.add_(1)

                with torch.no_grad():
                    baseline_tableau_version = getattr(batched_tableau, "version", None)
                    baseline_fused_apply_call_count = self._fused_apply_call_count
                    warm = torch.cuda.Stream(device=self.device)
                    cur = torch.cuda.current_stream(self.device)
                    graph_pool = torch.cuda.graph_pool_handle()

                    # GC hygiene around capture: collect NOW, then keep the collector
                    # off until every per-K capture has ended. If a cyclic-garbage
                    # CUDAGraph is destroyed by an allocation-triggered GC pass while
                    # a capture is active, its teardown can invalidate the in-flight
                    # capture; capture_end then raises after capture_prologue already
                    # ran, leaving the default CUDA generator marked "capturing"
                    # process-wide and failing every later torch-RNG CUDA op.
                    # Capture-time only — zero per-step cost.
                    gc.collect()
                    gc_was_enabled = gc.isenabled()
                    gc.disable()
                    try:
                        for bucket_K in bucket_sizes:
                            bucket = graph_buckets[bucket_K]
                            warm.wait_stream(cur)
                            with torch.cuda.stream(warm):
                                for _ in range(3):
                                    reset_graph_state(bucket_K)
                                    compute_graph_step(bucket)
                            cur.wait_stream(warm)
                            torch.cuda.synchronize(self.device)

                            if (
                                self._fused_apply_kernel_failed
                                or not self._effective_fused_apply_kernel()
                            ):
                                raise RuntimeError(
                                    "bucketed graph capture requires the CT fused apply; "
                                    "it latched off during warmup (need the "
                                    "tableau_batch_adapter/CT backend)"
                                )

                            reset_graph_state(bucket_K)
                            warm.wait_stream(cur)
                            graph = torch.cuda.CUDAGraph()
                            with torch.cuda.graph(graph, pool=graph_pool, stream=warm):
                                compute_graph_step(bucket)
                            cur.wait_stream(warm)
                            torch.cuda.synchronize(self.device)
                            bucket["_graph"] = graph
                            _graph[bucket_K] = graph
                            self._bucketed_graph_capture_count += 1
                    finally:
                        if gc_was_enabled:
                            gc.enable()

                    reset_graph_state()
                    if baseline_tableau_version is not None:
                        batched_tableau.version = baseline_tableau_version
                    self._fused_apply_call_count = baseline_fused_apply_call_count

                return {
                    "batched_tableau": batched_tableau,
                    "trajectory_batch": trajectory_batch,
                    "active_idx": active_idx,
                    "next_active_idx": next_active_idx,
                    "active_count_dev": active_count_dev,
                    "next_count_scalar": next_count_scalar,
                    "entering_count_dev": entering_count_dev,
                    "arange_full": arange_full,
                    "device_step": device_step,
                    "ar_step_dev": device_step,
                    "length_step_dev": length_step_dev,
                    "full_indices": full_indices,
                    "graph_buckets": graph_buckets,
                    "bucket_sizes": bucket_sizes,
                    "_graph": _graph,
                    "graphs": _graph,
                    "_graph_inputs": _graph_inputs,
                    "_graph_masks": _graph_masks,
                    "w_unpacked_buf": w_unpacked_buf,
                    "reset_mask": reset_mask,
                    "reset_graph_state": reset_graph_state,
                    "reset_run_state": reset_run_state,
                    "compute_graph_step": compute_graph_step,
                    "full_bucket": full_bucket,
                    "total_rows": total_rows,
                    "max_length": max_length,
                    "max_depth": max_depth,
                    "n_measurements": n_measurements,
                    "use_fused_mask_counts": use_fused_mask_counts,
                    "policy_orig": policy_orig,
                    "baseline_tableau_version": baseline_tableau_version,
                }

            cache_enabled = bool(
                getattr(self, "_bucketed_graph_cache_enabled", False)
            )
            entry = None
            if cache_enabled:
                # A key hit IS a freshness proof: the key embeds
                # id(current_policy_orig) (and the entry holds a strong ref, so the id
                # cannot be recycled), plus max_length, batch_size and n_measurements.
                # Adaptive max_length growth therefore arrives as a cache MISS, never
                # a stale hit. Do NOT drop max_length from the key — a same-key entry
                # with a different buffer shape would replay a wrong-shaped graph.
                entry = self._bucketed_graph_cache.get(cache_key)
                if entry is not None:
                    self._bucketed_graph_cache.move_to_end(cache_key)

            if entry is None:
                entry = _bucketed_graph_entry()
                if cache_enabled:
                    self._bucketed_graph_cache[cache_key] = entry
                    while len(self._bucketed_graph_cache) > self._bucketed_graph_cache_max:
                        self._bucketed_graph_cache.popitem(last=False)

            def clone_trajectory_snapshot(src: TrajectoryBatch) -> TrajectoryBatch:
                dst = TrajectoryBatch(
                    batch_size=src.batch_size,
                    n_measurements=src.n_measurements,
                    max_length=src.max_length,
                    n_qubits=src.n_qubits,
                    device=src.device,
                )
                dst.actions_time_major.copy_(src.actions_time_major)
                dst.lengths.copy_(src.lengths)
                dst.active.copy_(src.active)
                dst.masks.copy_(src.masks)
                dst.circuit_depths.copy_(src.circuit_depths)
                dst.current_layer_qubits.copy_(src.current_layer_qubits)
                dst.qubit_last_layer.copy_(src.qubit_last_layer)
                dst.last_single_qubit_gates.copy_(src.last_single_qubit_gates)
                dst.qubit_last_use_step.copy_(src.qubit_last_use_step)
                dst.action_qubits.copy_(src.action_qubits)
                dst.cache_enabled = bool(src.cache_enabled)
                dst.cache_states_uint8 = bool(
                    getattr(src, "cache_states_uint8", False)
                )
                # OWNERSHIP TRANSFER, not clone: every tensor in these per-call lists
                # was created fresh this call by the replay loop's
                # ``.detach.clone`` copy-outs, so none is graph-bound and the next
                # replay cannot touch them. Cloning again doubled the largest
                # per-call allocation (~1 GB at 20q/M=1000) for the whole flow phase.
                # The entry's lists are reset here and re-initialized each call, so
                # the transfer leaves the cache entry consistent.
                dst.cached_states = src.cached_states
                src.cached_states = []
                dst.cached_masks = src.cached_masks
                src.cached_masks = []
                dst.cached_backward_valid_counts = src.cached_backward_valid_counts
                src.cached_backward_valid_counts = []
                dst.bucketed_active_idx_history = src.bucketed_active_idx_history
                src.bucketed_active_idx_history = []
                dst.bucketed_active_count_history = list(
                    src.bucketed_active_count_history
                )
                src.bucketed_active_count_history = []
                dst.bucketed_row_valid_history = []
                dst.bucketed_cached_flat_history = src.bucketed_cached_flat_history
                src.bucketed_cached_flat_history = []
                dst.bucketed_K = src.bucketed_K
                dst.bucketed_K_history = list(src.bucketed_K_history)
                dst.bucketed_iterations_run = src.bucketed_iterations_run
                dst.bucketed_kept_steps = src.bucketed_kept_steps

                src_tableau = src.batched_tableau
                if hasattr(src_tableau, "copy"):
                    snapshot_tableau = src_tableau.copy()
                    self._configure_sampling_tableau(snapshot_tableau)
                else:
                    snapshot_tableau = self._tableau_cls(
                        n_qubits=self.n_qubits,
                        batch_size=dst.batch_size,
                        n_measurements=dst.n_measurements,
                        device=str(self.device),
                    )
                    self._configure_sampling_tableau(snapshot_tableau)
                    with torch.no_grad():
                        for replay_step in range(dst.max_length):
                            step_active = replay_step < dst.lengths
                            if hasattr(snapshot_tableau, "active"):
                                snapshot_tableau.active.copy_(step_active)
                            snapshot_tableau.apply_actions_step(
                                dst.actions_time_major[replay_step],
                                self.action_mapping,
                                step_active,
                            )
                        if hasattr(snapshot_tableau, "active"):
                            snapshot_tableau.active.copy_(dst.active)
                        torch.cuda.current_stream(self.device).synchronize()
                dst.batched_tableau = snapshot_tableau
                return dst

            with torch.no_grad():
                entry["reset_run_state"]()
                batched_tableau = entry["batched_tableau"]
                trajectory_batch = entry["trajectory_batch"]
                baseline_tableau_version = entry.get("baseline_tableau_version")
                if baseline_tableau_version is not None:
                    batched_tableau.version = baseline_tableau_version

                trajectory_batch.cache_enabled = bool(cache_for_flows)
                trajectory_batch.cache_states_uint8 = (
                    self._effective_uint8_state_cache()
                )
                trajectory_batch.cached_states = []
                trajectory_batch.cached_masks = []
                trajectory_batch.cached_backward_valid_counts = []
                trajectory_batch.bucketed_active_idx_history = []
                trajectory_batch.bucketed_active_count_history = []
                trajectory_batch.bucketed_row_valid_history = []
                trajectory_batch.bucketed_cached_flat_history = []
                trajectory_batch.bucketed_K = entry["full_bucket"]
                trajectory_batch.bucketed_K_history = []

                active_count_dev = entry["active_count_dev"]
                entering_count_dev = entry["entering_count_dev"]
                graph_buckets = entry["graph_buckets"]
                full_bucket = entry["full_bucket"]
                max_length = entry["max_length"]

                K = full_bucket
                active_count_host = full_bucket
                window = self._bucketed_k_window
                kept = 0
                iterations_run = 0
                # (step, idx_cb, indices_cb, states_cb, masks_cb, backward, count_dev, K)
                pending = []
                # replay launches on the current stream, as do the cache copy-outs
                # and the window-boundary ``active_count_dev.item``. The CT/CuPy
                # adapter wraps its kernels in the same stream via ExternalStream, so
                # whole-device fences here would serialize every sampled layer and
                # erase the graph replay win.

                def flush_pending_cache() -> None:
                    # Mirrors the eager compacted sampler's flush: ONE batched
                    # device->host sync per window resolves the entering counts.
                    # Zero-entering no-op replays past full termination are DROPPED
                    # (trap 2); their terminal-filled action rows are zeroed
                    # post-loop (trap 3).
                    nonlocal kept, pending
                    if not pending:
                        return
                    counts_host = (
                        torch.stack([p[6] for p in pending]).cpu().tolist()
                    )
                    for (
                        p_step,
                        idx_cb,
                        indices_cb,
                        states_cb,
                        masks_cb,
                        p_backward,
                        _count_dev,
                        p_K,
                    ), count in zip(pending, counts_host):
                        count = int(count)
                        if count == 0:
                            continue
                        kept += 1
                        active_flat = idx_cb[:count].clone()
                        trajectory_batch.bucketed_active_idx_history.append(
                            active_flat
                        )
                        trajectory_batch.bucketed_active_count_history.append(
                            count
                        )
                        trajectory_batch.bucketed_K_history.append(p_K)
                        if cache_for_flows:
                            trajectory_batch.cache_step_data(
                                p_step,
                                states_cb[:count].clone(),
                                indices_cb[:count].clone(),
                                masks_cb[:count].clone(),
                                p_backward,
                            )
                            trajectory_batch.bucketed_cached_flat_history.append(
                                active_flat.clone()
                            )
                    pending = []

                for step in range(max_length):
                    if step == 0 or (step % window == 0):
                        flush_pending_cache()
                        active_count_host = int(active_count_dev.item())
                        if active_count_host == 0:
                            break
                        if step > 0:
                            K = min(
                                K,
                                self._smallest_pow2_bucket(
                                    active_count_host, full_bucket
                                ),
                            )
                    if active_count_host > K:
                        raise RuntimeError(
                            f"bucketed graph K overflow: active_count_host="
                            f"{active_count_host} > K={K} at step={step}; "
                            f"full_bucket={full_bucket}"
                        )

                    bucket = graph_buckets[K]
                    # Features, masks, backward counts, RNG, policy, sample,
                    # apply, and the queue partition are ALL inside the
                    # captured graph now — one replay launch per sampled
                    # layer; only the cache copy-outs below stay eager.
                    need_counts = bool(cache_for_flows and step < max_length - 1)
                    bucket["_graph"].replay()
                    batched_tableau.version += 1
                    self._fused_apply_call_count += 1
                    trajectory_batch.actions_time_major[step].copy_(bucket["actions"])
                    iterations_run += 1

                    # Copy-outs must run before the NEXT replay overwrites the
                    # graph-owned buffers, but need no host sync: the boundary count
                    # bounds every entering count in this window (counts only
                    # decrease), so ``[:active_count_host]`` prefixes are host-sized.
                    # The exact entering count stays on DEVICE and is resolved at the
                    # flush, letting the host enqueue a whole window ahead.
                    idx_cb = bucket["idx"][:active_count_host].detach().clone()
                    if cache_for_flows:
                        states_cb = (
                            bucket["_graph_inputs"][:active_count_host]
                            .detach()
                            .clone()
                        )
                        indices_cb = (
                            bucket["indices"][:active_count_host].detach().clone()
                        )
                        masks_cb = (
                            bucket["_graph_masks"][:active_count_host]
                            .detach()
                            .clone()
                        )
                        backward_cb = (
                            bucket["backward"].detach().clone()
                            if need_counts
                            else None
                        )
                    else:
                        states_cb = None
                        indices_cb = None
                        masks_cb = None
                        backward_cb = None
                    pending.append((
                        step,
                        idx_cb,
                        indices_cb,
                        states_cb,
                        masks_cb,
                        backward_cb,
                        entering_count_dev.clone(),
                        K,
                    ))

                    if step == max_length - 1:
                        still_active = (
                            trajectory_batch.active
                            & (trajectory_batch.lengths == 0)
                        )
                        max_lengths = torch.full_like(
                            trajectory_batch.lengths, max_length
                        )
                        trajectory_batch.lengths.copy_(
                            torch.where(
                                still_active,
                                max_lengths,
                                trajectory_batch.lengths,
                            )
                        )
                        trajectory_batch.active.copy_(
                            torch.where(
                                still_active,
                                torch.zeros_like(trajectory_batch.active),
                                trajectory_batch.active,
                            )
                        )
                        if hasattr(batched_tableau, "active"):
                            batched_tableau.active.copy_(trajectory_batch.active)

                # Flush the final window. The per-step early-exit.item is
                # gone: full termination is detected at the next window
                # boundary instead, costing at most window-1 no-op replays
                # per sampling call (the same trade the eager compacted
                # sampler documents) in exchange for zero per-step syncs.
                flush_pending_cache()

            if kept < max_length:
                trajectory_batch.actions_time_major[kept:].zero_()
            trajectory_batch.bucketed_iterations_run = iterations_run
            trajectory_batch.bucketed_kept_steps = kept

            if self.adaptive_tracker is not None:
                self.adaptive_tracker.update_statistics(trajectory_batch)

            with torch.no_grad():
                self._bucketed_sample_invocation_buf.add_(1)

            return clone_trajectory_snapshot(trajectory_batch)

    def sample_trajectories(self,
                          batch_size: int,
                          n_measurements: int,
                          max_depth: int,
                          mode: SamplingMode = SamplingMode.ON_POLICY,
                          batch_data_list: Optional[List[Dict]] = None,
                          cache_for_flows: bool = True) -> TrajectoryBatch:
        """Sample trajectories with depth limit and adaptive buffer sizing."""
        if self._effective_sampling_mode == _SAMPLING_MODE_BUCKETED:
            return self._sample_trajectories_bucketed(
                batch_size=batch_size,
                n_measurements=n_measurements,
                max_depth=max_depth,
                mode=mode,
                batch_data_list=batch_data_list,
                cache_for_flows=cache_for_flows,
            )

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
            return self._replay_trajectories(batch_data_list, max_length)

        if (
            self.static_shape_sampling
            and self.device.type == 'cuda'
            and mode in (SamplingMode.ON_POLICY, SamplingMode.OFF_POLICY)
        ):
            return self._sample_trajectories_static_shape(
                batch_size=batch_size,
                n_measurements=n_measurements,
                max_depth=max_depth,
                mode=mode,
                max_length=max_length,
                cache_for_flows=cache_for_flows,
            )
        
        # Create tableau on the same device as the model
        batched_tableau = self._tableau_cls(
            n_qubits=self.n_qubits,
            batch_size=batch_size,
            n_measurements=n_measurements,
            device=str(self.device)
        )
        self._configure_sampling_tableau(batched_tableau)
        
        trajectory_batch = TrajectoryBatch(
            batch_size=batch_size,
            n_measurements=n_measurements,
            max_length=max_length,
            n_qubits=self.n_qubits,
            device=self.device
        )
        trajectory_batch.batched_tableau = batched_tableau
        
        # Enable caching if requested
        if cache_for_flows:
            trajectory_batch.enable_caching(
                states_uint8=self._effective_uint8_state_cache()
            )
        
        with torch.no_grad():
            for step in range(max_length):
                # NOTE: the previous ``if not trajectory_batch.active.any: break``
                # forced a host sync every step. The states-empty break below
                # already terminates correctly — ``to_flat_tensors_active_only``
                # internally calls ``nonzero`` whose output-shape resolution is
                # the same sync, so the extra ``.any`` was strictly redundant.
                states_tensor, indices = self._policy_features_active(batched_tableau)
                if states_tensor.shape[0] == 0:
                    break

                # Ensure contiguous memory layout for GPU performance
                if self.device.type in ['cuda', 'mps']:
                    states_tensor = states_tensor.contiguous()

                # Convert indices once; downstream active-only mask/cache paths
                # are positionally aligned with these rows.
                if isinstance(indices, torch.Tensor):
                    indices_tensor = indices.to(self.device)
                else:
                    indices_tensor = torch.as_tensor(indices, dtype=torch.long, device=self.device)

                # Fused active-mask + forward/backward valid-count kernel. Replaces
                # the three-call legacy path with one CuPy launch on CUDA, falling
                # back to the original chain on CPU / missing CuPy. The fused method
                # applies ``max_depth`` to the mask only — counts intentionally ignore
                # the depth cap to preserve the legacy semantic.
                need_counts = bool(cache_for_flows and step < max_length - 1)
                active_masks, _fwd_counts, backward_valid_counts = (
                    self.masking_engine.compute_masks_and_counts_fused(
                        trajectory_batch,
                        indices_tensor,
                        current_step=step + 1,
                        max_depth=max_depth,
                        compute_backward=need_counts,
                        use_fused_kernel=self._effective_fused_mask_counts_kernel(),
                    )
                )
                
                # Cache states if enabled
                if cache_for_flows:
                    trajectory_batch.cache_step_data(
                        step, states_tensor, indices_tensor, active_masks, backward_valid_counts
                    )

                # Pre-fill the per-step slice of ``actions_time_major`` (the
                # canonical action-history buffer, allocated once in
                # ``TrajectoryBatch.__init__``) with the terminal sentinel and scatter
                # only into that slice, instead of allocating a standalone
                # ``actions`` tensor and scattering twice. Downstream only reads
                # ``actions.shape`` / ``.view(-1)``, so a contiguous ``(B, M)`` view
                # is a drop-in.
                actions = trajectory_batch.actions_time_major[step]
                actions.fill_(self.terminal_index)

                if mode == SamplingMode.ON_POLICY:
                    # No bucket padding on the dynamic-active path: padding existed
                    # only to give CUDA Graph capture a fixed shape, which applies
                    # under ``static_shape_sampling`` — routed to
                    # ``_sample_trajectories_static_shape`` by the early-return guard
                    # above. Here the bucket math, zeros allocation, cat launch and
                    # indexed slice were all dead work paid every step.
                    # bf16 autocast (when enabled) on the no-grad sampling forward,
                    # optionally via bucketed CUDA-graph replay. The helper returns
                    # fp32 logits for the fused gumbel kernel; bf16 affects only the
                    # sampled-action distribution, not gradients. With
                    # use_cuda_graph_policy=False this is bit-identical.
                    logits = self._policy_forward_dynamic(states_tensor)

                    # Fused masked Gumbel-max sampling. The fused kernel reads
                    # the mask directly and never picks invalid slots, so we
                    # skip the explicit ``masked_fill`` clone the legacy path
                    # used. Falls back to the PyTorch chain on CPU or when
                    # CuPy is unavailable; both produce the same distribution.
                    sampled_actions = masked_gumbel_argmax(
                        logits,
                        active_masks,
                        terminal_index=self.terminal_index,
                        use_fused_kernel=self._effective_fused_sampling_kernel(),
                    )

                    # Single scatter into ``actions_time_major[step]``
                    # via the ``actions`` view; this also populates the
                    # per-step ``(B, M)`` tensor consumed by
                    # ``apply_actions_to_batch`` below (item 3).
                    actions[indices_tensor[:, 0], indices_tensor[:, 1]] = sampled_actions

                elif mode == SamplingMode.OFF_POLICY:
                    # Uniform-over-valid-actions specialization. ``off_logits`` is
                    # still allocated to give the fused kernel a zero-logit input, but
                    # the legacy ``off_logits[~active_masks] = -inf`` masked-fill is
                    # gone: the fused kernel reads ``active_masks`` directly and never
                    # picks an invalid slot. Reuses a cached zero buffer sliced to the
                    # active rows rather than a fresh zeros_like per step.
                    off_logits = self._offpolicy_zero_logits(
                        active_masks.shape[0], active_masks.shape[1], active_masks.device
                    )
                    sampled_actions = masked_gumbel_argmax(
                        off_logits,
                        active_masks,
                        terminal_index=self.terminal_index,
                        use_fused_kernel=self._effective_fused_sampling_kernel(),
                    )

                    # Single scatter into ``actions_time_major[step]``
                    # via the ``actions`` view (item 3).
                    actions[indices_tensor[:, 0], indices_tensor[:, 1]] = sampled_actions

                # Apply actions using depth-aware function with step tracking
                terminated = self.apply_actions_to_batch(
                    batched_tableau, actions, trajectory_batch, step=step
                )
                
                # Update lengths for terminated trajectories. ``step + 1`` is passed
                # as a scalar to ``torch.where`` rather than materializing a ``(B, M)``
                # int64 tensor per layer — PyTorch broadcasts the Python int inside
                # the kernel, saving an allocation and a fill per sampled step. The
                # ``max_lengths`` pattern at the bottom of the loop runs once and is
                # left as-is.
                newly_terminated = terminated & (trajectory_batch.lengths == 0)
                trajectory_batch.lengths = torch.where(
                    newly_terminated,
                    step + 1,
                    trajectory_batch.lengths,
                )
                
                # Handle max length reached
                if step == max_length - 1:
                    still_active = trajectory_batch.active & (trajectory_batch.lengths == 0)
                    max_lengths = torch.full_like(trajectory_batch.lengths, max_length)
                    trajectory_batch.lengths = torch.where(
                        still_active,
                        max_lengths,
                        trajectory_batch.lengths,
                    )
                    trajectory_batch.active = torch.where(
                        still_active,
                        torch.zeros_like(trajectory_batch.active),
                        trajectory_batch.active,
                    )
        
        # Update adaptive statistics if using adaptive strategy
        if self.adaptive_tracker is not None:
            self.adaptive_tracker.update_statistics(trajectory_batch)
        
        return trajectory_batch
