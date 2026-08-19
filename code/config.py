# -*- coding: utf-8 -*-
"""Experiment configuration for FlowMeas.

Extracted from main.py (which keeps a backward-compatible facade re-export).
This is a LIGHTWEIGHT LEAF module: it imports only stdlib + numpy and NO sibling
code/ module and NO heavy dependency (torch / GFNs / models / matplotlib), so
``import config`` (or ``from.config import ExperimentConfig``) is cheap.

The sampling-mode resolver accepts a lightweight structural device protocol,
so runtime type introspection remains resolvable without importing torch.
"""

import logging
import numpy as np
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Protocol


class _DeviceLike(Protocol):
    type: str


EVALUATOR_MODE_EXACT_SMALL = "exact_small"
EVALUATOR_MODE_SCALABLE_LARGE = "scalable_large"

def _coerce_bool_config(value: Any, field_name: str) -> bool:
    """Coerce JSON/CLI-friendly boolean config values."""
    if isinstance(value, (bool, np.bool_)):
        return bool(value)

    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "t", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "f", "no", "n", "off"}:
            return False

    if isinstance(value, int) and value in {0, 1}:
        return bool(value)

    raise ValueError(
        f"{field_name} must be a boolean value; got {value!r}"
    )


def _coerce_optional_bool_config(value: Any, field_name: str) -> Optional[bool]:
    """Coerce an optional JSON/CLI-friendly boolean config value."""
    if value is None:
        return None
    return _coerce_bool_config(value, field_name)

# NOTE: these sampling-mode constants + the coerce helper intentionally MIRROR code/GFNs.py (one
# logical contract; duplicated because GFNs cannot import main without a cycle). If you add/rename a
# mode or alias, update BOTH. No double alias-warning occurs in the normal flow: __post_init__
# canonicalizes ExperimentConfig.sampling_mode before it reaches the GFlowNet, so GFNs' coerce never
# re-sees a raw alias.
_SAMPLING_MODE_DYNAMIC_ACTIVE = "dynamic_active"
_SAMPLING_MODE_STATIC_FULL = "static_full"
_SAMPLING_MODE_BUCKETED = "bucketed"
_SAMPLING_MODE_VALUES = frozenset({
    _SAMPLING_MODE_DYNAMIC_ACTIVE,
    _SAMPLING_MODE_STATIC_FULL,
    _SAMPLING_MODE_BUCKETED,
})
_SAMPLING_MODE_ALIASES = {
    "dynamic": _SAMPLING_MODE_DYNAMIC_ACTIVE,
    "static_shape": _SAMPLING_MODE_STATIC_FULL,
    "static": _SAMPLING_MODE_STATIC_FULL,
}
_SAMPLING_MODE_ALIAS_WARNED = set()


def _coerce_sampling_mode_config(
    value: Any,
    field_name: str = "sampling_mode",
    *,
    warn_alias: bool = True,
) -> Optional[str]:
    """Coerce the optional sampler selector to the canonical vocabulary."""
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(
            f"{field_name} must be one of "
            f"{sorted(_SAMPLING_MODE_VALUES)}; got {value!r}"
        )
    normalized = value.strip().lower()
    if normalized in _SAMPLING_MODE_ALIASES:
        canonical = _SAMPLING_MODE_ALIASES[normalized]
        if warn_alias and normalized not in _SAMPLING_MODE_ALIAS_WARNED:
            logging.warning(
                "%s=%r is deprecated; use %r",
                field_name,
                value,
                canonical,
            )
            _SAMPLING_MODE_ALIAS_WARNED.add(normalized)
        return canonical
    if normalized not in _SAMPLING_MODE_VALUES:
        raise ValueError(
            f"{field_name} must be one of "
            f"{sorted(_SAMPLING_MODE_VALUES)}; got {value!r}"
        )
    return normalized


def _legacy_sampler_alias(effective_sampling_mode: str) -> str:
    if effective_sampling_mode == _SAMPLING_MODE_STATIC_FULL:
        return "static_shape"
    return effective_sampling_mode


def _resolve_sampling_mode_controls(
    static_shape_sampling: Any,
    cuda_graph_sampling: Any,
    sampling_mode: Any,
    device: _DeviceLike,
    *,
    warn_inconsistent: bool,
) -> Dict[str, Any]:
    requested_static = _coerce_optional_bool_config(
        static_shape_sampling,
        "static_shape_sampling",
    )
    static_requested = False if requested_static is None else bool(requested_static)

    canonical_requested_mode = _coerce_sampling_mode_config(
        sampling_mode,
        warn_alias=False,
    )
    if canonical_requested_mode is None:
        selected_mode = (
            _SAMPLING_MODE_STATIC_FULL
            if static_requested
            else _SAMPLING_MODE_DYNAMIC_ACTIVE
        )
        requested_sampling_mode = selected_mode
    else:
        selected_mode = canonical_requested_mode
        requested_sampling_mode = sampling_mode
        if requested_static is not None:
            legacy_mode = (
                _SAMPLING_MODE_STATIC_FULL
                if static_requested
                else _SAMPLING_MODE_DYNAMIC_ACTIVE
            )
            if legacy_mode != selected_mode and warn_inconsistent:
                logging.warning(
                    "sampling_mode=%r conflicts with "
                    "static_shape_sampling=%r (legacy mode %r); using "
                    "sampling_mode=%r",
                    sampling_mode,
                    requested_static,
                    legacy_mode,
                    selected_mode,
                )

    static_effective = (
        selected_mode == _SAMPLING_MODE_STATIC_FULL
        and device.type == "cuda"
    )
    if selected_mode == _SAMPLING_MODE_BUCKETED:
        effective_sampling_mode = _SAMPLING_MODE_BUCKETED
    elif static_effective:
        effective_sampling_mode = _SAMPLING_MODE_STATIC_FULL
    else:
        effective_sampling_mode = _SAMPLING_MODE_DYNAMIC_ACTIVE

    requested_graph = _coerce_optional_bool_config(
        cuda_graph_sampling,
        "cuda_graph_sampling",
    )
    graph_default = (
        effective_sampling_mode == _SAMPLING_MODE_STATIC_FULL
        and device.type == "cuda"
    )
    graph_requested = (
        graph_default
        if requested_graph is None
        else bool(requested_graph)
    )
    # Request + mode/device eligibility only; GFNs.py enforces the
    # CT-backend/default-feature preconditions for bucketed graph capture.
    graph_eligible = (
        effective_sampling_mode in (
            _SAMPLING_MODE_STATIC_FULL,
            _SAMPLING_MODE_BUCKETED,
        )
        and device.type == "cuda"
    )
    graph_effective = graph_requested and graph_eligible

    return {
        "requested_static_shape_sampling": requested_static,
        "effective_static_shape_sampling": static_effective,
        "requested_cuda_graph_sampling": requested_graph,
        "cuda_graph_sampling": graph_effective,
        "requested_sampling_mode": requested_sampling_mode,
        "effective_sampling_mode": effective_sampling_mode,
        "effective_sampler": _legacy_sampler_alias(effective_sampling_mode),
    }


def resolve_evaluator_mode(large_hubbard_mode: bool) -> str:
    """Return the configured evaluator mode name."""
    return (
        EVALUATOR_MODE_SCALABLE_LARGE
        if _coerce_bool_config(large_hubbard_mode, "large_hubbard_mode")
        else EVALUATOR_MODE_EXACT_SMALL
    )

@dataclass
class ExperimentConfig:
    """Configuration for experiments"""
    hamiltonian_path: str
    eval_every: int = 1000
    n_updates: int = 10000
    n_measurements: int = 1000  # Circuits per batch element
    update_freq: int = 5  # Number of batch elements per update
    max_depth: int = 8
    beta: float = 1e3
    hidden_dim: int = 1024
    num_hidden_layers: int = 3
    lr: float = 1e-3
    weight_decay: float = 1e-5
    device_preference: str = "auto"
    results_dir: str = "./experiment_results"
    n_eval_top_k_batch_elements: int = 5  # Number of top batch elements to evaluate
    replay_every: int = 25
    offpolicy_every: int = 20
    checkpoint_every: int = 100
    reward_type: str = "log"  # Options: "default", "exponential", "threshold"
    reward_kwargs: Dict = field(default_factory=lambda: {"alpha": 1.0})
    cost_type: str = "exponential"
    cost_kwargs: Dict = field(default_factory=dict)  # Arguments for cost computation
    # Opt-in: zero the compact-encoding stabilizer PENALTY terms' weights in the cost
    # BEFORE normalization (sum-normalization then redistributes over the physical terms
    # only). Stabilizer terms have Gamma == 0 on the code space (zero measurement
    # information) yet carry |c| = lambda/2, wasting circuit budget.
    # TRAINING-AFFECTING (changes the reward landscape) -> lives in all_params.
    # Requires a metadata.json 'stabilizer_penalty' block next to the Hamiltonian
    # (detect_stabilizer_terms fails fast otherwise).
    zero_stabilizer_cost_weights: bool = False
    objective_type: str = "tb"
    objective_kwargs: Dict = field(default_factory=lambda: {"loss_type": "squared"})
    n_simulations: int = 10  # Number of simulation runs for error estimation
    resume: bool = True  # Whether to resume from existing experiment
    experiment_dir: Optional[str] = None  # Specific experiment directory to resume from
    warm_start_from: Optional[str] = None  # Load NN weights and logZ from this experiment (no hyperparameter check)
    model_type: str = "clifford_mlp"
    # Extra model constructor kwargs forwarded to ``create_clifford_model`` (e.g.
    # ``row_embed_dim`` / ``pool`` for ``packed_w_rowtoken``; ``covariant_shaping``
    # for the GIPTE hit models). Architecture-affecting -> part of the resume
    # compatibility check.
    model_kwargs: Dict = field(default_factory=dict)
    async_eval: bool = False  # Enable asynchronous evaluation
    eval_poll_interval: int = 30  # Seconds between checkpoint polls
    eval_process_timeout: int = 300  # Timeout for evaluator process shutdown
    large_hubbard_mode: bool = False  # Select scalable-large Hubbard execution mode
    measurement_backend: Optional[str] = None  # None/auto => CT on CUDA, legacy on CPU
    # The sampler default stays on the dynamic active-row path. The static-shape +
    # CUDA-graph sampler runs the full B*M policy forward every step, which is slower
    # for clifford_mlp on high-termination envelopes, and its manual policy-graph
    # capture nests badly with reduce-overhead (latching ``_policy_graph_failed``).
    # Static-shape remains an explicit opt-in.
    static_shape_sampling: Optional[bool] = None  # None => dynamic-active CUDA default (measured best)
    cuda_graph_sampling: Optional[bool] = None  # None => enabled only when static_shape_sampling is true
    sampling_mode: Optional[str] = None  # canonical: dynamic_active, static_full, bucketed
    # Opt-in bucketed CUDA-graph replay for the dynamic-active policy
    # forward. Unlike cuda_graph_sampling, this does not force full-row static
    # sampling; it graph-replays only small active-row buckets and falls back to
    # eager above cuda_graph_policy_max_rows.
    use_cuda_graph_policy: bool = False
    cuda_graph_policy_max_rows: int = 2048
    use_fused_metadata_kernel: bool = True
    use_fused_sampling_kernel: bool = True  # Fused masked Gumbel-max CuPy kernel
    use_fused_mask_counts_kernel: bool = True  # Fused active-mask + fwd/bwd valid-counts kernel
    use_fused_counter_rng_kernel: bool = True  # bucketed counter-RNG: 1-launch CuPy SplitMix64 vs ~160-launch torch chain
    use_fused_partition_update_kernel: bool = True  # Phase-4 CT fused bucketed queue partition + lengths update (1 launch vs ~14)
    use_fused_apply_kernel: bool = True  # CT-side fused action lowering + apply + metadata
    # bf16 autocast on the dynamic sampling policy forward only (~2x the
    # GEMM-bound, GPU-saturated sample phase on tensor cores). Sampling is
    # no_grad and only feeds the gumbel-argmax action choice, so this changes
    # only the stochastic exploration distribution, NOT gradients (the loss
    # path stays fp32/TF32). Unlike the bit-identical fused-kernel knobs it is
    # NOT parity-identical, but its effect is within run-to-run exploration
    # variance, so it is treated as a non-forking sampler perf knob for resume
    # (logged in the resume diff, omitted from ``all_params``).
    use_bf16_sampling: bool = True
    # EXPERIMENTAL (default OFF): bf16 autocast on the gradient-path policy
    # forward -> bf16 backward GEMMs (~71% of the step). Unlike use_bf16_sampling
    # this is on the GRADIENT path, so it must clear a convergence + gradient-parity
    # study before any default flip. Opt-in via config until then.
    use_bf16_backward: bool = False
    # Cached-flow backward checkpointing (re-runs the policy forward in backward
    # to cap memory — required at n=52/hd=2048).: the default (``None`` =
    # auto) now checkpoints **only** under large-system memory pressure
    # (``large_hubbard_mode``); small/medium systems that fit skip the per-step
    # recompute automatically. Set ``True``/``False`` to force it either way
    # (e.g. ``False`` for a memory-light ``packed_w_rowtoken`` run that would
    # otherwise auto-checkpoint). Numerically identical; a bit-identical perf
    # knob (omitted from the resume compatibility check).
    use_activation_checkpointing: Optional[bool] = None
    # uint8 cached-state compression: store the cached flow states (flattened-W
    # 0/1 features) as uint8 instead of fp32 — 4x smaller cached flows, the
    # per-batch memory term that scales with trajectory row-steps (the 20q
    # whole-step OOM driver). Bit-identical (0/1 <-> fp32 round-trip is exact);
    # auto-disabled for GIPTE / packed-W feature modes inside GFNs.
    use_uint8_state_cache: bool = True
    transfer_weights_on_depth_change: bool = False  # If False, don't transfer NN weights when max_depth changes
    # DMRG-backed reference for scalable-large runs. Two-tier resolution:
    #   - ``dmrg_reference_energy`` (Optional[float]) is an explicit scalar
    #     override. Wins if it normalises to a finite float.
    #   - ``dmrg_reference_path`` (Optional[str]) points at a sidecar JSON
    #     written by ``python -m code.dmrg_reference compute --hamiltonian...``.
    #     Hash-verified against ``hamiltonian_path`` at resolution time.
    # When neither is set the run stays on the ``structural`` validation tier.
    # See ``code/dmrg_reference.py`` for the producer and sidecar layout.
    dmrg_reference_energy: Optional[float] = None
    dmrg_reference_path: Optional[str] = None

    def __post_init__(self) -> None:
        self.large_hubbard_mode = _coerce_bool_config(
            self.large_hubbard_mode,
            "large_hubbard_mode",
        )
        self.use_fused_metadata_kernel = _coerce_bool_config(
            self.use_fused_metadata_kernel,
            "use_fused_metadata_kernel",
        )
        self.use_fused_mask_counts_kernel = _coerce_bool_config(
            self.use_fused_mask_counts_kernel,
            "use_fused_mask_counts_kernel",
        )
        self.use_fused_counter_rng_kernel = _coerce_bool_config(
            self.use_fused_counter_rng_kernel,
            "use_fused_counter_rng_kernel",
        )
        self.use_fused_partition_update_kernel = _coerce_bool_config(
            self.use_fused_partition_update_kernel,
            "use_fused_partition_update_kernel",
        )
        self.use_fused_apply_kernel = _coerce_bool_config(
            self.use_fused_apply_kernel,
            "use_fused_apply_kernel",
        )
        self.use_fused_sampling_kernel = _coerce_bool_config(
            self.use_fused_sampling_kernel,
            "use_fused_sampling_kernel",
        )
        self.use_bf16_sampling = _coerce_bool_config(
            self.use_bf16_sampling,
            "use_bf16_sampling",
        )
        self.use_bf16_backward = _coerce_bool_config(
            self.use_bf16_backward,
            "use_bf16_backward",
        )
        self.zero_stabilizer_cost_weights = _coerce_bool_config(
            self.zero_stabilizer_cost_weights,
            "zero_stabilizer_cost_weights",
        )
        # ``None`` (auto) ⇒ checkpoint only under large-system memory
        # pressure (large_hubbard_mode, coerced just above); explicit True/False
        # is honored. Resolving here (where both knobs are visible) keeps the
        # downstream model_config value a concrete bool.
        _ckpt = _coerce_optional_bool_config(
            self.use_activation_checkpointing,
            "use_activation_checkpointing",
        )
        self.use_activation_checkpointing = (
            bool(self.large_hubbard_mode) if _ckpt is None else _ckpt
        )
        self.use_uint8_state_cache = _coerce_bool_config(
            self.use_uint8_state_cache,
            "use_uint8_state_cache",
        )
        self.static_shape_sampling = _coerce_optional_bool_config(
            self.static_shape_sampling,
            "static_shape_sampling",
        )
        self.cuda_graph_sampling = _coerce_optional_bool_config(
            self.cuda_graph_sampling,
            "cuda_graph_sampling",
        )
        self._requested_sampling_mode_raw = self.sampling_mode
        self.sampling_mode = _coerce_sampling_mode_config(
            self.sampling_mode,
            "sampling_mode",
        )
        self.use_cuda_graph_policy = _coerce_bool_config(
            self.use_cuda_graph_policy,
            "use_cuda_graph_policy",
        )
        try:
            self.cuda_graph_policy_max_rows = int(self.cuda_graph_policy_max_rows)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "cuda_graph_policy_max_rows must be a positive integer; "
                f"got {self.cuda_graph_policy_max_rows!r}"
            ) from exc
        if self.cuda_graph_policy_max_rows < 1:
            raise ValueError(
                "cuda_graph_policy_max_rows must be a positive integer; "
                f"got {self.cuda_graph_policy_max_rows!r}"
            )

    @property
    def evaluator_mode(self) -> str:
        return resolve_evaluator_mode(self.large_hubbard_mode)
