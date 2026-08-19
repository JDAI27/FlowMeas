# -*- coding: utf-8 -*-
"""Shared runtime infrastructure for the GFlowNet training stack.

Split out of ``GFNs.py`` (which remains the backward-compatible facade).
Owns: optional-dependency imports (models, objectives, masking engine,
measurement-adapter kernels, bucketed sampler primitives), sampling-mode
constants and coercion, device helpers, reward functions, the
``FlowMeasTableau`` protocol, and profiler record-function shims.

Hot-path note: downstream modules import these names into their own
namespace (plain module-global lookups), so this split adds no per-step
indirection and changes no kernel launch counts.
"""

import torch
import logging

# Enable TensorFloat32 for better performance on Ampere and newer GPUs
if torch.cuda.is_available():
    torch.set_float32_matmul_precision('high')
from enum import Enum
from typing import List, Tuple, Dict, Optional, Any, Protocol
from contextlib import contextmanager, nullcontext

try:
    from torch.profiler import record_function as torch_record_function
except ImportError:  # pragma: no cover - torch profiler optional
    def torch_record_function(_name):
        @contextmanager
        def _noop():
            yield
        return _noop()

record_function = torch_record_function


_SAMPLING_MODE_DYNAMIC_ACTIVE = "dynamic_active"
_SAMPLING_MODE_STATIC_FULL = "static_full"
_SAMPLING_MODE_BUCKETED = "bucketed"
_SAMPLING_MODE_VALUES = frozenset({
    _SAMPLING_MODE_DYNAMIC_ACTIVE,
    _SAMPLING_MODE_STATIC_FULL,
    _SAMPLING_MODE_BUCKETED,
})
class BucketedGraphPreflightError(RuntimeError):
    """Pre-capture capability failure inside the bucketed CUDA-graph path.

    Raised ONLY at probe sites that run before any shared GFlowNet/tableau
    state is mutated (the graph entry's private tableau/buffers are still
    discardable), so the public bucketed dispatch can catch it and degrade to
    the eager bucketed sampler within the SAME call. First-use runtime
    failures (CuPy import, NVRTC compile, launch skew) also latch the
    corresponding fused-kernel gate off, so subsequent calls are refused by
    ``_effective_bucketed_graph`` without entering the graph path at all.
    Failures after warmup/capture has begun stay hard RuntimeErrors — by then
    the entry is mid-construction and silently degrading could hide a real
    capture bug.
    """


_SAMPLING_MODE_ALIASES = {
    "dynamic": _SAMPLING_MODE_DYNAMIC_ACTIVE,
    "static_shape": _SAMPLING_MODE_STATIC_FULL,
    "static": _SAMPLING_MODE_STATIC_FULL,
}
_SAMPLING_MODE_ALIAS_WARNED = set()


def _coerce_sampling_mode(
    value: Any,
    field_name: str = "sampling_mode",
    *,
    warn_alias: bool = True,
) -> Optional[str]:
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


# Outside an active ``torch.profiler``
# capture, every ``record_function`` enter/exit is a host-side range push/pop
# that nobody reads — pure overhead between the kernel-issuing phases of an
# issue-bound step. ``EfficientGFNTrainer.train`` binds the real
# ``record_function`` by default (``profile=True``) and swaps in this no-op only
# when a caller explicitly passes ``profile=False``. The production runner
# (``main.run_experiment``'s inline loop) uses no ``record_function`` at all, so
# this gating helps only ``train(profile=False)`` callers; production benefits
# instead from the metrics-trim slack below. ``nullcontext`` is stateless and
# reentrant, so one shared instance is safe across every ``with`` site.
_NULL_RECORD_CONTEXT = nullcontext()


def _null_record_function(_name):
    return _NULL_RECORD_CONTEXT


def _resolve_cuda_graph_policy_enabled(
    requested: bool,
    device_type: str,
    static_shape_sampling: bool,
    sampling_mode: Optional[str] = None,
) -> bool:
    """Whether the dynamic-active CUDA graph policy can run."""
    dynamic_active = (
        not bool(static_shape_sampling)
        if sampling_mode is None
        else sampling_mode == _SAMPLING_MODE_DYNAMIC_ACTIVE
    )
    return bool(requested) and device_type == 'cuda' and dynamic_active


# How far ``metrics_history`` / ``timing_history`` may overflow ``metrics_window``
# before the O(window) list trim runs. Trimming every update slices the whole
# list each step; amortizing over this slack makes it ~1/slack as frequent while
# always retaining at least ``metrics_window`` most-recent entries.
_METRICS_TRIM_SLACK = 64


class FlowMeasTableau(Protocol):
    """Tableau surface used by FlowMeas training/cost paths."""

    batch_size: int
    n_measurements: int
    device: torch.device
    active: torch.Tensor

    def apply_actions_step(
        self,
        actions: torch.Tensor,
        action_map: Dict[int, Tuple],
        mask: Optional[torch.Tensor] = None,
    ) -> None:
        ...

    def to_flat_tensors_active_only(self) -> Tuple[torch.Tensor, torch.Tensor]:
        ...

    def to_flat_tensors(self) -> Tuple[torch.Tensor, torch.Tensor]:
        ...

    def _pauli_string_to_symplectic(self, pauli_strings: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        ...

    def transform_paulis(self, pauli_vecs: torch.Tensor) -> torch.Tensor:
        ...

    def prob_P_multi(self, p_strs: List[str]) -> torch.Tensor:
        ...

# Configure logging to show debug messages when debug mode is enabled
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

try:
    from .models import DiscreteUniform, CliffordMLP, create_clifford_model
    from .gfn_objectives import GFlowNetObjective, create_gfn_objective
    from .cost_computer import CostComputer, CostFunction, ThresholdCost
    from .quantum_action_mapping import build_action_mapping
    from .masking_engine import MaskingEngine
    from .measurement_adapter import resolve_tableau_backend
    try:
        from .measurement_adapter.metadata_kernel import (
            apply_metadata_kernel,
            fused_kernel_persistently_unavailable as _fused_metadata_persistently_unavailable,
        )
    except ImportError:  # pragma: no cover - optional CUDA/CuPy fast path
        # Symmetric with the absolute-import branch below: guard to None so the
        # two execution modes (python -m code.X vs python code/X.py) behave
        # identically when the optional fused metadata kernel is unavailable.
        # The latch probe reports unavailable so the sibling
        # ``_effective_fused_metadata_kernel`` gate cannot claim that a missing
        # callable is eligible.
        apply_metadata_kernel = None

        def _fused_metadata_persistently_unavailable() -> bool:
            return True
    from .measurement_adapter.sampling_kernel import (
        masked_gumbel_argmax,
        fused_kernel_persistently_unavailable as _fused_sampling_persistently_unavailable,
    )
    from .measurement_adapter.mask_counts_kernel import (
        compute_mask_counts_fused as _compute_mask_counts_fused,
        fused_kernel_persistently_unavailable as _fused_mask_counts_persistently_unavailable,
    )
    from .measurement_adapter.counter_rng_kernel import (
        fused_kernel_persistently_unavailable as _fused_counter_rng_persistently_unavailable,
    )
    from .measurement_adapter.partition_update_adapter import (
        partition_update_bucketed_torch as _partition_update_bucketed_torch,
        fused_kernel_persistently_unavailable as _fused_partition_update_persistently_unavailable,
    )
    from .measurement_adapter import fused_apply_adapter as _fused_apply_adapter
    from .bucketed_sampler import (
        ActiveQueue,
        build_fixed_k_features,
        counter_uniforms,
        flat_to_bm,
        ordered_compact,
        ordered_partition_scatter,
    )
except ImportError:
    # Fallback to absolute imports for direct execution
    from models import DiscreteUniform, CliffordMLP, create_clifford_model
    from gfn_objectives import GFlowNetObjective, create_gfn_objective
    from cost_computer import CostComputer, CostFunction, ThresholdCost
    from quantum_action_mapping import build_action_mapping
    from masking_engine import MaskingEngine
    from measurement_adapter import resolve_tableau_backend
    try:
        from measurement_adapter.metadata_kernel import (
            apply_metadata_kernel,
            fused_kernel_persistently_unavailable as _fused_metadata_persistently_unavailable,
        )
    except ImportError:  # pragma: no cover - optional CUDA/CuPy fast path
        apply_metadata_kernel = None

        def _fused_metadata_persistently_unavailable() -> bool:
            return True
    from measurement_adapter.sampling_kernel import (
        masked_gumbel_argmax,
        fused_kernel_persistently_unavailable as _fused_sampling_persistently_unavailable,
    )
    try:
        from measurement_adapter.mask_counts_kernel import (
            compute_mask_counts_fused as _compute_mask_counts_fused,
            fused_kernel_persistently_unavailable as _fused_mask_counts_persistently_unavailable,
        )
    except ImportError:  # pragma: no cover - optional CUDA/CuPy fast path
        def _compute_mask_counts_fused(*args, **kwargs):
            return None
        def _fused_mask_counts_persistently_unavailable() -> bool:
            return False
    try:
        from measurement_adapter.counter_rng_kernel import (
            fused_kernel_persistently_unavailable as _fused_counter_rng_persistently_unavailable,
        )
    except ImportError:  # pragma: no cover - optional CUDA/CuPy fast path
        def _fused_counter_rng_persistently_unavailable() -> bool:
            return False
    try:
        from measurement_adapter.partition_update_adapter import (
            partition_update_bucketed_torch as _partition_update_bucketed_torch,
            fused_kernel_persistently_unavailable as _fused_partition_update_persistently_unavailable,
        )
    except ImportError:  # pragma: no cover - optional CUDA/CuPy fast path
        def _partition_update_bucketed_torch(*args, **kwargs):
            return None
        def _fused_partition_update_persistently_unavailable() -> bool:
            return False
    try:
        from measurement_adapter import fused_apply_adapter as _fused_apply_adapter
    except ImportError:  # pragma: no cover - optional CUDA/CuPy fast path
        _fused_apply_adapter = None
    from bucketed_sampler import (
        ActiveQueue,
        build_fixed_k_features,
        counter_uniforms,
        flat_to_bm,
        ordered_compact,
        ordered_partition_scatter,
    )


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
    """Get the best available device.

    Accepts ``"cpu"``, ``"mps"``, ``"cuda"``, or ``"cuda:<index>"``. An
    unindexed ``"cuda"`` (or the auto-detect branch) is normalized to
    ``cuda:<current_device>`` so that downstream caches keyed on
    ``device.index`` (CUDA-graph cache, multi-GPU dispatch) don't fall back
    to ambient ``torch.cuda.current_device`` state — previously
    ``"cuda:1"`` silently fell through to auto-detect and could capture
    graphs on the wrong GPU.
    """
    if device_preference:
        if device_preference == "cpu":
            return torch.device("cpu")
        if device_preference == "mps" and torch.backends.mps.is_available():
            return torch.device("mps")
        if device_preference == "cuda" and torch.cuda.is_available():
            return torch.device(f"cuda:{torch.cuda.current_device()}")
        if device_preference.startswith("cuda:") and torch.cuda.is_available():
            return torch.device(device_preference)

    # Auto-detect best device
    if torch.cuda.is_available():
        return torch.device(f"cuda:{torch.cuda.current_device()}")
    return torch.device("cpu")


def _resolve_device(d: torch.device) -> torch.device:
    """Resolve ``cuda`` (no index) to ``cuda:<current_device>`` so two
    ``torch.device`` instances that point at the same physical device
    compare equal under ``==``.

    PyTorch returns ``torch.device('cuda:0')`` from a tensor created with
    ``device=torch.device('cuda')`` (no index), but
    ``torch.device('cuda') == torch.device('cuda:0')`` is False. Anywhere
    we cache or compare devices for "same hardware" we need to normalize.
    """
    if d.type == "cuda" and d.index is None and torch.cuda.is_available():
        return torch.device("cuda", torch.cuda.current_device())
    return d


def default_reward_fn(costs: torch.Tensor, beta: float = 1.0, alpha: float = 5e-3 , **kwargs) -> torch.Tensor:
    """Default reward function: linear transformation of costs."""
    return beta * (alpha - costs)

def log_reward_fn(costs: torch.Tensor, beta: float = 1.0, alpha: float = 1.0, **kwargs) -> torch.Tensor:
    """Logarithmic reward function for stronger differentiation."""
    # Avoid log(0) by adding a small epsilon
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
