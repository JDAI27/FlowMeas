"""FlowMeas measurement adapter package.

Bridges FlowMeas's legacy (B, M)-shaped per-step training stack to the
clifford-tableau shared core. Per (v1.1).

Runtime dependencies (not in base requirements.txt — see
``requirements-measurement-adapter.txt`` at the repo root):
- ``cupy-cuda12x`` (or ``cupy-cuda11x`` matching your toolkit)
- ``clifford-tableau`` (sibling repo, ``pip install -e../clifford-tableau``)

This package is GPU-only at runtime, but ``import measurement_adapter``
itself is safe on CPU-only hosts and hosts without ``clifford_tableau``.
The ``ActionAdapter`` / ``TableauBatchAdapter`` symbols are resolved
lazily via ``__getattr__`` so they only trigger CT/CuPy imports when
actually accessed. This keeps CPU-only test collection and tooling
imports unaffected by the optional GPU stack.
"""

__all__ = [
    "AUTO_BACKEND",
    "LEGACY_BACKEND",
    "MPS_NATIVE_BACKEND",
    "MeasurementBackendSelection",
    "TABLEAU_BATCH_BACKEND",
    "VALID_MEASUREMENT_BACKENDS",
    "ActionAdapter",
    "CuTensorNetMPSBackend",
    "CuTensorNetMPSOps",
    "EstimatorBackend",
    "MPSNativeBackend",
    "MPSOps",
    "TableauBatchAdapter",
    "TableauFeatureExtractor",
    "TransformedPauliBatch",
    "create_estimator_backend",
    "get_legacy_tableau_class",
    "resolve_tableau_backend",
]


def __getattr__(name):
    if name in {
        "AUTO_BACKEND",
        "LEGACY_BACKEND",
        "MPS_NATIVE_BACKEND",
        "MeasurementBackendSelection",
        "TABLEAU_BATCH_BACKEND",
        "VALID_MEASUREMENT_BACKENDS",
        "get_legacy_tableau_class",
        "resolve_tableau_backend",
    }:
        from . import backends
        return getattr(backends, name)
    if name == "ActionAdapter":
        from .action_adapter import ActionAdapter
        return ActionAdapter
    if name == "TableauBatchAdapter":
        from .tableau_batch_adapter import TableauBatchAdapter
        return TableauBatchAdapter
    if name == "TableauFeatureExtractor":
        from .gipte_features import TableauFeatureExtractor
        return TableauFeatureExtractor
    if name in {"EstimatorBackend", "TransformedPauliBatch", "create_estimator_backend"}:
        from . import estimator_backend
        return getattr(estimator_backend, name)
    if name in {"MPSNativeBackend", "MPSOps"}:
        # Lazy import keeps CPU-only tooling cheap while still exposing the
        # torch-only dense-reference MPS backend on demand.
        from . import mps_native_backend
        return getattr(mps_native_backend, name)
    if name in {"CuTensorNetMPSBackend", "CuTensorNetMPSOps"}:
        from . import mps_native_cutensornet
        return getattr(mps_native_cutensornet, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
