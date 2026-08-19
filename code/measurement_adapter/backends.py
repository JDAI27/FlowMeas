"""Measurement backend resolution for FlowMeas.

This module is the single place that maps a user/config backend choice to the
tableau implementation FlowMeas should instantiate. Keeping the policy
here prevents training and evaluation call sites from each carrying their own
copy of the legacy-vs-CT validation rules.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch


LEGACY_BACKEND = "clifford_map"
TABLEAU_BATCH_BACKEND = "tableau_batch_adapter"
MPS_NATIVE_BACKEND = "mps_native"
AUTO_BACKEND = "auto"
VALID_MEASUREMENT_BACKENDS: Tuple[str, ...] = (
    LEGACY_BACKEND,
    TABLEAU_BATCH_BACKEND,
    MPS_NATIVE_BACKEND,
)


@dataclass(frozen=True)
class MeasurementBackendSelection:
    """Resolved measurement backend and concrete tableau class."""

    name: str
    tableau_cls: type
    auto_selected: bool


def _device_from(device: Union[str, torch.device]) -> torch.device:
    return device if isinstance(device, torch.device) else torch.device(device)


def _import_legacy_tableau() -> type:
    try:
        from ..clifford_map import CliffordMap
    except ImportError:
        from clifford_map import CliffordMap
    return CliffordMap


def get_legacy_tableau_class() -> type:
    """Return the transition-era CPU tableau class through an explicit shim.

    New app-layer code should normally call ``resolve_tableau_backend`` so the
    shared measurement-core path can be selected by configuration. This helper
    exists for legacy-only utilities that still need ``CliffordMap`` semantics
    while keeping the direct import of ``code/clifford_map.py`` centralized.
    """

    return _import_legacy_tableau()


def _import_ct_tableau() -> type:
    from .tableau_batch_adapter import TableauBatchAdapter
    return TableauBatchAdapter


def _import_cutensornet():
    """Probe cuTensorNet availability with a narrow ImportError catch.

    Returns the module that owns ``contract_decompose`` (the gate-split
    primitive the MPS-native backend actually consumes). cuQuantum 25.03+
    exposes this at ``cuquantum.tensornet.experimental``; older releases
    only had ``cuquantum.cutensornet.experimental``. Probe the new path
    first, fall back to the old path, raise ``ImportError`` only if both
    are missing.

    The old ``cuquantum.cutensornet`` module emits a ``DeprecationWarning``
    on import; probing it only when the new path is unavailable keeps the
    common case warning-free and avoids a future failure when cuQuantum
    eventually drops the alias.

    Tests monkeypatch this helper to drive both branches of the resolver
    without needing the real cuQuantum stack.
    """
    cause: Optional[ImportError] = None
    with warnings.catch_warnings():
        # cuquantum-python 25.03 emits package-level deprecation warnings while
        # importing its top-level package, even when callers use the new
        # ``cuquantum.tensornet.experimental`` path. This is an availability
        # probe; keep it quiet so resolver tests and validation scripts do not
        # report warnings unrelated to the selected API.
        warnings.filterwarnings(
            "ignore",
            category=DeprecationWarning,
            module=r"cuquantum(\..*)?",
        )
        try:
            from cuquantum.tensornet.experimental import (  # type: ignore[import-not-found]
                contract_decompose as _cd_new,
            )
            from cuquantum import tensornet  # type: ignore[import-not-found]
            return tensornet
        except ImportError:
            pass
        try:
            from cuquantum.cutensornet.experimental import (  # type: ignore[import-not-found]
                contract_decompose as _cd_old,
            )
            from cuquantum import cutensornet  # type: ignore[import-not-found]
            return cutensornet
        except ImportError as e:
            cause = e
    if cause is not None:
        raise ImportError(
            "measurement_backend='mps_native' requires the cuQuantum / "
            "cutensornet runtime with the ``contract_decompose`` gate-split "
            "primitive (cuquantum.tensornet.experimental on 25.03+; "
            "cuquantum.cutensornet.experimental on older releases). Install "
            "cuquantum-python>=23.10 to enable this backend."
        ) from cause


def resolve_tableau_backend(
    measurement_backend: Optional[str],
    device: Union[str, torch.device],
) -> MeasurementBackendSelection:
    """Resolve a backend name to the tableau implementation class.

    ``None`` and ``"auto"`` select the production default: CT-backed
    ``TableauBatchAdapter`` on CUDA and legacy ``CliffordMap`` on CPU. The CT
    backend remains CUDA-only and fails before any expensive simulator setup if
    selected on CPU.
    """

    resolved_device = _device_from(device)
    auto_selected = measurement_backend is None or measurement_backend == AUTO_BACKEND

    if auto_selected:
        # NOTE: ``auto`` deliberately never resolves to ``mps_native`` in this
        # layer; it has to be selected explicitly.
        name = (
            TABLEAU_BATCH_BACKEND
            if resolved_device.type == "cuda"
            else LEGACY_BACKEND
        )
    else:
        name = str(measurement_backend)

    if name not in VALID_MEASUREMENT_BACKENDS:
        valid = "', '".join(VALID_MEASUREMENT_BACKENDS)
        raise ValueError(
            f"measurement_backend must be '{valid}' or 'auto'; got {measurement_backend!r}"
        )

    if name == TABLEAU_BATCH_BACKEND:
        if resolved_device.type != "cuda":
            raise ValueError(
                "measurement_backend='tableau_batch_adapter' requires a "
                f"CUDA device; got device={resolved_device}. The CT backend "
                "is GPU-only."
            )
        tableau_cls = _import_ct_tableau()
    elif name == MPS_NATIVE_BACKEND:
        if resolved_device.type != "cuda":
            raise ValueError(
                "measurement_backend='mps_native' requires a CUDA device; "
                f"got device={resolved_device}. The MPS-native backend depends "
                "on cuTensorNet primitives."
            )
        # Probe cuTensorNet availability before any expensive setup; raises
        # ImportError (with chained cause) when the runtime is missing.
        _import_cutensornet()
        # MPS-native only swaps state sampling, not Pauli transformation.
        # The estimator still needs a tableau implementation for ``can_measure``,
        # signs, and z_masks. Use the same tableau ``auto`` would pick on
        # CUDA (TableauBatchAdapter).
        tableau_cls = _import_ct_tableau()
    else:
        tableau_cls = _import_legacy_tableau()

    return MeasurementBackendSelection(
        name=name,
        tableau_cls=tableau_cls,
        auto_selected=auto_selected,
    )
