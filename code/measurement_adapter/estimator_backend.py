"""Estimator-time compatibility shim for legacy Pauli transforms.

FlowMeas's energy estimator still needs the legacy ``Pauli.apply_clifford``
phase model while the shared measurement-core estimator surface is being wired
in. Keep that dependency behind this module so app-layer estimator code depends
on an explicit adapter boundary instead of importing legacy measurement
internals directly.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Optional, Protocol, Type

import torch


@dataclass
class TransformedPauliBatch:
    """Batched output of applying a tableau batch to Hamiltonian Paulis.

    Treat these tensors as read-only; clone fields before handing them to
    asynchronous consumers or caches that may mutate them.
    """

    x: torch.Tensor
    z: torch.Tensor
    phase: torch.Tensor


class TableauForEstimator(Protocol):
    """Legacy-Pauli-compatible tableau surface consumed by the shim.

    ``EstimatorBackend`` passes the tableau to ``Pauli.apply_clifford``; these
    are the attributes that legacy implementation reads for transforms and
    phases. Future CT-native estimator code should replace this structural seam
    rather than add more app-layer reads of legacy internals.
    """

    batch_size: int
    n_measurements: int
    device: torch.device
    W: torch.Tensor
    heis_phase_vec: torch.Tensor


class EstimatorBackend:
    """Compatibility facade for estimator Pauli transforms.

    This class is intentionally small: it owns the import of
    ``pauli_tracker.Pauli`` and returns plain tensors to callers. It currently
    covers the Pauli-transform/phase slice of the broader estimator
    boundary; hit tables, sign extraction, and sampling can be
    added here as those paths move to the shared measurement core.
    """

    # Shared class cache avoids rebinding the legacy Pauli import on the hot
    # estimator path. Tests that monkey-patch Pauli should reset this to None.
    _pauli_cls: ClassVar[Optional[Type]] = None

    def compute_transformed_paulis(
        self,
        tableau: TableauForEstimator,
        pauli_vecs: torch.Tensor,
        pauli_phases: torch.Tensor,
    ) -> TransformedPauliBatch:
        """Apply ``tableau`` to symplectic Pauli rows and return tensor fields."""

        self._validate_pauli_inputs(pauli_vecs, pauli_phases)
        pauli_cls = self._pauli_class()
        n_qubits = pauli_vecs.shape[-1] // 2
        x_in = pauli_vecs[:, :n_qubits]
        z_in = pauli_vecs[:, n_qubits:]
        pauli_in = pauli_cls(x_in, z_in, pauli_phases)
        pauli_out = pauli_in.apply_clifford(tableau)
        return TransformedPauliBatch(
            x=pauli_out.x,
            z=pauli_out.z,
            phase=pauli_out.phase,
        )

    def heisenberg_phase_row(
        self,
        tableau: TableauForEstimator,
        batch_idx: int = 0,
        circuit_idx: int = 0,
    ) -> torch.Tensor:
        """Expose legacy phase-row debug data without leaking the import site."""

        return tableau.heis_phase_vec[batch_idx, circuit_idx]

    @staticmethod
    def _validate_pauli_inputs(
        pauli_vecs: torch.Tensor,
        pauli_phases: torch.Tensor,
    ) -> None:
        if pauli_vecs.ndim != 2:
            raise ValueError(
                "pauli_vecs must be 2-D with shape (K, 2n); "
                f"got {tuple(pauli_vecs.shape)}"
            )
        if pauli_vecs.shape[-1] % 2:
            raise ValueError(
                "pauli_vecs width must be even (2n); "
                f"got {pauli_vecs.shape[-1]}"
            )
        expected_phase_shape = (pauli_vecs.shape[0],)
        if tuple(pauli_phases.shape) != expected_phase_shape:
            raise ValueError(
                f"pauli_phases must have shape {expected_phase_shape}; "
                f"got {tuple(pauli_phases.shape)}"
            )

    @classmethod
    def _pauli_class(cls):
        if cls._pauli_cls is not None:
            return cls._pauli_cls
        try:
            from ..pauli_tracker import Pauli
        except ImportError:
            from pauli_tracker import Pauli
        cls._pauli_cls = Pauli
        return Pauli


def create_estimator_backend() -> EstimatorBackend:
    """Factory used by app-layer estimators to make the shim explicit."""

    return EstimatorBackend()
