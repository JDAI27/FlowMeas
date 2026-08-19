"""Guardrails for exact full-state evaluation paths.

Large Hubbard workloads must not accidentally enter code paths that allocate
ground-state vectors, dense Hamiltonians, or sparse exact-diagonalization
workspaces. This module keeps the shared cutoff and diagnostic messages in one
place so orchestration, helpers, and estimators fail the same way.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional, Union

try:
    import psutil
except ImportError:  # pragma: no cover - psutil is a declared dependency here
    psutil = None


EXACT_FULL_STATE_QUBIT_LIMIT = 26
COMPLEX128_BYTES = 16
MEMORY_FRACTION_LIMIT = 0.80


class ExactFullStateGuardError(RuntimeError):
    """Raised when a large-system exact full-state path is requested."""


def _mb(num_bytes: float) -> float:
    """Return binary MiB for memory diagnostics."""
    return num_bytes / (1024**2)


def is_large_full_state_system(
    n_qubits: int,
    *,
    qubit_limit: int = EXACT_FULL_STATE_QUBIT_LIMIT,
) -> bool:
    """Return True when exact full-state evaluation is disallowed by size."""
    return int(n_qubits) >= int(qubit_limit)


def estimate_full_state_memory(
    n_qubits: int,
    *,
    n_terms: int = 0,
    method: str = "state_vector",
) -> Dict[str, Optional[float]]:
    """Estimate memory pressure for exact full-state paths.

    Estimates are intentionally conservative diagnostics, not scheduling
    decisions. They are used in guard messages so accidental exact paths explain
    how large the requested state/matrix would have been.
    """
    dim = 2 ** int(n_qubits)
    method = (method or "state_vector").lower()

    estimates: Dict[str, Optional[float]] = {
        "n_qubits": float(n_qubits),
        "dimension": float(dim),
        "state_vector_mb": _mb(dim * COMPLEX128_BYTES),
        "sparse_matrix_mb": _mb(max(int(n_terms), 0) * dim * COMPLEX128_BYTES),
        "workspace_mb": 0.0,
        "dense_matrix_mb": 0.0,
        "available_mb": None,
        "feasible": None,
    }

    if method == "lobpcg":
        estimates["workspace_mb"] = _mb(5 * dim * COMPLEX128_BYTES)
    elif method == "dense":
        estimates["dense_matrix_mb"] = _mb(dim * dim * COMPLEX128_BYTES)
    elif method == "state_vector":
        pass
    else:
        estimates["workspace_mb"] = _mb(3 * dim * COMPLEX128_BYTES)

    estimates["total_mb"] = (
        (estimates["state_vector_mb"] or 0.0)
        + (estimates["sparse_matrix_mb"] or 0.0)
        + (estimates["workspace_mb"] or 0.0)
        + (estimates["dense_matrix_mb"] or 0.0)
    )

    if psutil is not None:
        available_mb = _mb(psutil.virtual_memory().available)
        estimates["available_mb"] = available_mb
        estimates["feasible"] = estimates["total_mb"] < (
            MEMORY_FRACTION_LIMIT * available_mb
        )

    return estimates


def _path_hint(filepath: Optional[Union[str, Path]]) -> str:
    if filepath is None:
        return ""
    path = Path(filepath)
    return f" for {path}"


def format_full_state_guard_message(
    *,
    context: str,
    n_qubits: int,
    n_terms: int = 0,
    method: str = "state_vector",
    filepath: Optional[Union[str, Path]] = None,
    qubit_limit: int = EXACT_FULL_STATE_QUBIT_LIMIT,
) -> str:
    estimates = estimate_full_state_memory(
        n_qubits,
        n_terms=n_terms,
        method=method,
    )
    state_gb = (estimates["state_vector_mb"] or 0.0) / 1024
    total_gb = (estimates["total_mb"] or 0.0) / 1024
    available = estimates.get("available_mb")
    available_text = (
        f", available memory about {available / 1024:.1f} GiB"
        if available is not None
        else ""
    )
    return (
        f"{context}: exact full-state evaluation is blocked for "
        f"n_qubits={n_qubits} >= {qubit_limit}{_path_hint(filepath)}. "
        f"Requested method={method!r} would use a state dimension of "
        f"2^{n_qubits} and at least {state_gb:.1f} GiB for one complex128 "
        f"state vector; estimated total for this path is {total_gb:.1f} GiB"
        f"{available_text}. Use large_hubbard_mode=True / scalable_large "
        "structural reporting, or provide a cached scalar exact energy instead "
        "of requesting ground_state_vector or exact diagonalization. For large "
        "reference energies, use compute_ground_state_dmrg to cache a scalar "
        "DMRG result."
    )


def guard_exact_full_state_request(
    *,
    context: str,
    n_qubits: int,
    n_terms: int = 0,
    method: str = "state_vector",
    filepath: Optional[Union[str, Path]] = None,
    qubit_limit: int = EXACT_FULL_STATE_QUBIT_LIMIT,
) -> None:
    """Raise if a large-system exact full-state path is requested."""
    if not is_large_full_state_system(n_qubits, qubit_limit=qubit_limit):
        return

    message = format_full_state_guard_message(
        context=context,
        n_qubits=n_qubits,
        n_terms=n_terms,
        method=method,
        filepath=filepath,
        qubit_limit=qubit_limit,
    )
    logging.warning("Blocked exact full-state path: %s", message)
    raise ExactFullStateGuardError(message)
