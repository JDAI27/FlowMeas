#!/usr/bin/env python3
"""
DSS estimator for energy: averages over snapshots that hit each Pauli (Eq. B2).
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Pauli

from quantum_hardware_exp.circuits.hit_detection import conjugate_pauli_by_circuit, eigenvalue_from_bitstring


def estimate_pauli_expectations(
    bitstrings: List[str],
    circuits: List[QuantumCircuit],
    pauli_strings: List[str],
) -> Dict[int, float]:
    """Return ô(P) for each Pauli index using DSS averaging over hits.

    ô(P) = (1/h(P)) Σ_i ⟨b_i| U_i P U_i^† |b_i⟩ with h(P) = # hits.
    """
    assert len(bitstrings) == len(circuits)
    nP = len(pauli_strings)
    # Precompute conjugations per circuit
    conj_cache: List[List[Tuple[Pauli, bool]]] = []
    for u in circuits:
        row = []
        for p_label in pauli_strings:
            p_prime, diag = conjugate_pauli_by_circuit(p_label, u)
            row.append((p_prime, diag))
        conj_cache.append(row)

    estimates: Dict[int, float] = {}
    for p_idx, p_label in enumerate(pauli_strings):
        acc = 0.0
        hits = 0
        for i, (b, u) in enumerate(zip(bitstrings, circuits)):
            p_prime, diag = conj_cache[i][p_idx]
            if not diag:
                continue
            val = eigenvalue_from_bitstring(p_prime, b)
            # eigenvalue_from_bitstring may return 0 for non-real phases; skip
            if val == 0:
                continue
            acc += val
            hits += 1
        estimates[p_idx] = (acc / hits) if hits > 0 else 0.0
    return estimates


def estimate_energy(
    bitstrings: List[str],
    circuits: List[QuantumCircuit],
    pauli_strings: List[str],
    coefficients: List[float],
    identity_weight: float,
) -> Tuple[float, Dict[int, float]]:
    """Compute DSS energy estimate from snapshot outcomes and circuits."""
    exp_map = estimate_pauli_expectations(bitstrings, circuits, pauli_strings)
    energy = identity_weight
    for i, c in enumerate(coefficients):
        energy += float(c) * float(exp_map.get(i, 0.0))
    return float(energy), exp_map

