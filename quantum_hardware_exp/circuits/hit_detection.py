#!/usr/bin/env python3
"""
Hit detection and eigenvalue extraction for DSS estimator.

For each circuit U and Pauli P, we compute P' = U P U^† using Clifford conjugation.
If P' is diagonal (Z/I only), the circuit "hits" P. Given a measured bitstring b,
the eigenvalue is sign(P') * (-1)^{parity_{Z}(b)}.
"""

from __future__ import annotations

from typing import Tuple

from qiskit import QuantumCircuit
from qiskit.quantum_info import Pauli, Clifford


def conjugate_pauli_by_circuit(pauli_label: str, circuit: QuantumCircuit) -> Tuple[Pauli, bool]:
    """Return the conjugated Pauli and whether it is diagonal (Z/I only)."""
    p = Pauli(pauli_label)
    c = Clifford(circuit)
    # Evolve (conjugate) Pauli by Clifford: P' = C P C^
    p_prime = p.evolve(c)
    # Diagonal if no X on any qubit (i.e., x vector is all False) and no Y (encoded by both x and z True)
    x = p_prime.x
    z = p_prime.z
    diagonal = not x.any()
    return p_prime, diagonal


def eigenvalue_from_bitstring(conjugated: Pauli, bitstr: str) -> int:
    """Compute ±1 eigenvalue of conjugated Pauli on computational bitstring bitstr.

    Assumes conjugated is diagonal (no X components).
    """
    # Qiskit uses little-endian ordering internally (qubit 0 -> index 0),
    # whereas bitstrings are reported msb-first. Reverse to align with Pauli.z.
    z = conjugated.z
    parity = 0
    for idx, bit in enumerate(reversed(bitstr)):
        if z[idx]:
            parity ^= (bit == '1')
    phase = conjugated.phase % 4  # 0->1, 1->i, 2->-1, 3->-i
    if phase in (1, 3):
        # Should not occur for diagonal Hermitian Paulis; treat as 0 contribution
        # or map to nearest real sign; we choose to ignore (caller should check)
        return 0
    sign_phase = 1 if phase == 0 else -1
    return sign_phase * (1 if parity == 0 else -1)
