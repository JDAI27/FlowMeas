#!/usr/bin/env python3
"""
Diagnostic script to identify DSS estimation issues.
"""

import numpy as np
from pathlib import Path
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, SparsePauliOp, Pauli, Clifford

from quantum_hardware_exp.hamiltonian_loader import HamiltonianLoader
from quantum_hardware_exp.state_preparation import StatePreparator
from quantum_hardware_exp.circuits.dss_loader import load_circuits_from_checkpoint
from quantum_hardware_exp.circuits.hit_detection import conjugate_pauli_by_circuit


def test_true_energy():
    """Compute the actual ground state energy using Statevector."""
    print("=" * 70)
    print("TEST 1: True Energy Verification")
    print("=" * 70)

    loader = HamiltonianLoader()
    prep = StatePreparator()
    Ham = loader.load_hub_square_U2()

    try:
        ansatz = prep.load_ground_state_hub_2x2_U2()
        print(f"Loaded ansatz: {ansatz.num_qubits} qubits, {ansatz.num_parameters} params")
    except Exception as e:
        print(f"Could not load ansatz: {e}")
        return

    # Compute exact energy using statevector
    sv = Statevector.from_instruction(ansatz)
    exact_E = sv.expectation_value(Ham).real
    print(f"\nExact energy from Statevector: {exact_E:.8f} Ha")
    print(f"Hardcoded true_E in code:      -4.20267211 Ha")
    print(f"User reported true_E:          -3.52757248 Ha")

    return exact_E


def test_pauli_convention():
    """Test if Pauli string convention is correct."""
    print("\n" + "=" * 70)
    print("TEST 2: Pauli String Convention")
    print("=" * 70)

    loader = HamiltonianLoader()
    Ham = loader.load_hub_square_U2()
    paulis, coeffs, cI = loader.format_spo(Ham)

    print(f"Number of Pauli terms: {len(paulis)}")
    print(f"Identity coefficient (cI): {cI}")
    print(f"\nFirst 5 Pauli strings from Hamiltonian:")
    for i, (p, c) in enumerate(zip(paulis[:5], coeffs[:5])):
        print(f"  {p} : {c:.4f}")

    # Test: create Pauli directly vs reversed
    test_pauli = paulis[0]
    p_direct = Pauli(test_pauli)
    p_reversed = Pauli(test_pauli[::-1])

    print(f"\nTest Pauli: '{test_pauli}'")
    print(f"  Direct Pauli.z: {p_direct.z}")
    print(f"  Reversed Pauli.z: {p_reversed.z}")


def test_hit_rates():
    """Check hit rates for Flow-Shadow circuits."""
    print("\n" + "=" * 70)
    print("TEST 3: Hit Rate Analysis")
    print("=" * 70)

    loader = HamiltonianLoader()
    Ham = loader.load_hub_square_U2()
    paulis, coeffs, cI = loader.format_spo(Ham)

    ckpt_path = "quantum_hardware_exp/data/checkpoint_square_U2.pth"
    fs_circuits, n = load_circuits_from_checkpoint(ckpt_path, limit=100)
    print(f"Loaded {len(fs_circuits)} Flow-Shadow circuits ({n} qubits)")

    # Count hits per Pauli
    hits_per_pauli = {i: 0 for i in range(len(paulis))}

    for u in fs_circuits:
        for p_idx, p_label in enumerate(paulis):
            _, diag = conjugate_pauli_by_circuit(p_label, u)
            if diag:
                hits_per_pauli[p_idx] += 1

    print(f"\nHit rates (out of {len(fs_circuits)} circuits):")
    total_hits = 0
    zero_hit_paulis = []
    for p_idx in range(len(paulis)):
        hits = hits_per_pauli[p_idx]
        total_hits += hits
        rate = hits / len(fs_circuits) * 100
        if hits == 0:
            zero_hit_paulis.append(p_idx)
        if p_idx < 10 or hits == 0:
            print(f"  Pauli {p_idx} ({paulis[p_idx]}): {hits} hits ({rate:.1f}%)")

    avg_hits = total_hits / len(paulis)
    print(f"\nAverage hits per Pauli: {avg_hits:.1f}")
    print(f"Paulis with ZERO hits: {len(zero_hit_paulis)}")
    if zero_hit_paulis:
        print(f"  Zero-hit Paulis: {zero_hit_paulis[:10]}{'...' if len(zero_hit_paulis) > 10 else ''}")


def test_frame_convention():
    """Test Schrödinger vs Heisenberg frame."""
    print("\n" + "=" * 70)
    print("TEST 4: Frame Convention (U P U† vs U† P U)")
    print("=" * 70)

    # Simple test circuit
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)

    p = Pauli("ZZ")
    c = Clifford(qc)

    p_schrodinger = p.evolve(c, frame='s')  # U P U†
    p_heisenberg = p.evolve(c, frame='h')   # U† P U (default)

    print(f"Test circuit: H(0), CX(0,1)")
    print(f"Test Pauli: ZZ")
    print(f"\nSchrödinger frame (U P U†):")
    print(f"  Result: {p_schrodinger.to_label()}, phase={p_schrodinger.phase}")
    print(f"  Diagonal: {not p_schrodinger.x.any()}")

    print(f"\nHeisenberg frame (U† P U):")
    print(f"  Result: {p_heisenberg.to_label()}, phase={p_heisenberg.phase}")
    print(f"  Diagonal: {not p_heisenberg.x.any()}")


def test_bitstring_eigenvalue():
    """Test eigenvalue computation from bitstrings."""
    print("\n" + "=" * 70)
    print("TEST 5: Eigenvalue Computation")
    print("=" * 70)

    from quantum_hardware_exp.circuits.hit_detection import eigenvalue_from_bitstring

    # Test case: Z on qubit 0
    p = Pauli("ZI")  # Z on qubit 1 in Qiskit convention (big-endian)
    print(f"Pauli: ZI (Z on qubit 1 in Qiskit big-endian)")
    print(f"  p.z = {p.z}")

    for bitstr in ["00", "01", "10", "11"]:
        ev = eigenvalue_from_bitstring(p, bitstr)
        print(f"  Bitstring '{bitstr}': eigenvalue = {ev}")

    # Test case: Z on qubit 0 in little-endian
    p2 = Pauli("IZ")  # Z on qubit 0 in Qiskit convention
    print(f"\nPauli: IZ (Z on qubit 0 in Qiskit big-endian)")
    print(f"  p.z = {p2.z}")

    for bitstr in ["00", "01", "10", "11"]:
        ev = eigenvalue_from_bitstring(p2, bitstr)
        print(f"  Bitstring '{bitstr}': eigenvalue = {ev}")


def test_simple_expectation():
    """Test expectation value estimation on a simple case."""
    print("\n" + "=" * 70)
    print("TEST 6: Simple Expectation Value Test")
    print("=" * 70)

    from quantum_hardware_exp.estimation.dss_estimator import estimate_energy

    # Prepare |+⟩ state on 2 qubits
    ansatz = QuantumCircuit(2)
    ansatz.h(0)
    ansatz.h(1)

    # Measure with identity circuit (Z basis)
    circuits = [QuantumCircuit(2) for _ in range(100)]

    # Simulate bitstrings (|+⟩ gives uniform distribution)
    np.random.seed(42)
    bitstrings = [format(np.random.randint(4), '02b') for _ in range(100)]

    # Test Pauli: ZI (should give 0 for |+⟩)
    paulis = ["ZI"]
    coeffs = [1.0]
    cI = 0.0

    E, exp_map = estimate_energy(bitstrings, circuits, paulis, coeffs, cI)
    print(f"State: |++⟩")
    print(f"Observable: ZI")
    print(f"Expected: 0.0")
    print(f"Estimated: {exp_map[0]:.4f} (from {100} samples)")


if __name__ == "__main__":
    exact_E = test_true_energy()
    test_pauli_convention()
    test_hit_rates()
    test_frame_convention()
    test_bitstring_eigenvalue()
    test_simple_expectation()

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    if exact_E is not None:
        print(f"Computed exact energy: {exact_E:.8f} Ha")
        print(f"\nIf this differs from -3.527... or -4.202..., the ansatz")
        print(f"may not be the ground state, or there's a convention issue.")
