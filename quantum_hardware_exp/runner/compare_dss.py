#!/usr/bin/env python3
"""
Compare Flow-Shadow DSS circuits vs Qiskit commuting-group baseline under a fixed
snapshot budget N, using Qiskit simulators or IBM hardware via Sampler.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
from qiskit.quantum_info import SparsePauliOp

from quantum_hardware_exp.hamiltonian_loader import HamiltonianLoader
from quantum_hardware_exp.state_preparation import StatePreparator
from quantum_hardware_exp.circuits.dss_loader import load_circuits_from_checkpoint
from quantum_hardware_exp.runner.snapshot_runner import run_snapshots_simulator, run_snapshots_ibm, run_snapshots_estimator
from quantum_hardware_exp.estimation.dss_estimator import estimate_energy
from qiskit_ibm_runtime.options import TwirlingOptions, SamplerOptions

def _truncate_or_repeat(circuits: List, N: int) -> List:
    if len(circuits) >= N:
        return circuits[:N]
    out = []
    i = 0
    while len(out) < N:
        out.append(circuits[i % len(circuits)])
        i += 1
    return out


def build_qiskit_commuting_baseline(pauli_strings: List[str]) -> List:
    """Build Qiskit's default qubit-wise commuting baseline with single-qubit rotations.

    - Group Pauli strings into qubit-wise commuting sets via SparsePauliOp.group_commuting(qubit_wise=True).
    - For each group, choose a per-qubit measurement basis that diagonalizes all strings in the group:
      if any string has X on qubit q -> X basis; elif any has Y -> Y basis; else Z basis.
    - Construct a unitary circuit per group that applies the basis change (H for X, S† then H for Y).
    """
    from qiskit import QuantumCircuit
    groups = SparsePauliOp.from_list([(p, 1.0) for p in pauli_strings]).group_commuting(qubit_wise=True)
    circuits = []
    for g in groups:
        labels = [lab for lab, _ in g.to_list()]
        n = len(labels[0])
        # Decide per-qubit basis from the entire group
        basis = ['Z'] * n
        for lab in labels:
            for q, op in enumerate(lab):
                if op == 'X':
                    basis[q] = 'X'
                elif op == 'Y' and basis[q] != 'X':
                    # X has precedence; otherwise Y
                    basis[q] = 'Y'
                elif op == 'Z':
                    basis[q] = basis[q]  # keep existing (Z if none)
        qc = QuantumCircuit(n)
        for q, b in enumerate(basis):
            if b == 'X':
                qc.h(q)
            elif b == 'Y':
                qc.sdg(q)
                qc.h(q)
        circuits.append(qc)
    return circuits


def _set_default_shots(estimator: Any, shots: int) -> None:
    """Best-effort assignment of default shots on a Qiskit Estimator-like primitive."""
    options = getattr(estimator, "options", None)
    if options is None:
        return
    # Direct attribute access (e.g., dataclass-style)
    if hasattr(options, "default_shots"):
        try:
            setattr(options, "default_shots", shots)
        except Exception:
            pass
    # Mapping-style (e.g., dict-like Options)
    for key in ("default_shots", "shots"):
        try:
            options[key] = shots
            break
        except Exception:
            continue
    # Update method fallback
    if hasattr(options, "update"):
        try:
            options.update({"default_shots": shots})
        except Exception:
            pass


def _initialize_ibm_estimator(backend, shots: int):
    """Instantiate an IBM Runtime Estimator primitive and configure default shots."""
    estimator = None
    import_errors = []
    try:
        from qiskit_ibm_runtime import EstimatorV2  # type: ignore
        estimator = EstimatorV2(backend=backend)
    except ImportError as exc:
        import_errors.append(exc)
    if estimator is None:
        try:
            from qiskit_ibm_runtime import Estimator, Options  # type: ignore
            try:
                options = Options()
            except Exception:
                options = None
            estimator = Estimator(backend=backend, options=options)
        except ImportError as exc:
            import_errors.append(exc)
    if estimator is None:
        raise ImportError(
            "Failed to import an Estimator primitive from qiskit_ibm_runtime. "
            f"Import errors: {import_errors}"
        )
    _set_default_shots(estimator, shots)
    return estimator


def _run_estimator(estimator: Any, circuit, observable):
    """Execute estimator.run with broad compatibility across Qiskit versions."""
    try:
        job = estimator.run(circuits=[circuit], observables=[observable])
    except TypeError:
        try:
            job = estimator.run([(circuit, observable)])
        except TypeError:
            try:
                from qiskit_ibm_runtime import EstimatorPub  # type: ignore
                pub = EstimatorPub(circuit=circuit, observables=observable)
                job = estimator.run([pub])
            except Exception as exc:
                raise RuntimeError("Unsupported estimator.run signature for provided primitive") from exc
    return job


def _extract_estimator_value(result: Any) -> float:
    """Extract the expectation value from various Estimator result shapes."""
    # Direct access (legacy Estimator)
    if hasattr(result, "values"):
        values = result.values
        if isinstance(values, (list, tuple, np.ndarray)):
            return float(values[0])
        return float(values)
    # IBM Runtime V2 returns list-like container
    if isinstance(result, list) and result:
        return _extract_estimator_value(result[0])
    if hasattr(result, "data"):
        data = result.data
        if isinstance(data, list) and data:
            return _extract_estimator_value(data[0])
        if hasattr(data, "evs"):
            evs = data.evs
            if isinstance(evs, (list, tuple, np.ndarray)):
                return float(evs[0])
            return float(evs)
        if hasattr(data, "value"):
            return float(data.value)
        if hasattr(data, "values"):
            vals = data.values
            if isinstance(vals, (list, tuple, np.ndarray)):
                return float(vals[0])
            return float(vals)
        if isinstance(data, dict):
            for key in ("value", "values", "evs"):
                if key in data:
                    val = data[key]
                    if isinstance(val, (list, tuple, np.ndarray)):
                        return float(val[0])
                    return float(val)
    raise RuntimeError("Unable to extract expectation value from Estimator result")


def estimate_energy_estimator(
    estimator,
    circuit,
    pauli_strings: List[str],
    coefficients: List[float],
    identity_weight: float,
) -> float:
    """Evaluate Hamiltonian energy using a Qiskit Estimator primitive."""
    if not pauli_strings:
        return float(identity_weight)
    observable = SparsePauliOp.from_list([(p, float(c)) for p, c in zip(pauli_strings, coefficients)])
    job = _run_estimator(estimator, circuit, observable)
    res = job.result()
    exp_val = _extract_estimator_value(res)
    return float(identity_weight + exp_val)


def main():
    ap = argparse.ArgumentParser(description="DSS-style comparison runner (Flow-Shadow vs Qiskit baseline)")
    ap.add_argument("--budget", type=int, default=1000, help="Snapshot budget N (one bitstring per circuit)")
    ap.add_argument("--checkpoint", type=Path, default=Path("quantum_hardware_exp/data/checkpoint_square_U2.pth"))
    ap.add_argument("--backend", type=str, default=None, help="IBM backend name for hardware; omit for simulator")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--repeats", type=int, default=1, help="Repeat full N-snapshot run K times for statistics")
    ap.add_argument("--out", type=Path, default=None, help="Optional JSON filepath to save results; defaults under quantum_hardware_exp/results/")
    args = ap.parse_args()

    # Load H2 Hamiltonian and ansatz
    loader = HamiltonianLoader()
    #cI, paulis, coeffs = loader.load_h2_8qubit()
    prep = StatePreparator()
    #state, true_E = prep.load_ground_state("H2", 8)
    #ansatz = prep.create_state_preparation_circuit(state, 8)
    Ham = loader.load_hub_square_U2()
    ansatz = prep.load_ground_state_hub_2x2_U2()
    paulis, coeffs, cI = loader.format_spo(Ham)
    true_E = -4.202672114583806
    # Flow-Shadow circuits
    fs_circuits, n = load_circuits_from_checkpoint(str(args.checkpoint))
    fs_sched = _truncate_or_repeat(fs_circuits, args.budget)

    # Baseline circuits (Qiskit's qubit-wise commuting grouping)
    qk_circuits = build_qiskit_commuting_baseline(paulis)
    qk_sched = _truncate_or_repeat(qk_circuits, args.budget)

    # Execute snapshots (with repeats for statistics)
    fs_energies: List[float] = []
    qk_energies: List[float] = []
    qk_estim_energies: List[float] = []
    if args.backend or args.recover:
        from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2, Session, EstimatorV2
        svc = QiskitRuntimeService()
    if args.backend:
        backend = svc.backend(args.backend)
        with Session(backend=backend) as session:
            sampler = SamplerV2(session)
            for r in range(args.repeats):
                fs_bits = run_snapshots_ibm(ansatz, fs_sched, sampler, backend)
                qk_bits = run_snapshots_ibm(ansatz, qk_sched, sampler, backend)
                fs_E, _ = estimate_energy(fs_bits, fs_sched, paulis, coeffs, cI)
                qk_E, _ = estimate_energy(qk_bits, qk_sched, paulis, coeffs, cI)
                fs_energies.append(fs_E)
                qk_energies.append(qk_E)
            qk_estim_energies = run_snapshots_estimator(ansatz, Ham, args.repeats, args.budget, session, backend)
    else:
        for r in range(args.repeats):
            fs_bits = run_snapshots_simulator(ansatz, fs_sched, seed=args.seed + r)
            qk_bits = run_snapshots_simulator(ansatz, qk_sched, seed=args.seed + 1000 + r)
            fs_E, _ = estimate_energy(fs_bits, fs_sched, paulis, coeffs, cI)
            qk_E, _ = estimate_energy(qk_bits, qk_sched, paulis, coeffs, cI)
            fs_energies.append(fs_E)
            qk_energies.append(qk_E)

    fs_E = float(np.mean(fs_energies))
    qk_E = float(np.mean(qk_energies))
    qk_estim_E = float(np.mean(qk_estim_energies))
    fs_std = float(np.std(fs_energies, ddof=1)) if len(fs_energies) > 1 else 0.0
    qk_std = float(np.std(qk_energies, ddof=1)) if len(qk_energies) > 1 else 0.0
    qk_estim_std = float(np.std(qk_estim_energies, ddof=1)) if len(qk_estim_energies) > 1 else 0.0
    def ci95(std, n):
        return 1.96 * std / np.sqrt(n) if n > 1 else 0.0
    fs_ci = ci95(fs_std, len(fs_energies))
    qk_ci = ci95(qk_std, len(qk_energies))
    qk_estim_ci = ci95(qk_estim_std, len(qk_estim_energies))

    # Report
    print("\n" + "=" * 70)
    print("DSS COMPARISON (SNAPSHOT BUDGET)")
    print("=" * 70)
    print(f"Budget N: {args.budget} snapshots; repeats: {args.repeats}; device: {'hardware:'+args.backend if args.backend else 'simulator'}")
    print(f"True energy: {true_E:.6f} Ha")
    print("-" * 70)
    print(f"Flow-Shadow: mean {fs_E:+.6f} Ha  (std {fs_std:.6f}, CI95 ±{fs_ci:.6f})  | err {abs(fs_E-true_E):.6f}")
    print(f"Qiskit Base: mean {qk_E:+.6f} Ha  (std {qk_std:.6f}, CI95 ±{qk_ci:.6f})  | err {abs(qk_E-true_E):.6f}")
    print(f"Qiskit Estim Base: mean {qk_estim_E:+.6f} Ha  (std {qk_estim_std:.6f}, CI95 ±{qk_estim_ci:.6f})  | err {abs(qk_estim_E-true_E):.6f}")
    impr = (abs(qk_E-true_E) - abs(fs_E-true_E)) / max(1e-12, abs(qk_E-true_E))
    print("-" * 70)
    print(f"Relative improvement: {impr*100:.2f}%")

    result = {
        "flow_shadow": {"energies": fs_energies, "mean": fs_E, "std": fs_std, "ci95": fs_ci, "abs_err": abs(fs_E-true_E)},
        "qiskit": {"energies": qk_energies, "mean": qk_E, "std": qk_std, "ci95": qk_ci, "abs_err": abs(qk_E-true_E)},
        "budget": args.budget,
        "repeats": args.repeats,
        "backend": args.backend,
        "checkpoint": str(args.checkpoint),
        "seed": args.seed,
        "true_energy": true_E,
    }

    # Optionally write JSON to results directory
    try:
        import json, os
        from datetime import datetime
        out_path = args.out
        if out_path is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            base = Path("quantum_hardware_exp/results")
            base.mkdir(parents=True, exist_ok=True)
            out_path = base / f"dss_compare_{ts}.json"
        else:
            out_dir = Path(out_path).parent
            out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nSaved results to: {out_path}")
    except Exception as e:
        print(f"Warning: failed to save results JSON: {e}")

    return result


if __name__ == "__main__":
    main()
