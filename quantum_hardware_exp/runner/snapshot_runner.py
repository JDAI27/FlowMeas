#!/usr/bin/env python3
"""
Snapshot runner: executes a schedule of unitary circuits (one snapshot per circuit)
on top of a state-preparation ansatz and returns a single bitstring per circuit.
"""

from __future__ import annotations

from typing import List, Tuple

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector


def run_snapshots_simulator(ansatz: QuantumCircuit, circuits: List[QuantumCircuit], seed: int = 42) -> List[str]:
    """Run one snapshot per circuit on a statevector simulator and return bitstrings."""
    out: List[str] = []
    import numpy as np

    rng = np.random.default_rng(seed)
    for u in circuits:
        n = u.num_qubits
        qc = QuantumCircuit(n)
        qc.compose(ansatz, inplace=True)
        qc.compose(u, inplace=True)
        # Probabilities in computational basis
        sv = Statevector.from_instruction(qc)
        probs = (sv.data.conj() * sv.data).real
        idx = int(rng.choice(2**n, p=probs))
        bitstr = format(idx, f"0{n}b")
        out.append(bitstr)
    return out


def run_snapshots_ibm(ansatz: QuantumCircuit, circuits: List[QuantumCircuit], sampler, seed: int | None = None) -> List[str]:
    """Run one snapshot per circuit on IBM Sampler-like primitive and return bitstrings.

    - If the primitive supports shots, we request shots=1 and read counts.
    - Otherwise, we sample a single bitstring from quasi-probabilities for each circuit.
    """
    from typing import List, Dict
    import numpy as np

    # Build measured circuits (compose ansatz and circuit, then measure)
    measured: List[QuantumCircuit] = []
    for u in circuits:
        n = u.num_qubits
        qc = QuantumCircuit(n)
        qc.compose(ansatz, inplace=True)
        qc.compose(u, inplace=True)
        qc.measure_all()
        measured.append(qc)

    # Try run with shots=1; if not supported, fall back
    try:
        job = sampler.run(measured, shots=1)
        res = job.result()
        out: List[str] = []
        for pub in res:
            if hasattr(pub, "data") and hasattr(pub.data, "meas"):
                cnts = pub.data.meas.get_counts()
                # pick the observed bitstring (only one expected)
                bitstr = max(cnts.items(), key=lambda kv: kv[1])[0]
                out.append(bitstr)
            else:
                # unknown pub shape; fall back to quasi dist path
                raise TypeError("Unknown pub result shape")
        return out
    except Exception:
        pass

    # Fall back: quasi-probabilities path
    job = sampler.run(measured)
    res = job.result()
    rng = np.random.default_rng(seed)
    out2: List[str] = []
    # Qiskit SamplerResult API exposes quasi_dists or similar
    if hasattr(res, "quasi_dists"):
        for i, qd in enumerate(res.quasi_dists):
            n = measured[i].num_qubits
            dist = np.zeros(2**n)
            for k, v in qd.items():
                dist[int(k)] = max(0.0, float(v))
            s = dist.sum()
            dist = dist / s if s > 0 else np.full_like(dist, 1.0 / len(dist))
            idx = int(rng.choice(2**n, p=dist))
            bitstr = format(idx, f"0{n}b")
            out2.append(bitstr)
        return out2

    # Last resort: try to read counts arrays
    try:
        out3: List[str] = []
        for pub in res:
            if hasattr(pub, "data") and hasattr(pub.data, "get_counts"):
                cnts = pub.data.get_counts()
                bitstr = max(cnts.items(), key=lambda kv: kv[1])[0]
                out3.append(bitstr)
        if out3:
            return out3
    except Exception:
        pass

    raise RuntimeError("Unsupported sampler result format for snapshot extraction")
