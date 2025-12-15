#!/usr/bin/env python3
"""
Load DSS-style shallow Clifford measurement circuits from a Flow-Shadow checkpoint.

Builds unitary-only QuantumCircuits (no measurements). Each circuit corresponds
to a snapshot U_i in the DSS schedule. Composite single-qubit gates are expanded
explicitly (HS = S then H, SH = H then S, HSH = H then S then H).
"""

from __future__ import annotations

from typing import List, Tuple

import torch
from qiskit import QuantumCircuit


def _append_single(qc: QuantumCircuit, gate: str, q: int):
    if gate == "H":
        qc.h(q)
    elif gate == "S":
        qc.s(q)
    elif gate == "HS":
        qc.s(q)
        qc.h(q)
    elif gate == "SH":
        qc.h(q)
        qc.s(q)
    elif gate == "HSH":
        qc.h(q)
        qc.s(q)
        qc.h(q)
    else:
        raise ValueError(f"Unknown single-qubit gate {gate}")


def _append_two(qc: QuantumCircuit, gate: str, c: int, t: int):
    if gate == "CNOT":
        qc.cx(c, t)
    else:
        raise ValueError(f"Unknown two-qubit gate {gate}")


def load_circuits_from_checkpoint(ckpt_path: str, limit: int | None = None) -> Tuple[List[QuantumCircuit], int]:
    """
    Load DSS measurement circuits from Flow-Shadow checkpoint.

    Returns a list of unitary-only circuits and the number of qubits.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu")
    n_qubits = int(ckpt.get("n_qubits", 0) or 0)
    if n_qubits <= 0:
        raise ValueError("Checkpoint missing n_qubits")
    action_mapping = ckpt["action_mapping"]
    terminal = ckpt.get("terminal_index", None)

    # Choose first top_trajectories entry
    traj = ckpt["top_trajectories"][0]
    actions_2d = traj["actions"]  # [n_circuits, max_len]
    lengths_1d = traj["lengths"]
    n_rows = int(lengths_1d.shape[0])
    if limit is not None:
        n_rows = min(n_rows, limit)

    circuits: List[QuantumCircuit] = []
    for i in range(n_rows):
        L = int(lengths_1d[i])
        row = actions_2d[i].tolist()
        qc = QuantumCircuit(n_qubits)
        for step in range(L):
            aid = int(row[step])
            if terminal is not None and aid == terminal:
                break
            spec = action_mapping.get(aid)
            if spec is None:
                continue
            gate = spec[0]
            if gate == "terminal":
                break
            if len(spec) == 2:
                _append_single(qc, gate, int(spec[1]))
            elif len(spec) == 3:
                _append_two(qc, gate, int(spec[1]), int(spec[2]))
            else:
                raise ValueError(f"Invalid action mapping entry: {spec}")
        circuits.append(qc)

    return circuits, n_qubits

