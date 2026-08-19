#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quantum action mapping for GFlowNet circuit generation.

This module defines the mapping between discrete GFlowNet actions and quantum gates.
It is the single source of truth used by both GFlowNet training and energy estimation.

Action Space:
=============
For n qubits, the action space consists of:

1. Single-qubit Clifford gates (5 gates × n qubits):
   - H (Hadamard): X↔Z, Y→-Y
   - S (Phase): X→Y, Z→Z, Y→-X
   - HS (H then S): X→Y, Y→-Z, Z→X
   - SH (S then H): X→Z, Y→X, Z→-Y
   - HSH: X→X, Y→-Y, Z→-Z

2. Two-qubit gates (CNOT, nearest-neighbor):
   - CNOT(q, q+1): Control q, target q+1
   - CNOT(q+1, q): Control q+1, target q

3. Terminal action:
   - Signals end of circuit generation

Action Index Layout:
    [0, 5n-1]: Single-qubit gates (H, S, HS, SH, HSH on each qubit)
    [5n, 5n + 2(n-1) - 1]: CNOT gates (forward and reverse for each pair)
    [5n + 2(n-1)]: Terminal action

Example (n=2 qubits):
    0: H on qubit 0      5: HS on qubit 0     10: CNOT(0,1)
    1: H on qubit 1      6: HS on qubit 1     11: CNOT(1,0)
    2: S on qubit 0      7: SH on qubit 0     12: terminal
    3: S on qubit 1      8: SH on qubit 1
    4:...               9: HSH on qubit 0
"""

from typing import Dict, Tuple, Union


def build_action_mapping(n_qubits: int) -> Tuple[Dict[int, Union[Tuple[str], Tuple[str, int], Tuple[str, int, int]]], int]:
    """
    Build action mapping for quantum gates.

    This is the single source of truth for action-to-gate mapping used by both
    GFlowNet training (GFNs.py) and energy estimation (energy_estimator.py).

    Args:
        n_qubits: Number of qubits in the system

    Returns:
        actions: Dictionary mapping action indices to gate tuples:
                 - Single qubit: (gate_name, qubit)
                 - Two qubit: (gate_name, control, target)
                 - Terminal: ("terminal",)
        terminal_index: Index of the terminal action

    Example:
        >>> actions, terminal = build_action_mapping(2)
        >>> actions[0]
        ('H', 0)
        >>> actions[terminal]
        ('terminal',)
    """
    actions = {}
    idx = 0
    
    # Single qubit gates - applied to each qubit
    for gate in ["H", "S", "HS", "SH", "HSH"]:
        for q in range(n_qubits):
            actions[idx] = (gate, q)
            idx += 1
    
    # Two qubit gates - nearest neighbor connectivity
    for gate in ["CNOT"]:
        for q1 in range(n_qubits - 1):
            # Apply gate to neighboring qubits in both directions
            actions[idx] = (gate, q1, q1 + 1)    # forward direction
            idx += 1
            actions[idx] = (gate, q1 + 1, q1)    # reverse direction
            idx += 1
    
    # Terminal action
    actions[idx] = ("terminal",)
    terminal_index = idx
    
    return actions, terminal_index
