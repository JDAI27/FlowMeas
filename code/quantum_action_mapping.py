#!/usr/bin/env python3
# quantum_action_mapping.py

from typing import Dict, Tuple, List, Union


def build_action_mapping(n_qubits: int) -> Tuple[Dict[int, Union[Tuple[str], Tuple[str, int], Tuple[str, int, int]]], int]:
    """Build action mapping: single source of truth for action-to-gate mapping."""
    actions = {}
    idx = 0
    
    for gate in ["H", "S", "HS", "SH", "HSH"]:
        for q in range(n_qubits):
            actions[idx] = (gate, q)
            idx += 1
    
    for gate in ["CNOT"]:
        for q1 in range(n_qubits - 1):
            actions[idx] = (gate, q1, q1 + 1)
            idx += 1
            actions[idx] = (gate, q1 + 1, q1)
            idx += 1
    
    actions[idx] = ("terminal",)
    terminal_index = idx
    
    return actions, terminal_index


def get_gate_types() -> Tuple[set, set]:
    """Get sets of single and two qubit gate names."""
    return {"H", "S", "HS", "SH", "HSH"}, {"CNOT"}


def action_to_gate_spec(action_tuple: Tuple) -> Tuple[str, List[int]]:
    """Convert action tuple to (gate_name, qubits) format."""
    gate_name = action_tuple[0]
    
    if gate_name == "terminal":
        return gate_name, []
    elif len(action_tuple) == 2:
        return gate_name, [action_tuple[1]]
    elif len(action_tuple) == 3:
        return gate_name, [action_tuple[1], action_tuple[2]]
    else:
        raise ValueError(f"Invalid action tuple: {action_tuple}")


def validate_action_consistency(mapping1: Dict[int, Tuple], 
                               mapping2: Dict[int, Tuple]) -> bool:
    """Validate that two action mappings are identical."""
    if len(mapping1) != len(mapping2):
        return False
    
    for idx in mapping1:
        if idx not in mapping2:
            return False
        if mapping1[idx] != mapping2[idx]:
            return False
    
    return True
