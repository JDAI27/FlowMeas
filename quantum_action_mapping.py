#!/usr/bin/env python3
# quantum_action_mapping.py

from typing import Dict, Tuple, List


def build_action_mapping(n_qubits: int) -> Tuple[Dict[int, Tuple], int]:
    """
    Build action mapping for quantum gates.
    
    This is the single source of truth for action-to-gate mapping used by both
    GFlowNet training (GFNs.py) and energy estimation (clifford_energy_estimator.py).
    
    Args:
        n_qubits: Number of qubits in the system
        
    Returns:
        actions: Dictionary mapping action indices to (gate_name, qubit(s)) tuples
        terminal_index: Index of the terminal action
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


def get_gate_types() -> Tuple[set, set]:
    """
    Get sets of single and two qubit gate names.
    
    Returns:
        single_qubit_gates: Set of single qubit gate names
        two_qubit_gates: Set of two qubit gate names
    """
    single_qubit_gates = {"H", "S", "HS", "SH", "HSH"}
    two_qubit_gates = {"CNOT"}  # SWAP commented out
    
    return single_qubit_gates, two_qubit_gates


def action_to_gate_spec(action_tuple: Tuple) -> Tuple[str, List[int]]:
    """
    Convert action tuple to gate specification format.
    
    Args:
        action_tuple: Tuple from action mapping (gate_name, qubit(s))
        
    Returns:
        gate_name: Name of the gate
        qubits: List of qubit indices
    """
    gate_name = action_tuple[0]
    
    if gate_name == "terminal":
        return gate_name, []
    elif len(action_tuple) == 2:
        # Single qubit gate
        return gate_name, [action_tuple[1]]
    elif len(action_tuple) == 3:
        # Two qubit gate
        return gate_name, [action_tuple[1], action_tuple[2]]
    else:
        raise ValueError(f"Invalid action tuple: {action_tuple}")


def validate_action_consistency(mapping1: Dict[int, Tuple], 
                               mapping2: Dict[int, Tuple]) -> bool:
    """
    Validate that two action mappings are consistent.
    
    Args:
        mapping1: First action mapping
        mapping2: Second action mapping
        
    Returns:
        True if mappings are identical, False otherwise
    """
    if len(mapping1) != len(mapping2):
        return False
    
    for idx in mapping1:
        if idx not in mapping2:
            return False
        if mapping1[idx] != mapping2[idx]:
            return False
    
    return True
