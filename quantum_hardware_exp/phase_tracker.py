"""
Phase tracking module for Pauli operator measurements.
Handles the transformation of Pauli operators through Clifford gates.

Uses Stim sign convention (verified to match code/pauli_tracker.py):
- Y = XZ (no intrinsic i factor)
- k₀ = 0 for all Paulis, including Y
- Phase: Global exponent e in {0, 1, 2, 3} representing {+1, +i, -1, -i}
"""

import numpy as np
from typing import Tuple
from qiskit import QuantumCircuit


class PhaseTracker:
    """Tracks phases of Pauli operators through gate transformations.
    
    Uses Stim convention where Y = XZ (no intrinsic i factor).
    Initial phase k₀ = 0 for all Pauli strings including Y.
    """
    
    @staticmethod
    def pauli_to_symplectic(pauli_str: str) -> Tuple[np.ndarray, int]:
        """
        Convert Pauli string to symplectic representation with initial phase.
        
        Uses Stim convention: Y = XZ (no intrinsic i), k₀ = 0 for all Paulis.
        
        Args:
            pauli_str: String of Pauli operators (e.g., "XYZII")
            
        Returns:
            Tuple of (symplectic vector, initial phase)
        """
        n_qubits = len(pauli_str)
        vec = np.zeros(2 * n_qubits, dtype=bool)
        phase = 0  # Stim convention: k₀ = 0 for all Paulis (no intrinsic Y phase)
        
        for i, p in enumerate(pauli_str):
            if p == 'X':
                vec[i] = True
            elif p == 'Y':
                vec[i] = True
                vec[n_qubits + i] = True
                # Stim convention: Y = XZ, no intrinsic phase contribution
            elif p == 'Z':
                vec[n_qubits + i] = True
        
        return vec, phase
    
    @staticmethod
    def apply_measurement_basis_transform(
        symplectic_vec: np.ndarray, 
        phase: int,
        measurement_circuit: QuantumCircuit
    ) -> Tuple[np.ndarray, int]:
        """
        Transform Pauli operator through measurement basis changes.
        
        Args:
            symplectic_vec: Symplectic representation of Pauli operator
            phase: Current phase (0, 1, 2, or 3)
            measurement_circuit: Circuit with basis rotations
            
        Returns:
            Tuple of (transformed symplectic vector, final phase)
        """
        n_qubits = len(symplectic_vec) // 2
        vec = symplectic_vec.copy()
        current_phase = phase
        
        for instruction in measurement_circuit.data:
            # Skip non-gate operations
            if instruction.operation.name in ['measure', 'barrier', 'state_preparation']:
                continue
                
            gate = instruction.operation.name
            qubits = [q._index for q in instruction.qubits]
            
            if gate == 'h' and len(qubits) == 1:
                q = qubits[0]
                old_x = vec[q]
                old_z = vec[n_qubits + q]
                
                # Hadamard: X <-> Z, Y -> -Y
                vec[q] = old_z
                vec[n_qubits + q] = old_x
                
                # Y operator gets phase π
                if old_x and old_z:
                    current_phase = (current_phase + 2) % 4
                    
            elif gate == 's' and len(qubits) == 1:
                q = qubits[0]
                old_x = vec[q]
                old_z = vec[n_qubits + q]
                
                # S gate: X -> Y, Y -> -X, Z -> Z
                if old_x and not old_z:  # X -> Y
                    vec[n_qubits + q] = True
                    current_phase = (current_phase + 1) % 4
                elif old_x and old_z:  # Y -> -X
                    vec[n_qubits + q] = False
                    current_phase = (current_phase + 3) % 4
            
            elif gate == 'sdg' and len(qubits) == 1:
                q = qubits[0]
                old_x = vec[q]
                old_z = vec[n_qubits + q]
                
                # S†: X -> -Y, Y -> X, Z -> Z
                if old_x and not old_z:  # X -> -Y
                    vec[n_qubits + q] = True
                    current_phase = (current_phase + 3) % 4
                elif old_x and old_z:  # Y -> X
                    vec[n_qubits + q] = False
                    current_phase = (current_phase + 1) % 4
        
        return vec, current_phase
    
    @staticmethod
    def compute_pauli_expectation(
        pauli_str: str,
        measurement_circuit: QuantumCircuit,
        measurement_outcome: int
    ) -> float:
        """
        Compute Pauli expectation value with proper phase tracking.
        
        Args:
            pauli_str: Pauli operator string
            measurement_circuit: Circuit used for measurement
            measurement_outcome: Integer outcome from measurement
            
        Returns:
            Expectation value (-1 or +1)
        """
        # Convert to symplectic representation
        symplectic_vec, initial_phase = PhaseTracker.pauli_to_symplectic(pauli_str)
        
        # Transform through measurement basis
        transformed_vec, final_phase = PhaseTracker.apply_measurement_basis_transform(
            symplectic_vec, initial_phase, measurement_circuit
        )
        
        # Extract Z part after transformation
        n_qubits = len(pauli_str)
        z_part = transformed_vec[n_qubits:]
        
        # Compute parity of outcome with Z mask
        parity = 0
        for i, has_z in enumerate(z_part):
            if has_z:
                bit = (measurement_outcome >> i) & 1
                parity ^= bit
        
        # Convert parity to eigenvalue
        eigenvalue = 1 - 2 * parity
        
        # Apply phase as sign
        # For diagonal (measurable) Hermitian operators, phases must be even:
        # phase 0 → +1, phase 2 → -1
        # Odd phases (1, 3) indicate non-Hermitian operators (shouldn't happen)
        if final_phase in (1, 3):
            # Odd phase indicates non-Hermitian; return 0 to flag issue
            return 0.0
        sign = 1 if final_phase == 0 else -1  # phase 0 → +1, phase 2 → -1
        
        return sign * eigenvalue