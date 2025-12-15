"""
Hamiltonian loader module.
Handles loading molecular Hamiltonians from various formats.
"""

import json
import pickle
import numpy as np
from typing import Tuple, List, Dict, Optional
from pathlib import Path
from collections import defaultdict


class HamiltonianLoader:
    """Loads and manages quantum Hamiltonians."""
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialize Hamiltonian loader.
        
        Args:
            data_dir: Directory containing Hamiltonian data files
        """
        self.data_dir = Path(data_dir)
    
    def load_h2_8qubit(self) -> Tuple[float, List[str], List[float]]:
        """
        Load the H2 8-qubit Hamiltonian.
        
        Returns:
            Tuple of (identity_weight, pauli_strings, coefficients)
        """
        # Try multiple possible locations
        possible_paths = [
            self.data_dir / "hamiltonian_h2_8q.txt",
            Path("quantum_hardware_exp/data/hamiltonian_h2_8q.txt"),
            Path("data/hamiltonian_h2_8q.txt"),
        ]
        
        ham_path = None
        for path in possible_paths:
            if path.exists():
                ham_path = path
                break
        
        if ham_path is None:
            raise FileNotFoundError(
                f"H2 Hamiltonian not found. Searched: {possible_paths}"
            )
        
        return self.load_from_json(ham_path)
    
    def load_from_json(self, filepath: Path) -> Tuple[float, List[str], List[float]]:
        """
        Load Hamiltonian from JSON format.
        
        Args:
            filepath: Path to JSON file
            
        Returns:
            Tuple of (identity_weight, pauli_strings, coefficients)
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        pauli_strings = []
        coefficients = []
        identity_weight = 0.0
        
        for term in data['paulis']:
            pauli_str = term['label']
            
            # Handle complex coefficients
            if isinstance(term['coeff'], dict):
                coeff = term['coeff']['real']
            else:
                coeff = float(term['coeff'])
            
            if pauli_str == "I" * len(pauli_str):  # Identity term
                identity_weight = coeff
            else:
                pauli_strings.append(pauli_str)
                coefficients.append(coeff)
        
        
        return identity_weight, pauli_strings, coefficients
    
    def load_from_pickle(self, filepath: Path) -> Tuple[float, List[str], List[float]]:
        """
        Load Hamiltonian from pickle format.
        
        Args:
            filepath: Path to pickle file
            
        Returns:
            Tuple of (identity_weight, pauli_strings, coefficients)
        """
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        if isinstance(data, dict):
            # Extract from dictionary format
            if 'identity_weight' in data:
                identity_weight = data['identity_weight']
            else:
                identity_weight = 0.0
            
            if 'pauli_str_list' in data:
                pauli_strings = data['pauli_str_list']
                coefficients = [complex(c).real for c in data['w_list']]
            elif 'paulis' in data:
                pauli_strings = data['paulis']
                coefficients = data['coeffs']
            else:
                raise ValueError(f"Unknown pickle format in {filepath}")
        else:
            raise ValueError(f"Unexpected data type in {filepath}: {type(data)}")
        
        
        return identity_weight, pauli_strings, coefficients
    
    def validate_hamiltonian(
        self, 
        identity_weight: float, 
        pauli_strings: List[str], 
        coefficients: List[float]
    ) -> bool:
        """
        Validate Hamiltonian data.
        
        Args:
            identity_weight: Weight of identity operator
            pauli_strings: List of Pauli strings
            coefficients: List of coefficients
            
        Returns:
            True if valid, raises exception otherwise
        """
        # Check lengths match
        if len(pauli_strings) != len(coefficients):
            raise ValueError(
                f"Mismatch: {len(pauli_strings)} Paulis vs {len(coefficients)} coefficients"
            )
        
        # Check Pauli strings are valid
        valid_ops = {'I', 'X', 'Y', 'Z'}
        for pauli in pauli_strings:
            if not all(op in valid_ops for op in pauli):
                raise ValueError(f"Invalid Pauli string: {pauli}")
        
        # Check all strings have same length
        if pauli_strings:
            n_qubits = len(pauli_strings[0])
            if not all(len(p) == n_qubits for p in pauli_strings):
                raise ValueError("Inconsistent qubit counts in Pauli strings")
        
        # Check coefficients are real
        if not all(np.isreal(c) for c in coefficients):
            print("Warning: Complex coefficients found, using real parts")
        
        return True
    
    def get_hamiltonian_stats(
        self, 
        pauli_strings: List[str], 
        coefficients: List[float]
    ) -> Dict:
        """
        Get statistics about the Hamiltonian.
        
        Args:
            pauli_strings: List of Pauli strings
            coefficients: List of coefficients
            
        Returns:
            Dictionary of statistics
        """
        stats = {
            'n_terms': len(pauli_strings),
            'n_qubits': len(pauli_strings[0]) if pauli_strings else 0,
            'max_coeff': max(abs(c) for c in coefficients) if coefficients else 0,
            'min_coeff': min(abs(c) for c in coefficients) if coefficients else 0,
            'sum_abs_coeffs': sum(abs(c) for c in coefficients),
        }
        
        # Count operator types
        op_counts = {'X': 0, 'Y': 0, 'Z': 0}
        for pauli in pauli_strings:
            for op in pauli:
                if op in op_counts:
                    op_counts[op] += 1
        
        stats['operator_counts'] = op_counts
        
        # Count term weights
        weight_counts = defaultdict(int)
        for pauli in pauli_strings:
            weight = sum(1 for op in pauli if op != 'I')
            weight_counts[weight] += 1
        
        stats['weight_distribution'] = dict(weight_counts)
        
        return stats