import numpy as np
from scipy.sparse import csr_matrix, kron, identity
from scipy.sparse.linalg import eigsh
import json
import re
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Union
import logging

class PauliHamiltonianHelper:
    """Helper class for parsing and analyzing Pauli Hamiltonian files.
    
    Supports multiple file formats:
    1. Two-line format: Pauli string on one line, coefficient on the next
    2. CSV format: coefficient,Pauli_string on each line
    3. JSON format: {"paulis": [{"label": "...", "coeff": {"real": ..., "imag": ...}}]}
    """
    
    # Pauli matrices
    I = np.array([[1, 0], [0, 1]], dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    
    def __init__(self, filepath: Union[str, Path]):
        """
        Initialize the helper with a Hamiltonian file.
        
        Args:
            filepath: Path to the Hamiltonian file
        """
        self.filepath = Path(filepath)
        self.pauli_str_list = []
        self.w_list = []
        self.n_qubits = None
        self._ground_state_energy = None
        self._ground_state_vector = None
        
        # Parse the file
        self._parse_hamiltonian_file()
        
    def _detect_format(self, content: str, lines: List[str]) -> str:
        """Detect the format of the Hamiltonian file."""
        # Check if it's JSON
        if content.strip().startswith('{') and 'paulis' in content:
            return 'json'
        
        # Check if it's CSV format (coefficient,Pauli_string)
        if lines and ',' in lines[0]:
            parts = lines[0].split(',')
            if len(parts) == 2:
                try:
                    float(parts[0])
                    if all(c in 'IXYZ' for c in parts[1].strip()):
                        return 'csv'
                except ValueError:
                    pass
        
        # Check if it's two-line format
        if len(lines) >= 2:
            # First line should be Pauli string
            if all(c in 'IXYZ' for c in lines[0]):
                # Second line should be coefficient
                try:
                    coeff_str = lines[1]
                    if coeff_str.startswith('(') and coeff_str.endswith(')'):
                        complex(coeff_str[1:-1])
                        return 'two_line'
                    else:
                        complex(coeff_str)
                        return 'two_line'
                except ValueError:
                    pass
        
        raise ValueError(f"Could not detect format of file {self.filepath}")
        
    def _parse_hamiltonian_file(self):
        """Parse the Hamiltonian file to extract Pauli strings and coefficients."""
        with open(self.filepath, 'r') as f:
            content = f.read()
        
        lines = [line.strip() for line in content.strip().split('\n') if line.strip()]
        
        if not lines:
            raise ValueError(f"File {self.filepath} appears to be empty")
        
        # Detect format
        format_type = self._detect_format(content, lines)
        
        if format_type == 'json':
            self._parse_json_format(content)
        elif format_type == 'csv':
            self._parse_csv_format(lines)
        elif format_type == 'two_line':
            self._parse_two_line_format(lines)
        
        logging.info(f"Parsed {len(self.pauli_str_list)} Pauli terms for {self.n_qubits} qubits (format: {format_type})")
        
        # Check if parsing was successful
        if len(self.pauli_str_list) == 0:
            raise ValueError(f"No valid Pauli terms found in {self.filepath}")
    
    def _parse_json_format(self, content: str):
        """Parse JSON format Hamiltonian."""
        data = json.loads(content)
        
        for term in data['paulis']:
            pauli_str = term['label']
            coeff = complex(term['coeff']['real'], term['coeff']['imag'])
            
            self.pauli_str_list.append(pauli_str)
            self.w_list.append(coeff)
            
            if self.n_qubits is None:
                self.n_qubits = len(pauli_str)
    
    def _parse_csv_format(self, lines: List[str]):
        """Parse CSV format: coefficient,Pauli_string."""
        for line in lines:
            if ',' not in line:
                continue
                
            parts = line.split(',', 1)  # Split only on first comma
            if len(parts) != 2:
                continue
                
            try:
                coeff = float(parts[0])
                pauli_str = parts[1].strip()
                
                # Validate Pauli string
                if all(c in 'IXYZ' for c in pauli_str) and len(pauli_str) > 0:
                    self.pauli_str_list.append(pauli_str)
                    self.w_list.append(complex(coeff, 0))
                    
                    if self.n_qubits is None:
                        self.n_qubits = len(pauli_str)
                        
            except ValueError:
                logging.warning(f"Could not parse line: {line}")
    
    def _parse_two_line_format(self, lines: List[str]):
        """Parse two-line format: Pauli string on one line, coefficient on next."""
        idx = 0
        while idx < len(lines) - 1:
            pauli_str = lines[idx]
            
            # Check if this is a valid Pauli string
            if all(c in 'IXYZ' for c in pauli_str) and len(pauli_str) > 0:
                coeff_str = lines[idx + 1]
                
                try:
                    # Remove parentheses if present
                    if coeff_str.startswith('(') and coeff_str.endswith(')'):
                        coeff_str = coeff_str[1:-1]
                    
                    # Parse complex number
                    coefficient = complex(coeff_str)
                    
                    self.pauli_str_list.append(pauli_str)
                    self.w_list.append(coefficient)
                    
                    if self.n_qubits is None:
                        self.n_qubits = len(pauli_str)
                    
                    idx += 2
                except ValueError as e:
                    logging.warning(f"Could not parse coefficient '{lines[idx+1]}': {e}")
                    idx += 1
            else:
                idx += 1
    
    def _pauli_string_to_matrix(self, pauli_str: str) -> np.ndarray:
        """
        Convert a Pauli string to its matrix representation.
        
        Args:
            pauli_str: String of Pauli operators (e.g., 'XIYZ')
            
        Returns:
            Matrix representation of the Pauli string
        """
        pauli_map = {'I': self.I, 'X': self.X, 'Y': self.Y, 'Z': self.Z}
        
        # Start with the first qubit
        matrix = pauli_map[pauli_str[0]]
        
        # Kronecker product for remaining qubits
        for char in pauli_str[1:]:
            matrix = np.kron(matrix, pauli_map[char])
            
        return matrix
    
    def _pauli_string_to_sparse_matrix(self, pauli_str: str) -> csr_matrix:
        """
        Convert a Pauli string to its sparse matrix representation.
        More efficient for large systems.
        
        Args:
            pauli_str: String of Pauli operators
            
        Returns:
            Sparse matrix representation
        """
        pauli_map = {
            'I': csr_matrix(self.I),
            'X': csr_matrix(self.X),
            'Y': csr_matrix(self.Y),
            'Z': csr_matrix(self.Z)
        }
        
        # Start with identity of size 1
        matrix = csr_matrix(np.array([[1.0 + 0j]]))
        
        # Kronecker product for each qubit
        for char in pauli_str:
            matrix = kron(matrix, pauli_map[char])
            
        return matrix
    
    def get_hamiltonian_matrix(self, sparse: bool = True) -> Union[np.ndarray, csr_matrix]:
        """
        Construct the full Hamiltonian matrix.
        
        Args:
            sparse: If True, return sparse matrix (recommended for large systems)
            
        Returns:
            Hamiltonian matrix
        """
        dim = 2 ** self.n_qubits
        
        if sparse:
            H = csr_matrix((dim, dim), dtype=complex)
            for pauli_str, coeff in zip(self.pauli_str_list, self.w_list):
                H += coeff * self._pauli_string_to_sparse_matrix(pauli_str)
        else:
            H = np.zeros((dim, dim), dtype=complex)
            for pauli_str, coeff in zip(self.pauli_str_list, self.w_list):
                H += coeff * self._pauli_string_to_matrix(pauli_str)
        
        return H
    
    def compute_ground_state(self, sparse: bool = True, k: int = 1) -> Tuple[float, np.ndarray]:
        """
        Compute the ground state energy and state vector.
        
        Args:
            sparse: If True, use sparse matrix methods
            k: Number of lowest eigenvalues to compute
            
        Returns:
            Tuple of (ground_state_energy, ground_state_vector)
        """
        if self._ground_state_energy is not None and k == 1:
            return self._ground_state_energy, self._ground_state_vector
        
        H = self.get_hamiltonian_matrix(sparse=sparse)
        
        if sparse:
            # Use sparse eigenvalue solver for lowest k eigenvalues
            eigenvalues, eigenvectors = eigsh(H, k=k, which='SA')
        else:
            # Use dense eigenvalue solver
            eigenvalues, eigenvectors = np.linalg.eigh(H)
            eigenvalues = eigenvalues[:k]
            eigenvectors = eigenvectors[:, :k]
        
        # Ground state is the lowest eigenvalue
        self._ground_state_energy = eigenvalues[0].real
        self._ground_state_vector = eigenvectors[:, 0]
        
        return self._ground_state_energy, self._ground_state_vector
    
    @property
    def ground_state_energy(self) -> float:
        """Get the ground state energy (computing if necessary)."""
        if self._ground_state_energy is None:
            self.compute_ground_state()
        return self._ground_state_energy
    
    @property
    def ground_state_vector(self) -> np.ndarray:
        """Get the ground state vector (computing if necessary)."""
        if self._ground_state_vector is None:
            self.compute_ground_state()
        return self._ground_state_vector
    
    def get_expectation_value(self, state: np.ndarray, pauli_str: str) -> complex:
        """
        Compute expectation value of a Pauli string for a given state.
        
        Args:
            state: Quantum state vector
            pauli_str: Pauli string operator
            
        Returns:
            Expectation value <state|pauli_str|state>
        """
        if len(pauli_str) != self.n_qubits:
            raise ValueError(f"Pauli string length {len(pauli_str)} doesn't match n_qubits {self.n_qubits}")
        
        op_matrix = self._pauli_string_to_matrix(pauli_str)
        return np.vdot(state, op_matrix @ state)
    
    def verify_ground_state_energy(self) -> float:
        """
        Verify ground state energy by computing <ψ|H|ψ> using Pauli decomposition.
        
        Returns:
            Ground state energy computed from expectation values
        """
        gs_vector = self.ground_state_vector
        energy = 0.0
        
        for pauli_str, coeff in zip(self.pauli_str_list, self.w_list):
            exp_val = self.get_expectation_value(gs_vector, pauli_str)
            energy += coeff * exp_val
            
        return energy.real
    
    def get_exact_energy_from_file(self) -> Optional[Dict[str, float]]:
        """
        Try to read the exact energy from ExactEnergy.txt file if it exists.
        
        Returns:
            Dictionary with energy values or None if file doesn't exist
        """
        exact_energy_path = self.filepath.parent / 'ExactEnergy.txt'
        if not exact_energy_path.exists():
            return None
    
    def get_hartree_fock_bitstring(self) -> Optional[Dict[str, str]]:
        """
        Read Hartree-Fock bitstrings from hartree_fock_bitstrings.txt file.
        
        Returns:
            Dictionary mapping transformation type to bitstring, or None if file doesn't exist
        """
        hf_path = self.filepath.parent / 'hartree_fock_bitstrings.txt'
        if not hf_path.exists():
            return None
        
        bitstrings = {}
        current_transform = None
        
        with open(hf_path, 'r') as f:
            lines = [line.strip() for line in f.readlines()]
        
        i = 0
        while i < len(lines):
            line = lines[i]
            
            # Skip empty lines and header
            if not line or line.startswith('Hartree-Fock') or line == '===':
                i += 1
                continue
            
            # Check if this is a transformation type (jw, parity, bk)
            if line in ['jw', 'parity', 'bk']:
                current_transform = line
                # Look for bitstring in next non-empty, non-separator line
                j = i + 1
                while j < len(lines):
                    if lines[j] and lines[j] != '===':
                        # This should be the bitstring
                        if all(c in '01' for c in lines[j]):
                            bitstrings[current_transform] = lines[j]
                        break
                    j += 1
            
            i += 1
        
        return bitstrings if bitstrings else None
            
    def compute_expectations(
        self,
        state: Optional[np.ndarray] = None,
        pauli_strings: Optional[List[str]] = None,
        use_sparse: bool = False
    ) -> Dict[str, complex]:
        """
        Return a dictionary {pauli_string: ⟨state|P|state⟩}.

        Args
        ----
        state
            State vector.  If None, the stored ground-state vector is used.
        pauli_strings
            Iterable of Pauli strings.  If None, all terms in the Hamiltonian
            are evaluated.
        use_sparse
            If True, construct operators with the sparse helper to save memory
            on larger systems (slower for very small ones).

        Returns
        -------
        Dict[str, complex]
            Expectation values keyed by Pauli label.
        """
        if state is None:
            state = self.ground_state_vector
        if pauli_strings is None:
            pauli_strings = self.pauli_str_list

        expectations = {}
        for p in pauli_strings:
            if use_sparse:
                op = self._pauli_string_to_sparse_matrix(p)
            else:
                op = self._pauli_string_to_matrix(p)
            expectations[p] = np.vdot(state, op @ state) # round to 9 decimal places and convert to float
            expectations[p] = np.round(expectations[p].real, 9)
        return expectations
    
    def summary(self) -> Dict:
        """
        Get a summary of the Hamiltonian properties.
        
        Returns:
            Dictionary with summary information
        """
        exact_energies = self.get_exact_energy_from_file()
        computed_energy = self.ground_state_energy
        
        # Get the total energy (including nuclear repulsion) if available
        exact_total_energy = None
        if exact_energies:
            exact_total_energy = exact_energies.get('total_energy', 
                                                   exact_energies.get('electronic_energy'))
        
        # Get HF bitstring for current transformation
        hf_bitstrings = self.get_hartree_fock_bitstring()
        hf_bitstring = None
        if hf_bitstrings:
            # Try to match transformation name
            transform = self.filepath.stem
            hf_bitstring = hf_bitstrings.get(transform)
        
        summary = {
            'molecule': self.filepath.parent.name,
            'transformation': self.filepath.stem,
            'n_qubits': self.n_qubits,
            'n_terms': len(self.pauli_str_list),
            'ground_state_energy': computed_energy,
            'exact_energies': exact_energies,
            'exact_total_energy': exact_total_energy,
            'energy_difference': abs(computed_energy - exact_total_energy) if exact_total_energy else None,
            'hf_bitstring': hf_bitstring,
            'largest_coefficient': max(abs(w) for w in self.w_list),
            'smallest_coefficient': min(abs(w) for w in self.w_list if abs(w) > 1e-10)
        }
        
        return summary
    
    def save_to_format(self, filepath: str, format: str = 'two_line'):
        """
        Save the Hamiltonian to a specific format.
        
        Args:
            filepath: Output file path
            format: 'two_line', 'csv', or 'json'
        """
        with open(filepath, 'w') as f:
            if format == 'two_line':
                for pauli_str, coeff in zip(self.pauli_str_list, self.w_list):
                    f.write(f"{pauli_str}\n")
                    f.write(f"({coeff.real}{coeff.imag:+}j)\n")
                    
            elif format == 'csv':
                for pauli_str, coeff in zip(self.pauli_str_list, self.w_list):
                    f.write(f"{coeff.real},{pauli_str}\n")
                    
            elif format == 'json':
                data = {
                    'paulis': [
                        {
                            'label': pauli_str,
                            'coeff': {'real': coeff.real, 'imag': coeff.imag}
                        }
                        for pauli_str, coeff in zip(self.pauli_str_list, self.w_list)
                    ]
                }
                json.dump(data, f, indent=2)
            else:
                raise ValueError(f"Unknown format: {format}")
    
    def __repr__(self) -> str:
        return f"PauliHamiltonianHelper(molecule={self.filepath.parent.name}, n_qubits={self.n_qubits}, n_terms={len(self.pauli_str_list)})"


# Example usage
if __name__ == "__main__":
    # Test with different file formats
    test_files = "../Hamiltonians/H2_STO3g_4qubits/jw.txt",  # Two-line format
    
    for filepath in test_files:
        if Path(filepath).exists():
            logging.info(f"\nTesting {filepath}:")
            try:
                helper = PauliHamiltonianHelper(filepath)
                logging.info(f"  Success! Found {len(helper.pauli_str_list)} terms")
                logging.info(f"  Ground state energy: {helper.ground_state_energy:.10f}")
                logging.info(f"  Summary: {helper.summary()}")
                # print the expectation values for all Pauli strings
                expectations = helper.compute_expectations()
                logging.info(f"  Expectation values: {expectations}")
            except Exception as e:
                logging.info(f"  Error: {e}")
                
