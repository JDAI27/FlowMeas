import numpy as np
from scipy.sparse import csr_matrix, kron, identity
from scipy.sparse.linalg import eigsh, lobpcg, LinearOperator
import json
import re
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Union, Any
import logging
import hashlib
import os
from datetime import datetime
import psutil

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
    
    def __init__(self, filepath: Union[str, Path], cache_dir: Optional[str] = None):
        """
        Initialize the helper with a Hamiltonian file.
        
        Args:
            filepath: Path to the Hamiltonian file
            cache_dir: Directory for caching ground states. If None, uses 'cache/ground_states/'
        """
        self.filepath = Path(filepath)
        self.pauli_str_list: List[str] = []
        self.w_list: List[complex] = []
        self.n_qubits: Optional[int] = None
        self._ground_state_energy = None
        self._ground_state_vector = None
        self.cache_dir = Path(cache_dir) if cache_dir else Path('cache/ground_states')
        
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
        if self.n_qubits is None:
            raise ValueError("n_qubits not set - no Pauli strings loaded")
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
    
    def _get_hamiltonian_hash(self) -> str:
        """Generate a hash of the Hamiltonian for caching."""
        hasher = hashlib.sha256()
        for pauli_str, coeff in zip(self.pauli_str_list, self.w_list):
            hasher.update(f"{pauli_str}{coeff.real}{coeff.imag}".encode())
        return hasher.hexdigest()[:16]
    
    def _get_cache_path(self) -> Path:
        """Get the cache directory path for this Hamiltonian."""
        molecule = self.filepath.parent.name
        transformation = self.filepath.stem
        hamiltonian_hash = self._get_hamiltonian_hash()
        return self.cache_dir / f"{molecule}_{self.n_qubits}q_{hamiltonian_hash}_{transformation}"
    
    def _save_ground_state(self, energy: float, vector: np.ndarray, method: str = "", suffix: str = ""):
        """Save ground state to cache."""
        cache_path = self._get_cache_path()
        cache_path.mkdir(parents=True, exist_ok=True)
        
        # Create filename with optional suffix
        filename = "ground_state"
        if suffix:
            filename += f"_{suffix}"
        
        # Save to temporary file first, then move atomically
        temp_file = cache_path / f"{filename}.tmp"
        final_file = cache_path / f"{filename}.npz"
        
        np.savez_compressed(
            str(temp_file),
            energy=energy,
            vector=vector,
            method=method,
            timestamp=datetime.now().isoformat(),
            n_qubits=self.n_qubits,
            n_terms=len(self.pauli_str_list)
        )
        
        # np.savez_compressed adds .npz extension automatically
        temp_file_with_ext = cache_path / f"{filename}.tmp.npz"
        
        # Atomic move
        os.replace(str(temp_file_with_ext), str(final_file))
        logging.info(f"Saved ground state to cache: {final_file}")
    
    def _load_ground_state(self, suffix: str = "") -> Optional[Tuple[float, np.ndarray]]:
        """Load ground state from cache if available."""
        filename = "ground_state"
        if suffix:
            filename += f"_{suffix}"
        cache_file = self._get_cache_path() / f"{filename}.npz"

        if not cache_file.exists():
            return None

        try:
            data = np.load(cache_file)
            energy = float(data['energy'])
            vector = data['vector'] if 'vector' in data and len(data['vector']) > 0 else None
            method = str(data.get('method', 'unknown'))
            # DMRG caches store MPS/CI vectors, not qubit statevectors; drop them.
            if method and 'dmrg' in method.lower():
                vector = None
            logging.info(f"Loaded ground state from cache: {cache_file}")
            logging.info(f"  Method: {method}")
            logging.info(f"  Timestamp: {data.get('timestamp', 'unknown')}")
            return energy, vector
        except Exception as e:
            logging.warning(f"Failed to load cache: {e}")
            return None

    def _load_best_ground_state(self) -> Optional[Tuple[float, np.ndarray, str]]:
        """
        Load the best available ground state from cache with priority.

        Priority order:
        1. DMRG with highest bond dimension
        2. FCI (exact within basis)
        3. Regular cached ground state

        Returns:
            Tuple of (energy, vector, method) or None if no cache found
        """
        cache_path = self._get_cache_path()
        if not cache_path.exists():
            return None

        # Look for all cached ground state files
        cache_files = list(cache_path.glob("ground_state*.npz"))

        if not cache_files:
            return None

        # Categorize cache files by method
        dmrg_files = []
        fci_files = []
        other_files = []

        for cache_file in cache_files:
            filename = cache_file.stem
            if 'dmrg' in filename.lower():
                # Extract bond dimension if present
                import re
                match = re.search(r'M(\d+)', filename)
                bond_dim = int(match.group(1)) if match else 0
                dmrg_files.append((cache_file, bond_dim))
            elif 'fci' in filename.lower():
                fci_files.append(cache_file)
            else:
                other_files.append(cache_file)

        # Try DMRG files first (highest bond dimension)
        if dmrg_files:
            dmrg_files.sort(key=lambda x: x[1], reverse=True)
            best_dmrg_file = dmrg_files[0][0]
            try:
                data = np.load(best_dmrg_file)
                energy = float(data['energy'])
                vector = data['vector'] if 'vector' in data and len(data['vector']) > 0 else None
                method = str(data.get('method', 'dmrg'))
                # DMRG caches may store an MPS/CI vector, not a qubit statevector.
                # Drop it to avoid misinterpreting as a computational basis vector.
                if method and 'dmrg' in method.lower():
                    vector = None
                logging.info(f"Loaded DMRG ground state from cache: {best_dmrg_file}")
                logging.info(f"  Method: {method}, Bond dim: {dmrg_files[0][1]}")
                logging.info(f"  Energy: {energy:.10f}")
                return energy, vector, method
            except Exception as e:
                logging.warning(f"Failed to load DMRG cache: {e}")

        # Try FCI files next
        if fci_files:
            try:
                data = np.load(fci_files[0])
                energy = float(data['energy'])
                vector = data['vector'] if 'vector' in data and len(data['vector']) > 0 else None
                method = str(data.get('method', 'fci'))
                logging.info(f"Loaded FCI ground state from cache: {fci_files[0]}")
                logging.info(f"  Method: {method}")
                logging.info(f"  Energy: {energy:.10f}")
                return energy, vector, method
            except Exception as e:
                logging.warning(f"Failed to load FCI cache: {e}")

        # Try other files last
        if other_files:
            try:
                data = np.load(other_files[0])
                energy = float(data['energy'])
                vector = data['vector'] if 'vector' in data and len(data['vector']) > 0 else None
                method = str(data.get('method', 'unknown'))
                logging.info(f"Loaded ground state from cache: {other_files[0]}")
                logging.info(f"  Method: {method}")
                logging.info(f"  Energy: {energy:.10f}")
                return energy, vector, method
            except Exception as e:
                logging.warning(f"Failed to load cache: {e}")

        return None
    
    def estimate_memory_usage(self, method: str = 'eigsh') -> Dict[str, float]:
        """Estimate memory usage for ground state computation."""
        dim = 2 ** self.n_qubits
        complex_size = 16  # bytes for complex128
        
        estimates = {
            'state_vector_mb': (dim * complex_size) / (1024**2),
            'sparse_matrix_mb': (len(self.pauli_str_list) * dim * complex_size) / (1024**2),
        }
        
        if method == 'eigsh':
            # Arnoldi vectors: ~2-3 vectors of size dim
            estimates['workspace_mb'] = (3 * dim * complex_size) / (1024**2)
        elif method == 'lobpcg':
            # LOBPCG: ~3-5 vectors of size dim
            estimates['workspace_mb'] = (5 * dim * complex_size) / (1024**2)
        elif method == 'dense':
            # Dense matrix: dim x dim
            estimates['dense_matrix_mb'] = (dim * dim * complex_size) / (1024**2)
            estimates['workspace_mb'] = estimates['dense_matrix_mb']
        
        estimates['total_mb'] = sum(estimates.values())
        
        # Get available memory
        available_mb = psutil.virtual_memory().available / (1024**2)
        estimates['available_mb'] = available_mb
        estimates['feasible'] = estimates['total_mb'] < 0.8 * available_mb
        
        return estimates
    
    def compute_ground_state(self, sparse: bool = True, k: int = 1, method: str = 'auto', 
                           use_cache: bool = True, force_recompute: bool = False) -> Tuple[float, np.ndarray]:
        """
        Compute the ground state energy and state vector.
        
        Args:
            sparse: If True, use sparse matrix methods
            k: Number of lowest eigenvalues to compute
            method: 'eigsh', 'lobpcg', 'dense', or 'auto' (automatically choose based on size)
            use_cache: Whether to use cached results if available
            force_recompute: Force recomputation even if cached
            
        Returns:
            Tuple of (ground_state_energy, ground_state_vector)
        """
        # Check cache first - prioritize DMRG/FCI caches
        if use_cache and not force_recompute and k == 1:
            # First try loading best available cache (DMRG > FCI > other)
            cached_best = self._load_best_ground_state()
            if cached_best is not None:
                energy, vector, method = cached_best
                self._ground_state_energy = energy
                self._ground_state_vector = vector if vector is not None else np.array([])
                logging.info(f"Using cached ground state from {method}")
                return self._ground_state_energy, self._ground_state_vector

            # Fallback to regular cache loading
            cached = self._load_ground_state()
            if cached is not None:
                self._ground_state_energy, self._ground_state_vector = cached
                return self._ground_state_energy, self._ground_state_vector
        
        # If already computed and k=1, return cached values
        if self._ground_state_energy is not None and k == 1 and not force_recompute:
            return self._ground_state_energy, self._ground_state_vector
        
        # Auto-select method based on system size
        if method == 'auto':
            mem_est = self.estimate_memory_usage('eigsh')
            if self.n_qubits <= 12:
                method = 'dense'
            elif not mem_est['feasible']:
                logging.warning(f"System may be too large for eigsh (estimated {mem_est['total_mb']:.1f} MB needed, "
                              f"{mem_est['available_mb']:.1f} MB available). Trying LOBPCG...")
                method = 'lobpcg'
            else:
                method = 'eigsh'
        
        # Check memory before proceeding
        mem_est = self.estimate_memory_usage(method)
        if not mem_est['feasible']:
            raise MemoryError(
                f"Insufficient memory for {method} method. "
                f"Estimated: {mem_est['total_mb']:.1f} MB, Available: {mem_est['available_mb']:.1f} MB. "
                f"System has {self.n_qubits} qubits ({2**self.n_qubits:,} states). "
                f"Consider using LOBPCG method or reducing the problem size."
            )
        
        logging.info(f"Computing ground state using {method} method...")
        logging.info(f"Memory estimate: {mem_est['total_mb']:.1f} MB")
        
        try:
            if method == 'dense':
                H = self.get_hamiltonian_matrix(sparse=False)
                eigenvalues, eigenvectors = np.linalg.eigh(H)
                eigenvalues = eigenvalues[:k]
                eigenvectors = eigenvectors[:, :k]
            
            elif method == 'eigsh':
                H = self.get_hamiltonian_matrix(sparse=True)
                eigenvalues, eigenvectors = eigsh(H, k=k, which='SA')
            
            elif method == 'lobpcg':
                H = self.get_hamiltonian_matrix(sparse=True)
                # Random initial guess
                X = np.random.rand(H.shape[0], k)
                # Normalize
                for i in range(k):
                    X[:, i] /= np.linalg.norm(X[:, i])
                eigenvalues, eigenvectors = lobpcg(H, X, largest=False, maxiter=500)
            
            else:
                raise ValueError(f"Unknown method: {method}")
            
            # Ground state is the lowest eigenvalue
            self._ground_state_energy = eigenvalues[0].real
            self._ground_state_vector = eigenvectors[:, 0]
            
            # Save to cache if k=1
            if use_cache and k == 1:
                self._save_ground_state(self._ground_state_energy, self._ground_state_vector, method)
            
            return self._ground_state_energy, self._ground_state_vector
            
        except MemoryError as e:
            logging.error(f"Memory error with {method} method: {e}")
            if method != 'lobpcg':
                logging.info("Attempting with LOBPCG method...")
                return self.compute_ground_state(sparse=True, k=k, method='lobpcg', 
                                               use_cache=use_cache, force_recompute=force_recompute)
            else:
                raise
    
    @property
    def ground_state_energy(self) -> float:
        """Get the ground state energy (computing if necessary).

        Priority order for loading:
        1. Already computed in memory
        2. DMRG cache (highest bond dimension)
        3. FCI cache (exact within basis)
        4. Other cached methods
        5. Compute new (auto-select method)
        """
        if self._ground_state_energy is None:
            # Try loading from cache first
            cached_best = self._load_best_ground_state()
            if cached_best is not None:
                energy, vector, method = cached_best
                self._ground_state_energy = energy
                self._ground_state_vector = vector if vector is not None else np.array([])
                logging.info(f"Loaded ground state energy from {method} cache: {energy:.10f}")
            else:
                # Compute if no cache available
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

        energies = {}
        with open(exact_energy_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                match = re.search(r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", line)
                if not match:
                    continue
                val = float(match.group(1))

                lower = line.lower()
                if 'static coulomb repulsion' in lower:
                    energies['coulomb_repulsion'] = val
                elif 'total' in lower and 'hartree fock' in lower:
                    energies['total_hartree_fock_energy'] = val
                elif 'total' in lower:
                    energies['total_energy'] = val
                elif 'hartree fock energy' in lower:
                    energies['hartree_fock_energy'] = val
                elif 'exact energy is' in lower or 'exact ground state energy (electronic)' in lower or 'exact ground state electronic energy' in lower:
                    energies['electronic_energy'] = val

        return energies if energies else None
    
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
            expectations[p] = np.vdot(state, op @ state)
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
    
    def compute_ground_state_fci(self, n_electrons: Union[int, Tuple[int, int]], 
                                mapping: str = 'jordan_wigner',
                                use_cache: bool = True, 
                                force_recompute: bool = False,
                                **fci_kwargs) -> Tuple[float, np.ndarray]:
        """
        Compute ground state using Full Configuration Interaction (FCI) with PySCF.
        
        This method converts the qubit Hamiltonian back to fermionic representation,
        extracts one- and two-electron integrals, and runs FCI calculation.
        
        Args:
            n_electrons: Total number of electrons or tuple of (n_alpha, n_beta)
            mapping: Fermion-to-qubit mapping used ('jordan_wigner' or 'bravyi_kitaev')
            use_cache: Whether to use cached results if available
            force_recompute: Force recomputation even if cached
            **fci_kwargs: Additional arguments for FCI solver (conv_tol, max_cycle)
            
        Returns:
            Tuple of (ground_state_energy, ground_state_vector)
            
        Note:
            Requires pyscf and openfermion packages to be installed.
        """
        # Check cache first (using same cache mechanism as regular compute_ground_state)
        cache_key = f"fci_{mapping}_{n_electrons}"
        if use_cache and not force_recompute:
            cached = self._load_ground_state(suffix=cache_key)
            if cached is not None:
                return cached
        
        try:
            try:
                from .fci_solver import FCISolver
            except ImportError:
                from fci_solver import FCISolver
        except ImportError:
            raise ImportError("FCI solver requires pyscf and openfermion. "
                            "Install with: pip install pyscf openfermion")
        
        # Handle n_electrons input
        if isinstance(n_electrons, int):
            n_alpha = (n_electrons + 1) // 2
            n_beta = n_electrons // 2
            n_electrons_tuple = (n_alpha, n_beta)
        else:
            n_electrons_tuple = n_electrons
        
        # Create FCI solver
        solver = FCISolver(
            self.pauli_str_list, 
            self.w_list, 
            self.n_qubits, 
            n_electrons_tuple,
            mapping
        )
        
        # Run FCI calculation
        fci_energy, ci_vector = solver.run_fci(**fci_kwargs)
        
        # Cache the result
        if use_cache:
            self._save_ground_state(fci_energy, ci_vector, suffix=cache_key)
        
        return fci_energy, ci_vector
    
    def compute_ground_state_fci_with_analysis(self, n_electrons: Union[int, Tuple[int, int]], 
                                              mapping: str = 'jordan_wigner',
                                              **fci_kwargs) -> Dict[str, Union[float, np.ndarray]]:
        """
        Compute ground state using FCI and return detailed analysis.
        
        Args:
            n_electrons: Total number of electrons or tuple of (n_alpha, n_beta)
            mapping: Fermion-to-qubit mapping used
            **fci_kwargs: Additional arguments for FCI solver
            
        Returns:
            Dictionary with:
                - 'energy': Ground state energy
                - 'ci_vector': Ground state wavefunction
                - 'no_occupations': Natural orbital occupations
                - 'rdm1': One-particle reduced density matrix
                - 'rdm2': Two-particle reduced density matrix
        """
        try:
            try:
                from .fci_solver import pauli_strings_to_fci_energy
            except ImportError:
                from fci_solver import pauli_strings_to_fci_energy
        except ImportError:
            raise ImportError("FCI solver requires pyscf and openfermion. "
                            "Install with: pip install pyscf openfermion")
        
        return pauli_strings_to_fci_energy(
            self.pauli_str_list,
            self.w_list,
            self.n_qubits,
            n_electrons,
            mapping,
            **fci_kwargs
        )
    
    
    def compute_ground_state_post_hf(self, n_electrons: Union[int, Tuple[int, int]], 
                                     method: str = 'ccsd',
                                     mapping: str = 'jordan_wigner',
                                     use_cache: bool = True, 
                                     force_recompute: bool = False,
                                     **kwargs) -> Dict[str, Any]:
        """
        Compute ground state using post-Hartree-Fock methods.
        
        Args:
            n_electrons: Total number of electrons or tuple of (n_alpha, n_beta)
            method: Post-HF method ('ccsd', 'mp2', 'cisd', or 'all')
            mapping: Fermion-to-qubit mapping used
            use_cache: Whether to use cached results if available
            force_recompute: Force recomputation even if cached
            **kwargs: Additional arguments for the specific method
            
        Returns:
            Dictionary with method-specific results including energy and properties
        """
        # Check cache first
        cache_key = f"{method}_{mapping}_{n_electrons}"
        if use_cache and not force_recompute and method != 'all':
            cached = self._load_ground_state(suffix=cache_key)
            if cached is not None:
                energy, data = cached
                # Reconstruct the result dictionary from cached data
                return {
                    'energy': energy,
                    'method': method,
                    'cached': True,
                    'data': data
                }
        
        try:
            try:
                from .post_hf_solver import pauli_strings_to_post_hf
            except ImportError:
                from post_hf_solver import pauli_strings_to_post_hf
        except ImportError:
            raise ImportError("Post-HF solver requires pyscf and openfermion. "
                            "Install with: pip install pyscf openfermion")
        
        # Run post-HF calculation
        result = pauli_strings_to_post_hf(
            self.pauli_str_list,
            self.w_list,
            self.n_qubits,
            n_electrons,
            method,
            mapping,
            **kwargs
        )
        
        # Cache the result if requested and not comparing all methods
        if use_cache and method != 'all':
            # Store energy and essential data
            self._save_ground_state(result['energy'], result.get('rdm1', np.array([])), suffix=cache_key)
        
        return result
    
    def compute_ground_state_ccsd(self, n_electrons: Union[int, Tuple[int, int]], 
                                  mapping: str = 'jordan_wigner',
                                  use_cache: bool = True, 
                                  force_recompute: bool = False,
                                  **ccsd_kwargs) -> Tuple[float, Dict[str, Any]]:
        """
        Compute ground state using CCSD (Coupled Cluster Singles and Doubles).
        
        Args:
            n_electrons: Total number of electrons or tuple of (n_alpha, n_beta)
            mapping: Fermion-to-qubit mapping used
            use_cache: Whether to use cached results if available
            force_recompute: Force recomputation even if cached
            **ccsd_kwargs: Additional arguments for CCSD solver
            
        Returns:
            Tuple of (CCSD energy, dictionary with additional results)
        """
        result = self.compute_ground_state_post_hf(
            n_electrons, 'ccsd', mapping, use_cache, force_recompute, **ccsd_kwargs
        )
        return result['energy'], result
    
    def compute_ground_state_mp2(self, n_electrons: Union[int, Tuple[int, int]], 
                                mapping: str = 'jordan_wigner',
                                use_cache: bool = True, 
                                force_recompute: bool = False,
                                **mp2_kwargs) -> Tuple[float, Dict[str, Any]]:
        """
        Compute ground state using MP2 (Møller-Plesset 2nd order perturbation theory).
        
        Args:
            n_electrons: Total number of electrons or tuple of (n_alpha, n_beta)
            mapping: Fermion-to-qubit mapping used
            use_cache: Whether to use cached results if available
            force_recompute: Force recomputation even if cached
            **mp2_kwargs: Additional arguments for MP2 solver
            
        Returns:
            Tuple of (MP2 energy, dictionary with additional results)
        """
        result = self.compute_ground_state_post_hf(
            n_electrons, 'mp2', mapping, use_cache, force_recompute, **mp2_kwargs
        )
        return result['energy'], result
    
    def compute_ground_state_cisd(self, n_electrons: Union[int, Tuple[int, int]], 
                                 mapping: str = 'jordan_wigner',
                                 use_cache: bool = True, 
                                 force_recompute: bool = False,
                                 **cisd_kwargs) -> Tuple[float, Dict[str, Any]]:
        """
        Compute ground state using CISD (Configuration Interaction Singles and Doubles).
        
        Args:
            n_electrons: Total number of electrons or tuple of (n_alpha, n_beta)
            mapping: Fermion-to-qubit mapping used
            use_cache: Whether to use cached results if available
            force_recompute: Force recomputation even if cached
            **cisd_kwargs: Additional arguments for CISD solver
            
        Returns:
            Tuple of (CISD energy, dictionary with additional results)
        """
        result = self.compute_ground_state_post_hf(
            n_electrons, 'cisd', mapping, use_cache, force_recompute, **cisd_kwargs
        )
        return result['energy'], result
    
    def compare_all_methods(self, n_electrons: Union[int, Tuple[int, int]], 
                           mapping: str = 'jordan_wigner',
                           include_fci: bool = True) -> Dict[str, Dict[str, Any]]:
        """
        Compare all available quantum chemistry methods.
        
        Args:
            n_electrons: Total number of electrons
            mapping: Fermion-to-qubit mapping used
            include_fci: Whether to include FCI in comparison (can be expensive)
            
        Returns:
            Dictionary with results from each method
        """
        results = {}
        
        # Direct diagonalization
        try:
            e_direct, _ = self.compute_ground_state(method='auto', use_cache=True)
            results['Direct'] = {
                'energy': e_direct,
                'method': 'Direct diagonalization',
                'description': 'Exact diagonalization of qubit Hamiltonian'
            }
        except Exception as e:
            logger.warning(f"Direct diagonalization failed: {e}")
        
        # Post-HF methods
        post_hf_results = self.compute_ground_state_post_hf(
            n_electrons, 'all', mapping, use_cache=False
        )
        results.update(post_hf_results)
        
        # FCI if requested and feasible
        if include_fci and self.n_qubits <= 16:  # Practical limit
            try:
                e_fci, _ = self.compute_ground_state_fci(
                    n_electrons, mapping, use_cache=True
                )
                results['FCI'] = {
                    'energy': e_fci,
                    'method': 'Full Configuration Interaction',
                    'description': 'Exact within basis set'
                }
            except Exception as e:
                logger.warning(f"FCI failed: {e}")
        
        return results
    
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
                
