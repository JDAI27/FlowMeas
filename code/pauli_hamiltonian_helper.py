import numpy as np
from scipy.sparse import csr_matrix, kron, identity
from scipy.sparse.linalg import eigsh, lobpcg, LinearOperator
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Union, Any
import logging
import hashlib
import os
from datetime import datetime


@dataclass(frozen=True)
class CachedMPSGroundState:
    """Result of a non-computing DMRG-MPS cache read.

    Returned by ``PauliHamiltonianHelper.load_cached_dmrg_ground_state_mps``.
    All fields come straight from a cached ``ground_state_dmrg_*.npz``
    payload (validated by ``_load_best_ground_state``); the loader does not
    call ``compute_ground_state`` on cache miss. Required so MPS-native
    ``EnergyEstimator`` startup fails fast when the cache is missing rather
    than triggering an unbounded DMRG at constructor time.

    ``mps`` is a list of complex128 numpy arrays of shape
    ``(chi_left, 2, chi_right)`` on CPU. ``method`` records which cache
    file the payload came from (``"dmrg"``, etc.).

    Optional metadata (``bond_dim``, ``converged``, ``final_trunc_err``,
    ``n_sweeps``) is populated when available on the cache file and is
    ``None`` otherwise. Useful for logging and for downstream sanity checks
    (e.g. warning when training on an unconverged cache).
    """

    energy: float
    mps: List[np.ndarray]
    method: str
    bond_dim: Optional[int] = None
    converged: Optional[bool] = None
    final_trunc_err: Optional[float] = None
    n_sweeps: Optional[int] = None

    @property
    def max_bond_dim_from_mps(self) -> int:
        """Compute the effective max bond dim by inspecting the loaded
        tensors (independent of the cache-file metadata field)."""
        m = 0
        for t in self.mps:
            m = max(m, int(t.shape[0]), int(t.shape[2]))
        return m
try:
    from .full_state_guard import (
        EXACT_FULL_STATE_QUBIT_LIMIT,
        estimate_full_state_memory,
        guard_exact_full_state_request,
        is_large_full_state_system,
    )
except ImportError:
    from full_state_guard import (
        EXACT_FULL_STATE_QUBIT_LIMIT,
        estimate_full_state_memory,
        guard_exact_full_state_request,
        is_large_full_state_system,
    )

logger = logging.getLogger(__name__)


PREPROCESSING_EXACT_SOLVER_QUBIT_LIMIT = EXACT_FULL_STATE_QUBIT_LIMIT
"""Systems at or above this qubit count must not enter O(2^n) code paths
(full-state vector, dense/sparse Hamiltonian matrix, exact diagonalization)
during preprocessing or normal loading.

Aligned with ``main.EXACT_FULL_STATE_QUBIT_LIMIT`` (= 26) so there is a
single threshold for the full-state boundary.  At 26 qubits the state
vector is 2^26 ≈ 64 M entries (1 GiB complex128); above that, memory
and runtime become impractical for exact diag.
"""


class PauliHamiltonianHelper:
    """Helper class for parsing and analyzing Pauli Hamiltonian files.

    Supports multiple file formats:
    1. Two-line format: Pauli string on one line, coefficient on the next
    2. CSV format: coefficient,Pauli_string on each line
    3. JSON format: {"paulis": [{"label": "...", "coeff": {"real":..., "imag":...}}]}

    Method map (— this is a large helper; navigate by cluster rather than
    line number. Sectioning is intentionally doc-only: a full module split is
    deferred because of the many external references to private methods):
      * File parsing & Pauli->matrix construction: ``_detect_format``,
        ``_parse_{hamiltonian_file,json_format,csv_format,two_line_format}``,
        ``_pauli_string_to_{matrix,sparse_matrix}``, ``get_hamiltonian_matrix``.
      * Caching keys & large-system safety guard: ``_get_hamiltonian_hash``,
        ``_get_cache_path``, ``_guard_exact_full_state``, ``_resolve_auto_method``.
        NOTE: ``_guard_exact_full_state`` is the 26-qubit O(2^n) tripwire — do
        not weaken it; ``_resolve_auto_method`` picks dense/eigsh/DMRG using
        both system size and the medium-system memory estimate.
      * Ground-state cache I/O: ``_save_ground_state``/``_save_ground_state_dmrg``,
        ``_cache_payload_*``, ``_load_cached_vector``,
        ``_in_memory_state_matches_method``, ``_npz_path_has_valid_mps``,
        ``_load_mps_from_npz``, ``_load_ground_state``,
        ``_methods_are_compatible_for_exact_cache``, ``_load_best_ground_state``.
      * Memory estimation: ``estimate_memory_usage``, ``_format_available_memory``.
      * Ground-state computation: ``compute_ground_state``.
      * Public ground-state properties: ``ground_state_{energy,vector,mps}``.
      * Non-computing cache access: ``load_cached_dmrg_ground_state_mps``.
        Private MPS conversion helper: ``_mps_to_dense_vector_numpy``.
      * Expectation values & verification: ``get_expectation_value``,
        ``verify_ground_state_energy``, ``_has_dmrg_cache``,
        ``get_exact_energy_from_file``, ``get_hartree_fock_bitstring``.
      * Reporting & summaries: ``compute_expectations``, ``summary``,
        ``safe_summary``, ``save_to_format``.
      * Classical-chemistry references (PySCF): ``compute_ground_state_{fci,
        fci_with_analysis,post_hf,ccsd,mp2,cisd}``, ``compare_all_methods``.
    """
    
    # Pauli matrices
    I = np.array([[1, 0], [0, 1]], dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    
    def __init__(self, filepath: Union[str, Path], cache_dir: Optional[str] = None,
                 allow_exact_solver: Optional[bool] = None):
        """
        Initialize the helper with a Hamiltonian file.

        Args:
            filepath: Path to the Hamiltonian file
            cache_dir: Directory for caching ground states. If None, uses 'cache/ground_states/'
            allow_exact_solver: Explicit opt-in/out for O(2^n) operations.
                If None (default), automatically refused for systems with
                n_qubits >= PREPROCESSING_EXACT_SOLVER_QUBIT_LIMIT.
                Set to True to override (e.g. for testing on medium systems).
        """
        self.filepath = Path(filepath)
        self.pauli_str_list: List[str] = []
        self.w_list: List[complex] = []
        self.n_qubits: Optional[int] = None
        self._ground_state_energy = None
        self._ground_state_vector = None
        # MPS representation (DMRG output). list[np.ndarray] of shape
        # (chi_L, d, chi_R) per site, complex128 on CPU. Set when DMRG runs
        # or when a cached DMRG entry is loaded. Surfaced via
        # ``ground_state_mps`` property; the MPS-native path uses this for
        # MPS-native expectation values without materializing a dense state.
        self._ground_state_mps: Optional[List[np.ndarray]] = None
        # Absolute path of the DMRG cache file that the last ``_load_best_ground_state``
        # call committed to. ``load_cached_dmrg_ground_state_mps`` reads
        # bond_dim / converged / final_trunc_err / n_sweeps metadata from
        # this exact file so the surfaced metadata always agrees with the
        # loaded MPS. ``None`` until a DMRG cache load succeeds.
        self._ground_state_cache_file: Optional[Path] = None
        # Track the method that produced the in-memory (energy, vector, mps) triple.
        # Used by the "already computed" shortcut in ``compute_ground_state`` (so a
        # DMRG request ignores a previous exact result and vice versa) and by
        # ``verify_ground_state_energy`` / ``compute_expectations`` to pick
        # MPS-native vs dense verification from the ACTIVE source.
        # Values: None, 'dmrg', 'dense', 'eigsh', 'lobpcg', 'fci', or a cache tag.
        self._ground_state_method: Optional[str] = None
        self._allow_exact_solver = allow_exact_solver
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
        self._guard_exact_full_state(
            "PauliHamiltonianHelper.get_hamiltonian_matrix",
            method="sparse_matrix" if sparse else "dense",
        )
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

    def _guard_exact_full_state(self, context: str, method: str = "state_vector") -> None:
        if self._allow_exact_solver is True:
            return
        guard_exact_full_state_request(
            context=context,
            n_qubits=self.n_qubits,
            n_terms=len(self.pauli_str_list),
            method=method,
            filepath=self.filepath,
        )

    # ``method='auto'`` must NOT unconditionally pick DMRG: at chi=16 a degenerate
    # Z2 sector can return a non-ground energy, so auto would let small Hamiltonians
    # silently cache a wrong reference. Policy:
    #   - Tiny (``n_qubits <= 12``): ``'dense'`` — fast, exact, parity check.
    #   - Medium (``12 < n_qubits < 26``): ``'eigsh'`` — sparse Lanczos, exact.
    #   - Large (``n_qubits >= 26``): ``'dmrg'`` — exact full-state is guard-blocked.
    # Callers can still bypass this by passing an explicit method.
    EXACT_AUTO_DENSE_MAX_QUBITS = 12

    def _resolve_auto_method(self) -> str:
        """Resolve ``method='auto'`` to a concrete solver based on system size.

        Returns one of ``'dense'``, ``'eigsh'``, or ``'dmrg'``. The policy:

        * ``n_qubits <= EXACT_AUTO_DENSE_MAX_QUBITS`` (12): ``'dense'`` —
          fast + exact, always memory-feasible (4096x4096 complex128 ≈ 256 MB).
        * ``EXACT_AUTO_DENSE_MAX_QUBITS < n_qubits < EXACT_FULL_STATE_QUBIT_LIMIT``
          (12 < n < 26): ``'eigsh'`` if the memory estimator says it fits,
          else ``'dmrg'``. This guards against the case where eigsh
          construction at 16q needed >100 GB of host RAM.
        * ``n_qubits >= EXACT_FULL_STATE_QUBIT_LIMIT`` (26): ``'dmrg'`` —
          exact full-state is guard-blocked anyway.
        """
        if self.n_qubits is None:
            return 'dmrg'
        if is_large_full_state_system(self.n_qubits):
            return 'dmrg'
        if self.n_qubits <= self.EXACT_AUTO_DENSE_MAX_QUBITS:
            return 'dense'
        # Medium range: prefer eigsh only when the estimator says it fits.
        # estimate_memory_usage returns ``feasible=None`` when host memory
        # is unknown — treat that as "go ahead and try" rather than
        # silently downgrading to DMRG: exact methods should win whenever
        # they are feasible.
        try:
            mem_est = self.estimate_memory_usage('eigsh')
        except Exception:
            mem_est = {'feasible': None}
        if mem_est.get('feasible') is False:
            logging.info(
                f"auto: eigsh estimated infeasible for n={self.n_qubits} "
                f"(needs {mem_est.get('total_mb')} MB, have "
                f"{mem_est.get('available_mb')} MB). Falling back to DMRG."
            )
            return 'dmrg'
        return 'eigsh'
    
    def _save_ground_state(self, energy: float, vector: np.ndarray, method: str = "", suffix: str = ""):
        """Save ground state to cache (eigsh / dense / FCI / post-HF path).

        For DMRG ground states the canonical save is:meth:`_save_ground_state_dmrg`
        — that path stores the **MPS as the primary state representation**, with
        the dense vector being optional. This method here is the legacy
        statevector-centric path used by methods that produce a dense vector
        natively (eigsh, exact diag, FCI ci-vector, etc.).
        """
        if (
            vector is not None
            and len(vector) > 1
            and self.n_qubits is not None
            and is_large_full_state_system(self.n_qubits)
            and self._cache_payload_kind(method, suffix=suffix)
            in {"qubit_statevector", "exact_vector"}
        ):
            self._guard_exact_full_state(
                "PauliHamiltonianHelper._save_ground_state",
                method=method or "state_vector",
            )

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
            n_terms=len(self.pauli_str_list),
        )

        # np.savez_compressed adds.npz extension automatically
        temp_file_with_ext = cache_path / f"{filename}.tmp.npz"

        # Atomic move
        os.replace(str(temp_file_with_ext), str(final_file))
        logging.info(f"Saved ground state to cache: {final_file}")

    def _save_ground_state_dmrg(
        self,
        energy: float,
        mps: List[np.ndarray],
        *,
        method: str,
        suffix: Optional[str] = None,
        converged: bool = True,
        n_sweeps: int = 0,
        final_trunc_err: float = 0.0,
    ):
        """Save a DMRG ground state to cache.

        Cache contract for DMRG ground states:
          - ``energy``: scalar (real)
          - ``mps``: REQUIRED. List of L numpy arrays, complex128, each shape
            (chi_L, d, chi_R). Stored in the npz under keys ``mps_n_tensors``,
            ``mps_0``, ``mps_1``,... This is the canonical state.
          - ``vector``: always stored as an empty array for DMRG. Callers
            that need a dense state vector contract the cached MPS on the
            fly via:prop:`ground_state_vector` (n<26 only).
          - DMRG metadata: ``converged``, ``n_sweeps``, ``final_trunc_err``,
            ``method`` (e.g. ``"dmrg_M64"``).

        The MPS is what (MPS-native ``EnergyEstimator``) reads at
        training time to compute expectation values without ever
        materialising the 2^n dense state.
        """
        if mps is None or len(mps) == 0:
            raise ValueError("_save_ground_state_dmrg requires a non-empty MPS")

        # Default suffix matches the method tag so concurrent runs at
        # different bond_dim land in distinct files (e.g. ground_state_dmrg_M64.npz).
        if suffix is None:
            suffix = method

        # Dense vector is always empty for the DMRG cache — the MPS is the
        # canonical state, the dense form is contracted on demand for n<26.
        # Dropped the dead
        # ``dense_vector`` kwarg that was never passed.
        stored_vector = np.array([], dtype=complex)

        cache_path = self._get_cache_path()
        cache_path.mkdir(parents=True, exist_ok=True)

        filename = "ground_state"
        if suffix:
            filename += f"_{suffix}"
        temp_file = cache_path / f"{filename}.tmp"
        final_file = cache_path / f"{filename}.npz"

        # Cache collision safety: if a cache file already exists at this path AND has
        # strictly better convergence metrics, REFUSE to overwrite. Two DMRG runs
        # with different bond_dim / max_sweeps / energy_tol that end at the same
        # ``final_chi`` previously overwrote each other, sometimes downgrading a
        # converged cache.
        if final_file.exists():
            # The "has MPS" check must round-trip through the full payload
            # loader, not just look for the mps_n_tensors key — a truncated
            # cache claiming mps_n_tensors=8 while missing mps_1..7 would
            # otherwise pass and block legitimate overwrites.
            ex_has_mps = self._npz_path_has_valid_mps(final_file)
            ex_conv: bool = False
            ex_trunc: float = float("inf")
            try:
                with np.load(final_file) as existing:
                    ex_conv = bool(existing.get("converged", False)) \
                        if "converged" in existing.files else True
                    ex_trunc = float(existing.get("final_trunc_err", float("inf"))) \
                        if "final_trunc_err" in existing.files else float("inf")
            except Exception:
                ex_conv, ex_trunc = False, float("inf")
            new_conv = bool(converged)
            new_trunc = float(final_trunc_err)
            # A payload-less existing file (corrupt) is never honored as "better":
            # its metadata can claim converged=True with trunc=0 while having no MPS,
            # so it cannot satisfy the cache contract and must be overwritten. Without
            # this, such a file blocks every legitimate save and the loader recomputes
            # forever.
            existing_strictly_better = ex_has_mps and (
                # Strictly-better rules (MPS-bearing caches only):
                #   1. converged wins over non-converged.
                #   2. Same convergence status: lower trunc_err wins.
                (ex_conv and not new_conv) or
                (ex_conv == new_conv and ex_trunc + 1e-15 < new_trunc)
            )
            if existing_strictly_better:
                logging.info(
                    f"Existing DMRG cache at {final_file} has stricter metrics "
                    f"(converged={ex_conv}, trunc={ex_trunc:.3e}) than the new "
                    f"result (converged={new_conv}, trunc={new_trunc:.3e}); "
                    f"keeping the existing cache."
                )
                return

        save_kwargs: Dict[str, Any] = dict(
            energy=energy,
            vector=stored_vector,
            method=method,
            timestamp=datetime.now().isoformat(),
            n_qubits=self.n_qubits,
            n_terms=len(self.pauli_str_list),
            converged=bool(converged),
            n_sweeps=int(n_sweeps),
            final_trunc_err=float(final_trunc_err),
            mps_n_tensors=len(mps),
        )
        for i, t in enumerate(mps):
            save_kwargs[f"mps_{i}"] = np.asarray(t, dtype=np.complex128)

        np.savez_compressed(str(temp_file), **save_kwargs)
        temp_file_with_ext = cache_path / f"{filename}.tmp.npz"
        os.replace(str(temp_file_with_ext), str(final_file))
        logging.info(
            f"Saved DMRG ground state to cache: {final_file} "
            f"(MPS: {len(mps)} tensors, dense_vector: "
            f"{'stored' if stored_vector.size else 'empty (>=26q or omitted)'})"
        )

    @staticmethod
    def _cache_payload_key(
        method: str = "",
        *,
        cache_file: Optional[Path] = None,
        suffix: str = "",
    ) -> str:
        """Return the method key that owns a cached payload."""
        for descriptor in (method, suffix, cache_file.stem if cache_file else ""):
            if not descriptor:
                continue
            descriptor = descriptor.lower()
            if descriptor.startswith("ground_state"):
                descriptor = descriptor[len("ground_state"):].lstrip("_")
            token = re.split(r"[^a-z0-9]+", descriptor, maxsplit=1)[0]
            if token:
                return token
        return "state_vector"

    @classmethod
    def _cache_payload_kind(
        cls,
        method: str = "",
        *,
        cache_file: Optional[Path] = None,
        suffix: str = "",
    ) -> str:
        """Classify cached payloads so exact vectors and auxiliary data are not conflated."""
        key = cls._cache_payload_key(method, cache_file=cache_file, suffix=suffix)
        if key == "dmrg":
            return "dmrg"
        if key in {"ccsd", "mp2", "cisd"}:
            return "auxiliary"
        if key == "fci":
            return "exact_vector"
        return "qubit_statevector"

    @classmethod
    def _cache_payload_is_qubit_statevector(
        cls,
        method: str = "",
        *,
        cache_file: Optional[Path] = None,
        suffix: str = "",
    ) -> bool:
        return (
            cls._cache_payload_kind(method, cache_file=cache_file, suffix=suffix)
            == "qubit_statevector"
        )

    def _load_cached_vector(
        self,
        data: Any,
        method: str,
        cache_file: Path,
        *,
        allow_auxiliary: bool = False,
    ) -> Optional[np.ndarray]:
        """Load a cached vector only when it is safe to materialize as a qubit statevector."""
        if 'vector' not in data.files:
            return None

        payload_kind = self._cache_payload_kind(method, cache_file=cache_file)
        if payload_kind == "dmrg":
            return None
        if payload_kind == "auxiliary" and not allow_auxiliary:
            return None

        if (
            self.n_qubits is not None
            and is_large_full_state_system(self.n_qubits)
            and payload_kind in {"qubit_statevector", "exact_vector"}
        ):
            logging.warning(
                "Skipping cached exact ground-state vector from %s for %d-qubit large system; "
                "full-state vectors are blocked.",
                cache_file,
                self.n_qubits,
            )
            return None

        vector = data['vector']
        if len(vector) == 0:
            return None

        if (
            self.n_qubits is not None
            and payload_kind == "qubit_statevector"
            and len(vector) != 2 ** self.n_qubits
        ):
            logging.warning(
                "Skipping cached ground-state vector from %s because length %d "
                "does not match 2^%d.",
                cache_file,
                len(vector),
                self.n_qubits,
            )
            return None

        return vector
    
    def _in_memory_state_matches_method(self, requested: str) -> bool:
        """Does the in-memory (energy, vector, mps) triple match the requested
        method?  Used by ``compute_ground_state`` to gate the "already
        computed" shortcut.

        - ``requested='auto'``: any source matches; auto resolved to dmrg
          earlier so this branch is unreachable, but keep the safe default.
        - ``requested='dmrg'``: the in-memory state must include an MPS;
          a leftover dense-only result from a prior exact run does not
          satisfy a DMRG request.
        - explicit exact method: the active source must be the same exact
          method. This keeps method-specific validation/profiling requests
          from being satisfied by whichever exact solver last left a dense
          vector in memory.
        """
        if requested == 'auto':
            return True
        active = self._ground_state_method
        if requested == 'dmrg':
            # DMRG request needs the MPS in memory.
            return self._ground_state_mps is not None
        # Explicit exact method: match by name only.
        if active is None:
            return False
        if active == requested:
            return True
        return False

    @staticmethod
    def _npz_path_has_valid_mps(path) -> bool:
        """True iff the file at ``path`` is loadable AND every ``mps_i``
        array can be converted to ``complex128``.

        This is the single source of truth for "is this file a valid DMRG
        cache" — used by ``_save_ground_state_dmrg`` overwrite checks,
        ``_has_dmrg_cache``, ``_load_best_ground_state`` prefilter, and the
        regen tool. The predicate must round-trip the *actual* loader: an
        ``mps_0=np.array(['bad'])`` payload passes a key-only check but
        crashes ``_load_mps_from_npz`` on the dtype conversion. Walking
        ``np.asarray(..., dtype=complex128)`` for every tensor here keeps the
        three sites strictly consistent.
        """
        try:
            with np.load(str(path)) as data:
                if 'mps_n_tensors' not in data.files:
                    return False
                try:
                    n = int(data['mps_n_tensors'])
                except Exception:
                    return False
                for i in range(n):
                    key = f'mps_{i}'
                    if key not in data.files:
                        return False
                    try:
                        # Read + convert — same conversion the actual loader
                        # performs. A malformed string array, an
                        # uncastable object dtype, or a dimensionality
                        # mismatch will raise here.
                        arr = np.asarray(data[key], dtype=np.complex128)
                        if arr.ndim != 3:
                            return False
                    except Exception:
                        return False
                return True
        except Exception:
            return False

    def _load_mps_from_npz(self, data) -> Optional[List[np.ndarray]]:
        """Extract MPS tensors from a loaded npz cache, if present.

        Layout: an MPS-bearing cache has key ``mps_n_tensors`` (int)
        plus ``mps_0``, ``mps_1``,... arrays. Returns the list of numpy
        arrays (CPU, complex128), or None if no MPS is stored, the payload
        is truncated, or any ``mps_i`` array cannot be converted to
        complex128 / has wrong rank. Keep this loader and the
        ``_npz_path_has_valid_mps`` predicate strictly consistent.
        """
        if 'mps_n_tensors' not in data.files:
            return None
        try:
            n = int(data['mps_n_tensors'])
        except Exception:
            return None
        tensors: List[np.ndarray] = []
        for i in range(n):
            key = f"mps_{i}"
            if key not in data.files:
                logging.warning(f"Cache MPS truncated: missing {key}")
                return None
            try:
                arr = np.asarray(data[key], dtype=np.complex128)
            except Exception as e:
                logging.warning(f"Cache MPS unreadable: {key} cannot cast to complex128 ({e})")
                return None
            if arr.ndim != 3:
                logging.warning(
                    f"Cache MPS malformed: {key} has ndim={arr.ndim}, expected 3"
                )
                return None
            tensors.append(arr)
        return tensors

    def _load_ground_state(
        self,
        suffix: str = "",
        expected_method: Optional[str] = None,
    ) -> Optional[Tuple[float, Optional[np.ndarray]]]:
        """Load ground state from cache if available.

        Args:
            suffix: cache filename suffix (e.g. an FCI / post-HF cache key).
            expected_method: if set, the cache's recorded ``method`` field
                must match the requested solver under
:meth:`_methods_are_compatible_for_exact_cache`. A
                mismatched cache (e.g. ``method='dense'`` requested but the
                npz says ``method='eigsh'``) returns ``None`` so the caller
                recomputes against the requested solver. Pinned by
                Previously, explicit-exact callers reused
                whatever exact-vector cache existed and silently mislabeled
                ``self._ground_state_method`` as the requested one.

        Side effect: if the cache contains an MPS payload,
        ``self._ground_state_mps`` is populated. Callers can read it via the
        ``ground_state_mps`` property.
        """
        filename = "ground_state"
        if suffix:
            filename += f"_{suffix}"
        cache_file = self._get_cache_path() / f"{filename}.npz"

        if not cache_file.exists():
            return None

        try:
            with np.load(cache_file) as data:
                energy = float(data['energy'])
                method = str(data.get('method', 'unknown'))
                if expected_method is not None and not self._methods_are_compatible_for_exact_cache(
                    requested=expected_method, cached=method
                ):
                    logging.info(
                        f"Skipping cache {cache_file}: cached method={method!r} "
                        f"does not match requested method={expected_method!r}."
                    )
                    return None
                vector = self._load_cached_vector(
                    data,
                    method,
                    cache_file,
                    allow_auxiliary=(
                        self._cache_payload_kind(method, cache_file=cache_file, suffix=suffix)
                        == "auxiliary"
                    ),
                )
                mps = self._load_mps_from_npz(data)
                if mps is not None:
                    self._ground_state_mps = mps
                logging.info(f"Loaded ground state from cache: {cache_file}")
                logging.info(f"  Method: {method}")
                logging.info(f"  Timestamp: {data.get('timestamp', 'unknown')}")
                if mps is not None:
                    logging.info(f"  MPS: {len(mps)} tensors")
            return energy, vector
        except Exception as e:
            logging.warning(f"Failed to load cache: {e}")
            return None

    @staticmethod
    def _methods_are_compatible_for_exact_cache(requested: str, cached: str) -> bool:
        """Return True iff ``cached`` is a valid hit for an explicit ``requested`` solver.

        Two solvers are compatible only when they produce a state vector
        that legitimately represents the same operator's ground state under
        the same numerical contract. Examples:

        * ``'dense'`` is satisfied only by another ``'dense'`` cache: a
          caller explicitly asking for dense diagonalization (e.g. for an
          exact-vs-DMRG parity test) must not be silently handed an
          ``'eigsh'`` Lanczos vector.
        * ``'eigsh'`` is similarly strict.
        * ``'fci'`` may be satisfied by a recorded ``'fci'`` cache (the FCI
          path threads its own ``suffix``, so file-name disambiguation does
          the routing; this hook is the second line of defense).

        Previously ``_load_ground_state`` returned
        whatever ``ground_state.npz`` existed regardless of solver tag, so
        ``compute_ground_state(method='dense', use_cache=True)`` after an
        ``'eigsh'`` run returned the eigsh vector while overwriting
        ``self._ground_state_method`` with the requested ``'dense'``.
        """
        if not requested or not cached:
            return False
        if cached == 'unknown':
            # Pre-PR caches lacked the ``method`` field. Conservatively
            # reject those for explicit-exact requests rather than risk
            # honouring a stale dense cache as eigsh (or vice versa).
            return False
        if requested == cached:
            return True
        # No cross-solver equivalence is honoured for explicit-exact requests.
        # DMRG caches carry an ``M{chi}`` suffix on the recorded method and
        # are routed through ``_load_best_ground_state`` instead, so they
        # are never compared here.
        return False

    def _load_best_ground_state(self) -> Optional[Tuple[float, Optional[np.ndarray], str]]:
        """
        Load the best available ground state from cache with priority.

        Priority order:
        1. DMRG: converged first, then descending bond dimension. DMRG
           entries lacking an ``mps_n_tensors`` payload key are rejected at
           prefilter time; candidates whose
           ``mps_i`` arrays fail to load fully fall through to the next
           DMRG entry rather than returning a half-loaded result.
        2. FCI (exact within basis)
        3. Regular cached ground state

        Returns:
            Tuple of (energy, vector, method) or None if no cache found.
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
                # Extract bond dimension and convergence metadata.
                import re
                match = re.search(r'M(\d+)', filename)
                bond_dim = int(match.group(1)) if match else 0
                converged = False
                trunc_err = float('inf')
                # DMRG entries must carry a *complete* MPS payload (every
                # mps_i present, not just the mps_n_tensors header). Use the
                # same validity predicate as _save_ground_state_dmrg and
                # _has_dmrg_cache so all three sites agree on what
                # constitutes a valid DMRG cache.
                if not self._npz_path_has_valid_mps(cache_file):
                    logging.warning(
                        f"Skipping DMRG cache file with missing / truncated MPS payload: "
                        f"{cache_file}. Run will regenerate the entry."
                    )
                    continue
                try:
                    with np.load(cache_file) as _peek:
                        if 'converged' in _peek.files:
                            converged = bool(_peek['converged'])
                        else:
                            # Caches with MPS but no convergence field
                            # (legacy molecular DMRG) are treated as converged.
                            converged = True
                        if 'final_trunc_err' in _peek.files:
                            trunc_err = float(_peek['final_trunc_err'])
                        else:
                            # No trunc metadata: treat as 0.0 (legacy molecular
                            # DMRG caches were always written as fully converged
                            # to machine precision; ranking by trunc is then a
                            # no-op against the bond_dim tie-breaker).
                            trunc_err = 0.0
                except Exception as e:
                    logging.warning(
                        f"Skipping unreadable DMRG cache file {cache_file}: {e}"
                    )
                    continue
                dmrg_files.append((cache_file, bond_dim, converged, trunc_err))
            elif 'fci' in filename.lower():
                fci_files.append(cache_file)
            else:
                other_files.append(cache_file)

        # Try DMRG files first. Preference order:
        #   1. converged caches (the converged flag is the rank-1 signal)
        #   2. within a bucket, lower final_trunc_err wins
        #   3. tie-break on bond_dim descending
        # Consistent with ``_save_ground_state_dmrg``'s "strictly better" rule.
        if dmrg_files:
            dmrg_files.sort(key=lambda x: (not x[2], x[3], -x[1]))
            # Walk candidates in priority order; an entry whose MPS payload fails to
            # fully load is not a valid DMRG cache and must not be honored — fall
            # through to the next-best file. The MPS is the canonical cached state,
            # so DMRG-without-MPS is corruption, not a hit.
            for best_dmrg_file, best_chi, best_converged, best_trunc in dmrg_files:
                try:
                    with np.load(best_dmrg_file) as data:
                        energy = float(data['energy'])
                        method = str(data.get('method', 'dmrg'))
                        vector = self._load_cached_vector(data, method, best_dmrg_file)
                        mps = self._load_mps_from_npz(data)
                        if mps is None:
                            logging.warning(
                                f"DMRG cache {best_dmrg_file} has 'mps_n_tensors' "
                                f"but the MPS payload is incomplete; skipping."
                            )
                            continue
                        self._ground_state_mps = mps
                        # Stash the chosen file so the metadata pass reads from the
                        # SAME payload the loaded MPS came from. A "highest-M file in
                        # the directory" heuristic could mismatch the loader's ranking
                        # and surface metadata from a different cache.
                        self._ground_state_cache_file = best_dmrg_file
                        logging.info(f"Loaded DMRG ground state from cache: {best_dmrg_file}")
                        logging.info(
                            f"  Method: {method}, Bond dim: {best_chi}, converged={best_converged}"
                        )
                        if not best_converged:
                            logging.warning(
                                f"  Loaded UNCONVERGED DMRG cache (final_trunc_err may be > svd_min). "
                                f"Consider rerunning with higher bond_dim or implementing lazy env caching."
                            )
                        logging.info(f"  Energy: {energy:.10f}")
                        logging.info(f"  MPS: {len(mps)} tensors")
                    return energy, vector, method
                except Exception as e:
                    logging.warning(f"Failed to load DMRG cache {best_dmrg_file}: {e}")
                    continue

        # Try FCI files next
        if fci_files:
            try:
                with np.load(fci_files[0]) as data:
                    energy = float(data['energy'])
                    method = str(data.get('method', 'fci'))
                    vector = self._load_cached_vector(data, method, fci_files[0])
                    logging.info(f"Loaded FCI ground state from cache: {fci_files[0]}")
                    logging.info(f"  Method: {method}")
                    logging.info(f"  Energy: {energy:.10f}")
                return energy, vector, method
            except Exception as e:
                logging.warning(f"Failed to load FCI cache: {e}")

        # Try other files last
        if other_files:
            try:
                with np.load(other_files[0]) as data:
                    energy = float(data['energy'])
                    method = str(data.get('method', 'unknown'))
                    vector = self._load_cached_vector(data, method, other_files[0])
                    logging.info(f"Loaded ground state from cache: {other_files[0]}")
                    logging.info(f"  Method: {method}")
                    logging.info(f"  Energy: {energy:.10f}")
                return energy, vector, method
            except Exception as e:
                logging.warning(f"Failed to load cache: {e}")

        return None
    
    def estimate_memory_usage(self, method: str = 'eigsh') -> Dict[str, Optional[float]]:
        """Estimate memory usage for ground state computation."""
        return estimate_full_state_memory(
            self.n_qubits,
            n_terms=len(self.pauli_str_list),
            method=method,
        )

    @staticmethod
    def _format_available_memory(mem_est: Dict[str, Optional[float]]) -> str:
        available_mb = mem_est.get('available_mb')
        if available_mb is None:
            return "unknown"
        return f"{available_mb:.1f} MB"
    
    def compute_ground_state(self, sparse: bool = True, k: int = 1, method: str = 'auto',
                           use_cache: bool = True, force_recompute: bool = False,
                           device: Optional[str] = None) -> Tuple[float, np.ndarray]:
        """
        Compute the ground state energy and state vector.

        Args:
            sparse: If True, use sparse matrix methods (legacy paths only).
            k: Number of lowest eigenvalues to compute.
            method: 'dmrg' (default via 'auto'), 'eigsh', 'lobpcg', 'dense',
                or 'auto' (resolves to 'dmrg').
            use_cache: Whether to use cached results if available.
            force_recompute: Force recomputation even if cached.
            device: Torch device string ('cuda' / 'cpu') for the DMRG path.
                None (default) lets DMRG auto-select cuda when available.
                Ignored by the legacy eigsh / dense / lobpcg paths.

        Returns:
            Tuple of (ground_state_energy, ground_state_vector).
            For n_qubits < 26: ``ground_state_vector`` is a dense 2**n
            complex128 array (contracted from the MPS for the DMRG path).
            For n_qubits >= 26: ``ground_state_vector`` is an empty array
            (the full-state guard would refuse to materialize 2**n entries);
            call:prop:`ground_state_mps` to get the MPS instead.

        Notes:
            DMRG is the default for ALL system sizes. The MPS is
            stored in the cache as the canonical state; the dense vector is
            contracted from the MPS on the fly for n<26 and is NOT cached on
            disk for DMRG entries.
            Legacy methods (eigsh / dense / lobpcg) are still available for
            small-system parity testing by passing ``method`` explicitly;
            those paths cache a dense vector as before.
        """
        # Resolve 'auto' first:
        #   - n <= EXACT_AUTO_DENSE_MAX_QUBITS (12): 'dense' (fast + exact)
        #   - 12 < n < EXACT_FULL_STATE_QUBIT_LIMIT (26): 'eigsh' (sparse + exact)
        #   - n >= 26: 'dmrg' (exact full-state is guard-blocked)
        # Explicit DMRG remains available; auto never silently downgrades a
        # feasible-exact system to the known-stall DMRG path.
        if method == 'auto':
            method = self._resolve_auto_method()

        # The O(2^n) guard only fires for full-state methods. DMRG / FCI
        # explicitly avoid the 2^n dense path.
        if method not in ('dmrg', 'fci'):
            self._guard_exact_full_state(
                "PauliHamiltonianHelper.compute_ground_state",
                method=method,
            )

        # Cache lookup honors the explicit method requested by the caller: an explicit
        # 'dense' / 'eigsh' / 'lobpcg' request must NOT be satisfied by a DMRG cache,
        # or exact-vs-DMRG parity tests silently compare DMRG against itself.
        #   - ``method='dmrg'``: honor only DMRG cache hits.
        #   - ``method`` in {'dense','eigsh','lobpcg','fci'}: skip the DMRG-preferring
        #     loader entirely, load only a matching dense-vector cache, and clear any
        #     in-memory DMRG MPS so it cannot satisfy downstream calls.
        explicit_exact = method in ('dense', 'eigsh', 'lobpcg', 'fci')

        # Explicit exact method: load only the matching dense-vector cache
        # (``ground_state.npz``) and clear any stale DMRG MPS that may have
        # been left in memory from a previous call. Skip the DMRG-preferring
        # ``_load_best_ground_state`` entirely so we don't even read the
        # DMRG cache file's MPS as a side effect.
        if (
            use_cache and not force_recompute and k == 1 and explicit_exact
        ):
            self._ground_state_mps = None
            # Keep ``_ground_state_cache_file`` paired with ``_ground_state_mps``
            # so a stale DMRG-cache pointer can't be read after the explicit
            # exact path clears the in-memory MPS (hygiene).
            self._ground_state_cache_file = None
            # Filter the cache by the *requested* solver
            # so an explicit ``method='dense'`` is never satisfied by an
            # eigsh cache (and vice versa). Without this, exact-vs-DMRG
            # parity tests can silently reuse the wrong solver's vector.
            cached = self._load_ground_state(expected_method=method)
            if cached is not None:
                self._ground_state_energy, self._ground_state_vector = cached
                if self._ground_state_vector is not None:
                    # Also clear MPS again — _load_ground_state may have
                    # populated it from an unrelated cache as a side effect.
                    if self._cache_payload_kind(method) != 'dmrg':
                        self._ground_state_mps = None
                        self._ground_state_cache_file = None
                    self._ground_state_method = method
                    return self._ground_state_energy, self._ground_state_vector
                logging.info(
                    f"Cached scalar present but no vector; recomputing under method={method!r}."
                )
            # No matching exact cache. Clear any DMRG-populated in-memory state too,
            # or the "already computed" shortcut below would return the stale DMRG
            # result for an explicit-exact request within the same helper instance.
            self._ground_state_energy = None
            self._ground_state_vector = None
            self._ground_state_method = None
            # Keep ``_ground_state_cache_file`` paired with the above reset
            # so the metadata pass cannot read from a stale DMRG
            # cache pointer after the helper has been explicitly steered
            # away from DMRG.
            self._ground_state_cache_file = None

        if use_cache and not force_recompute and k == 1 and not explicit_exact:
            cached_best = self._load_best_ground_state()
            if cached_best is not None:
                energy, vector, cached_method = cached_best
                cached_kind = self._cache_payload_kind(cached_method)
                if method == 'dmrg' and cached_kind != 'dmrg':
                    logging.info(
                        f"Ignoring cached {cached_method} entry: method='dmrg' "
                        f"was explicitly requested, will run DMRG."
                    )
                else:
                    self._ground_state_energy = energy
                    # For DMRG entries the cache loader returns vector=None
                    # (MPS is canonical). Tuple contract:
                    #  - n<26: contract MPS -> dense vector on the fly
                    #  - n>=26: empty array (full_state_guard blocks dense)
                    if vector is not None and vector.size > 0:
                        self._ground_state_vector = vector
                    elif (
                        self._ground_state_mps is not None
                        and self.n_qubits is not None
                        and not is_large_full_state_system(self.n_qubits)
                    ):
                        self._ground_state_vector = self._mps_to_dense_vector_numpy(
                            self._ground_state_mps
                        )
                    else:
                        self._ground_state_vector = np.array([], dtype=complex)
                    self._ground_state_method = cached_method
                    logging.info(f"Using cached ground state from {cached_method}")
                    return self._ground_state_energy, self._ground_state_vector

            # Fallback to regular cache loading (legacy path; non-DMRG).
            # Only the legacy methods consult this path — ``method='dmrg'``
            # must not be satisfied by a non-DMRG cache.
            if method != 'dmrg':
                cached = self._load_ground_state()
                if cached is not None:
                    self._ground_state_energy, self._ground_state_vector = cached
                    if self._ground_state_vector is not None:
                        self._ground_state_method = method
                        return self._ground_state_energy, self._ground_state_vector
                    logging.info(
                        "Using cached scalar ground state; recomputing vector for this exact request."
                    )

        # If already computed and k=1, return cached values — but only when the cached
        # result matches the requested method. Otherwise ``method='dmrg'`` after a
        # prior ``'dense'`` returns the dense vector without producing the canonical
        # MPS payload, and an explicit exact method could short-circuit on a leftover
        # DMRG result.
        if (
            self._ground_state_energy is not None
            and self._ground_state_vector is not None
            and k == 1
            and not force_recompute
            and self._in_memory_state_matches_method(method)
        ):
            return self._ground_state_energy, self._ground_state_vector

        # ---------------------------------------------------------------
        # DMRG path (default).
        # ---------------------------------------------------------------
        if method == 'dmrg':
            # DMRG solver repointed from the in-house torch Pauli-MPO backend to the
            # TeNPy backend. Same ``compute_ground_state_dmrg`` contract; TeNPy is
            # NumPy/CPU-only, so ``device`` is accepted-and-ignored on this offline
            # precompute path. The cache subsystem is unchanged.
            try:
                from .tenpy_dmrg import compute_ground_state_dmrg
            except ImportError:
                from tenpy_dmrg import compute_ground_state_dmrg

            logging.info(
                f"Computing ground state via DMRG (n_qubits={self.n_qubits}, "
                f"n_terms={len(self.pauli_str_list)})..."
            )
            # For n<26: ask the DMRG path to also contract the MPS to a dense state
            # vector. The cache stays MPS-canonical (no dense stored on disk), but the
            # in-memory tuple returns a real dense vector for legacy callers. For
            # n>=26 the full_state_guard would refuse to materialize it anyway, so
            # return an empty array per the existing convention.
            want_dense = (self.n_qubits is not None
                          and not is_large_full_state_system(self.n_qubits))
            energy, vec_or_none, info = compute_ground_state_dmrg(
                self.pauli_str_list,
                self.w_list,
                self.n_qubits,
                bond_dim=None,            # auto-pick from n_qubits
                return_dense_vector=want_dense,
                device=device,            # None -> DMRG auto-selects cuda
            )
            mps_numpy: List[np.ndarray] = info["mps_numpy"]
            final_chi = int(info["final_chi"])
            self._ground_state_energy = energy
            self._ground_state_vector = (
                vec_or_none if vec_or_none is not None
                else np.array([], dtype=complex)
            )
            self._ground_state_mps = mps_numpy
            self._ground_state_method = f"dmrg_M{final_chi}"
            logging.info(
                f"DMRG: E={energy:.10f}, chi={final_chi}, sweeps={info['n_sweeps']}, "
                f"trunc={info['final_trunc_err']:.3e}, converged={info['converged']}"
            )

            if use_cache and k == 1:
                self._save_ground_state_dmrg(
                    self._ground_state_energy,
                    mps_numpy,
                    method=f"dmrg_M{final_chi}",
                    converged=bool(info["converged"]),
                    n_sweeps=int(info["n_sweeps"]),
                    final_trunc_err=float(info["final_trunc_err"]),
                )
            return self._ground_state_energy, self._ground_state_vector

        # ---------------------------------------------------------------
        # Legacy paths: dense / eigsh / lobpcg.
        # ---------------------------------------------------------------
        # Check memory before proceeding
        mem_est = self.estimate_memory_usage(method)
        if mem_est['feasible'] is None:
            logging.info(
                "Memory availability is unknown; proceeding with %s without a feasibility check.",
                method,
            )
        elif mem_est['feasible'] is False:
            raise MemoryError(
                f"Insufficient memory for {method} method. "
                f"Estimated: {mem_est['total_mb']:.1f} MB, Available: {self._format_available_memory(mem_est)}. "
                f"System has {self.n_qubits} qubits ({2**self.n_qubits:,} states). "
                f"Consider using DMRG (method='dmrg') or LOBPCG."
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
            # Clear any in-memory DMRG MPS — this exact result supersedes any
            # stale MPS left from a previous DMRG run on the same helper.
            # Otherwise downstream verify_ground_state / compute_expectations
            # may consult the stale MPS rather than the new exact vector.
            self._ground_state_mps = None
            # Keep the cache-file pointer paired with the MPS reset.
            self._ground_state_cache_file = None
            self._ground_state_method = method

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
        5. Compute new via ``method='auto'`` — see ``_resolve_auto_method``
           for the size-based policy (dense / eigsh / dmrg). No O(2^n) guard
           is needed here because ``compute_ground_state`` re-applies it for
           the explicit dense/eigsh/lobpcg paths internally.
        """
        if self._ground_state_energy is None:
            # Try loading from cache first
            cached_best = self._load_best_ground_state()
            if cached_best is not None:
                energy, vector, method = cached_best
                self._ground_state_energy = energy
                self._ground_state_vector = vector
                self._ground_state_method = method
                logging.info(f"Loaded ground state energy from {method} cache: {energy:.10f}")
            else:
                # No cache -> compute via auto policy. dense / eigsh is the
                # fast-exact route for feasible sizes; DMRG only applies for
                # n >= EXACT_FULL_STATE_QUBIT_LIMIT (26) where exact full-state is
                # guard-blocked. Without this, 4-12q helpers could cache a DMRG-stall
                # energy as the canonical reference.
                self.compute_ground_state(method='auto')
        return self._ground_state_energy
    
    @property
    def ground_state_vector(self) -> np.ndarray:
        """Get the dense ground state vector (computing or contracting if necessary).

        For n_qubits >= 26 this raises (full-state guard). For n_qubits < 26:
        - If a dense vector is already computed/cached, return it.
        - Otherwise, if a DMRG MPS is available (in memory or via cache),
          contract it to a 2**n vector on demand and memoise the result.
        - Otherwise, fall through to ``compute_ground_state`` which (with
          DMRG as the default) will produce an MPS and then contract.

        Legacy callers reading ``ground_state_vector`` continue to work; new
        callers should prefer:prop:`ground_state_mps` to avoid materialising
        the 2**n vector.
        """
        self._guard_exact_full_state(
            "PauliHamiltonianHelper.ground_state_vector",
            method="state_vector",
        )
        # 1. Already have a non-empty dense vector → return it.
        if (
            self._ground_state_vector is not None
            and isinstance(self._ground_state_vector, np.ndarray)
            and self._ground_state_vector.size > 0
        ):
            return self._ground_state_vector

        # 2. Have an MPS (either fresh or loaded from cache) → contract.
        mps_np = self._ground_state_mps
        if mps_np is None:
            # Try loading from cache before falling through to compute.
            loaded = self._load_best_ground_state()
            if loaded is not None:
                energy, vec, method = loaded
                if self._ground_state_energy is None:
                    self._ground_state_energy = energy
                self._ground_state_method = method
                if vec is not None and vec.size > 0:
                    self._ground_state_vector = vec
                    return self._ground_state_vector
                mps_np = self._ground_state_mps  # set as side effect of loader

        if mps_np is not None:
            # Contract MPS -> dense vector on CPU.
            self._ground_state_vector = self._mps_to_dense_vector_numpy(mps_np)
            return self._ground_state_vector

        # 3. No MPS, no dense vector. Compute now (DMRG default) and recurse once.
        if self._ground_state_vector is None or self._ground_state_vector.size == 0:
            self.compute_ground_state()
        # compute_ground_state may have populated either a dense vector (eigsh
        # path) or an MPS (DMRG path). Contract the MPS if needed.
        if (
            (self._ground_state_vector is None or self._ground_state_vector.size == 0)
            and self._ground_state_mps is not None
        ):
            self._ground_state_vector = self._mps_to_dense_vector_numpy(self._ground_state_mps)
        return self._ground_state_vector

    @property
    def ground_state_mps(self) -> Optional[List[Any]]:
        """Get the ground-state MPS tensors (DMRG output), as torch tensors.

        Returns ``None`` when no DMRG cache or in-memory MPS is available
        (e.g. the cache was produced by an eigsh / dense / FCI path that
        only stores a dense state vector).

        The MPS is returned as a list of torch.complex128 tensors on CPU.
        Callers needing a specific device should ``.to(device)`` each tensor.

        This is the primitive (MPS-native ``EnergyEstimator``) reads
        at training time to compute expectation values without materialising
        the 2^n dense vector. Callers that only need the energy do not need
        to touch the MPS.
        """
        if self._ground_state_mps is None:
            # Try cache first.
            loaded = self._load_best_ground_state()
            if loaded is not None:
                # Side effect: _ground_state_mps may be populated.
                energy, _vec, method = loaded
                if self._ground_state_energy is None:
                    self._ground_state_energy = energy
                self._ground_state_method = method
        if self._ground_state_mps is None:
            # Nothing cached and nothing computed yet. Run DMRG (default).
            if self._ground_state_energy is None:
                self.compute_ground_state(method='dmrg')
        if self._ground_state_mps is None:
            return None
        try:
            import torch
        except ImportError as e:
            raise ImportError(
                "ground_state_mps requires torch (DMRG path uses torch tensors). "
                "Install with: pip install torch"
            ) from e
        return [
            torch.tensor(arr, dtype=torch.complex128)
            for arr in self._ground_state_mps
        ]

    def load_cached_dmrg_ground_state_mps(self) -> Optional[CachedMPSGroundState]:
        """Non-computing DMRG-MPS cache read for MPS-native startup.

        Unlike the ``ground_state_mps`` property, this method never calls
        ``compute_ground_state`` and never reads ``self.ground_state_energy``.
        It walks the cache directory via the existing
        ``_load_best_ground_state`` helper (which enforces the DMRG
        payload validity rules) and returns:

        - ``CachedMPSGroundState(energy, mps, method)`` when a valid DMRG
          cache is present (``self._ground_state_mps`` is populated as a
          side effect of ``_load_best_ground_state``).
        - ``None`` when no usable cache exists. Two sub-cases that callers
          may want to discriminate via ``self._ground_state_vector``:

          1. ``_load_best_ground_state`` itself returned ``None`` (no
             cache at all) — both ``_ground_state_vector`` and
             ``_ground_state_mps`` remain ``None``.
          2. ``_load_best_ground_state`` returned a non-MPS hit (FCI /
             other dense cache; only reachable for
             ``n_qubits < EXACT_FULL_STATE_QUBIT_LIMIT`` because dense
             vectors are filtered out for large systems) — the loader
             may have populated ``_ground_state_vector`` but
             ``_ground_state_mps`` is still ``None``.

        Used by MPS-native ``EnergyEstimator`` to fail fast with an
        actionable error rather than starting a long DMRG compute during
        construction.
        """
        loaded = self._load_best_ground_state()
        if loaded is None:
            return None
        energy, _vec, method = loaded
        # Verify the loader's hit is actually DMRG-backed. ``_load_best_ground_state``
        # only sets ``self._ground_state_mps`` on the DMRG branch, but a reused helper
        # instance can pair a fresh non-DMRG hit with a stale in-memory MPS from an
        # earlier load, silently corrupting MPS-native startup. Guard on the payload
        # kind so non-DMRG hits always return ``None``.
        if self._cache_payload_kind(method) != 'dmrg':
            return None
        if self._ground_state_mps is None:
            # Defensive: a DMRG hit should always populate ``_ground_state_mps``;
            # if it did not, treat the cache as unusable for MPS-native rather
            # than returning a half-loaded payload.
            return None
        # Cache the scalar energy on the instance so subsequent accesses
        # via the ``ground_state_energy`` property do not re-enter the
        # DMRG compute branch.
        if self._ground_state_energy is None:
            self._ground_state_energy = energy
        if getattr(self, "_ground_state_method", None) is None:
            self._ground_state_method = method
        # Best-effort metadata read. ``_load_best_ground_state`` validated the file is
        # loadable and stashed its absolute path, so use that exact file and the
        # surfaced metadata always corresponds to the loaded MPS rather than another
        # cache in the same directory. Failures fall back to ``None`` rather than
        # re-raising, since the MPS itself loaded fine.
        bond_dim_meta: Optional[int] = None
        converged_meta: Optional[bool] = None
        trunc_meta: Optional[float] = None
        sweeps_meta: Optional[int] = None
        chosen_file = self._ground_state_cache_file
        try:
            if chosen_file is not None and chosen_file.exists():
                bond_dim_from_filename: Optional[int] = None
                m = re.search(r"M(\d+)", chosen_file.stem)
                if m:
                    bond_dim_from_filename = int(m.group(1))
                with np.load(chosen_file) as data:
                    if "bond_dim" in data.files:
                        bond_dim_meta = int(data["bond_dim"])
                    elif bond_dim_from_filename is not None:
                        bond_dim_meta = bond_dim_from_filename
                    if "converged" in data.files:
                        converged_meta = bool(data["converged"])
                    if "final_trunc_err" in data.files:
                        trunc_meta = float(data["final_trunc_err"])
                    if "n_sweeps" in data.files:
                        sweeps_meta = int(data["n_sweeps"])
        except Exception as e:
            logger.debug("Optional MPS-cache metadata read failed: %s", e)
        return CachedMPSGroundState(
            energy=float(energy),
            mps=self._ground_state_mps,
            method=str(method),
            bond_dim=bond_dim_meta,
            converged=converged_meta,
            final_trunc_err=trunc_meta,
            n_sweeps=sweeps_meta,
        )

    @staticmethod
    def _mps_to_dense_vector_numpy(mps_np: List[np.ndarray]) -> np.ndarray:
        """Contract a list of MPS tensors (numpy) into a dense 2**L state vector.

        Each tensor has shape (chi_L, d, chi_R). Caller is responsible for
        the 26-qubit guard (this helper does not check).
        """
        if not mps_np:
            raise ValueError("empty MPS")
        # Sequential contraction: state of shape (1, 2^i, chi_i)
        state = mps_np[0]  # (1, d, chi_1)
        for i in range(1, len(mps_np)):
            # state (1, S, chi_i) * mps_i (chi_i, d, chi_{i+1}) → (1, S*d, chi_{i+1})
            _, S, chi_i = state.shape
            _, d, chi_next = mps_np[i].shape
            state = np.einsum('axc, cyd -> axyd', state, mps_np[i])
            state = state.reshape(1, S * d, chi_next)
        # Final state shape: (1, 2^L, 1) → flatten
        return state.reshape(-1).astype(np.complex128)
    
    def get_expectation_value(self, state: np.ndarray, pauli_str: str) -> complex:
        """
        Compute expectation value of a Pauli string for a given state.

        Args:
            state: Quantum state vector
            pauli_str: Pauli string operator

        Returns:
            Expectation value <state|pauli_str|state>
        """
        self._guard_exact_full_state(
            "PauliHamiltonianHelper.get_expectation_value",
            method="state_vector",
        )
        if len(pauli_str) != self.n_qubits:
            raise ValueError(f"Pauli string length {len(pauli_str)} doesn't match n_qubits {self.n_qubits}")
        
        op_matrix = self._pauli_string_to_matrix(pauli_str)
        return np.vdot(state, op_matrix @ state)
    
    def verify_ground_state_energy(self) -> float:
        """
        Verify ground state energy by computing <ψ|H|ψ>.

        Selection rules:

        1. If an in-memory exact dense vector is the ACTIVE state (eigsh,
           dense, lobpcg, fci recompute), use the dense-Pauli path — even
           when a DMRG cache happens to exist on disk. The disk DMRG cache
           is for a *different* representation of the same Hamiltonian and
           must not silently override a fresh exact verification.
        2. Otherwise prefer the MPS-native MPS-MPO sandwich whenever an
           in-memory MPS exists or a valid DMRG cache is on disk.
        3. Fall back to dense-Pauli path for non-DMRG cached states.

        For n_qubits >= 26 the dense fallback would hit the full-state
        guard; only the MPS path is meaningful at large n. If the MPS is
        not available the guard will raise.
        """
        # Rule 1: active in-memory dense exact result wins over disk DMRG.
        active_is_exact_in_memory = (
            self._ground_state_vector is not None
            and getattr(self._ground_state_vector, "size", 0) > 0
            and self._ground_state_mps is None
            and self._ground_state_method is not None
            and self._cache_payload_kind(self._ground_state_method) != 'dmrg'
        )

        # Rule 2: prefer MPS-native unless rule 1 fired.
        if not active_is_exact_in_memory and (
            self._ground_state_mps is not None or self._has_dmrg_cache()
        ):
            try:
                try:
                    from .pauli_mpo_dmrg import mps_mpo_expectation, pauli_sum_to_mpo
                except ImportError:
                    from pauli_mpo_dmrg import mps_mpo_expectation, pauli_sum_to_mpo
                import torch as _torch
            except ImportError:
                pass  # fall through to dense path
            else:
                # Trigger property to load/compute MPS if needed.
                mps_torch = self.ground_state_mps
                if mps_torch is not None:
                    mpo = pauli_sum_to_mpo(
                        self.pauli_str_list,
                        self.w_list,
                        self.n_qubits,
                        device=mps_torch[0].device,
                    )
                    # Pass assume_normalized=False: the MPS in memory may have
                    # come from a truncated DMRG sweep and thus carry norm < 1.
                    val = mps_mpo_expectation(mps_torch, mpo, assume_normalized=False)
                    return float(val.real)

        # Legacy dense-vector path (eigsh / dense / FCI cached state, or
        # an active in-memory exact recompute that rule 1 routed here).
        gs_vector = self.ground_state_vector
        energy = 0.0
        for pauli_str, coeff in zip(self.pauli_str_list, self.w_list):
            exp_val = self.get_expectation_value(gs_vector, pauli_str)
            energy += coeff * exp_val
        return energy.real

    def _has_dmrg_cache(self) -> bool:
        """Does a DMRG cache file with a real MPS payload exist for this Hamiltonian?

        Uses the same validity predicate as the loader / save guard
: the file must round-trip
        through ``_load_mps_from_npz`` — every ``mps_i`` array must be
        present. A truncated cache (``mps_n_tensors`` set but ``mps_i``
        missing for some i) is NOT a valid DMRG cache and must trigger a
        regen, not a "skipped (already cached)" no-op.
        """
        cache_path = self._get_cache_path()
        if not cache_path.exists():
            return False
        for p in cache_path.glob("ground_state*.npz"):
            if "dmrg" not in p.stem.lower():
                continue
            if self._npz_path_has_valid_mps(p):
                return True
        return False
    
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
        # Prefer the MPS-native path when the caller passed no explicit ``state`` AND
        # the active in-memory result is not a freshly-computed exact dense vector.
        # Same rule as ``verify_ground_state_energy``: an active eigsh/dense/lobpcg/fci
        # recompute must not be overridden by a disk DMRG cache.
        active_is_exact_in_memory = (
            self._ground_state_vector is not None
            and getattr(self._ground_state_vector, "size", 0) > 0
            and self._ground_state_mps is None
            and self._ground_state_method is not None
            and self._cache_payload_kind(self._ground_state_method) != 'dmrg'
        )
        if state is None and not active_is_exact_in_memory and (
            self._ground_state_mps is not None or self._has_dmrg_cache()
        ):
            try:
                try:
                    from .pauli_mpo_dmrg import mps_pauli_expectation
                except ImportError:
                    from pauli_mpo_dmrg import mps_pauli_expectation
            except ImportError:
                pass
            else:
                mps_torch = self.ground_state_mps
                if mps_torch is not None:
                    if pauli_strings is None:
                        pauli_strings = self.pauli_str_list
                    expectations: Dict[str, complex] = {}
                    for p in pauli_strings:
                        # Truncated DMRG MPS may have norm < 1; normalize.
                        val = mps_pauli_expectation(
                            mps_torch, p, assume_normalized=False,
                        )
                        expectations[p] = np.round(val.real, 9)
                    return expectations

        # Legacy dense-vector path.
        self._guard_exact_full_state(
            "PauliHamiltonianHelper.compute_expectations",
            method="state_vector",
        )
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

        For systems with n_qubits >= PREPROCESSING_EXACT_SOLVER_QUBIT_LIMIT
        (unless ``allow_exact_solver=True``), use:meth:`safe_summary` instead
        to avoid triggering O(2^n) ground-state computation.

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

    def safe_summary(self) -> Dict:
        """Return a summary that never triggers O(2^n) computation.

        Unlike:meth:`summary`, this skips ground-state energy and any
        other path that would materialise a full-state vector.  Safe to
        call on arbitrarily large systems.
        """
        exact_energies = self.get_exact_energy_from_file()
        exact_total_energy = None
        if exact_energies:
            exact_total_energy = exact_energies.get(
                'total_energy', exact_energies.get('electronic_energy')
            )

        hf_bitstrings = self.get_hartree_fock_bitstring()
        hf_bitstring = None
        if hf_bitstrings:
            transform = self.filepath.stem
            hf_bitstring = hf_bitstrings.get(transform)

        nonzero_coeffs = [abs(w) for w in self.w_list if abs(w) > 1e-10]

        return {
            'molecule': self.filepath.parent.name,
            'transformation': self.filepath.stem,
            'n_qubits': self.n_qubits,
            'n_terms': len(self.pauli_str_list),
            'ground_state_energy': None,
            'ground_state_energy_status': 'not_computed_large_system',
            'exact_energies': exact_energies,
            'exact_total_energy': exact_total_energy,
            'hf_bitstring': hf_bitstring,
            'largest_coefficient': max(abs(w) for w in self.w_list) if self.w_list else None,
            'smallest_coefficient': min(nonzero_coeffs) if nonzero_coeffs else None,
        }
    
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
                energy, vector = cached
                return energy, vector if vector is not None else np.array([])

        self._guard_exact_full_state(
            "PauliHamiltonianHelper.compute_ground_state_fci",
            method="fci",
        )
        
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

        Note:
            This analysis path is intentionally blocked for large systems. Use
            compute_ground_state_fci(..., use_cache=True) or ground_state_energy
            when only a cached scalar FCI energy is needed.
        """
        self._guard_exact_full_state(
            "PauliHamiltonianHelper.compute_ground_state_fci_with_analysis",
            method="fci",
        )

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
                    'data': data if data is not None else np.array([])
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
        self._guard_exact_full_state(
            "PauliHamiltonianHelper.compare_all_methods",
            method="auto",
        )
        results = {}

        # Exact diagonalization baseline. The contract is explicit here rather than
        # via ``method='auto'``: ``'dense'`` for the smallest systems (where
        # ``compare_all_methods`` is meant to run), so the label and solver always
        # agree. For ``n_qubits > 12`` fall back to ``'eigsh'`` (Lanczos), still an
        # exact qubit diagonalization up to convergence tolerance; the label below
        # reflects which solver was used.
        if self.n_qubits is not None and self.n_qubits <= self.EXACT_AUTO_DENSE_MAX_QUBITS:
            direct_method = 'dense'
            direct_label = 'Dense diagonalization'
        else:
            direct_method = 'eigsh'
            direct_label = 'Sparse Lanczos (eigsh)'
        try:
            e_direct, _ = self.compute_ground_state(method=direct_method, use_cache=True)
            results['Direct'] = {
                'energy': e_direct,
                'method': direct_label,
                'description': f'Exact diagonalization of qubit Hamiltonian via {direct_method}',
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
                
