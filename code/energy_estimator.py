# -*- coding: utf-8 -*-
"""
Energy estimator for quantum circuits using shadow tomography.

Notation:
=========

Shadow Tomography Protocol:
    Given Hamiltonian H = Σ_P w_P · P and ground state |ψ⟩:

    1. For each measurement circuit U_i:
       - Apply U_i to state: |ψ_i⟩ = U_i |ψ⟩ (Schrödinger picture)
       - Sample ONE outcome |b_i⟩ from |⟨b|ψ_i⟩|² (single-shot measurement)

    2. For each Pauli P in Hamiltonian:
       - Transform: P' = U_i P U_i† (Heisenberg picture)
       - If P' is diagonal (no X component): circuit can measure P
       - Eigenvalue: (-1)^{popcount(b_i & z_mask_P')} where z_mask is Z-part of P'
       - Sign: from phase of P' (must be even: 0→+1, 2→-1)

    3. Aggregate:
       - Hitting count N_P = number of circuits that can measure P
       - Estimate: ⟨P⟩ ≈ (1/N_P) Σ_{i: can_measure(i,P)} sign_i · eigenvalue_i

    4. Energy:
       E = Σ_P w_P · ⟨P⟩

Key Invariant:
    ⟨ψ|P|ψ⟩ = ⟨U_i ψ| U_i P U_i† |U_i ψ⟩ = ⟨ψ_i|P'|ψ_i⟩

    Both Heisenberg (P→P') and Schrödinger (|ψ⟩→|ψ_i⟩) transformations
    must use the same circuit U_i for correctness.

Tensor Shapes:
    batch_actions: (B, C, L) - B batches, C circuits, L max length
    batch_lengths: (B, C) - actual length of each circuit
    can_measure: (B, C, K) - 1.0 if circuit c can measure Pauli k
    z_masks: (B, C, K) - Z-part masks as integers for eigenvalue computation
    signs: (B, C, K) - ±1 from phase conversion
    outcomes: (B, C) - measurement outcomes as integer bit strings

Phase Invariant:
    Measurable Paulis (diagonal operators) must have even phase (0 or 2).
    Odd phases indicate non-Hermitian operators (implementation bug if seen).
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, field
import warnings
import logging

import math

try:
    # Try relative imports first (when run as module)
    from .measurement_adapter import (
        MPS_NATIVE_BACKEND,
        create_estimator_backend,
        resolve_tableau_backend,
    )
    from .quantum_action_mapping import build_action_mapping
    from .pauli_hamiltonian_helper import PauliHamiltonianHelper
    from .full_state_guard import guard_exact_full_state_request
except ImportError:
    # Fall back to absolute imports (when run directly)
    from measurement_adapter import (
        MPS_NATIVE_BACKEND,
        create_estimator_backend,
        resolve_tableau_backend,
    )
    from quantum_action_mapping import build_action_mapping
    from pauli_hamiltonian_helper import PauliHamiltonianHelper
    from full_state_guard import guard_exact_full_state_request

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# --------------------------------------------------------------------

@dataclass
class SimulationResult:
    """Results from multiple simulation runs.

    ``mean_absolute_error`` (MAE) and ``rmse`` measure *different*
    quantities; they are not drop-in replacements. Pick the one that
    matches what you are reporting.
    """
    energy_estimates: List[float]
    absolute_errors: List[float]
    rmse: float  # Root Mean Squared Error: sqrt(mean((E_m - E*)^2))
    std_absolute_error: float  # Sample std of |E_m - E*| across the M sims (ddof=1; 0 at M=1)
    mean_energy_estimate: float
    std_energy_estimate: float

    @property
    def mean_absolute_error(self) -> float:
        """Mean Absolute Error: mean(|E_m - E*|) across the M simulations.

        Different from ``rmse`` whenever the per-sim errors are not all
        equal. Use ``rmse`` when you want the L2 aggregate and ``mae``
        (this) when you want the L1 aggregate.
        """
        return float(np.mean(self.absolute_errors)) if self.absolute_errors else 0.0

@dataclass
class BatchElementEnergyResult:
    """Unified energy estimation result with optional extended metrics.

    semantics for ``energy_difference`` / ``absolute_error``:

    * M=1 (single simulation): ``|E - E_exact|`` for that simulation.
    * M>1 (aggregated): the Mean Absolute Error across the M simulations,
      ``mean(|E_m - E_exact|)`` — same quantity as ``mae`` below.

    The field name describes an absolute error, so the value must be one
    too. For the L2-aggregate (RMSE) at M>1, read ``rmse`` directly.
    """
    # Core energy metrics
    energy_estimate: float

    # Required batch processing fields (for backward compatibility)
    update: int
    batch_element_rank: int
    n_circuits: int
    # Shot count = n_circuits * M. At M=1 this equals n_circuits
    # (so legacy tests pinning n_circuits at M=1 still pass).
    total_measurements: int
    energy_difference: float  # |E - E*| at M=1, MAE at M>1.
    pauli_estimates: Dict[str, float]
    hitting_counts: Dict[str, int]
    circuit_lengths: List[int]
    mean_circuit_length: float
    # Defaults so loaders that omit these keys
    # (notably the legacy ``eval_results_update_*.json`` fixtures used
    # in test_run_config_success_path) reconstruct cleanly through the
    # shared ``_extended_result_from_record`` helper. Every live writer
    # passes both explicitly, so this is purely a robustness change.
    batch_cost: float = 0.0
    convergence_metrics: Dict = field(default_factory=dict)

    # Extended error metrics (optional)
    absolute_error: Optional[float] = None
    relative_error: Optional[float] = None
    variance: Optional[float] = None
    # Explicit aggregate-error fields populated by
    # estimate_energy_with_simulations at M>1; None at M=1.
    rmse: Optional[float] = None
    mae: Optional[float] = None

    # Trajectory information (optional)
    trajectory_index: Optional[int] = None
    circuit_depth: Optional[float] = None
    n_gates: Optional[float] = None

    # Circuit statistics (optional)
    circuit_depth_min: Optional[int] = None
    circuit_depth_max: Optional[int] = None
    circuit_depth_std: Optional[float] = None
    n_gates_min: Optional[int] = None
    n_gates_max: Optional[int] = None
    n_gates_std: Optional[float] = None

    # Simulation results (optional)
    simulation_results: Optional[SimulationResult] = None

    def __post_init__(self):
        """Ensure compatibility between energy_difference and absolute_error."""
        if self.absolute_error is None and self.energy_difference is not None:
            self.absolute_error = self.energy_difference
        elif self.energy_difference is None and self.absolute_error is not None:
            self.energy_difference = self.absolute_error


@dataclass
class PreparedCircuitData:
    """Cached data from circuit preparation for efficient i.i.d. sampling.

    This dataclass holds the SMALL precomputed data needed to perform
    multiple independent measurement simulations. Importantly, we do NOT
    cache the probability distributions (probs) as they are O(2^n) per circuit
    and would consume massive memory for large qubit systems.

    Memory-efficient design:
    - Pauli transformation data: O(batch × circuits × n_paulis) - SMALL
    - State vectors/probs: O(batch × circuits × 2^n) - NOT CACHED

    For 20 qubits with 100 circuits:
    - Pauli data (~600 terms): ~240 KB
    - Probs (if cached): ~4 GB  <-- avoided!

    Memory optimization via unique z_masks:
    - Many Paulis share the same z_mask pattern after Clifford transformation
    - We compute eigenvalues only for unique masks, then scatter to all Paulis
    - Typical reduction: 5-10x for molecular Hamiltonians
    """
    # Which Paulis can be measured by each circuit
    can_measure: torch.Tensor  # (batch_size, n_circuits, n_paulis)
    
    # Signs from phase conversion (0→+1, 2→-1)
    signs: torch.Tensor  # (batch_size, n_circuits, n_paulis)
    
    # Z masks for eigenvalue computation
    z_masks: torch.Tensor  # (batch_size, n_circuits, n_paulis)
    
    # Hitting counts per Pauli (how many circuits measure each Pauli)
    hits: torch.Tensor  # (batch_size, n_paulis)
    
    # Circuit actions and lengths (needed for re-applying circuits)
    batch_actions: torch.Tensor  # (batch_size, n_circuits, max_length)
    batch_lengths: torch.Tensor  # (batch_size, n_circuits)
    
    # Shape info
    batch_size: int
    n_circuits: int
    n_paulis: int
    
    # === Unique z_mask optimization (reduces memory by 5-10x) ===
    # For each circuit, we store unique z_mask values and a mapping back to Paulis
    # This avoids computing eigenvalues for duplicate z_masks
    
    # Unique z_masks per circuit: (batch_size, n_circuits, max_unique_masks)
    # Padded with -1 for circuits with fewer unique masks
    unique_z_masks: Optional[torch.Tensor] = None
    
    # Number of unique masks per circuit: (batch_size, n_circuits)
    n_unique_per_circuit: Optional[torch.Tensor] = None
    
    # Mapping from Pauli index to unique mask index: (batch_size, n_circuits, n_paulis)
    pauli_to_unique_idx: Optional[torch.Tensor] = None
    
    # Maximum number of unique masks across all circuits (for tensor sizing)
    max_unique_masks: Optional[int] = None

class EnergyEstimator:
    def __init__(self,
                 hamiltonian_helper: 'PauliHamiltonianHelper',
                 n_qubits: int,
                 device: Optional[Union[torch.device, str]] = None,
                 debug: bool = False,
                 force_cpu: bool = False,
                 measurement_backend: Optional[str] = 'clifford_map',
                 mps_max_bond_dim: Optional[int] = None,
                 mps_truncation_tol: float = 1e-10):
        """
        Initialize EnergyEstimator with optional CPU-only mode for async evaluation.

        Args:
            hamiltonian_helper: Hamiltonian helper object
            n_qubits: Number of qubits
            device: Device to use (can be string like 'cpu', 'cuda', or torch.device)
            debug: Enable debug logging
            force_cpu: Force CPU usage even if GPU is available (for async evaluation)
            measurement_backend: which tableau implementation to use.
                'clifford_map' (default): legacy code/clifford_map.py:CliffordMap.
                'tableau_batch_adapter': CT-backed shim from code/measurement_adapter/.
                'mps_native': MPS-factored state path,
                    requires CUDA + cuTensorNet at runtime.
                None/'auto': CT on CUDA, legacy on CPU. ``auto`` deliberately
                    does not pick ``mps_native`` in this layer; opt in
                    explicitly.
                Both ``tableau_batch_adapter`` and ``mps_native`` require
                CUDA at runtime and are rejected with ValueError when
                ``force_cpu`` is True or the resolved device is CPU. The
                legacy ``clifford_map`` backend runs on either device.
                Mirrors the GFlowNet flag in GFNs.py so direct/synchronous
                evaluation can use the same backend as training.
            mps_max_bond_dim: Optional cap on MPS bond dimension during
                gate application. Only consumed by ``mps_native``; ignored
                otherwise. ``None`` lets bond dim grow until the
                ``mps_truncation_tol`` threshold takes effect.
            mps_truncation_tol: SVD truncation tolerance for two-site MPS
                gate application. Only consumed by ``mps_native``.
        """
        self.hamiltonian_helper = hamiltonian_helper
        self.n_qubits = n_qubits
        self.debug = debug
        
        # Handle device selection with CPU forcing option
        if force_cpu:
            self.device = torch.device('cpu')
            logging.info("EnergyEstimator: Forced to use CPU for async evaluation")
        elif device is None:
            # Auto-detect best device
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            else:
                self.device = torch.device('cpu')
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device
            
        logging.info(f"EnergyEstimator initialized on device: {self.device}")

        helper_n_qubits = getattr(hamiltonian_helper, "n_qubits", n_qubits)
        if helper_n_qubits is not None and int(helper_n_qubits) != int(n_qubits):
            raise ValueError(
                f"EnergyEstimator n_qubits={n_qubits} does not match "
                f"hamiltonian_helper.n_qubits={helper_n_qubits}"
            )

        # Measurement backend selection (eval-time migration).
        # MUST resolve before the full-state guard so an explicit
        # ``mps_native`` request can bypass the state-vector guard for
        # ``n_qubits >= 26``. Dense backends still go through the guard
        # below.
        backend_selection = resolve_tableau_backend(measurement_backend, self.device)
        self.measurement_backend = backend_selection.name
        self._tableau_cls = backend_selection.tableau_cls
        self._estimator_backend = create_estimator_backend()
        logging.info(f"EnergyEstimator measurement backend: {self.measurement_backend}")

        # ``EnergyEstimator`` is the exact state-vector evaluation path for
        # dense backends. Block direct construction for large systems before
        # touching ``hamiltonian_helper.ground_state_energy`` or any
        # state-vector tensor. The MPS-native backend does not
        # materialize a 2**n vector, so it skips this guard.
        if self.measurement_backend != MPS_NATIVE_BACKEND:
            guard_exact_full_state_request(
                context="EnergyEstimator.__init__",
                n_qubits=int(helper_n_qubits),
                n_terms=len(getattr(hamiltonian_helper, "pauli_str_list", []) or []),
                method="state_vector",
                filepath=getattr(hamiltonian_helper, "filepath", None),
            )

        # Ground-state energy assignment. The dense path reads the property
        # unconditionally, and on a cache miss that property runs an unbounded DMRG
        # at constructor time. Use the non-computing cache reader for
        # ``mps_native`` so we fail fast instead.
        self._ground_state_mps_loaded: Optional[List[torch.Tensor]] = None
        if self.measurement_backend == MPS_NATIVE_BACKEND:
            cached_mps = self._load_cached_mps_for_native_backend(hamiltonian_helper)
            self.ground_state_energy = float(cached_mps.energy)
            # Promote to torch on the target device with the backend's dtype.
            self._ground_state_mps_loaded = [
                torch.tensor(arr, dtype=torch.complex128).to(self.device)
                for arr in cached_mps.mps
            ]
            # Surface MPS cache metadata for operator visibility.
            logging.info(
                "EnergyEstimator (mps_native): loaded MPS ground state "
                "from %s cache (bond_dim=%s, converged=%s, "
                "final_trunc_err=%s, n_sweeps=%s)",
                cached_mps.method,
                cached_mps.bond_dim,
                cached_mps.converged,
                cached_mps.final_trunc_err,
                cached_mps.n_sweeps,
            )
            if cached_mps.converged is False:
                warnings.warn(
                    "EnergyEstimator (mps_native) is using an UNCONVERGED "
                    "DMRG MPS cache (final_trunc_err="
                    f"{cached_mps.final_trunc_err}). Estimates will inherit "
                    "the DMRG residual; rerun precompute with a "
                    "higher bond_dim to converge."
                )
            # Instantiate the MPS-native sampling backend lazily later
            # (after action_map / terminal_action are built below).
            self._mps_native_backend = None  # populated below
        else:
            self.ground_state_energy = hamiltonian_helper.ground_state_energy
            self._mps_native_backend = None
        self.all_pauli_strings = hamiltonian_helper.pauli_str_list
        self.all_pauli_coeffs = [w.real for w in hamiltonian_helper.w_list]
        
        # Handle identity weight
        identity_str = "I" * n_qubits
        self.identity_weight = self.all_pauli_coeffs[self.all_pauli_strings.index(identity_str)] if identity_str in self.all_pauli_strings else 0.0

        # Keep identity in pauli_strings for proper coverage reporting
        # (it's always measurable and always has expectation value +1)
        self.pauli_strings = self.all_pauli_strings.copy()
        self.pauli_to_coeff = {p: c for p, c in zip(self.all_pauli_strings, self.all_pauli_coeffs)}
        
        # Build and store the action mapping
        self.action_map, self.terminal_action = build_action_mapping(self.n_qubits)
        
        # Convert Pauli strings to symplectic representation
        self.pauli_vecs = self._pauli_string_to_symplectic(self.pauli_strings)
        self.pauli_phases = self._get_pauli_phases(self.pauli_strings)
        
        # Setup gates and precompute masks
        self._setup_torch_quantum_gates()
        self._precompute_masks()
        
        # Cache for memory efficiency: holds an instance of self._tableau_cls
        # (CliffordMap or TableauBatchAdapter depending on measurement_backend).
        # Reused across calls when (batch_size, n_circuits) match; reset() is
        # called on cache hit at _get_or_create_clifford_map line 418.
        self._cached_clifford_map: Optional[object] = None
        self._cached_clifford_dims: Optional[Tuple[int, int]] = None  # (batch_size, n_circuits)

        # Instantiate the MPS-native sampling backend
        # after ``action_map`` and ``terminal_action`` are available. CUDA
        # runs use the cuTensorNet gate-split adapter when available; the
        # dense-reference fallback keeps CPU example/test paths on the same
        # high-level routing.
        if self.measurement_backend == MPS_NATIVE_BACKEND:
            try:
                from .measurement_adapter import (
                    CuTensorNetMPSBackend,
                    MPSNativeBackend,
                )
            except ImportError:
                from measurement_adapter import (  # type: ignore[no-redef]
                    CuTensorNetMPSBackend,
                    MPSNativeBackend,
                )
            # Pick the production GPU adapter when CUDA + cuQuantum are
            # available; otherwise use the dense-reference MPS fallback
            # backend (reachable in tests and the CPU example's local
            # resolver substitution).
            if (
                self.device.type == "cuda"
                and CuTensorNetMPSBackend.is_available(self.device)
            ):
                self._mps_native_backend = CuTensorNetMPSBackend(
                    n_qubits=self.n_qubits,
                    action_map=self.action_map,
                    terminal_action=self.terminal_action,
                    device=self.device,
                    max_bond_dim=mps_max_bond_dim,
                    truncation_tol=mps_truncation_tol,
                )
                logging.info(
                    "EnergyEstimator (mps_native): using CuTensorNetMPSBackend "
                    "(production GPU path)"
                )
            else:
                self._mps_native_backend = MPSNativeBackend(
                    n_qubits=self.n_qubits,
                    action_map=self.action_map,
                    terminal_action=self.terminal_action,
                    device=self.device,
                    max_bond_dim=mps_max_bond_dim,
                    truncation_tol=mps_truncation_tol,
                )
                logging.info(
                    "EnergyEstimator (mps_native): using MPSNativeBackend "
                    "with DenseReferenceMPSOps fallback path"
                )

    @staticmethod
    def _sync_mps_backend_action_mapping(estimator: "EnergyEstimator") -> None:
        """Keep MPS backend decode tables aligned with checkpoint overrides."""
        backend = getattr(estimator, "_mps_native_backend", None)
        if backend is None:
            return
        targets = [backend]
        delegate = getattr(backend, "_delegate", None)
        if delegate is not None:
            targets.append(delegate)
        for target in targets:
            target.action_map = estimator.action_map
            target.terminal_action = int(estimator.terminal_action)

    @torch.no_grad()
    def _pauli_string_to_symplectic(self, p_strs: List[str]) -> torch.Tensor:
        """Convert Pauli strings to symplectic representation."""
        vecs = torch.zeros(len(p_strs), 2 * self.n_qubits, dtype=torch.bool, device=self.device)
        for i, s in enumerate(p_strs):
            for j, c in enumerate(s):
                if c == 'X': 
                    vecs[i, j] = True
                elif c == 'Y': 
                    vecs[i, j] = True
                    vecs[i, self.n_qubits + j] = True
                elif c == 'Z': 
                    vecs[i, self.n_qubits + j] = True
        return vecs
    
    @torch.no_grad()
    def _get_pauli_phases(self, p_strs: List[str]) -> torch.Tensor:
        """Get the initial phases of Pauli strings.

        For Pauli strings from the Hamiltonian, the phase is always 0.
        The Hamiltonian specifies operators like "XY" or "Z" directly with their
        coefficients, not as products of generators.

        **Stim Convention (Verified)**:
        - Y = XZ (no intrinsic i stored in Y)
        - k₀ = 0 for all Paulis, including Y
        - Verified: stim.PauliString("Y") has sign +1, stim.PauliString("YY") has sign +1
        - If k₀ = y_count, then YY would have k₀ = 2 (sign = -1), but Stim shows +1

        Note: The intrinsic Y-phase (k₀ = y_count) would only apply if we were
        constructing Paulis by multiplying tableau generators, which is NOT the
        case for Hamiltonian terms.

        BUG FIX: Previously used y_count, which caused odd phases on measurable
        operators when Y-containing Paulis were transformed to Z-only operators
        (e.g., SH: Y → -Z). This violated the invariant that measurable (diagonal)
        operators must have even phases.
        """
        # All Hamiltonian Pauli strings have phase 0
        phases = torch.zeros(len(p_strs), dtype=torch.int8, device=self.device)

        if self.debug:
            print(f"\n[DEBUG] Initial Pauli phases (all zeros for Hamiltonian terms):")
            for i, (p_str, phase) in enumerate(zip(p_strs, phases)):
                print(f"  {p_str}: phase = {phase}")

        return phases

    @torch.no_grad()
    def _setup_torch_quantum_gates(self):
        """Setup quantum gates as torch tensors."""
        H = torch.tensor([[1, 1], [1, -1]], dtype=torch.complex64, device=self.device) / math.sqrt(2)
        S = torch.tensor([[1, 0], [0, 1j]], dtype=torch.complex64, device=self.device)
        self.torch_gates = {
            'H': H,
            'S': S,
            'HS': H @ S,
            'SH': S @ H,
            'HSH': H @ S @ H
        }

        # Setup adjoint gates for correct circuit application in reverse
        # When applying circuit in reverse order, we need adjoints
        # H† = H, S† = S^(-1), CNOT† = CNOT (self-adjoint)
        # (HS)† = S†H† (reverse order and take adjoints)
        self.torch_gates_adjoint = {
            'H': H,  # H is self-adjoint
            'S': S.conj().T,  # S† = S^(-1)
            'HS': (H @ S).conj().T,  # (HS)† = S†H†
            'SH': (S @ H).conj().T,  # (SH)† = H†S†
            'HSH': (H @ S @ H).conj().T  # (HSH)†
        }
        
    def _precompute_masks(self):
        """Precompute masks for efficient bitwise operations."""
        self.dim = 2 ** self.n_qubits
        # ``self.basis`` is a ``(2**n_qubits,)`` index tensor used only by the dense
        # state-vector CNOT path. The MPS-native backend does not consume it, so skip
        # the allocation there — at 52 qubits a length-``2**52`` int64 tensor would
        # OOM immediately, defeating the memory-ceiling lift.
        if self.measurement_backend == MPS_NATIVE_BACKEND:
            self.basis = None
        else:
            self.basis = torch.arange(self.dim, device=self.device, dtype=torch.long)
        
        # Precompute qubit masks for CNOT operations
        self.qubit_masks = torch.zeros(self.n_qubits, dtype=torch.long, device=self.device)
        for i in range(self.n_qubits):
            self.qubit_masks[i] = 1 << (self.n_qubits - 1 - i)
        
        # Precompute bit_shifts for _compute_z_masks_vectorized (avoids repeated allocation)
        self._bit_shifts = torch.arange(self.n_qubits - 1, -1, -1, device=self.device)
        self._bit_shifts_view = self._bit_shifts.view(1, 1, 1, -1)
        
        # GPU OPTIMIZATION: Pre-compute Pauli coefficients tensor to avoid Python loop in _estimate_energy
        # This tensor maps Pauli string index to its coefficient (excluding identity)
        n_paulis = len(self.pauli_strings)
        identity_str = "I" * self.n_qubits
        self._pauli_coeffs_tensor = torch.zeros(n_paulis, device=self.device, dtype=torch.float32)
        for p_idx, p_str in enumerate(self.pauli_strings):
            if p_str in self.pauli_to_coeff and p_str != identity_str:
                self._pauli_coeffs_tensor[p_idx] = self.pauli_to_coeff[p_str]
    
    def _get_or_create_clifford_map(self, batch_size: int, n_circuits: int) -> 'CliffordMap':
        """
        Get a CliffordMap instance, reusing cached one if dimensions match.

        This significantly reduces memory allocations in tight loops by reusing
        the same CliffordMap instance when batch_size and n_circuits match.

        Args:
            batch_size: Number of batch elements
            n_circuits: Number of circuits per batch element

        Returns:
            A reset CliffordMap ready for use
        """
        dims = (batch_size, n_circuits)
        
        # Check if cached map has matching dimensions
        if (self._cached_clifford_map is not None and 
            self._cached_clifford_dims == dims):
            # Reuse cached map - just reset it to identity state
            self._cached_clifford_map.reset()
            if self.debug:
                logging.debug(f"Reusing cached CliffordMap with dims {dims}")
            return self._cached_clifford_map
        
        # Create new CliffordMap and cache it
        if self.debug:
            logging.debug(f"Creating new CliffordMap with dims {dims} (previous: {self._cached_clifford_dims})")
        
        # Clear old cached map to free memory before creating new one
        if self._cached_clifford_map is not None:
            del self._cached_clifford_map
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
        
        self._cached_clifford_map = self._tableau_cls(
            self.n_qubits, batch_size, n_circuits, str(self.device)
        )
        self._cached_clifford_dims = dims
        
        return self._cached_clifford_map
    
    def clear_clifford_cache(self):
        """
        Explicitly clear the CliffordMap cache to free memory.

        Call this when you're done with a series of evaluations and want to
        release the memory held by the cached CliffordMap.
        """
        if self._cached_clifford_map is not None:
            del self._cached_clifford_map
            self._cached_clifford_map = None
            self._cached_clifford_dims = None
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
            if self.debug:
                logging.debug("Cleared CliffordMap cache")

    def get_ground_state(self) -> torch.Tensor:
        """Get the ground state vector (dense, 2**n).

        Legacy accessor — materialises a 2**n state vector and only works
        for n_qubits < 26 (per the full-state guard). Used by the dense
        evaluation path inside ``EnergyEstimator``.

        Calling this under ``measurement_backend='mps_native'`` raises
        immediately: the MPS-native path's whole point is to avoid the
        2**n allocation, so a stray call here would defeat the memory
        ceiling lift. Use ``self._ground_state_mps_loaded`` or
        ``self.hamiltonian_helper.ground_state_mps`` instead.
        """
        if self.measurement_backend == MPS_NATIVE_BACKEND:
            raise RuntimeError(
                "EnergyEstimator.get_ground_state() is the dense state-vector "
                "accessor and is forbidden in measurement_backend='mps_native' "
                "mode (memory ceiling). Use "
                "self._ground_state_mps_loaded for the cached MPS tensors, or "
                "self.hamiltonian_helper.ground_state_mps for the property "
                "form."
            )
        if hasattr(self.hamiltonian_helper, 'ground_state_vector'):
            return torch.tensor(self.hamiltonian_helper.ground_state_vector,
                              dtype=torch.complex64, device=self.device)
        raise ValueError("Hamiltonian helper does not have a ground state vector.")

    def _load_cached_mps_for_native_backend(self, hamiltonian_helper):
        """Non-computing DMRG-MPS cache read for MPS-native startup.

        Calls the helper's public ``load_cached_dmrg_ground_state_mps``
        wrapper around ``_load_best_ground_state``. Raises ``RuntimeError``
        with an actionable message if no MPS cache is available, with two
        sub-cases discriminated by ``helper._ground_state_vector`` state
        after the cache walk.
        """
        loader = getattr(hamiltonian_helper, "load_cached_dmrg_ground_state_mps", None)
        if loader is None:
            raise RuntimeError(
                "measurement_backend='mps_native' requires "
                "PauliHamiltonianHelper.load_cached_dmrg_ground_state_mps() "
                "Update to a helper version that provides it."
            )
        cached = loader()
        if cached is not None:
            return cached
        cache_dir = getattr(hamiltonian_helper, "_get_cache_path", lambda: "<unknown>")()
        # Discriminate "no cache at all" from "dense / FCI cache only" by re-probing
        # the cache walker: ``_load_best_ground_state`` returns ``None`` when no cache
        # files exist and a ``(energy, vector, method)`` tuple otherwise (the DMRG
        # branch also populates ``_ground_state_mps``). A ``None`` wrapper result
        # alongside an available tuple means the cache exists but has no MPS — the
        # dense-only case, reachable only for ``n_qubits < 26``.
        underlying = getattr(hamiltonian_helper, "_load_best_ground_state", None)
        had_some_cache = False
        if underlying is not None:
            try:
                probe = underlying()
                had_some_cache = probe is not None
            except Exception:  # pragma: no cover - defensive
                had_some_cache = False
        if had_some_cache:
            raise RuntimeError(
                "measurement_backend='mps_native' requires a DMRG-MPS "
                f"cache, but cache directory {cache_dir!s} only contains a "
                "dense / FCI ground-state cache. Run DMRG explicitly: "
                "helper.compute_ground_state(method='dmrg', use_cache=True), "
                "or precompute via the DMRG cache build."
            )
        raise RuntimeError(
            "measurement_backend='mps_native' requires a DMRG-MPS cache, "
            f"but no cache files were found at {cache_dir!s}. Run the "
            " DMRG precompute step, or call "
            "helper.compute_ground_state(method='dmrg', use_cache=True)."
        )

    def get_ground_state_mps(self) -> Optional[list]:
        """Return the ground-state MPS tensors (hook).

        Returns the MPS as a list of torch.complex128 tensors (on CPU; caller
        ``.to(self.device)`` as needed), or ``None`` if no MPS is cached
        (e.g. ground state came from eigsh, dense diag, or FCI).

        The MPS-native path reads this in place of ``get_ground_state()`` to
        avoid materialising the 2**n state vector. The MPS-MPO sandwich for
        per-Pauli expectations is in ``code.pauli_mpo_dmrg.mps_pauli_expectation``.
        """
        if not hasattr(self.hamiltonian_helper, 'ground_state_mps'):
            return None
        return self.hamiltonian_helper.ground_state_mps
    
    @torch.no_grad()
    def _apply_circuit_to_state_debug(self, state: torch.Tensor, circuit_actions: torch.Tensor, circuit_length: int) -> torch.Tensor:
        """Non-vectorized circuit application for debugging - matches original logic exactly."""
        state = state.clone()
        apply_ = False
        
        if self.debug:
            print(f"\n[DEBUG] _apply_circuit_to_state_debug:")
            print(f"  circuit_length: {circuit_length}")
            print(f"  circuit_actions: {circuit_actions}")
            print(f"  initial state: {state}")
        
        for step in range(circuit_length,-1,-1):
            step_ = step
            if circuit_length == circuit_actions.shape[0]:
                continue
            
            action_idx = circuit_actions[step_].item()
            gate_tuple = self.action_map.get(action_idx)
            
            if gate_tuple and gate_tuple[0] == "terminal":
                apply_ = True
                if self.debug:
                    print(f"  Found terminal at step {step_}")
            elif apply_:
                gate_name = gate_tuple[0]
                qubits = gate_tuple[1:]
                
                if self.debug:
                    print(f"  Applying {gate_name} on qubits {qubits} at step {step_}")

                if gate_name == "CNOT":
                    control, target = qubits[0], qubits[1]
                    dim = 2**self.n_qubits
                    basis = torch.arange(dim, device=self.device, dtype=torch.long)
                    c_mask, t_mask = 1 << (self.n_qubits - 1 - control), 1 << (self.n_qubits - 1 - target)
                    new_basis = torch.where((basis & c_mask) != 0, basis ^ t_mask, basis)
                    state = state[new_basis]
                else:  # Single-qubit gates
                    qubit = qubits[0]
                    gate = self.torch_gates[gate_name]
                    s_before, s_after = 2**qubit, 2**(self.n_qubits - qubit - 1)
                    state = state.reshape(s_before, 2, s_after)
                    state = torch.tensordot(gate, state, dims=([1], [1])).permute(1, 0, 2).reshape(-1)
        
        if self.debug:
            print(f"  final state: {state}")
            
        return state
    
    @torch.no_grad()
    def _apply_circuits_to_states(self, states: torch.Tensor, 
                                            batch_actions: torch.Tensor, 
                                            batch_lengths: torch.Tensor) -> torch.Tensor:
        """
        Vectorized application of U|ψ⟩ to quantum states.

        CliffordMap/Stim use the Heisenberg picture: P' = UPU†
        For consistency, we apply U to the state (gates in forward order).

        This ensures: ⟨Uψ|(UPU†)|Uψ⟩ = ⟨ψ|U†(UPU†)U|ψ⟩ = ⟨ψ|P|ψ⟩

        Args:
            states: Shape (batch_size, n_circuits, 2^n) or (2^n) for single state
            batch_actions: Shape (batch_size, n_circuits, max_length)
            batch_lengths: Shape (batch_size, n_circuits)

        Returns:
            Transformed states with shape (batch_size, n_circuits, 2^n)
        """
        batch_size, n_circuits, max_length = batch_actions.shape
        
        # Ensure all tensors are on the correct device
        batch_actions = batch_actions.to(self.device)
        batch_lengths = batch_lengths.to(self.device)
        
        # Broadcast state if needed
        if states.dim() == 1:
            states = states.unsqueeze(0).unsqueeze(0).expand(batch_size, n_circuits, -1)
        elif states.dim() == 2:
            states = states.unsqueeze(1).expand(-1, n_circuits, -1)

        # Clone once at the beginning to avoid modifying input and ensure we can do in-place updates
        states = states.to(self.device).clone()
        
        # Find terminal positions (vectorized). For rows with no terminal,
        # ``torch.where(..., max_length)`` falls back to max_length, so the
        # min-reduce here returns max_length — meaning "no terminal in row".
        terminal_mask = batch_actions == self.terminal_action
        positions = torch.arange(max_length, device=self.device).unsqueeze(0).unsqueeze(0)
        terminal_positions = torch.where(terminal_mask, positions, max_length).min(dim=2)[0]

        # Combine the terminal cutoff with batch_lengths so this path mirrors
        # ``_apply_circuits_to_map``: a gate at position s runs iff
        # s < min(batch_lengths[b, c], terminal_positions[b, c]). Otherwise rows
        # declared shorter than max_length but lacking a terminal token would apply
        # all max_length gates here while the tableau path stops at batch_lengths.
        effective_end = torch.minimum(batch_lengths.to(self.device), terminal_positions)

        if self.debug:
            print(f"\n[DEBUG] _apply_circuits_to_states (applying U):")
            print(f"  terminal_positions: {terminal_positions}")
            print(f"  batch_lengths: {batch_lengths}")
            print(f"  effective_end: {effective_end}")

        # Apply gates in FORWARD order (applies U)
        # CliffordMap computes P' = UPU† (Heisenberg picture, matching Stim)
        # State evolution: |ψ⟩ → U|ψ⟩
        # Combined: ⟨Uψ|UPU†|Uψ⟩ = ⟨ψ|P|ψ⟩
        for step in range(max_length):
            # Gate runs iff step is before BOTH the per-row length and the
            # terminal-token cutoff (matches the tableau-path semantics).
            step_active = step < effective_end
            if not step_active.any():
                break
                
            actions = batch_actions[:, :, step]
            
            # Process each action type
            for action_idx, gate_tuple in self.action_map.items():
                if gate_tuple[0] == "terminal":
                    continue
                    
                action_mask = (actions == action_idx) & step_active
                if not action_mask.any():
                    continue
                
                gate_name = gate_tuple[0]
                qubits = gate_tuple[1:]
                
                if self.debug and action_mask.any():
                    print(f"  Applying {gate_name} on qubits {qubits} at step {step}")
                
                if gate_name == "CNOT":
                    control, target = qubits[0], qubits[1]
                    c_mask = self.qubit_masks[control]
                    t_mask = self.qubit_masks[target]
                    
                    # Apply CNOT only where action_mask is True
                    affected_states = states[action_mask]  # Shape: (N, 2^n)
                    
                    # Create basis for each affected state
                    basis_expanded = self.basis.unsqueeze(0).expand(affected_states.shape[0], -1)
                    control_set = (basis_expanded & c_mask) != 0
                    new_basis = torch.where(control_set, basis_expanded ^ t_mask, basis_expanded)
                    
                    # Gather to apply CNOT
                    states[action_mask] = affected_states.gather(-1, new_basis)

                else:
                    # Single-qubit gates - use regular gates (not adjoints)
                    qubit = qubits[0]
                    gate = self.torch_gates[gate_name]  # Use regular gates

                    # Reshape for matrix multiplication
                    s_before = 2 ** qubit
                    s_after = 2 ** (self.n_qubits - qubit - 1)

                    affected_states = states[action_mask].reshape(-1, s_before, 2, s_after)
                    transformed = torch.einsum('ij,nbja->nbia', gate, affected_states)
                    states[action_mask] = transformed.reshape(-1, self.dim)
        
        if self.debug:
            print(f"  State[0,0] (identity): {states[0,0]}")
            print(f"  State[0,4] (H⊗H): {states[0,4]}")
        
        return states
    
    @torch.no_grad()
    def _apply_circuits_to_map(self, clifford_map: 'CliffordMap',
                              batch_actions: torch.Tensor,
                              batch_lengths: torch.Tensor):
        """Apply all circuits in the batch to the Clifford tableau.

        Uses the fully vectorized implementation for 10-15x speedup.
        """
        # Ensure tensors are on the correct device
        batch_actions = batch_actions.to(self.device)
        batch_lengths = batch_lengths.to(self.device)

        # Use the vectorized apply_action method directly if available
        if hasattr(clifford_map, 'apply_action'):
            # This will auto-precompute action tensors on first call
            clifford_map.apply_action(batch_actions, batch_lengths, self.action_map)
            return

        # Fallback to original implementation if vectorized method not available
        for b_idx in range(clifford_map.batch_size):
            for c_idx in range(clifford_map.n_measurements):
                mask = torch.zeros(clifford_map.batch_size, clifford_map.n_measurements,
                                 dtype=torch.bool, device=self.device)
                mask[b_idx, c_idx] = True

                # Find terminal position
                terminal_pos = -1
                for step in range(batch_lengths[b_idx, c_idx]):
                    if batch_actions[b_idx, c_idx, step] == self.terminal_action:
                        terminal_pos = step
                        break

                # Apply gates in forward order up to (but not including) terminal
                end_pos = terminal_pos if terminal_pos >= 0 else batch_lengths[b_idx, c_idx]

                for step in range(end_pos):
                    action_idx = batch_actions[b_idx, c_idx, step].item()
                    gate_tuple = self.action_map.get(action_idx)

                    if not gate_tuple or gate_tuple[0] == "terminal":
                        break

                    gate_name = gate_tuple[0]
                    qubits = gate_tuple[1:]

                    if gate_name == "H":
                        clifford_map.apply_H(qubits[0], mask)
                    elif gate_name == "S":
                        clifford_map.apply_S(qubits[0], mask)
                    elif gate_name == "HS":
                        clifford_map.apply_HS(qubits[0], mask)
                    elif gate_name == "SH":
                        clifford_map.apply_SH(qubits[0], mask)
                    elif gate_name == "HSH":
                        clifford_map.apply_HSH(qubits[0], mask)
                    elif gate_name == "CNOT":
                        clifford_map.apply_CNOT(qubits[0], qubits[1], mask)

    @torch.no_grad()
    def _phases_to_signs(self, phases: torch.Tensor) -> torch.Tensor:
        """
        Convert Z4 phases to real ±1 signs for the operator P' = U P U†.
        Assumes phases are those returned by `_compute_pauli_phases` (already conjugated).
        Mapping: 0 -> +1, 2 -> -1.
        If odd phases (1 or 3) are encountered (should not occur for measurable Paulis),
        set sign to 0 to avoid injecting a wrong ±1 and optionally warn in debug mode.

        GPU-optimized: Uses vectorized operations without creating intermediate tensors.
        """
        # Ensure integer type
        ph = phases.to(torch.int8)

        # Even/odd mask
        odd_mask = (ph & 1).bool()

        # Strict even-phase mapping: 0 -> +1, 2 -> -1, selected from the second bit
        # (phase=0 -> (ph & 2)==0 -> +1; phase=2 -> (ph & 2)==2 -> -1). Computed
        # directly rather than via full_like to reduce allocations:
        # sign = 1 - 2*((ph >> 1) & 1).
        even_signs = (1.0 - ((ph & 2).float())).to(torch.float32)

        # For odd phases, zero out (non-Hermitian), this should not occur for measurable terms.
        signs = torch.where(odd_mask,
                            torch.zeros_like(even_signs),
                            even_signs)

        if self.debug and odd_mask.any():
            print("[DEBUG] Odd phases (1 or 3) encountered in phase→sign conversion; "
                  "setting their signs to 0. This indicates an upstream phase-model issue for those terms.")
        return signs

    @torch.no_grad()
    def _compute_pauli_phases(self, clifford_map: 'CliffordMap',
                                       p_out: torch.Tensor,
                                       pauli_vecs: Optional[torch.Tensor] = None,
                                       pauli_phases: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Vectorized Z₄ phase of Q = U P U† for every Pauli P in the batch.

        This method now uses the estimator-backend shim for consistent phase
        computation that matches Stim exactly. The shim currently delegates to
        the transition-era Pauli phase model behind an adapter boundary.

        Args:
            clifford_map: Clifford tableau
            p_out: Transformed Pauli bits Q = U P U†, shape (B, C, K, 2n)
                   (used for shape inference; transform runs through the estimator backend)
            pauli_vecs: Optional custom Pauli vectors (K, 2n), defaults to self.pauli_vecs
            pauli_phases: Optional custom Pauli phases (K,), defaults to self.pauli_phases

        Returns:
            phases: Shape (batch_size, n_circuits, n_paulis) with values in {0,1,2,3}
                   representing phase exponents {+1, +i, -1, -i}
        """
        batch_size, n_circuits, n_paulis, _ = p_out.shape

        # Allow tests to pass custom Pauli subsets
        if pauli_vecs is None:
            pauli_vecs = self.pauli_vecs
        if pauli_phases is None:
            pauli_phases = self.pauli_phases

        if pauli_vecs.shape[0] != n_paulis or pauli_phases.shape[0] != n_paulis:
            raise ValueError(
                f"Mismatch between provided Pauli tensors and p_out. "
                f"Expected {n_paulis} rows, got "
                f"{pauli_vecs.shape[0]} (vecs) / {pauli_phases.shape[0]} (phases)."
            )

        transformed_paulis = self._estimator_backend.compute_transformed_paulis(
            clifford_map,
            pauli_vecs,
            pauli_phases,
        )
        phases = transformed_paulis.phase

        if self.debug:
            print(f"\n[DEBUG] _compute_pauli_phases (using estimator backend):")
            print(f"  initial phases (k): {pauli_phases}")
            print(
                "  phase_vec[0,0]: "
                f"{self._estimator_backend.heisenberg_phase_row(clifford_map, 0, 0)}"
            )
            if n_circuits > 4:
                print(
                    "  phase_vec[0,4] (example circuit): "
                    f"{self._estimator_backend.heisenberg_phase_row(clifford_map, 0, 4)}"
                )
            print(f"  computed phases[0,0][:min(8,{n_paulis})]: {phases[0,0,:min(8,n_paulis)]}")

        return phases

    @torch.no_grad()
    def _prepare_circuits_for_sampling(self, batch_actions: torch.Tensor,
                                        batch_lengths: torch.Tensor,
                                        ground_state: Optional[torch.Tensor]) -> PreparedCircuitData:
        """Prepare Pauli transformation data for efficient i.i.d. sampling.

        This method computes only the SMALL data that is independent of the random
        measurement outcomes (Pauli transformations). The large state vectors and
        probability distributions are NOT cached to save memory.

        Memory savings for 20 qubits with 100 circuits:
        - Before: ~4 GB (cached probs)
        - After: ~240 KB (just Pauli data)

        Args:
            batch_actions: Circuit actions (batch_size, n_circuits, max_length)
            batch_lengths: Circuit lengths (batch_size, n_circuits)
            ground_state: Ground state vector (used only for validation, not cached)

        Returns:
            PreparedCircuitData containing cached Pauli transformation data
        """
        batch_size, n_circuits, _ = batch_actions.shape
        n_paulis = len(self.pauli_strings)

        # === Step 1: Apply circuits to Clifford tableau ===
        clifford_map = self._get_or_create_clifford_map(batch_size, n_circuits)
        self._apply_circuits_to_map(clifford_map, batch_actions, batch_lengths)

        # === Step 2+3: Single Pauli transform: P' = UPU† (Heisenberg picture) ===
        # The estimator backend owns the compatibility path for legacy Pauli
        # phase tracking. Its output already contains transformed X / Z bits
        # plus phases, so prob_P_multi is redundant for measurability here.
        pauli_out = self._estimator_backend.compute_transformed_paulis(
            clifford_map,
            self.pauli_vecs,
            self.pauli_phases,
        )

        # Measurability: Z-basis-measurable iff transformed X half is all zero.
        # Equivalent to ``clifford_map.prob_P_multi(self.pauli_strings)`` bit-for-bit.
        can_measure = (~pauli_out.x.any(dim=-1)).float()  # (B, C, K) float32

        # Phases: (B, C, K)
        phases = pauli_out.phase

        # === Step 4: Validate phases (NO state computation here - saves memory!) ===
        # Invariant: measurable terms must have even phase (diagonal operators)
        meas = can_measure.bool()
        odd_phase = (phases & 1).bool()
        if (odd_phase & meas).any():
            raise AssertionError("Odd phase detected on diagonal (measurable) operator - phase calculation bug!")

        # Convert phases to signs: 0→+1, 2→-1
        signs = self._phases_to_signs(phases)

        # === Step 5: Compute Z masks for eigenvalue calculation ===
        # Pass pauli_out.z directly — no need to materialize the (B, C, K, 2n)
        # concat-then-slice round-trip; it's already (B, C, K, n).
        z_masks = self._compute_z_masks_vectorized(pauli_out.z)

        # === Step 6: Precompute hitting counts ===
        hits = can_measure.sum(dim=1)  # (batch, n_paulis)

        # === Step 7: Compute unique z_mask mapping for memory optimization ===
        # Many Paulis share the same z_mask, so we compute eigenvalues only for unique masks
        unique_z_masks, n_unique_per_circuit, pauli_to_unique_idx, max_unique_masks = \
            self._compute_unique_z_mask_mapping(z_masks)

        # NOTE: We do NOT cache probs or states - they are O(2^n) per circuit!
        # Instead, we store batch_actions so we can recompute states on-the-fly
        return PreparedCircuitData(
            can_measure=can_measure,
            signs=signs,
            z_masks=z_masks,
            hits=hits,
            batch_actions=batch_actions,
            batch_lengths=batch_lengths,
            batch_size=batch_size,
            n_circuits=n_circuits,
            n_paulis=n_paulis,
            unique_z_masks=unique_z_masks,
            n_unique_per_circuit=n_unique_per_circuit,
            pauli_to_unique_idx=pauli_to_unique_idx,
            max_unique_masks=max_unique_masks
        )

    @torch.no_grad()
    def _sample_outcomes_from_prepared(
        self,
        prepared: 'PreparedCircuitData',
        ground_state: Optional[torch.Tensor],
        M: int,
    ) -> torch.Tensor:
        """Generate ``(batch_size, n_circuits, M)`` integer outcomes.

        This is the single point where dense and MPS-native backends differ.
        Dense path materializes a ``(B, C, 2^n)`` state tensor and samples
        via ``torch.multinomial``. MPS-native delegates to
        ``self._mps_native_backend.sample_outcomes`` and reads the cached
        ground-state MPS stored on ``self._ground_state_mps_loaded``.

        Outcomes use big-endian qubit ordering
        (``1 << (n_qubits - 1 - q)``) — identical to the dense sampler.

        Args:
            prepared: PreparedCircuitData from
                ``_prepare_circuits_for_sampling``. ``batch_actions`` and
                ``batch_lengths`` are read here; the remaining fields are
                consumed by the shared accumulator in
                ``_sample_energies_from_prepared``.
            ground_state: dense ground state (used by the dense backend).
                Ignored for MPS-native; that path reads
                ``self._ground_state_mps_loaded`` instead.
            M: number of i.i.d. simulations per circuit.

        Returns:
            ``(batch_size, n_circuits, M)`` ``torch.long`` outcomes on
            ``self.device``.
        """
        if self.measurement_backend == MPS_NATIVE_BACKEND:
            if self._mps_native_backend is None or self._ground_state_mps_loaded is None:
                raise RuntimeError(
                    "measurement_backend='mps_native' but the MPS-native "
                    "backend was not initialized; did the constructor's "
                    "non-computing cache reader succeed?"
                )
            return self._mps_native_backend.sample_outcomes(
                self._ground_state_mps_loaded,
                prepared.batch_actions,
                prepared.batch_lengths,
                M,
            )

        # Dense state-vector path (unchanged contract).
        if ground_state is None:
            raise ValueError(
                "Dense backend requires a ground_state tensor; got None"
            )

        # Opt-in circuit chunking (``dense_sampling_circuit_chunk``; 0/unset uses the
        # legacy single-shot path below). The dense stage's only product is
        # ``outcomes`` and per-circuit draws are independent given the ground state,
        # so slicing the circuit axis is exact — statistically identical, though RNG
        # consumption order differs. Needed for 24q evals, where the unchunked path
        # materializes a (B*C, 2^n) complex128 state (~268 GB at C=1000, n=24).
        cc = int(getattr(self, "dense_sampling_circuit_chunk", 0) or 0)
        if cc > 0 and prepared.n_circuits > cc:
            outs = []
            for c0 in range(0, prepared.n_circuits, cc):
                c1 = min(c0 + cc, prepared.n_circuits)
                states = self._apply_circuits_to_states(
                    ground_state,
                    prepared.batch_actions[:, c0:c1],
                    prepared.batch_lengths[:, c0:c1],
                )
                probs = torch.abs(states) ** 2
                del states
                probs_flat = probs.reshape(-1, self.dim)
                outcomes_flat = torch.multinomial(probs_flat, M, replacement=True)
                outs.append(outcomes_flat.reshape(
                    prepared.batch_size, c1 - c0, M))
                del probs, probs_flat, outcomes_flat
            return torch.cat(outs, dim=1)

        states = self._apply_circuits_to_states(
            ground_state, prepared.batch_actions, prepared.batch_lengths
        )
        probs = torch.abs(states) ** 2  # (batch, circuits, 2^n)
        del states  # Free immediately

        batch_size = prepared.batch_size
        n_circuits = prepared.n_circuits
        probs_flat = probs.reshape(-1, self.dim)
        outcomes_flat = torch.multinomial(probs_flat, M, replacement=True)
        outcomes = outcomes_flat.reshape(batch_size, n_circuits, M)

        # Drop references; don't force a hot-path ``gc.collect()`` /
        # ``torch.cuda.empty_cache()``. The latter defeats the CUDA allocator's reuse
        # and forces a synchronization; the deletes alone let refcounting release the
        # tensors at the next allocator request. ``cleanup_gpu_memory`` is available
        # for callers under memory pressure.
        del probs, probs_flat, outcomes_flat
        return outcomes

    @torch.no_grad()
    def _sample_energies_from_prepared(self, prepared: PreparedCircuitData,
                                        ground_state: Optional[torch.Tensor],
                                        M: int = 1,
                                        compute_pauli_estimates: bool = True) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Sample M i.i.d. measurement outcomes and compute energy estimates.

        Memory-optimized implementation that:
        1. Computes states on-the-fly and releases immediately after sampling
        2. Processes simulations in chunks to avoid large intermediate tensors
        3. Uses in-place operations where possible

        Args:
            prepared: PreparedCircuitData from _prepare_circuits_for_sampling
            ground_state: Ground state vector
            M: Number of i.i.d. simulations to run
            compute_pauli_estimates: If True (default), materialize and return
                the per-Pauli estimates tensor of shape ``(batch_size, M,
                n_paulis)``. If False, skip both the allocation AND the
                per-chunk write into it — saves the ``(B, M, K) float32``
                allocation entirely. Energy values are unchanged either way;
                this only controls whether the per-Pauli decomposition is
                kept. The legacy ``_estimate_energy`` reporting path needs
                it; the new ``estimate_energy_tensor`` fast path does not.

        Returns:
            Tuple of:
                - energy_estimates: (batch_size, M) energy estimates
                - estimates_all: (batch_size, M, n_paulis) Pauli estimates
                  per simulation if ``compute_pauli_estimates`` is True,
                  else ``None``.
        """
        batch_size = prepared.batch_size
        n_circuits = prepared.n_circuits
        n_paulis = prepared.n_paulis

        # === Generate (B, C, M) integer outcomes ===
        # The outcome-generation step is the only thing that differs
        # between the dense state-vector backend and MPS-native.
        # Energy accumulation (signs / measurability /
        # unique-mask reduction) is shared across backends.
        outcomes_all = self._sample_outcomes_from_prepared(
            prepared, ground_state, M
        )

        # === Memory-efficient measurement computation using unique z_masks ===
        # Process in chunks over M to avoid huge intermediate tensors
        # With unique z_mask optimization, we compute eigenvalues for unique masks only
        # then scatter back to all Paulis, reducing memory by 5-10x
        
        # Use unique masks if available, otherwise fall back to full computation
        use_unique_optimization = (prepared.unique_z_masks is not None and 
                                   prepared.max_unique_masks is not None and
                                   prepared.max_unique_masks < n_paulis)
        
        if use_unique_optimization:
            # Memory estimate uses max_unique_masks instead of n_paulis
            bytes_per_sim = batch_size * n_circuits * prepared.max_unique_masks * 4
        else:
            bytes_per_sim = batch_size * n_circuits * n_paulis * 4
        
        target_bytes = 500 * 1024 * 1024  # 500 MB
        chunk_size = max(1, min(M, target_bytes // max(bytes_per_sim, 1)))
        
        # Pre-allocate output tensors. Skip the (B, M, K) float32
        # estimates_all when the caller doesn't need it — this is the
        # actual perf win behind ``estimate_energy_tensor(return_pauli_estimates=False)``.
        estimates_all = (
            torch.zeros(batch_size, M, n_paulis, device=self.device, dtype=torch.float32)
            if compute_pauli_estimates else None
        )
        energy_estimates = torch.zeros(batch_size, M, device=self.device, dtype=torch.float32)
        
        # Pre-compute hits (doesn't change across simulations)
        hits = prepared.hits  # (batch_size, n_paulis)
        
        # Process in chunks
        for m_start in range(0, M, chunk_size):
            m_end = min(m_start + chunk_size, M)
            m_chunk = m_end - m_start
            
            # Get outcomes for this chunk: (batch_size, n_circuits, m_chunk)
            outcomes_chunk = outcomes_all[:, :, m_start:m_end]
            
            if use_unique_optimization:
                # === OPTIMIZED PATH: Use unique z_masks ===
                eigenvalues = self._compute_eigenvalues_with_unique_masks(
                    outcomes_chunk, prepared
                )
            else:
                # === FALLBACK PATH: Original full computation ===
                eigenvalues = self._compute_eigenvalues_full(
                    outcomes_chunk, prepared.z_masks
                )
            
            # Apply signs and measurement mask efficiently
            # signs: (batch_size, n_circuits, n_paulis) -> expand to (batch_size, n_circuits, m_chunk, n_paulis)
            # Use expand (view) not repeat (copy)
            signs_expanded = prepared.signs.unsqueeze(2)
            can_measure_expanded = prepared.can_measure.unsqueeze(2)
            
            # Compute measurements: eigenvalues already float32
            eigenvalues.mul_(signs_expanded)  # In-place multiply by signs
            eigenvalues.mul_(can_measure_expanded.float())  # In-place multiply by measurable mask
            
            # Sum across circuits: (batch_size, m_chunk, n_paulis)
            sums_chunk = eigenvalues.sum(dim=1)
            del eigenvalues
            
            # Compute estimates for this chunk
            hits_expanded = hits.unsqueeze(1)  # (batch_size, 1, n_paulis)
            estimates_chunk = torch.where(
                hits_expanded > 0,
                sums_chunk / hits_expanded.float(),
                torch.zeros_like(sums_chunk)
            )
            del sums_chunk
            
            # Store in output tensor only if the caller asked for it.
            if estimates_all is not None:
                estimates_all[:, m_start:m_end, :] = estimates_chunk

            # Compute energies for this chunk
            energy_contrib = estimates_chunk * self._pauli_coeffs_tensor.view(1, 1, -1)
            energy_estimates[:, m_start:m_end] = self.identity_weight + energy_contrib.sum(dim=-1)
            del estimates_chunk, energy_contrib

        del outcomes_all
        return energy_estimates, estimates_all

    @torch.no_grad()
    def _estimate_energy(self, batch_actions: torch.Tensor,
                         batch_lengths: torch.Tensor,
                         ground_state: Optional[torch.Tensor],
                         M: int = 1) -> Tuple[List[List[BatchElementEnergyResult]], List[Dict[str, float]]]:
        """Unified energy estimation method handling both single and multiple simulations.

        This method first prepares the transformed states and Paulis (computed once),
        then performs M i.i.d. sampling operations using the cached data.

        Args:
            batch_actions: Circuit actions (batch_size, n_circuits, max_length)
            batch_lengths: Circuit lengths (batch_size, n_circuits)
            ground_state: Ground state vector
            M: Number of simulations (default: 1)

        Returns:
            Tuple ``(batch_simulation_results, mean_pauli_estimates_per_b)``:
              - ``batch_simulation_results``: list (batch elements) of lists
                (M simulations) of ``BatchElementEnergyResult``. Only the
                final simulation per batch element carries a populated
                ``pauli_estimates`` dict (it becomes ``final_results_object``);
                intermediate sims carry ``{}``.
              - ``mean_pauli_estimates_per_b``: list of per-batch-element dicts
                mapping each Pauli string to its mean estimate across the M
                simulations.

        Sole caller is ``estimate_energy_with_simulations``.
        """
        batch_size, n_circuits, _ = batch_actions.shape

        # === Phase 1: Prepare Pauli transformation data (computed once, small memory) ===
        prepared = self._prepare_circuits_for_sampling(batch_actions, batch_lengths, ground_state)

        # === Phase 2: I.I.D. sampling (computes states on-the-fly, releases immediately) ===
        energy_estimates, estimates_all = self._sample_energies_from_prepared(prepared, ground_state, M)

        # === Phase 3: Create result objects (CPU bundle assembly) ===
        try:
            from .energy_reporting import assemble_batch_element_results
        except ImportError:
            from energy_reporting import assemble_batch_element_results
        return assemble_batch_element_results(
            pauli_strings=self.pauli_strings,
            pauli_to_coeff=self.pauli_to_coeff,
            ground_state_energy=self.ground_state_energy,
            batch_lengths=batch_lengths,
            hits=prepared.hits,
            estimates_all=estimates_all,
            energy_estimates=energy_estimates,
            batch_size=batch_size,
            n_circuits=n_circuits,
            M=M,
            result_cls=BatchElementEnergyResult,
        )

    def _compute_z_masks_vectorized(self, z_parts: torch.Tensor) -> torch.Tensor:
        """Compute Z masks using vectorized bit operations.

        Args:
            z_parts: Z parts of transformed Paulis (batch, circuits, paulis, n_qubits)

        Returns:
            Z masks tensor (batch, circuits, paulis)
        """
        # Use precomputed bit_shifts to avoid repeated tensor allocation
        # Compute masks in one operation
        z_masks = (z_parts.long() << self._bit_shifts_view).sum(dim=-1)
        return z_masks

    def _compute_parities_vectorized(self, masked_outcomes: torch.Tensor) -> torch.Tensor:
        """Compute parities using fully vectorized bit counting.

        Args:
            masked_outcomes: Masked outcome values (any shape, last dims are preserved)

        Returns:
            Parity values (0 or 1) with same shape as input
        """
        # Fully vectorized parity computation using bit counting
        # Count set bits using built-in bit_count (PyTorch 1.9+)
        # For older PyTorch versions, use manual counting
        if hasattr(torch, 'bit_count'):
            # Use built-in bit_count for maximum efficiency
            parities = torch.bit_count(masked_outcomes.long()).to(torch.long) % 2
        else:
            # Fallback: vectorized XOR-based parity computation
            # Create bit masks for all positions at once
            bit_positions = torch.arange(self.n_qubits, device=masked_outcomes.device, dtype=torch.long)
            bit_masks = (masked_outcomes.unsqueeze(-1) >> bit_positions) & 1  # (..., n_qubits)
            parities = bit_masks.sum(dim=-1) % 2  # Sum bits and take mod 2
        
        return parities

    @torch.no_grad()
    def _compute_unique_z_mask_mapping(self, z_masks: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """Compute unique z_mask values and mapping for memory-efficient eigenvalue computation.

        Many Paulis share the same z_mask pattern after Clifford transformation.
        By computing eigenvalues only for unique masks, we reduce memory usage by 5-10x.

        Args:
            z_masks: Z masks tensor (batch_size, n_circuits, n_paulis)

        Returns:
            Tuple of:
                - unique_z_masks: (batch_size, n_circuits, max_unique_masks) - unique mask values, padded with 0
                - n_unique_per_circuit: (batch_size, n_circuits) - count of unique masks per circuit
                - pauli_to_unique_idx: (batch_size, n_circuits, n_paulis) - mapping from Pauli to unique index
                - max_unique_masks: int - maximum unique masks across all circuits
        """
        batch_size, n_circuits, n_paulis = z_masks.shape
        device = z_masks.device
        if batch_size == 0 or n_circuits == 0 or n_paulis == 0:
            unique_z_masks = torch.zeros(
                batch_size, n_circuits, 0, dtype=z_masks.dtype, device=device
            )
            n_unique_per_circuit = torch.zeros(
                batch_size, n_circuits, dtype=torch.long, device=device
            )
            pauli_to_unique_idx = torch.zeros(
                batch_size, n_circuits, n_paulis, dtype=torch.long, device=device
            )
            return unique_z_masks, n_unique_per_circuit, pauli_to_unique_idx, 0

        # Fully vectorized unique-per-(batch, circuit) mapping. The previous
        # implementation looped over every (b, c) pair calling ``torch.unique`` twice,
        # costing ``2 * batch_size * n_circuits`` GPU->host syncs on the eval prep hot
        # path. This sort-based version computes the same result with a single sync,
        # preserving the contract relied on by ``_compute_eigenvalues_with_unique_masks``:
        #     unique_z_masks[b, c, pauli_to_unique_idx[b, c, k]] == z_masks[b, c, k]
        # so the optimized eigenvalues stay bit-identical to the full path.
        #
        sorted_vals, sort_idx = torch.sort(z_masks, dim=-1)  # (B, C, K)

        # Boundary mask: position k starts a new unique value iff it differs
        # from its predecessor in sorted order (position 0 always does).
        is_new = torch.ones_like(sorted_vals, dtype=torch.bool)
        if n_paulis > 1:
            is_new[..., 1:] = sorted_vals[..., 1:] != sorted_vals[..., :-1]

        # Dense unique id (0-based) for each element in sorted order, and the
        # per-circuit unique count.
        unique_id_sorted = is_new.long().cumsum(dim=-1) - 1  # (B, C, K), values in [0, n_unique-1]
        n_unique_per_circuit = is_new.sum(dim=-1).to(torch.long)  # (B, C)

        # Single host sync: the global max sizes the padded output tensor.
        max_unique = int(n_unique_per_circuit.max().item()) if n_paulis > 0 else 0

        # Inverse map in ORIGINAL pauli order: scatter the sorted-order ids
        # back through the sort permutation.
        pauli_to_unique_idx = torch.empty_like(z_masks, dtype=torch.long)
        pauli_to_unique_idx.scatter_(-1, sort_idx, unique_id_sorted)

        # Padded unique values (pad with 0, matching the legacy contract).
        # Duplicate writes all carry the same value, so the scatter is
        # idempotent; padding columns [n_unique, max_unique) stay 0 and are
        # never gathered (no pauli maps to a padding id).
        unique_z_masks = torch.zeros(
            batch_size, n_circuits, max_unique, dtype=z_masks.dtype, device=device
        )
        if max_unique > 0:
            unique_z_masks.scatter_(-1, unique_id_sorted, sorted_vals)

        if self.debug:
            total_paulis = batch_size * n_circuits * n_paulis
            total_unique = n_unique_per_circuit.sum().item()
            reduction = total_paulis / total_unique if total_unique > 0 else 1.0
            logging.debug(f"Unique z_mask optimization: {total_paulis} -> {total_unique} "
                         f"({reduction:.1f}x reduction, max_unique={max_unique})")

        return unique_z_masks, n_unique_per_circuit, pauli_to_unique_idx, max_unique

    @torch.no_grad()
    def _compute_eigenvalues_full(self, outcomes_chunk: torch.Tensor, 
                                   z_masks: torch.Tensor) -> torch.Tensor:
        """Compute eigenvalues using full z_masks (original method).

        Args:
            outcomes_chunk: Measurement outcomes (batch_size, n_circuits, m_chunk)
            z_masks: Z masks (batch_size, n_circuits, n_paulis)

        Returns:
            eigenvalues: (batch_size, n_circuits, m_chunk, n_paulis)
        """
        # Expand for broadcasting: (batch_size, n_circuits, m_chunk, 1)
        outcomes_expanded = outcomes_chunk.unsqueeze(3)
        
        # z_masks: (batch_size, n_circuits, 1, n_paulis)
        z_masks_expanded = z_masks.unsqueeze(2)
        
        # Compute masked outcomes and parities
        masked = outcomes_expanded & z_masks_expanded
        parities = self._compute_parities_vectorized(masked)
        del masked  # Free immediately
        
        # Compute eigenvalues: 1 - 2*parity
        eigenvalues = parities.float()
        eigenvalues.mul_(-2.0).add_(1.0)
        del parities
        
        return eigenvalues

    @torch.no_grad()
    def _compute_eigenvalues_with_unique_masks(self, outcomes_chunk: torch.Tensor,
                                                prepared: 'PreparedCircuitData') -> torch.Tensor:
        """Compute eigenvalues using unique z_masks for memory efficiency.

        This method computes eigenvalues only for unique z_mask patterns, then
        scatters the results back to all Paulis. This reduces memory usage by
        5-10x for typical molecular Hamiltonians.

        Args:
            outcomes_chunk: Measurement outcomes (batch_size, n_circuits, m_chunk)
            prepared: PreparedCircuitData with unique mask mapping

        Returns:
            eigenvalues: (batch_size, n_circuits, m_chunk, n_paulis)
        """
        batch_size, n_circuits, m_chunk = outcomes_chunk.shape
        n_paulis = prepared.n_paulis
        max_unique = prepared.max_unique_masks
        
        # Step 1: Compute eigenvalues for unique masks only
        # outcomes: (batch_size, n_circuits, m_chunk, 1)
        outcomes_expanded = outcomes_chunk.unsqueeze(3)
        
        # unique_z_masks: (batch_size, n_circuits, 1, max_unique)
        unique_masks_expanded = prepared.unique_z_masks.unsqueeze(2)
        
        # Compute masked outcomes for unique masks: (batch_size, n_circuits, m_chunk, max_unique)
        masked_unique = outcomes_expanded & unique_masks_expanded
        parities_unique = self._compute_parities_vectorized(masked_unique)
        del masked_unique
        
        # Compute eigenvalues for unique masks: (batch_size, n_circuits, m_chunk, max_unique)
        eigenvalues_unique = parities_unique.float()
        eigenvalues_unique.mul_(-2.0).add_(1.0)
        del parities_unique
        
        # Step 2: Scatter eigenvalues back to all Paulis using the mapping
        # pauli_to_unique_idx: (batch_size, n_circuits, n_paulis)
        # Need to expand to (batch_size, n_circuits, m_chunk, n_paulis)
        idx_expanded = prepared.pauli_to_unique_idx.unsqueeze(2).expand(-1, -1, m_chunk, -1)
        
        # Gather from eigenvalues_unique using the mapping
        # eigenvalues_unique: (batch_size, n_circuits, m_chunk, max_unique)
        # idx_expanded: (batch_size, n_circuits, m_chunk, n_paulis)
        # Result: (batch_size, n_circuits, m_chunk, n_paulis)
        eigenvalues = torch.gather(eigenvalues_unique, dim=3, index=idx_expanded)
        del eigenvalues_unique, idx_expanded
        
        return eigenvalues

    @torch.no_grad()
    def prepare_circuits(self, batch_actions: torch.Tensor,
                         batch_lengths: torch.Tensor) -> PreparedCircuitData:
        """
        Prepare Pauli transformation data for efficient repeated sampling.

        Call this once, then call sample_from_prepared() multiple times with
        different M values. This is more memory-efficient than calling
        estimate_energy_with_simulations() multiple times because:
        1. Clifford tableau computation is done only once
        2. Pauli transformations are computed only once
        3. State vectors are computed on-the-fly for each sample batch

        Example:
            prepared = estimator.prepare_circuits(batch_actions, batch_lengths)
            for batch_idx in range(10):
                energies = estimator.sample_from_prepared(prepared, M=100)
                # accumulate energies...

        Args:
            batch_actions: Circuit actions tensor
            batch_lengths: Circuit lengths tensor

        Returns:
            PreparedCircuitData for use with sample_from_prepared()
        """
        batch_actions = batch_actions.to(self.device)
        batch_lengths = batch_lengths.to(self.device)
        # MPS-native skips ``get_ground_state()`` to preserve the memory
        # ceiling; ``_prepare_circuits_for_sampling`` only uses
        # ``ground_state`` for the legacy dense-validation hook.
        if self.measurement_backend == MPS_NATIVE_BACKEND:
            ground_state = None
        else:
            ground_state = self.get_ground_state()
        return self._prepare_circuits_for_sampling(batch_actions, batch_lengths, ground_state)
    
    @torch.no_grad()
    def sample_from_prepared(self, prepared: PreparedCircuitData,
                             M: int = 1,
                             ground_state: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample M i.i.d. energy estimates from prepared circuit data.

        This computes state vectors on-the-fly (not cached) to save memory,
        then immediately releases them after sampling.

        Args:
            prepared: PreparedCircuitData from prepare_circuits()
            M: Number of i.i.d. simulations
            ground_state: Optional ground state (uses Hamiltonian ground state if None)

        Returns:
            Tuple of (energy_estimates, pauli_estimates):
                - energy_estimates: (batch_size, M) energy values
                - pauli_estimates: (batch_size, M, n_paulis) per-Pauli estimates
        """
        # For ``mps_native`` the cached MPS is stored on
        # ``self._ground_state_mps_loaded``; the dense ``get_ground_state()``
        # would materialize a 2^n vector and defeat the memory-ceiling fix.
        if self.measurement_backend == MPS_NATIVE_BACKEND:
            return self._sample_energies_from_prepared(prepared, None, M)
        if ground_state is None:
            ground_state = self.get_ground_state()
        ground_state = ground_state.to(self.device)
        return self._sample_energies_from_prepared(prepared, ground_state, M)

    @torch.no_grad()
    def estimate_energy_tensor(self,
                               batch_actions: torch.Tensor,
                               batch_lengths: torch.Tensor,
                               M: int = 1,
                               ground_state: Optional[torch.Tensor] = None,
                               return_pauli_estimates: bool = False):
        """Tensor-only fast path for energy estimation.

        ``estimate_energy_with_simulations`` constructs a per-batch-per-M-per-Pauli
        Python dict of ``BatchElementEnergyResult`` objects and assembles
        ``mean_pauli_estimates`` / ``individual_*`` summaries. For training loops
        that just consume the mean energy (the common case), that work is pure
        overhead — for large M or many Paulis it can dominate total runtime via
        CPU transfers, dict allocations, and per-element warning emissions.

        This method returns the raw tensors from the sampling pipeline directly,
        skipping all dict / summary construction. Use it when you need energy
        values for downstream optimization; use
        ``estimate_energy_with_simulations`` when you need the reporting bundle.

        Args:
            batch_actions: ``(batch_size, n_circuits, max_length)``
            batch_lengths: ``(batch_size, n_circuits)``
            M: Number of i.i.d. measurement simulations per circuit
            ground_state: Optional ground state vector; defaults to
                ``self.get_ground_state()``.
            return_pauli_estimates: If True, also returns the per-Pauli estimate
                tensor of shape ``(batch_size, M, n_paulis)``. Default False to
                avoid materializing it when only the energy is needed.

        Returns:
            ``energy_estimates`` of shape ``(batch_size, M)`` (float32, on
            ``self.device``). If ``return_pauli_estimates`` is True, returns
            ``(energy_estimates, pauli_estimates)`` where ``pauli_estimates``
            has shape ``(batch_size, M, n_paulis)``.

        Caveats:
            This path intentionally does NOT emit the per-Pauli
            "never measured" warnings that ``estimate_energy_with_simulations``
            produces during dict construction — those are part of the
            reporting bundle, not the energy computation. If you rely on
            those diagnostics, keep using the legacy path or call
            ``estimate_energy_with_simulations`` periodically for spot
            checks. Likewise, the ``BatchElementEnergyResult`` objects,
            ``mean_pauli_estimates``, and the various ``individual_*``
            summaries are not produced here.
        """
        if M <= 0:
            raise ValueError("M must be positive")
        batch_actions = batch_actions.to(self.device)
        batch_lengths = batch_lengths.to(self.device)
        # MPS-native skips ``get_ground_state()`` to preserve the memory
        # ceiling; the cached MPS lives on ``self._ground_state_mps_loaded``.
        if self.measurement_backend == MPS_NATIVE_BACKEND:
            ground_state = None
        else:
            ground_state = self.get_ground_state() if ground_state is None else ground_state
            ground_state = ground_state.to(self.device)

        prepared = self._prepare_circuits_for_sampling(batch_actions, batch_lengths, ground_state)
        # Thread the no-pauli-estimates flag all the way through — the
        # docstring promised "avoid materializing it when only the energy
        # is needed", so the (B, M, K) float32 allocation must actually
        # be skipped, not just hidden from the return value.
        energy_estimates, estimates_all = self._sample_energies_from_prepared(
            prepared, ground_state, M,
            compute_pauli_estimates=return_pauli_estimates,
        )

        if return_pauli_estimates:
            return energy_estimates, estimates_all
        return energy_estimates

    def estimate_energy_with_simulations(self, batch_actions: torch.Tensor,
                                         batch_lengths: torch.Tensor,
                                         M: int = 1,
                                         ground_state: Optional[torch.Tensor] = None):
        """
        Estimate energy with M simulation runs.

        Args:
            batch_actions: Circuit actions tensor
            batch_lengths: Circuit lengths tensor
            M: Number of simulation runs
            ground_state: Optional ground state (uses Hamiltonian ground state if None)

        Returns:
            List of summary dictionaries, one per batch element
        """
        batch_size = batch_actions.shape[0]
        if M <= 0:
            raise ValueError("M must be positive")

        # Ensure tensors are on the correct device
        batch_actions = batch_actions.to(self.device)
        batch_lengths = batch_lengths.to(self.device)

        # MPS-native skips ``get_ground_state()`` to preserve the memory
        # ceiling; the cached MPS lives on ``self._ground_state_mps_loaded``.
        if self.measurement_backend == MPS_NATIVE_BACKEND:
            ground_state = None
        else:
            ground_state = self.get_ground_state() if ground_state is None else ground_state
            ground_state = ground_state.to(self.device)

        # Use unified _estimate_energy method which handles both single and multiple simulations efficiently
        batch_simulation_results, mean_pauli_estimates_per_b = self._estimate_energy(
            batch_actions, batch_lengths, ground_state, M)

        # Aggregate results (CPU summary assembly)
        try:
            from .energy_reporting import summarize_simulations
        except ImportError:
            from energy_reporting import summarize_simulations
        return summarize_simulations(
            batch_simulation_results=batch_simulation_results,
            mean_pauli_estimates_per_b=mean_pauli_estimates_per_b,
            batch_size=batch_size,
            M=M,
            ground_state_energy=self.ground_state_energy,
        )

    def cleanup_gpu_memory(self, clear_cache: bool = False):
        """
        Explicitly release GPU memory after large computations.
        Call this periodically during long training loops to prevent fragmentation.

        Args:
            clear_cache: If True, also clears the CliffordMap cache. Set to True
                        when switching to significantly different batch sizes, or
                        when you want to free all possible memory.
        """
        if clear_cache:
            self.clear_clifford_cache()
        
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            if self.debug:
                logging.info(f"GPU memory cleaned. Allocated: {torch.cuda.memory_allocated(self.device) / 1e9:.2f} GB")

    @classmethod
    def from_checkpoint_data(cls, hamiltonian_helper: 'PauliHamiltonianHelper',
                           n_qubits: int,
                           checkpoint_data: Dict,
                           force_cpu: bool = True,
                           measurement_backend: Optional[str] = None) -> 'EnergyEstimator':
        """
        Create an EnergyEstimator instance from checkpoint data.
        Useful for async evaluation without needing the full GFN model.

        Args:
            hamiltonian_helper: Hamiltonian helper object
            n_qubits: Number of qubits
            checkpoint_data: Dictionary containing checkpoint data
            force_cpu: Force CPU usage for async evaluation
            measurement_backend: optional backend override (e.g.
                'mps_native' for the TN sampling path, which reads a
                cached DMRG MPS instead of a dense ground state). ``None``
                keeps the constructor default.

        Returns:
            EnergyEstimator instance
        """
        device = 'cpu' if force_cpu else None
        ctor_kwargs = {}
        if measurement_backend is not None:
            ctor_kwargs["measurement_backend"] = measurement_backend
        estimator = cls(hamiltonian_helper, n_qubits, device=device,
                        force_cpu=force_cpu, **ctor_kwargs)
        
        # Update action mapping if provided in checkpoint
        if 'action_mapping' in checkpoint_data:
            estimator.action_map = checkpoint_data['action_mapping']
            logging.info("Loaded action mapping from checkpoint")
            
        if 'terminal_index' in checkpoint_data:
            estimator.terminal_action = checkpoint_data['terminal_index']
            logging.info(f"Loaded terminal action index: {estimator.terminal_action}")

        # The MPS-native backend snapshots action_map / terminal_action at
        # construction; keep its decode tables in sync with the checkpoint
        # overrides above or circuits would be misdecoded silently.
        cls._sync_mps_backend_action_mapping(estimator)

        return estimator


# Separate debug test function outside of main block
def debug_single_measurement():
    """Debug a single measurement to understand the issue."""
    import numpy as np
    
    class PauliHamiltonianHelper:
        def __init__(self):
            self.n_qubits = 2
            self.pauli_str_list = ['II', 'XX', 'YY', 'ZZ']
            self.w_list = [0.25, 0.25, 0.25, 0.25]
            self.ground_state_energy = -0.5
            # Bell state |ψ-⟩ = (|01⟩ - |10⟩)/√2
            self.ground_state_vector = np.array([0, 1, -1, 0], dtype=complex) / np.sqrt(2)
    
    print("\n=== Debug Single Measurement ===")
    device = torch.device('cpu')
    hamiltonian = PauliHamiltonianHelper()
    estimator = EnergyEstimator(hamiltonian, n_qubits=2, device=device, debug=False)
    
    # Create a single H⊗H circuit
    action_map, _ = build_action_mapping(n_qubits=2)
    reverse_action_map = {v: k for k, v in action_map.items()}
    
    batch_actions = torch.zeros(1, 1, 3, dtype=torch.long, device=device)
    batch_lengths = torch.ones(1, 1, dtype=torch.long, device=device) * 3
    
    # H⊗H circuit with terminal
    batch_actions[0, 0, 0] = reverse_action_map[('H', 0)]
    batch_actions[0, 0, 1] = reverse_action_map[('H', 1)]
    batch_actions[0, 0, 2] = estimator.terminal_action
    
    # Get ground state
    ground_state = estimator.get_ground_state()
    print(f"Ground state (|ψ-⟩): {ground_state}")
    
    # Apply circuit manually
    state = ground_state.clone()
    # Apply H to qubit 1 first (reverse order)
    H = estimator.torch_gates['H']
    state = state.reshape(2, 2)
    state = torch.matmul(state, H.T)  # H on qubit 1
    state = state.reshape(4)
    # Apply H to qubit 0
    state = state.reshape(2, 2)
    state = torch.matmul(H, state)  # H on qubit 0
    state = state.reshape(4)
    print(f"After H⊗H (manual): {state}")
    
    # Test measurement
    probs = torch.abs(state)**2
    print(f"Probabilities: {probs}")
    
    # For XX measurement (becomes ZZ after H⊗H)
    # Outcome 1 (|01⟩): ZZ eigenvalue = -1
    # Outcome 2 (|10⟩): ZZ eigenvalue = -1
    print("\nExpected: XX measurement should give -1")
    
    # Run the actual estimation
    summaries = estimator.estimate_energy_with_simulations(
        batch_actions=batch_actions, batch_lengths=batch_lengths, M=1
    )
    
    result = summaries[0]
    print(f"\nActual XX estimate: {result['mean_pauli_estimates'].get('XX', 0.0)}")
    print(f"Energy estimate: {result['mean_energy']:.6f}")
