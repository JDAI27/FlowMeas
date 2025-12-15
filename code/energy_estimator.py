# -*- coding: utf-8 -*-
"""Energy estimator for quantum circuits using derandomized measurement circuits."""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from collections import defaultdict
import warnings
import logging

import math

try:
    from .clifford_map import CliffordMap
    from .quantum_action_mapping import build_action_mapping
    from .pauli_hamiltonian_helper import PauliHamiltonianHelper
    from .pauli_tracker import Pauli
except ImportError:
    from clifford_map import CliffordMap
    from quantum_action_mapping import build_action_mapping
    from pauli_hamiltonian_helper import PauliHamiltonianHelper
    from pauli_tracker import Pauli

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


@dataclass
class SimulationResult:
    """Results from multiple simulation runs."""
    energy_estimates: List[float]
    absolute_errors: List[float]
    rmse: float
    std_absolute_error: float
    mean_energy_estimate: float
    std_energy_estimate: float

    @property
    def mean_absolute_error(self) -> float:
        """Return MAE for backward compatibility."""
        return np.mean(self.absolute_errors) if self.absolute_errors else 0.0

@dataclass
class BatchElementEnergyResult:
    """Energy estimation result for a batch element."""
    energy_estimate: float
    update: int
    batch_element_rank: int
    n_circuits: int
    total_measurements: int
    energy_difference: float
    pauli_estimates: Dict[str, float]
    hitting_counts: Dict[str, int]
    circuit_lengths: List[int]
    mean_circuit_length: float
    batch_cost: float
    convergence_metrics: Dict
    absolute_error: Optional[float] = None
    relative_error: Optional[float] = None
    variance: Optional[float] = None
    trajectory_index: Optional[int] = None
    circuit_depth: Optional[float] = None
    n_gates: Optional[float] = None
    circuit_depth_min: Optional[int] = None
    circuit_depth_max: Optional[int] = None
    circuit_depth_std: Optional[float] = None
    n_gates_min: Optional[int] = None
    n_gates_max: Optional[int] = None
    n_gates_std: Optional[float] = None
    simulation_results: Optional[SimulationResult] = None

    def __post_init__(self):
        if self.absolute_error is None and self.energy_difference is not None:
            self.absolute_error = self.energy_difference
        elif self.energy_difference is None and self.absolute_error is not None:
            self.energy_difference = self.absolute_error


@dataclass
class PreparedCircuitData:
    """Cached Pauli transformation data for efficient i.i.d. sampling.
    
    Memory-efficient design: caches only Pauli transformation data O(batch × circuits × n_paulis),
    not state vectors/probs O(batch × circuits × 2^n).
    """
    can_measure: torch.Tensor
    signs: torch.Tensor
    z_masks: torch.Tensor
    hits: torch.Tensor
    batch_actions: torch.Tensor
    batch_lengths: torch.Tensor
    batch_size: int
    n_circuits: int
    n_paulis: int
    unique_z_masks: Optional[torch.Tensor] = None
    n_unique_per_circuit: Optional[torch.Tensor] = None
    pauli_to_unique_idx: Optional[torch.Tensor] = None
    max_unique_masks: Optional[int] = None

class EnergyEstimator:
    def __init__(self, 
                 hamiltonian_helper: 'PauliHamiltonianHelper', 
                 n_qubits: int, 
                 device: Optional[Union[torch.device, str]] = None, 
                 debug: bool = False,
                 force_cpu: bool = False):
        """Initialize EnergyEstimator with optional CPU-only mode for async evaluation."""
        self.hamiltonian_helper = hamiltonian_helper
        self.n_qubits = n_qubits
        self.debug = debug
        
        if force_cpu:
            self.device = torch.device('cpu')
            logging.info("EnergyEstimator: Forced to use CPU for async evaluation")
        elif device is None:
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            else:
                self.device = torch.device('cpu')
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device
            
        logging.info(f"EnergyEstimator initialized on device: {self.device}")
        
        self.ground_state_energy = hamiltonian_helper.ground_state_energy
        self.all_pauli_strings = hamiltonian_helper.pauli_str_list
        self.all_pauli_coeffs = [w.real for w in hamiltonian_helper.w_list]
        
        identity_str = "I" * n_qubits
        self.identity_weight = self.all_pauli_coeffs[self.all_pauli_strings.index(identity_str)] if identity_str in self.all_pauli_strings else 0.0
        self.pauli_strings = self.all_pauli_strings.copy()
        self.pauli_to_coeff = {p: c for p, c in zip(self.all_pauli_strings, self.all_pauli_coeffs)}
        
        self.action_map, self.terminal_action = build_action_mapping(self.n_qubits)
        self.pauli_vecs = self._pauli_string_to_symplectic(self.pauli_strings)
        self.pauli_phases = self._get_pauli_phases(self.pauli_strings)
        
        self._setup_torch_quantum_gates()
        self._precompute_masks()
        
        self._cached_clifford_map: Optional['CliffordMap'] = None
        self._cached_clifford_dims: Optional[Tuple[int, int]] = None
        
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
        """Get initial phases of Pauli strings (always 0 for Hamiltonian terms)."""
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
            'H': H, 'S': S, 'HS': H @ S, 'SH': S @ H, 'HSH': H @ S @ H
        }
        self.torch_gates_adjoint = {
            'H': H, 'S': S.conj().T, 'HS': (H @ S).conj().T,
            'SH': (S @ H).conj().T, 'HSH': (H @ S @ H).conj().T
        }
        
    def _precompute_masks(self):
        """Precompute masks for efficient bitwise operations."""
        self.dim = 2 ** self.n_qubits
        self.basis = torch.arange(self.dim, device=self.device, dtype=torch.long)
        
        self.qubit_masks = torch.zeros(self.n_qubits, dtype=torch.long, device=self.device)
        for i in range(self.n_qubits):
            self.qubit_masks[i] = 1 << (self.n_qubits - 1 - i)
        
        self._bit_shifts = torch.arange(self.n_qubits - 1, -1, -1, device=self.device)
        self._bit_shifts_view = self._bit_shifts.view(1, 1, 1, -1)
        
        n_paulis = len(self.pauli_strings)
        identity_str = "I" * self.n_qubits
        self._pauli_coeffs_tensor = torch.zeros(n_paulis, device=self.device, dtype=torch.float32)
        for p_idx, p_str in enumerate(self.pauli_strings):
            if p_str in self.pauli_to_coeff and p_str != identity_str:
                self._pauli_coeffs_tensor[p_idx] = self.pauli_to_coeff[p_str]
    
    def _get_or_create_clifford_map(self, batch_size: int, n_circuits: int) -> 'CliffordMap':
        """Get a CliffordMap instance, reusing cached one if dimensions match."""
        dims = (batch_size, n_circuits)
        
        if self._cached_clifford_map is not None and self._cached_clifford_dims == dims:
            self._cached_clifford_map.reset()
            if self.debug:
                logging.debug(f"Reusing cached CliffordMap with dims {dims}")
            return self._cached_clifford_map
        
        if self.debug:
            logging.debug(f"Creating new CliffordMap with dims {dims} (previous: {self._cached_clifford_dims})")
        
        if self._cached_clifford_map is not None:
            del self._cached_clifford_map
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
        
        self._cached_clifford_map = CliffordMap(
            self.n_qubits, batch_size, n_circuits, str(self.device)
        )
        self._cached_clifford_dims = dims
        
        return self._cached_clifford_map
    
    def clear_clifford_cache(self):
        """Clear the CliffordMap cache to free memory."""
        if self._cached_clifford_map is not None:
            del self._cached_clifford_map
            self._cached_clifford_map = None
            self._cached_clifford_dims = None
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
            if self.debug:
                logging.debug("Cleared CliffordMap cache")

    def get_ground_state(self) -> torch.Tensor:
        """Get the ground state vector."""
        if hasattr(self.hamiltonian_helper, 'ground_state_vector'):
            return torch.tensor(self.hamiltonian_helper.ground_state_vector, 
                              dtype=torch.complex64, device=self.device)
        raise ValueError("Hamiltonian helper does not have a ground state vector.")
    
    @torch.no_grad()
    def _apply_circuit_to_state_debug(self, state: torch.Tensor, circuit_actions: torch.Tensor, circuit_length: int) -> torch.Tensor:
        """Non-vectorized circuit application for debugging."""
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
        """Vectorized application of U|ψ⟩ to quantum states."""
        batch_size, n_circuits, max_length = batch_actions.shape
        
        batch_actions = batch_actions.to(self.device)
        batch_lengths = batch_lengths.to(self.device)
        
        if states.dim() == 1:
            states = states.unsqueeze(0).unsqueeze(0).expand(batch_size, n_circuits, -1)
        elif states.dim() == 2:
            states = states.unsqueeze(1).expand(-1, n_circuits, -1)

        states = states.to(self.device).clone()
        
        terminal_mask = batch_actions == self.terminal_action
        positions = torch.arange(max_length, device=self.device).unsqueeze(0).unsqueeze(0)
        terminal_positions = torch.where(terminal_mask, positions, max_length).min(dim=2)[0]

        if self.debug:
            print(f"\n[DEBUG] _apply_circuits_to_states (applying U):")
            print(f"  terminal_positions: {terminal_positions}")
            print(f"  batch_lengths: {batch_lengths}")

        for step in range(max_length):
            step_active = step < terminal_positions
            if not step_active.any():
                continue
                
            actions = batch_actions[:, :, step]
            
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
                    
                    affected_states = states[action_mask]
                    basis_expanded = self.basis.unsqueeze(0).expand(affected_states.shape[0], -1)
                    control_set = (basis_expanded & c_mask) != 0
                    new_basis = torch.where(control_set, basis_expanded ^ t_mask, basis_expanded)
                    states[action_mask] = affected_states.gather(-1, new_basis)
                else:
                    qubit = qubits[0]
                    gate = self.torch_gates[gate_name]
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
        """Apply all circuits in the batch to the Clifford tableau."""
        batch_actions = batch_actions.to(self.device)
        batch_lengths = batch_lengths.to(self.device)

        if hasattr(clifford_map, 'apply_action'):
            clifford_map.apply_action(batch_actions, batch_lengths, self.action_map)
            return

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
        """Convert Z4 phases to real ±1 signs. Mapping: 0→+1, 2→-1, odd→0."""
        ph = phases.to(torch.int8)
        odd_mask = (ph & 1).bool()
        even_signs = (1.0 - ((ph & 2).float())).to(torch.float32)
        signs = torch.where(odd_mask, torch.zeros_like(even_signs), even_signs)
        if self.debug and odd_mask.any():
            print("[DEBUG] Odd phases encountered in phase→sign conversion; setting to 0.")
        return signs

    @torch.no_grad()
    def _compute_pauli_phases(self, clifford_map: 'CliffordMap',
                                       p_out: torch.Tensor,
                                       pauli_vecs: Optional[torch.Tensor] = None,
                                       pauli_phases: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute Z₄ phase of Q = U† P U for every Pauli P in the batch."""
        batch_size, n_circuits, n_paulis, _ = p_out.shape

        if pauli_vecs is None:
            pauli_vecs = self.pauli_vecs
        if pauli_phases is None:
            pauli_phases = self.pauli_phases

        if pauli_vecs.shape[0] != n_paulis or pauli_phases.shape[0] != n_paulis:
            raise ValueError(f"Mismatch between Pauli tensors and p_out.")

        x_in = pauli_vecs[:, :self.n_qubits]
        z_in = pauli_vecs[:, self.n_qubits:]
        pauli_in = Pauli(x_in, z_in, pauli_phases)
        pauli_out = pauli_in.apply_clifford(clifford_map)
        phases = pauli_out.phase

        if self.debug:
            print(f"\n[DEBUG] _compute_pauli_phases (using pauli_tracker):")
            print(f"  initial phases (k): {pauli_phases}")
            print(f"  computed phases[0,0][:min(8,{n_paulis})]: {phases[0,0,:min(8,n_paulis)]}")

        return phases

    @torch.no_grad()
    def _compute_pauli_expectations(self, clifford_map: 'CliffordMap', 
                                             ground_state: torch.Tensor,
                                             batch_actions: torch.Tensor, 
                                             batch_lengths: torch.Tensor) -> Tuple[List[Dict], List[Dict]]:
        """Compute Pauli expectations using shadow tomography."""
        batch_size, n_circuits, _ = batch_actions.shape
        n_paulis = len(self.pauli_strings)
        
        can_measure = clifford_map.prob_P_multi(self.pauli_strings)
        
        x_in = self.pauli_vecs[:, :self.n_qubits]
        z_in = self.pauli_vecs[:, self.n_qubits:]
        pauli_in = Pauli(x_in, z_in, self.pauli_phases)
        pauli_out = pauli_in.apply_clifford(clifford_map)
        p_out = torch.cat([pauli_out.x, pauli_out.z], dim=-1).byte()
        phases = pauli_out.phase

        if self.debug:
            print(f"\n  Clifford tableau check:")
            print(f"    Original XX: {self.pauli_vecs[0]}")
            print(f"    Original YY: {self.pauli_vecs[1]}")  
            print(f"    Original ZZ: {self.pauli_vecs[2]}")
            print(f"    Transformed XX for H circuit[0,4]: {p_out[0,4,0]}")
            print(f"    Transformed YY for H circuit[0,4]: {p_out[0,4,1]}")
            print(f"    Transformed ZZ for H circuit[0,4]: {p_out[0,4,2]}")
            print(f"    Phase vector content for some circuits:")
            for i in [0, 4, 8]:  # Identity, H, HSH circuits
                if i < n_circuits:
                    print(f"      Circuit {i} phase_vec: {clifford_map.heis_phase_vec[0,i]}")
            
            # Check W matrix for H circuit
            print(f"\n    W matrix for identity circuit [0,0]:")
            print(f"    {clifford_map.W[0,0]}")
            print(f"    W matrix for H circuit [0,4]:")
            print(f"    {clifford_map.W[0,4]}")
        
        states = self._apply_circuits_to_states(ground_state, batch_actions, batch_lengths)
        
        if self.debug:
            debug_state = self._apply_circuit_to_state_debug(
                ground_state, batch_actions[0, 0], batch_lengths[0, 0]
            )
            print(f"\n[DEBUG] State comparison for circuit[0,0]:")
            print(f"  Vectorized state: {states[0,0]}")
            print(f"  Debug state: {debug_state}")
            print(f"  States match: {torch.allclose(states[0,0], debug_state)}")
        
        probs = torch.abs(states) ** 2
        outcomes = torch.multinomial(probs.reshape(-1, self.dim), 1, replacement=True)
        outcomes = outcomes.reshape(batch_size, n_circuits)
        
        if self.debug:
            print(f"\n[DEBUG] _compute_pauli_expectations:")
            print(f"  can_measure[0,4:8]: {can_measure[0,4:8]}")  # H circuits
            print(f"  outcomes[0]: {outcomes[0]}")
            print(f"  outcomes binary: {[f'{o:02b}' for o in outcomes[0]]}")
            print(f"  probs[0,0]: {probs[0,0]}")
            print(f"  probs[0,4]: {probs[0,4]}")
        
        meas = can_measure.bool()
        odd_phase = (phases & 1).bool()
        if (odd_phase & meas).any():
            if self.debug:
                print(f"DEBUG: Odd phases found: {phases[odd_phase & meas]}")
            raise AssertionError("Odd phase detected on diagonal (measurable) operator")

        signs = self._phases_to_signs(phases)
        z_parts = p_out[:, :, :, self.n_qubits:]
        z_masks = self._compute_z_masks_vectorized(z_parts)
        outcomes_expanded = outcomes.unsqueeze(2)
        masked_outcomes = outcomes_expanded & z_masks
        parities = self._compute_parities_vectorized(masked_outcomes)
        eigenvalues = 1.0 - 2.0 * parities.float()
        
        if self.debug:
            print(f"  z_parts[0,4,0] (XX->ZZ): {z_parts[0,4,0]}")
            print(f"  z_masks[0,4,0]: {z_masks[0,4,0]:04b}")
            print(f"  outcome[0,4]: {outcomes[0,4]:04b}")
            print(f"  masked_outcome[0,4,0]: {masked_outcomes[0,4,0]:04b}")
            print(f"  parity[0,4,0]: {parities[0,4,0]}")
            print(f"  Raw eigenvalues[0,4:8,0] (before sign): {eigenvalues[0,4:8,0]}")
        
        # Combine with signs and measurement capability
        measurements = signs * eigenvalues * can_measure.float()
        
        if self.debug:
            print(f"  phases[0,4:8,0]: {phases[0,4:8,0]}")  # XX phases for H circuits
            print(f"  signs[0,4:8,0]: {signs[0,4:8,0]}")  # XX signs for H circuits  
            print(f"  eigenvalues[0,4:8,0]: {eigenvalues[0,4:8,0]}")  # XX eigenvalues
            print(f"  measurements[0,4:8,0]: {measurements[0,4:8,0]}")  # XX measurements
            print(f"  p_out[0,4,0]: {p_out[0,4,0]}")  # Transformed XX for circuit 4
            
            # Detailed debug for one H circuit measurement
            print(f"\n  Detailed debug for H circuit 4, XX measurement:")
            print(f"    Outcome: {outcomes[0,4]} = {outcomes[0,4]:02b}")
            print(f"    Z-part of transformed XX: {z_parts[0,4,0]}")
            print(f"    Z-mask: {z_masks[0,4,0]} = {z_masks[0,4,0]:02b}")
            print(f"    Masked: {masked_outcomes[0,4,0]} = {masked_outcomes[0,4,0]:02b}")
            print(f"    Parity: {parities[0,4,0]}")
            print(f"    Eigenvalue: {eigenvalues[0,4,0]}")
            print(f"    Phase: {phases[0,4,0]}")
            print(f"    Sign: {signs[0,4,0]}")
            print(f"    Final measurement: {measurements[0,4,0]}")
        
        batch_pauli_estimates = []
        batch_hitting_counts = []
        
        if self.debug:
            print(f"\n  Accumulation debug:")
            print(f"    Total measurements shape: {measurements.shape}")
            print(f"    Measurements[0,:,0] (XX for all circuits): {measurements[0,:,0]}")
            print(f"    Can_measure[0,:,0] (XX measurability): {can_measure[0,:,0]}")
        
        for b_idx in range(batch_size):
            hits = can_measure[b_idx].sum(dim=0)
            sums = measurements[b_idx].sum(dim=0)
            
            if self.debug and b_idx == 0:
                print(f"    Batch {b_idx} - hits: {hits}")
                print(f"    Batch {b_idx} - sums: {sums}")
            
            pauli_estimates = {}
            hitting_counts = {}
            
            for p_idx, p_str in enumerate(self.pauli_strings):
                count = hits[p_idx].item()
                hitting_counts[p_str] = count
                
                if count == 0:
                    pauli_estimates[p_str] = 0.0
                    if p_str in self.pauli_to_coeff:
                        warnings.warn(f"Pauli {p_str} was never measured (N_P = 0)")
                else:
                    pauli_estimates[p_str] = float(sums[p_idx].item()) / float(count)
            
            batch_pauli_estimates.append(pauli_estimates)
            batch_hitting_counts.append(hitting_counts)
        
        return batch_pauli_estimates, batch_hitting_counts

    @torch.no_grad()
    def _prepare_circuits_for_sampling(self, batch_actions: torch.Tensor,
                                        batch_lengths: torch.Tensor,
                                        ground_state: torch.Tensor) -> PreparedCircuitData:
        """Prepare Pauli transformation data for efficient i.i.d. sampling."""
        batch_size, n_circuits, _ = batch_actions.shape
        n_paulis = len(self.pauli_strings)

        clifford_map = self._get_or_create_clifford_map(batch_size, n_circuits)
        self._apply_circuits_to_map(clifford_map, batch_actions, batch_lengths)
        can_measure = clifford_map.prob_P_multi(self.pauli_strings)

        x_in = self.pauli_vecs[:, :self.n_qubits]
        z_in = self.pauli_vecs[:, self.n_qubits:]
        pauli_in = Pauli(x_in, z_in, self.pauli_phases)
        pauli_out = pauli_in.apply_clifford(clifford_map)
        p_out = torch.cat([pauli_out.x, pauli_out.z], dim=-1).byte()
        phases = pauli_out.phase

        meas = can_measure.bool()
        odd_phase = (phases & 1).bool()
        if (odd_phase & meas).any():
            raise AssertionError("Odd phase detected on diagonal (measurable) operator")

        signs = self._phases_to_signs(phases)
        z_parts = p_out[:, :, :, self.n_qubits:]
        z_masks = self._compute_z_masks_vectorized(z_parts)
        hits = can_measure.sum(dim=1)

        unique_z_masks, n_unique_per_circuit, pauli_to_unique_idx, max_unique_masks = \
            self._compute_unique_z_mask_mapping(z_masks)

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
    def _sample_energies_from_prepared(self, prepared: PreparedCircuitData,
                                        ground_state: torch.Tensor,
                                        M: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample M i.i.d. measurement outcomes and compute energy estimates."""
        import gc
        
        batch_size = prepared.batch_size
        n_circuits = prepared.n_circuits
        n_paulis = prepared.n_paulis
        
        states = self._apply_circuits_to_states(
            ground_state, prepared.batch_actions, prepared.batch_lengths
        )
        probs = torch.abs(states) ** 2
        del states
        
        probs_flat = probs.reshape(-1, self.dim)
        outcomes_all_flat = torch.multinomial(probs_flat, M, replacement=True)
        outcomes_all = outcomes_all_flat.reshape(batch_size, n_circuits, M)
        
        del probs, probs_flat, outcomes_all_flat
        gc.collect()
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        
        use_unique_optimization = (prepared.unique_z_masks is not None and 
                                   prepared.max_unique_masks is not None and
                                   prepared.max_unique_masks < n_paulis)
        
        if use_unique_optimization:
            bytes_per_sim = batch_size * n_circuits * prepared.max_unique_masks * 4
        else:
            bytes_per_sim = batch_size * n_circuits * n_paulis * 4
        
        target_bytes = 500 * 1024 * 1024
        chunk_size = max(1, min(M, target_bytes // max(bytes_per_sim, 1)))
        
        estimates_all = torch.zeros(batch_size, M, n_paulis, device=self.device, dtype=torch.float32)
        energy_estimates = torch.zeros(batch_size, M, device=self.device, dtype=torch.float32)
        hits = prepared.hits
        
        for m_start in range(0, M, chunk_size):
            m_end = min(m_start + chunk_size, M)
            m_chunk = m_end - m_start
            outcomes_chunk = outcomes_all[:, :, m_start:m_end]
            
            if use_unique_optimization:
                eigenvalues = self._compute_eigenvalues_with_unique_masks(outcomes_chunk, prepared)
            else:
                eigenvalues = self._compute_eigenvalues_full(outcomes_chunk, prepared.z_masks)
            
            signs_expanded = prepared.signs.unsqueeze(2)
            can_measure_expanded = prepared.can_measure.unsqueeze(2)
            
            eigenvalues.mul_(signs_expanded)
            eigenvalues.mul_(can_measure_expanded.float())
            
            sums_chunk = eigenvalues.sum(dim=1)
            del eigenvalues
            
            hits_expanded = hits.unsqueeze(1)
            estimates_chunk = torch.where(
                hits_expanded > 0,
                sums_chunk / hits_expanded.float(),
                torch.zeros_like(sums_chunk)
            )
            del sums_chunk
            
            estimates_all[:, m_start:m_end, :] = estimates_chunk
            energy_contrib = estimates_chunk * self._pauli_coeffs_tensor.view(1, 1, -1)
            energy_estimates[:, m_start:m_end] = self.identity_weight + energy_contrib.sum(dim=-1)
            del estimates_chunk, energy_contrib
        
        del outcomes_all
        gc.collect()
        
        return energy_estimates, estimates_all

    @torch.no_grad()
    def _estimate_energy(self, batch_actions: torch.Tensor,
                         batch_lengths: torch.Tensor,
                         ground_state: torch.Tensor,
                         M: int = 1) -> List[List[BatchElementEnergyResult]]:
        """Unified energy estimation with M i.i.d. simulations."""
        batch_size, n_circuits, _ = batch_actions.shape
        n_paulis = len(self.pauli_strings)

        prepared = self._prepare_circuits_for_sampling(batch_actions, batch_lengths, ground_state)
        energy_estimates, estimates_all = self._sample_energies_from_prepared(prepared, ground_state, M)

        batch_lengths_cpu = batch_lengths.cpu().numpy()
        hits_cpu = prepared.hits.cpu().numpy()
        estimates_all_cpu = estimates_all.cpu().numpy()
        energy_estimates_cpu = energy_estimates.cpu().numpy()

        batch_simulation_results = []
        for b_idx in range(batch_size):
            c_lens = batch_lengths_cpu[b_idx].tolist()
            hitting_counts_dict = {
                self.pauli_strings[p_idx]: int(hits_cpu[b_idx, p_idx])
                for p_idx in range(n_paulis)
            }

            for p_idx, p_str in enumerate(self.pauli_strings):
                if hits_cpu[b_idx, p_idx] == 0 and p_str in self.pauli_to_coeff:
                    warnings.warn(f"Pauli {p_str} was never measured (N_P = 0)")

            measured_paulis = [p for p, n in hitting_counts_dict.items() if n > 0]
            hitting_values = [n for n in hitting_counts_dict.values() if n > 0]

            results_for_batch = []
            for m in range(M):
                pauli_estimates_dict = {
                    self.pauli_strings[p_idx]: float(estimates_all_cpu[b_idx, m, p_idx])
                    for p_idx in range(n_paulis)
                }

                result = BatchElementEnergyResult(
                    update=0,
                    batch_element_rank=b_idx,
                    n_circuits=n_circuits,
                    total_measurements=n_circuits,
                    energy_estimate=float(energy_estimates_cpu[b_idx, m]),
                    energy_difference=abs(float(energy_estimates_cpu[b_idx, m]) - self.ground_state_energy),
                    pauli_estimates=pauli_estimates_dict,
                    hitting_counts=hitting_counts_dict,
                    circuit_lengths=c_lens,
                    mean_circuit_length=np.mean(c_lens) if c_lens else 0,
                    batch_cost=0.0,
                    convergence_metrics={
                        'coverage': len(measured_paulis) / len(self.pauli_strings) if self.pauli_strings else 0,
                        'avg_hitting_count': np.mean(hitting_values) if hitting_values else 0,
                    }
                )
                results_for_batch.append(result)

            batch_simulation_results.append(results_for_batch)

        return batch_simulation_results

    def _compute_z_masks_vectorized(self, z_parts: torch.Tensor) -> torch.Tensor:
        """Compute Z masks using vectorized bit operations."""
        z_masks = (z_parts.long() << self._bit_shifts_view).sum(dim=-1)
        return z_masks

    def _compute_parities_vectorized(self, masked_outcomes: torch.Tensor) -> torch.Tensor:
        """Compute parities using vectorized bit counting."""
        if hasattr(torch, 'bit_count'):
            parities = torch.bit_count(masked_outcomes.long()).to(torch.long) % 2
        else:
            bit_positions = torch.arange(self.n_qubits, device=masked_outcomes.device, dtype=torch.long)
            bit_masks = (masked_outcomes.unsqueeze(-1) >> bit_positions) & 1
            parities = bit_masks.sum(dim=-1) % 2
        return parities

    @torch.no_grad()
    def _compute_unique_z_mask_mapping(self, z_masks: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """Compute unique z_mask values and mapping for memory-efficient eigenvalue computation."""
        batch_size, n_circuits, n_paulis = z_masks.shape
        device = z_masks.device
        
        max_unique = 0
        for b in range(batch_size):
            for c in range(n_circuits):
                n_unique = torch.unique(z_masks[b, c]).numel()
                max_unique = max(max_unique, n_unique)
        
        unique_z_masks = torch.zeros(batch_size, n_circuits, max_unique, dtype=z_masks.dtype, device=device)
        n_unique_per_circuit = torch.zeros(batch_size, n_circuits, dtype=torch.long, device=device)
        pauli_to_unique_idx = torch.zeros(batch_size, n_circuits, n_paulis, dtype=torch.long, device=device)
        
        for b in range(batch_size):
            for c in range(n_circuits):
                unique_vals, inverse_idx = torch.unique(z_masks[b, c], return_inverse=True)
                n_unique = unique_vals.numel()
                unique_z_masks[b, c, :n_unique] = unique_vals
                n_unique_per_circuit[b, c] = n_unique
                pauli_to_unique_idx[b, c] = inverse_idx
        
        if self.debug:
            total_paulis = batch_size * n_circuits * n_paulis
            total_unique = n_unique_per_circuit.sum().item()
            reduction = total_paulis / total_unique if total_unique > 0 else 1.0
            logging.debug(f"Unique z_mask optimization: {reduction:.1f}x reduction")
        
        return unique_z_masks, n_unique_per_circuit, pauli_to_unique_idx, max_unique

    @torch.no_grad()
    def _compute_eigenvalues_full(self, outcomes_chunk: torch.Tensor, 
                                   z_masks: torch.Tensor) -> torch.Tensor:
        """Compute eigenvalues using full z_masks."""
        outcomes_expanded = outcomes_chunk.unsqueeze(3)
        z_masks_expanded = z_masks.unsqueeze(2)
        masked = outcomes_expanded & z_masks_expanded
        parities = self._compute_parities_vectorized(masked)
        del masked
        eigenvalues = parities.float()
        eigenvalues.mul_(-2.0).add_(1.0)
        del parities
        return eigenvalues

    @torch.no_grad()
    def _compute_eigenvalues_with_unique_masks(self, outcomes_chunk: torch.Tensor,
                                                prepared: 'PreparedCircuitData') -> torch.Tensor:
        """Compute eigenvalues using unique z_masks for memory efficiency."""
        batch_size, n_circuits, m_chunk = outcomes_chunk.shape
        n_paulis = prepared.n_paulis
        max_unique = prepared.max_unique_masks
        
        outcomes_expanded = outcomes_chunk.unsqueeze(3)
        unique_masks_expanded = prepared.unique_z_masks.unsqueeze(2)
        
        masked_unique = outcomes_expanded & unique_masks_expanded
        parities_unique = self._compute_parities_vectorized(masked_unique)
        del masked_unique
        
        eigenvalues_unique = parities_unique.float()
        eigenvalues_unique.mul_(-2.0).add_(1.0)
        del parities_unique
        
        idx_expanded = prepared.pauli_to_unique_idx.unsqueeze(2).expand(-1, -1, m_chunk, -1)
        eigenvalues = torch.gather(eigenvalues_unique, dim=3, index=idx_expanded)
        del eigenvalues_unique, idx_expanded
        
        return eigenvalues

    @torch.no_grad()
    def prepare_circuits(self, batch_actions: torch.Tensor,
                         batch_lengths: torch.Tensor) -> PreparedCircuitData:
        """Prepare Pauli transformation data for efficient repeated sampling."""
        batch_actions = batch_actions.to(self.device)
        batch_lengths = batch_lengths.to(self.device)
        ground_state = self.get_ground_state()
        return self._prepare_circuits_for_sampling(batch_actions, batch_lengths, ground_state)
    
    @torch.no_grad()
    def sample_from_prepared(self, prepared: PreparedCircuitData,
                             M: int = 1,
                             ground_state: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample M i.i.d. energy estimates from prepared circuit data."""
        if ground_state is None:
            ground_state = self.get_ground_state()
        ground_state = ground_state.to(self.device)
        return self._sample_energies_from_prepared(prepared, ground_state, M)

    @torch.no_grad()
    def estimate_energy_with_simulations(self, batch_actions: torch.Tensor, 
                                         batch_lengths: torch.Tensor,
                                         M: int = 1, 
                                         ground_state: Optional[torch.Tensor] = None):
        """Estimate energy with M simulation runs."""
        batch_size = batch_actions.shape[0]
        if M <= 0: 
            raise ValueError("M must be positive")
        
        batch_actions = batch_actions.to(self.device)
        batch_lengths = batch_lengths.to(self.device)
        ground_state = self.get_ground_state() if ground_state is None else ground_state
        ground_state = ground_state.to(self.device)

        batch_simulation_results = self._estimate_energy(
            batch_actions, batch_lengths, ground_state, M)
        
        final_summaries = []
        for b_idx in range(batch_size):
            results_for_el = batch_simulation_results[b_idx]
            energies = [res.energy_estimate for res in results_for_el]
            mean_energy = sum(energies) / M
            
            agg_pauli_estimates = defaultdict(list)
            for res in results_for_el:
                for pauli, val in res.pauli_estimates.items():
                    agg_pauli_estimates[pauli].append(val)
            
            individual_absolute_errors = [abs(e - self.ground_state_energy) for e in energies]
            individual_squared_errors = [(e - self.ground_state_energy) ** 2 for e in energies]
            rmse = np.sqrt(np.mean(individual_squared_errors))

            summary = {
                'batch_index': b_idx,
                'mean_energy': mean_energy,
                'energy_variance': np.var(energies, ddof=1) if M > 1 else 0.0,
                'rmse': rmse,
                'energy_difference': rmse,
                'std_absolute_error': np.std(individual_absolute_errors, ddof=1) if M > 1 else 0.0,
                'num_simulations': M,
                'mean_pauli_estimates': {p: sum(v) / M for p, v in agg_pauli_estimates.items()},
                'final_results_object': results_for_el[-1],
                'individual_energies': energies,
                'individual_absolute_errors': individual_absolute_errors,
                'individual_squared_errors': individual_squared_errors
            }
            final_summaries.append(summary)
            
        return final_summaries

    def cleanup_gpu_memory(self, clear_cache: bool = False):
        """Release GPU memory after large computations."""
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
                           force_cpu: bool = True) -> 'EnergyEstimator':
        """Create an EnergyEstimator instance from checkpoint data."""
        device = 'cpu' if force_cpu else None
        estimator = cls(hamiltonian_helper, n_qubits, device=device, force_cpu=force_cpu)
        
        if 'action_mapping' in checkpoint_data:
            estimator.action_map = checkpoint_data['action_mapping']
            logging.info("Loaded action mapping from checkpoint")
            
        if 'terminal_index' in checkpoint_data:
            estimator.terminal_action = checkpoint_data['terminal_index']
            logging.info(f"Loaded terminal action index: {estimator.terminal_action}")
            
        return estimator


def debug_single_measurement():
    """Debug a single measurement for testing."""
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