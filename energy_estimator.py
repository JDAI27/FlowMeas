# -*- coding: utf-8 -*-
"""
Energy estimator for quantum circuits using derandomized measurement circuits.
Implements energy estimation following the classical shadow tomography protocol.

The protocol is implemented as follows:
1. Given a batch of circuit sets {U_i}, determine which Pauli terms each circuit can measure.
2. Prepare the ground state |ψ⟩ and apply each circuit U_i.
3. Measure in the computational basis to get ONE outcome |b_i⟩ per circuit.
4. Estimate ⟨b_{i,j}|P|b_{i,j}⟩ for each Pauli P for each element in the batch.
5. Compute the total energy E_j = ∑_P w_P ⟨b_{i,j}|P|b_{i,j}⟩ for each element in the batch.
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from collections import defaultdict
import warnings

import math
from clifford_map import CliffordMap
from quantum_action_mapping import build_action_mapping
from gf2_ops import GF2Ops

from pauli_hamiltonian_helper import PauliHamiltonianHelper

# --------------------------------------------------------------------

@dataclass
class BatchElementEnergyResult:
    update: int; batch_element_rank: int; n_circuits: int; total_measurements: int
    energy_estimate: float; energy_difference: float; pauli_estimates: Dict[str, float]
    hitting_counts: Dict[str, int]; circuit_lengths: List[int]; mean_circuit_length: float
    batch_cost: float; convergence_metrics: Dict

class EnergyEstimator:
    def __init__(self, hamiltonian_helper: 'PauliHamiltonianHelper', n_qubits: int, device: torch.device, debug: bool = False):
        self.hamiltonian_helper = hamiltonian_helper
        self.n_qubits = n_qubits
        self.device = device
        self.debug = debug
        self.ground_state_energy = hamiltonian_helper.ground_state_energy
        self.all_pauli_strings = hamiltonian_helper.pauli_str_list
        self.all_pauli_coeffs = [w.real for w in hamiltonian_helper.w_list]
        
        # Handle identity weight
        identity_str = "I" * n_qubits
        self.identity_weight = self.all_pauli_coeffs[self.all_pauli_strings.index(identity_str)] if identity_str in self.all_pauli_strings else 0.0
        self.pauli_strings = [p for p in self.all_pauli_strings if p != identity_str]
        self.pauli_to_coeff = {p: c for p, c in zip(self.all_pauli_strings, self.all_pauli_coeffs) if p != identity_str}
        
        # Build and store the action mapping
        self.action_map, self.terminal_action = build_action_mapping(self.n_qubits)
        
        # Convert Pauli strings to symplectic representation
        self.pauli_vecs = self._pauli_string_to_symplectic_vectorized(self.pauli_strings)
        self.pauli_phases = self._get_pauli_phases(self.pauli_strings)
        
        # Setup gates and precompute masks
        self._setup_torch_quantum_gates()
        self._precompute_masks()
        
    def _pauli_string_to_symplectic_vectorized(self, p_strs: List[str]) -> torch.Tensor:
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
    
    def _get_pauli_phases(self, p_strs: List[str]) -> torch.Tensor:
        """Get the initial phases of Pauli strings (accounting for Y = iXZ)."""
        phases = torch.zeros(len(p_strs), dtype=torch.int8, device=self.device)
        for i, s in enumerate(p_strs):
            # Count Y operators, each contributes a factor of i (phase 1)
            y_count = s.count('Y')
            phases[i] = y_count % 4
            
        if self.debug:
            print(f"\n[DEBUG] Initial Pauli phases:")
            for i, (p_str, phase) in enumerate(zip(p_strs, phases)):
                print(f"  {p_str}: phase = {phase}")
                
        return phases

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
        
    def _precompute_masks(self):
        """Precompute masks for efficient bitwise operations."""
        self.dim = 2 ** self.n_qubits
        self.basis = torch.arange(self.dim, device=self.device, dtype=torch.long)
        
        # Precompute qubit masks for CNOT operations
        self.qubit_masks = torch.zeros(self.n_qubits, dtype=torch.long, device=self.device)
        for i in range(self.n_qubits):
            self.qubit_masks[i] = 1 << (self.n_qubits - 1 - i)
    
    def get_ground_state(self) -> torch.Tensor:
        """Get the ground state vector."""
        if hasattr(self.hamiltonian_helper, 'ground_state_vector'):
            return torch.tensor(self.hamiltonian_helper.ground_state_vector, 
                              dtype=torch.complex64, device=self.device)
        raise ValueError("Hamiltonian helper does not have a ground state vector.")
    
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
    def _apply_circuits_to_states_vectorized(self, states: torch.Tensor, 
                                            batch_actions: torch.Tensor, 
                                            batch_lengths: torch.Tensor) -> torch.Tensor:
        """
        Vectorized application of circuits to quantum states.
        NOTE: Gates are applied in reverse order after finding terminal action.
        
        Args:
            states: Shape (batch_size, n_circuits, 2^n) or (2^n) for single state
            batch_actions: Shape (batch_size, n_circuits, max_length)
            batch_lengths: Shape (batch_size, n_circuits)
            
        Returns:
            Transformed states with shape (batch_size, n_circuits, 2^n)
        """
        batch_size, n_circuits, max_length = batch_actions.shape
        
        # Broadcast state if needed
        if states.dim() == 1:
            states = states.unsqueeze(0).unsqueeze(0).expand(batch_size, n_circuits, -1)
        elif states.dim() == 2:
            states = states.unsqueeze(1).expand(-1, n_circuits, -1)
            
        # Clone to avoid modifying input
        states = states.clone()
        
        # Find terminal positions (vectorized)
        terminal_mask = batch_actions == self.terminal_action
        positions = torch.arange(max_length, device=self.device).unsqueeze(0).unsqueeze(0)
        terminal_positions = torch.where(terminal_mask, positions, max_length).min(dim=2)[0]
        
        # Gates are applied from position terminal_pos-1 down to 0
        # If no terminal found, no gates are applied (matching original behavior)
        
        if self.debug:
            print(f"\n[DEBUG] _apply_circuits_to_states_vectorized:")
            print(f"  terminal_positions: {terminal_positions}")
            print(f"  batch_lengths: {batch_lengths}")
        
        # Apply gates in reverse order
        for step in range(max_length-1, -1, -1):
            # Check which circuits should apply gate at this step
            # Gate is applied if step < terminal_position (found terminal after this step)
            step_active = step < terminal_positions
            if not step_active.any():
                continue
                
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
                
                if self.debug and action_mask[0, 4]:  # Debug H circuit
                    print(f"  Applying {gate_name} on qubits {qubits} at step {step}")
                
                if gate_name == "CNOT":
                    # Vectorized CNOT application
                    control, target = qubits[0], qubits[1]
                    c_mask = self.qubit_masks[control]
                    t_mask = self.qubit_masks[target]
                    
                    # Apply CNOT only where action_mask is True
                    # Get the affected states
                    affected_states = states[action_mask]  # Shape: (N, 2^n)
                    
                    # Create basis for each affected state
                    basis_expanded = self.basis.unsqueeze(0).expand(affected_states.shape[0], -1)
                    control_set = (basis_expanded & c_mask) != 0
                    new_basis = torch.where(control_set, basis_expanded ^ t_mask, basis_expanded)
                    
                    # Gather to apply CNOT
                    states[action_mask] = affected_states.gather(-1, new_basis)
                    
                else:
                    # Single-qubit gates
                    qubit = qubits[0]
                    gate = self.torch_gates[gate_name]
                    
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
    
    def _apply_circuits_to_map(self, clifford_map: 'CliffordMap', 
                              batch_actions: torch.Tensor, 
                              batch_lengths: torch.Tensor):
        """Apply all circuits in the batch to the Clifford tableau."""
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
    def _compute_pauli_phases_vectorized(self, clifford_map: 'CliffordMap', 
                                       p_out: torch.Tensor) -> torch.Tensor:
        """
        Vectorized computation of Pauli phases.
        
        Returns:
            phases: Shape (batch_size, n_circuits, n_paulis) with values in {0,1,2,3}
        """
        batch_size, n_circuits, n_paulis, _ = p_out.shape
        n = self.n_qubits
        
        # Initial phases from Pauli strings
        phases = self.pauli_phases.unsqueeze(0).unsqueeze(0).expand(batch_size, n_circuits, -1)
        
        # Linear contribution: Σ_i v'ᵢ φ_i
        # Only sum phase_vec elements where p_out is True (matching original logic)
        # Expand phase_vec for broadcasting
        phase_vec_expanded = clifford_map.heis_phase_vec.unsqueeze(2)  # (batch, circuits, 1, 2n)
        
        # Mask phase_vec with p_out and sum
        masked_phases = phase_vec_expanded * p_out.float()  # Element-wise multiply
        linear_contrib = masked_phases.sum(dim=-1)  # Sum over the 2n dimension
        
        # NOTE: Quadratic contribution commented out in original code
        # The CliffordMap implementation doesn't include heis_quad_mask
        phases = (phases + linear_contrib).long() % 4
        
        if self.debug:
            print(f"\n[DEBUG] _compute_pauli_phases_vectorized:")
            print(f"  initial phases: {self.pauli_phases}")
            print(f"  phase_vec[0,0]: {clifford_map.heis_phase_vec[0,0]}")
            print(f"  phase_vec[0,4] (H circuit): {clifford_map.heis_phase_vec[0,4]}")
            print(f"  computed phases[0,0]: {phases[0,0]}")
            
            # Check the linear contribution calculation
            print(f"\n  Linear contribution details for H circuit [0,4]:")
            for p_idx in range(min(3, n_paulis)):
                p_str = self.pauli_strings[p_idx]
                v_prime = p_out[0, 4, p_idx]
                phase_vec = clifford_map.heis_phase_vec[0, 4]
                
                # Manual dot product
                manual_contrib = 0
                for i in range(len(v_prime)):
                    if v_prime[i]:
                        manual_contrib += phase_vec[i].item()
                
                vec_contrib = linear_contrib[0, 4, p_idx].item()
                print(f"    {p_str}: v'={v_prime.cpu().numpy()}, "
                      f"manual_contrib={manual_contrib}, vec_contrib={vec_contrib}")
            
            # Manual calculation for verification
            for p_idx in range(min(3, n_paulis)):
                k = self.pauli_phases[p_idx].item()
                v_prime = p_out[0, 4, p_idx]  # H circuit
                p_vec = clifford_map.heis_phase_vec[0, 4]
                contrib = (p_vec[v_prime.bool()].sum().item()) if v_prime.bool().any() else 0
                manual_phase = (k + contrib) % 4
                print(f"  Manual phase calc for Pauli {p_idx}: init={k}, contrib={contrib}, final={manual_phase}")
                print(f"  Vectorized phase: {phases[0,4,p_idx].item()}")
        
        return phases

    @torch.no_grad()
    def _compute_pauli_expectations_vectorized(self, clifford_map: 'CliffordMap', 
                                             ground_state: torch.Tensor,
                                             batch_actions: torch.Tensor, 
                                             batch_lengths: torch.Tensor) -> Tuple[List[Dict], List[Dict]]:
        """
        Fully vectorized computation of Pauli expectations following shadow tomography.
        """
        batch_size, n_circuits, _ = batch_actions.shape
        n_paulis = len(self.pauli_strings)
        
        # Get measurement capability
        can_measure = clifford_map.prob_P_multi(self.pauli_strings)  # (batch, circuits, paulis)
        
        # Compute transformed Pauli vectors
        W = clifford_map.W
        W_inverse = clifford_map.W_inv #GF2Ops.invert_matrix(W)
        p_out = torch.einsum('bcij,pj->bcpi', W_inverse.byte(), self.pauli_vecs.byte()) % 2
        
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
            print(f"    {W[0,0]}")
            print(f"    W matrix for H circuit [0,4]:")  
            print(f"    {W[0,4]}")
        
        # Apply circuits to states vectorized
        states = self._apply_circuits_to_states_vectorized(ground_state, batch_actions, batch_lengths)
        
        # Debug: Compare with non-vectorized version for first circuit
        if self.debug:
            debug_state = self._apply_circuit_to_state_debug(
                ground_state, batch_actions[0, 0], batch_lengths[0, 0]
            )
            print(f"\n[DEBUG] State comparison for circuit[0,0]:")
            print(f"  Vectorized state: {states[0,0]}")
            print(f"  Debug state: {debug_state}")
            print(f"  States match: {torch.allclose(states[0,0], debug_state)}")
        
        probs = torch.abs(states) ** 2  # Shape: (batch, circuits, 2^n)
        
        # Sample outcomes for all circuits at once
        outcomes = torch.multinomial(probs.reshape(-1, self.dim), 1, replacement=True)
        outcomes = outcomes.reshape(batch_size, n_circuits)  # Shape: (batch, circuits)
        
        if self.debug:
            print(f"\n[DEBUG] _compute_pauli_expectations_vectorized:")
            print(f"  can_measure[0,4:8]: {can_measure[0,4:8]}")  # H circuits
            print(f"  outcomes[0]: {outcomes[0]}")
            print(f"  outcomes binary: {[f'{o:02b}' for o in outcomes[0]]}")
            print(f"  probs[0,0]: {probs[0,0]}")
            print(f"  probs[0,4]: {probs[0,4]}")  # H circuit probabilities
        
        # Compute phases for all Paulis
        phases = self._compute_pauli_phases_vectorized(clifford_map, p_out)
        
        # Convert phases to signs (matching original code's convention)
        # Phase 0 → +1, Phase 1 → -1, Phase 2 → -1, Phase 3 → +1
        # But in the original: phases 0,3 → +1 and phases 1,2 → -1
        signs = torch.where((phases == 0) | (phases == 3), 1.0, -1.0)
        
        # Compute parities efficiently
        # We need to use the transformed Pauli's Z part, not the original
        # Extract Z parts from p_out
        z_parts = p_out[:, :, :, self.n_qubits:]  # Shape: (batch, circuits, paulis, n)
        
        # Compute Z masks for each transformed Pauli
        z_masks = torch.zeros(batch_size, n_circuits, n_paulis, dtype=torch.long, device=self.device)
        for i in range(self.n_qubits):
            z_masks += (z_parts[:, :, :, i].long() << (self.n_qubits - 1 - i))
        
        # Expand outcomes for broadcasting
        outcomes_expanded = outcomes.unsqueeze(2)  # (batch, circuits, 1)
        
        # Compute masked outcomes
        masked_outcomes = outcomes_expanded & z_masks  # (batch, circuits, paulis)
        
        # Count set bits for parity
        parities = torch.zeros_like(masked_outcomes, dtype=torch.float32)
        for i in range(self.n_qubits):
            parities += ((masked_outcomes >> i) & 1).float()
        parities = parities.long() % 2
        
        # Compute eigenvalues
        eigenvalues = 1.0 - 2.0 * parities.float()  # Shape: (batch, circuits, paulis)
        
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
            
            # Manual verification
            outcome_val = outcomes[0,4].item()
            z_mask_val = z_masks[0,4,0].item()
            manual_masked = outcome_val & z_mask_val
            manual_parity = bin(manual_masked).count('1') % 2
            manual_eigenvalue = 1.0 - 2.0 * manual_parity
            print(f"    Manual calculation: outcome={outcome_val}, mask={z_mask_val}, "
                  f"masked={manual_masked}, parity={manual_parity}, eigenvalue={manual_eigenvalue}")
            
            # Check what the original Bell state should give
            print(f"\n  Expected behavior for Bell state |ψ-⟩:")
            print(f"    After H⊗H: (-|01⟩ + |10⟩)/√2")
            print(f"    Possible outcomes: 1 (|01⟩) or 2 (|10⟩)")
            print(f"    ZZ on |01⟩: eigenvalue = -1")
            print(f"    ZZ on |10⟩: eigenvalue = -1")
            print(f"    Expected XX measurement via H circuit: -1")
            
            # Debug bit ordering
            print(f"\n  Bit ordering check:")
            for i in range(4):
                print(f"    State {i} = |{i:02b}⟩, ZZ eigenvalue = {1 - 2*(bin(i & 3).count('1') % 2)}")
            
            # Manual check for specific measurement
            if outcomes[0,4] == 1:  # If we got |01⟩
                print(f"\n  Manual check: Got outcome |01⟩ from H circuit")
                print(f"    ZZ eigenvalue should be: -1")
                print(f"    Initial phase of XX: 0")
                print(f"    Phase after H⊗H transform: {phases[0,4,0].item()}")
                print(f"    Sign from phase: {signs[0,4,0].item()}")
                print(f"    Final measurement: {signs[0,4,0].item()} * (-1) = {measurements[0,4,0].item()}")
        
        # Accumulate results per batch element
        batch_pauli_estimates = []
        batch_hitting_counts = []
        
        if self.debug:
            print(f"\n  Accumulation debug:")
            print(f"    Total measurements shape: {measurements.shape}")
            print(f"    Measurements[0,:,0] (XX for all circuits): {measurements[0,:,0]}")
            print(f"    Can_measure[0,:,0] (XX measurability): {can_measure[0,:,0]}")
        
        for b_idx in range(batch_size):
            # Count hits per Pauli
            hits = can_measure[b_idx].sum(dim=0)  # Shape: (paulis,)
            
            # Sum measurements per Pauli
            sums = measurements[b_idx].sum(dim=0)  # Shape: (paulis,)
            
            if self.debug and b_idx == 0:
                print(f"    Batch {b_idx} - hits: {hits}")
                print(f"    Batch {b_idx} - sums: {sums}")
                print(f"    XX: sum={sums[0]}, count={hits[0]}, estimate={sums[0]/hits[0] if hits[0] > 0 else 0}")
            
            # Compute estimates
            pauli_estimates = {}
            hitting_counts = {}
            
            for p_idx, p_str in enumerate(self.pauli_strings):
                count = hits[p_idx].item()
                hitting_counts[p_str] = count
                
                if count == 0:
                    pauli_estimates[p_str] = 0.0
                    if p_str in self.pauli_to_coeff:  # Only warn for Hamiltonian terms
                        warnings.warn(f"Pauli {p_str} was never measured (N_P = 0)")
                else:
                    # Use same type conversion as original
                    pauli_estimates[p_str] = float(sums[p_idx].item()) / float(count)
            
            batch_pauli_estimates.append(pauli_estimates)
            batch_hitting_counts.append(hitting_counts)
        
        return batch_pauli_estimates, batch_hitting_counts

    async def _estimate_energy_single_run(self, batch_actions: torch.Tensor, 
                                              batch_lengths: torch.Tensor,
                                              ground_state: torch.Tensor) -> List[BatchElementEnergyResult]:
        """Single run of energy estimation for the batch."""
        batch_size, n_circuits, _ = batch_actions.shape
        
        # Create Clifford map
        clifford_map = CliffordMap(self.n_qubits, batch_size, n_circuits, str(self.device))
        self._apply_circuits_to_map(clifford_map, batch_actions, batch_lengths)
        
        # Compute Pauli expectations (vectorized)
        batch_pauli_estimates, batch_hitting_counts = self._compute_pauli_expectations_vectorized(
            clifford_map, ground_state, batch_actions, batch_lengths)
        
        # Create results
        batch_results = []
        for b_idx in range(batch_size):
            pauli_estimates = batch_pauli_estimates[b_idx]
            hitting_counts = batch_hitting_counts[b_idx]
            
            # Compute energy estimate
            energy_estimate = self.identity_weight + sum(
                self.pauli_to_coeff[p] * v for p, v in pauli_estimates.items()
            )
            
            # Compute metrics
            c_lens = batch_lengths[b_idx].cpu().numpy().tolist()
            measured_paulis = [p for p, n in hitting_counts.items() if n > 0]
            hitting_values = [n for n in hitting_counts.values() if n > 0]
            
            result = BatchElementEnergyResult(
                update=0, 
                batch_element_rank=b_idx, 
                n_circuits=n_circuits,
                total_measurements=n_circuits,  # One measurement per circuit
                energy_estimate=energy_estimate, 
                energy_difference=abs(energy_estimate - self.ground_state_energy),
                pauli_estimates=pauli_estimates, 
                hitting_counts=hitting_counts, 
                circuit_lengths=c_lens,
                mean_circuit_length=np.mean(c_lens) if c_lens else 0, 
                batch_cost=0.0,
                convergence_metrics={
                    'coverage': len(measured_paulis) / len(self.pauli_strings) if self.pauli_strings else 0,
                    'avg_hitting_count': np.mean(hitting_values) if hitting_values else 0,
                })
            batch_results.append(result)
            
        return batch_results

    async def estimate_energy_with_simulations(self, batch_actions: torch.Tensor, 
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
            
        ground_state = self.get_ground_state() if ground_state is None else ground_state
        
        # Run M simulations
        batch_simulation_results = [[] for _ in range(batch_size)]
        for _ in range(M):
            single_run_results = await self._estimate_energy_single_run(
                batch_actions, batch_lengths, ground_state)
            for b_idx in range(batch_size):
                batch_simulation_results[b_idx].append(single_run_results[b_idx])
        
        # Aggregate results
        final_summaries = []
        for b_idx in range(batch_size):
            results_for_el = batch_simulation_results[b_idx]
            energies = [res.energy_estimate for res in results_for_el]
            mean_energy = sum(energies) / M
            
            # Aggregate Pauli estimates
            agg_pauli_estimates = defaultdict(list)
            for res in results_for_el:
                for pauli, val in res.pauli_estimates.items():
                    agg_pauli_estimates[pauli].append(val)
            
            summary = {
                'batch_index': b_idx, 
                'mean_energy': mean_energy,
                'energy_variance': np.var(energies, ddof=1) if M > 1 else 0.0,
                'energy_difference': sum([abs(e - self.ground_state_energy) for e in energies]) / M,
                'num_simulations': M,
                'mean_pauli_estimates': {p: sum(v) / M for p, v in agg_pauli_estimates.items()},
                'final_results_object': results_for_el[-1]
            }
            final_summaries.append(summary)
            
        return final_summaries


# Separate debug test function outside of main block
async def debug_single_measurement():
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
    summaries = await estimator.estimate_energy_with_simulations(
        batch_actions=batch_actions, batch_lengths=batch_lengths, M=1
    )
    
    result = summaries[0]
    print(f"\nActual XX estimate: {result['mean_pauli_estimates'].get('XX', 0.0)}")
    print(f"Energy estimate: {result['mean_energy']:.6f}")
