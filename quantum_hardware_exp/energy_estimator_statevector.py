#!/usr/bin/env python3
"""
Energy Estimator using Qiskit Statevector Simulations

This module provides energy estimation methods for quantum systems:
1. Exact statevector computation: ⟨ψ|H|ψ⟩
2. Qiskit Estimator with configurable shot budgets

Compatible with PauliHamiltonianHelper from the main codebase.

Author: Quantum Hardware Experiments
"""

from __future__ import annotations

import sys
import json
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Union, Any
from dataclasses import dataclass

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, SparsePauliOp, Pauli, Operator
from qiskit.primitives import StatevectorEstimator

# Add code directory to path for imports
CODE_DIR = Path(__file__).parent.parent / "code"
if CODE_DIR.exists() and str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))


@dataclass
class EnergyResult:
    """Container for energy estimation results."""
    energy: float
    exact_energy: Optional[float] = None
    error: Optional[float] = None
    std_error: Optional[float] = None
    n_shots: Optional[int] = None
    method: str = "statevector"
    pauli_expectations: Optional[Dict[str, float]] = None


@dataclass
class SimulationStats:
    """Statistics from multiple independent simulation runs."""
    energies: List[float]
    mean_energy: float
    std_energy: float
    rmse: float
    bias: float
    exact_energy: float
    n_runs: int
    n_circuits: int
    
    def summary(self) -> str:
        """Return a formatted summary string."""
        return (
            f"SimulationStats(n_runs={self.n_runs}, n_circuits={self.n_circuits})\n"
            f"  Mean Energy: {self.mean_energy:.8f} Ha\n"
            f"  Std Energy:  {self.std_energy:.8f} Ha\n"
            f"  RMSE:        {self.rmse:.8f} Ha ({self.rmse*1000:.4f} mHa)\n"
            f"  Bias:        {self.bias:.8f} Ha ({self.bias*1000:.4f} mHa)\n"
            f"  Exact:       {self.exact_energy:.8f} Ha"
        )


class StatevectorEnergyEstimator:
    """
    Energy estimator using Qiskit statevector simulations and Estimator primitive.
    
    Supports:
    - Exact expectation value computation (infinite shots)
    - Qiskit Estimator with configurable shot budgets
    """
    
    def __init__(self, n_qubits: int):
        """
        Initialize the energy estimator.
        
        Args:
            n_qubits: Number of qubits in the system
        """
        self.n_qubits = n_qubits
    
    @classmethod
    def from_hamiltonian_file(cls, filepath: Union[str, Path]) -> Tuple['StatevectorEnergyEstimator', 'HamiltonianData']:
        """
        Create estimator from a Hamiltonian JSON file.
        
        Args:
            filepath: Path to Hamiltonian JSON file
            
        Returns:
            Tuple of (estimator, hamiltonian_data)
        """
        ham_data = HamiltonianData.load_json(filepath)
        return cls(ham_data.n_qubits), ham_data
    
    def compute_exact_energy(
        self,
        state: Union[np.ndarray, QuantumCircuit, Statevector],
        hamiltonian: 'HamiltonianData'
    ) -> EnergyResult:
        """
        Compute exact energy ⟨ψ|H|ψ⟩ using statevector simulation (infinite precision).
        
        Args:
            state: Input state as vector, circuit, or Statevector
            hamiltonian: Hamiltonian data containing Pauli terms
            
        Returns:
            EnergyResult with exact energy
        """
        sv = self._to_statevector(state)
        
        # Build SparsePauliOp for efficient computation
        sparse_op = hamiltonian.to_sparse_pauli_op()
        
        # Compute expectation value: ⟨ψ|H|ψ⟩
        energy = sv.expectation_value(sparse_op).real
        
        # Also compute individual Pauli expectations for analysis
        # Note: Qiskit's SparsePauliOp.from_list label convention matches ours
        pauli_exp = {}
        for pauli_str, coeff in zip(hamiltonian.pauli_strings, hamiltonian.coefficients):
            pauli_op = SparsePauliOp.from_list([(pauli_str, 1.0)])
            exp_val = sv.expectation_value(pauli_op).real
            pauli_exp[pauli_str] = exp_val
        
        return EnergyResult(
            energy=float(energy),
            exact_energy=float(energy),
            error=0.0,
            std_error=0.0,
            method="exact_statevector",
            pauli_expectations=pauli_exp
        )
    
    def estimate_energy_with_shots(
        self,
        state: Union[np.ndarray, QuantumCircuit, Statevector],
        hamiltonian: 'HamiltonianData',
        n_shots: int = 10000,
        seed: Optional[int] = None
    ) -> EnergyResult:
        """
        Estimate energy using Qiskit StatevectorEstimator with finite shots.
        
        Uses Qiskit's built-in Estimator primitive which handles measurement
        grouping and shot allocation automatically.
        
        Args:
            state: Input state as vector, circuit, or Statevector
            hamiltonian: Hamiltonian data
            n_shots: Total number of measurement shots
            seed: Random seed for reproducibility
            
        Returns:
            EnergyResult with estimated energy and standard error
        """
        # Convert state to circuit if needed
        circuit = self._to_circuit(state)
        
        # Build observable
        observable = hamiltonian.to_sparse_pauli_op()
        
        # Create estimator with precision based on shots
        # StatevectorEstimator uses precision parameter (standard error target)
        # precision ≈ 1/sqrt(shots) for shot-based estimation
        precision = 1.0 / np.sqrt(n_shots) if n_shots > 0 else 0.0
        
        estimator = StatevectorEstimator(seed=seed)
        
        # Run estimation
        job = estimator.run([(circuit, observable)], precision=precision)
        result = job.result()
        
        # Extract results
        energy = float(result[0].data.evs)
        std_error = float(result[0].data.stds) if hasattr(result[0].data, 'stds') else None
        
        # Compute exact energy for error calculation
        exact = self.compute_exact_energy(state, hamiltonian)
        
        return EnergyResult(
            energy=energy,
            exact_energy=exact.energy,
            error=abs(energy - exact.energy),
            std_error=std_error,
            n_shots=n_shots,
            method="qiskit_estimator",
            pauli_expectations=None  # Not available from Estimator
        )
    
    def estimate_energy_multiple_budgets(
        self,
        state: Union[np.ndarray, QuantumCircuit, Statevector],
        hamiltonian: 'HamiltonianData',
        shot_budgets: List[int],
        n_repeats: int = 10,
        seed: Optional[int] = None
    ) -> Dict[int, Dict[str, float]]:
        """
        Estimate energy for multiple shot budgets with statistics.
        
        Args:
            state: Input state
            hamiltonian: Hamiltonian data
            shot_budgets: List of shot budgets to test
            n_repeats: Number of repetitions per budget for statistics
            seed: Base random seed
            
        Returns:
            Dictionary mapping shot budget to statistics:
            {budget: {'mean': ..., 'std': ..., 'rmse': ..., 'bias': ...}}
        """
        rng = np.random.default_rng(seed)
        exact = self.compute_exact_energy(state, hamiltonian)
        
        results = {}
        for budget in shot_budgets:
            energies = []
            for i in range(n_repeats):
                rep_seed = int(rng.integers(0, 2**31))
                result = self.estimate_energy_with_shots(
                    state, hamiltonian, n_shots=budget, seed=rep_seed
                )
                energies.append(result.energy)
            
            energies = np.array(energies)
            errors = energies - exact.energy
            
            results[budget] = {
                'mean': float(np.mean(energies)),
                'std': float(np.std(energies, ddof=1)) if n_repeats > 1 else 0.0,
                'rmse': float(np.sqrt(np.mean(errors**2))),
                'bias': float(np.mean(errors)),
                'exact': exact.energy
            }
        
        return results
    
    def _to_statevector(
        self, 
        state: Union[np.ndarray, QuantumCircuit, Statevector]
    ) -> Statevector:
        """Convert input to Statevector."""
        if isinstance(state, Statevector):
            return state
        elif isinstance(state, np.ndarray):
            return Statevector(state)
        elif isinstance(state, QuantumCircuit):
            return Statevector.from_instruction(state)
        else:
            raise TypeError(f"Unsupported state type: {type(state)}")
    
    def _to_circuit(
        self,
        state: Union[np.ndarray, QuantumCircuit, Statevector]
    ) -> QuantumCircuit:
        """Convert input to QuantumCircuit."""
        if isinstance(state, QuantumCircuit):
            return state
        elif isinstance(state, (np.ndarray, Statevector)):
            sv = self._to_statevector(state)
            qc = QuantumCircuit(self.n_qubits)
            qc.initialize(sv.data)
            return qc
        else:
            raise TypeError(f"Unsupported state type: {type(state)}")
    
    def run_dss_simulation(
        self,
        ground_state: Union[np.ndarray, QuantumCircuit, Statevector],
        measurement_circuits: List[QuantumCircuit],
        hamiltonian: 'HamiltonianData',
        seed: Optional[int] = None,
        batch_actions: Optional[Any] = None,  # torch.Tensor (unused, for API compatibility)
        batch_lengths: Optional[Any] = None,  # torch.Tensor (unused, for API compatibility)
        action_map: Optional[Dict] = None  # unused, for API compatibility
    ) -> Tuple[float, Dict[str, float], Dict[str, int]]:
        """
        Run a single DSS-style energy estimation with given measurement circuits.
        
        For each circuit U_i:
        1. Prepare |ψ⟩ (ground state)
        2. Apply U_i
        3. Measure in computational basis → get ONE bitstring b_i
        4. For each Pauli P where U_i P U_i† is diagonal, estimate ⟨P⟩
        
        Uses Qiskit Clifford for tracking both measurability and phase/sign.
        
        Args:
            ground_state: The state to measure
            measurement_circuits: List of Clifford measurement circuits
            hamiltonian: Hamiltonian data
            seed: Random seed
            
        Returns:
            Tuple of (energy_estimate, pauli_estimates, hitting_counts)
        """
        from qiskit.quantum_info import Clifford
        
        sv = self._to_statevector(ground_state)
        rng = np.random.default_rng(seed)
        
        n_circuits = len(measurement_circuits)
        
        # Pre-compute Clifford objects and inverse circuits
        # CliffordMap convention: U† P U (Heisenberg picture, backward evolution)
        # So we need to apply U† to the state and check if U† P U is diagonal
        inverse_circuits = [circuit.inverse() for circuit in measurement_circuits]
        cliffords = [Clifford(circuit) for circuit in measurement_circuits]
        
        # Sample bitstrings using Qiskit statevector
        # Apply U† (inverse circuit) to state, matching CliffordMap convention
        bitstrings = []
        for inv_circuit in inverse_circuits:
            evolved_sv = sv.evolve(inv_circuit)  # U†|ψ⟩
            probs = np.abs(evolved_sv.data)**2
            outcome = rng.choice(2**self.n_qubits, p=probs)
            # Store as binary string (Qiskit convention: MSB = qubit n-1)
            bitstrings.append(format(outcome, f'0{self.n_qubits}b'))
        
        # For each Pauli, determine which circuits can measure it and accumulate
        pauli_estimates = {}
        hitting_counts = {}
        
        for pauli_str in hamiltonian.pauli_strings:
            # Convert to Qiskit Pauli
            # Use Hamiltonian convention (label): string[i] = qubit (n-1-i)
            # This matches SparsePauliOp.from_list and the ground state computation
            qiskit_pauli = self._pauli_str_to_qiskit(pauli_str, use_clifford_convention=False)
            
            total = 0.0
            hits = 0
            
            for c_idx, clifford in enumerate(cliffords):
                # Transform Pauli: P' = U† P U (CliffordMap convention)
                # For energy estimation with U†|ψ⟩: ⟨ψ|P|ψ⟩ = ⟨U†ψ|(U† P U)|U†ψ⟩
                # Use frame='s': computes C† @ P @ C = U† P U
                # This matches CliffordMap convention exactly!
                transformed = qiskit_pauli.evolve(clifford, frame='s')
                
                # Check if diagonal (no X component)
                if not transformed.x.any():
                    # Compute eigenvalue including phase
                    eigenvalue = self._compute_eigenvalue_from_bitstring(
                        transformed, bitstrings[c_idx]
                    )
                    
                    total += eigenvalue
                    hits += 1
            
            hitting_counts[pauli_str] = hits
            pauli_estimates[pauli_str] = total / hits if hits > 0 else 0.0
        
        # Compute energy
        energy = hamiltonian.identity_weight
        for pauli_str, coeff in zip(hamiltonian.pauli_strings, hamiltonian.coefficients):
            energy += coeff * pauli_estimates[pauli_str]
        
        return float(energy), pauli_estimates, hitting_counts
    
    def _pauli_str_to_qiskit(self, pauli_str: str, use_clifford_convention: bool = False) -> Pauli:
        """
        Convert Pauli string to Qiskit Pauli using array construction.
        
        Two conventions exist:
        1. Hamiltonian (Qiskit label): pauli_str[i] = operator on qubit (n-1-i)
        2. CliffordMap: pauli_str[i] = operator on qubit i
        
        Args:
            pauli_str: Pauli string in Hamiltonian convention
            use_clifford_convention: If True, first convert to CliffordMap convention (reverse string)
            
        Qiskit array: x[i], z[i] = operator on qubit i
        """
        n = len(pauli_str)
        x_arr = np.zeros(n, dtype=bool)
        z_arr = np.zeros(n, dtype=bool)
        
        # If using CliffordMap convention, reverse the string first
        # This converts from Hamiltonian convention to CliffordMap convention
        if use_clifford_convention:
            pauli_str = pauli_str[::-1]  # Now string[i] = qubit i
            
        for i, op in enumerate(pauli_str):
            if use_clifford_convention:
                # After reversal: pauli_str[i] = qubit i = arr[i]
                arr_idx = i
            else:
                # Hamiltonian convention: pauli_str[i] = qubit (n-1-i)
                arr_idx = n - 1 - i
                
            if op == 'X':
                x_arr[arr_idx] = True
            elif op == 'Z':
                z_arr[arr_idx] = True
            elif op == 'Y':
                x_arr[arr_idx] = True
                z_arr[arr_idx] = True
            # 'I' leaves both False
        
        return Pauli((z_arr, x_arr))
    
    def _compute_eigenvalue_from_bitstring(
        self,
        transformed_pauli: Pauli,
        bitstring: str
    ) -> float:
        """
        Compute eigenvalue of transformed Pauli on computational basis state.
        
        Args:
            transformed_pauli: The transformed Pauli (should be diagonal, in Qiskit convention)
            bitstring: Measurement outcome as binary string (MSB-first)
            
        Returns:
            Eigenvalue (+1 or -1)
        """
        # Phase from Pauli (0→+1, 1→+i, 2→-1, 3→-i)
        phase = transformed_pauli.phase
        if phase in (1, 3):
            # Should not happen for diagonal Hermitian operators
            return 0.0
        sign = 1.0 if phase == 0 else -1.0
        
        # Compute parity from Z positions
        # In Qiskit: transformed_pauli.z[i] is True if there's a Z on qubit i
        # Bitstring is MSB-first: bitstring[0] = qubit n-1, bitstring[n-1] = qubit 0
        # So qubit i's bit is at bitstring[n-1-i]
        n = len(bitstring)
        parity = 0
        for i, has_z in enumerate(transformed_pauli.z):
            if has_z:
                # Qubit i's measurement result is at position n-1-i in bitstring
                bit = int(bitstring[n - 1 - i])
                parity ^= bit
        
        return sign * ((-1) ** parity)
    
    def run_independent_simulations(
        self,
        ground_state: Union[np.ndarray, QuantumCircuit, Statevector],
        measurement_circuits: List[QuantumCircuit],
        hamiltonian: 'HamiltonianData',
        M: int = 100,
        seed: Optional[int] = None,
        checkpoint_data: Optional['CheckpointData'] = None
    ) -> SimulationStats:
        """
        Run M independent DSS-style simulations and compute statistics.
        
        Each simulation:
        - Uses the same measurement circuits
        - Samples new random bitstrings for each circuit
        - Computes an independent energy estimate
        
        Args:
            ground_state: The state to measure
            measurement_circuits: List of Clifford measurement circuits
            hamiltonian: Hamiltonian data
            M: Number of independent simulation runs
            seed: Base random seed
            checkpoint_data: Optional checkpoint data for using CliffordMap
            
        Returns:
            SimulationStats with RMSE, variance, etc.
        """
        rng = np.random.default_rng(seed)
        
        # Get exact energy for error calculation
        exact = self.compute_exact_energy(ground_state, hamiltonian)
        exact_energy = exact.energy
        
        # Prepare CliffordMap parameters if checkpoint data is available
        import torch
        batch_actions = None
        batch_lengths = None
        action_map = None
        if checkpoint_data is not None:
            batch_actions = torch.from_numpy(checkpoint_data.actions)
            batch_lengths = torch.from_numpy(checkpoint_data.lengths)
            action_map = checkpoint_data.action_map
        
        # Run M independent simulations
        energies = []
        for m in range(M):
            sim_seed = int(rng.integers(0, 2**31))
            energy, _, _ = self.run_dss_simulation(
                ground_state, measurement_circuits, hamiltonian, seed=sim_seed,
                batch_actions=batch_actions, batch_lengths=batch_lengths, action_map=action_map
            )
            energies.append(energy)
        
        energies = np.array(energies)
        errors = energies - exact_energy
        
        return SimulationStats(
            energies=energies.tolist(),
            mean_energy=float(np.mean(energies)),
            std_energy=float(np.std(energies, ddof=1)) if M > 1 else 0.0,
            rmse=float(np.sqrt(np.mean(errors**2))),
            bias=float(np.mean(errors)),
            exact_energy=exact_energy,
            n_runs=M,
            n_circuits=len(measurement_circuits)
        )
    
    def run_convergence_study(
        self,
        ground_state: Union[np.ndarray, QuantumCircuit, Statevector],
        measurement_circuits: List[QuantumCircuit],
        hamiltonian: 'HamiltonianData',
        circuit_budgets: List[int],
        M: int = 100,
        seed: Optional[int] = None
    ) -> Dict[int, SimulationStats]:
        """
        Run convergence study with different numbers of circuits.
        
        Args:
            ground_state: The state to measure
            measurement_circuits: Full list of measurement circuits
            hamiltonian: Hamiltonian data
            circuit_budgets: List of circuit counts to test
            M: Number of independent runs per budget
            seed: Base random seed
            
        Returns:
            Dictionary mapping circuit budget to SimulationStats
        """
        rng = np.random.default_rng(seed)
        results = {}
        
        for n_circuits in circuit_budgets:
            # Use first n_circuits from the list
            circuits_subset = measurement_circuits[:n_circuits]
            
            # Or repeat if we don't have enough
            if len(circuits_subset) < n_circuits:
                repeats = (n_circuits // len(measurement_circuits)) + 1
                circuits_subset = (measurement_circuits * repeats)[:n_circuits]
            
            sub_seed = int(rng.integers(0, 2**31))
            stats = self.run_independent_simulations(
                ground_state, circuits_subset, hamiltonian, M=M, seed=sub_seed
            )
            results[n_circuits] = stats
        
        return results


class HamiltonianData:
    """Container for Hamiltonian data."""
    
    def __init__(
        self,
        identity_weight: float,
        pauli_strings: List[str],
        coefficients: List[float],
        n_qubits: Optional[int] = None,
        ground_state_energy: Optional[float] = None,
        ground_state_vector: Optional[np.ndarray] = None
    ):
        self.identity_weight = identity_weight
        self.pauli_strings = pauli_strings
        self.coefficients = coefficients
        self.n_qubits = n_qubits or (len(pauli_strings[0]) if pauli_strings else 0)
        self.ground_state_energy = ground_state_energy
        self.ground_state_vector = ground_state_vector
    
    @classmethod
    def load_json(cls, filepath: Union[str, Path]) -> 'HamiltonianData':
        """Load Hamiltonian from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        pauli_strings = []
        coefficients = []
        identity_weight = 0.0
        
        for term in data['paulis']:
            label = term['label']
            
            if isinstance(term['coeff'], dict):
                coeff = term['coeff']['real']
            else:
                coeff = float(term['coeff'])
            
            if label == 'I' * len(label):
                identity_weight = coeff
            else:
                pauli_strings.append(label)
                coefficients.append(coeff)
        
        return cls(identity_weight, pauli_strings, coefficients)
    
    @classmethod
    def load_txt(cls, filepath: Union[str, Path]) -> 'HamiltonianData':
        """
        Load Hamiltonian from text file format.
        
        Format: alternating lines of Pauli string and complex coefficient.
        Example:
            IIII
            (-0.81+0j)
            ZIII
            (0.17+0j)
        """
        with open(filepath, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]
        
        pauli_strings = []
        coefficients = []
        identity_weight = 0.0
        
        for i in range(0, len(lines), 2):
            if i + 1 >= len(lines):
                break
            
            pauli_str = lines[i]
            coeff_str = lines[i + 1]
            
            # Parse complex coefficient
            coeff = complex(coeff_str).real
            
            if pauli_str == 'I' * len(pauli_str):
                identity_weight = coeff
            else:
                pauli_strings.append(pauli_str)
                coefficients.append(coeff)
        
        return cls(identity_weight, pauli_strings, coefficients)
    
    @classmethod
    def load(cls, filepath: Union[str, Path]) -> 'HamiltonianData':
        """Load Hamiltonian from file (auto-detect format)."""
        filepath = Path(filepath)
        
        if filepath.suffix == '.json':
            # Try JSON format
            try:
                return cls.load_json(filepath)
            except (json.JSONDecodeError, KeyError):
                pass
        
        if filepath.suffix == '.txt':
            return cls.load_txt(filepath)
        
        # Try to auto-detect
        with open(filepath, 'r') as f:
            first_line = f.readline().strip()
        
        if first_line.startswith('{'):
            return cls.load_json(filepath)
        else:
            return cls.load_txt(filepath)
    
    @classmethod
    def from_pauli_hamiltonian_helper(cls, helper) -> 'HamiltonianData':
        """
        Create HamiltonianData from a PauliHamiltonianHelper instance.
        
        Args:
            helper: PauliHamiltonianHelper instance from code/pauli_hamiltonian_helper.py
            
        Returns:
            HamiltonianData instance
        """
        identity_str = "I" * helper.n_qubits
        
        pauli_strings = []
        coefficients = []
        identity_weight = 0.0
        
        for p_str, coeff in zip(helper.pauli_str_list, helper.w_list):
            coeff_real = coeff.real if hasattr(coeff, 'real') else float(coeff)
            if p_str == identity_str:
                identity_weight = coeff_real
            else:
                pauli_strings.append(p_str)
                coefficients.append(coeff_real)
        
        # Get ground state info if available
        gs_energy = None
        gs_vector = None
        try:
            gs_energy = helper.ground_state_energy
            gs_vector = helper.ground_state_vector
        except Exception:
            pass
        
        return cls(
            identity_weight=identity_weight,
            pauli_strings=pauli_strings,
            coefficients=coefficients,
            n_qubits=helper.n_qubits,
            ground_state_energy=gs_energy,
            ground_state_vector=gs_vector
        )
    
    @classmethod
    def from_sparse_pauli_op(cls, op: SparsePauliOp) -> 'HamiltonianData':
        """Create from Qiskit SparsePauliOp."""
        pauli_strings = []
        coefficients = []
        identity_weight = 0.0
        
        for label, coeff in op.to_list():
            if label == 'I' * len(label):
                identity_weight = coeff.real
            else:
                pauli_strings.append(label)
                coefficients.append(coeff.real)
        
        return cls(identity_weight, pauli_strings, coefficients)
    
    def to_sparse_pauli_op(self) -> SparsePauliOp:
        """Convert to Qiskit SparsePauliOp.
        
        Note: Qiskit's SparsePauliOp.from_list label convention matches ours:
        label[i] = operator on qubit i
        """
        terms = [(p, c) for p, c in zip(self.pauli_strings, self.coefficients)]
        if self.identity_weight != 0:
            terms.append(('I' * self.n_qubits, self.identity_weight))
        return SparsePauliOp.from_list(terms)
    
    def get_stats(self) -> Dict:
        """Get Hamiltonian statistics."""
        return {
            'n_qubits': self.n_qubits,
            'n_terms': len(self.pauli_strings) + (1 if self.identity_weight != 0 else 0),
            'identity_weight': self.identity_weight,
            'max_coeff': max(abs(c) for c in self.coefficients) if self.coefficients else 0,
            'sum_abs_coeffs': sum(abs(c) for c in self.coefficients),
        }


def create_test_state(n_qubits: int, state_type: str = "hf") -> QuantumCircuit:
    """
    Create common test states.
    
    Args:
        n_qubits: Number of qubits
        state_type: Type of state - "hf" (Hartree-Fock), "random", "ghz", "uniform"
        
    Returns:
        QuantumCircuit preparing the state
    """
    qc = QuantumCircuit(n_qubits)
    
    if state_type == "hf":
        # Simple Hartree-Fock-like state (alternating occupation)
        for i in range(0, n_qubits, 2):
            qc.x(i)
    
    elif state_type == "ghz":
        qc.h(0)
        for i in range(1, n_qubits):
            qc.cx(i-1, i)
    
    elif state_type == "uniform":
        for i in range(n_qubits):
            qc.h(i)
    
    elif state_type == "random":
        # Random product state with rotations
        rng = np.random.default_rng(42)
        for i in range(n_qubits):
            qc.ry(rng.uniform(0, np.pi), i)
            qc.rz(rng.uniform(0, 2*np.pi), i)
    
    return qc


def create_hf_state_from_bitstring(n_qubits: int, bitstring: str) -> QuantumCircuit:
    """
    Create Hartree-Fock state from a bitstring.
    
    Args:
        n_qubits: Number of qubits
        bitstring: Binary string representing occupation (e.g., "1010")
        
    Returns:
        QuantumCircuit preparing the HF state
    """
    qc = QuantumCircuit(n_qubits)
    for i, bit in enumerate(reversed(bitstring)):
        if bit == '1':
            qc.x(i)
    return qc


def create_simple_hamiltonian(n_qubits: int, model: str = "ising") -> HamiltonianData:
    """
    Create simple test Hamiltonians.
    
    Args:
        n_qubits: Number of qubits
        model: Model type - "ising", "heisenberg", "tfim"
        
    Returns:
        HamiltonianData for the model
    """
    pauli_strings = []
    coefficients = []
    identity_weight = 0.0
    
    if model == "ising":
        # H = -J Σ Z_i Z_{i+1} - h Σ X_i
        J, h = 1.0, 0.5
        for i in range(n_qubits - 1):
            pauli = ['I'] * n_qubits
            pauli[i] = 'Z'
            pauli[i+1] = 'Z'
            pauli_strings.append(''.join(pauli))
            coefficients.append(-J)
        
        for i in range(n_qubits):
            pauli = ['I'] * n_qubits
            pauli[i] = 'X'
            pauli_strings.append(''.join(pauli))
            coefficients.append(-h)
    
    elif model == "heisenberg":
        # H = J Σ (X_i X_{i+1} + Y_i Y_{i+1} + Z_i Z_{i+1})
        J = 1.0
        for i in range(n_qubits - 1):
            for op in ['X', 'Y', 'Z']:
                pauli = ['I'] * n_qubits
                pauli[i] = op
                pauli[i+1] = op
                pauli_strings.append(''.join(pauli))
                coefficients.append(J)
    
    elif model == "tfim":
        # Transverse-field Ising model
        J, g = 1.0, 1.0
        for i in range(n_qubits - 1):
            pauli = ['I'] * n_qubits
            pauli[i] = 'Z'
            pauli[i+1] = 'Z'
            pauli_strings.append(''.join(pauli))
            coefficients.append(-J)
        
        for i in range(n_qubits):
            pauli = ['I'] * n_qubits
            pauli[i] = 'X'
            pauli_strings.append(''.join(pauli))
            coefficients.append(-g)
    
    return HamiltonianData(identity_weight, pauli_strings, coefficients, n_qubits)


def demo_energy_estimation():
    """Demonstrate energy estimation methods."""
    print("=" * 70)
    print("ENERGY ESTIMATION DEMO - QISKIT STATEVECTOR")
    print("=" * 70)
    
    # Test 1: Simple Ising model
    print("\n--- Test 1: Transverse-Field Ising Model (4 qubits) ---")
    n_qubits = 4
    hamiltonian = create_simple_hamiltonian(n_qubits, "tfim")
    estimator = StatevectorEnergyEstimator(n_qubits)
    
    print(f"Hamiltonian stats: {hamiltonian.get_stats()}")
    
    # Test with Hartree-Fock state
    state_circuit = create_test_state(n_qubits, "hf")
    
    # Exact energy
    result_exact = estimator.compute_exact_energy(state_circuit, hamiltonian)
    print(f"\nExact energy: {result_exact.energy:.6f} Ha")
    
    # Sampling-based estimation
    result_sampling = estimator.estimate_energy_sampling(
        state_circuit, hamiltonian, n_shots=10000, seed=42
    )
    print(f"Sampling ({result_sampling.n_shots} shots): {result_sampling.energy:.6f} Ha "
          f"(error: {result_sampling.error:.6f})")
    
    # Shadow estimation
    result_shadow = estimator.estimate_energy_shadow(
        state_circuit, hamiltonian, n_snapshots=5000, seed=42
    )
    print(f"Shadow ({result_shadow.n_shots} snapshots): {result_shadow.energy:.6f} Ha "
          f"(error: {result_shadow.error:.6f})")
    
    # Test 2: Load H2 Hamiltonian if available
    print("\n--- Test 2: H2 Molecule (8 qubits) ---")
    h2_path = Path(__file__).parent / "data" / "hamiltonian_h2_8q.txt"
    
    if h2_path.exists():
        estimator_h2, hamiltonian_h2 = StatevectorEnergyEstimator.from_hamiltonian_file(h2_path)
        print(f"H2 Hamiltonian loaded: {hamiltonian_h2.get_stats()}")
        
        # Create approximate ground state
        state_h2 = create_test_state(8, "hf")
        
        result_h2 = estimator_h2.compute_exact_energy(state_h2, hamiltonian_h2)
        print(f"HF state energy: {result_h2.energy:.6f} Ha")
        
        result_h2_shadow = estimator_h2.estimate_energy_shadow(
            state_h2, hamiltonian_h2, n_snapshots=2000, seed=42
        )
        print(f"Shadow estimate: {result_h2_shadow.energy:.6f} Ha (error: {result_h2_shadow.error:.6f})")
    else:
        print(f"H2 Hamiltonian file not found at {h2_path}")
    
    # Test 3: Convergence study
    print("\n--- Test 3: Convergence Study ---")
    n_qubits = 6
    hamiltonian = create_simple_hamiltonian(n_qubits, "heisenberg")
    estimator = StatevectorEnergyEstimator(n_qubits)
    state = create_test_state(n_qubits, "random")
    
    exact = estimator.compute_exact_energy(state, hamiltonian)
    print(f"Exact energy: {exact.energy:.6f}")
    
    print("\nShot/Snapshot | Sampling Error | Shadow Error")
    print("-" * 50)
    for n in [100, 500, 1000, 5000, 10000]:
        sampling = estimator.estimate_energy_sampling(state, hamiltonian, n_shots=n, seed=42)
        shadow = estimator.estimate_energy_shadow(state, hamiltonian, n_snapshots=n, seed=42)
        print(f"{n:>13} | {sampling.error:>14.6f} | {shadow.error:>12.6f}")
    
    print("\n" + "=" * 70)
    print("Demo complete!")
    return result_exact, result_sampling, result_shadow


def test_with_molecule(
    hamiltonian_path: Union[str, Path],
    ground_state_energy: Optional[float] = None,
    hf_bitstring: Optional[str] = None,
    n_shots: int = 10000,
    n_snapshots: int = 5000
):
    """
    Test energy estimation with a molecular Hamiltonian.
    
    Args:
        hamiltonian_path: Path to Hamiltonian file (.txt or .json)
        ground_state_energy: Known ground state energy for comparison
        hf_bitstring: Hartree-Fock reference bitstring (e.g., "1010")
        n_shots: Number of shots for sampling estimation
        n_snapshots: Number of snapshots for shadow estimation
    """
    print("=" * 70)
    print("MOLECULAR ENERGY ESTIMATION TEST")
    print("=" * 70)
    
    # Load Hamiltonian
    hamiltonian = HamiltonianData.load(hamiltonian_path)
    n_qubits = hamiltonian.n_qubits
    estimator = StatevectorEnergyEstimator(n_qubits)
    
    print(f"\nHamiltonian: {Path(hamiltonian_path).stem}")
    print(f"  - Number of qubits: {n_qubits}")
    print(f"  - Number of Pauli terms: {len(hamiltonian.pauli_strings)}")
    print(f"  - Identity weight: {hamiltonian.identity_weight:.6f}")
    if ground_state_energy is not None:
        print(f"  - Known ground state energy: {ground_state_energy:.6f} Ha")
    
    # Prepare initial state
    if hf_bitstring:
        # Create HF state from bitstring
        qc = QuantumCircuit(n_qubits)
        for i, bit in enumerate(reversed(hf_bitstring)):
            if bit == '1':
                qc.x(i)
        state_circuit = qc
        print(f"\nUsing Hartree-Fock state: |{hf_bitstring}⟩")
    else:
        state_circuit = create_test_state(n_qubits, "hf")
        print(f"\nUsing default HF-like state")
    
    # Compute exact energy with statevector
    print("\n--- Statevector Computation ---")
    result_exact = estimator.compute_exact_energy(state_circuit, hamiltonian)
    print(f"Exact ⟨ψ|H|ψ⟩: {result_exact.energy:.8f} Ha")
    
    if ground_state_energy is not None:
        error_from_gs = result_exact.energy - ground_state_energy
        print(f"Error from ground state: {error_from_gs:.8f} Ha ({error_from_gs*1000:.4f} mHa)")
    
    # Sampling-based estimation
    print(f"\n--- Sampling-Based Estimation ({n_shots} shots) ---")
    result_sampling = estimator.estimate_energy_sampling(
        state_circuit, hamiltonian, n_shots=n_shots, seed=42
    )
    print(f"Estimated energy: {result_sampling.energy:.8f} Ha")
    print(f"Estimation error: {result_sampling.error:.8f} Ha ({result_sampling.error*1000:.4f} mHa)")
    
    # Shadow estimation
    print(f"\n--- Classical Shadow Estimation ({n_snapshots} snapshots) ---")
    result_shadow = estimator.estimate_energy_shadow(
        state_circuit, hamiltonian, n_snapshots=n_snapshots, seed=42
    )
    print(f"Estimated energy: {result_shadow.energy:.8f} Ha")
    print(f"Estimation error: {result_shadow.error:.8f} Ha ({result_shadow.error*1000:.4f} mHa)")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Method':<25} {'Energy (Ha)':<15} {'Error (mHa)':<15}")
    print("-" * 55)
    print(f"{'Exact Statevector':<25} {result_exact.energy:<15.8f} {'---':<15}")
    print(f"{'Sampling':<25} {result_sampling.energy:<15.8f} {result_sampling.error*1000:<15.4f}")
    print(f"{'Classical Shadow':<25} {result_shadow.energy:<15.8f} {result_shadow.error*1000:<15.4f}")
    
    if ground_state_energy is not None:
        print("-" * 55)
        print(f"{'Ground State (ref)':<25} {ground_state_energy:<15.8f}")
    
    return {
        'exact': result_exact,
        'sampling': result_sampling,
        'shadow': result_shadow
    }


@dataclass
class CheckpointData:
    """Data loaded from a Flow-Shadow checkpoint."""
    circuits: List[QuantumCircuit]
    n_qubits: int
    actions: 'np.ndarray'  # (n_circuits, max_len)
    lengths: 'np.ndarray'  # (n_circuits,)
    action_map: Dict


def load_circuits_from_checkpoint(checkpoint_path: Union[str, Path]) -> CheckpointData:
    """
    Load DSS measurement circuits from a Flow-Shadow checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint .pth file
        
    Returns:
        CheckpointData with circuits and metadata for CliffordMap usage
    """
    import torch
    
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    n_qubits = int(ckpt.get("n_qubits", 0) or 0)
    if n_qubits <= 0:
        raise ValueError("Checkpoint missing n_qubits")
    
    action_mapping = ckpt["action_mapping"]
    terminal = ckpt.get("terminal_index", None)
    
    # Get trajectories from checkpoint
    traj = ckpt["top_trajectories"][0]
    actions_2d = traj["actions"]
    lengths_1d = traj["lengths"]
    n_rows = int(lengths_1d.shape[0])
    
    # Convert to numpy for storage
    actions_np = actions_2d.cpu().numpy() if hasattr(actions_2d, 'cpu') else np.array(actions_2d)
    lengths_np = lengths_1d.cpu().numpy() if hasattr(lengths_1d, 'cpu') else np.array(lengths_1d)
    
    def append_single(qc, gate, q):
        if gate == "H":
            qc.h(q)
        elif gate == "S":
            qc.s(q)
        elif gate == "HS":
            qc.s(q)
            qc.h(q)
        elif gate == "SH":
            qc.h(q)
            qc.s(q)
        elif gate == "HSH":
            qc.h(q)
            qc.s(q)
            qc.h(q)
    
    def append_two(qc, gate, c, t):
        if gate == "CNOT":
            qc.cx(c, t)
    
    circuits = []
    for i in range(n_rows):
        L = int(lengths_np[i])
        row = actions_np[i].tolist()
        qc = QuantumCircuit(n_qubits)
        
        for step in range(L):
            aid = int(row[step])
            if terminal is not None and aid == terminal:
                break
            spec = action_mapping.get(aid)
            if spec is None:
                continue
            gate = spec[0]
            if gate == "terminal":
                break
            if len(spec) == 2:
                append_single(qc, gate, int(spec[1]))
            elif len(spec) == 3:
                append_two(qc, gate, int(spec[1]), int(spec[2]))
        
        circuits.append(qc)
    
    return CheckpointData(
        circuits=circuits,
        n_qubits=n_qubits,
        actions=actions_np,
        lengths=lengths_np,
        action_map=action_mapping
    )


def test_with_checkpoint(
    checkpoint_path: Union[str, Path],
    hamiltonian_path: Union[str, Path],
    M: int = 100,
    circuit_budgets: Optional[List[int]] = None
):
    """
    Test energy estimation using circuits from a Flow-Shadow checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint .pth file
        hamiltonian_path: Path to Hamiltonian file
        M: Number of independent simulation runs
        circuit_budgets: List of circuit counts for convergence study
    """
    print("=" * 70)
    print("DSS ENERGY ESTIMATION TEST (Using Checkpoint Circuits)")
    print("=" * 70)
    
    # Try to import PauliHamiltonianHelper
    try:
        from pauli_hamiltonian_helper import PauliHamiltonianHelper
    except ImportError:
        print("Error: Could not import PauliHamiltonianHelper.")
        return None
    
    # Load Hamiltonian
    print(f"\nLoading Hamiltonian from: {hamiltonian_path}")
    helper = PauliHamiltonianHelper(hamiltonian_path)
    hamiltonian = HamiltonianData.from_pauli_hamiltonian_helper(helper)
    
    # Load circuits from checkpoint
    print(f"Loading circuits from: {checkpoint_path}")
    ckpt_data = load_circuits_from_checkpoint(checkpoint_path)
    circuits = ckpt_data.circuits
    n_qubits = ckpt_data.n_qubits
    print(f"  Loaded {len(circuits)} circuits for {n_qubits} qubits")
    
    # Create estimator
    estimator = StatevectorEnergyEstimator(n_qubits)
    
    print(f"\n--- Hamiltonian Info ---")
    print(f"  Number of qubits: {n_qubits}")
    print(f"  Number of Pauli terms: {len(hamiltonian.pauli_strings)}")
    print(f"  Ground state energy: {hamiltonian.ground_state_energy:.8f} Ha")
    
    # Get ground state
    if hamiltonian.ground_state_vector is not None and len(hamiltonian.ground_state_vector) > 0:
        ground_state = hamiltonian.ground_state_vector
        print(f"  Using exact ground state vector")
    else:
        hf_bitstrings = helper.get_hartree_fock_bitstring()
        transform = Path(hamiltonian_path).stem
        hf_bitstring = hf_bitstrings.get(transform) if hf_bitstrings else None
        if hf_bitstring:
            ground_state = create_hf_state_from_bitstring(n_qubits, hf_bitstring)
            ground_state = Statevector.from_instruction(ground_state).data
            print(f"  Using Hartree-Fock state: |{hf_bitstring}⟩")
        else:
            ground_state = np.zeros(2**n_qubits, dtype=complex)
            ground_state[0] = 1.0
            print(f"  Using |0⟩ state")
    
    # Compute exact energy
    print("\n--- Exact Statevector Energy ---")
    exact_result = estimator.compute_exact_energy(ground_state, hamiltonian)
    print(f"Exact ⟨ψ|H|ψ⟩: {exact_result.energy:.10f} Ha")
    
    # Prepare CliffordMap parameters
    import torch
    batch_actions = torch.from_numpy(ckpt_data.actions)
    batch_lengths = torch.from_numpy(ckpt_data.lengths)
    
    # Run M independent simulations
    print(f"\n--- DSS Simulation ({M} independent runs, {len(circuits)} circuits) ---")
    stats = estimator.run_independent_simulations(
        ground_state, circuits, hamiltonian, M=M, seed=42,
        checkpoint_data=ckpt_data
    )
    print(stats.summary())
    
    # Run one simulation to get hitting counts for diagnostics
    _, pauli_est, hitting_counts = estimator.run_dss_simulation(
        ground_state, circuits, hamiltonian, seed=42,
        batch_actions=batch_actions, batch_lengths=batch_lengths, action_map=ckpt_data.action_map
    )
    
    # Show hitting counts
    print("\n--- Pauli Coverage ---")
    total_paulis = len(hamiltonian.pauli_strings)
    covered = sum(1 for c in hitting_counts.values() if c > 0)
    print(f"  Paulis covered: {covered}/{total_paulis} ({100*covered/total_paulis:.1f}%)")
    
    # Compare estimated vs true expectations for covered Paulis
    print("\n  --- Pauli Expectation Comparison (covered only) ---")
    true_expectations = exact_result.pauli_expectations
    total_covered_error = 0.0
    print(f"  {'Pauli':<8} {'Hits':<6} {'Coeff':>10} {'True ⟨P⟩':>10} {'Est ⟨P⟩':>10} {'Error':>10}")
    print(f"  {'-'*62}")
    for p_str, hits in hitting_counts.items():
        if hits > 0:
            idx = hamiltonian.pauli_strings.index(p_str)
            coeff = hamiltonian.coefficients[idx]
            true_exp = true_expectations.get(p_str, 0.0)
            est_exp = pauli_est.get(p_str, 0.0)
            error = (est_exp - true_exp) * coeff
            total_covered_error += error
            print(f"  {p_str:<8} {hits:<6} {coeff:>10.6f} {true_exp:>10.6f} {est_exp:>10.6f} {error*1000:>10.2f} mHa")
    print(f"  Total error from covered Paulis: {total_covered_error*1000:.2f} mHa")
    
    # Show uncovered Paulis with detailed diagnostics
    uncovered = [p for p, c in hitting_counts.items() if c == 0]
    if uncovered:
        print(f"  Uncovered Paulis: {uncovered[:5]}{'...' if len(uncovered) > 5 else ''}")
        
        # Show coefficients and true expectations for uncovered Paulis
        print("\n  --- Uncovered Pauli Analysis ---")
        true_expectations = exact_result.pauli_expectations
        total_uncovered_contribution = 0.0
        for p_str in uncovered:
            idx = hamiltonian.pauli_strings.index(p_str)
            coeff = hamiltonian.coefficients[idx]
            true_exp = true_expectations.get(p_str, 0.0)
            contribution = coeff * true_exp
            total_uncovered_contribution += contribution
            print(f"    {p_str}: coeff={coeff:+.6f}, ⟨P⟩={true_exp:+.6f}, contribution={contribution:+.6f} Ha")
        print(f"  Total energy from uncovered Paulis: {total_uncovered_contribution:.6f} Ha ({total_uncovered_contribution*1000:.4f} mHa)")
        print(f"  This explains the bias! (uncovered Paulis contribute ~{total_uncovered_contribution*1000:.1f} mHa)")
    
    # Show coverage statistics
    counts = list(hitting_counts.values())
    if counts:
        print(f"\n  Min/Max/Mean hits: {min(counts)}/{max(counts)}/{np.mean(counts):.1f}")
    
    # Convergence study if budgets provided
    if circuit_budgets:
        print(f"\n--- Convergence Study ---")
        print(f"{'N_circuits':<12} {'Mean (Ha)':<15} {'Std (Ha)':<12} {'RMSE (mHa)':<12} {'Bias (mHa)':<12}")
        print("-" * 63)
        
        conv_results = estimator.run_convergence_study(
            ground_state, circuits, hamiltonian,
            circuit_budgets=circuit_budgets, M=M, seed=42
        )
        
        for n_circ in sorted(conv_results.keys()):
            s = conv_results[n_circ]
            print(f"{n_circ:<12} {s.mean_energy:<15.8f} {s.std_energy:<12.6f} "
                  f"{s.rmse*1000:<12.4f} {s.bias*1000:<12.4f}")
    
    print("\n" + "=" * 70)
    return stats


def test_with_pauli_hamiltonian_helper(
    hamiltonian_path: Union[str, Path],
    n_shots: int = 10000
):
    """
    Test energy estimation using PauliHamiltonianHelper from the main codebase.
    
    Args:
        hamiltonian_path: Path to Hamiltonian file (e.g., Hamiltonians/H2_STO3g_4qubits/jw.txt)
        n_shots: Number of shots for Qiskit Estimator
    """
    print("=" * 70)
    print("ENERGY ESTIMATION TEST (Using PauliHamiltonianHelper)")
    print("=" * 70)
    
    # Try to import PauliHamiltonianHelper
    try:
        from pauli_hamiltonian_helper import PauliHamiltonianHelper
    except ImportError:
        print("Error: Could not import PauliHamiltonianHelper.")
        print("Make sure to run from the project root or add code/ to PYTHONPATH.")
        return None
    
    # Load Hamiltonian using PauliHamiltonianHelper
    print(f"\nLoading Hamiltonian from: {hamiltonian_path}")
    helper = PauliHamiltonianHelper(hamiltonian_path)
    
    # Create HamiltonianData from helper
    hamiltonian = HamiltonianData.from_pauli_hamiltonian_helper(helper)
    
    n_qubits = hamiltonian.n_qubits
    estimator = StatevectorEnergyEstimator(n_qubits)
    
    print(f"\n--- Hamiltonian Info ---")
    print(f"  Number of qubits: {n_qubits}")
    print(f"  Number of Pauli terms: {len(hamiltonian.pauli_strings)} (excluding identity)")
    print(f"  Identity weight: {hamiltonian.identity_weight:.8f}")
    print(f"  Ground state energy: {hamiltonian.ground_state_energy:.8f} Ha")
    
    # Use the exact ground state from PauliHamiltonianHelper
    if hamiltonian.ground_state_vector is not None and len(hamiltonian.ground_state_vector) > 0:
        ground_state = hamiltonian.ground_state_vector
        print(f"  Using exact ground state vector from cache/computation")
    else:
        # Fall back to Hartree-Fock state
        hf_bitstrings = helper.get_hartree_fock_bitstring()
        transform = Path(hamiltonian_path).stem
        hf_bitstring = hf_bitstrings.get(transform) if hf_bitstrings else None
        
        if hf_bitstring:
            print(f"  Using Hartree-Fock state: |{hf_bitstring}⟩")
            qc = QuantumCircuit(n_qubits)
            for i, bit in enumerate(reversed(hf_bitstring)):
                if bit == '1':
                    qc.x(i)
            ground_state = Statevector.from_instruction(qc).data
        else:
            print(f"  Using default |0⟩ state")
            ground_state = np.zeros(2**n_qubits, dtype=complex)
            ground_state[0] = 1.0
    
    # Compute exact energy with statevector
    print("\n--- Exact Statevector Computation ---")
    result_exact = estimator.compute_exact_energy(ground_state, hamiltonian)
    print(f"Exact ⟨ψ|H|ψ⟩: {result_exact.energy:.10f} Ha")
    
    error_from_gs = result_exact.energy - hamiltonian.ground_state_energy
    print(f"Error from known ground state: {error_from_gs:.10f} Ha ({abs(error_from_gs)*1000:.6f} mHa)")
    
    # Qiskit Estimator with shots
    print(f"\n--- Qiskit Estimator ({n_shots} effective shots) ---")
    result_estimator = estimator.estimate_energy_with_shots(
        ground_state, hamiltonian, n_shots=n_shots, seed=42
    )
    print(f"Estimated energy: {result_estimator.energy:.10f} Ha")
    print(f"Estimation error: {result_estimator.error:.10f} Ha ({result_estimator.error*1000:.6f} mHa)")
    if result_estimator.std_error:
        print(f"Std error: {result_estimator.std_error:.10f} Ha")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Method':<25} {'Energy (Ha)':<18} {'Error (mHa)':<15}")
    print("-" * 58)
    print(f"{'Exact Statevector':<25} {result_exact.energy:<18.10f} {abs(error_from_gs)*1000:<15.6f}")
    print(f"{'Qiskit Estimator':<25} {result_estimator.energy:<18.10f} {result_estimator.error*1000:<15.6f}")
    print("-" * 58)
    print(f"{'Ground State (ref)':<25} {hamiltonian.ground_state_energy:<18.10f}")
    
    return {
        'exact': result_exact,
        'estimator': result_estimator,
        'ground_state_energy': hamiltonian.ground_state_energy
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Energy estimation with Qiskit statevector")
    parser.add_argument("--hamiltonian", type=str, default=None,
                        help="Path to Hamiltonian file")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to Flow-Shadow checkpoint file")
    parser.add_argument("--results-dir", type=str, default=None,
                        help="Path to results directory (auto-finds checkpoint and hamiltonian)")
    parser.add_argument("--n-shots", type=int, default=10000,
                        help="Number of shots for Qiskit Estimator")
    parser.add_argument("--M", type=int, default=100,
                        help="Number of independent simulation runs")
    parser.add_argument("--demo", action="store_true",
                        help="Run demo with simple test cases")
    parser.add_argument("--convergence", action="store_true",
                        help="Run convergence study with different circuit budgets")
    
    args = parser.parse_args()
    
    if args.demo:
        demo_energy_estimation()
    elif args.results_dir is not None:
        # Auto-find checkpoint and hamiltonian from results directory
        results_dir = Path(args.results_dir)
        
        # Find the experiment directory (most recent)
        exp_dirs = sorted(results_dir.glob("experiment_*"))
        if not exp_dirs:
            print(f"No experiment directories found in {results_dir}")
            sys.exit(1)
        exp_dir = exp_dirs[-1]
        
        # Find checkpoint
        checkpoint_path = exp_dir / "checkpoint_update.pth"
        if not checkpoint_path.exists():
            print(f"Checkpoint not found: {checkpoint_path}")
            sys.exit(1)
        
        # Find hamiltonian from config
        config_path = exp_dir / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
            hamiltonian_path = Path(config.get("hamiltonian_path", ""))
            if not hamiltonian_path.exists():
                # Try relative to project root
                hamiltonian_path = Path(__file__).parent.parent / config.get("hamiltonian_path", "")
        else:
            print(f"Config not found: {config_path}")
            sys.exit(1)
        
        # Circuit budgets for convergence study
        circuit_budgets = [10, 25, 50, 100, 200, 500, 1000] if args.convergence else None
        
        test_with_checkpoint(
            checkpoint_path=checkpoint_path,
            hamiltonian_path=hamiltonian_path,
            M=args.M,
            circuit_budgets=circuit_budgets
        )
    elif args.checkpoint is not None and args.hamiltonian is not None:
        circuit_budgets = [10, 25, 50, 100, 200, 500, 1000] if args.convergence else None
        test_with_checkpoint(
            checkpoint_path=args.checkpoint,
            hamiltonian_path=args.hamiltonian,
            M=args.M,
            circuit_budgets=circuit_budgets
        )
    elif args.hamiltonian is not None:
        test_with_pauli_hamiltonian_helper(
            hamiltonian_path=args.hamiltonian,
            n_shots=args.n_shots
        )
    else:
        # Default: test with H2_STO3g_4qubits
        default_results = Path(__file__).parent.parent / "results_0" / "H2_STO3g_4qubits"
        if default_results.exists():
            print(f"Running test with default results: {default_results}")
            # Find experiment dir
            exp_dirs = sorted(default_results.glob("experiment_*"))
            if exp_dirs:
                exp_dir = exp_dirs[-1]
                checkpoint_path = exp_dir / "checkpoint_update.pth"
                config_path = exp_dir / "config.json"
                
                if checkpoint_path.exists() and config_path.exists():
                    with open(config_path) as f:
                        config = json.load(f)
                    hamiltonian_path = Path(__file__).parent.parent / config.get("hamiltonian_path", "")
                    
                    circuit_budgets = [10, 25, 50, 100, 200, 500] if args.convergence else None
                    test_with_checkpoint(
                        checkpoint_path=checkpoint_path,
                        hamiltonian_path=hamiltonian_path,
                        M=args.M,
                        circuit_budgets=circuit_budgets
                    )
                else:
                    demo_energy_estimation()
            else:
                demo_energy_estimation()
        else:
            print("No results directory found. Running demo...")
            demo_energy_estimation()

