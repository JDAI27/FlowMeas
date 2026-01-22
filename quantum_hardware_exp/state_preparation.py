"""
State preparation module.
Handles quantum state preparation for molecular systems.
"""

import numpy as np
from typing import Optional, Tuple
from pathlib import Path
from qiskit import QuantumCircuit
from qiskit.circuit.library import StatePreparation
from qiskit.quantum_info import Statevector
from qiskit.circuit.library import StatePreparation, TwoLocal
from qiskit.quantum_info import Statevector, SparsePauliOp
from scipy.optimize import minimize
import pickle
from qiskit.circuit.library import real_amplitudes


class StatePreparator:
    """Manages quantum state preparation."""
    
    def __init__(self, cache_dir: str = "cache/ground_states"):
        """
        Initialize state preparator.
        
        Args:
            cache_dir: Directory containing cached ground states
        """
        self.cache_dir = Path(cache_dir)

    def load_ground_state_hub_2x2_U2(self):
        """Load the optimized ansatz for 2x2 Hubbard model with U=2.

        First tries to load from pickle. If that fails (version incompatibility),
        rebuilds the ansatz from the known optimal parameters.
        """
        possible_paths = [
            Path("data/hub_res.pkl"),
            Path("quantum_hardware_exp/data/hub_res.pkl"),
        ]

        # Try loading from pickle first
        for path in possible_paths:
            if path.exists():
                try:
                    with open(path, 'rb') as file:
                        qc = pickle.load(file)
                    return qc
                except (ValueError, pickle.UnpicklingError) as e:
                    # Pickle incompatible with current Qiskit version
                    # Fall through to build from parameters
                    pass

        # Build ansatz from known optimal parameters
        return self._build_hub_2x2_U2_ansatz()

    def _build_hub_2x2_U2_ansatz(self):
        """Build the optimized ansatz for 2x2 Hubbard U=2 from scratch.

        Architecture:
        - 8 qubits
        - Layer 1: RX-RZ on each qubit (16 params)
        - CZ entanglement: (0,1), (2,3), (4,5), (6,7) then (1,2), (3,4), (5,6)
        - Layer 2: RX on each qubit (8 params)

        Optimal parameters found via VQE optimization achieving E ≈ -3.527 Ha
        """
        from qiskit.circuit import ParameterVector

        # Optimal parameters from VQE optimization
        optimal_params = np.array([
            3.83997831e-01,  1.00240740e-04,  7.53225813e-01,  2.04375783e-04,
            1.19081583e+00,  1.32265803e-05,  1.57117741e-01,  1.76796837e-03,
            1.57033269e-01,  4.84630113e-04,  1.19088668e+00,  2.43974551e-06,
            7.53080019e-01,  1.67981696e-04,  3.84051508e-01, -2.69691550e-20,
            1.05396922e+00,  8.66888314e-02, -2.67174854e-20,  4.79379735e-01,
            4.79229296e-01, -2.67418676e-20,  8.66300478e-02,  1.05401347e+00
        ])

        # Build parameterized circuit
        qc = QuantumCircuit(8)
        theta = ParameterVector('theta', 24)

        # First layer: RX-RZ on each qubit
        for i in range(8):
            qc.rx(theta[2*i], i)
            qc.rz(theta[2*i + 1], i)

        # CZ entanglement - pairs
        for i in range(4):
            qc.cz(2*i, 2*i + 1)

        # CZ entanglement - shifted pairs
        for i in range(3):
            qc.cz(2*i + 1, 2*i + 2)

        # Second layer: RX on each qubit
        for i in range(8):
            qc.rx(theta[16 + i], i)

        # Bind optimal parameters
        return qc.assign_parameters(optimal_params)
    
    def load_ground_state(
        self, 
        molecule: str = "H2", 
        n_qubits: int = 8
    ) -> Tuple[np.ndarray, float]:
        """
        Load ground state from cache.
        
        Args:
            molecule: Molecule name
            n_qubits: Number of qubits
            
        Returns:
            Tuple of (ground_state_vector, ground_state_energy)
        """
        # Try to find cached ground state
        possible_paths = [
            self.cache_dir / f"{molecule}_6-31G_{n_qubits}qubits_8q_7aa2c8c1041e5343_jw/ground_state.npz",
            self.cache_dir / f"{molecule}_{n_qubits}q/ground_state.npz",
            Path(f"cache/ground_states/{molecule}_6-31G_{n_qubits}qubits_8q_7aa2c8c1041e5343_jw/ground_state.npz"),
        ]
        
        for path in possible_paths:
            if path.exists():
                return self._load_npz_state(path)
        
        # If not found, return approximate state
        return self._create_approximate_state(molecule, n_qubits)
    
    def _load_npz_state(self, filepath: Path) -> Tuple[np.ndarray, float]:
        """
        Load state from NPZ file.
        
        Args:
            filepath: Path to NPZ file
            
        Returns:
            Tuple of (state_vector, energy)
        """
        data = np.load(filepath)
        
        if 'vector' in data:
            state = data['vector']
        elif 'ground_state' in data:
            state = data['ground_state']
        else:
            raise ValueError(f"No state vector found in {filepath}")
        
        energy = data.get('energy', -1.86)  # Default H2 energy
        
        
        return state, float(energy)
    
    def _create_approximate_state(
        self, 
        molecule: str, 
        n_qubits: int
    ) -> Tuple[np.ndarray, float]:
        """
        Create approximate ground state.
        
        Args:
            molecule: Molecule name
            n_qubits: Number of qubits
            
        Returns:
            Tuple of (state_vector, energy)
        """
        dim = 2**n_qubits
        state = np.zeros(dim, dtype=complex)
        
        if molecule == "H2" and n_qubits == 8:
            # H2 ground state approximation
            state[0b10001000] = 0.993  # Dominant configuration
            state[0b01000100] = -0.118  # First excited configuration
            energy = -1.860861
        else:
            # Generic Hartree-Fock state
            state[0] = 1.0
            energy = 0.0
        
        # Normalize
        state = state / np.linalg.norm(state)
        
        return state, energy
    
    def create_state_preparation_circuit(
        self, 
        state_vector: np.ndarray,
        n_qubits: Optional[int] = None
    ) -> QuantumCircuit:
        """
        Create circuit that prepares the given state.
        
        Args:
            state_vector: Target state vector
            n_qubits: Number of qubits (inferred if None)
            
        Returns:
            QuantumCircuit that prepares the state
        """
        if n_qubits is None:
            n_qubits = int(np.log2(len(state_vector)))
        
        circuit = QuantumCircuit(n_qubits)
        
        # Use Qiskit's StatePreparation
        state_prep = StatePreparation(state_vector)
        circuit.append(state_prep, range(n_qubits))
        
        # Verify preparation
        prepared_state = Statevector.from_instruction(circuit)
        fidelity = np.abs(np.vdot(state_vector, prepared_state.data))**2
        
        if fidelity < 0.999:
            print(f"Warning: State preparation fidelity = {fidelity:.6f}")
        
        return circuit
    
    def create_hartree_fock_circuit(
        self, 
        n_electrons: int, 
        n_qubits: int
    ) -> QuantumCircuit:
        """
        Create Hartree-Fock reference state.
        
        Args:
            n_electrons: Number of electrons
            n_qubits: Number of qubits (spin-orbitals)
            
        Returns:
            QuantumCircuit preparing HF state
        """
        circuit = QuantumCircuit(n_qubits)
        
        # Place electrons in lowest orbitals
        for i in range(n_electrons):
            circuit.x(i)
        
        return circuit
    
    def create_uccsd_ansatz(
        self, 
        n_electrons: int, 
        n_qubits: int,
        theta: Optional[np.ndarray] = None
    ) -> QuantumCircuit:
        """
        Create UCCSD ansatz circuit.
        
        Args:
            n_electrons: Number of electrons
            n_qubits: Number of qubits
            theta: Variational parameters
            
        Returns:
            UCCSD ansatz circuit
        """
        circuit = self.create_hartree_fock_circuit(n_electrons, n_qubits)
        
        if theta is None:
            # Default parameters
            theta = np.zeros(10)  # Simplified
        
        # Add excitation operators (simplified version)
        # In practice, would implement full UCCSD
        
        # Example: Double excitation
        if n_qubits >= 4:
            circuit.ry(theta[0], 0)
            circuit.cx(0, 1)
            circuit.cx(1, 2)
            circuit.cx(2, 3)
        
        return circuit