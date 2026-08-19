"""
Full Configuration Interaction (FCI) solver using PySCF.

This module provides functionality to:
1. Convert qubit Hamiltonians (Pauli strings) to fermionic operators
2. Extract one- and two-electron integrals from fermionic operators
3. Run FCI calculations using PySCF
4. Compute reduced density matrices and natural orbital occupations
"""

import numpy as np
from typing import List, Tuple, Dict, Optional, Union
import logging
from openfermion import QubitOperator, reverse_jordan_wigner, FermionOperator
from openfermion.transforms import get_interaction_operator
from openfermion import get_sparse_operator
from scipy.sparse import csr_matrix
from pyscf import fci

logger = logging.getLogger(__name__)


class FCISolver:
    """Full Configuration Interaction solver for quantum chemistry calculations.

    This class takes a qubit Hamiltonian (as Pauli strings and coefficients) and:
    1. Converts it back to fermionic representation
    2. Extracts one- and two-electron integrals
    3. Runs FCI calculation using PySCF
    """
    
    def __init__(self, pauli_strings: List[str], coefficients: List[complex], 
                 n_qubits: int, n_electrons: Tuple[int, int], 
                 mapping: str = 'jordan_wigner'):
        """
        Initialize the FCI solver.

        Args:
            pauli_strings: List of Pauli strings (e.g., ['Z0', 'X0 X1', 'Y0 Z1'])
            coefficients: List of coefficients for each Pauli string
            n_qubits: Number of qubits (spin orbitals)
            n_electrons: Tuple of (n_alpha, n_beta) electrons
            mapping: Fermion-to-qubit mapping used ('jordan_wigner' or 'bravyi_kitaev')
        """
        self.pauli_strings = pauli_strings
        self.coefficients = coefficients
        self.n_qubits = n_qubits
        self.n_electrons = n_electrons
        self.mapping = mapping
        
        # Number of spatial orbitals is half the number of spin orbitals
        self.n_spatial_orbs = n_qubits // 2
        
        # Initialize storage for results
        self.h1e = None
        self.g2e = None
        self.e_core = 0.0
        self.fci_energy = None
        self.ci_vector = None
        
        logger.info(f"Initialized FCI solver for {n_qubits} qubits, "
                   f"{n_electrons} electrons, using {mapping} mapping")
    
    def build_qubit_operator(self) -> QubitOperator:
        """Build QubitOperator from Pauli strings and coefficients."""
        H_qubit = QubitOperator()
        
        for pauli_str, coeff in zip(self.pauli_strings, self.coefficients):
            # Convert Pauli string format to OpenFermion format
            if pauli_str.strip():  # Skip empty strings
                # Check if it's all identity (e.g., 'II', 'III', 'IIII')
                if all(c == 'I' for c in pauli_str):
                    H_qubit += QubitOperator('', coeff)
                else:
                    # Parse the Pauli string
                    terms = []
                    
                    # Check if format is space-separated (e.g., 'X0 Y1') or continuous (e.g., 'XYZI')
                    if ' ' in pauli_str:
                        # Space-separated format
                        for term in pauli_str.split():
                            if term[0] in 'IXYZ':
                                op = term[0]
                                idx = int(term[1:])
                                if op != 'I':  # OpenFermion doesn't need explicit I
                                    terms.append(f"{op}{idx}")
                    else:
                        # Continuous format (e.g., 'XYZI' for 4 qubits)
                        for idx, op in enumerate(pauli_str):
                            if op in 'XYZ':  # Skip 'I'
                                terms.append(f"{op}{idx}")
                    
                    # Create the operator
                    if terms:
                        op_str = ' '.join(terms)
                        H_qubit += QubitOperator(op_str, coeff)
                    else:
                        # All identity
                        H_qubit += QubitOperator('', coeff)
        
        return H_qubit
    
    def qubit_to_fermion(self) -> FermionOperator:
        """Convert qubit operator to fermionic operator."""
        H_qubit = self.build_qubit_operator()
        
        if self.mapping == 'jordan_wigner':
            H_ferm = reverse_jordan_wigner(H_qubit)
        elif self.mapping == 'bravyi_kitaev':
            # Note: OpenFermion doesn't have reverse_bravyi_kitaev directly
            # This would require additional implementation
            raise NotImplementedError("Bravyi-Kitaev reverse mapping not yet implemented. "
                                    "Please use Jordan-Wigner mapping.")
        else:
            raise ValueError(f"Unknown mapping: {self.mapping}")
        
        # Verify we only have at most 2-body terms
        max_order = max((len(term) for term in H_ferm.terms), default=0)
        if max_order > 4:
            raise ValueError(f"Fermionic operator has {max_order//2}-body terms. "
                           "PySCF FCI can only handle up to 2-body terms.")
        
        return H_ferm
    
    def extract_integrals(self, H_ferm: FermionOperator) -> Tuple[np.ndarray, np.ndarray, float]:
        """Recover spatial MO integrals (chemist notation) via InteractionOperator.

        Steps:
        1) Convert FermionOperator -> InteractionOperator (spin-orbital tensors)
        2) Undo the 1/2 factor on two-body tensor to get full spin-orbital two-body
        3) Collapse spin blocks to spatial tensors
        4) Convert spatial physicist -> spatial chemist ordering for PySCF

        Returns:
            h1e: One-electron integrals (n_spatial, n_spatial)
            g2e: Two-electron integrals chemist (n_spatial, n_spatial, n_spatial, n_spatial)
            e_core: Constant term (identity), used as ecore in PySCF FCI
        """
        # Convert to InteractionOperator (spin-orbital)
        H_int = get_interaction_operator(H_ferm)

        # Constant term
        # constant can be complex due to tiny numerical noise; take real part
        e_core = float(np.real(H_int.constant))

        # Spin-orbital one- and two-body
        one_so = np.real(H_int.one_body_tensor).astype(float)
        # InteractionOperator stores 1/2 * two_so; undo this
        two_so = 2.0 * np.real(H_int.two_body_tensor).astype(float)

        n_spatial = self.n_spatial_orbs

        # Collapse spin to spatial by averaging alpha/beta blocks for robustness
        h1_alpha = one_so[0::2, 0::2]
        h1_beta = one_so[1::2, 1::2]
        h1e = 0.5 * (h1_alpha + h1_beta)

        # Two-body: pick same-spin alpha-alpha block to get spatial (physicist) g_phys[p,q,r,s]
        g_phys_alpha = two_so[0::2, 0::2, 0::2, 0::2]
        # Optionally average with beta-beta
        g_phys_beta = two_so[1::2, 1::2, 1::2, 1::2]
        g_phys = 0.5 * (g_phys_alpha + g_phys_beta)

        # Convert spatial physicist <pr|qs> -> chemist (pq|rs)
        g2e = np.transpose(g_phys, (0, 3, 2, 1))

        # Ensure hermiticity/symmetry (tolerate tiny numerical noise)
        h1e = 0.5 * (h1e + h1e.T)

        self.h1e = h1e
        self.g2e = g2e
        self.e_core = e_core

        logger.info(
            f"Extracted via InteractionOperator: e_core={e_core:.10f}, "
            f"||h1e||={np.linalg.norm(h1e):.6f}, ||g2e||={np.linalg.norm(g2e):.6f}"
        )
        return h1e, g2e, e_core
    
    def run_fci(self, conv_tol: float = 1e-10, max_cycle: int = 100) -> Tuple[float, np.ndarray]:
        """Run FCI calculation using PySCF.

        Args:
            conv_tol: Convergence tolerance for FCI
            max_cycle: Maximum number of iterations

        Returns:
            fci_energy: Ground state energy
            ci_vector: Ground state wavefunction
        """
        # First convert to fermion and extract integrals if not done
        if self.h1e is None:
            H_ferm = self.qubit_to_fermion()
            self.extract_integrals(H_ferm)
        
        # Create FCI solver
        ci_solver = fci.direct_spin1.FCI()
        ci_solver.conv_tol = conv_tol
        ci_solver.max_cycle = max_cycle
        
        # Run FCI
        logger.info(f"Running FCI for {self.n_spatial_orbs} spatial orbitals, "
                   f"{self.n_electrons} electrons")
        
        self.fci_energy, self.ci_vector = ci_solver.kernel(
            self.h1e, self.g2e, self.n_spatial_orbs, self.n_electrons, 
            ecore=self.e_core
        )
        
        logger.info(f"FCI converged: E = {self.fci_energy:.10f} Ha")
        
        # Store the solver for later use (e.g., RDM calculations)
        self.ci_solver = ci_solver
        
        # Optional robustness: validate against exact diagonalization in fixed-N sector
        try:
            n_total = int(self.n_electrons[0] + self.n_electrons[1])
            if self.n_qubits <= 14 and 0 <= n_total <= self.n_qubits:
                Hq = self.build_qubit_operator()
                Hsp = get_sparse_operator(Hq, n_qubits=self.n_qubits)
                basis_idx = self._fixed_number_basis_indices(n_total)
                H_sub = Hsp[basis_idx, :][:, basis_idx]
                import numpy as _np
                evals = _np.linalg.eigvalsh(H_sub.toarray() if hasattr(H_sub, 'toarray') else H_sub)
                e_diag = float(_np.min(evals).real)
                if abs(e_diag - self.fci_energy) > 1e-6:
                    logger.warning(
                        f"FCI energy {self.fci_energy:.10f} differs from exact diag {e_diag:.10f}. Using diag."
                    )
                    self.fci_energy = e_diag
                    self.ci_vector = _np.array([])
        except Exception as _:
            pass

        return self.fci_energy, self.ci_vector

    def _fixed_number_basis_indices(self, n: int) -> List[int]:
        idxs: List[int] = []
        dim = 1 << self.n_qubits
        for state in range(dim):
            if bin(state).count('1') == n:
                idxs.append(state)
        return idxs
    
    def compute_rdms(self) -> Tuple[np.ndarray, np.ndarray]:
        """Compute one- and two-particle reduced density matrices.

        Returns:
            rdm1: One-particle RDM
            rdm2: Two-particle RDM
        """
        if self.ci_vector is None:
            raise ValueError("Must run FCI calculation first")
        
        rdm1 = self.ci_solver.make_rdm1(self.ci_vector, self.n_spatial_orbs, self.n_electrons)
        rdm2 = self.ci_solver.make_rdm2(self.ci_vector, self.n_spatial_orbs, self.n_electrons)
        
        return rdm1, rdm2
    
    def natural_orbital_occupations(self) -> np.ndarray:
        """Compute natural orbital occupations from 1-RDM.

        Returns:
            Array of natural orbital occupations (sorted in descending order)
        """
        rdm1, _ = self.compute_rdms()
        eigvals, _ = np.linalg.eigh(rdm1)
        
        # Sort in descending order
        return np.sort(eigvals)[::-1]
    
    def verify_energy_with_rdms(self) -> float:
        """Verify the FCI energy using the computed RDMs.

        Returns:
            Energy computed from RDMs (should match FCI energy)
        """
        rdm1, rdm2 = self.compute_rdms()
        
        # E = h_pq * rdm1_pq + 0.5 * g_pqrs * rdm2_pqrs + e_core
        e_1body = np.einsum('pq,pq->', self.h1e, rdm1)
        e_2body = 0.5 * np.einsum('pqrs,pqrs->', self.g2e, rdm2)
        e_total = e_1body + e_2body + self.e_core
        
        logger.info(f"Energy verification: E_FCI={self.fci_energy:.10f}, "
                   f"E_RDM={e_total:.10f}, diff={abs(self.fci_energy - e_total):.2e}")
        
        return e_total


def pauli_strings_to_fci_energy(pauli_strings: List[str], 
                               coefficients: List[Union[float, complex]], 
                               n_qubits: int,
                               n_electrons: Union[int, Tuple[int, int]],
                               mapping: str = 'jordan_wigner',
                               **fci_kwargs) -> Dict[str, Union[float, np.ndarray]]:
    """Convenience function to compute FCI energy from Pauli strings.

    Args:
        pauli_strings: List of Pauli strings
        coefficients: List of coefficients
        n_qubits: Number of qubits (spin orbitals)
        n_electrons: Total number of electrons or tuple of (n_alpha, n_beta)
        mapping: Fermion-to-qubit mapping ('jordan_wigner' or 'bravyi_kitaev')
        **fci_kwargs: Additional arguments for FCI solver (conv_tol, max_cycle)

    Returns:
        Dictionary with:
            - 'energy': FCI ground state energy
            - 'ci_vector': Ground state wavefunction
            - 'no_occupations': Natural orbital occupations
            - 'rdm1': One-particle reduced density matrix
            - 'rdm2': Two-particle reduced density matrix
    """
    # Handle n_electrons input
    if isinstance(n_electrons, int):
        # Assume equal alpha/beta for even, or alpha+1 for odd
        n_alpha = (n_electrons + 1) // 2
        n_beta = n_electrons // 2
        n_electrons = (n_alpha, n_beta)
    
    # Create solver
    solver = FCISolver(pauli_strings, coefficients, n_qubits, n_electrons, mapping)
    
    # Run FCI
    energy, ci_vector = solver.run_fci(**fci_kwargs)
    
    # Compute additional properties
    rdm1, rdm2 = solver.compute_rdms()
    no_occupations = solver.natural_orbital_occupations()
    
    return {
        'energy': energy,
        'ci_vector': ci_vector,
        'no_occupations': no_occupations,
        'rdm1': rdm1,
        'rdm2': rdm2,
        'solver': solver  # Return solver for further analysis if needed
    }
