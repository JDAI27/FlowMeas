"""
Post-Hartree-Fock solver module for quantum chemistry calculations.

This module provides implementations of post-HF methods including:
- CCSD (Coupled Cluster Singles and Doubles)
- MP2 (Møller-Plesset Second-Order Perturbation Theory)
- CISD (Configuration Interaction Singles and Doubles)

These methods use one- and two-electron integrals extracted from
qubit Hamiltonians to perform high-accuracy quantum chemistry calculations.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional, Union, Any
import logging
from pyscf import gto, scf, ao2mo, cc, mp, ci
try:
    from .fci_solver import FCISolver
except ImportError:
    from fci_solver import FCISolver

logger = logging.getLogger(__name__)


class PostHFSolver(FCISolver):
    """Post-Hartree-Fock solver for quantum chemistry calculations.

    This class extends FCISolver to provide additional post-HF methods
    including CCSD, MP2, and CISD. It handles the conversion from qubit
    Hamiltonians to molecular integrals and runs the calculations using PySCF.
    """
    
    def __init__(self, pauli_strings: List[str], coefficients: List[complex], 
                 n_qubits: int, n_electrons: Tuple[int, int], 
                 mapping: str = 'jordan_wigner'):
        """
        Initialize the post-HF solver.

        Args:
            pauli_strings: List of Pauli strings
            coefficients: List of coefficients for each Pauli string
            n_qubits: Number of qubits (spin orbitals)
            n_electrons: Tuple of (n_alpha, n_beta) electrons
            mapping: Fermion-to-qubit mapping used
        """
        super().__init__(pauli_strings, coefficients, n_qubits, n_electrons, mapping)
        
        # HF reference state attributes
        self.hf_energy = None
        self.mo_coeff = None
        self.mo_energy = None
        self.mf = None  # PySCF mean-field object
        
        # Post-HF results storage
        self.ccsd_energy = None
        self.ccsd_t1 = None
        self.ccsd_t2 = None
        self.mp2_energy = None
        self.mp2_t2 = None
        self.cisd_energy = None
        self.cisd_civec = None
        
        logger.info(f"Initialized PostHFSolver for {n_qubits} qubits, "
                   f"{n_electrons} electrons, using {mapping} mapping")
    
    def _setup_pyscf_mol(self) -> gto.Mole:
        """Create a dummy PySCF molecule object for custom Hamiltonians."""
        mol = gto.M(verbose=4)
        mol.nelectron = sum(self.n_electrons)
        mol.spin = self.n_electrons[0] - self.n_electrons[1]
        mol.incore_anyway = True  # Essential for custom integrals
        return mol
    
    def run_hf(self, conv_tol: float = 1e-10, max_cycle: int = 100) -> float:
        """
        Run Hartree-Fock calculation as reference for post-HF methods.

        Args:
            conv_tol: Convergence tolerance
            max_cycle: Maximum number of SCF iterations

        Returns:
            HF energy
        """
        # Extract integrals if not done yet
        if self.h1e is None:
            H_ferm = self.qubit_to_fermion()
            self.extract_integrals(H_ferm)
        
        # Setup PySCF molecule
        mol = self._setup_pyscf_mol()
        
        # Create RHF/UHF object based on spin
        if mol.spin == 0:
            self.mf = scf.RHF(mol)
        else:
            self.mf = scf.UHF(mol)
        
        # Set custom integrals
        self.mf.get_hcore = lambda *args: self.h1e
        self.mf.get_ovlp = lambda *args: np.eye(self.n_spatial_orbs)
        
        # For two-electron integrals, we need to ensure proper formatting
        # PySCF expects 8-fold symmetry by default
        eri_ao = ao2mo.restore(8, self.g2e, self.n_spatial_orbs)
        self.mf._eri = eri_ao
        
        # Set convergence criteria
        self.mf.conv_tol = conv_tol
        self.mf.max_cycle = max_cycle
        
        # Run HF calculation
        logger.info(f"Running HF calculation for {self.n_spatial_orbs} spatial orbitals")
        self.hf_energy = self.mf.kernel()
        
        # Store results
        self.mo_coeff = self.mf.mo_coeff
        self.mo_energy = self.mf.mo_energy
        
        # Add nuclear repulsion
        self.hf_energy += self.e_core
        
        logger.info(f"HF converged: E = {self.hf_energy:.10f} Ha")
        
        return self.hf_energy
    
    def run_ccsd(self, conv_tol: float = 1e-7, max_cycle: int = 100,
                 diis_space: int = 8, **kwargs) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Run CCSD (Coupled Cluster Singles and Doubles) calculation.

        Args:
            conv_tol: Convergence tolerance for CC amplitudes
            max_cycle: Maximum number of CC iterations
            diis_space: DIIS space size
            **kwargs: Additional arguments for PySCF CCSD

        Returns:
            Tuple of (CCSD energy, T1 amplitudes, T2 amplitudes)
        """
        # Run HF if not done
        if self.mf is None:
            self.run_hf()
        
        # Create CCSD object
        mycc = cc.CCSD(self.mf)
        mycc.conv_tol = conv_tol
        mycc.max_cycle = max_cycle
        mycc.diis_space = diis_space
        
        # Apply any additional kwargs
        for key, value in kwargs.items():
            setattr(mycc, key, value)
        
        logger.info("Running CCSD calculation")
        
        # Run CCSD
        self.ccsd_energy, self.ccsd_t1, self.ccsd_t2 = mycc.kernel()
        
        # Total energy includes nuclear repulsion
        total_energy = self.hf_energy + self.ccsd_energy
        
        logger.info(f"CCSD converged: E_corr = {self.ccsd_energy:.10f} Ha")
        logger.info(f"CCSD total energy: E = {total_energy:.10f} Ha")
        
        # Store the solver for further analysis
        self.ccsd_solver = mycc
        
        return total_energy, self.ccsd_t1, self.ccsd_t2
    
    def run_mp2(self, frozen: Optional[int] = None) -> Tuple[float, np.ndarray]:
        """
        Run MP2 (Møller-Plesset Second-Order Perturbation Theory) calculation.

        Args:
            frozen: Number of frozen core orbitals

        Returns:
            Tuple of (MP2 energy, T2 amplitudes)
        """
        # Run HF if not done
        if self.mf is None:
            self.run_hf()
        
        # Create MP2 object
        mymp = mp.MP2(self.mf, frozen=frozen)
        
        logger.info("Running MP2 calculation")
        
        # Run MP2
        self.mp2_energy, self.mp2_t2 = mymp.kernel()
        
        # Total energy
        total_energy = self.hf_energy + self.mp2_energy
        
        logger.info(f"MP2 converged: E_corr = {self.mp2_energy:.10f} Ha")
        logger.info(f"MP2 total energy: E = {total_energy:.10f} Ha")
        
        # Store the solver
        self.mp2_solver = mymp
        
        return total_energy, self.mp2_t2
    
    def run_cisd(self, conv_tol: float = 1e-9, max_cycle: int = 100,
                 frozen: Optional[int] = None) -> Tuple[float, np.ndarray]:
        """
        Run CISD (Configuration Interaction Singles and Doubles) calculation.

        Args:
            conv_tol: Convergence tolerance
            max_cycle: Maximum number of iterations
            frozen: Number of frozen core orbitals

        Returns:
            Tuple of (CISD energy, CI vector)
        """
        # Run HF if not done
        if self.mf is None:
            self.run_hf()
        
        # Create CISD object
        myci = ci.CISD(self.mf, frozen=frozen)
        myci.conv_tol = conv_tol
        myci.max_cycle = max_cycle
        
        logger.info("Running CISD calculation")
        
        # Run CISD
        self.cisd_energy, self.cisd_civec = myci.kernel()
        
        # Total energy includes HF energy
        total_energy = self.cisd_energy
        
        logger.info(f"CISD converged: E = {total_energy:.10f} Ha")
        
        # Store the solver
        self.cisd_solver = myci
        
        return total_energy, self.cisd_civec
    
    def compute_ccsd_rdms(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute CCSD reduced density matrices.

        Returns:
            Tuple of (1-RDM, 2-RDM)
        """
        if self.ccsd_solver is None:
            raise ValueError("Must run CCSD calculation first")
        
        # Check for edge case: very small systems can cause numerical issues
        nocc = self.ccsd_solver.nocc
        nmo = self.ccsd_solver.nmo
        nvir = nmo - nocc
        
        # Edge case: no virtual orbitals or trivial system
        if nvir == 0 or nmo <= 1:
            logger.warning("System too small for CCSD RDMs, returning HF RDMs")
            # Return HF density matrix
            dm = self.mf.make_rdm1()
            # For 2-RDM, use antisymmetrized product of 1-RDM
            rdm2 = np.zeros((nmo, nmo, nmo, nmo))
            for i in range(nmo):
                for j in range(nmo):
                    for k in range(nmo):
                        for l in range(nmo):
                            rdm2[i,j,k,l] = dm[i,k] * dm[j,l] - dm[i,l] * dm[j,k]
            return dm, rdm2
        
        try:
            # Lambda equations for accurate RDMs
            l1, l2 = self.ccsd_solver.solve_lambda(self.ccsd_t1, self.ccsd_t2)
            
            # Compute RDMs
            rdm1 = self.ccsd_solver.make_rdm1(self.ccsd_t1, self.ccsd_t2, l1, l2)
            rdm2 = self.ccsd_solver.make_rdm2(self.ccsd_t1, self.ccsd_t2, l1, l2)
            
            return rdm1, rdm2
        except (ZeroDivisionError, ValueError) as e:
            logger.warning(f"Failed to compute CCSD RDMs: {e}. Returning HF RDMs.")
            # Fallback to HF RDMs
            dm = self.mf.make_rdm1()
            rdm2 = np.zeros((nmo, nmo, nmo, nmo))
            for i in range(nmo):
                for j in range(nmo):
                    for k in range(nmo):
                        for l in range(nmo):
                            rdm2[i,j,k,l] = dm[i,k] * dm[j,l] - dm[i,l] * dm[j,k]
            return dm, rdm2
    
    def compute_mp2_rdms(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute MP2 reduced density matrices.

        Returns:
            Tuple of (1-RDM, 2-RDM)
        """
        if self.mp2_solver is None:
            raise ValueError("Must run MP2 calculation first")
        
        try:
            rdm1 = self.mp2_solver.make_rdm1()
            rdm2 = self.mp2_solver.make_rdm2()
            return rdm1, rdm2
        except Exception as e:
            logger.warning(f"Failed to compute MP2 RDMs: {e}. Returning HF RDMs.")
            # Fallback to HF RDMs
            dm = self.mf.make_rdm1()
            nmo = self.mf.mo_coeff.shape[1]
            rdm2 = np.zeros((nmo, nmo, nmo, nmo))
            for i in range(nmo):
                for j in range(nmo):
                    for k in range(nmo):
                        for l in range(nmo):
                            rdm2[i,j,k,l] = dm[i,k] * dm[j,l] - dm[i,l] * dm[j,k]
            return dm, rdm2
    
    def compute_cisd_rdms(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute CISD reduced density matrices.

        Returns:
            Tuple of (1-RDM, 2-RDM)
        """
        if self.cisd_solver is None:
            raise ValueError("Must run CISD calculation first")
        
        try:
            rdm1 = self.cisd_solver.make_rdm1(self.cisd_civec)
            rdm2 = self.cisd_solver.make_rdm2(self.cisd_civec)
            return rdm1, rdm2
        except Exception as e:
            logger.warning(f"Failed to compute CISD RDMs: {e}. Returning HF RDMs.")
            # Fallback to HF RDMs
            dm = self.mf.make_rdm1()
            nmo = self.mf.mo_coeff.shape[1]
            rdm2 = np.zeros((nmo, nmo, nmo, nmo))
            for i in range(nmo):
                for j in range(nmo):
                    for k in range(nmo):
                        for l in range(nmo):
                            rdm2[i,j,k,l] = dm[i,k] * dm[j,l] - dm[i,l] * dm[j,k]
            return dm, rdm2
    
    def compare_methods(self) -> Dict[str, Dict[str, Any]]:
        """
        Run and compare all available post-HF methods.

        Returns:
            Dictionary with results from each method
        """
        results = {}
        
        # FCI (if system is small enough)
        if self.n_spatial_orbs <= 8:  # Practical limit for FCI
            try:
                fci_energy, fci_vec = self.run_fci()
                results['FCI'] = {
                    'energy': fci_energy,
                    'method': 'Full CI',
                    'variational': True,
                    'size_extensive': False
                }
            except Exception as e:
                logger.warning(f"FCI failed: {e}")
        
        # CCSD
        try:
            ccsd_energy, _, _ = self.run_ccsd()
            results['CCSD'] = {
                'energy': ccsd_energy,
                'correlation_energy': self.ccsd_energy,
                'method': 'Coupled Cluster Singles and Doubles',
                'variational': False,
                'size_extensive': True
            }
        except Exception as e:
            logger.warning(f"CCSD failed: {e}")
        
        # MP2
        try:
            mp2_energy, _ = self.run_mp2()
            results['MP2'] = {
                'energy': mp2_energy,
                'correlation_energy': self.mp2_energy,
                'method': 'Møller-Plesset Second Order',
                'variational': False,
                'size_extensive': True
            }
        except Exception as e:
            logger.warning(f"MP2 failed: {e}")
        
        # CISD
        try:
            cisd_energy, _ = self.run_cisd()
            results['CISD'] = {
                'energy': cisd_energy,
                'method': 'Configuration Interaction Singles and Doubles',
                'variational': True,
                'size_extensive': False
            }
        except Exception as e:
            logger.warning(f"CISD failed: {e}")
        
        return results


def pauli_strings_to_post_hf(pauli_strings: List[str], 
                            coefficients: List[Union[float, complex]], 
                            n_qubits: int,
                            n_electrons: Union[int, Tuple[int, int]],
                            method: str = 'ccsd',
                            mapping: str = 'jordan_wigner',
                            **kwargs) -> Dict[str, Any]:
    """
    Convenience function to compute post-HF energy from Pauli strings.

    Args:
        pauli_strings: List of Pauli strings
        coefficients: List of coefficients
        n_qubits: Number of qubits (spin orbitals)
        n_electrons: Total number of electrons or tuple of (n_alpha, n_beta)
        method: Post-HF method to use ('ccsd', 'mp2', 'cisd', 'all')
        mapping: Fermion-to-qubit mapping
        **kwargs: Additional arguments for the specific method

    Returns:
        Dictionary with results from the calculation
    """
    # Handle n_electrons input
    if isinstance(n_electrons, int):
        n_alpha = (n_electrons + 1) // 2
        n_beta = n_electrons // 2
        n_electrons = (n_alpha, n_beta)
    
    # Create solver
    solver = PostHFSolver(pauli_strings, coefficients, n_qubits, n_electrons, mapping)
    
    # Run calculations based on method
    if method.lower() == 'all':
        return solver.compare_methods()
    
    elif method.lower() == 'ccsd':
        energy, t1, t2 = solver.run_ccsd(**kwargs)
        rdm1, rdm2 = solver.compute_ccsd_rdms()
        return {
            'energy': energy,
            'correlation_energy': solver.ccsd_energy,
            'hf_energy': solver.hf_energy,
            't1_amplitudes': t1,
            't2_amplitudes': t2,
            'rdm1': rdm1,
            'rdm2': rdm2,
            'solver': solver
        }
    
    elif method.lower() == 'mp2':
        energy, t2 = solver.run_mp2(**kwargs)
        rdm1, rdm2 = solver.compute_mp2_rdms()
        return {
            'energy': energy,
            'correlation_energy': solver.mp2_energy,
            'hf_energy': solver.hf_energy,
            't2_amplitudes': t2,
            'rdm1': rdm1,
            'rdm2': rdm2,
            'solver': solver
        }
    
    elif method.lower() == 'cisd':
        energy, civec = solver.run_cisd(**kwargs)
        rdm1, rdm2 = solver.compute_cisd_rdms()
        return {
            'energy': energy,
            'hf_energy': solver.hf_energy,
            'ci_vector': civec,
            'rdm1': rdm1,
            'rdm2': rdm2,
            'solver': solver
        }
    
    else:
        raise ValueError(f"Unknown method: {method}. Choose from 'ccsd', 'mp2', 'cisd', or 'all'")