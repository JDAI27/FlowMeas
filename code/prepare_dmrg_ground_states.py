#!/usr/bin/env python3
"""
Prepare ground states for all configured experiments.

This script:
1. Reads all Hamiltonians in the molecule configuration below
2. Computes ground states using best available method (FCI for small, sparse for larger)
3. Caches them in cache/ground_states/ directory
4. Reports progress and any errors

Run this script BEFORE submitting cluster jobs to ensure all ground states
are pre-computed and cached.

Methods used:
- FCI (Full Configuration Interaction): For systems ≤14 qubits - exact within basis
- Sparse diagonalization: For larger systems ≤16 qubits
- For systems >16 qubits: May require external DMRG or run without cache
"""

import sys
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import time
import json
from datetime import datetime

# Add code directory to path
sys.path.insert(0, str(Path(__file__).parent))

from pauli_hamiltonian_helper import PauliHamiltonianHelper

# DMRG is provided by the TeNPy-backed solver (code/tenpy_dmrg.py), wired into
# PauliHamiltonianHelper.compute_ground_state(method='dmrg') directly.

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Molecule configurations
MOLECULE_CONFIGS = {
    "H2_STO3g_4qubits": {
        "n_electrons": 2,
        "basis": "STO-3G",
        "description": "H2 with minimal basis"
    },
    "H2_6-31G_8qubits": {
        "n_electrons": 2,
        "basis": "6-31G",
        "description": "H2 with larger basis"
    },
    "LiH_STO3g_12qubits": {
        "n_electrons": 4,
        "basis": "STO-3G",
        "description": "LiH molecule"
    },
    "BeH2_STO3g_14qubits": {
        "n_electrons": 6,
        "basis": "STO-3G",
        "description": "BeH2 molecule"
    },
    "H2O_STO3g_14qubits": {
        "n_electrons": 10,
        "basis": "STO-3G",
        "description": "Water molecule"
    },
    "NH3_STO3g_16qubits": {
        "n_electrons": 10,
        "basis": "STO-3G",
        "description": "Ammonia molecule"
    },
    "C2_STO3g_20qubits": {
        "n_electrons": 12,
        "basis": "STO-3G",
        "description": "C2 molecule"
    },
    "HCl_STO3g_20qubits": {
        "n_electrons": 18,
        "basis": "STO-3G",
        "description": "HCl molecule"
    }
}

# Method selection based on system size
METHOD_SELECTION = {
    "small": {  # 4-8 qubits
        "methods": ["dense", "fci"],
        "description": "Small systems - use dense or FCI"
    },
    "medium": {  # 12-14 qubits
        "methods": ["fci", "sparse"],
        "description": "Medium systems - use FCI or sparse"
    },
    "large": {  # 16-20 qubits
        "methods": ["sparse", "lobpcg"],
        "description": "Large systems - use sparse or LOBPCG"
    }
}


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent


def get_best_method_for_system(n_qubits: int, n_electrons: int) -> str:
    """
    Determine the best method based on system size.

    Returns:
        Method name: 'dense' or 'auto' (sparse/dense)
    """
    if n_qubits <= 12:
        # Use dense diagonalization for small systems (exact and fast)
        return 'dense'
    else:
        # For larger systems, use sparse diagonalization
        return 'auto'


def find_hamiltonian_files() -> List[Tuple[str, Path]]:
    """Find all Hamiltonian files for configured molecules.

    Looks for both jw.txt and jw_generated.txt files.
    If both exist, processes both.
    """
    project_root = get_project_root()
    hamiltonians_dir = project_root / "Hamiltonians"

    if not hamiltonians_dir.exists():
        raise FileNotFoundError(f"Hamiltonians directory not found: {hamiltonians_dir}")

    found_hamiltonians = []

    for molecule_name in MOLECULE_CONFIGS.keys():
        molecule_dir = hamiltonians_dir / molecule_name

        if not molecule_dir.exists():
            logger.warning(f"Directory not found for {molecule_name}: {molecule_dir}")
            continue

        # Look for both jw.txt and jw_generated.txt
        jw_file = molecule_dir / "jw.txt"
        jw_generated_file = molecule_dir / "jw_generated.txt"

        files_found = []
        if jw_file.exists():
            files_found.append((molecule_name, jw_file))
        if jw_generated_file.exists():
            # Use a different name to distinguish in cache
            files_found.append((f"{molecule_name}_generated", jw_generated_file))

        if files_found:
            found_hamiltonians.extend(files_found)
        else:
            logger.warning(f"No JW Hamiltonian file found in {molecule_dir}")

    return found_hamiltonians


def compute_dmrg_ground_state(
    hamiltonian_path: Path,
    molecule_name: str,
    n_electrons: int,
    mapping: str = "jordan_wigner",
    use_cache: bool = True,
    force_recompute: bool = False,
    maxM: int = None
) -> Dict:
    """
    Compute DMRG ground state for a given Hamiltonian.

    Args:
        hamiltonian_path: Path to Hamiltonian file
        molecule_name: Name of molecule
        n_electrons: Number of electrons
        mapping: Fermion-to-qubit mapping
        use_cache: Whether to use cached results
        force_recompute: Force recomputation even if cached
        maxM: Maximum bond dimension (if None, determined automatically)

    Returns:
        Dictionary with results
    """
    logger.info(f"\n{'='*70}")
    logger.info(f"Processing: {molecule_name}")
    logger.info(f"File: {hamiltonian_path}")
    logger.info(f"{'='*70}")

    try:
        # Initialize helper
        helper = PauliHamiltonianHelper(hamiltonian_path)
        n_qubits = helper.n_qubits
        n_terms = len(helper.pauli_str_list)

        logger.info(f"  Qubits: {n_qubits}")
        logger.info(f"  Terms: {n_terms}")
        logger.info(f"  Electrons: {n_electrons}")

        # Determine best method for this system size
        best_method = get_best_method_for_system(n_qubits, n_electrons)
        logger.info(f"  Best method: {best_method}")

        # Check if already cached
        cache_path = helper._get_cache_path()
        logger.info(f"  Cache path: {cache_path}")

        # Try to compute/load ground state
        start_time = time.time()

        # For small systems (≤12 qubits), use dense diagonalization which is exact
        if best_method == 'dense':
            logger.info(f"\n  Using dense diagonalization (exact)...")
            try:
                energy, vector = helper.compute_ground_state(
                    sparse=False,
                    method='dense',
                    use_cache=use_cache,
                    force_recompute=force_recompute
                )
                computation_time = time.time() - start_time

                logger.info(f"  ✓ Dense diagonalization successful!")
                logger.info(f"    Ground state energy: {energy:.10f} Ha")
                logger.info(f"    Computation time: {computation_time:.2f} s")

                # Verify against exact energy if available
                exact_energies = helper.get_exact_energy_from_file()
                if exact_energies and 'electronic_energy' in exact_energies:
                    exact_energy = exact_energies['electronic_energy']
                    energy_diff = abs(energy - exact_energy)
                    logger.info(f"    Exact energy: {exact_energy:.10f} Ha")
                    logger.info(f"    Energy difference: {energy_diff:.10e} Ha")

                return {
                    "molecule": molecule_name,
                    "n_qubits": n_qubits,
                    "n_terms": n_terms,
                    "n_electrons": n_electrons,
                    "method": "dense_diagonalization",
                    "energy": energy,
                    "computation_time": computation_time,
                    "success": True,
                    "cache_path": str(cache_path),
                    "exact_energies": exact_energies
                }
            except Exception as e:
                logger.warning(f"  Dense diagonalization failed: {e}")
                logger.info(f"  Falling back to sparse diagonalization...")
                best_method = 'auto'

        # For larger systems, use sparse diagonalization
        if best_method in ['auto', 'sparse', 'lobpcg']:
            logger.info(f"\n  Using sparse matrix diagonalization...")

            try:
                energy, vector = helper.compute_ground_state(
                    sparse=True,
                    method='auto',
                    use_cache=use_cache,
                    force_recompute=force_recompute
                )
                computation_time = time.time() - start_time

                logger.info(f"  ✓ Sparse diagonalization successful!")
                logger.info(f"    Ground state energy: {energy:.10f} Ha")
                logger.info(f"    Computation time: {computation_time:.2f} s")

                # Verify against exact energy if available
                exact_energies = helper.get_exact_energy_from_file()
                if exact_energies and 'electronic_energy' in exact_energies:
                    exact_energy = exact_energies['electronic_energy']
                    energy_diff = abs(energy - exact_energy)
                    logger.info(f"    Exact energy: {exact_energy:.10f} Ha")
                    logger.info(f"    Energy difference: {energy_diff:.10e} Ha")

                return {
                    "molecule": molecule_name,
                    "n_qubits": n_qubits,
                    "n_terms": n_terms,
                    "n_electrons": n_electrons,
                    "method": "sparse_diagonalization",
                    "energy": energy,
                    "computation_time": computation_time,
                    "success": True,
                    "cache_path": str(cache_path),
                    "exact_energies": exact_energies
                }
            except MemoryError as e:
                logger.error(f"  ✗ Memory error: {e}")
                logger.error(f"    System too large for direct methods")
                logger.error(f"    Consider using external DMRG library (ITensor, dmrghandler)")
                return {
                    "molecule": molecule_name,
                    "n_qubits": n_qubits,
                    "success": False,
                    "error": "MemoryError",
                    "message": str(e)
                }
            except Exception as e:
                logger.error(f"  Sparse diagonalization failed: {e}")
                import traceback
                logger.error(traceback.format_exc())
                return {
                    "molecule": molecule_name,
                    "n_qubits": n_qubits,
                    "success": False,
                    "error": "Sparse_diag_failed",
                    "message": str(e)
                }

    except Exception as e:
        logger.error(f"  ✗ Error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {
            "molecule": molecule_name,
            "success": False,
            "error": type(e).__name__,
            "message": str(e)
        }


def main():
    """Main function to prepare all DMRG ground states."""
    print("=" * 70)
    print("DMRG Ground State Preparation")
    print("=" * 70)
    print(f"\nStart time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    project_root = get_project_root()
    cache_dir = project_root / "cache" / "ground_states"
    cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nProject root: {project_root}")
    print(f"Cache directory: {cache_dir}")

    # Find all Hamiltonian files
    print("\nSearching for Hamiltonian files...")
    try:
        hamiltonians = find_hamiltonian_files()
    except FileNotFoundError as e:
        logger.error(f"Error: {e}")
        return 1

    if not hamiltonians:
        logger.error("No Hamiltonian files found!")
        return 1

    print(f"Found {len(hamiltonians)} Hamiltonian file(s)")

    # Process each Hamiltonian
    results = []
    successful = 0
    failed = 0

    total_start_time = time.time()

    for i, (molecule_name, hamiltonian_path) in enumerate(hamiltonians, 1):
        print(f"\n{'='*70}")
        print(f"[{i}/{len(hamiltonians)}] {molecule_name}")
        print(f"{'='*70}")

        # Get molecule configuration
        # Strip _generated suffix to get base molecule name
        base_molecule_name = molecule_name.replace("_generated", "")
        molecule_config = MOLECULE_CONFIGS.get(base_molecule_name)

        if molecule_config is None:
            logger.error(f"Configuration not found for {base_molecule_name}")
            continue

        n_electrons = molecule_config["n_electrons"]

        # Compute ground state
        result = compute_dmrg_ground_state(
            hamiltonian_path=hamiltonian_path,
            molecule_name=molecule_name,
            n_electrons=n_electrons,
            mapping="jordan_wigner",
            use_cache=True,
            force_recompute=False
        )

        results.append(result)

        if result["success"]:
            successful += 1
            logger.info(f"  ✓ Success")
        else:
            failed += 1
            logger.error(f"  ✗ Failed")

    total_time = time.time() - total_start_time

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\nTotal time: {total_time:.2f} s")
    print(f"Successful: {successful}/{len(hamiltonians)}")
    print(f"Failed: {failed}/{len(hamiltonians)}")

    # Save results to JSON
    results_file = project_root / "cache" / "ground_states_preparation_log.json"
    results_data = {
        "timestamp": datetime.now().isoformat(),
        "total_time": total_time,
        "successful": successful,
        "failed": failed,
        "results": results
    }

    with open(results_file, 'w') as f:
        json.dump(results_data, f, indent=2)

    print(f"\nResults saved to: {results_file}")

    # Print details for successful computations
    if successful > 0:
        print("\n" + "=" * 70)
        print("SUCCESSFUL COMPUTATIONS")
        print("=" * 70)
        for result in results:
            if result["success"]:
                print(f"\n{result['molecule']}:")
                print(f"  Method: {result['method']}")
                print(f"  Energy: {result['energy']:.10f} Ha")
                print(f"  Time: {result['computation_time']:.2f} s")
                print(f"  Cache: {result['cache_path']}")

    # Print details for failed computations
    if failed > 0:
        print("\n" + "=" * 70)
        print("FAILED COMPUTATIONS")
        print("=" * 70)
        for result in results:
            if not result["success"]:
                print(f"\n{result['molecule']}:")
                print(f"  Error: {result.get('error', 'Unknown')}")
                print(f"  Message: {result.get('message', 'No message')}")

    # Final notes
    print("\n" + "=" * 70)
    print("NEXT STEPS")
    print("=" * 70)
    if failed > 0:
        print("\nSome computations failed. For large systems (>14 qubits):")
        print("  - Memory errors are expected for direct diagonalization")
        print("  - Consider using external DMRG libraries (ITensor, TenPy, etc.)")
        print("  - Or run experiments without pre-cached ground states")
    else:
        print("\n✓ All ground states computed successfully!")
        print("\nYou can now launch training runs with:")
        print("  python code/run_config.py --config <config.json>")

    print("=" * 70)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    exit(main())
