#!/usr/bin/env python3
"""
Generate Fermi-Hubbard model Hamiltonians and save in the same format as molecular Hamiltonians.

This script generates Hubbard model Hamiltonians for various lattice configurations
and saves them in JSON format compatible with the GFlowNet training pipeline.

The Fermi-Hubbard model describes interacting fermions on a lattice:
H = -t Σ_<i,j> (c†_i c_j + h.c.) + U Σ_i n_i↑ n_i↓ + μ Σ_i n_i

where:
- t: hopping parameter (uniform_interaction in Qiskit)
- U: on-site interaction (onsite_interaction in Qiskit)
- μ: chemical potential (uniform_onsite_potential in Qiskit)
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np

try:
    from qiskit_nature.second_q.hamiltonians.lattices import (
        LineLattice,
        SquareLattice,
        BoundaryCondition
    )
    from qiskit_nature.second_q.hamiltonians import FermiHubbardModel
    from qiskit_nature.second_q.mappers import JordanWignerMapper, ParityMapper, BravyiKitaevMapper
    from qiskit_nature.second_q.operators import FermionicOp
except ImportError:
    print("Error: Qiskit Nature is required. Install with:")
    print("  pip install qiskit-nature")
    exit(1)


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent


def pauli_op_to_json_format(pauli_op) -> Dict:
    """
    Convert a Qiskit Pauli operator to the JSON format used for molecular Hamiltonians.

    Args:
        pauli_op: Qiskit SparsePauliOp or similar

    Returns:
        Dictionary in the format: {"paulis": [{"label": "XXYY", "coeff": {"real": 0.5, "imag": 0.0}},...]}
    """
    paulis_list = []

    # SparsePauliOp has.paulis (PauliList) and.coeffs (array) attributes
    for pauli_term, coeff in zip(pauli_op.paulis, pauli_op.coeffs):
        # Use to_label() instead of str() — str() truncates long Paulis
        # with "..." which corrupts the label for 50+ qubit systems.
        pauli_string = pauli_term.to_label()

        # Convert coefficient to complex number
        coeff_complex = complex(coeff)

        pauli_dict = {
            "label": pauli_string,
            "coeff": {
                "real": float(coeff_complex.real),
                "imag": float(coeff_complex.imag)
            }
        }
        paulis_list.append(pauli_dict)

    return {"paulis": paulis_list}


def _resolve_mapper(mapper: str):
    """Resolve the ``mapper`` string to its qiskit-nature mapper instance and
    canonical name.

    Args:
        mapper: "jw" (Jordan-Wigner), "parity", or "bk" (Bravyi-Kitaev),
            case-insensitive.

    Returns:
        Tuple ``(mapper_obj, mapper_name)``.

    Raises:
        ValueError: if ``mapper`` is not one of the three supported strings.
    """
    key = mapper.lower()
    if key == "jw":
        return JordanWignerMapper(), "jw"
    elif key == "parity":
        return ParityMapper(), "parity"
    elif key == "bk":
        return BravyiKitaevMapper(), "bk"
    else:
        raise ValueError(f"Unknown mapper: {mapper}")


def generate_spinless_hubbard_hamiltonian(
    lattice_size: Tuple[int, int],
    hopping: float,
    nearest_neighbor_interaction: float,
    chemical_potential: float,
    boundary_condition: str = "open",
    mapper: str = "jw"
) -> Tuple[Dict, Dict]:
    """
    Generate a spinless Fermi-Hubbard model Hamiltonian on a square lattice.

    The spinless Hubbard model:
    H = -t Σ_<i,j> (c†_i c_j + h.c.) + V Σ_<i,j> n_i n_j + μ Σ_i n_i

    where:
    - t: hopping parameter
    - V: nearest-neighbor interaction
    - μ: chemical potential

    Args:
        lattice_size: Size of square lattice (rows, cols)
        hopping: Hopping parameter t (typically negative)
        nearest_neighbor_interaction: Nearest-neighbor interaction V
        chemical_potential: Chemical potential μ
        boundary_condition: "open" or "periodic"
        mapper: "jw" (Jordan-Wigner), "parity", or "bk" (Bravyi-Kitaev)

    Returns:
        Tuple of (hamiltonian_dict, metadata_dict)
    """
    if isinstance(lattice_size, int):
        rows = cols = lattice_size
    else:
        rows, cols = lattice_size

    n_sites = rows * cols
    is_periodic = boundary_condition.lower() == "periodic"

    def site_index(r, c):
        """Convert 2D coordinates to 1D site index."""
        return r * cols + c

    def get_neighbors(r, c):
        """Get neighboring sites for a given position."""
        neighbors = []
        # Right neighbor
        if c + 1 < cols:
            neighbors.append((r, c + 1))
        elif is_periodic:
            neighbors.append((r, 0))
        # Down neighbor
        if r + 1 < rows:
            neighbors.append((r + 1, c))
        elif is_periodic:
            neighbors.append((0, c))
        return neighbors

    # Build the Hamiltonian terms
    ham_terms = {}

    # Hopping terms: -t (c†_i c_j + c†_j c_i)
    for r in range(rows):
        for c in range(cols):
            i = site_index(r, c)
            for nr, nc in get_neighbors(r, c):
                j = site_index(nr, nc)
                if i != j:  # Avoid self-loops in periodic case
                    # c†_i c_j term
                    term_ij = f"+_{i} -_{j}"
                    if term_ij in ham_terms:
                        ham_terms[term_ij] += hopping
                    else:
                        ham_terms[term_ij] = hopping

                    # c†_j c_i term (hermitian conjugate)
                    term_ji = f"+_{j} -_{i}"
                    if term_ji in ham_terms:
                        ham_terms[term_ji] += hopping
                    else:
                        ham_terms[term_ji] = hopping

    # Nearest-neighbor interaction: V n_i n_j
    for r in range(rows):
        for c in range(cols):
            i = site_index(r, c)
            for nr, nc in get_neighbors(r, c):
                j = site_index(nr, nc)
                if i != j:  # Avoid self-loops
                    # n_i n_j = c†_i c_i c†_j c_j
                    term = f"+_{i} -_{i} +_{j} -_{j}"
                    if term in ham_terms:
                        ham_terms[term] += nearest_neighbor_interaction
                    else:
                        ham_terms[term] = nearest_neighbor_interaction

    # Chemical potential: μ Σ_i n_i
    for i in range(n_sites):
        term = f"+_{i} -_{i}"
        if term in ham_terms:
            ham_terms[term] += chemical_potential
        else:
            ham_terms[term] = chemical_potential

    # Remove zero terms
    ham_terms = {k: v for k, v in ham_terms.items() if abs(v) > 1e-12}

    # Create FermionicOp
    fermionic_op = FermionicOp(ham_terms, num_spin_orbitals=n_sites)

    # Choose mapper
    mapper_obj, mapper_name = _resolve_mapper(mapper)

    # Map to Pauli operators
    pauli_op = mapper_obj.map(fermionic_op)

    # Simplify the operator
    pauli_op = pauli_op.simplify()

    # Convert to JSON format
    hamiltonian_dict = pauli_op_to_json_format(pauli_op)

    # Create metadata
    n_qubits = pauli_op.num_qubits
    n_terms = len(hamiltonian_dict["paulis"])

    metadata = {
        "model": "Spinless-Hubbard",
        "lattice_type": "square",
        "lattice_size": [rows, cols],
        "boundary_condition": boundary_condition,
        "parameters": {
            "hopping": hopping,
            "nearest_neighbor_interaction": nearest_neighbor_interaction,
            "chemical_potential": chemical_potential
        },
        "mapper": mapper_name,
        "n_qubits": n_qubits,
        "n_terms": n_terms,
        "lattice_name": f"Square{rows}x{cols}_{boundary_condition}"
    }

    return hamiltonian_dict, metadata


def generate_hubbard_hamiltonian(
    lattice_type: str,
    lattice_size: Tuple[int, ...],
    hopping: float,
    onsite_interaction: float,
    uniform_onsite_potential: float,
    boundary_condition: str = "open",
    mapper: str = "jw"
) -> Tuple[Dict, Dict]:
    """
    Generate a Fermi-Hubbard model Hamiltonian.

    Args:
        lattice_type: "line" or "square"
        lattice_size: Size of lattice (int for line, tuple for square)
        hopping: Hopping parameter t (uniform_interaction in Qiskit, typically negative)
        onsite_interaction: On-site interaction U (positive for repulsive)
        uniform_onsite_potential: Chemical potential μ (uniform_onsite_potential in Qiskit)
        boundary_condition: "open" or "periodic"
        mapper: "jw" (Jordan-Wigner), "parity", or "bk" (Bravyi-Kitaev)

    Returns:
        Tuple of (hamiltonian_dict, metadata_dict)
    """
    # Create lattice
    bc = BoundaryCondition.OPEN if boundary_condition.lower() == "open" else BoundaryCondition.PERIODIC

    if lattice_type.lower() == "line":
        if isinstance(lattice_size, tuple):
            lattice_size = lattice_size[0]
        lattice = LineLattice(num_nodes=lattice_size, boundary_condition=bc)
        lattice_name = f"Line{lattice_size}_{boundary_condition}"
    elif lattice_type.lower() == "square":
        if isinstance(lattice_size, int):
            rows = cols = lattice_size
        else:
            rows, cols = lattice_size
        lattice = SquareLattice(rows=rows, cols=cols, boundary_condition=bc)
        lattice_name = f"Square{rows}x{cols}_{boundary_condition}"
    else:
        raise ValueError(f"Unknown lattice type: {lattice_type}")

    # Create Fermi-Hubbard model
    hubbard = FermiHubbardModel(
        lattice.uniform_parameters(
            uniform_interaction=hopping,  # -t in the Hamiltonian
            uniform_onsite_potential=uniform_onsite_potential,  # μ in the Hamiltonian
        ),
        onsite_interaction=onsite_interaction,  # U in the Hamiltonian
    )

    # Get second quantized operator
    second_q_op = hubbard.second_q_op()

    # Choose mapper
    mapper_obj, mapper_name = _resolve_mapper(mapper)

    # Map to Pauli operators
    pauli_op = mapper_obj.map(second_q_op)

    # Convert to JSON format
    hamiltonian_dict = pauli_op_to_json_format(pauli_op)

    # Create metadata
    n_qubits = pauli_op.num_qubits
    n_terms = len(hamiltonian_dict["paulis"])

    metadata = {
        "model": "Fermi-Hubbard",
        "lattice_type": lattice_type,
        "lattice_size": lattice_size if isinstance(lattice_size, (list, tuple)) else [lattice_size],
        "boundary_condition": boundary_condition,
        "parameters": {
            "hopping": hopping,
            "onsite_interaction": onsite_interaction,
            "uniform_onsite_potential": uniform_onsite_potential
        },
        "mapper": mapper_name,
        "n_qubits": n_qubits,
        "n_terms": n_terms,
        "lattice_name": lattice_name
    }

    return hamiltonian_dict, metadata


def save_hamiltonian(
    hamiltonian_dict: Dict,
    metadata: Dict,
    output_dir: Path,
    save_metadata: bool = True
):
    """Save Hamiltonian and metadata to files."""
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save Hamiltonian in the same format as molecular Hamiltonians
    mapper_name = metadata["mapper"]
    hamiltonian_path = output_dir / f"{mapper_name}.txt"

    with open(hamiltonian_path, 'w') as f:
        json.dump(hamiltonian_dict, f)

    print(f"  Saved Hamiltonian: {hamiltonian_path}")

    # Save metadata
    if save_metadata:
        metadata_path = output_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"  Saved metadata: {metadata_path}")

    return hamiltonian_path


def generate_standard_hubbard_models(base_dir: Path):
    """Generate a set of standard Hubbard model Hamiltonians."""

    print("=" * 60)
    print("Generating Standard Fermi-Hubbard Model Hamiltonians")
    print("=" * 60)

    # Standard configurations
    configs = [
        # Line lattices - small systems
        {
            "name": "Hubbard_Line4_U2",
            "lattice_type": "line",
            "lattice_size": 4,
            "hopping": -1.0,
            "onsite_interaction": 2.0,
            "uniform_onsite_potential": 0.0,
            "boundary_condition": "open"
        },
        {
            "name": "Hubbard_Line4_U4",
            "lattice_type": "line",
            "lattice_size": 4,
            "hopping": -1.0,
            "onsite_interaction": 4.0,
            "uniform_onsite_potential": 0.0,
            "boundary_condition": "open"
        },
        {
            "name": "Hubbard_Line6_U2",
            "lattice_type": "line",
            "lattice_size": 6,
            "hopping": -1.0,
            "onsite_interaction": 2.0,
            "uniform_onsite_potential": 0.0,
            "boundary_condition": "open"
        },

        # Square lattices - 2x2
        {
            "name": "Hubbard_Square2x2_U2",
            "lattice_type": "square",
            "lattice_size": (2, 2),
            "hopping": -1.0,
            "onsite_interaction": 2.0,
            "uniform_onsite_potential": 0.0,
            "boundary_condition": "open"
        },
        {
            "name": "Hubbard_Square2x2_U4",
            "lattice_type": "square",
            "lattice_size": (2, 2),
            "hopping": -1.0,
            "onsite_interaction": 4.0,
            "uniform_onsite_potential": 0.0,
            "boundary_condition": "open"
        },
        {
            "name": "Hubbard_Square2x2_U4_periodic",
            "lattice_type": "square",
            "lattice_size": (2, 2),
            "hopping": -1.0,
            "onsite_interaction": 4.0,
            "uniform_onsite_potential": 0.0,
            "boundary_condition": "periodic"
        },
        {
            "name": "Hubbard_Square2x2_U4_mu-2",
            "lattice_type": "square",
            "lattice_size": (2, 2),
            "hopping": -1.0,
            "onsite_interaction": 4.0,
            "uniform_onsite_potential": -2.0,
            "boundary_condition": "open"
        },

        # Square lattices - 3x2
        {
            "name": "Hubbard_Square3x2_U2",
            "lattice_type": "square",
            "lattice_size": (3, 2),
            "hopping": -1.0,
            "onsite_interaction": 2.0,
            "uniform_onsite_potential": 0.0,
            "boundary_condition": "open"
        },

        # With non-zero onsite potential
        {
            "name": "Hubbard_Line4_U2_mu1",
            "lattice_type": "line",
            "lattice_size": 4,
            "hopping": -1.0,
            "onsite_interaction": 2.0,
            "uniform_onsite_potential": -1.0,
            "boundary_condition": "open"
        },
    ]

    for config in configs:
        print(f"\nGenerating: {config['name']}")
        print(f"  Lattice: {config['lattice_type']}, size: {config['lattice_size']}")
        print(f"  Parameters: t={config['hopping']}, U={config['onsite_interaction']}, μ={config['uniform_onsite_potential']}")
        print(f"  Boundary: {config['boundary_condition']}")

        try:
            # Generate Hamiltonian
            ham_dict, metadata = generate_hubbard_hamiltonian(
                lattice_type=config['lattice_type'],
                lattice_size=config['lattice_size'],
                hopping=config['hopping'],
                onsite_interaction=config['onsite_interaction'],
                uniform_onsite_potential=config['uniform_onsite_potential'],
                boundary_condition=config['boundary_condition'],
                mapper="jw"
            )

            # Determine number of qubits for directory name
            n_qubits = metadata['n_qubits']
            dir_name = f"{config['name']}_{n_qubits}qubits"
            output_dir = base_dir / dir_name

            # Save
            save_hamiltonian(ham_dict, metadata, output_dir)

            print(f"  ✓ Success! {metadata['n_qubits']} qubits, {metadata['n_terms']} terms")

        except Exception as e:
            print(f"  ✗ Error: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 60)
    print("Generation complete!")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Generate Fermi-Hubbard model Hamiltonians"
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Standard generation command
    standard_parser = subparsers.add_parser(
        "standard",
        help="Generate standard set of Hubbard models"
    )
    standard_parser.add_argument(
        "--output-dir",
        type=str,
        default="Hamiltonians",
        help="Base directory for Hamiltonians (default: Hamiltonians)"
    )

    # Custom generation command
    custom_parser = subparsers.add_parser(
        "custom",
        help="Generate a custom Hubbard model"
    )
    custom_parser.add_argument(
        "--name",
        type=str,
        required=True,
        help="Name for the Hamiltonian"
    )
    custom_parser.add_argument(
        "--lattice-type",
        type=str,
        required=True,
        choices=["line", "square"],
        help="Lattice type"
    )
    custom_parser.add_argument(
        "--lattice-size",
        type=int,
        nargs="+",
        required=True,
        help="Lattice size (single int for line, two ints for square)"
    )
    custom_parser.add_argument(
        "--hopping",
        type=float,
        default=-1.0,
        help="Hopping parameter t (default: -1.0)"
    )
    custom_parser.add_argument(
        "--onsite-interaction",
        type=float,
        default=4.0,
        help="On-site interaction U (default: 4.0)"
    )
    custom_parser.add_argument(
        "--uniform-onsite-potential",
        type=float,
        default=0.0,
        help="Uniform onsite potential μ (default: 0.0)"
    )
    custom_parser.add_argument(
        "--boundary",
        type=str,
        default="open",
        choices=["open", "periodic"],
        help="Boundary condition (default: open)"
    )
    custom_parser.add_argument(
        "--mapper",
        type=str,
        default="jw",
        choices=["jw", "parity", "bk"],
        help="Fermion-to-qubit mapping (default: jw)"
    )
    custom_parser.add_argument(
        "--output-dir",
        type=str,
        default="Hamiltonians",
        help="Base directory for Hamiltonians (default: Hamiltonians)"
    )

    # Spinless Hubbard model command
    spinless_parser = subparsers.add_parser(
        "spinless",
        help="Generate a spinless Hubbard model (1 qubit per site)"
    )
    spinless_parser.add_argument(
        "--name",
        type=str,
        required=True,
        help="Name for the Hamiltonian"
    )
    spinless_parser.add_argument(
        "--lattice-size",
        type=int,
        nargs="+",
        required=True,
        help="Lattice size (rows cols for square lattice)"
    )
    spinless_parser.add_argument(
        "--hopping",
        type=float,
        default=-1.0,
        help="Hopping parameter t (default: -1.0)"
    )
    spinless_parser.add_argument(
        "--nn-interaction",
        type=float,
        default=1.0,
        help="Nearest-neighbor interaction V (default: 1.0)"
    )
    spinless_parser.add_argument(
        "--chemical-potential",
        type=float,
        default=0.0,
        help="Chemical potential μ (default: 0.0)"
    )
    spinless_parser.add_argument(
        "--boundary",
        type=str,
        default="open",
        choices=["open", "periodic"],
        help="Boundary condition (default: open)"
    )
    spinless_parser.add_argument(
        "--mapper",
        type=str,
        default="jw",
        choices=["jw", "parity", "bk"],
        help="Fermion-to-qubit mapping (default: jw)"
    )
    spinless_parser.add_argument(
        "--output-dir",
        type=str,
        default="Hamiltonians",
        help="Base directory for Hamiltonians (default: Hamiltonians)"
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    project_root = get_project_root()
    base_dir = project_root / args.output_dir

    if args.command == "standard":
        generate_standard_hubbard_models(base_dir)

    elif args.command == "custom":
        print("=" * 60)
        print(f"Generating Custom Hubbard Model: {args.name}")
        print("=" * 60)

        lattice_size = args.lattice_size[0] if len(args.lattice_size) == 1 else tuple(args.lattice_size)

        print(f"Lattice: {args.lattice_type}, size: {lattice_size}")
        print(f"Parameters: t={args.hopping}, U={args.onsite_interaction}, μ={args.uniform_onsite_potential}")
        print(f"Boundary: {args.boundary}")
        print(f"Mapper: {args.mapper}")

        try:
            ham_dict, metadata = generate_hubbard_hamiltonian(
                lattice_type=args.lattice_type,
                lattice_size=lattice_size,
                hopping=args.hopping,
                onsite_interaction=args.onsite_interaction,
                uniform_onsite_potential=args.uniform_onsite_potential,
                boundary_condition=args.boundary,
                mapper=args.mapper
            )

            n_qubits = metadata['n_qubits']
            dir_name = f"{args.name}_{n_qubits}qubits"
            output_dir = base_dir / dir_name

            save_hamiltonian(ham_dict, metadata, output_dir)

            print(f"\n✓ Success! {metadata['n_qubits']} qubits, {metadata['n_terms']} terms")
            print(f"Output directory: {output_dir}")

        except Exception as e:
            print(f"\n✗ Error: {e}")
            import traceback
            traceback.print_exc()
            return 1

    elif args.command == "spinless":
        print("=" * 60)
        print(f"Generating Spinless Hubbard Model: {args.name}")
        print("=" * 60)

        lattice_size = tuple(args.lattice_size) if len(args.lattice_size) == 2 else (args.lattice_size[0], args.lattice_size[0])

        print(f"Lattice: square, size: {lattice_size[0]}x{lattice_size[1]}")
        print(f"Parameters: t={args.hopping}, V={args.nn_interaction}, μ={args.chemical_potential}")
        print(f"Boundary: {args.boundary}")
        print(f"Mapper: {args.mapper}")

        try:
            ham_dict, metadata = generate_spinless_hubbard_hamiltonian(
                lattice_size=lattice_size,
                hopping=args.hopping,
                nearest_neighbor_interaction=args.nn_interaction,
                chemical_potential=args.chemical_potential,
                boundary_condition=args.boundary,
                mapper=args.mapper
            )

            n_qubits = metadata['n_qubits']
            dir_name = f"{args.name}_{n_qubits}qubits"
            output_dir = base_dir / dir_name

            save_hamiltonian(ham_dict, metadata, output_dir)

            print(f"\n✓ Success! {metadata['n_qubits']} qubits, {metadata['n_terms']} terms")
            print(f"Output directory: {output_dir}")

        except Exception as e:
            print(f"\n✗ Error: {e}")
            import traceback
            traceback.print_exc()
            return 1

    return 0


if __name__ == "__main__":
    exit(main())
