"""
Generate H2O Hamiltonians with different bond angles and bond lengths.
Saves to Hamiltonians/04Water/xxx/ where xxx is a descriptive name.
"""

import sys
import os
from pathlib import Path
import numpy as np

# Add code directory to path
sys.path.append(str(Path(__file__).parent))

from build_molecule_hamiltonian import generate_jw_format_file


def h2o_positions_from_geometry(angle_hoh_deg, distance_oh):
    """
    Generate H2O atomic positions from bond angle and bond length.

    Parameters:
    -----------
    angle_hoh_deg: float
        H-O-H bond angle in degrees
    distance_oh: float
        O-H bond length in Angstroms

    Returns:
    --------
    atom_labels: list
        List of atom labels ['O', 'H', 'H']
    positions: np.ndarray
        Array of atomic positions in Angstroms
    """
    atom_labels = ["O", "H", "H"]
    
    # Convert angle to radians
    angle_hoh_rad = np.deg2rad(angle_hoh_deg)
    
    # Calculate positions: O at origin, H atoms symmetric about y-axis
    # The angle is H-O-H, so each H makes angle_hoh/2 with the negative y-axis
    theta = (np.pi - angle_hoh_rad) / 2
    distance_oh_x = distance_oh * np.cos(theta)
    distance_oh_y = distance_oh * np.sin(theta)
    
    positions = np.array([
        [0, 0, 0],  # O at origin
        [-distance_oh_x, -distance_oh_y, 0],  # H1
        [distance_oh_x, -distance_oh_y, 0]    # H2
    ])
    
    return atom_labels, positions


def generate_h2o_hamiltonian(angle_hoh_deg, distance_oh, basis="sto-3g", 
                             output_dir=None, verbose=True):
    """
    Generate H2O Hamiltonian with specified geometry.

    Parameters:
    -----------
    angle_hoh_deg: float
        H-O-H bond angle in degrees
    distance_oh: float
        O-H bond length in Angstroms
    basis: str
        Basis set (default: 'sto-3g')
    output_dir: str or Path
        Output directory path (default: None, auto-generated)
    verbose: bool
        Print progress information

    Returns:
    --------
    output_str: str
        Formatted Hamiltonian string
    e_fci: float
        FCI ground state energy
    output_path: Path
        Path to the generated file
    """
    # Generate positions
    atom_labels, positions = h2o_positions_from_geometry(angle_hoh_deg, distance_oh)
    
    # Create output directory name
    if output_dir is None:
        # Format: angleXXX_bondYYY (e.g., angle104.5_bond0.958)
        angle_str = f"angle{angle_hoh_deg:.1f}".replace('.', 'p')
        bond_str = f"bond{distance_oh:.3f}".replace('.', 'p')
        config_name = f"{angle_str}_{bond_str}"
        output_dir = Path("Hamiltonians/04Water") / config_name
    else:
        output_dir = Path(output_dir)
    
    # Create directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Output file
    output_file = output_dir / "jw_generated.txt"
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Generating H2O Hamiltonian:")
        print(f"  Angle (H-O-H): {angle_hoh_deg:.2f}°")
        print(f"  Bond length (O-H): {distance_oh:.4f} Å")
        print(f"  Basis: {basis}")
        print(f"  Output: {output_file}")
        print(f"  Atom coordinates:")
        for label, pos in zip(atom_labels, positions):
            print(f"    {label} {pos[0]:8.4f} {pos[1]:8.4f} {pos[2]:8.4f}")
        print(f"{'='*60}")
    
    # Generate Hamiltonian
    output_str, e_fci = generate_jw_format_file(
        atom_labels=atom_labels,
        positions=positions,
        basis=basis,
        charge=0,
        spin=0,
        output_file=str(output_file)
    )
    
    # Save metadata
    metadata_file = output_dir / "geometry.txt"
    with open(metadata_file, 'w') as f:
        f.write(f"H2O Geometry Parameters\n")
        f.write(f"{'='*60}\n")
        f.write(f"Bond angle (H-O-H): {angle_hoh_deg:.4f} degrees\n")
        f.write(f"Bond length (O-H): {distance_oh:.6f} Angstroms\n")
        f.write(f"Basis set: {basis}\n")
        f.write(f"\nAtomic coordinates:\n")
        for label, pos in zip(atom_labels, positions):
            f.write(f"  {label:2s} {pos[0]:12.6f} {pos[1]:12.6f} {pos[2]:12.6f}\n")
        f.write(f"\nFCI Ground State Energy: {e_fci:.10f} Ha\n")
    
    if verbose:
        print(f"  FCI energy: {e_fci:.10f} Ha")
        print(f"  Geometry saved to: {metadata_file}")
    
    return output_str, e_fci, output_file


def generate_h2o_grid(angle_range=None, bond_range=None, basis="sto-3g", verbose=True):
    """
    Generate a grid of H2O Hamiltonians with different angles and bond lengths.

    Parameters:
    -----------
    angle_range: list or tuple
        [min_angle, max_angle, n_points] or list of specific angles
        Default: [90, 120, 7] (7 points from 90° to 120°)
    bond_range: list or tuple
        [min_bond, max_bond, n_points] or list of specific bond lengths
        Default: [0.85, 1.15, 7] (7 points from 0.85 to 1.15 Å)
    basis: str
        Basis set (default: 'sto-3g')
    verbose: bool
        Print progress information

    Returns:
    --------
    results: list
        List of dictionaries with results for each configuration
    """
    # Default ranges
    if angle_range is None:
        angle_range = [90, 120, 7]
    if bond_range is None:
        bond_range = [0.85, 1.15, 7]
    
    # Parse angle range
    if len(angle_range) == 3:
        angles = np.linspace(angle_range[0], angle_range[1], int(angle_range[2]))
    else:
        angles = np.array(angle_range)
    
    # Parse bond range
    if len(bond_range) == 3:
        bonds = np.linspace(bond_range[0], bond_range[1], int(bond_range[2]))
    else:
        bonds = np.array(bond_range)
    
    if verbose:
        print(f"\nGenerating H2O Hamiltonian grid:")
        print(f"  Angles: {len(angles)} points from {angles.min():.1f}° to {angles.max():.1f}°")
        print(f"  Bonds: {len(bonds)} points from {bonds.min():.3f} Å to {bonds.max():.3f} Å")
        print(f"  Total configurations: {len(angles) * len(bonds)}")
        print(f"  Basis: {basis}")
    
    results = []
    total = len(angles) * len(bonds)
    count = 0
    
    for angle in angles:
        for bond in bonds:
            count += 1
            if verbose:
                print(f"\n[{count}/{total}] Processing angle={angle:.2f}°, bond={bond:.4f} Å")
            
            try:
                output_str, e_fci, output_path = generate_h2o_hamiltonian(
                    angle_hoh_deg=angle,
                    distance_oh=bond,
                    basis=basis,
                    verbose=verbose
                )
                
                results.append({
                    'angle': float(angle),
                    'bond': float(bond),
                    'energy': float(e_fci),
                    'output_path': str(output_path),
                    'success': True
                })
            except Exception as e:
                print(f"  ERROR: {e}")
                results.append({
                    'angle': float(angle),
                    'bond': float(bond),
                    'energy': None,
                    'output_path': None,
                    'success': False,
                    'error': str(e)
                })
    
    # Summary
    if verbose:
        print(f"\n{'='*60}")
        print(f"Grid generation complete!")
        print(f"  Successful: {sum(1 for r in results if r['success'])}/{len(results)}")
        print(f"  Failed: {sum(1 for r in results if not r['success'])}/{len(results)}")
        
        if any(r['success'] for r in results):
            energies = [r['energy'] for r in results if r['success']]
            print(f"  Energy range: {min(energies):.6f} to {max(energies):.6f} Ha")
    
    return results


if __name__ == "__main__":
    import argparse
    
    # Ground state geometry
    GS_ANGLE = 104.5  # degrees
    GS_BOND = 0.9584  # Angstroms
    
    parser = argparse.ArgumentParser(description="Generate H2O Hamiltonians with different geometries")
    parser.add_argument("--angle-min", type=float, default=None, help="Minimum H-O-H angle (degrees)")
    parser.add_argument("--angle-max", type=float, default=None, help="Maximum H-O-H angle (degrees)")
    parser.add_argument("--angle-n", type=int, default=7, help="Number of angle points")
    parser.add_argument("--bond-min", type=float, default=None, help="Minimum O-H bond length (Å)")
    parser.add_argument("--bond-max", type=float, default=None, help="Maximum O-H bond length (Å)")
    parser.add_argument("--bond-n", type=int, default=7, help="Number of bond length points")
    parser.add_argument("--basis", type=str, default="sto-3g", help="Basis set")
    parser.add_argument("--single", action="store_true", help="Generate single configuration (use angle-min and bond-min)")
    parser.add_argument("--center", action="store_true", help="Center grid at GS geometry (104.5°, 0.9584 Å)")
    parser.add_argument("--angle-range", type=float, default=4.5, help="Angle range around center (±degrees, default: ±4.5°)")
    parser.add_argument("--bond-range", type=float, default=0.1, help="Bond range around center (±Å, default: ±0.1 Å)")
    
    args = parser.parse_args()
    
    if args.single:
        # Generate single configuration
        angle = args.angle_min if args.angle_min is not None else GS_ANGLE
        bond = args.bond_min if args.bond_min is not None else GS_BOND
        print(f"Generating single H2O Hamiltonian:")
        print(f"  Angle: {angle:.2f}°")
        print(f"  Bond: {bond:.4f} Å")
        generate_h2o_hamiltonian(
            angle_hoh_deg=angle,
            distance_oh=bond,
            basis=args.basis,
            verbose=True
        )
    else:
        # Determine ranges
        if args.center:
            # Center at GS geometry
            angle_min = GS_ANGLE - args.angle_range
            angle_max = GS_ANGLE + args.angle_range
            bond_min = GS_BOND - args.bond_range
            bond_max = GS_BOND + args.bond_range
            print(f"Centering grid at GS geometry:")
            print(f"  Center angle: {GS_ANGLE}° (range: {angle_min:.2f}° to {angle_max:.2f}°)")
            print(f"  Center bond: {GS_BOND:.4f} Å (range: {bond_min:.4f} to {bond_max:.4f} Å)")
        else:
            # Use provided or default values
            angle_min = args.angle_min if args.angle_min is not None else 90.0
            angle_max = args.angle_max if args.angle_max is not None else 120.0
            bond_min = args.bond_min if args.bond_min is not None else 0.85
            bond_max = args.bond_max if args.bond_max is not None else 1.15
        
        # Generate grid
        results = generate_h2o_grid(
            angle_range=[angle_min, angle_max, args.angle_n],
            bond_range=[bond_min, bond_max, args.bond_n],
            basis=args.basis,
            verbose=True
        )
        
        # Save summary
        summary_file = Path("Hamiltonians/04Water/grid_summary.txt")
        summary_file.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_file, 'w') as f:
            f.write("H2O Grid Generation Summary\n")
            f.write("="*60 + "\n\n")
            if args.center:
                f.write(f"Centered at GS geometry: {GS_ANGLE}°, {GS_BOND:.4f} Å\n")
                f.write(f"Angle range: {angle_min:.1f}° to {angle_max:.1f}° ({args.angle_n} points)\n")
                f.write(f"Bond range: {bond_min:.4f} to {bond_max:.4f} Å ({args.bond_n} points)\n")
            else:
                f.write(f"Angle range: {angle_min:.1f}° to {angle_max:.1f}° ({args.angle_n} points)\n")
                f.write(f"Bond range: {bond_min:.4f} to {bond_max:.4f} Å ({args.bond_n} points)\n")
            f.write(f"Basis: {args.basis}\n")
            f.write(f"Total configurations: {len(results)}\n\n")
            f.write(f"{'Angle (°)':>10} {'Bond (Å)':>10} {'Energy (Ha)':>15} {'Status':>10}\n")
            f.write("-"*60 + "\n")
            for r in results:
                if r['success']:
                    f.write(f"{r['angle']:10.2f} {r['bond']:10.4f} {r['energy']:15.10f} {'OK':>10}\n")
                else:
                    f.write(f"{r['angle']:10.2f} {r['bond']:10.4f} {'N/A':>15} {'FAILED':>10}\n")
        
        print(f"\nSummary saved to: {summary_file}")

