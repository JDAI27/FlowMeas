# %%


import numpy as np


def h2_labels_positions():
    distance_hh = 0.735
    atom_labels = ["H", "H"]
    positions = np.array([[0, 0, -distance_hh / 2], [0, 0, distance_hh / 2]])

    return atom_labels, positions


def lih_labels_positions():

    distance_lih = 1.6

    atom_labels = ["Li", "H"]
    positions = np.array([[0, 0, 0], [0, 0, distance_lih]])

    return atom_labels, positions


def h2o_labels_positions():

    angle_hoh = 104.5
    distance_oh = 0.9584

    atom_labels = ["O", "H", "H"]

    theta = (np.pi - angle_hoh) / 2
    distance_oh_x = distance_oh * np.cos(theta)
    distance_oh_y = distance_oh * np.sin(theta)

    positions = np.array([[0, 0, 0], [-distance_oh_x, -distance_oh_y, 0], [distance_oh_x, -distance_oh_y, 0]])

    return atom_labels, positions


def nh3_labels_positions():

    atom_labels = ["N", "H", "H", "H"]

    distance_nh, angle_hnh, phi = 1.017, 107.8 / 180 * np.pi, 0

    factor = (1 + 2 * np.sin(angle_hnh)) / 3
    height = distance_nh * np.sqrt(factor)
    rho = distance_nh * np.sqrt(1 - factor)
    rho_x_0 = rho * np.cos(phi)
    rho_y_0 = rho * np.sin(phi)
    rho_x_1 = rho * np.cos(2 * np.pi / 3 + phi)
    rho_y_1 = rho * np.sin(2 * np.pi / 3 + phi)
    rho_x_2 = rho * np.cos(-2 * np.pi / 3 + phi)
    rho_y_2 = rho * np.sin(-2 * np.pi / 3 + phi)
    positions = np.array([[0, 0, height], [rho_x_0, rho_y_0, 0], [rho_x_1, rho_y_1, 0], [rho_x_2, rho_y_2, 0]])

    return atom_labels, positions


def c2h2_labels_positions():

    distance_cc = 1.203
    distance_ch = 1.06

    atom_labels = ["C", "C", "H", "H"]
    half_distance_cc = distance_cc / 2
    positions = np.array(
        [
            [-half_distance_cc, 0, 0],
            [half_distance_cc, 0, 0],
            [-(half_distance_cc + distance_ch), 0, 0],
            [half_distance_cc + distance_ch, 0, 0],
        ]
    )

    return atom_labels, positions


def c2h4_labels_positions():

    distance_ch = 1.07
    distance_cc = 1.33
    angle_hcc = 121.8

    atom_labels = ["C", "C", "H", "H", "H", "H"]

    half_distance_cc = distance_cc / 2

    theta = np.pi - angle_hcc
    distance_ch_y = distance_ch * np.sin(theta)
    distance_ch_x = distance_ch * np.cos(theta)

    positions = np.array(
        [
            [-half_distance_cc, 0, 0],
            [half_distance_cc, 0, 0],
            [-(half_distance_cc + distance_ch_x), distance_ch_y, 0],
            [-(half_distance_cc + distance_ch_x), -distance_ch_y, 0],
            [half_distance_cc + distance_ch_x, distance_ch_y, 0],
            [half_distance_cc + distance_ch_x, -distance_ch_y, 0],
        ]
    )

    return atom_labels, positions


def c3h8_labels_positions():

    atom_labels = ["C", "C", "C", "H", "H", "H", "H", "H", "H", "H", "H"]

    positions = np.array(
        [
            [0.0000, -0.5689, 0.0000],
            [-1.2571, 0.2844, 0.0000],
            [1.2571, 0.2845, 0.0000],
            [0.0000, -1.2183, 0.8824],
            [0.0000, -1.2183, -0.8824],
            [-1.2969, 0.9244, 0.8873],
            [-1.2967, 0.9245, -0.8872],
            [-2.1475, -0.3520, -0.0001],
            [2.1475, -0.3520, 0.0000],
            [1.2968, 0.9245, 0.8872],
            [1.2968, 0.9245, -0.8872],
        ]
    )

    return atom_labels, positions


def n2_labels_positions():

    distance_nn = 1.09
    atom_labels = ["N", "N"]
    positions = np.array([[0, 0, -distance_nn / 2], [0, 0, distance_nn / 2]])

    return atom_labels, positions


def build_geometry(atom_labels, positions):
    geometry = [(label, pos) for label, pos in zip(atom_labels, positions)]

    return geometry


def build_mol_info(atom_labels, positions):
    atom_str = ""
    for label, pos in zip(atom_labels, positions):
        atom_str += f"{label} {pos[0]}  {pos[1]}  {pos[2]};"

    mol_info = {
        "atom": atom_str,
        "basis": "sto3g",
        "charge": 0,
        "spin": 0,
        # "unit": DistanceUnit.ANGSTROM,
    }

    return mol_info


def build_geostring(atom_labels, positions):

    geostring = ""
    for label, pos in zip(atom_labels, positions):
        geostring += f"{label} {pos[0]}  {pos[1]}  {pos[2]}\n"

    return geostring


# %%
# Old PauliArray code - commented out (now using OpenFermion directly)
# mol_labels_position = h2o_labels_positions
# mol_info = build_mol_info(*mol_labels_position())
# driver = PySCFDriver(**mol_info)
# problem = driver.run()
#..


# %%
def generate_jw_format_file(atom_labels, positions, basis="sto-3g", charge=0, spin=0, output_file=None):
    """
    Generate Hamiltonian in the same format as jw.txt files using OpenFermion + PySCF

    Parameters:
    -----------
    atom_labels: list
        List of atom labels (e.g., ['H', 'O', 'H'])
    positions: np.ndarray
        Array of atomic positions in Angstroms
    basis: str
        Basis set (default: 'sto-3g')
    charge: int
        Molecular charge (default: 0)
    spin: int
        Spin multiplicity (default: 0, singlet)
    output_file: str
        Output file path (default: None, returns string)

    Returns:
    --------
    output_str: str
        Formatted Hamiltonian string in jw.txt format
    """
    from pyscf import gto, scf, ao2mo
    from openfermion.transforms import jordan_wigner, get_fermion_operator
    from openfermion.ops import InteractionOperator
    from openfermion.chem.molecular_data import spinorb_from_spatial

    # Build geometry string for PySCF
    geometry = []
    for label, pos in zip(atom_labels, positions):
        geometry.append((label, tuple(pos)))

    # Create PySCF molecule
    mol = gto.Mole()
    mol.atom = geometry
    mol.basis = basis
    mol.charge = charge
    mol.spin = spin
    mol.build()

    # Run Hartree-Fock
    mf = scf.RHF(mol)
    mf.kernel()

    # Run FCI to get exact ground state energy
    from pyscf import fci
    cisolver = fci.FCI(mol, mf.mo_coeff)

    # For molecules with strong correlation (like C2), compute multiple roots
    # to ensure we get the true ground state
    n_electrons = mol.nelectron
    if n_electrons >= 12:  # For larger molecules, use nroots
        fci_result = cisolver.kernel(nroots=5)
        if isinstance(fci_result[0], (list, np.ndarray)):
            e_fci = fci_result[0][0]  # Take lowest energy
            ci_vec = fci_result[1][0]
        else:
            e_fci = fci_result[0]
            ci_vec = fci_result[1]
    else:
        e_fci, ci_vec = cisolver.kernel()

    print(f"Nuclear repulsion energy: {mol.energy_nuc():.10f}")
    print(f"Hartree-Fock energy: {mf.e_tot:.10f}")
    print(f"FCI ground state energy: {e_fci:.10f}")
    print(f"Correlation energy: {e_fci - mf.e_tot:.10f}")

    # Number of spatial orbitals and spin-orbitals
    n_orbitals = mol.nao
    n_qubits = 2 * n_orbitals

    # Transform one- and two-electron integrals to an orthonormal MO basis
    # One-electron (core) integrals in AO -> MO
    h1_ao = mf.get_hcore()  # T + V_ne in AO basis
    mo = mf.mo_coeff        # AO->MO transformation matrix
    h1_mo = mo.T @ h1_ao @ mo

    # Two-electron integrals in AO -> MO (chemist notation: (pq|rs))
    # Use compact=False to get a full 4-index tensor directly in MO basis
    eri_ao = mol.intor('int2e')
    eri_mo = ao2mo.incore.full(eri_ao, mo, compact=False)
    two_body_chem_mo = eri_mo  # (p,q,r,s) chemist notation (pq|rs)
    # Convert to physicist notation <pr|qs> expected by InteractionOperator
    two_body_phys_mo = two_body_chem_mo.transpose(0, 3, 2, 1)

    # Nuclear repulsion energy (constant)
    nuclear_repulsion = mol.energy_nuc()

    # Expand spatial integrals to spin-orbital tensors
    # Use the EXACT same convention as OpenFermion's MolecularData.get_molecular_hamiltonian()
    # which applies 1/2 factor to the two-body tensor
    one_body_so, two_body_so = spinorb_from_spatial(h1_mo, two_body_phys_mo)
    molecular_hamiltonian = InteractionOperator(
        constant=float(nuclear_repulsion),
        one_body_tensor=one_body_so,
        two_body_tensor=(1.0 / 2.0) * two_body_so,  # Match OpenFermion convention
    )

    # Convert to fermion operator (spin-orbital)
    fermion_hamiltonian = get_fermion_operator(molecular_hamiltonian)

    # Apply Jordan-Wigner transformation
    qubit_hamiltonian = jordan_wigner(fermion_hamiltonian)

    # Canonicalize: combine duplicate strings, drop tiny terms, sort consistently
    def canonicalize_qubit_operator(op, n_qubits, tol=1e-12):
        accum = {}
        for pauli_tuple, coeff in op.terms.items():
            c = complex(coeff)
            if abs(c.real) < tol and abs(c.imag) < tol:
                continue
            s = ['I'] * n_qubits
            for q, p in pauli_tuple:
                s[q] = p
            key = ''.join(s)
            accum[key] = accum.get(key, 0) + c
        # Remove near-zero after accumulation
        items = [(k, v) for k, v in accum.items() if abs(v.real) > tol or abs(v.imag) > tol]
        # Sort: identity first, then lexicographically by string
        items.sort(key=lambda kv: (kv[0] != 'I' * n_qubits, kv[0]))
        return items

    canon_terms = canonicalize_qubit_operator(qubit_hamiltonian, n_qubits)

    # Format output
    output_lines = []
    for pauli_string, coeff in canon_terms:
        coeff_str = f"({coeff.real}{coeff.imag:+}j)"
        output_lines.append(pauli_string)
        output_lines.append(coeff_str)

    # Join all lines
    output_str = '\n'.join(output_lines)

    # Write to file if specified
    if output_file:
        with open(output_file, 'w') as f:
            f.write(output_str)
        print(f"\nHamiltonian written to {output_file}")
        print(f"Number of terms: {len(canon_terms)}")
        print(f"Number of qubits: {n_qubits}")
        print(f"\nThe FCI energy from PySCF ({e_fci:.10f}) should match")
        print(f"the ground state energy obtained by diagonalizing this Hamiltonian.")

    return output_str, e_fci


# %%
def generate_h2o_sto3g_14qubits():
    """
    Generate H2O Hamiltonian with exact coordinates from the problem statement:
    H.0 0.769 -0.546
    O.0.0 0.137
    H.0 -0.769 -0.546
    """
    atom_labels = ["H", "O", "H"]
    positions = np.array([
        [0.0, 0.769, -0.546],
        [0.0, 0.0, 0.137],
        [0.0, -0.769, -0.546]
    ])

    output_file = "Hamiltonians/H2O_STO3g_14qubits/jw_generated.txt"

    print("Building H2O Hamiltonian...")
    print("Atom coordinates:")
    for label, pos in zip(atom_labels, positions):
        print(f"  {label} {pos[0]:.3f} {pos[1]:.3f} {pos[2]:.3f}")
    print("=" * 60)

    output_str, e_fci = generate_jw_format_file(
        atom_labels=atom_labels,
        positions=positions,
        basis="sto-3g",
        charge=0,
        spin=0,
        output_file=output_file
    )

    print(f"\nExpected from ExactEnergy.txt:")
    print(f"  Total ground state energy: -75.02329149983701")
    print(f"  FCI from PySCF: {e_fci:.10f}")
    print(f"  Difference: {abs(e_fci - (-75.02329149983701)):.2e}")

    return output_str, e_fci


# %%
def generate_nh3_sto3g_16qubits():
    """
    Generate NH3 Hamiltonian with exact coordinates from the problem statement:
    N.0.0 0.149
    H.0 0.947 -0.349
    H 0.820 -0.474 -0.349
    H -0.820 -0.474 -0.349
    """
    atom_labels = ["N", "H", "H", "H"]
    positions = np.array([
        [0.0, 0.0, 0.149],
        [0.0, 0.947, -0.349],
        [0.820, -0.474, -0.349],
        [-0.820, -0.474, -0.349]
    ])

    output_file = "Hamiltonians/NH3_STO3g_16qubits/jw_generated.txt"

    print("Building NH3 Hamiltonian...")
    print("Atom coordinates:")
    for label, pos in zip(atom_labels, positions):
        print(f"  {label} {pos[0]:.3f} {pos[1]:.3f} {pos[2]:.3f}")
    print("=" * 60)

    output_str, e_fci = generate_jw_format_file(
        atom_labels=atom_labels,
        positions=positions,
        basis="sto-3g",
        charge=0,
        spin=0,
        output_file=output_file
    )

    print(f"\nExpected from ExactEnergy.txt:")
    print(f"  Total ground state energy: -55.528228723477774")
    print(f"  FCI from PySCF: {e_fci:.10f}")
    print(f"  Difference: {abs(e_fci - (-55.528228723477774)):.2e}")

    return output_str, e_fci


# %%
def generate_h2_631g_8qubits():
    """
    Generate H2 Hamiltonian with 6-31G basis:
    H.0.0.0
    H 0 0.7462
    """
    atom_labels = ["H", "H"]
    positions = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.7462]
    ])

    output_file = "Hamiltonians/H2_6-31G_8qubits/jw_generated.txt"

    print("Building H2 Hamiltonian with 6-31G basis...")
    print("Atom coordinates:")
    for label, pos in zip(atom_labels, positions):
        print(f"  {label} {pos[0]:.3f} {pos[1]:.3f} {pos[2]:.3f}")
    print("=" * 60)

    output_str, e_fci = generate_jw_format_file(
        atom_labels=atom_labels,
        positions=positions,
        basis="6-31g",
        charge=0,
        spin=0,
        output_file=output_file
    )

    print(f"\nExpected from ExactEnergy.txt:")
    print(f"  Total ground state energy: -1.1516978499190276")
    print(f"  FCI from PySCF: {e_fci:.10f}")
    print(f"  Difference: {abs(e_fci - (-1.1516978499190276)):.2e}")

    return output_str, e_fci


# %%
def generate_h2_sto3g_4qubits():
    """
    Generate H2 Hamiltonian with STO-3G basis:
    H.0.0.0
    H.0.0.735
    """
    atom_labels = ["H", "H"]
    positions = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.735]
    ])

    output_file = "Hamiltonians/H2_STO3g_4qubits/jw_generated.txt"

    print("Building H2 Hamiltonian with STO-3G basis...")
    print("Atom coordinates:")
    for label, pos in zip(atom_labels, positions):
        print(f"  {label} {pos[0]:.3f} {pos[1]:.3f} {pos[2]:.3f}")
    print("=" * 60)

    output_str, e_fci = generate_jw_format_file(
        atom_labels=atom_labels,
        positions=positions,
        basis="sto-3g",
        charge=0,
        spin=0,
        output_file=output_file
    )

    print(f"\nExpected from ExactEnergy.txt:")
    print(f"  Total ground state energy: -1.1373060357533995")
    print(f"  FCI from PySCF: {e_fci:.10f}")
    print(f"  Difference: {abs(e_fci - (-1.1373060357533995)):.2e}")

    return output_str, e_fci


# %%
def generate_lih_sto3g_12qubits():
    """
    Generate LiH Hamiltonian with STO-3G basis:
    H.0.0.0
    Li.0.0 1.548
    """
    atom_labels = ["H", "Li"]
    positions = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.548]
    ])

    output_file = "Hamiltonians/LiH_STO3g_12qubits/jw_generated.txt"

    print("Building LiH Hamiltonian with STO-3G basis...")
    print("Atom coordinates:")
    for label, pos in zip(atom_labels, positions):
        print(f"  {label} {pos[0]:.3f} {pos[1]:.3f} {pos[2]:.3f}")
    print("=" * 60)

    output_str, e_fci = generate_jw_format_file(
        atom_labels=atom_labels,
        positions=positions,
        basis="sto-3g",
        charge=0,
        spin=0,
        output_file=output_file
    )

    return output_str, e_fci


# %%
def generate_beh2_sto3g_14qubits():
    """
    Generate BeH2 Hamiltonian with STO-3G basis:
    H 1.3038.0.0
    Be.0.0.0
    H -1.3038.0.0
    """
    atom_labels = ["H", "Be", "H"]
    positions = np.array([
        [1.3038, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [-1.3038, 0.0, 0.0]
    ])

    output_file = "Hamiltonians/BeH2_STO3g_14qubits/jw_generated.txt"

    print("Building BeH2 Hamiltonian with STO-3G basis...")
    print("Atom coordinates:")
    for label, pos in zip(atom_labels, positions):
        print(f"  {label} {pos[0]:.3f} {pos[1]:.3f} {pos[2]:.3f}")
    print("=" * 60)

    output_str, e_fci = generate_jw_format_file(
        atom_labels=atom_labels,
        positions=positions,
        basis="sto-3g",
        charge=0,
        spin=0,
        output_file=output_file
    )

    return output_str, e_fci


# %%
def generate_c2_sto3g_20qubits():
    """
    Generate C2 Hamiltonian with STO-3G basis:
    C.0.0.0
    C 0 0 1.2691
    """
    atom_labels = ["C", "C"]
    positions = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.2691]
    ])

    output_file = "Hamiltonians/C2_STO3g_20qubits/jw_generated.txt"

    print("Building C2 Hamiltonian with STO-3G basis...")
    print("Atom coordinates:")
    for label, pos in zip(atom_labels, positions):
        print(f"  {label} {pos[0]:.3f} {pos[1]:.3f} {pos[2]:.3f}")
    print("=" * 60)

    output_str, e_fci = generate_jw_format_file(
        atom_labels=atom_labels,
        positions=positions,
        basis="sto-3g",
        charge=0,
        spin=0,
        output_file=output_file
    )

    return output_str, e_fci


# %%
def generate_hcl_sto3g_20qubits():
    """
    Generate HCl Hamiltonian with STO-3G basis:
    Cl.0.0.075
    H.0.0 -1.268
    """
    atom_labels = ["Cl", "H"]
    positions = np.array([
        [0.0, 0.0, 0.075],
        [0.0, 0.0, -1.268]
    ])

    output_file = "Hamiltonians/HCl_STO3g_20qubits/jw_generated.txt"

    print("Building HCl Hamiltonian with STO-3G basis...")
    print("Atom coordinates:")
    for label, pos in zip(atom_labels, positions):
        print(f"  {label} {pos[0]:.3f} {pos[1]:.3f} {pos[2]:.3f}")
    print("=" * 60)

    output_str, e_fci = generate_jw_format_file(
        atom_labels=atom_labels,
        positions=positions,
        basis="sto-3g",
        charge=0,
        spin=0,
        output_file=output_file
    )

    return output_str, e_fci


# %%
if __name__ == "__main__":
    # Generate H2O Hamiltonian with exact coordinates
    # h2o_output = generate_h2o_sto3g_14qubits()

    # Generate NH3 Hamiltonian
    # nh3_output = generate_nh3_sto3g_16qubits()

    # Generate H2 with 6-31G basis
    # h2_output = generate_h2_631g_8qubits()

    pass
