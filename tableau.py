import torch
import math
from gf2_ops import GF2Ops
from abc import ABC, abstractmethod
from typing import Tuple, List, Optional, Dict, Union, Callable

class AbstractCliffordTableau(ABC):
    @abstractmethod
    def apply_I(self, qubit: int) -> None:
        pass

    @abstractmethod
    def apply_H(self, qubit: int) -> None:
        pass

    @abstractmethod
    def apply_S(self, qubit: int) -> None:
        pass

    @abstractmethod
    def apply_HS(self, qubit: int) -> None:
        pass

    @abstractmethod
    def apply_SH(self, qubit: int) -> None:
        pass

    @abstractmethod
    def apply_HSH(self, qubit: int) -> None:
        pass

    @abstractmethod
    def apply_CNOT(self, control: int, target: int) -> None:
        pass

    @abstractmethod
    def apply_SWAP(self, qubit1: int, qubit2: int) -> None:
        pass

    @abstractmethod
    def prob_P(self, pauli_string: str) -> float:
        pass

    @abstractmethod
    def empirical_sample(self, pauli_string: str) -> float:
        pass

    @abstractmethod
    def to_unitary(self) -> torch.Tensor:
        pass



class CliffordTableau(AbstractCliffordTableau):
    """
    This version matches the row/column usage in test_core.py:

      - For n qubits, matrix is 2n x 2n.
      - Columns [0..(n-1)] = X_0,...,X_{n-1}, Columns [n..(2n-1)] = Z_0,...,Z_{n-1}.
      - Rows the same pattern.
      - The S gate is effectively implemented as Z -> Z XOR X (rather than X->X XOR Z).
      - The combos HS, SH, HSH each match the test suite's definitions.
      - Two-qubit gates also follow the standard stabilizer column-flip logic used by the tests.
    """

    def __init__(self, n_qubits: int, device: str = 'cpu'):
        if n_qubits <= 0:
            raise ValueError("Number of qubits must be positive.")
        self.n_qubits = n_qubits
        self.device = device

        N2 = 2 * n_qubits
        # Use boolean to store GF(2) data
        self.matrix = torch.eye(N2, dtype=torch.bool, device=device)
        # Phase = length-2n bool vector
        self.phase = torch.zeros(N2, dtype=torch.bool, device=device)

        # Optional Heisenberg representation
        self.W = torch.eye(N2, dtype=torch.bool, device=device)
        self.heis_phase_vec = torch.zeros(N2, dtype=torch.bool, device=device)

        # Gate history for debugging / composition
        self.gate_history: List[Union[Tuple[str, int], Tuple[str, int, int]]] = []

        # Cache for gate matrices
        self._gate_matrix_cache: Dict[Tuple, Tuple[torch.Tensor, torch.Tensor]] = {}

    # --------------------- State Management ---------------------
    def reset(self) -> None:
        N2 = 2 * self.n_qubits
        self.matrix = torch.eye(N2, dtype=torch.bool, device=self.device)
        self.phase = torch.zeros(N2, dtype=torch.bool, device=self.device)
        self.W = torch.eye(N2, dtype=torch.bool, device=self.device)
        self.heis_phase_vec = torch.zeros(N2, dtype=torch.bool, device=self.device)
        self.gate_history = []

    def copy(self) -> 'CliffordTableau':
        new_tableau = CliffordTableau(self.n_qubits, self.device)
        new_tableau.matrix = self.matrix.clone()
        new_tableau.phase = self.phase.clone()
        new_tableau.W = self.W.clone()
        new_tableau.heis_phase_vec = self.heis_phase_vec.clone()
        new_tableau.gate_history = self.gate_history.copy()
        return new_tableau

    def to_device(self, device: str) -> 'CliffordTableau':
        if device == self.device:
            return self
        new_tableau = CliffordTableau(self.n_qubits, device)
        new_tableau.matrix = self.matrix.to(device)
        new_tableau.phase = self.phase.to(device)
        new_tableau.W = self.W.to(device)
        new_tableau.heis_phase_vec = self.heis_phase_vec.to(device)
        new_tableau.gate_history = self.gate_history.copy()
        return new_tableau

    def save_state(self, path: str) -> None:
        state = {
            'n_qubits': self.n_qubits,
            'matrix': self.matrix,
            'phase': self.phase,
            'W': self.W,
            'heis_phase_vec': self.heis_phase_vec,
            'gate_history': self.gate_history
        }
        torch.save(state, path)

    @classmethod
    def load_state(cls, path: str, device: Optional[str] = None) -> 'CliffordTableau':
        data = torch.load(path, map_location=device or 'cpu')
        tableau = cls(data['n_qubits'], device or 'cpu')
        tableau.matrix = data['matrix']
        tableau.phase = data['phase']
        tableau.W = data['W']
        tableau.heis_phase_vec = data['heis_phase_vec']
        tableau.gate_history = data['gate_history']
        return tableau

    # --------------------- Validation Helpers ---------------------
    def _validate_qubit(self, qubit: int) -> None:
        if not (0 <= qubit < self.n_qubits):
            raise ValueError(f"Qubit index {qubit} out of range for {self.n_qubits} qubits.")

    def _validate_two_qubit(self, q1: int, q2: int) -> None:
        self._validate_qubit(q1)
        self._validate_qubit(q2)
        if q1 == q2:
            raise ValueError("control and target must be distinct qubits.")

    def _validate_qubit_list(self, qubits: List[int]) -> None:
        if not qubits:
            raise ValueError("Qubit list cannot be empty.")
        for q in qubits:
            self._validate_qubit(q)
        if len(set(qubits)) != len(qubits):
            raise ValueError("Duplicate qubit indices not allowed.")

    # ----------------- Single-Qubit Gates -------------------
    def apply_I(self, qubit: int) -> None:
        """Identity gate: no change."""
        self._validate_qubit(qubit)
        self.gate_history.append(("I", qubit))

    def apply_H(self, qubit: int) -> None:
        r"""
        Hadamard: X->Z, Z->X, Y-> -Y.
        (x_j,\,z_j)\;\mapsto\;(z_j,\,x_j)
        \quad\text{and if }(x_j\wedge z_j)=1,\text{ then add }1\text{ (i.e.\ a sign flip) to the row{\prime}s phase.}
        """
        self._validate_qubit(qubit)
        self.gate_history.append(("H", qubit))

        x_idx = qubit
        z_idx = self.n_qubits + qubit

        x_row = self.matrix[:,x_idx].clone()
        z_row = self.matrix[:,z_idx].clone()

        # phase fix for Y
        flip_mask = x_row & z_row
        self.phase ^= flip_mask

        # swap row x_idx and row z_idx
        self.matrix[:,x_idx] = z_row
        self.matrix[:,z_idx] = x_row

        # Heisenberg update
        M_gate, s_vec = self._get_gate_matrix("H", qubit)
        self.W = GF2Ops.matmul(M_gate, self.W)
        self.heis_phase_vec ^= GF2Ops.matmul(M_gate, self.heis_phase_vec.unsqueeze(1)).squeeze(1) ^ s_vec

    def apply_S(self, qubit: int) -> None:
        r"""
        S: X-> Y, Y->-X, Z <->Z. 
        (x_j,\,z_j)\; \mapsto\;(x_j,\;z_j\oplus x_j) \quad\text{and if }(x_j\wedge z_j)=1,
        \text{ then add }1\text{ to the row{\prime}s phase.}
        """
        self._validate_qubit(qubit)
        self.gate_history.append(("S", qubit))

        x_idx = qubit
        z_idx = self.n_qubits + qubit

        x_row = self.matrix[:,x_idx].clone()
        z_row = self.matrix[:,z_idx].clone()

        # if x&z=1 => Y => flip phase
        flip_mask = x_row & z_row
        self.phase ^= flip_mask

        # Z-> Z ^ X
        self.matrix[:,z_idx] = z_row ^ x_row

        # Heisenberg
        M_gate, s_vec = self._get_gate_matrix("S", qubit)
        self.W = GF2Ops.matmul(M_gate, self.W)
        self.heis_phase_vec ^= GF2Ops.matmul(M_gate, self.heis_phase_vec.unsqueeze(1)).squeeze(1) ^ s_vec

    def apply_HS(self, qubit: int) -> None:
        """
        HS: X-> -Y, Y-> -Z, Z-> X.
        """
        self._validate_qubit(qubit)
        self.gate_history.append(("HS", qubit))

        x_idx = qubit
        z_idx = self.n_qubits + qubit

        old_x = self.matrix[x_idx].clone()  # bitmask for X
        old_z = self.matrix[z_idx].clone()  # bitmask for Z

        # Flip sign for old X or Y (i.e., whenever old_x=1)
        flip_mask = old_x
        self.phase ^= flip_mask

        # Map (x,z) -> (x⊕z, x)
        new_x = old_x ^ old_z
        new_z = old_x
        self.matrix[x_idx] = new_x
        self.matrix[z_idx] = new_z

        # Update the global transformation matrices
        M_gate, s_vec = self._get_gate_matrix("HS", qubit)
        self.W = GF2Ops.matmul(M_gate, self.W)
        self.heis_phase_vec ^= (
            GF2Ops.matmul(M_gate, self.heis_phase_vec.unsqueeze(1)).squeeze(1) ^ s_vec
        )

    def apply_SH(self, qubit: int) -> None:
        """
        SH => X->Z, Y->X, Z->Y.
        X->Z: The new row for X is the old row for Z.
        Z->Y: The new row for Z is old_x ⊕ old_z, i.e. the old Y.
        Y->X: Follows automatically because Y is stored as X ⊕ Z.
        """
        self._validate_qubit(qubit)
        self.gate_history.append(("SH", qubit))

        x_idx = qubit
        z_idx = self.n_qubits + qubit

        old_x = self.matrix[x_idx].clone()
        old_z = self.matrix[z_idx].clone()

        # Corrected matrix transformation for SH
        new_x = old_z            # X -> Z
        new_z = old_x ^ old_z    # Z -> Y

        self.matrix[x_idx] = new_x
        self.matrix[z_idx] = new_z

        # Keep the rest of the update logic the same
        M_gate, s_vec = self._get_gate_matrix("SH", qubit)
        self.W = GF2Ops.matmul(M_gate, self.W)
        self.heis_phase_vec ^= (
            GF2Ops.matmul(M_gate, self.heis_phase_vec.unsqueeze(1)).squeeze(1) ^ s_vec
        )

    def apply_HSH(self, qubit: int) -> None:
        """
        Implements the single-qubit map X->X, Y->Z, Z->-Y
        by toggling the bits (x,z) := (x ^ z, z) and then
        flipping phase wherever the new operator is Y.
        """
        self._validate_qubit(qubit)
        self.gate_history.append(("HSH", qubit))

        x_idx = qubit
        z_idx = self.n_qubits + qubit

        old_x = self.matrix[:,x_idx].clone()
        old_z = self.matrix[:,z_idx].clone()

        # First do (x,z) -> (x⊕z, z)
        new_x = old_x ^ old_z
        new_z = old_z

        # Now flip sign if the new operator is Y => new_x=1 and new_z=1
        flip_mask = new_x & new_z
        self.phase ^= flip_mask

        # Store them back
        self.matrix[:,x_idx] = new_x
        self.matrix[:,z_idx] = new_z

        # Finally update the global symplectic matrix and phases
        M_gate, s_vec = self._get_gate_matrix("HSH", qubit)
        self.W = GF2Ops.matmul(M_gate, self.W)
        self.heis_phase_vec ^= (
            GF2Ops.matmul(M_gate, self.heis_phase_vec.unsqueeze(1)).squeeze(1)
            ^ s_vec
        )

    # ----------------- Batch Gates -------------------
    def apply_gates_in_parallel(self, gate_type: str, qubits: List[int]) -> None:
        """
        The test code also checks parallel gates. We do the row changes in one pass,
        then update the Heisenberg representation sequentially.
        """
        self._validate_qubit_list(qubits)
        if gate_type not in ["I","H","S","HS","SH","HSH"]:
            raise ValueError(f"Unsupported gate type: {gate_type}")

        for q in qubits:
            self.gate_history.append((gate_type, q))

        if gate_type == "I":
            return

        x_inds = torch.tensor(qubits, device=self.device)
        z_inds = x_inds + self.n_qubits

        if gate_type == "H":
            # old_x => row x, old_z => row z
            old_x = self.matrix[x_inds].clone()
            old_z = self.matrix[z_inds].clone()
            # Flip phase for any Y
            for i in range(len(qubits)):
                flip_mask = old_x[i] & old_z[i]
                self.phase ^= flip_mask
            # swap
            self.matrix[x_inds] = old_z
            self.matrix[z_inds] = old_x

        elif gate_type == "S":
            for q in qubits:
                x_idx = q
                z_idx = q + self.n_qubits
                rowX = self.matrix[x_idx].clone()
                rowZ = self.matrix[z_idx].clone()
                flip_mask = rowX & rowZ
                self.phase ^= flip_mask
                self.matrix[z_idx] = rowZ ^ rowX

        elif gate_type == "HS":
            old_x = self.matrix[x_inds].clone()
            old_z = self.matrix[z_inds].clone()
            new_x = old_x ^ old_z
            new_z = old_x
            self.matrix[x_inds] = new_x
            self.matrix[z_inds] = new_z

        elif gate_type == "SH":
            old_x = self.matrix[x_inds].clone()
            old_z = self.matrix[z_inds].clone()
            new_x = old_z ^ old_x
            new_z = old_x
            self.matrix[x_inds] = new_x
            self.matrix[z_inds] = new_z

        elif gate_type == "HSH":
            old_x = self.matrix[x_inds].clone()
            old_z = self.matrix[z_inds].clone()
            new_x = old_x ^ old_z
            new_z = old_z
            self.matrix[x_inds] = new_x
            self.matrix[z_inds] = new_z

        # Heisenberg for each qubit
        for q in qubits:
            M_gate, s_vec = self._get_gate_matrix(gate_type, q)
            self.W = GF2Ops.matmul(M_gate, self.W)
            self.heis_phase_vec ^= GF2Ops.matmul(M_gate, self.heis_phase_vec.unsqueeze(1)).squeeze(1) ^ s_vec

    # ----------------- Two-Qubit Gates -------------------
    def apply_CNOT(self, control: int, target: int) -> None:
        self._validate_two_qubit(control, target)
        self.gate_history.append(("CNOT", control, target))

        # X_t => X_t ^ X_c
        # Z_c => Z_c ^ Z_t
        cx = control
        tx = target
        cz = self.n_qubits + control
        tz = self.n_qubits + target

        self.matrix[tx] ^= self.matrix[cx]
        self.matrix[cz] ^= self.matrix[tz]

        M_gate, s_vec = self._get_gate_matrix("CNOT", control, target)
        self.W = GF2Ops.matmul(M_gate, self.W)
        self.heis_phase_vec ^= GF2Ops.matmul(M_gate, self.heis_phase_vec.unsqueeze(1)).squeeze(1) ^ s_vec

    def apply_SWAP(self, q1: int, q2: int) -> None:
        self._validate_two_qubit(q1, q2)
        self.gate_history.append(("SWAP", q1, q2))

        # Swap X_q1 <-> X_q2, Z_q1 <-> Z_q2
        x1,x2 = q1,q2
        z1,z2 = q1+self.n_qubits, q2+self.n_qubits
        tmp = self.matrix[x1].clone()
        self.matrix[x1] = self.matrix[x2]
        self.matrix[x2] = tmp

        tmp = self.matrix[z1].clone()
        self.matrix[z1] = self.matrix[z2]
        self.matrix[z2] = tmp

        M_gate, s_vec = self._get_gate_matrix("SWAP", q1, q2)
        self.W = GF2Ops.matmul(M_gate, self.W)
        self.heis_phase_vec ^= GF2Ops.matmul(M_gate, self.heis_phase_vec.unsqueeze(1)).squeeze(1) ^ s_vec

    def apply_CZ(self, control: int, target: int) -> None:
        """
        Apply a CZ gate between qubit 'control' and qubit 'target'.
        Toggles Z bits on control/target if the other qubit has X = 1.
        Updates the global Heisenberg-W representation and phase vector
        using a dedicated gate matrix from self._get_gate_matrix(...).
        """
        self._validate_two_qubit(control, target)
        self.gate_history.append(("CZ", control, target))

        # Identify which columns in 'self.matrix' hold X vs Z for each qubit
        cx = control
        tx = target
        cz = self.n_qubits + control
        tz = self.n_qubits + target

        # z_c => z_c ^ x_t
        # z_t => z_t ^ x_c
        self.matrix[cz] ^= self.matrix[tx]
        self.matrix[tz] ^= self.matrix[cx]

        # Post-update the global W representation and phases
        M_gate, s_vec = self._get_gate_matrix("CZ", control, target)
        self.W = GF2Ops.matmul(M_gate, self.W)
        self.heis_phase_vec ^= (
            GF2Ops.matmul(M_gate, self.heis_phase_vec.unsqueeze(1)).squeeze(1)
            ^ s_vec
        )

    # -------------------- Gate Matrix Generators -------------
    def _get_gate_matrix(self, gate_type: str, *args) -> Tuple[torch.Tensor, torch.Tensor]:
        cache_key = (gate_type, *args)
        if cache_key in self._gate_matrix_cache:
            return self._gate_matrix_cache[cache_key]

        if gate_type == "I":
            N2 = 2*self.n_qubits
            M = torch.eye(N2, dtype=torch.bool, device=self.device)
            s_vec = torch.zeros(N2, dtype=torch.bool, device=self.device)
        elif gate_type == "H":
            M, s_vec = self._gate_matrix_H(args[0])
        elif gate_type == "S":
            M, s_vec = self._gate_matrix_S(args[0])
        elif gate_type in ("HS","SH","HSH"):
            if gate_type == "HS":
                M, s_vec = self._gate_matrix_HS(args[0])
            elif gate_type == "SH":
                M, s_vec = self._gate_matrix_SH(args[0])
            else:
                M, s_vec = self._gate_matrix_HSH(args[0])
        elif gate_type == "CNOT":
            M, s_vec = self._gate_matrix_CNOT(args[0], args[1])
        elif gate_type == "SWAP":
            M, s_vec = self._gate_matrix_SWAP(args[0], args[1])
        elif gate_type == "CZ":
            M, s_vec = self._gate_matrix_CZ(args[0], args[1])
        else:
            raise ValueError(f"Unknown gate type: {gate_type}")

        self._gate_matrix_cache[cache_key] = (M, s_vec)
        return (M, s_vec)

    def _gate_matrix_H(self, qubit: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        H: (x,z) -> (z,x).
        """
        N2 = 2*self.n_qubits
        M = torch.eye(N2, dtype=torch.bool, device=self.device)
        s_vec = torch.zeros(N2, dtype=torch.bool, device=self.device)

        x_idx = qubit
        z_idx = self.n_qubits + qubit

        # (x',z') = (old_z, old_x)
        M[x_idx, x_idx] = False
        M[x_idx, z_idx] = True
        M[z_idx, z_idx] = False
        M[z_idx, x_idx] = True

        return M, s_vec

    def _gate_matrix_S(self, qubit: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        S: (x,z) -> (x, z ^ x).
        """
        N2 = 2*self.n_qubits
        M = torch.eye(N2, dtype=torch.bool, device=self.device)
        s_vec = torch.zeros(N2, dtype=torch.bool, device=self.device)

        x_idx = qubit
        z_idx = self.n_qubits + qubit

        # (x',z') = (x, z ^ x)
        # The x' row is already identity => no change to x
        M[z_idx, x_idx] = True  # toggles z by x

        return M, s_vec

    def _gate_matrix_HS(self, qubit: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        HS: (x,z) -> (x ^ z, x).
        """
        N2 = 2*self.n_qubits
        M = torch.eye(N2, dtype=torch.bool, device=self.device)
        s_vec = torch.zeros(N2, dtype=torch.bool, device=self.device)

        x_idx = qubit
        z_idx = self.n_qubits + qubit

        # (x',z') = (old_x ^ old_z, old_x)
        M[x_idx, x_idx] = True
        M[x_idx, z_idx] = True
        M[z_idx, x_idx] = True
        M[z_idx, z_idx] = False

        return M, s_vec

    def _gate_matrix_SH(self, qubit: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        SH: (x,z) -> (z, x ^ z).
        """
        N2 = 2*self.n_qubits
        M = torch.eye(N2, dtype=torch.bool, device=self.device)
        s_vec = torch.zeros(N2, dtype=torch.bool, device=self.device)

        x_idx = qubit
        z_idx = self.n_qubits + qubit

        # (x',z') = (old_z, old_x ^ old_z)
        M[x_idx, x_idx] = False
        M[x_idx, z_idx] = True
        M[z_idx, x_idx] = True
        M[z_idx, z_idx] = True

        return M, s_vec

    def _gate_matrix_HSH(self, qubit: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        HSH: (x,z) -> (x ^ z, z).
        """
        N2 = 2*self.n_qubits
        M = torch.eye(N2, dtype=torch.bool, device=self.device)
        s_vec = torch.zeros(N2, dtype=torch.bool, device=self.device)

        x_idx = qubit
        z_idx = self.n_qubits + qubit

        # (x',z') = (old_x ^ old_z, old_z)
        M[x_idx, x_idx] = True
        M[x_idx, z_idx] = True
        M[z_idx, x_idx] = False
        M[z_idx, z_idx] = True

        return M, s_vec

    def _gate_matrix_CNOT(self, c: int, t: int):
        """
        CNOT => X_t->X_t ^ X_c, Z_c->Z_c ^ Z_t.
        """
        N2 = 2*self.n_qubits
        M = torch.eye(N2, dtype=torch.bool, device=self.device)
        s_vec = torch.zeros(N2, dtype=torch.bool, device=self.device)
        M[t, :] ^= M[c, :]
        M[self.n_qubits + c, :] ^= M[self.n_qubits + t, :]
        return M, s_vec

    def _gate_matrix_SWAP(self, q1: int, q2: int):
        """
        SWAP => X_q1<->X_q2, Z_q1<->Z_q2
        """
        N2 = 2*self.n_qubits
        M = torch.eye(N2, dtype=torch.bool, device=self.device)
        s_vec = torch.zeros(N2, dtype=torch.bool, device=self.device)
        M[[q1,q2]] = M[[q2,q1]]
        z1 = q1 + self.n_qubits
        z2 = q2 + self.n_qubits
        M[[z1,z2]] = M[[z2,z1]]
        return M, s_vec
    
    def _gate_matrix_CZ(self, c: int, t: int):
        """
        CZ => Z_c->Z_c ^ X_t, Z_t->Z_t ^ X_c
        """
        N2 = 2*self.n_qubits
        M = torch.eye(N2, dtype=torch.bool, device=self.device)
        s_vec = torch.zeros(N2, dtype=torch.bool, device=self.device)
        M[self.n_qubits + c, :] ^= M[t, :]
        M[self.n_qubits + t, :] ^= M[c, :]
        return M, s_vec

    # -------------------- Measurement & Etc. --------------------
    def _pauli_string_to_symplectic(self, pauli_string: str) -> Tuple[torch.Tensor,bool]:
        if len(pauli_string) != self.n_qubits:
            raise ValueError("Pauli string length must match number of qubits.")
        vec = torch.zeros(2*self.n_qubits, dtype=torch.bool, device=self.device)
        phase = False

        # optional ±i prefix
        if pauli_string.startswith("+i") or pauli_string.startswith("-i"):
            phase = (pauli_string[0]=='-')
            pauli_string = pauli_string[2:]
            if len(pauli_string)!= self.n_qubits:
                raise ValueError("Wrong length after ±i prefix in Pauli string.")

        for i,ch in enumerate(pauli_string):
            if ch=='I':
                continue
            elif ch=='X':
                vec[i] = True
            elif ch=='Y':
                vec[i] = True
                vec[self.n_qubits+i] = True
                # Y => i factor
                phase = not phase
            elif ch=='Z':
                vec[self.n_qubits+i] = True
            else:
                raise ValueError(f"Invalid Pauli char: {ch}")
        return vec, phase

    def prob_P(self, pauli_string: str) -> float:
        """
        Return 1.0 if state is +1 eigenstate of that Pauli, else 0.0.
        In stabilizer terms, if p_out[:n] has any 1 => off-diagonal => 0.0
        """
        p_vec, _ = self._pauli_string_to_symplectic(pauli_string)
        W_inv = GF2Ops.invert_matrix(self.W)
        p_out = GF2Ops.matmul(W_inv, p_vec.unsqueeze(1)).squeeze(1)
        if p_out[:self.n_qubits].any():
            return 0.0
        return 1.0

    def empirical_sample(self, pauli_string: str) -> float:
        """
        If p_out has any X bits => 0. else ±1 depending on random parity and net phase.
        """
        p_vec, in_phase = self._pauli_string_to_symplectic(pauli_string)
        W_inv = GF2Ops.invert_matrix(self.W)
        p_out = GF2Ops.matmul(W_inv, p_vec.unsqueeze(1)).squeeze(1)
        if p_out[:self.n_qubits].any():
            return 0.0
        heis_phase = (torch.sum(self.heis_phase_vec & p_vec) %2).item()==1
        phase_bit = in_phase ^ heis_phase
        bits = torch.randint(0,2,(self.n_qubits,),dtype=torch.bool,device=self.device)
        parity = (torch.sum(p_out[self.n_qubits:] & bits)%2).item()==1
        val = -1.0 if parity else 1.0
        return -val if phase_bit else val

    def apply_pauli_rotation(self, pauli_string: str, angle: float) -> None:
        """
        For angles in multiples of π/2, we decompose X->(H->Z->H), Y->(HS->Z->SH), etc.
        """
        norm_ang = angle%(2*math.pi)
        valid = [0,math.pi/2, math.pi, 3*math.pi/2, 2*math.pi]
        if not any(abs(norm_ang - v)<1e-6 for v in valid):
            raise ValueError("Angle must be multiple of pi/2 for a Clifford rotation.")

        if abs(norm_ang)<1e-6 or abs(norm_ang-2*math.pi)<1e-6:
            return
        for i,ch in enumerate(pauli_string):
            if ch=='I':
                continue
            if ch=='X':
                # map X->Z via H
                self.apply_H(i)
            elif ch=='Y':
                # map Y->Z via HS
                self.apply_HS(i)
            # rotation about Z
            if abs(norm_ang - math.pi/2)<1e-6:
                self.apply_S(i)
            elif abs(norm_ang - math.pi)<1e-6:
                self.apply_S(i); self.apply_S(i)
            elif abs(norm_ang - 3*math.pi/2)<1e-6:
                self.apply_S(i); self.apply_S(i); self.apply_S(i)
            # map back
            if ch=='X':
                self.apply_H(i)
            elif ch=='Y':
                self.apply_SH(i)

    def compose(self, other: 'CliffordTableau') -> 'CliffordTableau':
        """
        Compose self with other's gate_history.
        """
        if self.n_qubits != other.n_qubits:
            raise ValueError("Cannot compose tableaux with different n_qubits.")
        result = self.copy()
        for g in other.gate_history:
            gtype = g[0]
            if gtype=="I":
                result.apply_I(g[1])
            elif gtype in ("H","S","HS","SH","HSH"):
                qub = g[1]
                if gtype=="H":   result.apply_H(qub)
                elif gtype=="S": result.apply_S(qub)
                elif gtype=="HS":result.apply_HS(qub)
                elif gtype=="SH":result.apply_SH(qub)
                else:            result.apply_HSH(qub)
            elif gtype=="CNOT":
                _,c,t = g
                result.apply_CNOT(c,t)
            elif gtype=="SWAP":
                _,q1,q2 = g
                result.apply_SWAP(q1,q2)
            elif gtype=="CZ":
                _,c,t = g
                result.apply_CZ(c,t)
        return result

    def to_unitary(self) -> torch.Tensor:
        """
        Reconstruct the 2^n x 2^n unitary by replaying gate_history in a brute-force manner.
        """
        N = self.n_qubits
        dim = 2**N
        U = torch.eye(dim, dtype=torch.complex64, device=self.device)
        # define standard single-qubit unitaries
        H_ = (1.0/math.sqrt(2))*torch.tensor([[1,1],[1,-1]],dtype=torch.complex64,device=self.device)
        S_ = torch.tensor([[1,0],[0,1j]],dtype=torch.complex64,device=self.device)
        HS_ = H_ @ S_
        SH_ = S_ @ H_
        HSH_= H_ @ S_ @ H_
        for gate in self.gate_history:
            g = gate[0]
            if g=="I":
                continue
            elif g in ("H","S","HS","SH","HSH"):
                qub = gate[1]
                if g=="H":
                    U = self._apply_single_qubit(U,H_,qub,N)
                elif g=="S":
                    U = self._apply_single_qubit(U,S_,qub,N)
                elif g=="HS":
                    U = self._apply_single_qubit(U,HS_,qub,N)
                elif g=="SH":
                    U = self._apply_single_qubit(U,SH_,qub,N)
                else: #HSH
                    U = self._apply_single_qubit(U,HSH_,qub,N)
            elif g=="CNOT":
                _,c,t = gate
                U = self._apply_cnot(U,c,t,N)
            elif g=="SWAP":
                _,q1,q2 = gate
                U = self._apply_swap(U,q1,q2,N)
            # "CZ" => H->CNOT->H on target
        return U

    def _apply_single_qubit(self, U: torch.Tensor, gate: torch.Tensor,
                            qubit: int, n_qubits: int)->torch.Tensor:
        """
        Apply a single-qubit 2x2 gate to qubit 'qubit' by enumerating basis states.
        """
        dim = 2**n_qubits
        U_new = torch.zeros_like(U)
        for i in range(dim):
            bit = (i >> qubit)&1
            for b in range(2):
                j = (i & ~(1<<qubit)) | (b<<qubit)
                U_new[j] += gate[b,bit]*U[i]
        return U_new

    def _apply_cnot(self, U: torch.Tensor, control: int, target: int,
                    n_qubits: int)->torch.Tensor:
        """
        Flip target bit if control=1
        """
        dim = 2**n_qubits
        U_new = torch.zeros_like(U)
        for i in range(dim):
            c_bit = (i>>control)&1
            j = i ^ (c_bit<<target)
            U_new[j]+=U[i]
        return U_new

    def _apply_swap(self, U: torch.Tensor, q1: int, q2: int,
                    n_qubits: int)->torch.Tensor:
        """
        SWAP = CNOT(q1,q2) ; CNOT(q2,q1) ; CNOT(q1,q2)
        """
        U = self._apply_cnot(U,q1,q2,n_qubits)
        U = self._apply_cnot(U,q2,q1,n_qubits)
        U = self._apply_cnot(U,q1,q2,n_qubits)
        return U
    
    def get_stabilizer_tableau(self) -> Tuple:
        """
        Return the tableau as a 2n x 2n tensor and the phase vector.
        """
        return self.matrix.clone(), self.phase.clone()
    
    def to_flat_tensor(self):
        mat_flat= self.matrix.flatten().float()
        ph_flat = self.phase.flatten().float()
        return torch.cat([mat_flat, ph_flat], dim=0)