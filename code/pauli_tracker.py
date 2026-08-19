# -*- coding: utf-8 -*-
"""
DEPRECATED — transitional legacy migration shim (do NOT add features).

Part of the legacy measurement stack, retained for the CPU-fallback /
parity-oracle path. Reusable Pauli-conjugation / phase-tracking primitives
belong in the shared ``clifford-tableau`` core; the CUDA training/cost hot path
uses ``measurement_adapter`` (CT), not this. Per the project's
"treat the legacy measurement stack as transitional" rule, do not grow this
shim. See ``__deprecated__`` below.

Pauli string representation and manipulation with phase tracking.

Notation:
=========

Symplectic Representation:
    A Pauli string on n qubits is represented as:
    - x: bool tensor of shape (n,) - X component bits
    - z: bool tensor of shape (n,) - Z component bits
    - phase: int8 in {0,1,2,3} representing i^phase = {+1, +i, -1, -i}

    Pauli codes: I=0 (x=0,z=0), X=1 (x=1,z=0), Z=2 (x=0,z=1), Y=3 (x=1,z=1)

Stim Convention (Y = XZ, not iXZ):
    - stim.PauliString("Y") has sign +1, not +i
    - Phase k=0 for all basic Paulis including Y
    - Phase only accumulates during multiplication or Clifford conjugation

Clifford Application:
    apply_clifford() computes P' = U P U† (Heisenberg picture)
    The output phase follows the Aaronson-Gottesman formula.

Phase Table (_PAULI_PHASE_TABLE):
    Entry [a, b] gives the phase increment when multiplying Pauli code a by code b.

Code Table (_PAULI_CODE_TABLE):
    Entry [a, b] gives the resulting Pauli code when multiplying a by b.
"""

__deprecated__ = True  # legacy migration shim (transitional); see module docstring. Do not add features — reusable logic -> clifford-tableau.

import torch
from typing import Tuple, List, Optional, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from clifford_map import CliffordMap

# Phase increment table for Pauli multiplication
# _PAULI_PHASE_TABLE[a, b] = phase increment (mod 4) when multiplying Pauli codes a * b
# Codes: I=0, X=1, Z=2, Y=3
_PAULI_PHASE_TABLE = torch.tensor(
    [
        [0, 0, 0, 0],
        [0, 0, 3, 1],
        [0, 1, 0, 3],
        [0, 3, 1, 0],
    ],
    dtype=torch.int16,
)

_PAULI_CODE_TABLE = torch.tensor(
    [
        [0, 1, 2, 3],
        [1, 0, 3, 2],
        [2, 3, 0, 1],
        [3, 2, 1, 0],
    ],
    dtype=torch.int8,
)
class Pauli:
    """
    A class to represent and manipulate Pauli strings with phase tracking.

    Matches the Stim sign convention (verified):
    - Letters: I, X, Y, Z (Y has no intrinsic i)
    - Y = XZ (not iXZ): stim.PauliString("Y") has sign +1, not +i
    - Phase: Global exponent e in {0, 1, 2, 3} representing {+1, +i, -1, -i}
    - k₀ = 0 for all Paulis: stim.PauliString("YY") has sign +1 (not -1 from k₀=2)
    - Binary Symplectic Form: XZ order (first x bits, then z bits)

    Note: When multiplying X*Z in Stim, you get -iY due to anticommutation,
    but this is separate from Y's intrinsic phase. Y itself has no intrinsic i.
    """
    
    def __init__(self, x: torch.Tensor, z: torch.Tensor, phase: Union[int, torch.Tensor] = 0):
        """
        Initialize a Pauli string or batch of Pauli strings.

        Args:
            x: Boolean tensor of X bits (shape: [n_qubits] or [batch, n_qubits])
            z: Boolean tensor of Z bits (shape: [n_qubits] or [batch, n_qubits])
            phase: Integer or tensor for global phase exponent (mod 4).
        """
        self.x = x
        self.z = z
        
        # Ensure phase is a tensor if x/z are batched
        if isinstance(phase, int):
            if x.ndim > 1:
                self.phase = torch.full((x.shape[0],), phase, dtype=torch.int8, device=x.device)
            else:
                self.phase = torch.tensor(phase, dtype=torch.int8, device=x.device)
        else:
            self.phase = phase.to(dtype=torch.int8)
            
        self.n_qubits = x.shape[-1]
        self.device = x.device

    @classmethod
    def from_string(cls, pauli_str: str, device: Optional[torch.device] = None):
        """Create a Pauli object from a string (e.g., "IXYZ")."""
        n = len(pauli_str)
        x = torch.zeros(n, dtype=torch.bool, device=device)
        z = torch.zeros(n, dtype=torch.bool, device=device)
        
        for i, char in enumerate(pauli_str):
            if char == 'X':
                x[i] = True
            elif char == 'Z':
                z[i] = True
            elif char == 'Y':
                x[i] = True
                z[i] = True
                
        return cls(x, z, phase=0)

    def to_string(self) -> str:
        """Convert single Pauli to string representation."""
        if self.x.ndim > 1:
            raise ValueError("Cannot convert batched Pauli to single string.")
            
        chars = []
        phase_prefix = ["+", "+i", "-", "-i"][self.phase.item() % 4]
        
        for i in range(self.n_qubits):
            xi = self.x[i].item()
            zi = self.z[i].item()
            if not xi and not zi:
                chars.append('I')
            elif xi and not zi:
                chars.append('X')
            elif not xi and zi:
                chars.append('Z')
            else:
                chars.append('Y')
                
        return f"{phase_prefix}{''.join(chars)}"

    def apply_H(self, q: int):
        """
        Apply Hadamard gate on qubit q.

        Bit-level (XZ order):
        - Swap bits on q: x_q, z_q <- z_q, x_q
        - Phase increment: Delta = 2 * (x_q & z_q)
        """
        # Calculate phase increment: 2 if local Pauli is Y (x=1, z=1)
        # Use clone() to avoid in-place modification issues if x and z share memory or are used elsewhere
        xq = self.x[..., q].clone()
        zq = self.z[..., q].clone()
        
        delta = (xq & zq).to(torch.int8) * 2
        self.phase = (self.phase + delta) % 4
        
        # Swap X and Z
        self.x[..., q] = zq
        self.z[..., q] = xq
        
    def apply_S(self, q: int):
        """
        Apply Phase gate (S) on qubit q.

        Bit-level (XZ order):
        - Update Z part: z_q <- z_q XOR x_q
        - Phase increment: Delta = 2 * (x_q & z_q)
        """
        xq = self.x[..., q]
        zq = self.z[..., q]
        
        delta = (xq & zq).to(torch.int8) * 2
        self.phase = (self.phase + delta) % 4
        
        # Update Z
        self.z[..., q] = zq ^ xq

    def apply_CNOT(self, c: int, t: int):
        """
        Apply CNOT gate (control c, target t).

        Bit-level (XZ order):
        - x_t <- x_t XOR x_c
        - z_c <- z_c XOR z_t
        - Phase increment: Delta = 2 * (x_c & z_t & ~(x_t XOR z_c))

        The phase increment formula matches Stim sign convention:
        Δ = 2 if (x_c=1) and (z_t=1) and (x_t == z_c)
        """
        xc = self.x[..., c]
        zc = self.z[..., c]
        xt = self.x[..., t]
        zt = self.z[..., t]
        
        # Phase update: Δ = 2 * (x_c & z_t & ~(x_t ^ z_c))
        # ~(x_t ^ z_c) is equivalent to (x_t == z_c) for boolean values
        # Use explicit equality check for clarity and correctness with boolean tensors
        condition = xc & zt & (xt == zc)
        delta = condition.to(torch.int8) * 2
        self.phase = (self.phase + delta) % 4
        
        # Bit updates
        self.x[..., t] = xt ^ xc
        self.z[..., c] = zc ^ zt

    @staticmethod
    def _multiply_with_row(
        result_codes: torch.Tensor,
        result_phase: torch.Tensor,
        mask: torch.Tensor,
        row_codes: torch.Tensor,
        row_phase: torch.Tensor,
        phase_table: torch.Tensor,
        code_table: torch.Tensor,
    ) -> None:
        """In-place multiply selected Pauli strings by a tableau row (Stim style)."""
        if not torch.any(mask):
            return

        row_codes = row_codes.long()
        subset_codes = result_codes[mask].long()
        row_codes_exp = row_codes.unsqueeze(0).expand_as(subset_codes)
        phase_delta = phase_table[subset_codes, row_codes_exp].sum(dim=-1)
        phase_delta = phase_delta.to(result_phase.dtype)
        new_codes = code_table[subset_codes, row_codes_exp]

        row_phase_val = int(row_phase.item()) & 3
        row_phase_tensor = torch.tensor(
            row_phase_val,
            dtype=result_phase.dtype,
            device=result_phase.device,
        )
        result_phase[mask] = (result_phase[mask] + row_phase_tensor + phase_delta) & 3
        result_codes[mask] = new_codes.to(result_codes.dtype)

    def apply_clifford(self, clifford_map: 'CliffordMap') -> 'Pauli':
        """
        Apply a batch of Clifford tableaus to the Pauli strings.

        Fully vectorized GPU-friendly implementation that processes all batch,
        measurement, and Pauli dimensions in parallel.

        Args:
            clifford_map: A CliffordMap object containing batched tableaus.
                          W shape: (batch, measurements, 2n, 2n)
                          heis_phase_vec shape: (batch, measurements, 2n)

        Returns:
            A new Pauli object with shape (batch, measurements, n_paulis)
            representing U P U†.
        """
        # Dimensions
        # self.x: (K, n) or (n,)
        # clifford_map.W: (B, C, 2n, 2n)
        
        # Ensure we have batched Paulis
        if self.x.ndim == 1:
            x = self.x.unsqueeze(0)
            z = self.z.unsqueeze(0)
            phase = self.phase.unsqueeze(0)
        else:
            x = self.x
            z = self.z
            phase = self.phase
            
        n_paulis = x.shape[0]
        n_qubits = self.n_qubits
        batch_size = clifford_map.batch_size
        n_measurements = clifford_map.n_measurements
        
        # Construct symplectic vectors for Paulis: [x, z]
        # Shape: (K, 2n)
        device = clifford_map.device
        x = x.to(device)
        z = z.to(device)
        k0 = phase.to(device=device).to(torch.int16).view(1, 1, n_paulis).expand(batch_size, n_measurements, n_paulis)

        phase_table = _PAULI_PHASE_TABLE.to(device=device)
        code_table = _PAULI_CODE_TABLE.to(device=device)

        input_codes = (x + (z << 1)).to(torch.int8)  # (K, n)
        result_codes = torch.zeros(
            (batch_size, n_measurements, n_paulis, n_qubits), dtype=torch.int8, device=device
        )
        total_phase = k0.clone()

        W_bool = clifford_map.W.to(device=device, dtype=torch.bool)
        x_bits = W_bool[..., :n_qubits]
        z_bits = W_bool[..., n_qubits:]
        row_codes = x_bits.to(torch.int8) + (z_bits.to(torch.int8) << 1)  # (B, C, 2n, n)
        row_x_codes = row_codes[:, :, :n_qubits, :]  # (B, C, n, n)
        row_z_codes = row_codes[:, :, n_qubits:, :]  # (B, C, n, n)

        phi = clifford_map.heis_phase_vec.to(device=device).to(torch.int16) & 3
        row_x_phase = phi[:, :, :n_qubits]  # (B, C, n)
        row_z_phase = phi[:, :, n_qubits:]  # (B, C, n)

        # Vectorized implementation: process all qubits in parallel
        # Expand input_codes for broadcasting: (K, n) -> (B, C, K, n)
        input_codes_exp = input_codes.unsqueeze(0).unsqueeze(0).expand(batch_size, n_measurements, -1, -1)
        
        # Process each qubit position q
        for q in range(n_qubits):
            # Get input codes for this qubit: (B, C, K)
            code_q = input_codes_exp[:, :, :, q]
            
            # Get row codes and phases for this qubit: (B, C, n)
            row_x_codes_q = row_x_codes[:, :, q, :]  # (B, C, n)
            row_z_codes_q = row_z_codes[:, :, q, :]  # (B, C, n)
            row_x_phase_q = row_x_phase[:, :, q]  # (B, C)
            row_z_phase_q = row_z_phase[:, :, q]  # (B, C)
            
            # Masks for X, Z, Y components: (B, C, K)
            mask_x = (code_q == 1)
            mask_z = (code_q == 2)
            mask_y = (code_q == 3)
            
            # Process X components: multiply by X row
            if mask_x.any():
                # Expand row codes: (B, C, n) -> (B, C, K, n) where K is masked
                # We need to apply row_x_codes_q to each Pauli where mask_x is True
                row_x_exp = row_x_codes_q.unsqueeze(2).expand(-1, -1, n_paulis, -1)  # (B, C, K, n)
                row_x_phase_exp = row_x_phase_q.unsqueeze(2).expand(-1, -1, n_paulis)  # (B, C, K)
                
                # Apply multiplication for X components
                self._multiply_with_row_vectorized(
                    result_codes,
                    total_phase,
                    mask_x,
                    row_x_exp,
                    row_x_phase_exp,
                    phase_table,
                    code_table,
                )
            
            # Process Z components: multiply by Z row
            if mask_z.any():
                row_z_exp = row_z_codes_q.unsqueeze(2).expand(-1, -1, n_paulis, -1)  # (B, C, K, n)
                row_z_phase_exp = row_z_phase_q.unsqueeze(2).expand(-1, -1, n_paulis)  # (B, C, K)
                
                self._multiply_with_row_vectorized(
                    result_codes,
                    total_phase,
                    mask_z,
                    row_z_exp,
                    row_z_phase_exp,
                    phase_table,
                    code_table,
                )
            
            # Process Y components: add phase +1, then multiply by X and Z rows
            if mask_y.any():
                # Add phase increment for Y
                total_phase[mask_y] = (total_phase[mask_y] + 1) & 3
                
                # Multiply by X row
                row_x_exp = row_x_codes_q.unsqueeze(2).expand(-1, -1, n_paulis, -1)  # (B, C, K, n)
                row_x_phase_exp = row_x_phase_q.unsqueeze(2).expand(-1, -1, n_paulis)  # (B, C, K)
                
                self._multiply_with_row_vectorized(
                    result_codes,
                    total_phase,
                    mask_y,
                    row_x_exp,
                    row_x_phase_exp,
                    phase_table,
                    code_table,
                )
                
                # Multiply by Z row
                row_z_exp = row_z_codes_q.unsqueeze(2).expand(-1, -1, n_paulis, -1)  # (B, C, K, n)
                row_z_phase_exp = row_z_phase_q.unsqueeze(2).expand(-1, -1, n_paulis)  # (B, C, K)
                
                self._multiply_with_row_vectorized(
                    result_codes,
                    total_phase,
                    mask_y,
                    row_z_exp,
                    row_z_phase_exp,
                    phase_table,
                    code_table,
                )

        new_x = (result_codes & 1).bool()
        new_z = ((result_codes >> 1) & 1).bool()
        new_phase = total_phase.to(torch.int8)

        return Pauli(new_x, new_z, new_phase)
    
    @staticmethod
    def _multiply_with_row_vectorized(
        result_codes: torch.Tensor,
        result_phase: torch.Tensor,
        mask: torch.Tensor,
        row_codes: torch.Tensor,
        row_phase: torch.Tensor,
        phase_table: torch.Tensor,
        code_table: torch.Tensor,
    ) -> None:
        """
        Vectorized in-place multiply selected Pauli strings by tableau rows.

        Args:
            result_codes: (B, C, K, n) - result codes to update
            result_phase: (B, C, K) - result phases to update
            mask: (B, C, K) - boolean mask indicating which Paulis to update
            row_codes: (B, C, K, n) - row codes to multiply with
            row_phase: (B, C, K) - row phases to add
            phase_table: (4, 4) - phase lookup table
            code_table: (4, 4) - code lookup table
        """
        if not mask.any():
            return
        
        # Convert to long for indexing
        row_codes_long = row_codes.long()  # (B, C, K, n)
        result_codes_long = result_codes.long()  # (B, C, K, n)
        
        # Lookup phase and code updates for all entries: (B, C, K, n)
        phase_deltas = phase_table[result_codes_long, row_codes_long]  # (B, C, K, n)
        new_codes = code_table[result_codes_long, row_codes_long]  # (B, C, K, n)
        
        # Sum phase deltas across qubits: (B, C, K)
        phase_delta_sum = phase_deltas.sum(dim=-1).to(result_phase.dtype)
        
        # Add row phase: (B, C, K)
        row_phase_masked = (row_phase.to(result_phase.dtype) & 3)
        
        # Update phases only where mask is True: (B, C, K)
        result_phase[mask] = (result_phase[mask] + row_phase_masked[mask] + phase_delta_sum[mask]) & 3
        
        # Update codes only where mask is True: (B, C, K, n)
        # Expand mask to match result_codes shape: (B, C, K) -> (B, C, K, n)
        mask_expanded = mask.unsqueeze(-1).expand_as(result_codes)  # (B, C, K, n)
        result_codes[mask_expanded] = new_codes[mask_expanded].to(result_codes.dtype)

    #.. (rest of the class)
