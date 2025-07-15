# -*- coding: utf-8 -*-
# optimized_clifford_map.py
import torch
import math
from gf2_ops import GF2Ops
from typing import Tuple, List, Optional, Dict, Union
from collections import defaultdict


class TensorPool:
    """Memory pool for reusing temporary tensors with device awareness"""
    def __init__(self, device: torch.device):
        self.device = device
        self.bool_vectors = defaultdict(list)
        self.bool_matrices = defaultdict(list)
    
    def get_bool_vector(self, size: int) -> torch.Tensor:
        if self.bool_vectors[size]:
            return self.bool_vectors[size].pop().zero_()
        return torch.zeros(size, dtype=torch.bool, device=self.device)
    
    def get_bool_matrix(self, shape: Tuple[int, int]) -> torch.Tensor:
        if self.bool_matrices[shape]:
            return self.bool_matrices[shape].pop().zero_()
        return torch.zeros(shape, dtype=torch.bool, device=self.device)
    
    def return_tensor(self, tensor: torch.Tensor):
        if tensor.dim() == 1:
            self.bool_vectors[tensor.shape[0]].append(tensor)
        elif tensor.dim() == 2:
            self.bool_matrices[tensor.shape].append(tensor)


class CliffordMap:

    def __init__(self, n_qubits: int, batch_size: int, n_measurements: int, device: Union[str, torch.device] = 'cpu'):
        self.n_qubits = n_qubits
        self.batch_size = batch_size
        self.n_measurements = n_measurements
        self.device = torch.device(device) if isinstance(device, str) else device
        self.N2 = 2 * n_qubits
        
        # Heisenberg representation (the Clifford map)
        self.W = torch.eye(self.N2, dtype=torch.bool, device=self.device).unsqueeze(0).unsqueeze(0).expand(batch_size, n_measurements, -1, -1).contiguous()
        self.heis_phase_vec = torch.zeros(batch_size, n_measurements, self.N2, dtype=torch.int8, device=self.device)  # 0: +1, 1: +i, 2: -1, 3: -i

        # Active mask
        self.active = torch.ones(batch_size, n_measurements, dtype=torch.bool, device=self.device)
        
        # Pre-allocate buffers
        self._flat_buffer = None
        self._pauli_cache = {}
        
        # Create device-specific tensor pool
        self._pool = TensorPool(self.device)
        
        # Optimized flattened views for GPU operations
        self.total_tableaus = batch_size * n_measurements
        self._setup_flattened_views()
    
    def _pauli_commutation_phase(self, has_x1: bool, has_z1: bool, has_x2: bool, has_z2: bool) -> int:
        """
        Calculate the phase from commuting two Paulis.
        Returns phase mod 4 for P1 * P2 vs P2 * P1
        
        Using: XY = iZ, YZ = iX, ZX = iY
               YX = -iZ, ZY = -iX, XZ = -iY
        """
        # If they commute, return 0
        if (has_x1 and has_x2) or (has_z1 and has_z2) or not (has_x1 or has_z1) or not (has_x2 or has_z2):
            return 0
        
        # For Y operators, we need both X and Z
        is_y1 = has_x1 and has_z1
        is_y2 = has_x2 and has_z2
        
        # Handle all cases
        if has_x1 and not has_z1:  # P1 = X
            if has_z2 and not has_x2:  # P2 = Z
                return 3  # XZ = -iY, ZX = iY, so XZ - ZX gives phase -2i = 2
            elif is_y2:  # P2 = Y
                return 1  # XY = iZ
        elif has_z1 and not has_x1:  # P1 = Z
            if has_x2 and not has_z2:  # P2 = X
                return 1  # ZX = iY
            elif is_y2:  # P2 = Y
                return 3  # ZY = -iX
        elif is_y1:  # P1 = Y
            if has_x2 and not has_z2:  # P2 = X
                return 3  # YX = -iZ
            elif has_z2 and not has_x2:  # P2 = Z
                return 1  # YZ = iX
        
        return 0
    
    def _setup_flattened_views(self):
        """Create flattened views for efficient GPU operations"""
        # Create views without copying data
        self._W_flat = self.W.view(self.total_tableaus, self.N2, self.N2)
        self._heis_phase_vec_flat = self.heis_phase_vec.view(self.total_tableaus, self.N2)
        self._active_flat = self.active.view(self.total_tableaus)
        
        # Pre-compute index mappings
        self._flat_to = torch.arange(self.total_tableaus, device=self.device) // self.n_measurements
        self._flat_to_meas = torch.arange(self.total_tableaus, device=self.device) % self.n_measurements
    
    def _ensure_device(self, tensor: torch.Tensor) -> torch.Tensor:
        """Ensure tensor is on the correct device"""
        if tensor.device != self.device:
            return tensor.to(self.device)
        return tensor
    
    @torch.no_grad()
    def apply_H(self, qubit: int, mask: Optional[torch.Tensor] = None):
        """Vectorized Hadamard gate application: X → Z, Y → -Y, Z → X"""
        if mask is None:
            mask = self.active
        else:
            mask = self._ensure_device(mask)
        
        if not mask.any():
            return
        
        x_idx = qubit
        z_idx = self.n_qubits + qubit
        
        # Work with flattened tensors for efficiency
        mask_flat = mask.view(-1)
        active_indices = mask_flat.nonzero(as_tuple=True)[0]
        
        if len(active_indices) == 0:
            return
        
        # Get columns before swap
        x_cols = self._W_flat[active_indices, :, x_idx].clone()
        z_cols = self._W_flat[active_indices, :, z_idx].clone()
        
        # Track Y operators (where both x and z are set) for phase
        has_y = x_cols & z_cols
        
        # Swap columns: X ↔ Z
        self._W_flat[active_indices, :, x_idx] = z_cols
        self._W_flat[active_indices, :, z_idx] = x_cols
        
        # Phase update for all Y operators in one shot
        if has_y.any():
            phase_updates = has_y.to(torch.int8) * 2
            self._heis_phase_vec_flat[active_indices] ^= phase_updates
    
    @torch.no_grad()
    def apply_S(self, qubit: int, mask: Optional[torch.Tensor] = None):
        """Vectorized S gate application: X → Y(iXZ), Y → -X, Z → Z"""
        if mask is None:
            mask = self.active
        else:
            mask = self._ensure_device(mask)
        
        if not mask.any():
            return
        
        x_idx = qubit
        z_idx = self.n_qubits + qubit
        
        # Work with flattened tensors
        mask_flat = mask.view(-1)
        active_indices = mask_flat.nonzero(as_tuple=True)[0]
        
        if len(active_indices) == 0:
            return
        
        # S gate: X → Y, Y → -X, Z → Z
        # In symplectic: Y = iXZ, so X → iXZ, iXZ → -X (with phase)
        x_cols = self._W_flat[active_indices, :, x_idx].clone()
        z_cols = self._W_flat[active_indices, :, z_idx]#.clone()
        
        # Identify operator types BEFORE transformation
        has_x = x_cols #& ~z_cols
        #has_y = x_cols & z_cols
        
        # X → Y means column x becomes x ⊕ z
        self._W_flat[active_indices, :, x_idx] = x_cols ^ z_cols
        
        # Phase updates for X → Y = iXZ
        if has_x.any():
            phase_updates = has_x.to(torch.int8)
            self._heis_phase_vec_flat[active_indices] = (
                self._heis_phase_vec_flat[active_indices] + phase_updates
            ) % 4

        # Y → -X would also add 1, but is omitted as in the original code
    
    @torch.no_grad()
    def apply_HS(self, qubit: int, mask: Optional[torch.Tensor] = None):
        """Vectorized HS gate application (S then H): X → -Y, Y → -Z, Z → X"""
        if mask is None:
            mask = self.active
        else:
            mask = self._ensure_device(mask)
        
        if not mask.any():
            return

        self.apply_S(qubit, mask)
        self.apply_H(qubit, mask)

    @torch.no_grad()
    def apply_SH(self, qubit: int, mask: Optional[torch.Tensor] = None):
        """Vectorized SH gate application (H then S): X → Z, Y → X, Z → Y"""
        if mask is None:
            mask = self.active
        else:
            mask = self._ensure_device(mask)
        
        if not mask.any():
            return

        self.apply_H(qubit, mask)
        self.apply_S(qubit, mask)

    @torch.no_grad()
    def apply_HSH(self, qubit: int, mask: Optional[torch.Tensor] = None):
        """Vectorized HSH gate application (H then S then H)"""
        if mask is None:
            mask = self.active
        else:
            mask = self._ensure_device(mask)
        
        if not mask.any():
            return

        self.apply_H(qubit, mask)
        self.apply_S(qubit, mask)
        self.apply_H(qubit, mask)

    def apply_CNOT(self, control: int, target: int, mask: Optional[torch.Tensor] = None):
        """Vectorized CNOT application"""
        if mask is None:
            mask = self.active
        else:
            mask = self._ensure_device(mask)
        
        if not mask.any():
            return
        
        cx = control
        tx = target
        cz = self.n_qubits + control
        tz = self.n_qubits + target
        
        # quadratic phase mask
        b_idx, m_idx = mask.nonzero(as_tuple=True)          # 1-D index lists
        
        # Work with flattened tensors
        mask_flat = mask.view(-1)
        active_indices = mask_flat.nonzero(as_tuple=True)[0]
        
        if len(active_indices) == 0:
            return
        
        # Update W matrices
        self._W_flat[active_indices, tx] ^= self._W_flat[active_indices, cx]
        self._W_flat[active_indices, cz] ^= self._W_flat[active_indices, tz]

    def apply_actions_step(self,
                           actions: torch.Tensor,
                           action_map: Dict[int, Tuple],
                           mask: Optional[torch.Tensor] = None):
        """Apply a single layer of actions in a vectorized manner."""
        if mask is None:
            mask = self.active
        else:
            mask = self._ensure_device(mask)

        if not mask.any():
            return

        active_indices = mask.nonzero(as_tuple=True)
        step_actions = actions[active_indices]
        unique_actions, inverse = torch.unique(step_actions, return_inverse=True)

        for i, act in enumerate(unique_actions):
            gate_tuple = action_map.get(int(act))
            if gate_tuple is None or gate_tuple[0] == "terminal":
                continue

            sub_mask = torch.zeros_like(mask)
            sel = inverse == i
            sub_mask[active_indices[0][sel], active_indices[1][sel]] = True

            gate_name = gate_tuple[0]
            qubits = gate_tuple[1:]

            if gate_name == "H":
                self.apply_H(qubits[0], sub_mask)
            elif gate_name == "S":
                self.apply_S(qubits[0], sub_mask)
            elif gate_name == "HS":
                self.apply_HS(qubits[0], sub_mask)
            elif gate_name == "SH":
                self.apply_SH(qubits[0], sub_mask)
            elif gate_name == "HSH":
                self.apply_HSH(qubits[0], sub_mask)
            elif gate_name == "CNOT":
                self.apply_CNOT(qubits[0], qubits[1], sub_mask)

    def apply_action(self,
                             batch_actions: torch.Tensor,
                             batch_lengths: torch.Tensor,
                             action_map: Dict[int, Tuple]):
        """Apply a set of circuits in parallel using vectorized gate calls."""
        max_len = batch_actions.shape[2]
        for step in range(max_len):
            step_mask = batch_lengths > step
            if not step_mask.any():
                break

            actions_step = batch_actions[:, :, step]
            self.apply_actions_step(actions_step, action_map, step_mask)

    def _pauli_string_to_symplectic_vectorized(self, pauli_strings: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert Pauli strings to symplectic vectors"""
        n_strings = len(pauli_strings)
        vecs = torch.zeros(n_strings, self.N2, dtype=torch.bool, device=self.device)
        phases = torch.zeros(n_strings, dtype=torch.int8, device=self.device)  # FIXED: int8
        
        # Process each string
        for i, pauli_string in enumerate(pauli_strings):
            if pauli_string in self._pauli_cache:
                vec, phase = self._pauli_cache[pauli_string]
                vecs[i] = self._ensure_device(vec)
                phases[i] = phase
                continue
            
            phase = 0  # FIXED: Start with 0, not False
            ps = pauli_string
            
            if pauli_string.startswith(('+i', '-i', '+', '-')):
                # Parse phase prefix
                if pauli_string.startswith('+i'):
                    phase = 1
                    ps = pauli_string[2:]
                elif pauli_string.startswith('-i'):
                    phase = 3
                    ps = pauli_string[2:]
                elif pauli_string.startswith('+'):
                    phase = 0
                    ps = pauli_string[1:]
                elif pauli_string.startswith('-'):  # starts with '-'
                    phase = 2
                    ps = pauli_string[1:]
                else:
                    raise ValueError(f"Invalid Pauli string prefix: {pauli_string}")
            
            if len(ps) != self.n_qubits:
                raise ValueError(f"Pauli string length must match number of qubits. Got {len(ps)} for {self.n_qubits} qubits.")
            
            vec = torch.zeros(self.N2, dtype=torch.bool, device=self.device)  # FIXED: indentation
            
            for j, ch in enumerate(ps):
                if ch == 'X':
                    vec[j] = True
                elif ch == 'Y':
                    vec[j] = True
                    vec[self.n_qubits + j] = True
                    phase = (phase + 1) % 4  # FIXED: update phase, not phases[i]
                elif ch == 'Z':
                    vec[self.n_qubits + j] = True
                elif ch != 'I':
                    raise ValueError(f"Invalid Pauli char: {ch}")
            
            vecs[i] = vec
            phases[i] = phase
            
            # Cache the result
            self._pauli_cache[pauli_string] = (vec.clone(), phase)
        
        return vecs, phases

    def transform_paulis(self, pauli_vecs: torch.Tensor, chunk_size: int = 4096) -> torch.Tensor:
        """Transform Pauli vectors under all Clifford maps.

        Args:
            pauli_vecs: Tensor of shape (n_paulis, 2*n) in symplectic form.
            chunk_size: Process tableaus in chunks to save memory.

        Returns:
            Tensor of shape (batch_size, n_measurements, n_paulis, 2*n) containing
            the transformed Pauli vectors.
        """
        n_paulis = pauli_vecs.shape[0]

        # Flattened views for efficiency
        W_all = self._W_flat  # [total_tableaus, 2n, 2n]
        n_tableaus = self.total_tableaus

        result = torch.zeros(n_tableaus, n_paulis, self.N2,
                             dtype=torch.bool, device=self.device)

        pauli_vecs_t = pauli_vecs.T  # (2n, n_paulis)

        for start in range(0, n_tableaus, chunk_size):
            end = min(start + chunk_size, n_tableaus)
            W_chunk = W_all[start:end]

            # Invert chunk in one call
            W_inv_chunk = GF2Ops.invert_matrix(W_chunk, validate=False)

            # Multiply in GF(2) -> (chunk, 2n, n_paulis)
            p_out = GF2Ops.matmul(W_inv_chunk, pauli_vecs_t)

            # Reorder to (chunk, n_paulis, 2n)
            result[start:end] = p_out.transpose(1, 2)

        result = result.view(self.batch_size, self.n_measurements,
                             n_paulis, self.N2)
        return result

    def prob_P_multi(self, pauli_strings: List[str]) -> torch.Tensor:
        """Compute probability of Pauli strings being stabilizers for ALL tableaus"""
        n_strings = len(pauli_strings)

        # Convert Pauli strings
        p_vecs, _ = self._pauli_string_to_symplectic_vectorized(pauli_strings)

        # Transform all paulis in parallel
        p_out = self.transform_paulis(p_vecs)

        # Determine measurability (no X component in U^† P U)
        has_x = p_out[..., :self.n_qubits].any(dim=3)
        probs = (~has_x).float()
        return probs

    def prob_P_single(self, pauli_string: str) -> torch.Tensor:
        """Compute probability for a single Pauli string across all trajectories."""
        result = self.prob_P_multi([pauli_string])
        return result.squeeze(2)  # Remove the n_strings dimension
    
    def phase_to_complex(self, phase: int) -> complex:
        """Convert phase mod 4 to complex number"""
        return [1, 1j, -1, -1j][phase % 4]
    
    def to_flat_tensors_active_only(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """GPU-optimized flat tensor conversion using W matrix and phase vector"""
        mat_size = self.N2 * self.N2
        feature_size = mat_size + self.N2
        
        # Get active indices efficiently
        active_indices = self._active_flat.nonzero(as_tuple=True)[0]
        n_active = len(active_indices)
        
        if n_active == 0:
            return (torch.empty((0, feature_size), device=self.device), 
                    torch.empty((0, 2), dtype=torch.long, device=self.device))
        
        # Select active Clifford maps
        active_W = self._W_flat[active_indices].view(n_active, mat_size).float()
        active_phases = self._heis_phase_vec_flat[active_indices].float()
        
        # Concatenate
        output = torch.cat([active_W, active_phases], dim=1)
        
        # Get indices
        batch_indices = self._flat_to[active_indices]
        meas_indices = self._flat_to_meas[active_indices]
        indices = torch.stack([batch_indices, meas_indices], dim=1)
        
        return output, indices
    
    def forward_mlp_masked(self, mlp: torch.nn.Module) -> torch.Tensor:
        """GPU-optimized MLP forward pass using Clifford map representation"""
        # Get active tensors
        flat_active, indices = self.to_flat_tensors_active_only()
        
        if flat_active.shape[0] == 0:
            # Determine output dimension by running a dummy forward pass
            dummy_input = torch.zeros(1, self.N2 * self.N2 + self.N2, device=self.device)
            with torch.no_grad():
                output_dim = mlp(dummy_input).shape[-1]
            return torch.zeros(self.batch_size, self.n_measurements, output_dim, device=self.device)
        
        # Process through MLP
        mlp_output = mlp(flat_active)
        
        # Create full output tensor
        output_dim = mlp_output.shape[-1]
        full_output = torch.zeros(self.batch_size, self.n_measurements, output_dim, device=self.device)
        
        # Scatter results back
        batch_indices, meas_indices = indices[:, 0], indices[:, 1]
        full_output[batch_indices, meas_indices] = mlp_output
        
        return full_output
    
    def reset_measurement(self, batch_idx: int, meas_idx: int):
        """Reset a specific measurement to identity"""
        self.W[batch_idx, meas_idx] = torch.eye(self.N2, dtype=torch.bool, device=self.device)
        self.heis_phase_vec[batch_idx, meas_idx] = torch.zeros(self.N2, dtype=torch.int8, device=self.device)  # FIXED
        self.active[batch_idx, meas_idx] = True

    def reset(self):
        """Reset all measurements to identity"""
        # Reset all W matrices [batch_size, n_measurements, N2, N2]
        self.W[:] = torch.eye(self.N2, dtype=torch.bool, device=self.device)
        # Reset all phase vectors [batch_size, n_measurements, N2]  
        self.heis_phase_vec = torch.zeros(self.batch_size, self.n_measurements, self.N2, dtype=torch.int8, device=self.device) 
        # Reset active mask [batch_size, n_measurements]
        self.active.fill_(True)
    
    def get_clifford_map(self, batch_idx: int, meas_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a specific Clifford map and phase vector"""
        return self.W[batch_idx, meas_idx].clone(), self.heis_phase_vec[batch_idx, meas_idx].clone()
    
    def to_flat_tensors(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert Clifford maps to flat tensors for MLP input, with mask for active maps."""
        mat_size = self.N2 * self.N2
        
        # Use flattened views
        flat_W = self._W_flat.view(self.total_tableaus, mat_size).float()
        flat_phases = self._heis_phase_vec_flat.float()
        
        # Concatenate
        output = torch.cat([flat_W, flat_phases], dim=1)
        
        return output, self._active_flat.clone()
    
    def to_flat_tensors_vectorized(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Vectorized conversion for efficient batch processing."""
        mat_size = self.N2 * self.N2
        
        # Reshape W matrices and phases
        flat_W = self.W.view(self.batch_size, self.n_measurements, mat_size).float()
        flat_phases = self.heis_phase_vec.float()
        
        # Concatenate
        output = torch.cat([flat_W, flat_phases], dim=2)
        
        return output, self.active.clone()


    def _get_active_indices_flat(self, mask: torch.Tensor) -> Tuple[torch.Tensor, int]:
        """Convert 2D mask to flat indices for efficient GPU operations"""
        batch_indices, meas_indices = torch.where(mask)
        n_active = len(batch_indices)
        flat_indices = batch_indices * self.n_measurements + meas_indices
        return flat_indices, n_active


if __name__ == "__main__":
    # Example usage with the new class
    clifford_map = CliffordMap(n_qubits=10, batch_size=1000, 
                                      n_measurements=100, device='cpu')
    clifford_map.apply_H(0)
    clifford_map.apply_CNOT(0, 1)
    probs = clifford_map.prob_P_multi(['XIXI', 'ZZZZ'])
    
    # Get a specific Clifford map
    W, phase = clifford_map.get_clifford_map(0, 0)
