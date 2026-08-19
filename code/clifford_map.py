# -*- coding: utf-8 -*-
"""
DEPRECATED — legacy shim, retained only for the CPU-fallback / parity-oracle
path. The CUDA training and cost hot path uses
``measurement_adapter.TableauBatchAdapter`` (backed by ``clifford-tableau``).
Reusable primitives belong there, not here.

Clifford tableau tracking with batched GPU-accelerated operations.

Notation:
=========

W Matrix (Clifford Tableau):
    Shape: (B, C, 2n, 2n) for B batches, C circuits, n qubits

    Structure:
        W = [ W_XX  W_XZ ]
            [ W_ZX  W_ZZ ]

    - Row q (0 ≤ q < n): Image of X_q generator under the Clifford
    - Row n+q: Image of Z_q generator under the Clifford
    - Columns 0..n-1: X-component contributions
    - Columns n..2n-1: Z-component contributions

    Initially W = I_{2n} (identity Clifford).

Phase Vector (heis_phase_vec):
    Shape: (B, C, 2n)
    Stores Z4 phase exponents for each generator row.
    k ∈ {0,1,2,3} represents i^k = {+1, +i, -1, -i}

Heisenberg Picture:
    Gates transform Paulis as P' = U P U† via column operations on W.
    This matches Stim's tableau convention exactly.

Gate Update Rules:
    H(q):   Swap columns q and n+q; phase += 2 if Y component (x=z=1)
    S(q):   Column n+q ^= column q; phase += 2 if Y component before update
    CNOT(c,t): Column t ^= column c; Column n+c ^= column n+t
               Phase += 2 if x_c & z_t & (x_t == z_c)

Measurability:
    A Pauli P is measurable by circuit U if U P U† has no X component,
    i.e., the transformed Pauli is diagonal in Z basis.
"""

__deprecated__ = True  # legacy migration shim (transitional); see module docstring. Do not add features — reusable logic -> clifford-tableau.
import torch
import math
try:
    from .gf2_ops import GF2Ops
except ImportError:
    from gf2_ops import GF2Ops
from typing import Tuple, List, Optional, Dict, Union
from collections import defaultdict, OrderedDict
import threading

# Vectorized eager kernels for gate application. (Previously @torch.jit.script;
# dropped (not needed: CPU-fallback/parity-oracle path; eager dispatch is sufficient here;
# TorchScript compilation overhead not warranted) — these legacy clifford_map
# helpers run on the CPU-fallback / parity-oracle path, not the CUDA training
# hot path, so eager dispatch over the already-vectorized tensor ops is fine.)
def _apply_H_script(W_flat: torch.Tensor,
                    W_inv_flat: torch.Tensor,
                    phase_flat: torch.Tensor,
                    active_indices: torch.Tensor,
                    qubit: int,
                    n_qubits: int) -> None:
    # H gate on qubit a: Swap X_a and Z_a bits in each row of W
    # Vectorized implementation of Aaronson-Gottesman algorithm
    n = n_qubits

    # Get X and Z bits for qubit across all rows (columns of W matrix)
    # Only clone ONE for the swap (not both) - GPU optimization
    x_bits = W_flat[active_indices, :, qubit].clone()  # Clone this one for swap
    z_bits = W_flat[active_indices, :, n + qubit]  # No clone needed

    # Check for Y components (both x and z set) to update phase
    # H(Y) = -Y, which means phase should flip by 2 (represents -1)
    # With Y = iXZ convention, H conjugates Y to -Y, contributing phase +2
    has_y = x_bits & z_bits  # Element-wise AND across all rows
    phase_inc = (has_y.to(torch.int8) * 2)  # Add 2 for Y (phase flip from -1 factor)
    phase_flat[active_indices] = (phase_flat[active_indices] + phase_inc) & 3

    # Swap X_a and Z_a bits in each row (vectorized)
    W_flat[active_indices, :, qubit] = z_bits
    W_flat[active_indices, :, n + qubit] = x_bits

    # Update W_inv: swap corresponding rows (transpose of column operations)
    # Only clone ONE for the swap (not both) - GPU optimization
    x_rows_inv = W_inv_flat[active_indices, qubit, :].clone()  # Clone this one for swap
    z_rows_inv = W_inv_flat[active_indices, n + qubit, :]  # No clone needed
    W_inv_flat[active_indices, qubit, :] = z_rows_inv
    W_inv_flat[active_indices, n + qubit, :] = x_rows_inv


def _apply_S_script(W_flat: torch.Tensor,
                    W_inv_flat: torch.Tensor,
                    phase_flat: torch.Tensor,
                    active_indices: torch.Tensor,
                    qubit: int,
                    n_qubits: int) -> None:
    # S gate on qubit a: Update z_a based on x_a, update phase based on x_a * z_a
    # Vectorized implementation of Aaronson-Gottesman algorithm
    n = n_qubits

    # Get all x and z bits for this qubit across all rows (BEFORE update)
    x_bits = W_flat[active_indices, :, qubit]  # shape: (n_active, 2*n)
    z_bits = W_flat[active_indices, :, n + qubit]  # shape: (n_active, 2*n)

    # Aaronson-Gottesman S phase increment in Z4:
    # For each generator row i, compute: Δr_i = 2 * (x_i · z_i)
    # This accounts for reordering signs when S conjugates Paulis
    phase_inc = (x_bits & z_bits).to(torch.int8) * 2
    phase_flat[active_indices] = (phase_flat[active_indices] + phase_inc) & 3

    # Bit update: z ← z ⊕ x (vectorized)
    W_flat[active_indices, :, n + qubit] ^= x_bits

    # Update W_inv: For S gate, (M_S^{-1})^T = [[1,1],[0,1]] in GF(2)
    # Left multiply W_inv by (M_S^{-1})^T (transpose because W_inv = (W^{-1})^T):
    # X_a row ← X_a row ⊕ Z_a row, Z_a row unchanged
    z_row_inv = W_inv_flat[active_indices, n + qubit, :]  # Z_a row
    W_inv_flat[active_indices, qubit, :] ^= z_row_inv  # X_a row ← X_a ⊕ Z_a


def _apply_CNOT_script(W_flat: torch.Tensor,
                       W_inv_flat: torch.Tensor,
                       phase_flat: torch.Tensor,
                       active_indices: torch.Tensor,
                       control: int,
                       target: int,
                       n_qubits: int) -> None:
    # CNOT gate: X_c → X_c X_t, X_t → X_t, Z_c → Z_c, Z_t → Z_c Z_t
    # Using column operations on W (consistent with H and S gates)
    n = n_qubits
    cx = control
    tx = target
    cz = n + control
    tz = n + target

    # Aaronson-Gottesman CNOT phase increment in Z4:
    # For each generator row i, compute: Δr_i = 2 * [x_c · z_t · (x_t ⊕ z_c ⊕ 1)]
    # This accounts for reordering signs when CNOT conjugates Paulis
    # Read row bits BEFORE mutating the tableau
    cx_row_bits = W_flat[active_indices, :, cx]  # Row bits for X_control
    tx_row_bits = W_flat[active_indices, :, tx]  # Row bits for X_target
    cz_row_bits = W_flat[active_indices, :, cz]  # Row bits for Z_control
    tz_row_bits = W_flat[active_indices, :, tz]  # Row bits for Z_target

    phase_inc = (cx_row_bits & tz_row_bits & (tx_row_bits ^ cz_row_bits ^ 1)).to(torch.int8) * 2
    phase_flat[active_indices] = (phase_flat[active_indices] + phase_inc) & 3

    # Apply column transformations to W:
    # - Any Pauli with X_c gets X_t added: column tx ^= column cx
    # - Any Pauli with Z_t gets Z_c added: column cz ^= column tz
    W_flat[active_indices, :, tx] ^= cx_row_bits  # X_target column ^= X_control column
    W_flat[active_indices, :, cz] ^= tz_row_bits  # Z_control column ^= Z_target column

    # Update W_inv with corresponding row operations (transpose of column operations):
    # Row operations: row cx ^= row tx, row tz ^= row cz
    tx_rows_inv = W_inv_flat[active_indices, tx, :]  # Row tx (read-only)
    cz_rows_inv = W_inv_flat[active_indices, cz, :]  # Row cz (read-only)
    W_inv_flat[active_indices, cx, :] ^= tx_rows_inv  # Modify row cx
    W_inv_flat[active_indices, tz, :] ^= cz_rows_inv  # Modify row tz


class TensorPool:
    """Memory pool for reusing temporary tensors with device awareness"""
    def __init__(self, device: torch.device):
        self.device = device
        self.bool_vectors: defaultdict[int, List[torch.Tensor]] = defaultdict(list)
        self.bool_matrices: defaultdict[Tuple[int, int], List[torch.Tensor]] = defaultdict(list)
    
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


class LRUCache:
    """Thread-safe LRU cache implementation.

    Pickle-safe: ``threading.Lock`` is not picklable under multiprocessing's
    ``spawn`` start method, which is how ``gfn_async.py`` ships
    ``TrajectoryBatch`` (containing ``CliffordMap`` instances that embed
    these caches) from sampler to learner. Without the
    ``__getstate__/__setstate__`` overrides, a queue serialization failure
    fires in the feeder thread *after* ``Queue.put`` returns — the local
    ``try/except`` in ``sampler_worker`` never sees it, no ``ERROR``
    sentinel is sent, and the learner blocks forever on ``batch_queue.get``.
    Strip the lock on send (along with the cache contents, which are
    regenerated lazily on first access) and reconstruct it on receive.
    """
    def __init__(self, max_size: int):
        self.cache = OrderedDict()
        self.max_size = max_size
        self.lock = threading.Lock()

    def __getstate__(self):
        # Drop the lock entirely; also drop the cache contents — they're
        # lazy and tied to the sender's compute graph, so the receiver
        # will repopulate as needed instead of inheriting potentially
        # device-bound entries.
        return {"cache": OrderedDict(), "max_size": self.max_size}

    def __setstate__(self, state):
        self.cache = state["cache"]
        self.max_size = state["max_size"]
        self.lock = threading.Lock()

    def get(self, key):
        with self.lock:
            if key in self.cache:
                # Move to end (most recently used)
                self.cache.move_to_end(key)
                return self.cache[key]
            return None

    def put(self, key, value):
        with self.lock:
            if key in self.cache:
                # Update existing and move to end
                self.cache[key] = value
                self.cache.move_to_end(key)
            else:
                # Add new entry
                self.cache[key] = value
                if len(self.cache) > self.max_size:
                    # Remove least recently used
                    self.cache.popitem(last=False)

    def clear(self):
        with self.lock:
            self.cache.clear()
    
    def __contains__(self, key):
        with self.lock:
            return key in self.cache


class CliffordMap:

    def __init__(self, n_qubits: int, batch_size: int, n_measurements: int, 
                 device: Union[str, torch.device] = 'cpu',
                 pauli_cache_size: int = 10000,
                 prob_cache_size: int = 128,
                 enable_prob_cache: bool = True):
        with torch.no_grad():
            self.n_qubits = n_qubits
            self.batch_size = batch_size
            self.n_measurements = n_measurements
            self.device = torch.device(device) if isinstance(device, str) else device
            self.N2 = 2 * n_qubits
        
        # Version tracking for cache invalidation
        self.version = 0
        
        # Heisenberg representation (the Clifford map)
        with torch.no_grad():
            self.W = (
                torch.eye(self.N2, dtype=torch.int8, device=self.device)
                .unsqueeze(0)
                .unsqueeze(0)
                .expand(batch_size, n_measurements, -1, -1)
                .contiguous()
                .clone()
            )
            self.heis_phase_vec = torch.zeros(batch_size, n_measurements, self.N2, dtype=torch.int8, device=self.device)

            # NEW: Maintain the inverse of W
            self.W_inv = torch.eye(self.N2, dtype=torch.int8, device=self.device).unsqueeze(0).unsqueeze(0).expand(batch_size, n_measurements, -1, -1).contiguous()

            # Active mask
            self.active = torch.ones(batch_size, n_measurements, dtype=torch.bool, device=self.device)
        
            # Pre-allocate buffers
            self._flat_buffer = None
        
        # Caching with LRU
        self._pauli_cache = LRUCache(pauli_cache_size)
        self._prob_cache = LRUCache(prob_cache_size) if enable_prob_cache else None
        self.enable_prob_cache = enable_prob_cache
        
        # Create device-specific tensor pool
        self._pool = TensorPool(self.device)
        
        # Optimized flattened views for GPU operations
        self.total_tableaus = batch_size * n_measurements
        self._setup_flattened_views()
    
    def _increment_version(self):
        """Increment version when Clifford map changes"""
        self.version += 1
        # Clear probability cache on version change
        if self._prob_cache is not None:
            self._prob_cache.clear()
    
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
        
        # NEW: Flattened view for W_inv
        self._W_inv_flat = self.W_inv.view(self.total_tableaus, self.N2, self.N2)
        
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

        # Work with flattened tensors for efficiency
        mask_flat = mask.view(-1)
        active_indices = mask_flat.nonzero(as_tuple=True)[0]

        if len(active_indices) == 0:
            return

        _apply_H_script(
            self._W_flat,
            self._W_inv_flat,
            self._heis_phase_vec_flat,
            active_indices,
            qubit,
            self.n_qubits,
        )
        
        self._increment_version()
    
    @torch.no_grad()
    def apply_S(self, qubit: int, mask: Optional[torch.Tensor] = None):
        """Vectorized S gate application: X → Y(iXZ), Y → -X, Z → Z"""
        if mask is None:
            mask = self.active
        else:
            mask = self._ensure_device(mask)
        
        if not mask.any():
            return
        
        # Work with flattened tensors
        mask_flat = mask.view(-1)
        active_indices = mask_flat.nonzero(as_tuple=True)[0]

        if len(active_indices) == 0:
            return

        _apply_S_script(
            self._W_flat,
            self._W_inv_flat,
            self._heis_phase_vec_flat,
            active_indices,
            qubit,
            self.n_qubits,
        )
        
        self._increment_version()
    
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

        # Work with flattened tensors
        mask_flat = mask.view(-1)
        active_indices = mask_flat.nonzero(as_tuple=True)[0]

        if len(active_indices) == 0:
            return

        _apply_CNOT_script(
            self._W_flat,
            self._W_inv_flat,
            self._heis_phase_vec_flat,
            active_indices,
            control,
            target,
            self.n_qubits,
        )
        
        self._increment_version()

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

    def _pauli_string_to_symplectic(self, pauli_strings: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Vectorized conversion of Pauli strings to symplectic vectors"""
        n_strings = len(pauli_strings)
        vecs = torch.zeros(n_strings, self.N2, dtype=torch.bool, device=self.device)
        phases = torch.zeros(n_strings, dtype=torch.int8, device=self.device)
        
        # Separate cached and uncached strings
        cached_indices = []
        uncached_indices = []
        uncached_strings = []
        
        for i, ps in enumerate(pauli_strings):
            cached_result = self._pauli_cache.get(ps)
            if cached_result is not None:
                vec, phase = cached_result
                vecs[i] = self._ensure_device(vec)
                phases[i] = phase
                cached_indices.append(i)
            else:
                uncached_indices.append(i)
                uncached_strings.append(ps)
        
        if not uncached_strings:
            return vecs, phases
        
        # Vectorized processing of uncached strings
        n_uncached = len(uncached_strings)
        
        # GPU OPTIMIZATION: Build tensor on CPU first, then transfer in one bulk operation
        # This avoids creating many small tensors on GPU individually
        max_len = max(len(s) for s in uncached_strings) if uncached_strings else 0
        str_tensor_cpu = torch.zeros(n_uncached, max_len, dtype=torch.uint8)
        lengths_cpu = torch.zeros(n_uncached, dtype=torch.int)
        
        for i, s in enumerate(uncached_strings):
            bytes_s = s.encode('ascii')
            str_tensor_cpu[i, :len(bytes_s)] = torch.tensor(list(bytes_s), dtype=torch.uint8)
            lengths_cpu[i] = len(s)
        
        # Single bulk transfer to GPU
        str_tensor = str_tensor_cpu.to(self.device)
        lengths = lengths_cpu.to(self.device)
        
        # ASCII codes
        PLUS = ord('+')
        MINUS = ord('-')
        I_CHAR = ord('i')
        X_CHAR = ord('X')
        Y_CHAR = ord('Y')
        Z_CHAR = ord('Z')
        I_PAULI = ord('I')
        
        # Parse prefixes vectorized
        uncached_phases = torch.zeros(n_uncached, dtype=torch.int8, device=self.device)
        pauli_start_idx = torch.zeros(n_uncached, dtype=torch.int, device=self.device)
        
        # Check for different prefix patterns
        first_char = str_tensor[:, 0]
        second_char = str_tensor[:, 1] if max_len > 1 else torch.zeros_like(first_char)
        
        # +i prefix (phase = 1)
        plus_i_mask = (first_char == PLUS) & (second_char == I_CHAR)
        uncached_phases[plus_i_mask] = 1
        pauli_start_idx[plus_i_mask] = 2
        
        # -i prefix (phase = 3)
        minus_i_mask = (first_char == MINUS) & (second_char == I_CHAR)
        uncached_phases[minus_i_mask] = 3
        pauli_start_idx[minus_i_mask] = 2
        
        # + prefix (phase = 0)
        plus_mask = (first_char == PLUS) & ~plus_i_mask
        uncached_phases[plus_mask] = 0
        pauli_start_idx[plus_mask] = 1
        
        # - prefix (phase = 2)
        minus_mask = (first_char == MINUS) & ~minus_i_mask
        uncached_phases[minus_mask] = 2
        pauli_start_idx[minus_mask] = 1
        
        # No prefix (phase = 0, start at 0)
        no_prefix_mask = ~(plus_i_mask | minus_i_mask | plus_mask | minus_mask)
        pauli_start_idx[no_prefix_mask] = 0
        
        # Extract Pauli part for each string - optimized vectorized version
        uncached_vecs = torch.zeros(n_uncached, self.N2, dtype=torch.bool, device=self.device)
        
        # Vectorized extraction: handle variable start positions efficiently
        # Process strings grouped by start position for better GPU utilization
        max_start = int(pauli_start_idx.max().item()) if n_uncached > 0 else 0
        
        for start_val in range(max_start + 1):
            mask_start = (pauli_start_idx == start_val)
            if not mask_start.any():
                continue
            
            # Extract Pauli parts for strings with this start value
            indices = mask_start.nonzero(as_tuple=True)[0]
            pauli_parts = str_tensor[indices, start_val:start_val + self.n_qubits]  # (n_matching, n_qubits)
            
            # Vectorized character matching
            is_x = pauli_parts == X_CHAR
            is_y = pauli_parts == Y_CHAR
            is_z = pauli_parts == Z_CHAR
            
            # Set symplectic vector bits
            uncached_vecs[indices, :self.n_qubits] = is_x | is_y
            uncached_vecs[indices, self.n_qubits:] = is_z | is_y
            
            # Update phase for Y operators (vectorized)
            y_counts = is_y.sum(dim=1).to(uncached_phases.dtype)  # (n_matching,) - convert to int8
            uncached_phases[indices] = (uncached_phases[indices] + y_counts) % 4
        
        # Store results back and update cache
        for i, (idx, ps) in enumerate(zip(uncached_indices, uncached_strings)):
            vecs[idx] = uncached_vecs[i]
            phases[idx] = uncached_phases[i]
            # Update cache
            self._pauli_cache.put(ps, (uncached_vecs[i].clone(), uncached_phases[i].item()))
        
        return vecs, phases

    def transform_paulis(self, pauli_vecs: torch.Tensor, chunk_size: int = 4096) -> torch.Tensor:
        """Transform Pauli vectors under all Clifford maps using W matrix.

        Computes U P U† for each Pauli P and Clifford map U.

        Args:
            pauli_vecs: Tensor of shape (n_paulis, 2*n) in symplectic form.
            chunk_size: Process tableaus in chunks to save memory.

        Returns:
            Tensor of shape (batch_size, n_measurements, n_paulis, 2*n) containing
            the transformed Pauli vectors.
        """
        n_paulis = pauli_vecs.shape[0]
        n_tableaus = self.total_tableaus

        result = torch.zeros(n_tableaus, n_paulis, self.N2,
                             dtype=torch.bool, device=self.device)

        pauli_vecs_t = pauli_vecs.T  # (2n, n_paulis)

        # Use W^T (forward transformation): P → U P U† is P^T @ W = (W^T @ P)^T in symplectic form
        for start in range(0, n_tableaus, chunk_size):
            end = min(start + chunk_size, n_tableaus)

            # Use W^T for forward transformation (int8) - transpose because we use column vectors
            W_chunk = self._W_flat[start:end].transpose(1, 2)

            # Multiply in GF(2) -> (chunk, 2n, n_paulis)
            p_out = GF2Ops.matmul(W_chunk, pauli_vecs_t, validate=False)

            # Reorder to (chunk, n_paulis, 2n)
            result[start:end] = p_out.transpose(1, 2)

        result = result.view(self.batch_size, self.n_measurements,
                             n_paulis, self.N2)
        return result

    def prob_P_multi(self, pauli_strings: List[str]) -> torch.Tensor:
        """Compute probability of Pauli strings being stabilizers for ALL tableaus"""
        # Check probability cache if enabled
        if self.enable_prob_cache and self._prob_cache is not None:
            cache_key = (self.version, tuple(pauli_strings))
            cached_result = self._prob_cache.get(cache_key)
            if cached_result is not None:
                return cached_result
        
        n_strings = len(pauli_strings)

        # Convert Pauli strings
        p_vecs, _ = self._pauli_string_to_symplectic(pauli_strings)

        # Transform all paulis in parallel
        p_out = self.transform_paulis(p_vecs)

        # Determine measurability (no X component in U^† P U)
        has_x = p_out[..., :self.n_qubits].any(dim=3)
        probs = (~has_x).float()
        
        # Cache result if enabled
        if self.enable_prob_cache and self._prob_cache is not None:
            cache_key = (self.version, tuple(pauli_strings))
            self._prob_cache.put(cache_key, probs.clone())
        
        return probs

    def prob_P_single(self, pauli_string: str) -> torch.Tensor:
        """Compute probability for a single Pauli string across all trajectories."""
        result = self.prob_P_multi([pauli_string])
        return result.squeeze(2)  # Remove the n_strings dimension
    
    def phase_to_complex(self, phase: int) -> complex:
        """Convert phase mod 4 to complex number"""
        return [1, 1j, -1, -1j][phase % 4]
    
    def to_flat_tensors_active_only(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """GPU-optimized flat tensor conversion using only W matrix (2nx2n Clifford tableau)"""
        mat_size = self.N2 * self.N2
        feature_size = mat_size  # Only W matrix, no phase vector
        
        # Get active indices efficiently
        active_indices = self._active_flat.nonzero(as_tuple=True)[0]
        n_active = len(active_indices)
        
        if n_active == 0:
            return (torch.empty((0, feature_size), device=self.device), 
                    torch.empty((0, 2), dtype=torch.long, device=self.device))
        
        # Select active Clifford maps (only W matrix)
        output = self._W_flat[active_indices].view(n_active, mat_size).float()
        
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
            dummy_input = torch.zeros(1, self.N2 * self.N2, device=self.device)  # Only W matrix, no phase vector
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
        self.W[batch_idx, meas_idx] = torch.eye(self.N2, dtype=torch.int8, device=self.device)
        self.W_inv[batch_idx, meas_idx] = torch.eye(self.N2, dtype=torch.int8, device=self.device)
        self.heis_phase_vec[batch_idx, meas_idx] = torch.zeros(self.N2, dtype=torch.int8, device=self.device)
        self.active[batch_idx, meas_idx] = True
        self._increment_version()

    def reset(self):
        """Reset all measurements to identity"""
        # Reset all W matrices [batch_size, n_measurements, N2, N2]
        self.W[:] = torch.eye(self.N2, dtype=torch.int8, device=self.device)
        # Reset all W_inv matrices
        self.W_inv[:] = torch.eye(self.N2, dtype=torch.int8, device=self.device)
        # Reset all phase vectors [batch_size, n_measurements, N2]
        self.heis_phase_vec.zero_()
        # Reset active mask [batch_size, n_measurements]
        self.active.fill_(True)
        self._increment_version()
    
    def get_clifford_map(self, batch_idx: int, meas_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a specific Clifford map and phase vector"""
        return self.W[batch_idx, meas_idx].clone(), self.heis_phase_vec[batch_idx, meas_idx].clone()
    
    def to_flat_tensors(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Vectorized conversion for efficient batch processing using only W matrix (2nx2n Clifford tableau)."""
        mat_size = self.N2 * self.N2
        
        # Reshape W matrices only (no phase vector)
        output = self.W.view(self.batch_size, self.n_measurements, mat_size).float()
        
        return output, self.active.clone()

    def _get_active_indices_flat(self, mask: torch.Tensor) -> Tuple[torch.Tensor, int]:
        """Convert 2D mask to flat indices for efficient GPU operations"""
        batch_indices, meas_indices = torch.where(mask)
        n_active = len(batch_indices)
        flat_indices = batch_indices * self.n_measurements + meas_indices
        return flat_indices, n_active

    def validate_inverse(self, tolerance: float = 1e-6) -> bool:
        """
        Validate that W_inv is the correct left inverse of W for all active tableaus
        by checking a random sample.

        Verifies: W_inv @ W = I

        Note: W_inv is maintained as a left inverse through row operations.
        For a true inverse, both W @ W_inv = I and W_inv @ W = I hold, but
        our incremental updates guarantee the left-side identity.

        Returns:
            True if all sampled tableaus have valid inverses, False otherwise
        """
        active_indices = self._active_flat.nonzero(as_tuple=True)[0]

        if len(active_indices) == 0:
            return True

        # Check a sample of tableaus
        sample_size = min(10, len(active_indices))
        sample_indices = active_indices[torch.randperm(len(active_indices))[:sample_size]]

        for idx in sample_indices:
            W = self._W_flat[idx].to(torch.bool)
            W_inv = self._W_inv_flat[idx].to(torch.bool)

            # Check both W_inv @ W == I (left inverse) and W @ W_inv == I (right inverse)
            expected_identity = torch.eye(self.N2, dtype=torch.bool, device=self.device)

            lhs = GF2Ops.matmul(W_inv.unsqueeze(0),
                               W.unsqueeze(0),
                               validate=False).squeeze(0)
            rhs = GF2Ops.matmul(W.unsqueeze(0),
                               W_inv.unsqueeze(0),
                               validate=False).squeeze(0)

            if not torch.equal(lhs, expected_identity) or not torch.equal(rhs, expected_identity):
                print(f"Inverse validation failed for tableau {idx}")
                return False

        return True


if __name__ == "__main__":
    # Example usage with the new class
    clifford_map = CliffordMap(n_qubits=3, batch_size=100, 
                                      n_measurements=10, device='cpu',
                                      pauli_cache_size=10000,
                                      prob_cache_size=128,
                                      enable_prob_cache=True)
    
    # Apply some gates
    clifford_map.apply_H(0)
    clifford_map.apply_CNOT(0, 1)
    clifford_map.apply_S(1)
    
    # Validate that W_inv is correct
    print("W_inv validation:", clifford_map.validate_inverse())
    
    # Compute probabilities
    probs = clifford_map.prob_P_multi(['XII', 'ZZI', 'YYI'])
    print("Probabilities shape:", probs.shape)
    
    # Get a specific Clifford map
    W, phase = clifford_map.get_clifford_map(0, 0)
    print("W shape:", W.shape)
    
    # Test caching
    print("Version:", clifford_map.version)
    
    # Compute again - should use cache
    probs2 = clifford_map.prob_P_multi(['XII', 'ZZI', 'YYI'])
    print("Same result from cache:", torch.equal(probs, probs2))
    
    # Apply another gate - should invalidate cache
    clifford_map.apply_H(2)
    print("Version after gate:", clifford_map.version)
    
    # Test vectorized Pauli conversion
    test_strings = ['+iXYZ', '-XZI', 'YYY', '+ZXY', '-iIXZ']
    vecs, phases = clifford_map._pauli_string_to_symplectic(test_strings)
    print("Vectorized conversion done, shapes:", vecs.shape, phases.shape)
