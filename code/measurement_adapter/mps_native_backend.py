"""MPS-native sampling backend for ``EnergyEstimator``.

This module hosts:

* ``MPSOps`` — Protocol describing the low-level MPS operations
  ``MPSNativeBackend`` consumes: one-site and two-site gate application,
  truncation, and perfect sampling.
* ``DenseReferenceMPSOps`` — torch-only CPU reference implementation used by
  unit tests. Slow but exact; mirrors the dense state-vector path's gate
  semantics so parity tests on small systems can compare bit-for-bit.
* ``MPSNativeBackend`` — public surface consumed by
  ``EnergyEstimator``. Owns the ``action_map`` decoding, gate-tensor table,
  per-circuit MPS cloning, forward-order gate application with
  ``batch_lengths`` / terminal-action handling, and ``sample_outcomes`` that
  returns ``(B, C, M)`` integer outcomes matching the dense path's
  big-endian bit-order convention.

cuTensorNet integration lives in ``mps_native_cutensornet.py``; the Protocol
seam keeps the high-level circuit/sampling logic shared across dense-reference
and cuTensorNet implementations.

This lives under ``code/measurement_adapter/`` as a shim;
it is reusable measurement-core machinery that ultimately belongs in
``clifford-tableau`` once the API is stable.
"""
from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple

import numpy as np
import torch


logger = logging.getLogger(__name__)


# Backend dtype: complex128 to match the cached MPS payload
# (`helper.ground_state_mps` returns complex128 tensors). Gate tensors are
# promoted to complex128 here even though the dense path uses complex64
# (energy_estimator.py:362-369). Mixed-dtype ``apply_gate`` calls are
# disallowed, so the asymmetry is resolved at this boundary.
MPS_DTYPE: torch.dtype = torch.complex128


# ---------------------------------------------------------------------------
# MPSOps Protocol
# ---------------------------------------------------------------------------


class MPSOps(Protocol):
    """Low-level MPS operations consumed by ``MPSNativeBackend``.

    Implementations:

    * ``DenseReferenceMPSOps`` — torch-only CPU reference; used by parity
      tests and CPU-only smoke tests.
    * ``CuTensorNetMPSOps`` — production GPU implementation built on
      ``cuquantum.tensornet`` / ``cuquantum.cutensornet`` primitives.

    All methods take and return ``list[torch.Tensor]`` MPS representations
    where each tensor has shape ``(chi_left, 2, chi_right)`` and dtype
    ``MPS_DTYPE`` (complex128). The leftmost ``chi_left`` and rightmost
    ``chi_right`` are 1. Implementations may mutate the input list in place
    or return a new one; ``MPSNativeBackend`` clones before calling.
    """

    def apply_one_site_gate(
        self,
        mps: List[torch.Tensor],
        site: int,
        gate: torch.Tensor,
    ) -> List[torch.Tensor]:
        """Apply a 2x2 single-qubit gate at ``site`` (in place is fine)."""

    def apply_two_site_gate(
        self,
        mps: List[torch.Tensor],
        site: int,
        gate: torch.Tensor,
        *,
        max_bond_dim: Optional[int] = None,
        truncation_tol: float = 1e-10,
    ) -> List[torch.Tensor]:
        """Apply a 4x4 two-site gate at sites ``(site, site+1)``.

        The gate's index ordering is
        ``gate[i1', i2', i1, i2]`` (output, output, input, input) with each
        index in ``{0, 1}``. After contraction the merged two-site tensor is
        SVD-split with ``max_bond_dim`` / ``truncation_tol``.
        """

    def sample(
        self,
        mps: List[torch.Tensor],
        M: int,
        *,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """Draw ``M`` samples from ``|psi|^2`` and return an integer tensor.

        Returns a 1-D ``torch.long`` tensor of length ``M``. Bit ordering:
        qubit ``q`` contributes bit ``(n_qubits - 1 - q)`` to the integer
        (big-endian, matching the dense path's convention at
        ``energy_estimator.py:534-536``).
        """


# ---------------------------------------------------------------------------
# Dense reference implementation
# ---------------------------------------------------------------------------


def _clone_mps(mps: Sequence[torch.Tensor]) -> List[torch.Tensor]:
    return [t.clone() for t in mps]


class DenseReferenceMPSOps:
    """Torch-only CPU MPS ops for parity testing and CPU smoke tests.

    Not optimized — clarity over throughput. The production GPU
    implementation is ``CuTensorNetMPSOps``.

    Conventions:

    * Site tensor shape ``(chi_L, 2, chi_R)``; physical leg is index 1.
    * SVD truncation drops singular values below ``truncation_tol``; if
      ``max_bond_dim`` is provided, the kept rank is also capped.
    * Sampling uses perfect left-to-right marginal sampling, which is
      exact for ``|psi|^2`` and avoids forming the dense state.
    """

    def apply_one_site_gate(
        self,
        mps: List[torch.Tensor],
        site: int,
        gate: torch.Tensor,
    ) -> List[torch.Tensor]:
        # Contract gate (2x2) with site tensor over the physical leg.
        # site tensor: (chi_L, 2, chi_R); gate: (2, 2) with index order
        # ``gate[i_out, i_in]``. Result: (chi_L, 2, chi_R). Promote the gate
        # tensor to the MPS dtype + device explicitly so a caller that built
        # the gate on a different device (e.g. via a test stub) does not hit
        # a silent einsum mismatch.
        mps[site] = torch.einsum(
            "ij,ljk->lik",
            gate.to(dtype=mps[site].dtype, device=mps[site].device),
            mps[site],
        )
        return mps

    def apply_two_site_gate(
        self,
        mps: List[torch.Tensor],
        site: int,
        gate: torch.Tensor,
        *,
        max_bond_dim: Optional[int] = None,
        truncation_tol: float = 1e-10,
    ) -> List[torch.Tensor]:
        L = mps[site]      # (chi_L, 2, chi_M)
        R = mps[site + 1]  # (chi_M, 2, chi_R)
        chi_L, _, chi_M = L.shape
        _, _, chi_R = R.shape

        # Merge into a two-site tensor T[chi_L, p1, p2, chi_R].
        T = torch.einsum("aib,bjc->aijc", L, R)

        # Apply gate. gate index order: gate[p1', p2', p1, p2] -> contract
        # over (p1, p2) of T. Promote gate to T's dtype + device so a
        # caller-built gate on the wrong device fails fast at .to() rather
        # than silently misbehaving inside einsum.
        gate_c = gate.to(dtype=T.dtype, device=T.device).reshape(2, 2, 2, 2)
        T = torch.einsum("ijkl,akld->aijd", gate_c, T)

        # SVD-split. Reshape to a matrix M[(chi_L, p1), (p2, chi_R)].
        Mmat = T.reshape(chi_L * 2, 2 * chi_R)
        # Use full_matrices=False for thin SVD.
        U, S, Vh = torch.linalg.svd(Mmat, full_matrices=False)

        # Truncate. Log discarded weight at DEBUG so production runs stay
        # quiet but parity sweeps can inspect bond-dim growth.
        keep = (S > truncation_tol).sum().item()
        if max_bond_dim is not None:
            keep = min(keep, int(max_bond_dim))
        keep = max(keep, 1)
        if keep < S.shape[0]:
            # Discarded-singular-value squared sum; small => safe truncation.
            discarded_weight = float(
                (S[keep:].to(torch.float64) ** 2).sum().item()
            )
            if discarded_weight > truncation_tol * 100:
                logger.debug(
                    "MPS two-site gate truncation at site %d: kept %d / %d, "
                    "discarded weight %.3e (tol=%.3e, max_bond_dim=%s)",
                    site, keep, S.shape[0], discarded_weight,
                    truncation_tol, str(max_bond_dim),
                )
        U = U[:, :keep]
        S = S[:keep]
        Vh = Vh[:keep, :]

        # Reabsorb singular values into the right tensor (right-canonical-ish
        # for the next site).
        new_L = U.reshape(chi_L, 2, keep)
        new_R = (torch.diag(S).to(Vh.dtype) @ Vh).reshape(keep, 2, chi_R)

        mps[site] = new_L
        mps[site + 1] = new_R
        return mps

    def sample(
        self,
        mps: List[torch.Tensor],
        M: int,
        *,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        # Perfect left-to-right marginal sampling.
        #
        # Define right environments R_k[a, A] = sum over indices to the
        # right of site k:
        #   R_k[a, A] = sum_{i, b, B} T_k[a, i, b] * conj(T_k[A, i, B])
        #                              * R_{k+1}[b, B]
        # with R_{n_sites}[b, B] = delta(b, B) (right boundary).
        n_sites = len(mps)
        device = mps[0].device
        dtype = mps[0].dtype

        right_envs: List[torch.Tensor] = [None] * (n_sites + 1)  # type: ignore[list-item]
        last_chi = mps[-1].shape[2]
        right_envs[n_sites] = torch.eye(last_chi, dtype=dtype, device=device)
        for k in range(n_sites - 1, -1, -1):
            T = mps[k]  # (chi_L, 2, chi_R)
            right_envs[k] = torch.einsum(
                "aib,AiB,bB->aA", T, T.conj(), right_envs[k + 1]
            )

        # ``left_states[m, a]`` is the running left vector after conditioning
        # bits 0..k-1 for sample ``m``. Start at site 0 with the boundary
        # vector e_0 of length ``chi_left_of_site0 == 1``.
        outcomes = torch.zeros(M, dtype=torch.long, device=device)
        chi0 = mps[0].shape[0]
        left_states = torch.zeros(M, chi0, dtype=dtype, device=device)
        left_states[:, 0] = 1.0

        for k in range(n_sites):
            T = mps[k]  # (chi_L, 2, chi_R)
            R_next = right_envs[k + 1]  # (chi_R, chi_R)

            # psi_i[m, b] = sum_a L[m, a] * T[a, i, b] for i in {0, 1}.
            psi0 = torch.einsum("ma,ab->mb", left_states, T[:, 0, :])
            psi1 = torch.einsum("ma,ab->mb", left_states, T[:, 1, :])
            # weight_i[m] = sum_{b, B} psi_i[m, b] * conj(psi_i[m, B])
            #                          * R_next[b, B]
            w0 = torch.einsum("mb,mB,bB->m", psi0, psi0.conj(), R_next).real
            w1 = torch.einsum("mb,mB,bB->m", psi1, psi1.conj(), R_next).real
            # Fail fast on NaN / Inf rather than emitting garbage outcomes.
            # The dense path can fall over on a non-normalized state in
            # similar ways; we surface it here with a precise message.
            if not torch.isfinite(w0).all() or not torch.isfinite(w1).all():
                raise RuntimeError(
                    "MPS sampling produced non-finite marginal probabilities "
                    f"at site {k}; likely cause: non-normalized MPS or NaN "
                    "introduced by an upstream gate / truncation."
                )
            w0 = w0.clamp_min(0.0)
            w1 = w1.clamp_min(0.0)
            total = (w0 + w1).clamp_min(1e-300)
            prob_one = (w1 / total).to(torch.float64)

            rand = torch.rand(
                M, generator=generator, device=device, dtype=torch.float64
            )
            chosen = (rand < prob_one).to(torch.long)

            # Bit ordering: qubit ``k`` -> bit ``(n_sites - 1 - k)``.
            shift = n_sites - 1 - k
            outcomes |= chosen << shift

            # Move the running left vector to the chosen branch and
            # renormalize so the running magnitude stays O(1).
            chosen_c = chosen.unsqueeze(1).to(dtype)
            psi_chosen = (1 - chosen_c) * psi0 + chosen_c * psi1
            norms = torch.linalg.vector_norm(psi_chosen, dim=1, keepdim=True)
            psi_chosen = psi_chosen / norms.clamp_min(1e-300).to(dtype)
            left_states = psi_chosen

        return outcomes


# ---------------------------------------------------------------------------
# MPSNativeBackend
# ---------------------------------------------------------------------------


def _build_gate_table(device: torch.device) -> Dict[str, torch.Tensor]:
    """Build the one-site Clifford gates (complex128) matching the dense path.

    Mirrors ``EnergyEstimator._setup_torch_quantum_gates`` but at MPS-native
    dtype (complex128). Two-site CNOT is constructed separately because it
    is applied via ``apply_two_site_gate``.
    """
    H = torch.tensor(
        [[1, 1], [1, -1]], dtype=MPS_DTYPE, device=device
    ) / math.sqrt(2)
    S = torch.tensor([[1, 0], [0, 1j]], dtype=MPS_DTYPE, device=device)
    return {
        "H": H,
        "S": S,
        "HS": H @ S,
        "SH": S @ H,
        "HSH": H @ S @ H,
    }


def _build_cnot_gate(device: torch.device) -> torch.Tensor:
    """CNOT as a (2, 2, 2, 2) tensor with index order [c', t', c, t].

    Big-endian: when applied at adjacent sites (control_site, target_site)
    with control_site < target_site, ``apply_two_site_gate`` consumes
    physical indices (p1, p2) = (control, target).
    """
    g = torch.zeros(2, 2, 2, 2, dtype=MPS_DTYPE, device=device)
    # |00> -> |00>, |01> -> |01>, |10> -> |11>, |11> -> |10>.
    g[0, 0, 0, 0] = 1
    g[0, 1, 0, 1] = 1
    g[1, 1, 1, 0] = 1
    g[1, 0, 1, 1] = 1
    return g


def _cnot_target_first_gate(device: torch.device) -> torch.Tensor:
    """CNOT with index order [t', c', t, c] (target is the left physical leg).

    Used when the target qubit sits at a lower MPS site than the control.
    """
    g = torch.zeros(2, 2, 2, 2, dtype=MPS_DTYPE, device=device)
    # |t,c=0,0> -> |0,0>, |t,c=1,0> -> |1,0>,
    # |t,c=0,1> -> |1,1>, |t,c=1,1> -> |0,1>.
    g[0, 0, 0, 0] = 1
    g[1, 0, 1, 0] = 1
    g[1, 1, 0, 1] = 1
    g[0, 1, 1, 1] = 1
    return g


def _swap_gate(device: torch.device) -> torch.Tensor:
    """SWAP gate as a (2, 2, 2, 2) tensor with index order [a', b', a, b]."""
    g = torch.zeros(2, 2, 2, 2, dtype=MPS_DTYPE, device=device)
    for a in (0, 1):
        for b in (0, 1):
            g[b, a, a, b] = 1
    return g


class MPSNativeBackend:
    """MPS-native sampling backend used by ``EnergyEstimator`` in
    ``measurement_backend="mps_native"`` mode.

    The backend owns:

    * ``action_map`` — maps action indices to gate tuples (matches the
      ``EnergyEstimator``'s table).
    * ``terminal_action`` — sentinel value marking end of a circuit.
    * Gate tensor table at complex128 (one-site + CNOT).
    * ``ops`` — an ``MPSOps`` implementation; defaults to
      ``DenseReferenceMPSOps()``. Pass a cuTensorNet implementation here
      to enable the production GPU path.

    Public surface:

    * ``sample_outcomes(mps_ground_state, batch_actions, batch_lengths, M)``
      returns a ``(batch_size, n_circuits, M)`` ``torch.long`` integer
      tensor on the backend's device. Bit ordering matches the dense
      sampler exactly.
    """

    def __init__(
        self,
        n_qubits: int,
        action_map: Dict[int, Tuple],
        terminal_action: int,
        device: torch.device,
        *,
        ops: Optional[MPSOps] = None,
        max_bond_dim: Optional[int] = None,
        truncation_tol: float = 1e-10,
    ) -> None:
        self.n_qubits = int(n_qubits)
        self.action_map = action_map
        self.terminal_action = int(terminal_action)
        self.device = device
        self.ops: MPSOps = ops if ops is not None else DenseReferenceMPSOps()
        self.max_bond_dim = max_bond_dim
        self.truncation_tol = float(truncation_tol)

        self._one_site_gates = _build_gate_table(device)
        self._cnot_control_first = _build_cnot_gate(device)
        self._cnot_target_first = _cnot_target_first_gate(device)
        self._swap = _swap_gate(device)

    # ------------------------------------------------------------------
    # Gate dispatch
    # ------------------------------------------------------------------

    def _apply_gate(
        self,
        mps: List[torch.Tensor],
        gate_tuple: Tuple,
    ) -> List[torch.Tensor]:
        name = gate_tuple[0]
        if name == "CNOT":
            control, target = int(gate_tuple[1]), int(gate_tuple[2])
            return self._apply_cnot(mps, control, target)
        # Single-qubit gates.
        qubit = int(gate_tuple[1])
        gate = self._one_site_gates[name]
        return self.ops.apply_one_site_gate(mps, qubit, gate)

    def _apply_cnot(
        self,
        mps: List[torch.Tensor],
        control: int,
        target: int,
    ) -> List[torch.Tensor]:
        # Two-site gates require adjacent sites. SWAP-network the control
        # toward the target until they are neighbors, apply CNOT, then
        # un-SWAP. Cost is O(|control - target|) two-site applications;
        # acceptable for the dense-reference path. The cuTensorNet
        # implementation can use long-range MPO contraction instead.
        if control == target:
            raise ValueError(
                "CNOT control and target must differ; got both at site "
                f"{control}"
            )
        if abs(control - target) == 1:
            if control < target:
                return self.ops.apply_two_site_gate(
                    mps,
                    control,
                    self._cnot_control_first,
                    max_bond_dim=self.max_bond_dim,
                    truncation_tol=self.truncation_tol,
                )
            # control > target: target is the left physical leg.
            return self.ops.apply_two_site_gate(
                mps,
                target,
                self._cnot_target_first,
                max_bond_dim=self.max_bond_dim,
                truncation_tol=self.truncation_tol,
            )

        # Non-adjacent: SWAP control toward target, apply, un-SWAP.
        if control < target:
            # Move control rightward until it is at (target - 1).
            cur = control
            while cur + 1 < target:
                mps = self.ops.apply_two_site_gate(
                    mps,
                    cur,
                    self._swap,
                    max_bond_dim=self.max_bond_dim,
                    truncation_tol=self.truncation_tol,
                )
                cur += 1
            # Apply CNOT at (cur, target=cur+1) with control at the left.
            mps = self.ops.apply_two_site_gate(
                mps,
                cur,
                self._cnot_control_first,
                max_bond_dim=self.max_bond_dim,
                truncation_tol=self.truncation_tol,
            )
            # Un-SWAP back to ``control``.
            while cur > control:
                mps = self.ops.apply_two_site_gate(
                    mps,
                    cur - 1,
                    self._swap,
                    max_bond_dim=self.max_bond_dim,
                    truncation_tol=self.truncation_tol,
                )
                cur -= 1
            return mps

        # control > target: mirror.
        cur = control
        while cur - 1 > target:
            mps = self.ops.apply_two_site_gate(
                mps,
                cur - 1,
                self._swap,
                max_bond_dim=self.max_bond_dim,
                truncation_tol=self.truncation_tol,
            )
            cur -= 1
        # cur is at target+1; apply CNOT with target at the left physical leg.
        mps = self.ops.apply_two_site_gate(
            mps,
            target,
            self._cnot_target_first,
            max_bond_dim=self.max_bond_dim,
            truncation_tol=self.truncation_tol,
        )
        while cur < control:
            mps = self.ops.apply_two_site_gate(
                mps,
                cur,
                self._swap,
                max_bond_dim=self.max_bond_dim,
                truncation_tol=self.truncation_tol,
            )
            cur += 1
        return mps

    # ------------------------------------------------------------------
    # Public surface
    # ------------------------------------------------------------------

    def _prepare_mps(
        self,
        mps_ground_state: Sequence[Any],
        device: torch.device,
        dtype: torch.dtype = MPS_DTYPE,
    ) -> List[torch.Tensor]:
        """Convert a list of numpy / torch tensors to a fresh torch MPS on
        ``device`` with dtype ``MPS_DTYPE``. Validates:

        * ``len(mps) == self.n_qubits`` (catches stale or wrong-system caches).
        * Site shape ``(chi_L, 2, chi_R)`` (3-D, physical dim 2).
        * Boundary bonds: ``chi_left_of_first == 1``, ``chi_right_of_last == 1``.
        * Neighbor bond consistency.

        Also logs the max bond dimension at INFO so MPS-native runs surface
        the cache's bond-dim profile in the standard training log.
        """
        if len(mps_ground_state) != self.n_qubits:
            raise ValueError(
                f"MPS site count {len(mps_ground_state)} does not match "
                f"n_qubits={self.n_qubits}; cache may belong to a different "
                "system."
            )
        out: List[torch.Tensor] = []
        for i, t in enumerate(mps_ground_state):
            if isinstance(t, np.ndarray):
                t = torch.tensor(t, dtype=dtype, device=device)
            else:
                t = t.to(dtype=dtype, device=device).clone()
            if t.dim() != 3 or t.shape[1] != 2:
                raise ValueError(
                    f"MPS site {i} has invalid shape {tuple(t.shape)}; "
                    f"expected (chi_L, 2, chi_R)"
                )
            out.append(t)
        if out[0].shape[0] != 1:
            raise ValueError(
                f"MPS site 0 left bond must be 1; got {out[0].shape[0]}"
            )
        if out[-1].shape[2] != 1:
            raise ValueError(
                f"MPS final site right bond must be 1; got {out[-1].shape[2]}"
            )
        for i in range(len(out) - 1):
            if out[i].shape[2] != out[i + 1].shape[0]:
                raise ValueError(
                    f"MPS bond mismatch between sites {i} and {i+1}: "
                    f"{out[i].shape[2]} vs {out[i+1].shape[0]}"
                )
        max_chi = max(t.shape[0] for t in out)
        max_chi = max(max_chi, max(t.shape[2] for t in out))
        logger.info(
            "MPSNativeBackend: loaded %d-site MPS, max bond dim = %d "
            "(dtype=%s, device=%s)",
            len(out), max_chi, dtype, device,
        )
        return out

    @torch.no_grad()
    def sample_outcomes(
        self,
        mps_ground_state: Sequence[Any],
        batch_actions: torch.Tensor,
        batch_lengths: torch.Tensor,
        M: int,
        *,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """Apply per-(b, c) Clifford circuits to the ground-state MPS and
        sample ``M`` outcomes per circuit.

        Args:
            mps_ground_state: list of numpy / torch tensors representing the
                cached ground-state MPS (shape ``(chi_L, 2, chi_R)``,
                complex128, typically on CPU).
            batch_actions: ``(B, C, max_length)`` int tensor of action
                indices.
            batch_lengths: ``(B, C)`` int tensor of per-circuit lengths.
            M: number of i.i.d. outcomes per circuit.
            generator: optional torch ``Generator`` for reproducible
                sampling. Lives on ``self.device``.

        Returns:
            ``(B, C, M)`` ``torch.long`` tensor of integer outcomes on
            ``self.device``. Bit ordering matches the dense sampler:
            qubit ``q`` is bit ``(n_qubits - 1 - q)``.
        """
        if M <= 0:
            raise ValueError(f"M must be positive; got {M}")
        if batch_actions.dim() != 3:
            raise ValueError(
                f"batch_actions must be 3-D (B, C, max_length); got shape "
                f"{tuple(batch_actions.shape)}"
            )
        if batch_lengths.shape != batch_actions.shape[:2]:
            raise ValueError(
                f"batch_lengths shape {tuple(batch_lengths.shape)} does not "
                f"match batch_actions shape {tuple(batch_actions.shape)}"
            )

        B, C, max_length = batch_actions.shape
        device = self.device
        batch_actions = batch_actions.to(device)
        batch_lengths = batch_lengths.to(device)

        # Compute per-(b, c) effective end: min(batch_lengths, terminal_pos).
        # Mirrors energy_estimator.py:587-597 exactly so the MPS path stops
        # at the same step as the dense ``_apply_circuits_to_states``.
        terminal_mask = batch_actions == self.terminal_action
        positions = torch.arange(max_length, device=device).view(1, 1, -1)
        terminal_positions = torch.where(
            terminal_mask, positions, torch.tensor(max_length, device=device)
        ).min(dim=2)[0]
        effective_end = torch.minimum(batch_lengths, terminal_positions)

        # Move actions and lengths to CPU for the Python-driven dispatch
        # loop; the per-circuit MPS itself stays on ``device``.
        actions_cpu = batch_actions.cpu().tolist()
        end_cpu = effective_end.cpu().tolist()

        # Pre-load the ground-state MPS once; clone per (b, c).
        base_mps = self._prepare_mps(mps_ground_state, device)

        outcomes = torch.zeros(B, C, M, dtype=torch.long, device=device)
        for b in range(B):
            for c in range(C):
                mps = _clone_mps(base_mps)
                stop = int(end_cpu[b][c])
                row = actions_cpu[b][c]
                for step in range(stop):
                    action_idx = int(row[step])
                    gate_tuple = self.action_map.get(action_idx)
                    if gate_tuple is None or gate_tuple[0] == "terminal":
                        # ``effective_end`` already excludes terminals, but
                        # be defensive: skip unknown / terminal actions.
                        continue
                    mps = self._apply_gate(mps, gate_tuple)
                outcomes[b, c] = self.ops.sample(mps, M, generator=generator)
        return outcomes

    @classmethod
    def is_available(cls, device: torch.device) -> bool:
        """Return True iff cuTensorNet is importable and ``device`` is CUDA.

        This mirrors the resolver's gating policy and is exposed for callers
        that want to query availability without triggering the resolver
        error path.
        """
        if device.type != "cuda":
            return False
        try:
            from . import backends
            backends._import_cutensornet()
        except ImportError:
            return False
        return True
