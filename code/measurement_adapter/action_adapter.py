"""ActionAdapter: lower FlowMeas action tuples into clifford-tableau primitive gates.

FlowMeas's `action_map: Dict[int, Tuple]` uses string-keyed tuples like
("H", q), ("S", q), ("HS", q), ("SH", q), ("HSH", q), ("CNOT", c, t), and
("terminal",). The clifford-tableau core's BatchedCliffordSim accepts only
the primitive Clifford set {H, S, S_DAG, CX, CZ, SWAP} keyed by `GateType`.

This adapter pre-bakes a lookup table that maps each FlowMeas action id
to its primitive substep sequence. Composites HS/SH/HSH lower to 2-3
primitive substeps. Terminal and missing actions lower to zero substeps.

The translation is batched: given an array of per-circuit action ids of
shape (N,), `translate_step` returns a list of (gate_codes, q1, q2) cupy
arrays of shape (N,) — one tuple per primitive substep. Circuits whose
action has fewer primitives than the current substep get gate_code = -1,
which BatchedCliffordSim treats as a no-op.
"""
from __future__ import annotations

from typing import Dict, Tuple, List, Optional

import numpy as np

# clifford-tableau is an optional GPU-only dependency. Importing this module
# should not fail when CT is absent (e.g. on CPU-only CI). Resolution is
# deferred to the first ActionAdapter instantiation, which raises an
# ImportError with a pointer to requirements-measurement-adapter.txt.
try:
    from clifford_tableau.interfaces import GateType as _GateType  # noqa: F401
    _HAS_CT = True
except ImportError:
    _GateType = None
    _HAS_CT = False


_CT_INSTALL_HINT = (
    "clifford_tableau is not installed. Install per "
    "requirements-measurement-adapter.txt: pip install -e../clifford-tableau"
)


def _lowering_table() -> Dict[str, List[int]]:
    """Build the composite-gate lowering table from CT's GateType enum.

    Deferred so module import doesn't require ``clifford_tableau``.
    """
    if not _HAS_CT:
        raise ImportError(_CT_INSTALL_HINT)
    H = int(_GateType.H)
    S = int(_GateType.S)
    return {
        "H":   [H],
        "S":   [S],
        "HS":  [S, H],          # FlowMeas apply_HS = apply_S; apply_H
        "SH":  [H, S],          # FlowMeas apply_SH = apply_H; apply_S
        "HSH": [H, S, H],       # FlowMeas apply_HSH = apply_H; apply_S; apply_H
    }


class ActionAdapter:
    """Lower FlowMeas action ids into primitive (gate_code, q1, q2) substeps.

    Parameters
    ----------
    action_map:
        FlowMeas action map. Keys are int action ids; values are tuples
        like ("H", q), ("S", q), ("HS", q), ("SH", q), ("HSH", q),
        ("CNOT", c, t), or ("terminal",). Unknown gate names raise.
    n_qubits:
        Used only for sanity checks on qubit indices.
    """

    def __init__(
        self,
        action_map: Dict[int, Tuple],
        n_qubits: int,
        device: Optional["torch.device"] = None,
        cuda_index: Optional[int] = None,
        validate_action_ids: bool = True,
    ):
        """Parameters
        ----------
        action_map:
            FlowMeas action map.
        n_qubits:
            Number of qubits, for qubit-index range checking.
        device:
            Torch device that incoming action tensors will live on. Used to
            steer the CPU/GPU `.to()` in `translate_step` so we never move
            tensors to the wrong CUDA index in a multi-GPU setting. If None,
            translate_step accepts a tensor on any device but won't migrate.
        cuda_index:
            Explicit CUDA index for the cupy LUT and gather kernel. If None,
            falls back to `device.index` (or the current cupy device if both
            are unset).
        validate_action_ids:
            When True (default), ``translate_step`` performs a per-call host
            sync to verify every action id is in ``[0, max_id]``. The sync
            costs ~1 µs per layer and catches caller bugs like replay-buffer
            or checkpoint mismatches where a previously-stored action id is
            no longer valid under the current action_map size.

            Without the sync, CuPy's fancy indexing on out-of-range integers
            is undefined (per CuPy docs, fancy indexing does NOT check
            out-of-bounds indices for performance), so a stale id would
            silently execute the wrong gate sequence and corrupt
            trajectories instead of raising. Set ``False`` only in hot
            paths where the caller has *upstream* guaranteed that every
            id is in range (e.g. by constructing the id stream from the
            same ``action_map``'s keys within the same training run and
            explicitly forbidding replay from incompatible checkpoints).
        """
        if not _HAS_CT:
            raise ImportError(_CT_INSTALL_HINT)
        if not action_map:
            raise ValueError("action_map is empty")
        self.n_qubits = int(n_qubits)
        self._device = device  # torch device for incoming action tensors
        self._cuda_index = cuda_index
        self._validate_action_ids = bool(validate_action_ids)
        self._lowering = _lowering_table()
        self._cx_code = int(_GateType.CX)
        # Validate keys are non-negative integers before sizing the LUT.
        # Negative ids would silently index from the end of the sequences
        # list (Python negative indexing) and corrupt the LUT.
        for aid in action_map.keys():
            if not isinstance(aid, (int, np.integer)):
                raise TypeError(
                    f"action_map keys must be ints; got {type(aid).__name__}"
                )
            if int(aid) < 0:
                raise ValueError(
                    f"action_map keys must be non-negative; got {int(aid)}"
                )
        max_id = max(int(k) for k in action_map.keys())

        sequences: List[List[Tuple[int, int, int]]] = [[] for _ in range(max_id + 1)]
        for aid, tup in action_map.items():
            sequences[int(aid)] = self._lower_one(tup)
        self._max_depth = max((len(seq) for seq in sequences), default=1) or 1

        # Pre-baked LUT: (max_id+1, max_depth, 3) int32.
        # Triplet is (gate_code, q1, q2). -1 means no-op at that substep.
        lut = np.full((max_id + 1, self._max_depth, 3), -1, dtype=np.int32)
        for aid, seq in enumerate(sequences):
            for s, (g, q1, q2) in enumerate(seq):
                lut[aid, s, 0] = g
                lut[aid, s, 1] = q1
                lut[aid, s, 2] = q2
        self._lut_np = lut  # CPU master copy
        self._lut_cp = None  # lazily moved to GPU on first use, pinned to _cuda_index

    @property
    def max_depth(self) -> int:
        """Maximum number of primitive substeps for any single action id."""
        return self._max_depth

    def _lower_one(self, tup) -> List[Tuple[int, int, int]]:
        if tup is None or len(tup) == 0 or tup[0] == "terminal":
            return []
        gname = tup[0]
        qs = tup[1:]
        if gname in self._lowering:
            q = int(qs[0])
            if not (0 <= q < self.n_qubits):
                raise ValueError(f"qubit index {q} out of range for n_qubits={self.n_qubits}")
            return [(gcode, q, -1) for gcode in self._lowering[gname]]
        if gname == "CNOT":
            if len(qs) < 2:
                raise ValueError(f"CNOT requires (control, target); got {tup}")
            c, t = int(qs[0]), int(qs[1])
            if c == t:
                raise ValueError(f"CNOT control == target ({c})")
            for q in (c, t):
                if not (0 <= q < self.n_qubits):
                    raise ValueError(f"qubit index {q} out of range for n_qubits={self.n_qubits}")
            return [(self._cx_code, c, t)]
        raise ValueError(f"Unknown FlowMeas gate name: {gname!r}")

    def _resolve_cp_device(self):
        """Return the cupy.cuda.Device this adapter should run on."""
        import cupy as cp
        if self._cuda_index is not None:
            return cp.cuda.Device(self._cuda_index)
        if self._device is not None and getattr(self._device, "index", None) is not None:
            return cp.cuda.Device(self._device.index)
        return cp.cuda.Device()  # current device

    def _ensure_lut_cp(self):
        if self._lut_cp is None:
            import cupy as cp
            with self._resolve_cp_device():
                self._lut_cp = cp.asarray(self._lut_np)
        return self._lut_cp

    def translate_step(self, actions_flat):
        """Translate per-circuit action ids into primitive substeps.

        Parameters
        ----------
        actions_flat:
            Length-N array of int action ids. Accepts a torch tensor (any
            int dtype) or a cupy / numpy array. Values must lie in
            [0, max_id]; out-of-range ids will raise. Torch tensors are
            moved to the adapter's configured device if one was provided
            at construction; otherwise they must already be on a CUDA
            device matching the LUT's GPU.

        Returns
        -------
        List of (gate_codes, q1, q2) cupy int32 arrays, each shape (N,).
        Length of the list equals `max_depth`. Within each tuple, entries
        whose gate_code is -1 are no-ops for that circuit at that substep.
        """
        import cupy as cp

        try:
            import torch
        except ImportError:
            torch = None

        cp_device = self._resolve_cp_device()
        with cp_device:
            if torch is not None and isinstance(actions_flat, torch.Tensor):
                if actions_flat.dim() != 1:
                    raise ValueError(
                        f"actions_flat must be 1-D; got shape {tuple(actions_flat.shape)}"
                    )
                if self._device is not None:
                    actions_flat = actions_flat.to(self._device)
                elif not actions_flat.is_cuda:
                    actions_flat = actions_flat.cuda()
                actions_cp = cp.from_dlpack(actions_flat.to(torch.int32))
            else:
                actions_cp = cp.asarray(actions_flat, dtype=cp.int32)
                if actions_cp.ndim != 1:
                    raise ValueError(f"actions_flat must be 1-D; got shape {actions_cp.shape}")

            lut = self._ensure_lut_cp()  # (max_id+1, max_depth, 3) int32
            max_id = lut.shape[0] - 1
            # Default: validate. The per-layer sync costs ~1 µs and catches
            # replay-buffer / checkpoint mismatches where a stored id is no
            # longer in range under the current action_map size. CuPy fancy
            # indexing does not bounds-check, so an unvalidated stale id
            # would silently execute the wrong gate sequence. Hot paths
            # that have *upstream*-validated their id stream can opt out
            # via ``ActionAdapter(..., validate_action_ids=False)``.
            if self._validate_action_ids and actions_cp.size:
                bad = (actions_cp < 0) | (actions_cp > max_id)
                if bool(bad.any().item()):
                    raise ValueError(
                        f"action id out of range [0, {max_id}]; check for "
                        "replay-buffer or checkpoint mismatches (or pass "
                        "validate_action_ids=False if upstream validation "
                        "is guaranteed)"
                    )

            # Gather: result shape (N, max_depth, 3)
            gathered = lut[actions_cp]
            out: List[Tuple] = []
            for s in range(self._max_depth):
                g = cp.ascontiguousarray(gathered[:, s, 0])
                q1 = cp.ascontiguousarray(gathered[:, s, 1])
                q2 = cp.ascontiguousarray(gathered[:, s, 2])
                out.append((g, q1, q2))
        return out
