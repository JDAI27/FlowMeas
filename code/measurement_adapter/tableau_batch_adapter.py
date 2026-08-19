"""TableauBatchAdapter: state-evolution + W-readout + cost-path shim over BatchedCliffordSim.

Scope ( + Track-B mod-2/mod-4 bridge)
------------------------------------------------------------------
This adapter covers `code/clifford_map.py:CliffordMap`'s surface that drives
per-step state evolution, the policy network's W-matrix input, the
training-time cost path, and (since the Track-B mod-2 phase bridge) the
eval-time phase tensor consumed by ``energy_estimator.py`` /
``pauli_tracker.py``:

- ``apply_actions_step``, ``apply_action``            (per-step gate dispatch)
- ``to_flat_tensors_active_only``, ``to_flat_tensors`` (model input contract)
- ``reset``, ``reset_measurement``                     (lifecycle)
- ``.W``, ``.active``, ``.version``                    (read-only state)
- ``.heis_phase_vec``                                  (Z4 phase, mod-2 ⨯2 bridge to FlowMeas mod-4)
- ``_pauli_string_to_symplectic``, ``transform_paulis``
- ``prob_P_single``, ``prob_P_multi``                  (training cost path)

The adapter is now the CT-backed shared-core route for CUDA training and cost
paths. CPU-only paths still keep ``CliffordMap`` as a compatibility fallback
until the remaining estimator/parity-oracle callers are migrated.
"""
from __future__ import annotations

from collections import OrderedDict
from typing import Dict, Optional, Tuple, Union

import torch

try:
    from .cp_stream import current_external_stream
except ImportError:  # pragma: no cover - direct-execution mode
    from cp_stream import current_external_stream

from .action_adapter import ActionAdapter, _CT_INSTALL_HINT


_ACTION_ADAPTER_CACHE_MAXSIZE = 16
_PAULI_DICT_CACHE_MAXSIZE = 16

# Process-wide cache for the packed symplectic form of a (static) Pauli
# dictionary, keyed by (cuda device index, n_qubits, the Pauli strings). The
# packed dict is a pure function of those and INDEPENDENT of tableau state (W),
# so it is safe to share across the short-lived per-step ``TableauBatchAdapter``
# instances. It must be module-level, not per-instance: the trainer allocates a
# fresh adapter every sample/replay/loss call, so a per-instance cache would miss
# every step and re-pay the O(K*n) encode + pack + H2D for the static
# Hamiltonian. Values are tiny (K*ceil(2n/8) uint8 ~ 15 KB).
_PACKED_PAULI_DICT_CACHE: "OrderedDict[tuple, object]" = OrderedDict()


class TableauBatchAdapter:
    """FlowMeas (B, M)-shaped tableau batch backed by BatchedCliffordSim.

    Parameters
    ----------
    n_qubits:
        Number of qubits per circuit.
    batch_size:
        Outer batch dimension B.
    n_measurements:
        Inner measurement dimension M.
    device:
        Torch device. Must be CUDA; the underlying CT kernel is GPU-only.
    """

    def __init__(
        self,
        n_qubits: int,
        batch_size: int,
        n_measurements: int,
        device: Union[str, torch.device] = "cuda",
    ):
        try:
            from clifford_tableau.sim import BatchedCliffordSim
        except ImportError as e:
            raise ImportError(_CT_INSTALL_HINT) from e

        self.n_qubits = int(n_qubits)
        self.batch_size = int(batch_size)
        self.n_measurements = int(n_measurements)
        self.N2 = 2 * self.n_qubits
        self.total_tableaus = self.batch_size * self.n_measurements

        device_in = torch.device(device) if isinstance(device, str) else device
        if device_in.type != "cuda":
            raise ValueError("TableauBatchAdapter requires a CUDA device")

        # Resolve a concrete cuda index so torch / cupy / CT all agree on which
        # GPU we are on. torch.device('cuda') without an index resolves to the
        # current device; normalize self.device to the indexed form so later
        # tensor.to(self.device) doesn't follow torch's current-device drift
        # if it changes after the adapter is constructed.
        self._cuda_index = (
            device_in.index if device_in.index is not None
            else torch.cuda.current_device()
        )
        self._device_str = f"cuda:{self._cuda_index}"
        self.device = torch.device(self._device_str)

        import cupy as cp
        self._cp_device = cp.cuda.Device(self._cuda_index)
        with self._cp_device:
            self._sim = BatchedCliffordSim(
                n_qubits=self.n_qubits,
                batch_size=self.total_tableaus,
                device=self._device_str,
            )

        self.active = torch.ones(
            self.batch_size, self.n_measurements,
            dtype=torch.bool, device=self.device,
        )
        self.version = 0

        # Action-adapter cache. Keyed by ``id(action_map)``, which is cheap but
        # technically unsafe across object lifetimes — Python can reuse the
        # address of a garbage-collected dict for an unrelated one. FlowMeas
        # builds the action map once per training run and keeps it alive for
        # the duration, so the id-based key is safe in production. The cache
        # is bounded as a defensive cap (LRU eviction) and exposed via
        # ``clear_action_adapter_cache()`` for callers that rebuild action
        # maps and need to drop stale entries.
        self._action_adapter_cache: "OrderedDict[int, ActionAdapter]" = OrderedDict()
        self._validate_action_ids = True

        # Packed Pauli-dictionary cache for the GIPTE hit-feature cost path.
        # NOTE: the packed-Pauli-dict cache is now PROCESS-WIDE
        # (``_PACKED_PAULI_DICT_CACHE`` at module scope), not per-instance: the
        # trainer allocates a fresh adapter every step, so a per-instance cache
        # missed every call and re-paid the static Hamiltonian's host encode +
        # H2D copy each step. See ``_packed_pauli_dict``.

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _adapter_for(self, action_map: Dict[int, Tuple]) -> ActionAdapter:
        key = id(action_map)
        cache = self._action_adapter_cache
        ad = cache.get(key)
        if ad is not None:
            cache.move_to_end(key)  # LRU bump
            return ad
        ad = ActionAdapter(
            action_map,
            self.n_qubits,
            device=self.device,
            cuda_index=self._cuda_index,
            validate_action_ids=self.validate_action_ids,
        )
        cache[key] = ad
        while len(cache) > _ACTION_ADAPTER_CACHE_MAXSIZE:
            cache.popitem(last=False)  # evict LRU
        return ad

    @property
    def validate_action_ids(self) -> bool:
        return self._validate_action_ids

    @validate_action_ids.setter
    def validate_action_ids(self, value: bool) -> None:
        value = bool(value)
        if getattr(self, "_validate_action_ids", value) != value:
            self.clear_action_adapter_cache()
        self._validate_action_ids = value

    def clear_action_adapter_cache(self) -> None:
        """Drop all cached ``ActionAdapter`` instances.

        Call after rebuilding the action map (or if you suspect dict-id
        reuse across short-lived action maps). Safe to call at any time;
        the next ``apply_actions_step`` will rebuild from scratch.
        """
        self._action_adapter_cache.clear()

    def _flat_active_torch(self, mask: Optional[torch.Tensor]) -> torch.Tensor:
        """Resolve the active set for this layer as a flat torch bool tensor.

        Matches legacy ``CliffordMap.apply_actions_step`` semantics: when the
        caller supplies a mask, it is used directly without intersecting
        ``self.active``. When mask is None, ``self.active`` is the fallback.
        """
        if mask is None:
            return self.active.reshape(-1)
        mask = mask.to(device=self.device, dtype=torch.bool)
        if mask.shape != (self.batch_size, self.n_measurements):
            raise ValueError(
                f"mask shape {tuple(mask.shape)} != "
                f"({self.batch_size}, {self.n_measurements})"
            )
        return mask.reshape(-1).contiguous()

    def _torch_bool_to_cp(self, flat_bool_torch: torch.Tensor):
        """Convert a flat torch bool tensor to a cupy bool array on our device.

        The DLPack handoff is wrapped in ``_torch_stream()`` so the
        ``flat_bool_torch.to(torch.uint8)`` cast and the resulting
        ``cp.from_dlpack`` view are both ordered against PyTorch's current
        stream — the CT kernel wrap addressed the kernel launches
        but the DLPack handoffs were still happening outside the wrapped
        stream context, leaving producer/consumer ordering dependent on
        DLPack default-stream behavior.
        """
        import cupy as cp
        with self._cp_device, self._torch_stream():
            return cp.from_dlpack(flat_bool_torch.to(torch.uint8)).view(cp.bool_)

    def _torch_stream(self):
        """Return a CuPy ``ExternalStream`` wrapping the current torch CUDA stream.

        The fused metadata kernel already launches on PyTorch's current
        stream via this exact idiom; the CT simulator calls
        (``translate_step`` / ``apply_layer_batched`` / phase + W-bits reads)
        had been left under plain ``cp.cuda.Device``, so they could land on
        CuPy's default stream and race the PyTorch side. Wrapping every CT
        kernel handoff in this ``ExternalStream`` makes all GPU work issue
        onto a single, ordered stream — eliminating the race that would
        otherwise surface under non-default stream usage, graph capture, or
        any future scheduling change.

        Uses the cached wrapper (``cp_stream.current_external_stream``) —
        this method is called on every CT kernel handoff, and the per-call
        torch ``Stream`` + CuPy ``ExternalStream`` wrapper allocations were
        a measured slice of the fused-path launch overhead.
        """
        return current_external_stream(self._cuda_index)

    # ------------------------------------------------------------------
    # Core API — matches CliffordMap surface
    # ------------------------------------------------------------------

    @torch.no_grad()
    def apply_actions_step(
        self,
        actions: torch.Tensor,
        action_map: Dict[int, Tuple],
        mask: Optional[torch.Tensor] = None,
    ) -> None:
        """Apply a single per-circuit layer of actions.

        Parameters
        ----------
        actions:
            (B, M) int tensor of action ids. Active rows must hold valid
            keys of ``action_map``. Inactive rows are not inspected — they
            may carry padding sentinels (e.g. ``-1``) without raising,
            matching legacy ``CliffordMap.apply_actions_step`` which
            indexes ``actions[mask]`` before validation.
        action_map:
            FlowMeas action map.
        mask:
            Optional (B, M) bool mask. If supplied, used directly as the
            active set for this layer (matching legacy ``CliffordMap``,
            which does NOT intersect with ``self.active``). If None, falls
            back to ``self.active``.
        """
        if actions.shape != (self.batch_size, self.n_measurements):
            raise ValueError(
                f"actions shape {tuple(actions.shape)} != "
                f"({self.batch_size}, {self.n_measurements})"
            )
        active_flat_torch = self._flat_active_torch(mask)
        # No host-sync early-return on all-False ``active_flat_torch``:
        # the CT ``apply_layer_batched`` kernel reads ``active_mask`` per row
        # and no-ops where False, so calling with an all-False mask is just
        # one kernel launch with zero per-row work — cheaper than an
        # ``.any().item()`` sync every layer.

        # Zero out masked-out slots so padding sentinels (-1, stale ids) don't
        # trip the LUT's range check. The kernel will still skip those rows
        # because active_mask is False; the substituted 0 is just a safe
        # in-range index. Matches CliffordMap, which only validates
        # actions[mask].
        actions_flat = actions.to(self.device).reshape(-1)
        actions_flat = torch.where(
            active_flat_torch,
            actions_flat,
            torch.zeros((), dtype=actions_flat.dtype, device=self.device),
        )

        active_cp = self._torch_bool_to_cp(active_flat_torch)
        adapter = self._adapter_for(action_map)
        # Issue every CT kernel onto PyTorch's current stream so the
        # ``translate_step`` -> ``apply_layer_batched`` chain stays ordered
        # against any subsequent torch reads. Plain ``cp.cuda.Device`` would
        # let CuPy fall back to its default stream and race the torch side.
        with self._cp_device, self._torch_stream():
            substeps = adapter.translate_step(actions_flat)
            for (g_cp, q1_cp, q2_cp) in substeps:
                self._sim.apply_layer_batched(g_cp, q1_cp, q2_cp, active_mask=active_cp)
        # Call counter — ticks on every call, including all-False mask. The
        # cheaper alternative ("only tick when work was done") would require
        # a host sync on ``active_flat_torch.any()`` that the perf pass
        # explicitly removed. Cache consumers see at worst an extra rebuild
        # on no-op layers, never a missed invalidation.
        self.version += 1

    @torch.no_grad()
    def apply_action(
        self,
        batch_actions: torch.Tensor,
        batch_lengths: torch.Tensor,
        action_map: Dict[int, Tuple],
    ) -> None:
        """Apply a (B, M, max_len) sequence of actions, respecting per-row lengths.

        Mirrors `CliffordMap.apply_action`.
        """
        if batch_actions.dim() != 3:
            raise ValueError(f"batch_actions must be 3-D; got shape {tuple(batch_actions.shape)}")
        if batch_actions.shape[:2] != (self.batch_size, self.n_measurements):
            raise ValueError(
                f"batch_actions leading shape {tuple(batch_actions.shape[:2])} != "
                f"({self.batch_size}, {self.n_measurements})"
            )
        max_len = batch_actions.shape[2]
        batch_lengths = batch_lengths.to(self.device)
        # Early-break once every row has finished, matching the legacy
        # ``CliffordMap.apply_action`` (clifford_map.py:537-538). This is the
        # eval energy path only (sole caller: energy_estimator._apply_circuits_to_map);
        # it is NOT the CUDA-graph static sampler, so the per-step host sync is
        # acceptable. The tail layers are NOT free here: this adapter defaults to
        # ``validate_action_ids=True``, so each ``apply_actions_step`` already pays
        # a ``bad.any().item()`` validation sync plus a full B*M lowering in
        # ``translate_step``. On padded eval batches (actual lengths << max_len is
        # common) running every tail layer is strictly more work than breaking out.
        for step in range(max_len):
            step_mask = batch_lengths > step
            if not bool(step_mask.any().item()):
                break
            self.apply_actions_step(batch_actions[:, :, step], action_map, step_mask)

    # ------------------------------------------------------------------
    # Model input contract — to_flat_tensors_active_only & to_flat_tensors
    # ------------------------------------------------------------------

    def to_flat_tensors_active_only(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Active subset of W flattened to (n_active, 4n^2) float32.

        Matches `CliffordMap.to_flat_tensors_active_only` bit-for-bit (W only,
        no phase vector — preserves the model input contract).

        Returns
        -------
        (output, indices) where
          - output: (n_active, 4 * n_qubits ** 2) float32 on `self.device`
          - indices: (n_active, 2) long; columns are (batch_idx, meas_idx)
        """
        active_flat = self.active.view(-1)
        active_indices = active_flat.nonzero(as_tuple=True)[0]
        n_active = active_indices.numel()
        mat_size = self.N2 * self.N2
        if n_active == 0:
            return (
                torch.empty((0, mat_size), device=self.device, dtype=torch.float32),
                torch.empty((0, 2), device=self.device, dtype=torch.long),
            )

        W_torch = self._w_torch_view()  # (B*M, 2n, 2n) torch uint8 view
        W_active = W_torch[active_indices]
        output = W_active.contiguous().view(n_active, mat_size).to(torch.float32)
        output._source_refs = (W_torch, W_active)

        batch_indices = active_indices // self.n_measurements
        meas_indices = active_indices % self.n_measurements
        indices = torch.stack([batch_indices, meas_indices], dim=1)
        return output, indices

    def to_flat_tensors(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Full (B, M, 4n^2) float32 view of W plus a (B, M) active mask."""
        W_torch = self._w_torch_view()  # (B*M, 2n, 2n) uint8
        out = W_torch.view(
            self.batch_size, self.n_measurements, self.N2 * self.N2
        ).to(torch.float32)
        out._source_ref = W_torch
        return out, self.active.clone()

    def to_flat_tensors_into(self, out_unpacked: "cp.ndarray") -> torch.Tensor:
        """to_flat_tensors features (default-W path) unpacked into a caller-
        preallocated STATIC cupy buffer out_unpacked (uint8[B*M, 2n, 2n]),
        for CUDA-graph capture: the unpack kernel writes a FIXED address that is
        retained across several per-K graph captures (so it is never freed/reused
        between them, unlike the fresh cp.empty of the default get_W_bits).
        Returns (total_rows, (2n)^2) float32, byte-identical to the default-W
        _policy_features output.
        """
        with self._cp_device, self._torch_stream():
            W_cp = self._sim.get_W_bits(out=out_unpacked)
            W_torch = torch.from_dlpack(W_cp)
        total_rows = self.batch_size * self.n_measurements
        return W_torch.reshape(total_rows, self.N2 * self.N2).to(torch.float32).contiguous()

    # ------------------------------------------------------------------
    # State-shaped accessors (read-only)
    # ------------------------------------------------------------------

    def _w_torch_view(self) -> torch.Tensor:
        """Zero-copy torch view of the underlying packed W bits, shape (B*M, 2n, 2n) uint8."""
        # Keep BOTH the CT kernel launch AND the DLPack handoff inside
        # the wrapped stream context so the producer (CT ``get_W_bits``)
        # and consumer (``torch.from_dlpack``) execute on the same
        # ordered stream. wrapped only the CT call; the
        # ``from_dlpack`` was outside, leaving consumer ordering up to
        # DLPack's default-stream behavior.
        with self._cp_device, self._torch_stream():
            W_cp = self._sim.get_W_bits()
            return torch.from_dlpack(W_cp)

    @property
    def W(self) -> torch.Tensor:
        """Read-only (B, M, 2n, 2n) int8 view of the current Clifford map.

        The underlying storage is uint8 from CT; values are 0 or 1, so a
        view-cast to int8 via ``Tensor.view(dtype)`` is bit-exact and
        zero-copy. Verified on torch 2.11: same ``data_ptr``, no
        allocation. Modifying this tensor in place will desync the
        adapter — do not.

        Performance note: every ``.W`` access calls ``get_W_bits()``,
        which runs an unpack kernel on the packed tableau bits. Each
        call is cheap in absolute terms (~0.1 ms at n=52, B=1000) but
        it is *not* the free attribute access ``CliffordMap.W`` was.
        Callers that need to index into ``W`` repeatedly should grab a
        single reference first (``W = tba.W``) and reuse it rather than
        re-accessing in a loop.
        """
        W = self._w_torch_view().view(
            self.batch_size, self.n_measurements, self.N2, self.N2
        )
        return W.view(torch.int8)

    def _require_packed_w_getter(self) -> None:
        """Fail fast with an actionable message if the CT sim is too old.

        The GFlowNet ``packed_w_input`` guard only checks that the *adapter* class
        exposes ``policy_packed_w`` — which is always true here — so it cannot catch
        a clifford-tableau backend that predates ``get_W_bits_packed_u32``. Mirror
        the:meth:`hit_features` discipline so a CT-version skew raises a clean
        RuntimeError instead of a cryptic ``AttributeError`` on the first sampling
        forward.
        """
        if not hasattr(self._sim, "get_W_bits_packed_u32"):
            raise RuntimeError(
                "policy_packed_w (packed_w_rowtoken) requires a clifford-tableau "
                "backend exposing get_W_bits_packed_u32 (>= the packed-W release). "
                + _CT_INSTALL_HINT
            )

    def policy_packed_w(self) -> torch.Tensor:
        """Bit-packed W as a torch ``int32[B*M, 2n, ceil(2n/32)]`` tensor (zero-copy).

        The compact policy input for ``model_type='packed_w_rowtoken'``: the model
        unpacks these bits on-GPU, so the float ``(2n)^2`` ``to_flat_tensors``
        materialization (the ~32-128x inflation) is skipped entirely.

        Layout is CT's little-endian column packing (``BatchedCliffordSim
.get_W_bits_packed_u32``), which is bit-identical to the model's LSB-first
        ``_unpack`` contract — verified: ``model._unpack(this) == to_flat bits``.
        The ``uint32`` words are exposed through an ``int32`` DLPack view (same bit
        pattern, zero-copy); the model reads them as unsigned.

        Both the CT kernel launch and the DLPack handoff stay inside the wrapped
        stream context so producer/consumer share one ordered stream (same
        discipline as ``_w_torch_view``).

        Returns a FRESH buffer each call (CT ``get_W_bits_packed_u32(out=None)``
        ``cp.empty()``s a new array). Callers that cache the result across steps
        (GFNs ``cache_for_flows``) rely on this non-aliasing — see the comment at
        the packed-W ``cache_step_data`` site. Use:meth:`policy_packed_w_into`
        only where a single fixed buffer is intended (CUDA-graph capture).
        """
        self._require_packed_w_getter()
        import cupy as cp
        with self._cp_device, self._torch_stream():
            packed = self._sim.get_W_bits_packed_u32()  # cupy uint32 (B*M, 2n, row_words)
            return torch.from_dlpack(packed.view(cp.int32))

    def policy_packed_w_into(self, out: "cp.ndarray") -> "cp.ndarray":
        """Write packed W into a pre-allocated cupy ``uint32[B*M, 2n, ceil(2n/32)]``.

        Static-buffer variant of:meth:`policy_packed_w` for CUDA-graph capture:
        the GIPTE/packed-W capture path keeps a fixed-address ``out`` buffer that
        the captured graph reads, and refreshes it each step via this call. Keeps
        all ``_sim`` access inside the adapter (public-API boundary), and runs on
        the wrapped stream so the write is ordered before the graph replay.
        """
        self._require_packed_w_getter()
        with self._cp_device, self._torch_stream():
            self._sim.get_W_bits_packed_u32(out=out)
        return out

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @torch.no_grad()
    def copy(self) -> "TableauBatchAdapter":
        """Return an independent adapter with a direct GPU copy of CT state.

        CUDA-graph sampling needs a non-aliasing final tableau snapshot, but
        rebuilding that snapshot by replaying every sampled layer doubles the
        CT apply work. The underlying ``BatchedCliffordSim.copy()`` clones the
        packed simulator state directly, so expose the same operation at the
        FlowMeas adapter boundary.
        """
        new = object.__new__(type(self))
        new.n_qubits = self.n_qubits
        new.batch_size = self.batch_size
        new.n_measurements = self.n_measurements
        new.N2 = self.N2
        new.total_tableaus = self.total_tableaus
        new._cuda_index = self._cuda_index
        new._device_str = self._device_str
        new.device = self.device
        new._cp_device = self._cp_device
        with self._cp_device, self._torch_stream():
            new._sim = self._sim.copy()
        new.active = self.active.clone()
        new.version = self.version
        new._action_adapter_cache = OrderedDict()
        new._validate_action_ids = self._validate_action_ids
        return new

    @torch.no_grad()
    def reset(self) -> None:
        """Reset all (B, M) tableaus to identity and mark every row active."""
        import cupy as cp
        with self._cp_device, self._torch_stream():
            # CT's unmasked reset replaces ``self._sim._state`` with freshly
            # allocated arrays. CUDA graphs captured against this adapter hold
            # raw pointers to the existing state buffers, so replacing them makes
            # replay mutate stale storage while W readout observes the new state.
            # Use the masked reset path for all rows; it copies identity into the
            # existing buffers and preserves graph-captured addresses.
            mask_cp = cp.ones(self.total_tableaus, dtype=cp.bool_)
            self._sim.reset(mask=mask_cp)
            # ``BatchedCliffordSim.reset(mask=...)`` queues device copies from
            # temporary CuPy identity/mask arrays. Reset is a per-call setup step,
            # not the per-sampled-layer hot path, so synchronize here to keep
            # those temporaries alive until the copies have actually completed.
            torch.cuda.current_stream(self.device).synchronize()
        self.active.fill_(True)
        self.version += 1

    @torch.no_grad()
    def reset_inplace(self) -> None:
        """Reset all (B, M) tableaus to identity IN-PLACE, preserving the
        underlying ``_sim._state`` cupy buffers.

        ``reset()`` calls ``_sim.reset(mask=None)``, which REPLACES ``_state``
        with a fresh ``BatchedTableauState`` at new buffer addresses. That is
        fatal for CUDA-graph capture: the captured apply/unpack kernels bind the
        ``_state`` buffer addresses at capture time, so a buffer-replacing reset
        between captures makes every per-K graph evolve a SEPARATE, decoupled
        state copy (and leaves the live ``_state`` stuck at identity). This uses
        the in-place full-mask reset path so all captures + replays share one
        persistent ``_state`` buffer.
        """
        import cupy as cp
        with self._cp_device, self._torch_stream():
            full_mask = cp.ones(self.total_tableaus, dtype=cp.bool_)
            self._sim.reset(mask=full_mask)
            # Keep CT reset temporaries and the just-created mask alive until
            # the queued reset copies have completed on the torch stream.
            torch.cuda.current_stream(self.device).synchronize()
        self.active.fill_(True)
        self.version += 1

    @torch.no_grad()
    def reset_inplace_with_mask(self, full_mask) -> None:
        """Reset all (B, M) tableaus in-place using ``full_mask``.

        Precondition: ``full_mask`` is a caller-owned cupy bool array with shape
        ``(self.total_tableaus,)`` and dtype ``cp.bool_`` on this adapter's CUDA
        device. The mask is treated as an all-true reset mask and is not
        allocated here.
        """
        with self._cp_device, self._torch_stream():
            self._sim.reset(mask=full_mask)
            # ``BatchedCliffordSim.reset(mask=...)`` may queue copies from
            # temporary CuPy identity buffers. This helper is only used during
            # graph setup / call reset, not the per-layer hot path, so sync to
            # make stream ordering and temporary lifetimes explicit.
            torch.cuda.current_stream(self.device).synchronize()
        self.active.fill_(True)
        self.version += 1

    @torch.no_grad()
    def reset_measurement(self, batch_idx: int, meas_idx: int) -> None:
        """Reset a single (b, m) tableau to identity and mark it active."""
        import cupy as cp
        b = int(batch_idx)
        m = int(meas_idx)
        if not (0 <= b < self.batch_size):
            raise IndexError(
                f"batch_idx {b} out of range [0, {self.batch_size})"
            )
        if not (0 <= m < self.n_measurements):
            raise IndexError(
                f"meas_idx {m} out of range [0, {self.n_measurements})"
            )
        flat_idx = b * self.n_measurements + m
        with self._cp_device, self._torch_stream():
            mask_cp = cp.zeros(self.total_tableaus, dtype=cp.bool_)
            mask_cp[flat_idx] = True
            self._sim.reset(mask=mask_cp)
            torch.cuda.current_stream(self.device).synchronize()
        self.active[b, m] = True
        self.version += 1

    # ------------------------------------------------------------------
    # Cost-path methods
    # ------------------------------------------------------------------

    def _pauli_string_to_symplectic(
        self, pauli_strings: "list[str]"
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert a list of Pauli strings to symplectic + phase tensors.

        Mirrors ``CliffordMap._pauli_string_to_symplectic``. Recognised prefixes:
        ``+``, ``-``, ``+i``, ``-i``. Body characters are I/X/Y/Z over n qubits.
        Strings shorter or longer than ``n_qubits`` are silently
        zero-padded / truncated — same behavior as ``CliffordMap``.

        Performance
        -----------
        The parsing is a CPU Python loop, costing O(n_strings × n_qubits).
        At a Hubbard-benchmark scale (n=52, n_strings ~ few thousand) this is
        microseconds. At very large n_strings (>= 10k) it climbs to
        hundreds of milliseconds per call. The expected call cadence is
        once per training step against a static Hamiltonian list, so
        caching at the call site is the right optimization if perf
        matters; the FlowMeas-side ``_pauli_cache`` in ``CliffordMap`` plays
        that role and is intentionally NOT replicated here yet
        (deferred until measures real-world cost).

        Returns
        -------
        (vecs, phases) where
          - vecs: (n_strings, 2n) bool — first n entries are X-bits, last n are Z-bits
          - phases: (n_strings,) int8 mod 4 (0=+1, 1=+i, 2=-1, 3=-i)
        """
        n_strings = len(pauli_strings)

        # Build everything on CPU then bulk-transfer once. Pin the staging
        # buffers so the device copy is asynchronous on the default stream.
        vecs_cpu = torch.zeros(n_strings, self.N2, dtype=torch.bool, pin_memory=True)
        phases_cpu = torch.zeros(n_strings, dtype=torch.int8, pin_memory=True)
        for i, ps in enumerate(pauli_strings):
            phase = 0
            start = 0
            if len(ps) >= 2 and ps[0] in "+-" and ps[1] == "i":
                phase = 1 if ps[0] == "+" else 3
                start = 2
            elif len(ps) >= 1 and ps[0] in "+-":
                phase = 0 if ps[0] == "+" else 2
                start = 1
            pauli_part = ps[start:start + self.n_qubits]
            for j, ch in enumerate(pauli_part):
                if ch in ("X", "Y"):
                    vecs_cpu[i, j] = True
                if ch in ("Y", "Z"):
                    vecs_cpu[i, self.n_qubits + j] = True
                if ch == "Y":
                    phase = (phase + 1) % 4
            phases_cpu[i] = phase
        vecs = vecs_cpu.to(self.device, non_blocking=True)
        phases = phases_cpu.to(self.device, non_blocking=True)
        return vecs, phases

    def transform_paulis(
        self, pauli_vecs: torch.Tensor, chunk_size: int = 4096
    ) -> torch.Tensor:
        """Apply the current Clifford map to a batch of input Paulis.

        Computes ``U P U†`` in symplectic form for every (b, m) tableau, where
        ``U`` is the Clifford represented by ``self.W`` on that row. Phase is
        not tracked here (mirroring ``CliffordMap.transform_paulis``); use
        ``_pauli_string_to_symplectic`` for phase data.

        Parameters
        ----------
        pauli_vecs:
            ``(n_paulis, 2n)`` bool. First n columns are X-bits, last n are
            Z-bits, matching the AG convention.
        chunk_size:
            Tableaus per matmul chunk. Bounds peak memory at large ``B*M``.

        Returns
        -------
        ``(B, M, n_paulis, 2n)`` bool.

        Memory profile
        --------------
        Per chunk the intermediate ``prod`` is shape
        ``(chunk, n_paulis, 2n)`` in float32, costing
        ``chunk × n_paulis × 2n × 4`` bytes. At benchmark shape (n=52, B*M=1000,
        n_paulis ~ 4000) and the default ``chunk_size=4096``, that's a
        single chunk of ~1.7 GB. ``CliffordMap.transform_paulis`` uses
        bit-packed GF(2) ops and is 32× more memory-efficient at the same
        chunk size, so reduce ``chunk_size`` when ``n_paulis`` or ``B*M``
        is large.
        """
        if pauli_vecs.dim() != 2 or pauli_vecs.shape[1] != self.N2:
            raise ValueError(
                f"pauli_vecs must be (n_paulis, {self.N2}); got {tuple(pauli_vecs.shape)}"
            )
        if not isinstance(chunk_size, int) or chunk_size <= 0:
            raise ValueError(f"chunk_size must be a positive int; got {chunk_size!r}")
        pauli_vecs = pauli_vecs.to(self.device, dtype=torch.bool)
        n_paulis = pauli_vecs.shape[0]
        n_tableaus = self.total_tableaus

        W_flat = self._w_torch_view()  # (B*M, 2n, 2n) uint8 view of CT bits
        p_float = pauli_vecs.to(torch.float32)  # (n_paulis, 2n)

        out = torch.empty(
            n_tableaus, n_paulis, self.N2,
            dtype=torch.bool, device=self.device,
        )
        for start in range(0, n_tableaus, chunk_size):
            end = min(start + chunk_size, n_tableaus)
            W_chunk = W_flat[start:end].to(torch.float32)  # (chunk, 2n, 2n)
            # Per-tableau matmul: out[k, p, j] = sum_i p[p, i] * W[k, i, j]
            # broadcasting p over the chunk dim.
            prod = torch.matmul(p_float.unsqueeze(0), W_chunk)  # (chunk, n_paulis, 2n)
            # Cast to int32 (not int64) and extract parity via bitwise AND.
            # Max accumulator value is 2n ≤ ~10^4 in any practical setting,
            # so int32 has plenty of headroom and halves the intermediate
            # memory footprint vs int64.
            out[start:end] = (prod.to(torch.int32) & 1).to(torch.bool)
        return out.view(self.batch_size, self.n_measurements, n_paulis, self.N2)

    def _packed_pauli_dict(self, pauli_strings: "list[str]"):
        """Cached packed symplectic form of a Pauli dictionary, on this device.

        Returns a CuPy ``uint8[K, ceil(2n/8)]`` array packed in the same
        little-endian column layout as a W row (see
        ``clifford_tableau.measurement.pack_pauli_symplectic``), so the CT packed
        conjugation kernel can select W rows to XOR by the dictionary's set bits.

        Cache is safe only when all callers share the same CUDA stream at miss
        and hit time (FlowMeas single-process invariant): the cache-miss H2D
        copy is stream-ordered against the current torch stream, but the
        cache-hit path returns the stored array without any fence. Multi-stream
        callers must insert an explicit stream-sync after retrieving the cached
        array (or extend the cache key with the stream handle).
        """
        import cupy as cp
        from clifford_tableau.measurement import pack_pauli_symplectic

        # PROCESS-WIDE cache (not ``self._pauli_dict_cache``): the trainer
        # allocates a fresh adapter every step, so a per-instance cache misses
        # every call and re-pays the O(K*n) host encode + pack + H2D for the
        # STATIC Hamiltonian (see _PACKED_PAULI_DICT_CACHE).
        # The packed dict is state-independent, so keying on (device, n_qubits,
        # strings) is correct and shareable across adapter instances/tableaus.
        key = (self._cuda_index, self.n_qubits, tuple(pauli_strings))
        cache = _PACKED_PAULI_DICT_CACHE
        cached = cache.get(key)
        if cached is not None:
            cache.move_to_end(key)
            return cached

        # Reuse FlowMeas's parser so the X/Z bit convention exactly matches the legacy
        # float cost path (and CliffordMap). vecs: (K, 2n) bool.
        vecs, _ = self._pauli_string_to_symplectic(list(pauli_strings))
        packed_np = pack_pauli_symplectic(
            vecs.detach().cpu().numpy(), self.n_qubits
        )
        # Issue the H2D copy on PyTorch's current stream (same discipline as
        # every other CuPy alloc/handoff in this file — _w_torch_view, hit_features,
        # etc., stream-ordering fixes). Without _torch_stream() the
        # cp.asarray transfer lands on CuPy's default stream while the consuming
        # conjugate_dictionary_packed kernel (prob_P_multi) runs on the torch
        # stream, so the kernel could read dict_packed before the copy completes
        # (a first-call/cache-miss data race).
        with self._cp_device, self._torch_stream():
            dict_packed = cp.ascontiguousarray(cp.asarray(packed_np))
        cache[key] = dict_packed
        while len(cache) > _PAULI_DICT_CACHE_MAXSIZE:
            cache.popitem(last=False)
        return dict_packed

    def prob_P_multi(self, pauli_strings: list) -> torch.Tensor:
        """Probability each Pauli string is a Z-basis-measurable stabilizer
        of the current Clifford map.

        Returns ``(B, M, n_strings) float32``. Entry ``(b, m, k)`` is 1.0 iff
        ``U P_k U†`` has no X component (only Z and/or I) for tableau (b, m),
        else 0.0 — matching ``CliffordMap.prob_P_multi`` bit-for-bit.

        Fast path: when the CT backend exposes the packed
        dictionary-conjugation primitive, the hit indicator is computed directly
        by a fused GF(2) XOR/popcount kernel on the bit-packed tableau. This is
        bit-identical to the legacy ``transform_paulis``+``any`` float path but
        avoids casting the tableau to float32 and materializing the
        ``(B, M, K, 2n)`` bool intermediate (the float matmul intermediate the
        ``transform_paulis`` docstring flags as 32x heavier). Falls back to the
        legacy float path for non-CT backends or empty dictionaries.
        """
        if hasattr(self._sim, "conjugate_dictionary_packed") and len(pauli_strings) > 0:
            dict_packed = self._packed_pauli_dict(pauli_strings)
            # Keep the packed-W readout, the conjugation kernel, and the DLPack
            # handoff on PyTorch's current stream so the hit tensor is ordered
            # against subsequent torch reads (same discipline as ``_w_torch_view``).
            with self._cp_device, self._torch_stream():
                hit_cp, _xweight_cp = self._sim.conjugate_dictionary_packed(dict_packed)
                hit_torch = torch.from_dlpack(hit_cp)  # uint8 (B*M, K)
            return hit_torch.view(
                self.batch_size, self.n_measurements, -1
            ).to(torch.float32)

        # Legacy float fallback (non-CT backend, or K == 0).
        p_vecs, _ = self._pauli_string_to_symplectic(pauli_strings)
        p_out = self.transform_paulis(p_vecs)  # (B, M, n_strings, 2n)
        has_x = p_out[..., : self.n_qubits].any(dim=3)  # (B, M, n_strings)
        return (~has_x).to(torch.float32)

    def prob_P_single(self, pauli_string: str) -> torch.Tensor:
        """Single-Pauli specialization. Returns ``(B, M) float32``."""
        return self.prob_P_multi([pauli_string]).squeeze(2)

    def hit_features(
        self, pauli_strings: "list[str]", need_xweight: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """GIPTE hit features for a Pauli dictionary in a single fused kernel.

        Returns ``(hit, xweight)``:

        * ``hit[b, m, k]``     — ``(B, M, K) float32``; 1.0 iff ``P_k`` is
          measurable (Z-diagonal) under tableau ``(b, m)``; identical to
:meth:`prob_P_multi`. **Exactly invariant** to the stabilizer-group /
          reward gauge (it depends only on the measurable subspace
          ``M_U = leftnull(W[:,:n])``).
        * ``xweight[b, m, k]`` — ``(B, M, K) float32`` when ``need_xweight`` (the
          X-block popcount of ``U P_k U†``; ``>= 0``, ``== 0`` iff hit; a
          gauge-**covariant** distance-to-measurable shaping signal), else
          ``None``. The covariant channel is off by default, so callers that do
          not use it (``covariant_shaping=False``) pass ``need_xweight=False`` to
          skip the DLPack view + float32 cast of the unused ``(B, M, K)`` tensor.
          The fused kernel always computes the popcount; only the host-side
          materialization is elided.

        Requires a clifford-tableau backend exposing
        ``conjugate_dictionary_packed`` (the packed GIPTE Stage-0 primitive).
        """
        if not hasattr(self._sim, "conjugate_dictionary_packed"):
            raise RuntimeError(
                "hit_features requires a clifford-tableau backend exposing "
                "conjugate_dictionary_packed (>= GIPTE Stage 0). "
                + _CT_INSTALL_HINT
            )
        if len(pauli_strings) == 0:
            empty = torch.zeros(
                self.batch_size, self.n_measurements, 0,
                device=self.device, dtype=torch.float32,
            )
            return empty, (empty.clone() if need_xweight else None)

        dict_packed = self._packed_pauli_dict(pauli_strings)
        with self._cp_device, self._torch_stream():
            hit_cp, xweight_cp = self._sim.conjugate_dictionary_packed(dict_packed)
            hit_torch = torch.from_dlpack(hit_cp)        # uint8  (B*M, K)
            xweight_torch = (
                torch.from_dlpack(xweight_cp) if need_xweight else None  # int32 (B*M, K)
            )
        hit = hit_torch.view(self.batch_size, self.n_measurements, -1).to(torch.float32)
        xweight = (
            xweight_torch.view(self.batch_size, self.n_measurements, -1).to(torch.float32)
            if need_xweight else None
        )
        return hit, xweight

    # ------------------------------------------------------------------
    # Phase tensor (eval-time / pauli_tracker consumer)
    # ------------------------------------------------------------------

    @property
    def heis_phase_vec(self) -> torch.Tensor:
        """Z4 mod-4 phase tensor matching ``CliffordMap.heis_phase_vec``.

        Returns ``(B, M, 2n)`` int8 on ``self.device``. Each entry is the
        phase exponent for one row of the Heisenberg-picture image of a
        generator — 0 → +1, 2 → −1. CT's underlying simulator stores this
        as a single AG-style mod-2 sign bit per row; FlowMeas's convention is
        Z4 mod-4 (0=+1, 1=+i, 2=−1, 3=−i) with the same int8 dtype, and
        Heisenberg-conjugated Hermitian Paulis only ever populate the
        even residues {0, 2}. The bridge is therefore exact and lossless:
        ``mod4 = mod2 << 1``.

        Identity ``mod4 = 2 · mod2``
        ----------------------------
        FlowMeas gate phase updates always add 2·(qubit-local Y bit) and reduce
        mod 4 — see ``code/clifford_map.py`` apply_H/S/CNOT scripts. CT's
        AG update XORs the same per-qubit bit into ``r``. Starting from
        zeros and applying the same gate sequence, the FlowMeas phase after k
        gates is ``2·r_k mod 4``, and since ``r_k ∈ {0, 1}`` this is just
        ``2·r_k``. Empirically confirmed bit-exact across random circuits
        on the parity test suite.

        Bridge cost
        -----------
        Each access reads CT's packed sign-bit buffer, unpacks to a per-row
        uint8 array, multiplies by 2, view-casts to int8, and reshapes. The
        cost is the unpack pass; it is ~microseconds at benchmark scale but it
        is *not* the free attribute access ``CliffordMap.heis_phase_vec``
        was. Callers that need to index into it repeatedly should grab a
        single reference first (``phase = tba.heis_phase_vec``) rather
        than re-accessing in a loop. Mirrors the ``.W`` property's caveat.

        Mutability
        ----------
        Unlike ``CliffordMap.heis_phase_vec`` (which IS the storage and is
        mutated in-place by FlowMeas-internal lifecycle methods), the tensor
        returned here is a *fresh derivation* of CT's packed bits — the
        ``* 2`` allocates a new cupy buffer. Mutating the returned tensor
        is therefore harmless but also pointless: subsequent reads
        re-derive from CT's state, so a write through this view is silently
        lost. Use ``reset()`` / ``reset_measurement()`` to clear phase.
        """
        import cupy as cp
        # Keep the DLPack handoff INSIDE the wrapped stream so the read
        # is ordered against the latest CT writes AND consumer ordering
        # against the resulting torch tensor is well-defined.
        # left the ``torch.from_dlpack`` outside the context, leaving the
        # consumer side dependent on default-stream behavior.
        with self._cp_device, self._torch_stream():
            phase_cp = self._sim.get_phase_vec()  # (B*M, 2n) uint8 mod-2
            # Multiply by 2 to lift mod-2 → mod-4 (stays in uint8 since 1*2=2).
            phase_cp_mod4 = phase_cp * cp.uint8(2)
            phase_torch = torch.from_dlpack(phase_cp_mod4)  # uint8 (B*M, 2n)
        return phase_torch.view(
            self.batch_size, self.n_measurements, self.N2
        ).view(torch.int8)
