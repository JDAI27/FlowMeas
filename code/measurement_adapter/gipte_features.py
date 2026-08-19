"""GIPTE feature extractor: tableau -> gauge-invariant hit-feature set.

This is the *representation* half of the GIPTE encoder. Given a batched
Clifford tableau and a fixed Pauli dictionary, it builds the per-term
feature set

    H[b, m, k,:] = [ hit_k(b,m), |coeff_k|, sign(coeff_k), locality_k ]   (d=4)

where ``hit_k`` is the measurability indicator (1 iff ``P_k`` is Z-diagonal under
the Clifford) and the remaining three columns are W-independent constants. ``hit``
is exactly invariant to the stabilizer-group / reward gauge (it depends only on
the measurable subspace ``M_U``), and the metadata columns are gauge-independent
by construction, so ``H`` is an exactly gauge-invariant representation.

The optional ``covariant_shaping`` channel appends ``xweight_k / n`` (the X-block
popcount, a *covariant* distance-to-measurable signal). Enabling it deliberately
trades exact invariance for a denser gradient signal — keep it OFF for the
canonical, exactly-invariant policy.

Design notes
------------
* No float32 W materialization and no Gaussian elimination: hit features come
  from the packed GF(2) ``conjugate_dictionary_packed`` kernel via
  ``TableauBatchAdapter.hit_features``.
* The dictionary metadata is built once (a fixed, build-once artifact).
* ``extract(active_only=False)`` returns the full ``(B*M, K, d)`` set + full
  index map (the static-shape, CUDA-graph-friendly path); ``active_only=True``
  mirrors ``to_flat_tensors_active_only`` (a ``nonzero`` host sync) for callers
  that want the active subset.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import torch


class TableauFeatureExtractor:
    """Build the GIPTE gauge-invariant hit-feature set from a batched tableau."""

    def __init__(
        self,
        pauli_strings: Sequence[str],
        num_qubits: int,
        coeffs: Optional[Sequence[float]] = None,
        device: "torch.device | str" = "cuda",
        covariant_shaping: bool = False,
        normalize_xweight: bool = True,
    ):
        self.pauli_strings = list(pauli_strings)
        self.K = len(self.pauli_strings)
        self.n_qubits = int(num_qubits)
        self.device = torch.device(device) if isinstance(device, str) else device
        self.covariant_shaping = bool(covariant_shaping)
        self.normalize_xweight = bool(normalize_xweight)

        # feature_dim = hit (1) + [|coeff|, sign(coeff), locality] (3) [+ xweight (1)]
        self.feature_dim = 4 + (1 if self.covariant_shaping else 0)

        if coeffs is None:
            coeffs_t = torch.ones(self.K, dtype=torch.float32)
        else:
            coeffs_t = torch.as_tensor(list(coeffs), dtype=torch.float32)
            if coeffs_t.shape != (self.K,):
                raise ValueError(
                    f"coeffs must have length K={self.K}; got {tuple(coeffs_t.shape)}"
                )

        # Per-term W-independent metadata (gauge-invariant constants):
        #   |coeff|, sign(coeff), locality (Pauli weight / n).
        locality = torch.tensor(
            [self._pauli_weight(s) / max(1, self.n_qubits) for s in self.pauli_strings],
            dtype=torch.float32,
        )
        meta = torch.stack(
            [coeffs_t.abs(), torch.sign(coeffs_t), locality], dim=-1
        )  # (K, 3)
        self._meta = meta.to(self.device)

    def _symplectic_vecs(self) -> "np.ndarray":
        """``(K, 2n)`` uint8 symplectic matrix for the dictionary (X then Z bits).

        Matches ``TableauBatchAdapter._pauli_string_to_symplectic``: X-bit set for
        X/Y, Z-bit set for Y/Z; recognises +/-/+i/-i prefixes. Used to build the
        extractor's OWN static packed dictionary for the capture path (a fixed
        address that survives across tableau instances, unlike the per-call dict
        cached on the adapter).
        """
        import numpy as np
        n = self.n_qubits
        vecs = np.zeros((self.K, 2 * n), dtype=np.uint8)
        # Mirror TableauBatchAdapter._pauli_string_to_symplectic EXACTLY (no
        # strip/upper): the adapter parser also drives the cost/reward, so the
        # captured-mode dict must match it bit-for-bit on ALL inputs, not just
        # clean ones — otherwise captured-mode features would diverge from both
        # the eager fallback and the reward. (5-pass capture gate P3.)
        for i, ps in enumerate(self.pauli_strings):
            start = 0
            if len(ps) >= 2 and ps[0] in "+-" and ps[1] == "i":
                start = 2
            elif len(ps) >= 1 and ps[0] in "+-":
                start = 1
            for j, ch in enumerate(ps[start:start + n]):
                if ch in ("X", "Y"):
                    vecs[i, j] = 1
                if ch in ("Y", "Z"):
                    vecs[i, n + j] = 1
        return vecs

    def dict_packed_cupy(self):
        """Cached CuPy ``uint8[K, ceil(2n/8)]`` packed dictionary (fixed address).

        Built once (CUDA-only). The fixed address is required so a captured CUDA
        graph can read it across many sampling calls / tableau instances.
        """
        cached = getattr(self, "_dict_cp", None)
        if cached is None:
            import cupy as cp
            from clifford_tableau.measurement import pack_pauli_symplectic
            packed = pack_pauli_symplectic(self._symplectic_vecs(), self.n_qubits)
            cached = cp.asarray(packed)
            self._dict_cp = cached
        return cached

    def assemble_static(self, hit_view: torch.Tensor,
                        xweight_view: torch.Tensor, total_rows: int) -> torch.Tensor:
        """Build ``H`` (total_rows, K, feature_dim) from torch views of the static
        hit/xweight buffers + the constant metadata. Bit-identical to ``extract``'s
        assembly; used inside the CUDA-graph capture region.
        """
        meta = self._meta.view(1, self.K, 3).expand(total_rows, self.K, 3)
        feats = [hit_view.unsqueeze(-1).float(), meta]
        if self.covariant_shaping:
            xw = xweight_view.float()
            if self.normalize_xweight:
                xw = xw / float(self.n_qubits)
            feats.append(xw.unsqueeze(-1))
        return torch.cat(feats, dim=-1)

    def _pauli_weight(self, pauli: str) -> int:
        """Number of non-identity factors, parsed EXACTLY like ``_symplectic_vecs``
        so the ``locality`` metadata stays consistent with the hit computation for
        non-canonical Pauli strings.

        Matches the symplectic parser bit-for-bit: NO ``strip``/``upper`` (X/Y/Z
        are matched case-sensitively, exactly as the X/Z bits are set), truncate to
        the first ``n_qubits`` body characters, and the same ``+``/``-``/``+i``/
        ``-i`` prefix handling. (Whitespace/lowercase/extra-suffix inputs therefore
        count the same way the hit kernel sees them, not under a normalized view.)
        """
        start = 0
        if len(pauli) >= 2 and pauli[0] in "+-" and pauli[1] == "i":
            start = 2
        elif len(pauli) >= 1 and pauli[0] in "+-":
            start = 1
        body = pauli[start:start + self.n_qubits]
        return sum(1 for ch in body if ch in ("X", "Y", "Z"))

    def extract(
        self, batched_tableau, active_only: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(H, indices)``.

        ``H`` is ``(N, K, feature_dim)`` float32 and ``indices`` is ``(N, 2)``
        long with columns ``(batch_idx, meas_idx)``. ``N == B*M`` when
        ``active_only=False`` (static-shape path), else the active-row count.
        """
        # xweight is only consumed under covariant_shaping; skip its host-side
        # materialization otherwise (the fused kernel computes the popcount either
        # way — only the DLPack view + float32 cast of the unused tensor is elided).
        hit, xweight = batched_tableau.hit_features(
            self.pauli_strings, need_xweight=self.covariant_shaping
        )  # (B, M, K); xweight is None when covariant_shaping is False
        b, m, k = hit.shape
        if k != self.K:
            raise ValueError(
                f"dictionary size mismatch: tableau returned K={k}, expected {self.K}"
            )

        if active_only:
            # Replay / dynamic hot path: gather the active rows of the hit/xweight
            # channels FIRST, then assemble over only n_active rows, so the per-step
            # metadata cat allocates O(n_active) rather than the full O(B*M) (B,M,K,d)
            # buffer — matching the legacy to_flat_tensors_active_only
            # gather-before-materialize ordering. Bit-identical to building the full
            # tensor and gathering after (the hit kernel is full-batch either way).
            active = batched_tableau.active.reshape(-1)
            idx = active.nonzero(as_tuple=True)[0]
            indices = torch.stack([idx // m, idx % m], dim=1)
            hit_a = hit.reshape(b * m, self.K)[idx]
            feats_a = [
                hit_a.unsqueeze(-1),
                self._meta.view(1, self.K, 3).expand(idx.numel(), self.K, 3),
            ]
            if self.covariant_shaping:
                xw_a = xweight.reshape(b * m, self.K)[idx]
                xw_a = xw_a / float(self.n_qubits) if self.normalize_xweight else xw_a
                feats_a.append(xw_a.unsqueeze(-1))
            return torch.cat(feats_a, dim=-1), indices

        # Static-shape full path: stable (B*M, K, d) with the full index map.
        meta = self._meta.view(1, 1, self.K, 3).expand(b, m, self.K, 3)
        feats = [hit.unsqueeze(-1), meta]
        if self.covariant_shaping:
            xw = xweight / float(self.n_qubits) if self.normalize_xweight else xweight
            feats.append(xw.unsqueeze(-1))
        H = torch.cat(feats, dim=-1)  # (B, M, K, feature_dim)
        H_rows = H.reshape(b * m, self.K, self.feature_dim)
        bi = torch.arange(b, device=H.device).repeat_interleave(m)
        mi = torch.arange(m, device=H.device).repeat(b)
        indices = torch.stack([bi, mi], dim=1)
        return H_rows, indices
