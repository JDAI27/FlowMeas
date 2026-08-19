"""cuTensorNet-backed MPS sampling backend.

Production GPU path for MPS-native ``EnergyEstimator``. This module avoids
``NetworkState.set_initial_mps`` deliberately: cuQuantum/cuTensorNet 25.03 /
cutensornet-cu12 2.12.2 has an engine-side axis-convention bug where
``set_initial_mps`` gives wrong amplitudes in the documented ``pkn`` order
and can raise ``CUTENSORNET_STATUS_INTERNAL_ERROR`` for the empirically
correct transpose when ``n_qubits >= 4``.

Instead, this adapter uses lower-level cuTensorNet gate-split primitives:

* ``CuTensorNetMPSOps.apply_two_site_gate`` calls
  ``contract_decompose('aib,bjc,pqij->apx,xqc',...)`` to contract two
  neighboring MPS tensors with a two-qubit gate and split the result by SVD.
* ``CuTensorNetMPSBackend`` delegates the circuit/action/terminal/SWAP-network
  logic to ``MPSNativeBackend`` with ``CuTensorNetMPSOps`` injected.
* Sampling uses the existing perfect left-to-right MPS sampler. On CUDA
  tensors that sampler runs through torch CUDA kernels and still avoids any
  dense ``2**n`` state vector.

The public surface mirrors ``MPSNativeBackend.sample_outcomes`` so
``EnergyEstimator`` can choose this backend transparently when CUDA +
cuQuantum are available.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch

from .mps_native_backend import DenseReferenceMPSOps, MPSNativeBackend


logger = logging.getLogger(__name__)

ContractDecomposeFn = Callable[
    ..., Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]
]

# Module-level singleton for the perfect-MPS sampler. ``DenseReferenceMPSOps``
# is stateless (constructor takes no args), so every ``CuTensorNetMPSOps``
# instance can share the same sampler — no need for a per-op allocation.
_SHARED_SAMPLER = None


def _get_sampler():
    global _SHARED_SAMPLER
    if _SHARED_SAMPLER is None:
        _SHARED_SAMPLER = DenseReferenceMPSOps()
    return _SHARED_SAMPLER


def _import_contract_decompose() -> ContractDecomposeFn:
    """Import cuTensorNet's gate-split helper lazily.

    Prefer the non-deprecated ``cuquantum.tensornet.experimental`` path while
    retaining a fallback for older wheels that only expose
    ``cuquantum.cutensornet.experimental``.
    """
    from .backends import _import_cutensornet

    _import_cutensornet()
    try:
        from cuquantum.tensornet.experimental import contract_decompose
    except ImportError:
        from cuquantum.cutensornet.experimental import contract_decompose
    return contract_decompose


class CuTensorNetMPSOps:
    """cuTensorNet implementation of the ``MPSOps`` protocol.

    The two-site gate path is the production-relevant piece: it uses
    cuTensorNet's optimized contract+SVD gate split instead of torch's
    ``linalg.svd``. One-site gates and sampling remain simple torch tensor
    operations, which run on CUDA when the MPS tensors live on CUDA.
    """

    def __init__(
        self,
        *,
        contract_decompose_fn: Optional[ContractDecomposeFn] = None,
    ) -> None:
        self._contract_decompose = (
            contract_decompose_fn
            if contract_decompose_fn is not None
            else _import_contract_decompose()
        )
        # Sampling is delegated to a module-level singleton; the sampler
        # is stateless across calls.
        self._sampler = _get_sampler()

    @staticmethod
    def _svd_algorithm(
        *,
        max_bond_dim: Optional[int],
        truncation_tol: float,
    ) -> Dict[str, Any]:
        """Build cuTensorNet SVD options matching ``DenseReferenceMPSOps``.

        ``partition='V'`` absorbs singular values into the right tensor, the
        same convention as ``DenseReferenceMPSOps``. If a cuTensorNet version
        returns singular values explicitly anyway, ``apply_two_site_gate``
        handles that defensively.
        """
        svd_method: Dict[str, Any] = {"partition": "V"}
        if max_bond_dim is not None:
            svd_method["max_extent"] = int(max_bond_dim)
        if truncation_tol > 0.0:
            svd_method["abs_cutoff"] = float(truncation_tol)
        # ``qr_method = {}`` (empty dict, NOT ``None``) is the cuTensorNet
        # idiom for "enable QR-assisted SVD with default parameters". This
        # is the documented fast path for ternary contract+decompose splits
        # (``aib,bjc,pqij->apx,xqc``); the empty-dict toggle is materially
        # different from omitting the key (== disable QR). See cuQuantum
        # 25.03 docs for ``DecompositionOptions.qr_method``.
        return {
            "qr_method": {},
            "svd_method": svd_method,
        }

    def apply_one_site_gate(
        self,
        mps: List[torch.Tensor],
        site: int,
        gate: torch.Tensor,
    ) -> List[torch.Tensor]:
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
        if site < 0 or site + 1 >= len(mps):
            raise IndexError(
                f"two-site gate at site {site} is outside MPS of length "
                f"{len(mps)}"
            )

        left = mps[site]
        right = mps[site + 1]
        gate_c = gate.to(dtype=left.dtype, device=left.device).reshape(2, 2, 2, 2)
        algorithm = self._svd_algorithm(
            max_bond_dim=max_bond_dim,
            truncation_tol=truncation_tol,
        )

        # left:  a i b
        # right: b j c
        # gate:  p q i j  (outputs p/q, inputs i/j)
        # out:   a p x, x q c
        new_left, singular_values, new_right = self._contract_decompose(
            "aib,bjc,pqij->apx,xqc",
            left,
            right,
            gate_c,
            algorithm=algorithm,
        )
        if singular_values is not None:
            s = singular_values.to(dtype=new_right.dtype, device=new_right.device)
            new_right = s.reshape(-1, 1, 1) * new_right

        mps[site] = new_left.contiguous()
        mps[site + 1] = new_right.contiguous()
        logger.debug(
            "CuTensorNetMPSOps gate split at site %d: new bond dim = %d "
            "(max_bond_dim=%s, truncation_tol=%.3e)",
            site,
            int(mps[site].shape[2]),
            str(max_bond_dim),
            truncation_tol,
        )
        return mps

    def sample(
        self,
        mps: List[torch.Tensor],
        M: int,
        *,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        return self._sampler.sample(mps, M, generator=generator)


class CuTensorNetMPSBackend:
    """Production GPU MPS backend for ``EnergyEstimator``.

    This is intentionally a thin wrapper around ``MPSNativeBackend``. The
    high-level backend already owns action decoding, terminal handling, gate
    order, and SWAP-network CNOT semantics; this class only swaps the low-level
    adjacent two-site gate split to cuTensorNet.
    """

    def __init__(
        self,
        n_qubits: int,
        action_map: Dict[int, Tuple],
        terminal_action: int,
        device: torch.device,
        *,
        max_bond_dim: Optional[int] = None,
        truncation_tol: float = 1e-10,
        contract_decompose_fn: Optional[ContractDecomposeFn] = None,
    ) -> None:
        if device.type != "cuda":
            raise ValueError(
                f"CuTensorNetMPSBackend requires a CUDA device; got {device}"
            )

        contract_decompose = (
            contract_decompose_fn
            if contract_decompose_fn is not None
            else _import_contract_decompose()
        )
        self.ops = CuTensorNetMPSOps(contract_decompose_fn=contract_decompose)
        self._delegate = MPSNativeBackend(
            n_qubits=n_qubits,
            action_map=action_map,
            terminal_action=terminal_action,
            device=device,
            ops=self.ops,
            max_bond_dim=max_bond_dim,
            truncation_tol=truncation_tol,
        )

        self.n_qubits = self._delegate.n_qubits
        self.action_map = self._delegate.action_map
        self.terminal_action = self._delegate.terminal_action
        self.device = self._delegate.device
        self.max_bond_dim = self._delegate.max_bond_dim
        self.truncation_tol = self._delegate.truncation_tol

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
        return self._delegate.sample_outcomes(
            mps_ground_state,
            batch_actions,
            batch_lengths,
            M,
            generator=generator,
        )

    @classmethod
    def is_available(cls, device: torch.device) -> bool:
        """Return True iff CUDA and cuTensorNet gate-split APIs are usable."""
        if device.type != "cuda":
            return False
        if not torch.cuda.is_available():
            return False
        try:
            _import_contract_decompose()
        except ImportError:
            return False
        return True
