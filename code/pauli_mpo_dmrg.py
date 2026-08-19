"""Pauli-sum -> MPO -> two-site DMRG ground state. Offline precompute path.

Backend: torch (CPU or CUDA via ``device="cuda"``). Eigensolver is a hand-rolled
Lanczos with full Gram-Schmidt re-orthogonalization. complex128 by default;
complex64 opt-in via ``compute_ground_state_dmrg(dtype=...)`` / ``_precision``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import torch
import torch.backends.opt_einsum as torch_opt_einsum

# ``effective_h_matvec_dense``'s 5-operand einsum relies on torch's opt_einsum
# integration for a good contraction order; a naive left-to-right order folds all
# four outer bond legs together last and materializes a chi^4*d^4 tensor (~64 GiB
# at chi=128). Fail fast at import rather than OOM deep in a Lanczos sweep. An
# explicit raise (not ``assert``) so ``python -O`` cannot strip the guard.
if not torch_opt_einsum.is_available():
    raise ImportError(
        "opt_einsum missing — pauli_mpo_dmrg's dense-boundary matvec would OOM "
        "(chi^4*d^4 ~ 64 GiB at chi=128); pip install opt_einsum"
    )
if hasattr(torch_opt_einsum, "enabled") and not torch_opt_einsum.enabled:
    torch_opt_einsum.enabled = True
if hasattr(torch_opt_einsum, "enabled") and not torch_opt_einsum.enabled:
    raise ImportError(
        "torch.backends.opt_einsum is disabled — pauli_mpo_dmrg's "
        "dense-boundary matvec requires the optimized contraction planner"
    )


# ---------------------------------------------------------------------------
# Constants

_CPX = torch.complex128
_REAL = torch.float64

PAULI_TENSORS: dict[str, torch.Tensor] = {
    "I": torch.tensor([[1, 0], [0, 1]], dtype=_CPX),
    "X": torch.tensor([[0, 1], [1, 0]], dtype=_CPX),
    "Y": torch.tensor([[0, -1j], [1j, 0]], dtype=_CPX),
    "Z": torch.tensor([[1, 0], [0, -1]], dtype=_CPX),
}


class _precision:
    """Temporarily set the working complex/real dtypes for one DMRG run.

    Every tensor created in this module reads the module-global ``_CPX`` /
    ``_REAL`` by name at call time (late binding), so swapping them here makes
    the entire run -- MPO build, MPS init, environments, Lanczos -- use the
    requested precision with no per-call dtype threading. ``complex64`` halves
    device memory AND is faster on GPU (better bandwidth); it costs
    single-precision accuracy, so validate energy parity for the target system
    before trusting a complex64 reference. Save/restore so nested/sequential
    runs compose cleanly.
    """

    def __init__(self, dtype: torch.dtype):
        if dtype not in (torch.complex128, torch.complex64):
            raise ValueError(
                f"dtype must be torch.complex128 or torch.complex64, got {dtype}"
            )
        self._cpx = dtype
        self._real = torch.float64 if dtype == torch.complex128 else torch.float32

    def __enter__(self):
        global _CPX, _REAL
        self._prev = (_CPX, _REAL)
        _CPX, _REAL = self._cpx, self._real
        return self

    def __exit__(self, *exc):
        global _CPX, _REAL
        _CPX, _REAL = self._prev
        return False


# ---------------------------------------------------------------------------
# MPO

@dataclass
class PauliMPO:
    """Sum-of-strings MPO for H = sum_k c_k * tensor_i sigma_{P_k^(i)}.

    Stores both the dense (K_in, K_out, d, d) tensors AND a channel-diagonal
    (K, d, d) form for interior sites. Sweep driver routes to the fast
    channel-diagonal matvec at interior bonds and falls back to dense at
    the two boundary 2-site blocks.

    Fields:
        W: list of L tensors, W[i].shape == (K_in_i, K_out_i, d, d).
           K_in_0 = K_out_{L-1} = 1; K_in_i = K_out_{i-1} = K otherwise.
        W_diag: list of length L. W_diag[i] is the (K, d, d) channel-diagonal
           extract for interior sites 1..L-2; None at i=0 and i=L-1.
        n_qubits: L
        K: number of non-identity Pauli terms
        identity_offset: real scalar (sum of identity-only term coefficients)
        device
        pauli_index: (L, K) int64 tensor, pauli_index[i, k] in {0,1,2,3}.
        fanout_coeffs: (K,) complex128 — the c_k applied at the left boundary.
    """

    W: list[torch.Tensor]
    W_diag: list[Optional[torch.Tensor]]
    n_qubits: int
    K: int
    identity_offset: complex
    device: torch.device
    pauli_index: torch.Tensor
    fanout_coeffs: torch.Tensor


def pauli_sum_to_mpo(
    pauli_strings: list[str],
    coefficients: list[complex],
    n_qubits: int,
    device: torch.device | str = "cpu",
    dense_interior: bool = False,
) -> PauliMPO:
    """Build a sum-of-strings MPO from Pauli strings + complex coefficients.

    By default, the interior MPO tensors are stored ONLY in channel-diagonal
    form (K, d, d) — the dense (K, K, d, d) form would cost K^2 memory per
    site and is never needed at runtime because env updates and matvecs
    have channel-diagonal fast paths.

    Pass ``dense_interior=True`` to additionally build the dense (K, K, d, d)
    interior tensors; this is only needed for ``mpo_to_dense()`` round-trip
    tests on small systems (n_qubits <= 14ish).

    Assumes the input has been combined into Hermitian canonical form (unique
    strings, real coefficients). Imaginary parts must be < 1e-10 of |coeff|.

    Identity-only terms ('I' * n_qubits) are split off as the scalar
    ``identity_offset``.

    Pauli strings are zero-indexed from the leftmost site. So 'XIZ' means
    sigma_X at site 0, sigma_I at site 1, sigma_Z at site 2.
    """
    if not pauli_strings:
        raise ValueError("Empty Pauli list")
    if len(pauli_strings) != len(coefficients):
        raise ValueError("Strings/coefficients length mismatch")

    dev = torch.device(device) if not isinstance(device, torch.device) else device
    d = 2
    _PAULI_KEY = "IXYZ"

    identity_str = "I" * n_qubits
    identity_offset = 0.0 + 0.0j
    non_id_strings: list[str] = []
    non_id_coeffs: list[complex] = []

    for s, c in zip(pauli_strings, coefficients):
        if len(s) != n_qubits:
            raise ValueError(f"Pauli string {s!r} length != n_qubits={n_qubits}")
        if any(ch not in _PAULI_KEY for ch in s):
            raise ValueError(f"Pauli string {s!r} has invalid character (must be I/X/Y/Z)")
        if s == identity_str:
            identity_offset += complex(c)
        else:
            non_id_strings.append(s)
            non_id_coeffs.append(complex(c))

    # Hermiticity check on non-identity terms (real coeffs)
    for c in non_id_coeffs:
        if abs(c.imag) > 1e-10 * max(abs(c), 1.0):
            raise ValueError(
                f"Coefficient {c} has imaginary part {c.imag:.3e} > 1e-10. "
                "Input must be Hermitian canonical form."
            )
    # Identity offset must be real too
    if abs(identity_offset.imag) > 1e-10 * max(abs(identity_offset), 1.0):
        raise ValueError(f"Identity offset {identity_offset} has imaginary part")

    K = len(non_id_strings)
    if K == 0:
        # Pure identity Hamiltonian. K=1 degenerate MPO that contributes 0.
        # We still need a valid MPO structure; build a trivial one with
        # K=1 and a zero coefficient.
        W = [torch.zeros((1, 1, d, d), dtype=_CPX, device=dev) for _ in range(n_qubits)]
        W_diag = [None] * n_qubits  # boundary handling for trivial MPO via dense path
        pauli_index = torch.zeros((n_qubits, 1), dtype=torch.long, device=dev)
        fanout = torch.zeros((1,), dtype=_CPX, device=dev)
        return PauliMPO(
            W=W, W_diag=W_diag, n_qubits=n_qubits, K=1,
            identity_offset=complex(identity_offset.real),
            device=dev, pauli_index=pauli_index, fanout_coeffs=fanout,
        )

    fanout = torch.tensor(non_id_coeffs, dtype=_CPX, device=dev)  # (K,) coeffs
    pauli_index = torch.zeros((n_qubits, K), dtype=torch.long, device=dev)
    for k, s in enumerate(non_id_strings):
        for i, ch in enumerate(s):
            pauli_index[i, k] = _PAULI_KEY.index(ch)

    # Build W tensors.
    # W[0]: (1, K, d, d) fan-out: W[0][0, k] = c_k * sigma_{P_k^(0)}.
    # W[i] for 0<i<L-1: dense (K, K, d, d) channel-diagonal if dense_interior,
    #                   else None — runtime uses W_diag.
    # W[L-1]: (K, 1, d, d) fan-in: W[L-1][k, 0] = sigma_{P_k^(L-1)}.
    # n_qubits == 1 is special: with no separate fan-in site the single site must
    # hold a closed (1, 1, d, d) tensor that already sums over channels, or a
    # 1-qubit sandwich with K>1 silently drops all but the k=0 channel.
    # ``dtype=_CPX`` (not just ``.to(dev)``) so an active complex64 precision
    # context is honored — PAULI_TENSORS is built once at import as complex128.
    pauli_stack = torch.stack(
        [PAULI_TENSORS[ch] for ch in _PAULI_KEY]
    ).to(device=dev, dtype=_CPX)  # (4, 2, 2)
    W: list[Optional[torch.Tensor]] = [None] * n_qubits
    if n_qubits == 1:
        site_paulis = pauli_stack[pauli_index[0]]  # (K, 2, 2)
        # H = sum_k c_k * sigma_{P_k}; close fan-out into a (1, 1, d, d) op.
        H_single = (fanout[:, None, None] * site_paulis).sum(dim=0)  # (d, d)
        W[0] = H_single.unsqueeze(0).unsqueeze(0)  # (1, 1, d, d)
    else:
        for i in range(n_qubits):
            site_paulis = pauli_stack[pauli_index[i]]  # (K, 2, 2)
            if i == 0:
                W[i] = (fanout[:, None, None] * site_paulis).unsqueeze(0)  # (1, K, d, d)
            elif i == n_qubits - 1:
                W[i] = site_paulis.unsqueeze(1)  # (K, 1, d, d)
            else:
                if dense_interior:
                    Wi = torch.zeros((K, K, d, d), dtype=_CPX, device=dev)
                    idx = torch.arange(K, device=dev)
                    Wi[idx, idx] = site_paulis
                    W[i] = Wi
                # else: leave as None — runtime path uses W_diag

    # Channel-diagonal form for interior sites: always built (cheap).
    W_diag: list[Optional[torch.Tensor]] = [None] * n_qubits
    for i in range(1, n_qubits - 1):
        W_diag[i] = pauli_stack[pauli_index[i]]  # (K, d, d)
    return PauliMPO(
        W=W,
        W_diag=W_diag,
        n_qubits=n_qubits,
        K=K,
        identity_offset=complex(identity_offset.real),
        device=dev,
        pauli_index=pauli_index,
        fanout_coeffs=fanout,
    )


def pauli_sum_to_mpo_compressed(
    pauli_strings: list[str],
    coefficients: list[complex],
    n_qubits: int,
    device: torch.device | str = "cpu",
) -> PauliMPO:
    """Build a COMPRESSED MPO via quimb's numeric SVD compression, instead of
    ``pauli_sum_to_mpo``'s naive one-channel-per-term construction (K =
    number of non-identity terms).

    Method: assemble the naive K-channel sum-of-strings MPO arrays, hand
    them to ``quimb.tensor.MatrixProductOperator``, and SVD-compress. SVD
    compression is numerically optimal per bond (a symbolic FSA compiler
    like TeNPy's MPOGraph is not), and measured better on the 6x6
    compact-Hubbard Hamiltonian: max bond 65 / mean 38.9 in the original
    qubit order and 60 / 37.5 after RCM reordering, vs TeNPy's 99 / 66 and
    the raw K=235. The array leg convention is pinned against a dense 5q
    reference with exact operator parity (|diff| ~ 7e-15) at cutoff 1e-12.
    This directly cuts the dominant ``chi^2 * K`` environment-tensor memory
    cost -- the biggest lever for reaching higher chi within a fixed GPU
    memory budget, independent of the ``dtype``/complex64 lever
    (``_precision``).

    Returns a ``PauliMPO`` in the SAME dense per-site (K, K, d, d) format the
    existing ``dense_interior=True`` path already produces, with
    ``W_diag = [None] * n_qubits`` for every site (not just the two
    boundaries). ``dmrg_sweeps``/``_make_matvec``/``_update_left_env_auto``/
    ``_update_right_env_auto`` already dispatch to the dense path whenever
    ``W_diag[i] is None`` (verified by direct code reading -- no changes
    needed there); the compressed MPO is not channel-diagonal in the
    interior (that's precisely what makes it compact), so every site here
    goes through the dense path rather than just the two boundary blocks.

    Per-bond dimensions after compression vary along the chain; they are
    zero-padded to a common ``Kmax`` (the max over interior bonds) so the
    uniform-K tensor format ``pauli_sum_to_mpo`` callers already expect
    still applies. Padding embeds the real (smaller) block in the top-left
    corner of a ``(Kmax, Kmax, d, d)`` zero tensor -- the extra channels are
    simply unpopulated ("dead"), which is exact (not approximate): a
    smaller-bond MPO is trivially embeddable in a larger-uniform-bond one.

    ``pauli_index``/``fanout_coeffs`` on the returned ``PauliMPO`` are
    dummy placeholders (zeros) -- confirmed by direct code reading that
    nothing downstream of ``pauli_sum_to_mpo`` reads these fields again;
    they exist on the dataclass only as construction-time bookkeeping for
    the sum-of-strings builder.
    """
    try:
        import quimb.tensor as qtn
    except ImportError as ex:
        raise ImportError(
            "pauli_sum_to_mpo_compressed requires quimb (pip install quimb) "
            "for MPO SVD compression."
        ) from ex

    if not pauli_strings:
        raise ValueError("Empty Pauli list")
    if len(pauli_strings) != len(coefficients):
        raise ValueError("Strings/coefficients length mismatch")
    if n_qubits == 1:
        # No bonds to compress; the sum-of-strings builder already returns
        # the closed (1, 1, d, d) single-site MPO.
        return pauli_sum_to_mpo(pauli_strings, coefficients, 1, device=device)

    dev = torch.device(device) if not isinstance(device, torch.device) else device
    d = 2
    _PAULI_KEY = "IXYZ"

    identity_str = "I" * n_qubits
    identity_offset = 0.0 + 0.0j
    non_id_strings: list[str] = []
    non_id_coeffs: list[float] = []
    for s, c in zip(pauli_strings, coefficients):
        if len(s) != n_qubits:
            raise ValueError(f"Pauli string {s!r} length != n_qubits={n_qubits}")
        if any(ch not in _PAULI_KEY for ch in s):
            raise ValueError(f"Pauli string {s!r} has invalid character (must be I/X/Y/Z)")
        cc = complex(c)
        if s == identity_str:
            identity_offset += cc
            continue
        if abs(cc.imag) > 1e-10 * max(abs(cc), 1.0):
            raise ValueError(
                f"Coefficient {cc} on non-identity term {s!r} has imaginary part "
                f"{cc.imag:.3e} > 1e-10. Input must be Hermitian canonical form."
            )
        non_id_strings.append(s)
        non_id_coeffs.append(cc.real)
    if abs(identity_offset.imag) > 1e-10 * max(abs(identity_offset), 1.0):
        raise ValueError(f"Identity offset {identity_offset} has imaginary part")

    if not non_id_strings:
        # Pure identity Hamiltonian -- degenerate MPO, same convention as
        # pauli_sum_to_mpo's K==0 branch.
        W = [torch.zeros((1, 1, d, d), dtype=_CPX, device=dev) for _ in range(n_qubits)]
        W_diag = [None] * n_qubits
        pauli_index = torch.zeros((n_qubits, 1), dtype=torch.long, device=dev)
        fanout = torch.zeros((1,), dtype=_CPX, device=dev)
        return PauliMPO(
            W=W, W_diag=W_diag, n_qubits=n_qubits, K=1,
            identity_offset=complex(identity_offset.real),
            device=dev, pauli_index=pauli_index, fanout_coeffs=fanout,
        )

    # Naive K-channel sum-of-strings arrays, (l, r, up, down) legs with the
    # trivial outer bonds dropped at the boundaries (quimb's convention);
    # leg semantics pinned by the dense-parity recon. quimb then
    # SVD-compresses the ragged bonds.
    np_pauli = {ch: PAULI_TENSORS[ch].numpy() for ch in _PAULI_KEY}
    Kraw = len(non_id_strings)
    raw = []
    for i in range(n_qubits):
        if i == 0:
            a = np.zeros((Kraw, d, d), dtype=complex)  # (r, u, d)
            for k, (s, c) in enumerate(zip(non_id_strings, non_id_coeffs)):
                a[k] = c * np_pauli[s[0]]
        elif i == n_qubits - 1:
            a = np.zeros((Kraw, d, d), dtype=complex)  # (l, u, d)
            for k, s in enumerate(non_id_strings):
                a[k] = np_pauli[s[-1]]
        else:
            a = np.zeros((Kraw, Kraw, d, d), dtype=complex)
            for k, s in enumerate(non_id_strings):
                a[k, k] = np_pauli[s[i]]
        raw.append(a)
    mpo_q = qtn.MatrixProductOperator(raw, shape="lrud")
    mpo_q.compress(cutoff=1e-12)

    # Extract per-site arrays back in (l, r, u, d) order by index NAME (the
    # tensors' axis order is not guaranteed after compression), reinstating
    # the trivial boundary bonds.
    arrs = []
    for i in range(n_qubits):
        t = mpo_q[i]
        left = mpo_q.bond(i - 1, i) if i > 0 else None
        right = mpo_q.bond(i, i + 1) if i < n_qubits - 1 else None
        order = [ix for ix in (left, right, f"k{i}", f"b{i}") if ix is not None]
        data = t.transpose(*order).data
        if left is None:
            data = data.reshape(1, *data.shape)
        if right is None:
            data = data.reshape(data.shape[0], 1, *data.shape[1:])
        arrs.append(np.ascontiguousarray(data))

    # Zero-pad every interior bond to a common Kmax so the uniform-K
    # (K, K, d, d) format applies at every site.
    interior_dims = [arrs[i].shape[1] for i in range(n_qubits - 1)]  # bonds 1..L-1
    K = max(interior_dims) if interior_dims else 1

    def pad(a: np.ndarray, left_k: int, right_k: int) -> np.ndarray:
        out = np.zeros((left_k, right_k, d, d), dtype=complex)
        out[: a.shape[0], : a.shape[1]] = a
        return out

    W: list[torch.Tensor] = []
    for i in range(n_qubits):
        left_k = 1 if i == 0 else K
        right_k = 1 if i == n_qubits - 1 else K
        padded = pad(arrs[i], left_k, right_k)
        W.append(torch.tensor(padded, dtype=_CPX, device=dev))

    W_diag = [None] * n_qubits
    pauli_index = torch.zeros((n_qubits, K), dtype=torch.long, device=dev)
    fanout = torch.zeros((K,), dtype=_CPX, device=dev)
    return PauliMPO(
        W=W,
        W_diag=W_diag,
        n_qubits=n_qubits,
        K=K,
        identity_offset=complex(identity_offset.real),
        device=dev,
        pauli_index=pauli_index,
        fanout_coeffs=fanout,
    )


def mpo_to_dense(mpo: PauliMPO) -> torch.Tensor:
    """Contract the full MPO chain into a dense (2**n, 2**n) matrix.

    For correctness testing only — the dense matrix is 2**L by 2**L
    complex128, so memory grows as 16 * 4**L bytes:
      L=12 → 256 MiB, L=13 → 1 GiB, L=14 → 4 GiB.
    The ceiling here is L=12 (~256 MiB) which is well within test-runner
    memory; bump it if you really need a larger reference, but be aware
    of the quadratic-in-2^L growth. Requires that the MPO was built with
    ``dense_interior=True``.
    """
    L = mpo.n_qubits
    if L > 12:
        raise ValueError(
            f"mpo_to_dense refuses n_qubits={L}: dense matrix is "
            f"{16 * (1 << (2*L)) / 1024**2:.0f} MiB. Use mps_mpo_expectation "
            f"or get_hamiltonian_matrix(sparse=True) instead."
        )
    if any(w is None for w in mpo.W):
        raise ValueError(
            "mpo_to_dense requires dense interior W tensors. "
            "Rebuild the MPO with pauli_sum_to_mpo(..., dense_interior=True)."
        )
    # Sequential contraction: M[i] of shape (left_bond, 2**i, 2**i, right_bond)
    M = mpo.W[0].clone()  # (1, K, 2, 2)
    for i in range(1, L):
        Wi = mpo.W[i]  # (K_in, K_out, d, d)
        a, b, S, T = M.shape
        Ki, Ko, d, _ = Wi.shape
        assert b == Ki, f"bond mismatch at site {i}: {b} vs {Ki}"
        Mnew = torch.einsum("abST, bcuv -> acSuTv", M, Wi)
        M = Mnew.reshape(a, Ko, S * d, T * d)
    assert M.shape[:2] == (1, 1), f"unexpected final bond shape {M.shape[:2]}"
    H = M[0, 0]
    dim = 1 << L
    H = H + mpo.identity_offset * torch.eye(dim, dtype=_CPX, device=mpo.device)
    return H


# ---------------------------------------------------------------------------
# MPS construction & canonicalization

def init_mps_neel(
    n_qubits: int,
    *,
    device: torch.device | str = "cpu",
    perturb: float = 0.0,
    seed: int | None = None,
) -> list[torch.Tensor]:
    """Product state |0101...> as MPS with bond dim 1, then right-canonicalize.

    Each tensor has shape (chi_L, d, chi_R). Initial bond dim is 1.
    DMRG sweeps grow the bond dim via two-site SVD.

    When ``seed`` is provided the random perturbation is generated from a
    LOCAL ``torch.Generator`` rather than ``torch.manual_seed`` — the latter
    would mutate the global RNG and silently re-seed every downstream
    ``torch.randn`` call after a DMRG run.
    """
    dev = torch.device(device) if not isinstance(device, torch.device) else device
    d = 2
    # Local generator so we never touch the global torch RNG state.
    gen: Optional[torch.Generator] = None
    if seed is not None:
        gen = torch.Generator(device=dev)
        gen.manual_seed(int(seed))
    mps: list[torch.Tensor] = []
    for i in range(n_qubits):
        t = torch.zeros((1, d, 1), dtype=_CPX, device=dev)
        t[0, i % 2, 0] = 1.0
        if perturb > 0:
            # Complex noise from the local generator (torch.randn doesn't
            # accept complex dtype, so build real+imag separately).
            re = torch.randn(t.shape, generator=gen, dtype=_REAL, device=dev)
            im = torch.randn(t.shape, generator=gen, dtype=_REAL, device=dev)
            t = t + perturb * (re.to(_CPX) + 1j * im.to(_CPX))
        mps.append(t)
    # Right-canonicalize (right-to-left QR)
    return right_canonicalize(mps)


def right_canonicalize(mps: list[torch.Tensor]) -> list[torch.Tensor]:
    """Bring MPS to right-canonical form via right-to-left QR.

    After this, every tensor M_i satisfies sum_{s, b} M_i[a, s, b] * conj(M_i[a', s, b]) = delta_{a, a'}
    (right-orthogonal). The orthogonality center ends up at site 0.
    """
    L = len(mps)
    for i in range(L - 1, 0, -1):
        a, d, b = mps[i].shape
        M = mps[i].reshape(a, d * b)
        # QR of M^T means M = (R^T)(Q^T). We want M = R * Q with Q right-orthogonal.
        # Equivalently: M^H * M = R^H * R (cholesky of Gram). Or use SVD.
        # Use QR on M^T: M^T = Q' * R', so M = R'^T * Q'^T.
        # Then M_new = Q'^T (right-orthogonal because Q'^T * Q'^*^T = I after sum), and
        # the R'^T factor is absorbed into site i-1.
        Mt = M.T.conj()  # (d*b, a)
        Q, R = torch.linalg.qr(Mt, mode="reduced")
        # M = (Q*R)^H = R^H Q^H; new tensor at site i is Q^H of shape (a_new, d, b)
        # where a_new = min(d*b, a).
        a_new = Q.shape[1]
        new_M_i = Q.T.conj().reshape(a_new, d, b)
        absorb = R.T.conj()  # (a, a_new)
        mps[i] = new_M_i
        # Absorb into site i-1: mps[i-1] gets right-multiplied by absorb along its right bond.
        prev = mps[i - 1]  # (a_prev, d_prev, a)
        ap, dp, _ = prev.shape
        mps[i - 1] = torch.einsum("xyz, zw -> xyw", prev, absorb)
    return mps


def left_canonicalize(mps: list[torch.Tensor]) -> list[torch.Tensor]:
    """Bring MPS to left-canonical form via left-to-right QR."""
    L = len(mps)
    for i in range(L - 1):
        a, d, b = mps[i].shape
        M = mps[i].reshape(a * d, b)
        Q, R = torch.linalg.qr(M, mode="reduced")
        b_new = Q.shape[1]
        mps[i] = Q.reshape(a, d, b_new)
        nxt = mps[i + 1]
        _, dn, bn = nxt.shape
        mps[i + 1] = torch.einsum("xy, yzw -> xzw", R, nxt)
    return mps


# ---------------------------------------------------------------------------
# MPS arithmetic (Pauli application / addition / compression) — the
# primitives behind stabilizer-code-space projection of an initial state.

def apply_pauli_string(mps: list[torch.Tensor], pauli_string: str) -> list[torch.Tensor]:
    """Return P|psi> for a single Pauli string P (new list; input not mutated).

    Single-site operators act on the physical leg only, so bond dimensions
    are unchanged. Site convention matches ``pauli_sum_to_mpo``:
    ``pauli_string[q]`` is the Pauli on qubit q.
    """
    if len(pauli_string) != len(mps):
        raise ValueError(
            f"Pauli string length {len(pauli_string)} != MPS length {len(mps)}"
        )
    out = []
    for q, (ch, t) in enumerate(zip(pauli_string, mps)):
        if ch == "I":
            out.append(t)
            continue
        if ch not in PAULI_TENSORS:
            raise ValueError(f"invalid Pauli character {ch!r} at position {q}")
        P = PAULI_TENSORS[ch].to(device=t.device, dtype=t.dtype)
        out.append(torch.einsum("st, atb -> asb", P, t))
    return out


def mps_add(
    mps_a: list[torch.Tensor],
    mps_b: list[torch.Tensor],
    coeff_a: complex = 1.0,
    coeff_b: complex = 1.0,
) -> list[torch.Tensor]:
    """Return coeff_a*|a> + coeff_b*|b> as a block direct-sum MPS.

    Bond dimensions add (chi_a + chi_b per bond); compress afterwards with
:func:`mps_compress`. The coefficients are folded into site 0.
    """
    L = len(mps_a)
    if len(mps_b) != L:
        raise ValueError(f"MPS length mismatch: {L} vs {len(mps_b)}")
    if L == 1:
        return [coeff_a * mps_a[0] + coeff_b * mps_b[0]]
    out = []
    for i, (A, B) in enumerate(zip(mps_a, mps_b)):
        al, d, ar = A.shape
        bl, _, br = B.shape
        if i == 0:
            # (1, d, ar+br): concatenate along the right bond.
            out.append(torch.cat([coeff_a * A, coeff_b * B], dim=2))
        elif i == L - 1:
            # (al+bl, d, 1): concatenate along the left bond.
            out.append(torch.cat([A, B], dim=0))
        else:
            blk = torch.zeros((al + bl, d, ar + br), dtype=A.dtype, device=A.device)
            blk[:al, :, :ar] = A
            blk[al:, :, ar:] = B
            out.append(blk)
    return out


def mps_compress(
    mps: list[torch.Tensor],
    chi_max: int,
    svd_min: float = 1e-10,
) -> tuple[list[torch.Tensor], float]:
    """SVD-truncate an MPS to bond dimension <= chi_max.

    Right-canonicalizes (orthogonality center at site 0) then sweeps
    left-to-right with two-site SVD truncation — the right-orthogonal
    environment makes each local truncation optimal. Returns
    (compressed_mps, max_trunc_err). Norm is preserved up to truncation.
    """
    mps = right_canonicalize(list(mps))
    L = len(mps)
    max_err = 0.0
    for i in range(L - 1):
        T = torch.einsum("asx, xtb -> astb", mps[i], mps[i + 1])
        U, S, Vh, err = two_site_svd(T, chi_max, svd_min)
        max_err = max(max_err, err)
        mps[i] = U
        mps[i + 1] = torch.einsum("c, ctb -> ctb", S.to(Vh.dtype), Vh)
    return mps, max_err


def project_onto_stabilizers(
    mps: list[torch.Tensor],
    stabilizer_strings: list[str],
    *,
    chi_max: int = 256,
    svd_min: float = 1e-10,
    truncation_tol: float = 1e-12,
    renormalize: bool = True,
) -> list[torch.Tensor]:
    """Apply the code-space projector  P = prod_i (I + S_i)/2  to an MPS.

    The compact/toric-code-dressed encoding's ground state lives in the +1
    eigenspace of all stabilizers; DMRG started from a bare product state
    must *rotate* into that space through local two-site updates, which
    stalls (measured: <S>_mean plateaus ~0.89 on the 6x6/54q system, ~6 Ha
    of pure penalty leakage). Projecting the initial state instead makes
    every <S_i> = +1 exactly from sweep 0; H commutes with the stabilizers,
    so sweeps then optimize within the code space (truncation drift is
    self-correcting via the penalty).

    A projected computational-basis product state is a stabilizer state
    whose exact bond dimension is bounded (~2^(strip width)), so chi_max is
    a safety cap, not an approximation knob — for the 6x6 lattice the true
    chi stays well under 256.

    Raises if the input state is annihilated by the projector (norm ~ 0):
    that means the product state sits in a -1 eigensector of some stabilizer
    subgroup element; pick a different product state.
    """
    for lab in stabilizer_strings:
        s_psi = apply_pauli_string(mps, lab)
        mps = mps_add(mps, s_psi, 0.5, 0.5)
        mps, trunc_err = mps_compress(mps, chi_max, svd_min)
        if trunc_err > truncation_tol:
            raise ValueError(
                f"projector (I + {lab})/2 truncated the projected state "
                f"(discarded norm^2={trunc_err:.3e} > {truncation_tol:.3e}); "
                f"increase chi_max or relax truncation_tol only if an approximate "
                f"projected initial state is intended"
            )
        norm2 = mps_norm_squared(mps)
        if norm2 < 1e-12:
            raise ValueError(
                f"projector (I + {lab})/2 annihilated the state (norm^2="
                f"{norm2:.2e}); the initial product state is incompatible "
                f"with this stabilizer — choose a different product state"
            )
        if renormalize:
            mps[0] = mps[0] / math.sqrt(norm2)
    return mps


# ---------------------------------------------------------------------------
# Environments

def trivial_env(K: int, device: torch.device) -> torch.Tensor:
    """1x1x1 boundary env at the very edge. Only the (0,0,0) entry is 1."""
    e = torch.zeros((1, 1, 1), dtype=_CPX, device=device)
    e[0, 0, 0] = 1.0
    return e


def update_left_env(
    L_env: torch.Tensor,   # (a_left, m_left, a_left')
    psi_i: torch.Tensor,   # (a_left, d, a_right)
    W_i: torch.Tensor,     # (m_left, m_right, d, d)
) -> torch.Tensor:
    """Advance left env past site i (dense MPO). Returns (a_right, m_right, a_right')."""
    return torch.einsum("amA, asb, mnst, AtB -> bnB",
                        L_env, psi_i.conj(), W_i, psi_i)


def update_right_env(
    R_env: torch.Tensor,   # (b_right, m_right, b_right')
    psi_i: torch.Tensor,   # (b_left, d, b_right)
    W_i: torch.Tensor,     # (m_left, m_right, d, d)
) -> torch.Tensor:
    """Advance right env past site i (dense MPO, right-to-left). Returns (b_left, m_left, b_left')."""
    return torch.einsum("bnB, asb, mnst, AtB -> amA",
                        R_env, psi_i.conj(), W_i, psi_i)


def update_left_env_channel_diag(
    L_env: torch.Tensor,       # (a, m, A)  — m is the channel index
    psi_i: torch.Tensor,       # (a, s, b)
    W_i_diag: torch.Tensor,    # (m, s, s')  channel-diagonal at site i
) -> torch.Tensor:             # (b, m, B)
    """Advance left env at an interior site using the channel-diagonal MPO form.

    Cost: O(chi^3 * K * d) — saves a factor of K vs. dense env update.
    """
    return torch.einsum("amA, asb, mst, AtB -> bmB",
                        L_env, psi_i.conj(), W_i_diag, psi_i)


def update_right_env_channel_diag(
    R_env: torch.Tensor,       # (b, m, B)
    psi_i: torch.Tensor,       # (a, s, b)
    W_i_diag: torch.Tensor,    # (m, s, s')
) -> torch.Tensor:             # (a, m, A)
    """Advance right env at an interior site using the channel-diagonal MPO form."""
    return torch.einsum("bmB, asb, mst, AtB -> amA",
                        R_env, psi_i.conj(), W_i_diag, psi_i)


def _update_left_env_auto(
    L_env: torch.Tensor,
    psi_i: torch.Tensor,
    W_i_dense: Optional[torch.Tensor],
    W_i_diag: Optional[torch.Tensor],
) -> torch.Tensor:
    """Dispatch: channel-diagonal if available, else dense."""
    if W_i_diag is not None:
        return update_left_env_channel_diag(L_env, psi_i, W_i_diag)
    assert W_i_dense is not None, "neither dense nor diagonal MPO available at this site"
    return update_left_env(L_env, psi_i, W_i_dense)


def _update_right_env_auto(
    R_env: torch.Tensor,
    psi_i: torch.Tensor,
    W_i_dense: Optional[torch.Tensor],
    W_i_diag: Optional[torch.Tensor],
) -> torch.Tensor:
    if W_i_diag is not None:
        return update_right_env_channel_diag(R_env, psi_i, W_i_diag)
    assert W_i_dense is not None
    return update_right_env(R_env, psi_i, W_i_dense)


class LazyRightEnvs:
    """Checkpoint-based lazy R_env cache for forward DMRG sweeps.

    Builds R_env at sqrt(L) checkpoint positions; rebuilds intermediate
    positions on demand by walking down from the nearest checkpoint AHEAD
    of the requested position.

    Memory: ~sqrt(L) + 1 env tensors instead of L (eager).
    Compute: average ~sqrt(L)/2 extra ``update_right_env`` calls per forward
    sweep step (vs. eager: 0). Overhead is bounded by max gap between
    checkpoints.

    Use when ``L * chi^2 * K * 16 B`` would blow the device memory budget.
    For Hubbard 8x8 at chi=800, K=400: eager needs 262 GB; sliding with
    sqrt(64)=8 checkpoints needs ~33 GB.

    NOTE: assumes the MPS is static during forward sweep, which is the
    case for two-site DMRG (forward modifies sites i and i+1 only;
    R_envs depend on mps[i+2..L-1], unchanged). Caller must rebuild the
    LazyRightEnvs after each full forward sweep before another forward
    sweep can use it.
    """

    def __init__(self, mps: list[torch.Tensor], mpo: PauliMPO,
                 n_checkpoints: Optional[int] = None):
        L = len(mps)
        if n_checkpoints is None:
            n_checkpoints = max(2, int(math.sqrt(L)))
        # Checkpoint stride. We always have R_env[L] (trivial); add more
        # every ``step`` positions walking from L down to 1.
        step = max(1, L // n_checkpoints)
        self.checkpoints: dict[int, torch.Tensor] = {L: trivial_env(mpo.K, mpo.device)}
        current = self.checkpoints[L]
        for pos in range(L - 1, 0, -1):
            current = _update_right_env_auto(
                current, mps[pos], mpo.W[pos], mpo.W_diag[pos]
            )
            if (L - pos) % step == 0:
                self.checkpoints[pos] = current
        # Ensure pos=2 (smallest needed by forward sweep at site i=0) is
        # reachable via some checkpoint > 2.
        self.mps = mps
        self.mpo = mpo
        self.L = L

    def get(self, pos: int) -> torch.Tensor:
        """Return R_env at position ``pos`` (env to the right of site pos-1)."""
        if pos in self.checkpoints:
            return self.checkpoints[pos]
        # Find smallest checkpoint position > pos
        for p in sorted(self.checkpoints.keys()):
            if p > pos:
                nearest = p
                break
        else:
            raise ValueError(f"No checkpoint AHEAD of pos={pos}; L={self.L}, "
                             f"checkpoints={sorted(self.checkpoints.keys())}")
        # Walk down from `nearest` to `pos`
        env = self.checkpoints[nearest]
        for p in range(nearest - 1, pos - 1, -1):
            env = _update_right_env_auto(
                env, self.mps[p], self.mpo.W[p], self.mpo.W_diag[p]
            )
        return env

    @property
    def memory_estimate_bytes(self) -> int:
        """Total bytes held in checkpoints (complex128)."""
        return sum(t.element_size() * t.numel() for t in self.checkpoints.values())


def build_right_environments(
    mps: list[torch.Tensor],
    mpo: PauliMPO,
) -> list[torch.Tensor]:
    """Build R_env[L], R_env[L-1],..., R_env[1] eagerly.

    Dispatches between dense and channel-diagonal env updates per site.
    R_env[L] is the trivial right boundary (1, 1, 1).
    Returns list of length L+1 where R[i] = env to the right of site i-1.

    Use ``LazyRightEnvs`` instead when the eager memory budget (L * chi^2
    * K * 16 B) is too large.
    """
    L = len(mps)
    R = [None] * (L + 1)
    R[L] = trivial_env(mpo.K, mpo.device)
    for i in range(L - 1, 0, -1):
        R[i] = _update_right_env_auto(R[i + 1], mps[i], mpo.W[i], mpo.W_diag[i])
    # R[0] is never used downstream (the leftmost 2-site block uses L_env[0]
    # which is the trivial boundary). Skip it —
    # (nit: dead R[0] computation).
    return R


# ---------------------------------------------------------------------------
# Effective-H matvec (dense MPO)

def effective_h_matvec_dense(
    T: torch.Tensor,        # (a, s, t, b)
    L_env: torch.Tensor,    # (a, m, a')
    W_i: torch.Tensor,      # (m, n, s, s')
    W_ip1: torch.Tensor,    # (n, p, t, t')
    R_env: torch.Tensor,    # (b, p, b')
) -> torch.Tensor:          # (a, s, t, b)
    """Two-site effective Hamiltonian matvec using dense MPO tensors.

    Output shape matches T. Cost: 2*chi^3*K*d^2 + 2*chi^2*K^2*d^3 flops.
    """
    # Relies on the module-level opt_einsum assert above to avoid a naive
    # left-to-right contraction order; without it this materializes a
    # chi^4*d^4 tensor (~64 GiB at chi=128) and OOMs. Do not remove the assert.
    return torch.einsum(
        "amA, mnsS, nptT, bpB, ASTB -> astb",
        L_env, W_i, W_ip1, R_env, T,
    )


def effective_h_matvec_channel_diag(
    T: torch.Tensor,            # (a, s, t, b)
    L_env: torch.Tensor,        # (a, m, a') — m is the channel index (size K)
    W_i_diag: torch.Tensor,     # (K, s, s') — channel-diagonal: W[k,k',s,s'] = delta(k,k')*W_diag[k,s,s']
    W_ip1_diag: torch.Tensor,   # (K, t, t')
    R_env: torch.Tensor,        # (b, m, b') — same m as L_env (single shared channel)
) -> torch.Tensor:              # (a, s, t, b)
    """Channel-diagonal matvec. Interior sites only.

    For sum-of-strings MPO, the K x K bond block is diagonal:
    W[k, k', s, s'] = delta_{k,k'} * sigma_{P_k^(i)}[s, s'].
    All three internal MPO bonds (left of i, between i and i+1, right of
    i+1) collapse to a single channel index k.

    Cost: 2*chi^3*K*d^2 + 2*chi^2*K*d^3 flops -- saves the chi^2*K^2*d^3
    middle-step terms vs. dense.

    Speedup vs. dense at our chi/K ratios:
        chi=200, K=89:  ~1.88x
        chi=400, K=200: ~1.5x
        chi=400, K=300: ~2.5x
    """
    return torch.einsum(
        "amA, msS, mtT, bmB, ASTB -> astb",
        L_env, W_i_diag, W_ip1_diag, R_env, T,
    )


def extract_channel_diag(W: torch.Tensor, *, check: bool = True) -> torch.Tensor:
    """Extract the (K, d, d) diagonal from an interior MPO tensor (K, K, d, d).

    Assumes the input IS channel-diagonal (off-diagonal entries are zero).
    Used when converting a dense PauliMPO for channel-diagonal matvec.

    ``check=True`` (default): defensively assert the off-diagonal is zero,
    catching silent misuse on a non-diagonal MPO at ~zero runtime cost on
    the rare boundary path. Set ``check=False`` in hot loops.
    """
    K1, K2, d1, d2 = W.shape
    assert K1 == K2, f"expected square bond, got ({K1}, {K2})"
    idx = torch.arange(K1, device=W.device)
    diag = W[idx, idx]  # (K, d, d)
    if check:
        # Build the diagonal-only reconstruction and verify match.
        # Cost: one K^2 d^2 array allocation; only used at MPO setup.
        recon = torch.zeros_like(W)
        recon[idx, idx] = diag
        max_off = (W - recon).abs().max().item()
        ref = max(W.abs().max().item(), 1.0)
        if max_off > 1e-10 * ref:
            raise ValueError(
                f"extract_channel_diag called on non-diagonal MPO: "
                f"max off-diagonal magnitude {max_off:.3e} > 1e-10 * {ref:.3e}"
            )
    return diag


# ---------------------------------------------------------------------------
# Lanczos

def lanczos_smallest(
    matvec: Callable[[torch.Tensor], torch.Tensor],
    v0: torch.Tensor,
    *,
    k: int = 30,
    tol: float | None = None,
    full_reorth: bool = True,
) -> tuple[float, torch.Tensor, int]:
    """k-step Lanczos with full Gram-Schmidt re-orthogonalization.

    ``tol`` (None = precision-aware auto, keyed on ``v0.dtype``): the
    Ritz-convergence gate. A fixed 1e-10 relative gate sits below the float32
    noise floor (~1e-6 relative) and can NEVER fire for complex64 inputs, so
    every call would burn all ``k`` matvecs — the single dominant waste in
    c64 GPU runs. Auto: 1e-10 for double precision, 1e-6 for single.

    Returns (eigval, eigvec, n_iter). eigvec has the same shape as v0.
    """
    is_double = v0.dtype in (torch.complex128, torch.float64)
    if tol is None:
        tol = 1e-10 if is_double else 1e-6
    # Breakdown gate scaled to the working precision: 1e-14 is unreachable in
    # float32, where ||w|| after orthogonalizing a near-eigenvector bottoms
    # out at ~1e-6*scale — continuing past that point normalizes noise into a
    # garbage Krylov direction instead of terminating on the (converged)
    # invariant subspace.
    beta_floor = 1e-14 if is_double else 1e-6
    shape = v0.shape
    v = v0.flatten()
    v = v / torch.linalg.vector_norm(v)
    V = [v]                  # Krylov basis (flattened vectors)
    alphas: list[float] = []
    betas: list[float] = []
    n_iter = 0

    for j in range(k):
        w = matvec(v.reshape(shape)).flatten()
        if j > 0:
            w = w - betas[-1] * V[-2]
        alpha = torch.vdot(v, w).real.item()
        alphas.append(alpha)
        w = w - alpha * v
        if full_reorth:
            # Re-orthogonalize against all prior basis vectors
            for u in V:
                w = w - torch.vdot(u, w) * u
        beta = torch.linalg.vector_norm(w).real.item()
        n_iter = j + 1
        # Build tridiagonal and check convergence
        if j >= 1:
            T_tri = _build_tridiag(alphas, betas)
            eigvals = torch.linalg.eigvalsh(T_tri)
            if j >= 2:
                # Use scale-relative
                # tolerance so the gate doesn't trip on noise when |E| is
                # near zero (large Hubbard energies are O(10^1), but
                # arbitrary subsystem expectations can be near 0).
                prev_T = _build_tridiag(alphas[:-1], betas[:-1])
                prev_eig = torch.linalg.eigvalsh(prev_T)[0].item()
                scale = max(abs(eigvals[0].item()), abs(prev_eig), 1.0)
                if abs(eigvals[0].item() - prev_eig) < tol * scale:
                    break
        if beta < beta_floor or j == k - 1:
            break
        betas.append(beta)
        v = w / beta
        V.append(v)

    # Final eigendecomposition
    T_tri = _build_tridiag(alphas, betas)
    eigvals, eigvecs = torch.linalg.eigh(T_tri)
    smallest = eigvals[0].item()
    # Build eigenvector in original space
    coeffs = eigvecs[:, 0]  # (n_iter,)
    eigvec_flat = torch.zeros_like(V[0])
    for c, u in zip(coeffs, V):
        eigvec_flat = eigvec_flat + c.to(eigvec_flat.dtype) * u
    eigvec_flat = eigvec_flat / torch.linalg.vector_norm(eigvec_flat)
    return smallest, eigvec_flat.reshape(shape), n_iter


def _build_tridiag(alphas: list[float], betas: list[float]) -> torch.Tensor:
    # Always float64: alphas/betas are Python floats (exact here), the matrix
    # is tiny (k<=30) and CPU-side, and a float32 tridiag (under a complex64
    # _precision context) would add ~1e-6-relative eigensolve noise on top of
    # the matvec noise, making the Lanczos convergence gate unreliable.
    n = len(alphas)
    T = torch.zeros((n, n), dtype=torch.float64)
    for i in range(n):
        T[i, i] = alphas[i]
        if i < n - 1 and i < len(betas):
            T[i, i + 1] = betas[i]
            T[i + 1, i] = betas[i]
    return T


# ---------------------------------------------------------------------------
# Two-site update (SVD with chi_max truncation)

def two_site_svd(
    T: torch.Tensor,        # (a, d, d, b)
    chi_max: int,
    svd_min: float = 1e-10,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """SVD-truncate T into (U, S, Vh, trunc_err).

    Shapes:
        U:   (a, d, chi_new)
        S:   (chi_new,)
        Vh:  (chi_new, d, b)
        trunc_err: scalar (sum of squared discarded singular values)
    """
    a, d1, d2, b = T.shape
    M = T.reshape(a * d1, d2 * b)
    U, S, Vh = torch.linalg.svd(M, full_matrices=False)
    # Truncate
    keep = min(chi_max, S.shape[0])
    # Drop tiny singular values
    keep_mask = S[:keep] > svd_min * S[0] if S[0] > 0 else torch.zeros(keep, dtype=torch.bool, device=S.device)
    if keep_mask.sum().item() == 0:
        keep_eff = 1
    else:
        keep_eff = max(int(keep_mask.sum().item()), 1)
    trunc_err = float((S[keep_eff:] ** 2).sum().real.item()) if keep_eff < S.shape[0] else 0.0
    U_t = U[:, :keep_eff].reshape(a, d1, keep_eff)
    S_t = S[:keep_eff]
    Vh_t = Vh[:keep_eff].reshape(keep_eff, d2, b)
    return U_t, S_t, Vh_t, trunc_err


# ---------------------------------------------------------------------------
# DMRG sweep driver

@dataclass
class DMRGResult:
    energy: float          # excludes identity offset
    mps: list[torch.Tensor]
    converged: bool
    n_sweeps: int
    final_chi: int
    final_trunc_err: float
    energy_history: list[float]


def _diag_to_dense(W_diag: torch.Tensor) -> torch.Tensor:
    """(K, d, d) channel-diagonal -> (K, K, d, d) dense with delta_{kk'} structure."""
    K = W_diag.shape[0]
    dense = torch.zeros((K, K, *W_diag.shape[1:]), dtype=W_diag.dtype, device=W_diag.device)
    idx = torch.arange(K, device=W_diag.device)
    dense[idx, idx] = W_diag
    return dense


def _make_matvec(
    Le: torch.Tensor,
    Re: torch.Tensor,
    W_i_dense: Optional[torch.Tensor],
    W_ip1_dense: Optional[torch.Tensor],
    W_i_diag: Optional[torch.Tensor],
    W_ip1_diag: Optional[torch.Tensor],
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Pick the right matvec for the two-site block.

    Fast paths:
      - both sites interior (both W_diag set): channel-diagonal matvec
      - both sites boundary (both W_dense set, no W_diag): dense matvec
    Mixed (boundary + interior): reconstruct the missing dense W on demand
    from W_diag. The mixed case only happens at the two outermost 2-site
    blocks (i=0 with site 1 interior; i=L-2 with site L-2 interior), so
    the K^2 allocation is per sweep, not per matvec call.
    """
    if W_i_diag is not None and W_ip1_diag is not None:
        # All-interior fast path
        def mv(x):
            return effective_h_matvec_channel_diag(x, Le, W_i_diag, W_ip1_diag, Re)
        return mv
    # Mixed-boundary: materialize dense for the interior site(s).
    Wi = W_i_dense if W_i_dense is not None else _diag_to_dense(W_i_diag)
    Wp1 = W_ip1_dense if W_ip1_dense is not None else _diag_to_dense(W_ip1_diag)
    def mv(x):
        return effective_h_matvec_dense(x, Le, Wi, Wp1, Re)
    return mv


def _autopick_env_cache(L: int, bond_dim: int, K: int, *,
                        free_bytes: int, eager_threshold: float = 0.4) -> str:
    """Choose env caching strategy based on memory.

    Eager (``2*L*chi^2*K*sizeof(complex)``) is fastest but memory-intensive. Sliding
    (``sqrt(L)*chi^2*K*sizeof(complex)``) is ~sqrt(L)x slower per forward sweep step
    on the env-update path but uses far less peak memory.

    Returns 'all' if eager fits in ``eager_threshold * free_bytes``;
    'sliding' otherwise.
    """
    cpx_bytes = 16 if _CPX == torch.complex128 else 8
    # ``all`` keeps both eager L_envs and the full R_env_list resident.
    eager_bytes = 2 * L * bond_dim * bond_dim * K * cpx_bytes
    return "all" if eager_bytes < eager_threshold * free_bytes else "sliding"


def dmrg_sweeps(
    mpo: PauliMPO,
    *,
    bond_dim: int = 50,
    chi_list: dict[int, int] | None = None,
    max_sweeps: int = 30,
    energy_tol: float = 1e-9,
    svd_min: float = 1e-10,
    initial_state: str = "neel",
    initial_mps: Optional[list[torch.Tensor]] = None,
    lanczos_k: int = 30,
    lanczos_tol: float | None = None,
    use_channel_diagonal: bool = True,
    env_cache: str = "auto",
    seed: int | None = 0,
    verbose: bool = False,
    stall_tol: float = 0.0,
    stall_patience: int = 0,
) -> DMRGResult:
    """Two-site DMRG sweeps.

    ``initial_mps``: warm-start from an existing MPS (e.g. the previous rung
    of a chi ladder) instead of the Neel product state. The tensors are moved
    to the MPO's device/dtype and right-canonicalized. When warm-starting and
    ``chi_list`` is None, the ramp is skipped (``{0: bond_dim}``) — the
    default ramp would truncate a high-chi input back to ``bond_dim // 4`` on
    sweep 0 and destroy the warm start. Overrides ``initial_state``.

    ``lanczos_tol`` (None = precision-aware auto): the per-block Lanczos
    Ritz-convergence gate. A fixed 1e-10 relative gate sits below the float32
    noise floor (~1e-6 relative) and can never fire under complex64, so every
    two-site block would burn the full ``lanczos_k`` matvecs. Auto: 1e-10 for
    complex128, 1e-6 for complex64.

    ``stall_tol``/``stall_patience``: optional early termination for runs
    that flat-line far above ``energy_tol`` (truncation noise floor).
    When ``stall_patience > 0`` and chi has reached ``bond_dim``, the sweep
    loop stops after ``stall_patience`` CONSECUTIVE sweeps with
    ``|dE| < stall_tol * max(|E|, 1)``. This does NOT set ``converged``
    (a stall at trunc~1e-4 is not convergence); it just stops burning
    sweeps that no longer move the energy. Disabled by default (0).

    Convergence triggers only after current chi has reached ``bond_dim``
    (i.e., past the ramp).

    ``use_channel_diagonal`` (default True) routes interior matvecs to the
    channel-diagonal fast path; set False for regression-debugging — note
    that the dense fallback requires the MPO to have been built with
    ``dense_interior=True``, otherwise the W interior tensors are ``None``
    and the matvec construction would fail.

    ``env_cache`` controls the right-environment caching strategy:
      - ``'all'``: eager build of all L R_envs (fastest, ~L*chi^2*K*16 bytes)
      - ``'sliding'``: ~sqrt(L) checkpointed R_envs, rebuilt on demand
        (saves ~sqrt(L)x memory; pays ~sqrt(L) extra env updates per
        forward sweep step; enables higher chi 6+ systems)
      - ``'auto'`` (default): pick based on device memory budget — eager
        if it fits in ~40% of free memory, sliding otherwise.
    Backward sweep is always incremental (one R_env at a time) regardless
    of mode — it rebuilds R_envs from R_env[L]=trivial as it walks left.
    """
    L = mpo.n_qubits
    if not use_channel_diagonal:
        # Dense fallback path needs the interior dense W tensors. The default
        # ``pauli_sum_to_mpo(..., dense_interior=False)`` leaves W[i] as None
        # for interior sites — fail fast instead of crashing inside
        # ``_diag_to_dense(None)`` deep in the matvec construction.
        #
        if L >= 3 and any(mpo.W[i] is None for i in range(1, L - 1)):
            raise ValueError(
                "dmrg_sweeps(use_channel_diagonal=False) requires the MPO to "
                "have dense interior tensors. Rebuild the MPO with "
                "pauli_sum_to_mpo(..., dense_interior=True)."
            )
    # The MPO's dtype and the active precision context must agree: every tensor this
    # function creates uses the module-global ``_CPX``, so an MPO built under a
    # different precision context would type-promote every einsum to the wider dtype
    # (defeating complex64's memory halving) or crash mid-sweep. Fail fast instead.
    mpo_dtype = mpo.W[0].dtype
    if mpo_dtype != _CPX:
        raise ValueError(
            f"MPO dtype {mpo_dtype} != active precision {_CPX}. Build the MPO "
            f"and call dmrg_sweeps under the SAME _precision(dtype) context "
            f"(or use compute_ground_state_dmrg(dtype=...), which wraps both)."
        )
    if stall_patience > 0 and stall_tol <= 0:
        raise ValueError(
            f"stall_patience={stall_patience} requires stall_tol > 0 (got "
            f"{stall_tol}); the gate compares |dE| < stall_tol*scale and can "
            f"never fire at 0, silently burning max_sweeps."
        )
    if initial_mps is not None and chi_list is not None:
        raise ValueError(
            "initial_mps cannot be combined with an explicit chi_list: a ramp "
            "whose early caps sit below the warm-started MPS's bond dimension "
            "would truncate away the previous rung's work on sweep 0. Omit "
            "chi_list (warm starts run at full bond_dim from sweep 0)."
        )
    if chi_list is None:
        # The chi ramp must never exceed bond_dim, so for ``bond_dim < 16`` skip the
        # ramp entirely (it would otherwise inflate chi above the requested cap).
        # Warm starts skip it too: sweeping a warm-started MPS at a chi cap below
        # its own bond dim would truncate away the previous rung's work.
        if initial_mps is not None or bond_dim < 16:
            chi_list = {0: bond_dim}
        else:
            chi_list = {0: max(8, bond_dim // 4),
                        2: max(16, bond_dim // 2),
                        4: bond_dim}
    # Build initial MPS
    if initial_mps is not None:
        if len(initial_mps) != L:
            raise ValueError(
                f"initial_mps has {len(initial_mps)} tensors but the MPO has "
                f"{L} sites"
            )
        for i, t in enumerate(initial_mps):
            if t.ndim != 3 or t.shape[1] != 2:
                raise ValueError(
                    f"initial_mps[{i}] has shape {tuple(t.shape)}; expected "
                    f"(chi_L, 2, chi_R)"
                )
        mps = [t.to(device=mpo.device, dtype=_CPX) for t in initial_mps]
        mps = right_canonicalize(mps)
    elif initial_state == "neel":
        mps = init_mps_neel(L, device=mpo.device, perturb=1e-3, seed=seed)
    else:
        raise ValueError(f"Unknown initial_state={initial_state!r}")

    # Resolve env_cache
    if env_cache == "auto":
        try:
            if mpo.device.type == "cuda":
                free_bytes, _ = torch.cuda.mem_get_info(mpo.device)
            else:
                import psutil
                free_bytes = psutil.virtual_memory().available
        except Exception:
            free_bytes = 8 * 1024 ** 3
        env_cache = _autopick_env_cache(L, bond_dim, mpo.K, free_bytes=free_bytes)
        if verbose:
            print(f"[dmrg_sweeps] env_cache resolved to {env_cache!r}")
    if env_cache not in ("all", "sliding"):
        raise ValueError(f"env_cache must be 'all', 'sliding', or 'auto'; got {env_cache!r}")

    L_env = [trivial_env(mpo.K, mpo.device)] + [None] * L  # L_env[i] = env left of site i

    energy_history: list[float] = []
    current_chi = chi_list[0]
    prev_E = float("inf")
    converged = False
    final_trunc_err = 0.0
    stall_count = 0

    for sweep_n in range(max_sweeps):
        # Update chi according to ramp. When ``current_chi`` first reaches
        # ``bond_dim``, RESET ``prev_E`` to inf so the first full-chi sweep's
        # convergence check doesn't compare against a stale sub-cap sweep's
        # energy: that comparison spans a huge delta and never fires,
        # wasting one full-chi sweep.
        if sweep_n in chi_list:
            new_chi = chi_list[sweep_n]
            if new_chi >= bond_dim and current_chi < bond_dim:
                prev_E = float("inf")
            current_chi = new_chi

        # ---- Build right-env provider for this forward sweep ----
        if env_cache == "all":
            R_env_list = build_right_environments(mps, mpo)
            def _r_env_at(pos: int) -> torch.Tensor:
                return R_env_list[pos]
        else:
            lazy_r = LazyRightEnvs(mps, mpo)
            def _r_env_at(pos: int) -> torch.Tensor:
                return lazy_r.get(pos)

        sweep_energies: list[float] = []
        max_trunc = 0.0

        # Forward sweep: i = 0..L-2
        for i in range(L - 1):
            T = torch.einsum("asx, xtb -> astb", mps[i], mps[i + 1])
            Le = L_env[i]
            Re = _r_env_at(i + 2)
            Wi = mpo.W[i]
            Wip1 = mpo.W[i + 1]
            Wid = mpo.W_diag[i] if use_channel_diagonal else None
            Wip1d = mpo.W_diag[i + 1] if use_channel_diagonal else None
            mv = _make_matvec(Le, Re, Wi, Wip1, Wid, Wip1d)

            eval_, T_new, _ = lanczos_smallest(mv, T, k=lanczos_k, tol=lanczos_tol)
            sweep_energies.append(eval_)

            U, S, Vh, terr = two_site_svd(T_new, current_chi, svd_min)
            max_trunc = max(max_trunc, terr)
            mps[i] = U
            # Absorb S into right tensor so orthogonality center moves to i+1
            mps[i + 1] = torch.einsum("c, ctb -> ctb", S.to(_CPX), Vh)
            # Advance left env past site i
            L_env[i + 1] = _update_left_env_auto(Le, mps[i], Wi, Wid)

        # Free the per-sweep R_env provider before backward (releases memory
        # for sliding mode and detaches the eager list).
        if env_cache == "all":
            R_env_list = None
        else:
            lazy_r = None

        # Backward sweep: i = L-2..0. Build R_envs incrementally — one
        # tensor at a time — regardless of env_cache mode. The MPS just
        # changed during the forward sweep, so any cached R_envs are stale.
        # Start with R_env[L] = trivial (always valid), update as we walk.
        current_r_env = trivial_env(mpo.K, mpo.device)  # R_env[L]
        for i in range(L - 2, -1, -1):
            T = torch.einsum("asx, xtb -> astb", mps[i], mps[i + 1])
            Le = L_env[i]
            Re = current_r_env  # equals R_env[i+2] at this step
            Wi = mpo.W[i]
            Wip1 = mpo.W[i + 1]
            Wid = mpo.W_diag[i] if use_channel_diagonal else None
            Wip1d = mpo.W_diag[i + 1] if use_channel_diagonal else None
            mv = _make_matvec(Le, Re, Wi, Wip1, Wid, Wip1d)

            eval_, T_new, _ = lanczos_smallest(mv, T, k=lanczos_k, tol=lanczos_tol)
            sweep_energies.append(eval_)

            U, S, Vh, terr = two_site_svd(T_new, current_chi, svd_min)
            max_trunc = max(max_trunc, terr)
            # Now absorb S into left tensor so orthogonality center moves to i
            mps[i] = torch.einsum("asx, x -> asx", U, S.to(_CPX))
            mps[i + 1] = Vh
            # Advance the sliding R_env: R_env[i+1] = update(R_env[i+2], mps[i+1])
            current_r_env = _update_right_env_auto(
                current_r_env, mps[i + 1], Wip1, Wip1d
            )

        # Canonical sweep energy is the Rayleigh quotient of the CURRENT
        # (post-truncation) MPS, not the minimum local Ritz value — the min-Ritz value
        # drifts from the true MPS energy after truncation, which is exactly the trust
        # signal we want for convergence and cache metadata.
        # ``assume_normalized=False`` divides by <psi|psi>: SVD truncation leaves the
        # norm strictly below 1 when discarded singular values are nonzero, so the
        # un-normalized sandwich would under-estimate the quotient.
        E_full = mps_mpo_expectation(mps, mpo, assume_normalized=False).real
        # DMRGResult.energy historically "excludes identity_offset" — keep
        # that contract for diagnostics (energy_history is exposed to
        # callers); the offset is added back in compute_ground_state_dmrg
        # via a final mps_mpo_expectation call.
        E = E_full - float(mpo.identity_offset.real)
        energy_history.append(E)
        final_trunc_err = max_trunc
        if verbose:
            print(f"sweep {sweep_n}: E_mps={E:.10f} chi={current_chi} trunc={max_trunc:.2e} "
                  f"(min_ritz={min(sweep_energies):.10f})")

        # Convergence check ONLY at chi_max
        if current_chi >= bond_dim:
            if abs(E - prev_E) < energy_tol and max_trunc < svd_min:
                converged = True
                # Run one more sweep number for clarity
                break
            # Stall gate (opt-in): stop burning sweeps once the energy has
            # flat-lined at the truncation noise floor, far above energy_tol.
            # NOT convergence — converged stays False.
            if stall_patience > 0:
                if abs(E - prev_E) < stall_tol * max(abs(E), 1.0):
                    stall_count += 1
                    if stall_count >= stall_patience:
                        if verbose:
                            print(f"[dmrg_sweeps] stalled: |dE| < {stall_tol:g}*scale "
                                  f"for {stall_patience} consecutive sweeps — stopping")
                        break
                else:
                    stall_count = 0
        prev_E = E

    # ``final_chi`` reports the OBSERVED maximum bond dim in the converged MPS, not
    # the requested cap — returning the requested upper bound would mislead the cache
    # filename for ground states whose true MPS rank is below it. Computed as the max
    # bond dim across interior bonds, which is what cache readers care about.
    observed_chi = max(
        (t.shape[2] for t in mps[:-1]),
        default=1,
    ) if len(mps) >= 2 else 1
    return DMRGResult(
        energy=energy_history[-1],
        mps=mps,
        converged=converged,
        n_sweeps=len(energy_history),
        final_chi=int(observed_chi),
        final_trunc_err=final_trunc_err,
        energy_history=energy_history,
    )


# ---------------------------------------------------------------------------
# MPS -> dense vector

def mps_to_dense_vector(mps: list[torch.Tensor]) -> np.ndarray:
    """Contract the MPS chain into a dense 2**L-dim state vector.

    Refuses to materialize a dense vector of size 2**L when L exceeds the
    repo-wide full-state guard limit (see code/full_state_guard.py). The
    helper-level ``PauliHamiltonianHelper`` already skips this path for
    n>=26, but the public function and tools could otherwise accidentally
    request a 2**52 / 2**64 contraction.
    """
    L = len(mps)
    try:
        from .full_state_guard import (
            EXACT_FULL_STATE_QUBIT_LIMIT,
            is_large_full_state_system,
        )
    except ImportError:
        from full_state_guard import (
            EXACT_FULL_STATE_QUBIT_LIMIT,
            is_large_full_state_system,
        )
    if is_large_full_state_system(L):
        raise ValueError(
            f"mps_to_dense_vector refuses to materialize 2**{L} amplitudes "
            f"(~10^{int(L * 0.30103)}) for n_qubits={L} >= "
            f"{EXACT_FULL_STATE_QUBIT_LIMIT}. Use mps_mpo_expectation / "
            f"mps_pauli_expectation directly on the MPS instead of contracting "
            f"to a dense state vector."
        )
    state = mps[0]  # (1, d, chi_1)
    for i in range(1, L):
        # state shape: (1, d_prev, chi_i). Next: mps[i] (chi_i, d, chi_ip1)
        chi_prev = state.shape[-1]
        d_prev = state.shape[1]
        d, _, chi_next = mps[i].shape[1], 0, mps[i].shape[2]
        # Contract over chi_i
        state = torch.einsum("axc, cyd -> axyd", state, mps[i])
        # Merge physical dims
        state = state.reshape(1, d_prev * d, chi_next)
    # Final state: (1, 2**L, 1) -> flatten
    return state.reshape(-1).cpu().numpy().astype(np.complex128)


# ---------------------------------------------------------------------------
# MPS-MPO expectation values.
# These are the primitives the MPS-native EnergyEstimator reads. Shipping them
# here lets the MPS cache contract be validated end-to-end without going through
# dense state vectors.


def mps_norm_squared(mps: list[torch.Tensor]) -> float:
    """Compute <psi|psi> for an MPS. Real, non-negative.

    Cost O(L * chi^3). For a right-canonical or normalized MPS this is 1.0
    (or very close), but the function must work for arbitrary input so
    callers of ``mps_mpo_expectation`` can pass non-normalized states.
    """
    if not mps:
        return 0.0
    target_device = mps[0].device
    mps_d = [t.to(target_device) if t.device != target_device else t for t in mps]
    # Build the norm env from the left: env[a, A] = sum_{prefix} psi[..,a]* psi[..,A].
    env = torch.zeros((1, 1), dtype=mps_d[0].dtype, device=target_device)
    env[0, 0] = 1.0
    for t in mps_d:
        # env: (a, A); t: (a, s, b); new env: (b, B) = sum_{a A s} env[a,A] t*[a,s,b] t[A,s,B]
        env = torch.einsum("aA, asb, AsB -> bB", env, t.conj(), t)
    # env is (1, 1); the single entry is <psi|psi>.
    return float(env[0, 0].real.item())


def mps_mpo_expectation(
    mps: list[torch.Tensor],
    mpo: PauliMPO,
    *,
    assume_normalized: bool = True,
) -> complex:
    """Compute <psi|H|psi> where psi is an MPS and H is the sum-of-strings MPO.

    Includes the identity_offset (i.e., returns the same energy DMRG converges
    to). Cost is O(L * chi^3 * K) per sweep -- comparable to a single DMRG
    sweep's env update, far cheaper than materializing a dense state.

    Normalization:
        DMRG output is right-canonical so <psi|psi> == 1 to machine precision
        and the raw MPS-MPO sandwich plus ``identity_offset`` IS the
        expectation value. ``assume_normalized=True`` (default) keeps that
        fast path.

        For arbitrary user-supplied MPS (e.g. via ``mps_pauli_expectation``
        on a hand-built MPS, or after a non-canonical operation), pass
        ``assume_normalized=False`` to compute <psi|psi> and divide both
        the contraction and the identity offset by it — the correct
        unnormalized formula being
            <psi|H|psi> / <psi|psi>  =  (raw_sandwich + offset * <psi|psi>) / <psi|psi>.

    This is the primitive that replaces EnergyEstimator's dense
    <psi|H|psi> path. It extends to a Clifford-rotated MPS (the diagonal
    basis) and per-Pauli sandwiches; the contraction shape stays the same.
    """
    L = len(mps)
    if L != mpo.n_qubits:
        raise ValueError(f"MPS length {L} != MPO n_qubits {mpo.n_qubits}")
    # Move MPS to MPO device if needed
    target_device = mpo.device
    mps_d = [t.to(target_device) if t.device != target_device else t for t in mps]

    # Build L_env from the left, sweeping forward.
    # L_env[0] is the trivial (1, 1, 1) at the leftmost boundary.
    L_env = trivial_env(mpo.K, mpo.device)
    for i in range(L):
        L_env = _update_left_env_auto(L_env, mps_d[i], mpo.W[i], mpo.W_diag[i])
    # After processing all L sites, L_env has the boundary shape (1, 1, 1)
    # and its single entry equals <psi|H_no_identity|psi>.
    raw_sandwich = complex(L_env[0, 0, 0].item())
    offset = complex(mpo.identity_offset)
    if assume_normalized:
        return raw_sandwich + offset
    norm_sq = mps_norm_squared(mps_d)
    if norm_sq <= 0.0:
        raise ValueError(f"mps_mpo_expectation: <psi|psi> = {norm_sq} <= 0")
    return (raw_sandwich + offset * norm_sq) / norm_sq


def mps_pauli_expectation(
    mps: list[torch.Tensor],
    pauli_string: str,
    *,
    coefficient: complex = 1.0,
    device: torch.device | str | None = None,
    assume_normalized: bool = True,
) -> complex:
    """Compute <psi|c * P|psi> for a single Pauli string P.

    Constructs a trivial K=1 MPO with the Pauli at each site and runs the
    same MPS-MPO sandwich as ``mps_mpo_expectation``. This is the primitive
    used for per-Pauli measurement expectation values.

    Pass ``assume_normalized=False`` when ``mps`` is not guaranteed to be
    a right-canonical / unit-norm state (e.g. hand-built test states); the
    function will then normalize by <psi|psi> the same way
:func:`mps_mpo_expectation` does.
    """
    if device is None:
        device = mps[0].device if mps else torch.device("cpu")
    # Build a single-string MPO directly (K=1, no fanout coefficients besides `c`).
    mpo = pauli_sum_to_mpo([pauli_string], [coefficient], len(pauli_string), device=device)
    return mps_mpo_expectation(mps, mpo, assume_normalized=assume_normalized)


# ---------------------------------------------------------------------------
# Top-level entry point

def _solve_ground_state_n1(
    pauli_strings: list[str],
    coefficients: list[complex],
    device: torch.device | str,
    *,
    return_dense_vector: bool = True,
) -> tuple[float, Optional[np.ndarray], dict]:
    """Direct 2x2 diagonalization for the n_qubits == 1 case.

    The two-site DMRG sweep has no interior bonds for L=1; the sum-of-strings
    MPO construction also assumes L >= 2. Just build the dense 2x2 matrix and
    return its lowest eigenpair plus a trivial single-site MPS for the
    canonical cache contract.

    Validation mirrors ``pauli_sum_to_mpo``:
      - empty input is rejected;
      - characters outside ``IXYZ`` are rejected;
      - non-Hermitian (imag > 1e-10 * |c|) coefficients on non-identity
        terms are rejected;
      - identity-only terms are split off as the canonical
        ``identity_offset`` so cache metadata is correct rather than 0.0;
      - ``return_dense_vector=False`` honors the public-API contract
.
    """
    if len(pauli_strings) != len(coefficients):
        raise ValueError("strings/coefficients length mismatch")
    if not pauli_strings:
        raise ValueError("empty Pauli list")
    _PAULI_KEY = "IXYZ"
    identity_offset = 0.0 + 0.0j
    H = torch.zeros((2, 2), dtype=_CPX, device=device)
    for s, c in zip(pauli_strings, coefficients):
        if len(s) != 1 or s not in PAULI_TENSORS:
            raise ValueError(f"expected single-character Pauli for n=1, got {s!r}")
        if any(ch not in _PAULI_KEY for ch in s):
            raise ValueError(f"Pauli string {s!r} has invalid character (must be I/X/Y/Z)")
        cc = complex(c)
        if s == "I":
            identity_offset += cc
            continue
        if abs(cc.imag) > 1e-10 * max(abs(cc), 1.0):
            raise ValueError(
                f"Coefficient {cc} on non-identity term {s!r} has imag part "
                f"{cc.imag:.3e} > 1e-10; input must be Hermitian canonical form."
            )
        # dtype=_CPX (not just .to(device)) so an active complex64 precision
        # context is honored -- see _precision's docstring.
        H = H + cc * PAULI_TENSORS[s].to(device=device, dtype=_CPX)
    if abs(identity_offset.imag) > 1e-10 * max(abs(identity_offset), 1.0):
        raise ValueError(
            f"Identity offset {identity_offset} has imag part {identity_offset.imag:.3e}"
        )
    eigvals, eigvecs = torch.linalg.eigh(H)
    # H above EXCLUDES the identity offset (kept symmetric with the
    # pauli_sum_to_mpo convention); add it back for the reported energy.
    energy = float(eigvals[0].real.item()) + float(identity_offset.real)
    ground = eigvecs[:, 0]
    # Single-site MPS tensor of shape (1, 2, 1) with the eigenvector as the
    # physical leg; both bond legs are trivial size-1.
    mps_tensor = ground.reshape(1, 2, 1).contiguous()
    # MPS cache arrays follow the active precision, while dense vectors keep
    # the public n>1 contract from mps_to_dense_vector: numpy complex128.
    np_cpx = np.complex128 if _CPX == torch.complex128 else np.complex64
    mps_numpy = [mps_tensor.cpu().numpy().astype(np_cpx)]
    vec: Optional[np.ndarray] = None
    if return_dense_vector:
        vec = ground.cpu().numpy().astype(np.complex128)
    info = {
        "converged": True,
        "n_sweeps": 0,
        "final_chi": 1,
        "final_trunc_err": 0.0,
        "identity_offset": float(identity_offset.real),
        "energy_history": [energy],
        "mps_numpy": mps_numpy,
    }
    return energy, vec, info


def _autopick_bond_dim(
    n_qubits: int,
    K: int,
    *,
    device: torch.device | str,
    frac: float = 0.5,
    floor: int = 32,
    env_cache: str = "auto",
) -> int:
    """Pick the largest chi that fits within ``frac`` of the device's free memory,
    capped above by an "aspirational" target chi per system size.

    Memory model — BOTH L_envs and R_envs are accounted for. L_envs are
    always eager (one tensor per site, filled during the forward sweep);
    only R_envs can be sliding. Per-tensor cost: ``chi^2 * K * 16 B``.

      - ``env_cache='all'``: L (L_envs) + L (R_envs) = ``2*L*chi^2*K*16``.
      - ``env_cache='sliding'``: L (L_envs, still eager) + sqrt(L) (R_env
        checkpoints) = ``(L + sqrt(L))*chi^2*K*16``.

    Aspirational chi per system size (driven by entanglement scaling for
    spinless-Hubbard targets):
        n <= 12  -> 50
        n <= 16  -> 128   (Hubbard 4x4)
        n <= 25  -> 200
        n <= 49  -> 400
        n <= 64  -> 800   (only achievable with env_cache='sliding')
        n >  64  -> floor (caller should explicitly bump)

    ``env_cache='auto'``: picks 'sliding' if the eager memory cost would
    force a chi cap below the aspirational target, else 'all'.
    """
    aspirational = [(12, 50), (16, 128), (25, 200), (49, 400), (64, 800)]
    chi_target = floor
    for limit, chi in aspirational:
        if n_qubits <= limit:
            chi_target = chi
            break

    # Query free memory
    try:
        dev = torch.device(device) if not isinstance(device, torch.device) else device
        if dev.type == "cuda":
            free_bytes, _total = torch.cuda.mem_get_info(dev)
        else:
            import psutil
            free_bytes = psutil.virtual_memory().available
    except Exception:
        free_bytes = 8 * 1024 ** 3  # 8 GB safe fallback

    budget = frac * free_bytes
    sqrt_L = max(2, int(math.sqrt(n_qubits)))
    # Bytes per env element track the ACTIVE precision (complex128=16,
    # complex64=8). Hardcoding 16 under a complex64 run would under-estimate
    # the fittable chi (cap it at the complex128 ceiling), silently defeating
    # the memory headroom complex64 exists to unlock.
    cpx_bytes = 16 if _CPX == torch.complex128 else 8
    # Eager: L L_envs + L R_envs
    eager_tensors = 2 * n_qubits
    # Sliding: L L_envs (always eager) + sqrt(L) R_env checkpoints
    sliding_tensors = n_qubits + sqrt_L
    chi_max_fit_eager = int(math.sqrt(budget / max(eager_tensors * K * cpx_bytes, 1)))
    chi_max_fit_sliding = int(math.sqrt(budget / max(sliding_tensors * K * cpx_bytes, 1)))

    if env_cache == "all":
        chi = max(min(chi_target, chi_max_fit_eager), floor)
    elif env_cache == "sliding":
        # Cap by the L_env-inclusive sliding budget. The previous formula
        # used only ``sqrt(L) * chi^2 * K * 16`` (R_env-side only) which
        # over-estimated the headroom and let chi shoot above what fits
        # once L_envs filled in during the forward sweep.
        chi = max(min(chi_target, chi_max_fit_sliding), floor)
    elif env_cache == "auto":
        # Prefer eager if its chi can reach the aspirational target;
        # otherwise pick sliding.
        if chi_max_fit_eager >= chi_target:
            chi = chi_target
        else:
            chi = max(min(chi_target, chi_max_fit_sliding), floor)
    else:
        raise ValueError(f"env_cache must be 'all', 'sliding', or 'auto'; got {env_cache!r}")

    if chi < chi_target:
        import logging as _log
        _log.warning(
            f"DMRG: capped bond_dim {chi_target}->{chi} for n_qubits={n_qubits}, K={K} "
            f"(env_cache={env_cache!r}, budget {budget/1024**3:.2f} GB, "
            f"env per bond ~{chi**2 * K * cpx_bytes / 1024**2:.0f} MB)"
        )
    return chi


def compute_ground_state_dmrg(
    pauli_strings: list[str],
    coefficients: list[complex],
    n_qubits: int,
    *,
    bond_dim: int | None = None,
    max_sweeps: int = 30,
    energy_tol: float = 1e-9,
    svd_min: float = 1e-10,
    initial_state: str = "neel",
    return_dense_vector: bool | None = None,
    device: str | torch.device | None = None,
    seed: int | None = 0,
    verbose: bool = False,
    dtype: torch.dtype = torch.complex128,
) -> tuple[float, Optional[np.ndarray], dict]:
    """Run DMRG. Returns (energy, dense_vector | None, info).

    ``dtype`` (default complex128): pass ``torch.complex64`` to halve device
    memory (roughly doubling the reachable chi for a fixed memory budget) and
    run faster on GPU -- single precision, so validate energy parity against
    complex128 for the target system before trusting a complex64 reference.

    ``bond_dim`` defaults (None): memory-aware autopick via
:func:`_autopick_bond_dim`. The chi is the largest that fits in ~50% of
    the target device's free memory, capped by an aspirational ladder per
    system size (see:func:`_autopick_bond_dim` for the table).

    ``return_dense_vector`` defaults: True iff n_qubits < 26.

    Special-cases n_qubits == 1 to a direct 2x2 diagonalization
    (— DMRG's two-site sweep is undefined for L=1).

    The reported energy is the canonical normalized Rayleigh quotient
    ``<MPS_final | H | MPS_final> / <MPS_final | MPS_final>``, not the minimum of the per-sweep
    local Ritz values. SVD truncation after each two-site eigensolve leaves
    the post-truncation MPS norm strictly below 1 whenever discarded
    singular values are nonzero, so we explicitly normalize rather than
    relying on a unit-norm assumption.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Resolve return_dense_vector BEFORE the n=1 fast path so the fast path
    # honors the same contract (MED C13 #2 — was previously
    # ignored on n=1).
    if return_dense_vector is None:
        return_dense_vector = n_qubits < 26
    elif return_dense_vector and n_qubits >= 26:
        # Caller asked for a dense vector but the system is too large to
        # materialize it. Better to fail fast at the entry point than to
        # let mps_to_dense_vector blow up partway through a long DMRG run.
        #
        raise ValueError(
            f"compute_ground_state_dmrg: return_dense_vector=True is "
            f"incompatible with n_qubits={n_qubits} >= 26 — a 2**n dense "
            f"vector cannot be materialised under the repo-wide full-state "
            f"guard. Pass return_dense_vector=False (or leave as None) and "
            f"consume info['mps_numpy'] / mps_mpo_expectation instead."
        )

    # ----- n=1 fast path (Fix #6) -----
    if n_qubits == 1:
        # Honor dtype for n=1 MPS cache arrays too (else complex64 is silently
        # ignored). The optional dense vector still follows the n>1 public
        # contract and is returned as numpy complex128.
        with _precision(dtype):
            return _solve_ground_state_n1(
                pauli_strings, coefficients, device,
                return_dense_vector=return_dense_vector,
            )

    # Precision context (complex128 default / complex64 = half memory) wraps
    # the MPO build, bond-dim autopick, sweeps, and final expectation so every
    # tensor created for this run is at the requested precision.
    with _precision(dtype):
        mpo = pauli_sum_to_mpo(pauli_strings, coefficients, n_qubits, device=device)

        if bond_dim is None:
            bond_dim = _autopick_bond_dim(n_qubits, mpo.K, device=device)

        result = dmrg_sweeps(
            mpo,
            bond_dim=bond_dim,
            max_sweeps=max_sweeps,
            energy_tol=energy_tol,
            svd_min=svd_min,
            initial_state=initial_state,
            seed=seed,
            verbose=verbose,
        )
        # The canonical reported energy is <psi|H|psi> / <psi|psi> on the
        # post-truncation MPS, not the minimum sweep-Ritz energy. SVD truncation
        # leaves the norm strictly below 1 whenever a singular value is discarded, so
        # ``assume_normalized=False`` is required to recover the Rayleigh quotient.
        # ``mps_mpo_expectation`` already adds ``mpo.identity_offset`` after the
        # norm division.
        energy = mps_mpo_expectation(result.mps, mpo, assume_normalized=False).real
        # Move MPS to CPU as numpy arrays so callers can serialize / cache without
        # carrying device state. Callers that want torch tensors on a specific
        # device can convert via torch.tensor(arr, dtype=torch.complex128).to(device).
        mps_numpy = [t.detach().cpu().numpy() for t in result.mps]
        info = {
            "converged": result.converged,
            "n_sweeps": result.n_sweeps,
            "final_chi": result.final_chi,
            "final_trunc_err": result.final_trunc_err,
            "identity_offset": float(mpo.identity_offset.real),
            "energy_history": result.energy_history,
            "mps_numpy": mps_numpy,
        }
        if return_dense_vector:
            vec = mps_to_dense_vector(result.mps)
            return energy, vec, info
        return energy, None, info
