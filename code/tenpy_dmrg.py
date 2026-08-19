"""TeNPy-backed Pauli-sum DMRG ground-state solver (M-TENPY.2/.3).

Drop-in replacement for the in-house torch Pauli-MPO DMRG (``pauli_mpo_dmrg``).
DMRG here is an OFFLINE precompute path; released physics-tenpy is NumPy/CPU
only and that is accepted for this workload.

Public surface (binds the existing call sites in ``dmrg_reference.py`` and
``pauli_hamiltonian_helper.py`` unchanged):

    compute_ground_state_dmrg(pauli_strings, coefficients, n_qubits, *,
        bond_dim=None, max_sweeps=30, energy_tol=1e-9, svd_min=1e-10,
        initial_state="neel", return_dense_vector=None, device=None,
        seed=0, verbose=False) -> (energy, dense_vector | None, info)

    mps_to_dense_vector(mps_numpy, n_qubits) -> np.ndarray
    rayleigh_from_mps(mps_numpy, pauli_strings, coefficients, n_qubits) -> float
    compute_ground_state_fermion_dmrg -> (energy, dense_vector | None, info)

The ``info`` dict mirrors the in-house contract exactly:
    final_chi:int>=1, converged:bool, n_sweeps:int, final_trunc_err:float,
    identity_offset:float, energy_history:list[float],
    mps_numpy:list[complex128 (chi_L, 2, chi_R)].

Energy convention: identity-only Pauli terms ('I'*n) are split off into a
single REAL ``identity_offset`` (excluded from the MPO, added back to the
returned energy). The reported energy equals the normalized Rayleigh quotient
``<psi|H_no_id|psi>`` (TeNPy already normalizes the state) + identity_offset.
"""

from __future__ import annotations

from contextlib import contextmanager
import logging
import threading
import warnings
from typing import Iterator, List, Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Defaults mirrored from the in-house solver (code/pauli_mpo_dmrg.py).

_PAULI_KEY = "IXYZ"
# Single-qubit Pauli matrices, complex128, used for the dense fallback paths
# (n=1 direct diagonalization + Rayleigh-quotient verification).
_PAULI_MAT = {
    "I": np.array([[1, 0], [0, 1]], dtype=np.complex128),
    "X": np.array([[0, 1], [1, 0]], dtype=np.complex128),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=np.complex128),
    "Z": np.array([[1, 0], [0, -1]], dtype=np.complex128),
}
# TeNPy SpinHalfSite operator names: Sigma{x,y,z} are the bare Pauli matrices
# (verified: Sigmax == [[0,1],[1,0]]).
_PAULI_OPNAME = {"X": "Sigmax", "Y": "Sigmay", "Z": "Sigmaz"}

_DEFAULT_MAX_SWEEPS = 30
_DEFAULT_ENERGY_TOL = 1e-9
_DEFAULT_SVD_MIN = 1e-10

# TeNPy uses NumPy's process-global legacy RNG in a few solver paths. Protect
# save/seed/restore as one critical section so concurrent offline solves cannot
# restore each other's state.
_NUMPY_RNG_LOCK = threading.RLock()


# ---------------------------------------------------------------------------
# Input validation + identity-offset split.

def _validate_positive_int(value: object, name: str) -> int:
    """Return *value* as an int, rejecting bools, non-integrals, and <= 0."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return result


def _validate_positive_float(value: object, name: str) -> float:
    """Return a finite, strictly-positive solver parameter."""
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(
            f"{name} must be a positive finite number, got {value!r}"
        )
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            f"{name} must be a positive finite number, got {value!r}"
        ) from exc
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(
            f"{name} must be a positive finite number, got {value!r}"
        )
    return result


def _validate_nonnegative_float(value: object, name: str) -> float:
    """Return a finite, non-negative solver parameter."""
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(
            f"{name} must be a non-negative finite number, got {value!r}"
        )
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            f"{name} must be a non-negative finite number, got {value!r}"
        ) from exc
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(
            f"{name} must be a non-negative finite number, got {value!r}"
        )
    return result


def _validate_finite_float(value: object, name: str) -> float:
    """Return a finite real-valued model parameter."""
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite real number, got {value!r}")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            f"{name} must be a finite real number, got {value!r}"
        ) from exc
    if not np.isfinite(result):
        raise ValueError(f"{name} must be a finite real number, got {value!r}")
    return result


def _validate_seed(seed: Optional[int]) -> Optional[int]:
    """Validate a seed accepted by ``numpy.random.seed`` without truncation."""
    if seed is None:
        return None
    if isinstance(seed, (bool, np.bool_)) or not isinstance(
        seed, (int, np.integer)
    ):
        raise ValueError(f"seed must be an integer or None, got {seed!r}")
    result = int(seed)
    if result < 0 or result > 2**32 - 1:
        raise ValueError(
            f"seed must be in [0, {2**32 - 1}] or None, got {seed!r}"
        )
    return result


@contextmanager
def _preserved_numpy_rng(seed: Optional[int]) -> Iterator[None]:
    """Isolate TeNPy's use of NumPy's global RNG, including failure paths."""
    seed = _validate_seed(seed)
    with _NUMPY_RNG_LOCK:
        state = np.random.get_state()
        try:
            if seed is not None:
                np.random.seed(seed)
            yield
        finally:
            np.random.set_state(state)


def _validate_and_split(
    pauli_strings: Sequence[str],
    coefficients: Sequence[complex],
    n_qubits: int,
) -> Tuple[List[str], List[float], float]:
    """Validate the Pauli sum and split off the identity-only offset.

    Returns ``(non_id_strings, non_id_real_coeffs, identity_offset)``.

    Raises ValueError on: empty input, length mismatch, bad characters,
    wrong-length strings, or a non-identity term with a non-negligible
    imaginary coefficient (Hermiticity).
    """
    n_qubits = _validate_positive_int(n_qubits, "n_qubits")
    if len(pauli_strings) != len(coefficients):
        raise ValueError(
            f"pauli_strings and coefficients must be the same length: "
            f"{len(pauli_strings)} != {len(coefficients)}"
        )
    if len(pauli_strings) == 0:
        raise ValueError(
            "empty Pauli list: need at least one term in the Hamiltonian"
        )

    identity_str = "I" * n_qubits
    identity_offset = 0.0 + 0.0j
    non_id_strings: List[str] = []
    non_id_coeffs: List[float] = []

    for term_index, (s, c) in enumerate(zip(pauli_strings, coefficients)):
        if len(s) != n_qubits:
            raise ValueError(
                f"Pauli string {s!r} length {len(s)} != n_qubits={n_qubits}"
            )
        if any(ch not in _PAULI_KEY for ch in s):
            raise ValueError(
                f"Pauli string {s!r} has an invalid character (must be I/X/Y/Z)"
            )
        try:
            cc = complex(c)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                f"Coefficient {c!r} for term {term_index} is not numeric"
            ) from exc
        if not np.isfinite(cc.real) or not np.isfinite(cc.imag):
            raise ValueError(
                f"Coefficient {cc} for term {term_index} must be finite"
            )
        if s == identity_str:
            identity_offset += cc
            continue
        if abs(cc.imag) > 1e-10 * max(abs(cc), 1.0):
            raise ValueError(
                f"Coefficient {cc} on non-identity term {s!r} has imaginary "
                f"part {cc.imag:.3e} > 1e-10; input must be Hermitian "
                f"canonical form."
            )
        non_id_strings.append(s)
        non_id_coeffs.append(float(cc.real))

    if not np.isfinite(identity_offset.real) or not np.isfinite(
        identity_offset.imag
    ):
        raise ValueError(
            f"Summed identity offset {identity_offset} must be finite"
        )
    if abs(identity_offset.imag) > 1e-10 * max(abs(identity_offset), 1.0):
        raise ValueError(
            f"Identity offset {identity_offset} has imaginary part "
            f"{identity_offset.imag:.3e}; input must be Hermitian."
        )
    return non_id_strings, non_id_coeffs, float(identity_offset.real)


def _autopick_bond_dim(
    n_qubits: int,
    n_terms: Optional[int] = None,
    *,
    available_bytes: Optional[int] = None,
) -> int:
    """Choose an aspirational chi capped by a conservative CPU-memory model.

    Generic Pauli MPO environments scale approximately as
    ``2 * n * D_MPO * chi**2 * sizeof(complex128)``. For a term-list MPO,
    ``n_terms + 2`` is a conservative pre-build proxy for ``D_MPO``. Native
    local fermion models use a small constant proxy. At most 25% of currently
    available host memory is assigned to environments, leaving headroom for
    MPS tensors, SVD workspaces, Python, and other processes.
    """
    n_qubits = _validate_positive_int(n_qubits, "n_qubits")
    if n_terms is not None:
        n_terms = _validate_positive_int(n_terms, "n_terms")
    if n_qubits <= 25:
        target = 200
    elif n_qubits <= 49:
        target = 400
    else:
        target = 800

    if available_bytes is None:
        try:
            import psutil

            available_bytes = int(psutil.virtual_memory().available)
        except Exception:
            available_bytes = 8 * 1024**3
    available_bytes = _validate_positive_int(
        available_bytes,
        "available_bytes",
    )

    mpo_bond_proxy = max((n_terms + 2) if n_terms is not None else 8, 4)
    budget = 0.25 * available_bytes
    bytes_per_chi_squared = 2 * n_qubits * mpo_bond_proxy * 16
    memory_cap = int(np.sqrt(budget / max(bytes_per_chi_squared, 1)))
    floor = min(32, target)
    selected = max(min(target, memory_cap), floor)
    if selected < target:
        logging.warning(
            "tenpy_dmrg: capped automatic bond_dim %d->%d for n_qubits=%d, "
            "n_terms=%r using a %.2f GiB environment budget",
            target,
            selected,
            n_qubits,
            n_terms,
            budget / 1024**3,
        )
    return selected


# ---------------------------------------------------------------------------
# Dense-state full-state guard re-use.

def _is_large_full_state(n_qubits: int) -> bool:
    try:
        from .full_state_guard import is_large_full_state_system
    except ImportError:  # pragma: no cover - direct (non-package) import
        from full_state_guard import is_large_full_state_system
    return is_large_full_state_system(n_qubits)


def _full_state_limit() -> int:
    try:
        from .full_state_guard import EXACT_FULL_STATE_QUBIT_LIMIT
    except ImportError:  # pragma: no cover
        from full_state_guard import EXACT_FULL_STATE_QUBIT_LIMIT
    return EXACT_FULL_STATE_QUBIT_LIMIT


def _resolve_return_dense_vector(
    return_dense_vector: Optional[bool],
    n_qubits: int,
    *,
    caller: str,
) -> bool:
    """Resolve the dense-output default and enforce the full-state guard."""
    if return_dense_vector is None:
        return n_qubits < _full_state_limit()
    if not isinstance(return_dense_vector, (bool, np.bool_)):
        raise ValueError(
            f"{caller}: return_dense_vector must be bool or None, "
            f"got {return_dense_vector!r}"
        )
    if return_dense_vector and _is_large_full_state(n_qubits):
        raise ValueError(
            f"{caller}: return_dense_vector=True is incompatible with "
            f"n_qubits={n_qubits} >= {_full_state_limit()} — a 2**n dense "
            f"vector cannot be materialised under the repo-wide full-state "
            f"guard. Pass return_dense_vector=False and consume "
            f"info['mps_numpy'] instead."
        )
    return bool(return_dense_vector)


# ---------------------------------------------------------------------------
# MPS <-> dense helpers (operate on the public mps_numpy list format).

def mps_to_dense_vector(mps_numpy: List[np.ndarray], n_qubits: int) -> np.ndarray:
    """Contract an MPS (list of (chi_L, 2, chi_R) arrays) into a 2**n vector.

    Refuses to materialize when ``n_qubits`` crosses the repo-wide full-state
    guard limit (26), matching the in-house solver's behaviour. The error
    message contains 'refuses to materialize' for caller assertions.
    """
    n_qubits = _validate_positive_int(n_qubits, "n_qubits")
    if n_qubits != len(mps_numpy):
        raise ValueError(
            f"mps_to_dense_vector: n_qubits={n_qubits} != number of MPS "
            f"tensors {len(mps_numpy)}"
        )
    if _is_large_full_state(n_qubits):
        raise ValueError(
            f"mps_to_dense_vector refuses to materialize 2**{n_qubits} "
            f"amplitudes for n_qubits={n_qubits} >= {_full_state_limit()}. "
            f"Use rayleigh_from_mps / the MPS directly instead of a dense "
            f"state vector."
        )
    if not mps_numpy:
        raise ValueError("empty MPS")
    state = np.asarray(mps_numpy[0], dtype=np.complex128)  # (1, d, chi_1)
    for i in range(1, len(mps_numpy)):
        _, S, _ = state.shape
        _, d, chi_next = mps_numpy[i].shape
        state = np.einsum("axc, cyd -> axyd", state,
                          np.asarray(mps_numpy[i], dtype=np.complex128))
        state = state.reshape(1, S * d, chi_next)
    return state.reshape(-1).astype(np.complex128)


def _pauli_string_dense(s: str) -> np.ndarray:
    """Dense 2**n matrix for one Pauli string (small n only)."""
    m = np.array([[1.0 + 0j]], dtype=np.complex128)
    for ch in s:
        m = np.kron(m, _PAULI_MAT[ch])
    return m


def rayleigh_from_mps(
    mps_numpy: List[np.ndarray],
    pauli_strings: Sequence[str],
    coefficients: Sequence[complex],
    n_qubits: int,
) -> float:
    """Normalized Rayleigh quotient <psi|H|psi>/<psi|psi> from mps_numpy.

    Includes the identity offset. For small n only (contracts to a dense
    vector); the full-state guard applies. This is the cross-check the
    reference/cache tests use to confirm the reported energy equals the
    recomputed Rayleigh quotient.
    """
    non_id, real_c, offset = _validate_and_split(
        pauli_strings, coefficients, n_qubits
    )
    vec = mps_to_dense_vector(mps_numpy, n_qubits)
    norm_sq = float(np.vdot(vec, vec).real)
    if norm_sq <= 0.0:
        raise ValueError(f"rayleigh_from_mps: <psi|psi>={norm_sq} <= 0")
    acc = 0.0 + 0.0j
    for s, c in zip(non_id, real_c):
        Hs = _pauli_string_dense(s)
        acc += c * np.vdot(vec, Hs @ vec)
    return float((acc.real / norm_sq) + offset)


# ---------------------------------------------------------------------------
# TeNPy MPS -> mps_numpy extraction.

def _extract_mps_numpy(psi) -> List[np.ndarray]:
    """Pull B-form tensors out of a (canonicalized) TeNPy finite MPS.

    Returns one complex128 array per site, ndim==3, shape (chi_L, 2, chi_R).
    """
    out: List[np.ndarray] = []
    for i in range(psi.L):
        B = psi.get_B(i, form="B")
        arr = B.transpose(["vL", "p", "vR"]).to_ndarray().astype(np.complex128)
        out.append(np.ascontiguousarray(arr))
    return out


# ---------------------------------------------------------------------------
# n=1 direct diagonalization.

def _solve_n1(
    pauli_strings: Sequence[str],
    coefficients: Sequence[complex],
    *,
    return_dense_vector: bool,
) -> Tuple[float, Optional[np.ndarray], dict]:
    """Direct 2x2 diagonalization for n_qubits == 1.

    TeNPy two-site DMRG asserts L > 2 and even single-site DMRG is degenerate
    for a single site, so handle n=1 with a closed-form eig.
    """
    non_id, real_c, offset = _validate_and_split(pauli_strings, coefficients, 1)
    H = np.zeros((2, 2), dtype=np.complex128)
    for s, c in zip(non_id, real_c):
        H = H + c * _PAULI_MAT[s]
    evals, evecs = np.linalg.eigh(H)
    energy = float(evals[0].real) + offset
    ground = evecs[:, 0].astype(np.complex128)
    mps_numpy = [ground.reshape(1, 2, 1).astype(np.complex128)]
    vec: Optional[np.ndarray] = ground.copy() if return_dense_vector else None
    info = {
        "converged": True,
        "n_sweeps": 0,
        "final_chi": 1,
        "final_trunc_err": 0.0,
        "identity_offset": offset,
        "energy_history": [energy],
        "mps_numpy": mps_numpy,
    }
    return energy, vec, info


# ---------------------------------------------------------------------------
# Core TeNPy DMRG driver (n >= 2).

def _run_tenpy_dmrg(
    non_id_strings: List[str],
    real_coeffs: List[float],
    n_qubits: int,
    *,
    bond_dim: int,
    max_sweeps: int,
    energy_tol: float,
    svd_min: float,
    seed: Optional[int],
    verbose: bool,
):
    """Build the Pauli-sum MPO model and run TeNPy DMRG. Returns the engine
    info dict plus the converged ``psi``.

    n == 2 uses single-site DMRG (active_sites=1; two-site asserts L > 2);
    n >= 3 uses the default two-site engine.
    """
    # TeNPy uses NumPy's global RNG for mixer/random-init internals. The
    # context restores the caller's state on both successful and failed runs.
    with _preserved_numpy_rng(seed):
        from tenpy.networks.site import SpinHalfSite
        from tenpy.networks.terms import TermList
        from tenpy.networks.mpo import MPOGraph
        from tenpy.networks.mps import MPS
        from tenpy.models.lattice import Chain
        from tenpy.models.model import MPOModel
        import tenpy.algorithms.dmrg as dmrg

        # conserve=None is REQUIRED: generic X/Y Pauli terms break Sz
        # conservation.
        site = SpinHalfSite(conserve=None)
        sites = [site] * n_qubits

        terms = []
        strengths = []
        for s, c in zip(non_id_strings, real_coeffs):
            factors = [
                (_PAULI_OPNAME[ch], i)
                for i, ch in enumerate(s)
                if ch != "I"
            ]
            # Non-identity strings always have >= 1 factor (identity-only was
            # split).
            terms.append(factors)
            strengths.append(c)

        tl = TermList(terms, strengths)
        H = MPOGraph.from_term_list(
            tl,
            sites,
            bc="finite",
            unit_cell_width=n_qubits,
        ).build_MPO()
        model = MPOModel(
            Chain(n_qubits, site, bc="open", bc_MPS="finite"),
            H,
        )

        # Neel-like product start; truncated to n sites.
        p_state = (["up", "down"] * n_qubits)[:n_qubits]
        psi = MPS.from_product_state(
            sites,
            p_state,
            bc="finite",
            unit_cell_width=n_qubits,
        )

        dmrg_params = {
            "trunc_params": {"svd_min": svd_min, "chi_max": bond_dim},
            "max_sweeps": max_sweeps,
            # min_sweeps default is fine; convergence handled below.
            "max_E_err": energy_tol,
            "max_S_err": 1e-8,
            # Return a non-converged result (with TeNPy's warning) for an
            # intentionally small chi instead of aborting before callers can
            # inspect ``info['converged']`` and the final MPS.
            "max_trunc_err": None,
            # The density-matrix mixer is ESSENTIAL: without it the
            # single-site/product-start DMRG gets stuck in local minima.
            "mixer": True,
        }
        if n_qubits == 2:
            # Two-site DMRG asserts L > 2; use single-site for L == 2.
            dmrg_params["active_sites"] = 1

        with warnings.catch_warnings():
            # Single-site DMRG may report a non-canonical intermediate state;
            # canonical_form() immediately below resolves that known warning.
            warnings.filterwarnings(
                "ignore",
                message=r".*not in canonical form.*",
                module=r"tenpy(?:\..*)?",
            )
            info = dmrg.run(psi, model, dmrg_params)

        # Single-site DMRG leaves a non-canonical state; restore canonical
        # form before any energy or tensor extraction.
        psi.canonical_form()
        return info, psi, model


def compute_ground_state_dmrg(
    pauli_strings: Sequence[str],
    coefficients: Sequence[complex],
    n_qubits: int,
    *,
    bond_dim: Optional[int] = None,
    max_sweeps: int = _DEFAULT_MAX_SWEEPS,
    energy_tol: float = _DEFAULT_ENERGY_TOL,
    svd_min: float = _DEFAULT_SVD_MIN,
    initial_state: str = "neel",
    return_dense_vector: Optional[bool] = None,
    device: object = None,
    seed: Optional[int] = 0,
    verbose: bool = False,
) -> Tuple[float, Optional[np.ndarray], dict]:
    """TeNPy-backed ground state of a Pauli-sum Hamiltonian.

    Returns ``(energy, dense_vector | None, info)``. See module docstring for
    the ``info`` schema. ``device`` is accepted for call-site compatibility
    but ignored (released TeNPy is CPU/NumPy only — accepted for the offline
    precompute path). ``initial_state`` other than 'neel' is rejected.

    ``return_dense_vector`` defaults to True iff ``n_qubits < 26``; requesting
    True for n>=26 raises ValueError (message contains 'incompatible with
    n_qubits=').
    """
    n_qubits = _validate_positive_int(n_qubits, "n_qubits")
    if bond_dim is not None:
        bond_dim = _validate_positive_int(bond_dim, "bond_dim")
    max_sweeps = _validate_positive_int(max_sweeps, "max_sweeps")
    energy_tol = _validate_positive_float(energy_tol, "energy_tol")
    svd_min = _validate_nonnegative_float(svd_min, "svd_min")
    seed = _validate_seed(seed)

    if initial_state != "neel":
        raise ValueError(f"Unknown initial_state={initial_state!r} (only 'neel')")
    if device is not None and str(device) not in ("cpu", "None"):
        logging.info(
            "tenpy_dmrg: device=%r ignored; released TeNPy is CPU/NumPy only.",
            device,
        )

    return_dense_vector = _resolve_return_dense_vector(
        return_dense_vector,
        n_qubits,
        caller="compute_ground_state_dmrg",
    )

    # ---- n == 1 fast path ----
    if n_qubits == 1:
        return _solve_n1(
            pauli_strings, coefficients,
            return_dense_vector=return_dense_vector,
        )

    non_id, real_c, offset = _validate_and_split(
        pauli_strings, coefficients, n_qubits
    )

    # Pure-identity Hamiltonian: H = offset * I. Any product state is a ground
    # state; report the offset and a trivial bond-dim-1 MPS.
    if not non_id:
        d = 2
        mps_numpy = []
        for i in range(n_qubits):
            t = np.zeros((1, d, 1), dtype=np.complex128)
            t[0, i % 2, 0] = 1.0
            mps_numpy.append(t)
        info = {
            "converged": True,
            "n_sweeps": 0,
            "final_chi": 1,
            "final_trunc_err": 0.0,
            "identity_offset": offset,
            "energy_history": [offset],
            "mps_numpy": mps_numpy,
        }
        vec = mps_to_dense_vector(mps_numpy, n_qubits) if return_dense_vector else None
        return offset, vec, info

    if bond_dim is None:
        bond_dim = _autopick_bond_dim(n_qubits, len(non_id))

    info_t, psi, model = _run_tenpy_dmrg(
        non_id, real_c, n_qubits,
        bond_dim=bond_dim, max_sweeps=max_sweeps,
        energy_tol=energy_tol, svd_min=svd_min, seed=seed, verbose=verbose,
    )

    # Reported energy = normalized Rayleigh quotient + identity offset.
    raw_E = float(np.real(model.H_MPO.expectation_value(psi)))
    energy = raw_E + offset

    mps_numpy = _extract_mps_numpy(psi)
    energy, info = _finalize_info(
        info_t, psi, energy, offset, mps_numpy,
        bond_dim=bond_dim, energy_tol=energy_tol, svd_min=svd_min,
    )

    vec: Optional[np.ndarray] = None
    if return_dense_vector:
        vec = mps_to_dense_vector(mps_numpy, n_qubits)
    if verbose:
        logging.info(
            "tenpy_dmrg: E=%.10f chi=%d sweeps=%d trunc=%.3e converged=%s",
            energy, info["final_chi"], info["n_sweeps"],
            info["final_trunc_err"], info["converged"],
        )
    return energy, vec, info


def _finalize_info(
    info_t,
    psi,
    energy: float,
    offset: float,
    mps_numpy: List[np.ndarray],
    *,
    bond_dim: int,
    energy_tol: float,
    svd_min: float,
) -> Tuple[float, dict]:
    """Assemble the contract ``info`` dict from a TeNPy run's statistics."""
    stats = info_t.get("sweep_statistics", {}) or {}
    e_hist_raw = stats.get("E", [])
    energy_history = [float(np.real(e)) + offset for e in e_hist_raw]
    if not energy_history:
        energy_history = [energy]

    n_sweeps = int(len(energy_history))

    trunc_list = stats.get("max_trunc_err", [])
    final_trunc_err = float(np.real(trunc_list[-1])) if len(trunc_list) else 0.0

    # Observed max interior bond dim.
    chi = psi.chi  # list of bond dims (length L-1 for finite)
    final_chi = int(max(chi)) if len(chi) else 1
    final_chi = max(final_chi, 1)

    delta_list = stats.get("Delta_E", [])
    if len(delta_list):
        last_dE = abs(float(np.real(delta_list[-1])))
    elif len(energy_history) >= 2:
        last_dE = abs(energy_history[-1] - energy_history[-2])
    else:
        last_dE = float("inf")

    # converged:= the FINAL-sweep truncation error is at most svd_min AND the
    # bond dim reached the requested cap (or the exact MPS rank, when smaller
    # than the cap with no truncation) AND the final-sweep energy delta is below
    # energy_tol. The chi-ramp's early high-truncation sweeps are intentionally
    # excluded -- using the max truncation over ALL sweeps systematically
    # false-negatives on systems that need a chi ramp (the reported energy can
    # be the correct converged value while the max-over-sweeps stays > svd_min).
    # ``svd_min=0`` is a supported exact-truncation setting; use <= so an
    # exactly zero truncation error can still satisfy convergence.
    truncation_converged = final_trunc_err <= svd_min
    reached_cap = final_chi >= int(bond_dim) or truncation_converged
    converged = bool(
        reached_cap and last_dE < energy_tol and truncation_converged
    )

    info = {
        "converged": converged,
        "n_sweeps": n_sweeps,
        "final_chi": final_chi,
        "final_trunc_err": final_trunc_err,
        "identity_offset": offset,
        "energy_history": energy_history,
        "mps_numpy": mps_numpy,
    }
    return energy, info


# ---------------------------------------------------------------------------
# Charge-conserving (spinless Hubbard) route via FermionModel.

def _validate_fermion_lattice(
    *,
    n_qubits: int,
    lattice: str,
    L: Optional[int],
    Lx: Optional[int],
    Ly: Optional[int],
) -> dict:
    """Return validated TeNPy lattice parameters with exactly n_qubits sites."""
    if lattice not in ("Chain", "Ladder", "Square"):
        raise ValueError(
            "lattice must be one of 'Chain', 'Ladder', or 'Square', "
            f"got {lattice!r}"
        )

    if lattice in {"Chain", "Ladder"}:
        if Lx is not None or Ly is not None:
            raise ValueError(
                f"lattice={lattice!r} accepts L only; "
                f"got Lx={Lx!r}, Ly={Ly!r}"
            )
        if L is None:
            if lattice == "Ladder" and n_qubits % 2:
                raise ValueError(
                    "lattice='Ladder' requires an even n_qubits when L is "
                    f"omitted, got n_qubits={n_qubits}"
                )
            length = n_qubits if lattice == "Chain" else n_qubits // 2
        else:
            length = _validate_positive_int(L, "L")
        expected_sites = length if lattice == "Chain" else 2 * length
        dimensions = {"L": length}
    else:
        if L is not None:
            raise ValueError(
                f"lattice='Square' accepts Lx and Ly, not L={L!r}"
            )
        if Lx is None or Ly is None:
            raise ValueError(
                f"lattice='Square' requires Lx and Ly "
                f"(got Lx={Lx!r}, Ly={Ly!r})"
            )
        width = _validate_positive_int(Lx, "Lx")
        height = _validate_positive_int(Ly, "Ly")
        expected_sites = width * height
        dimensions = {"Lx": width, "Ly": height}

    if expected_sites != n_qubits:
        raise ValueError(
            f"lattice={lattice!r} dimensions define {expected_sites} sites, "
            f"but n_qubits={n_qubits}"
        )
    return dimensions


def _validate_conserve_and_particles(
    conserve: Optional[str],
    n_particles: Optional[int],
    n_sites: int,
) -> Tuple[Optional[str], Optional[int]]:
    """Validate TeNPy's conservation mode and an optional fixed-N sector."""
    if conserve not in ("N", "parity", None):
        raise ValueError(
            "conserve must be 'N', 'parity', or None, "
            f"got {conserve!r}"
        )
    if n_particles is None:
        return conserve, None
    if isinstance(n_particles, (bool, np.bool_)) or not isinstance(
        n_particles, (int, np.integer)
    ):
        raise ValueError(
            f"n_particles must be an integer or None, got {n_particles!r}"
        )
    count = int(n_particles)
    if count < 0 or count > n_sites:
        raise ValueError(
            f"n_particles must be in [0, {n_sites}], got {n_particles!r}"
        )
    if conserve != "N":
        raise ValueError(
            "a fixed n_particles sector requires conserve='N', "
            f"got conserve={conserve!r}"
        )
    return conserve, count


def _solve_fermion_n1(
    *,
    mu: float,
    conserve: Optional[str],
    n_particles: Optional[int],
    return_dense_vector: bool,
) -> Tuple[float, Optional[np.ndarray], dict]:
    """Exact one-site spinless-fermion solution in the requested sector.

    With no fixed count, conserved modes retain the Neel-like initial sector
    (occupied for site zero). With ``conserve=None``, choose the lower-energy
    occupation. This avoids routing a one-site model through TeNPy's two-site
    DMRG engine.
    """
    if n_particles is not None:
        occupied = n_particles == 1
    elif conserve in {"N", "parity"}:
        occupied = True
    else:
        occupied = mu > 0.0

    energy = -mu if occupied else 0.0
    ground = np.array(
        [0.0, 1.0] if occupied else [1.0, 0.0],
        dtype=np.complex128,
    )
    mps_numpy = [ground.reshape(1, 2, 1)]
    vec = ground.copy() if return_dense_vector else None
    info = {
        "converged": True,
        "n_sweeps": 0,
        "final_chi": 1,
        "final_trunc_err": 0.0,
        "identity_offset": 0.0,
        "energy_history": [float(energy)],
        "mps_numpy": mps_numpy,
    }
    return float(energy), vec, info


def compute_ground_state_fermion_dmrg(
    *,
    n_qubits: int,
    lattice: str = "Chain",
    L: Optional[int] = None,
    Lx: Optional[int] = None,
    Ly: Optional[int] = None,
    t: float = 1.0,
    V: float = 0.0,
    mu: float = 0.0,
    conserve: Optional[str] = "N",
    n_particles: Optional[int] = None,
    bond_dim: Optional[int] = None,
    max_sweeps: int = _DEFAULT_MAX_SWEEPS,
    energy_tol: float = _DEFAULT_ENERGY_TOL,
    svd_min: float = _DEFAULT_SVD_MIN,
    return_dense_vector: Optional[bool] = None,
    seed: Optional[int] = 0,
    verbose: bool = False,
) -> Tuple[float, Optional[np.ndarray], dict]:
    """Charge-conserving DMRG for a spinless-fermion (Hubbard) lattice.

    Uses ``tenpy.models.fermions_spinless.FermionModel`` which auto-inserts the
    Jordan-Wigner string (correct fermion signs) and supports U(1) particle
    number (``conserve='N'``) or parity (``conserve='parity'``) conservation to
    cut bond dimension / memory.

    Returns the SAME ``(energy, dense_vector | None, info)`` contract as
:func:`compute_ground_state_dmrg`. ``identity_offset`` is 0.0 (no Pauli
    identity split on this native-lattice route). ``n_qubits`` is the number
    of sites and feeds the full-state guard for the optional dense vector.
    A supplied ``n_particles`` selects an exact U(1) sector and therefore
    requires ``conserve='N'``. With no explicit count, conserved modes use the
    alternating product state's sector (ceil(n_qubits / 2) particles).

    ``H = -t sum_<ij> (c_i^dag c_j + h.c.) + V sum_<ij> n_i n_j - mu sum_i n_i``
    in TeNPy's FermionModel sign convention (J = t here).
    """
    n_qubits = _validate_positive_int(n_qubits, "n_qubits")
    dimensions = _validate_fermion_lattice(
        n_qubits=n_qubits,
        lattice=lattice,
        L=L,
        Lx=Lx,
        Ly=Ly,
    )
    t = _validate_finite_float(t, "t")
    V = _validate_finite_float(V, "V")
    mu = _validate_finite_float(mu, "mu")
    conserve, n_particles = _validate_conserve_and_particles(
        conserve,
        n_particles,
        n_qubits,
    )
    if bond_dim is not None:
        bond_dim = _validate_positive_int(bond_dim, "bond_dim")
    max_sweeps = _validate_positive_int(max_sweeps, "max_sweeps")
    energy_tol = _validate_positive_float(energy_tol, "energy_tol")
    svd_min = _validate_nonnegative_float(svd_min, "svd_min")
    seed = _validate_seed(seed)

    # Resolve the dense-state policy before importing TeNPy, constructing a
    # model, or running any solver work.
    return_dense_vector = _resolve_return_dense_vector(
        return_dense_vector,
        n_qubits,
        caller="compute_ground_state_fermion_dmrg",
    )

    if n_qubits == 1:
        return _solve_fermion_n1(
            mu=mu,
            conserve=conserve,
            n_particles=n_particles,
            return_dense_vector=return_dense_vector,
        )

    if bond_dim is None:
        bond_dim = _autopick_bond_dim(n_qubits)

    model_params = {
        "lattice": lattice,
        "J": t,
        "V": V,
        "mu": mu,
        "conserve": conserve,
        "bc_MPS": "finite",
        "bc_x": "open",
    }
    model_params.update(dimensions)
    if lattice == "Square":
        # TeNPy's Square lattice otherwise defaults to periodic/cylinder bc_y.
        model_params["bc_y"] = "open"

    # Keep TeNPy's process-global RNG use isolated for the complete model/run
    # operation, including exceptions and canonicalization.
    with _preserved_numpy_rng(seed):
        from tenpy.models.fermions_spinless import FermionModel
        from tenpy.networks.mps import MPS
        import tenpy.algorithms.dmrg as dmrg

        M = FermionModel(model_params)
        sites = M.lat.mps_sites()
        n_sites = len(sites)
        if n_sites != n_qubits:
            raise RuntimeError(
                f"TeNPy lattice={lattice!r} constructed {n_sites} sites, "
                f"expected exactly n_qubits={n_qubits}"
            )

        # Half-filling Neel-like start, or the validated fixed-N sector.
        if n_particles is None:
            p_state = [
                "full" if i % 2 == 0 else "empty"
                for i in range(n_sites)
            ]
        else:
            p_state = ["empty"] * n_sites
            for i in range(n_particles):
                p_state[i] = "full"
        psi = MPS.from_product_state(
            sites,
            p_state,
            bc="finite",
            unit_cell_width=M.lat.mps_unit_cell_width,
        )

        dmrg_params = {
            "trunc_params": {"svd_min": svd_min, "chi_max": bond_dim},
            "max_sweeps": max_sweeps,
            "max_E_err": energy_tol,
            "max_S_err": 1e-8,
            # Match the public result contract: expose large truncation
            # through a warning and ``info['converged']=False`` rather than
            # raising before the final MPS can be returned.
            "max_trunc_err": None,
            "mixer": True,
        }
        if n_sites == 2:
            dmrg_params["active_sites"] = 1

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r".*not in canonical form.*",
                module=r"tenpy(?:\..*)?",
            )
            info_t = dmrg.run(psi, M, dmrg_params)
        psi.canonical_form()

        # The DMRG result's ``E`` can be the pre-truncation sweep energy.
        # Recompute from the final canonical MPS so the scalar and returned
        # tensors obey the same Rayleigh-quotient contract.
        energy = float(np.real(M.H_MPO.expectation_value(psi)))
        mps_numpy = _extract_mps_numpy(psi)

    energy, info = _finalize_info(
        info_t, psi, energy, 0.0, mps_numpy,
        bond_dim=bond_dim, energy_tol=energy_tol, svd_min=svd_min,
    )

    vec: Optional[np.ndarray] = None
    if return_dense_vector:
        vec = mps_to_dense_vector(mps_numpy, n_qubits)

    if verbose:
        logging.info(
            "tenpy_dmrg(fermion): E=%.10f chi=%d sweeps=%d converged=%s",
            energy, info["final_chi"], info["n_sweeps"], info["converged"],
        )
    return energy, vec, info
