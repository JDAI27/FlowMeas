"""Energy-only DMRG reference for the canonical large-system workload.

This module is the public, narrow producer/consumer surface for DMRG-backed
scalar reference energies on Hubbard targets too large for full-state
evaluation (notably the frozen 52-qubit spin-less 26x2-ladder benchmark).

Design rules:

  - **Energy only.** The public API never returns an MPS or a dense state
    vector. ``compute_ground_state_dmrg`` from ``tenpy_dmrg`` is invoked
    with ``return_dense_vector=False``; the MPS payload is discarded after
    the Rayleigh quotient is computed so this module cannot leak full-state
    artefacts into the large-system path.
  - **Storage is separate from exact-state caches.** The sidecar lives next
    to the Hamiltonian source file (``<dir>/dmrg_reference.json``), not in
    ``cache/ground_states/`` where ``PauliHamiltonianHelper`` writes its
    MPS / dense / FCI artefacts. A DMRG reference run therefore never
    touches the exact-state cache, and a sidecar load never triggers
    full-state materialisation.
  - **Hash-pinned.** Each sidecar records a SHA-256 of the Hamiltonian
    source bytes. ``load_dmrg_reference`` refuses to return an energy
    whose hash does not match the on-disk Hamiltonian — a regenerated
    file silently invalidates the old reference.
  - **Reports distinguish exact / approximate / structural.** The scalar
    flows through ``ExperimentConfig.dmrg_reference_path`` →
    ``main.get_evaluator_mode_metadata`` →
    ``validation_tier.get_validation_tier_metadata``. ``dmrg_reference`` is
    the canonical "approximate" tier in the validation-tier vocabulary
    (DMRG is variational + bond-dim-truncated, so the reference is an
    approximate, not exact, scalar).

The reusable DMRG solver itself lives in ``tenpy_dmrg`` (M-TENPY.4 repointed
the reference solver from the in-house torch Pauli-MPO backend
``pauli_mpo_dmrg`` to the released physics-tenpy backend; DMRG is an OFFLINE
precompute path and released TeNPy is NumPy/CPU-only — accepted for this
workload). This module is the FlowMeas application-layer wrapper that owns
the *reference* contract (sidecar layout, hash pinning, config resolution,
report wiring).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    from .validation_tier import coerce_dmrg_reference_energy as _coerce_finite_float
except ImportError:
    from validation_tier import (  # type: ignore[no-redef]
        coerce_dmrg_reference_energy as _coerce_finite_float,
    )

logger = logging.getLogger(__name__)


# Sidecar filename, always located next to the Hamiltonian source file.
DMRG_REFERENCE_SIDECAR_NAME = "dmrg_reference.json"

# Sidecar schema version. Bump whenever the on-disk shape changes in a
# non-backwards-compatible way so loaders can refuse stale records.
DMRG_REFERENCE_SCHEMA_VERSION = 1

# FlowMeas repository root. Kept as a module constant so tests and
# alternate launchers can monkeypatch it without rewriting ``__file__``.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_NATIVE_NOTES_PREFIX = "native_fermion_dmrg:"


# ---------------------------------------------------------------------------
# Sidecar payload
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DMRGReferenceMetadata:
    """Immutable record of one DMRG reference computation.

    ``energy`` is the only numerical reference value consumed downstream;
    every other field is solver provenance, recorded so a reader can judge
    how tight the bond-dim-truncated approximation is.

    Fields:
      energy                 Rayleigh quotient ``<psi|H|psi>/<psi|psi>`` of the
                             converged (truncated) MPS, identity offset included.
      n_qubits, n_terms      Hamiltonian size (terms post-canonicalisation).
      bond_dim               Maximum bond dim realised in the final MPS
                             (``DMRGResult.final_chi``), not the requested cap.
      requested_bond_dim     Cap passed to ``dmrg_sweeps``; ``None`` if autopicked.
      n_sweeps, max_sweeps   Sweeps performed, and the budget.
      converged              Final-sweep energy delta below ``energy_tol`` and
                             truncation error at most ``svd_min``, once chi has
                             reached the cap (or the exact MPS rank, if smaller).
      energy_tol, svd_min    Tolerances passed to ``dmrg_sweeps``.
      final_trunc_err        Largest discarded singular-value weight at the final
                             sweep; ``0.0`` when nothing was truncated.
      identity_offset        Constant ``c_I`` split off from the Pauli sum, so
                             ``<psi|H_traceless|psi>`` can be rederived.
      compute_time_seconds   Seconds spent in ``compute_ground_state_dmrg``.
      device                 Device the solver ran on (``cpu`` or ``cuda``).
      hamiltonian_hash       SHA-256 of the Hamiltonian source bytes; the primary
                             loader-side guard against stale references.
      hamiltonian_path       Source path as passed in — traceability only, the
                             loader does not require it to match.
      hamiltonian_basename   Basename of the source file; cheap cross-check.
      schema_version         ``DMRG_REFERENCE_SCHEMA_VERSION``.
      generated_at_iso       UTC ISO-8601 timestamp of the run.
      solver_module          Module name of the DMRG implementation.
      solver_revision        Best-effort git revision, ``None`` outside a tree.
      energy_history_tail    Last ``min(5, n_sweeps)`` per-sweep energies.
      notes                  Free-form notes (CLI ``--notes``), empty by default.
    """

    energy: float
    n_qubits: int
    n_terms: int
    bond_dim: int
    requested_bond_dim: Optional[int]
    n_sweeps: int
    max_sweeps: int
    converged: bool
    energy_tol: float
    final_trunc_err: float
    svd_min: float
    identity_offset: float
    compute_time_seconds: float
    device: str
    hamiltonian_hash: str
    hamiltonian_path: str
    hamiltonian_basename: str
    schema_version: int = DMRG_REFERENCE_SCHEMA_VERSION
    generated_at_iso: str = ""
    solver_module: str = "code.tenpy_dmrg"
    solver_revision: Optional[str] = None
    energy_history_tail: Tuple[float, ...] = field(default_factory=tuple)
    notes: str = ""

    def to_json_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable dict (``energy_history_tail`` → list)."""
        payload = asdict(self)
        payload["energy_history_tail"] = list(self.energy_history_tail)
        return payload


# ---------------------------------------------------------------------------
# Sidecar I/O
# ---------------------------------------------------------------------------


def dmrg_reference_sidecar_path(hamiltonian_path: os.PathLike | str) -> Path:
    """Return the canonical sidecar location for a Hamiltonian file.

    The sidecar always lives in the same directory as the Hamiltonian
    source, e.g. ``Hamiltonians/SpinlessHubbard_26x2_ladder_52qubits/dmrg_reference.json``.
    This is deliberately *not* under ``cache/ground_states/`` so DMRG
    reference artefacts cannot be confused with — or accidentally consumed
    by — the exact-state cache layer in
    ``PauliHamiltonianHelper._save_ground_state``.
    """
    return Path(hamiltonian_path).resolve().parent / DMRG_REFERENCE_SIDECAR_NAME


def hamiltonian_sha256(hamiltonian_path: os.PathLike | str) -> str:
    """Hex SHA-256 of the Hamiltonian source bytes.

    Used to detect a regenerated Hamiltonian against an old sidecar.
    """
    h = hashlib.sha256()
    with open(hamiltonian_path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_short_revision() -> Optional[str]:
    """Best-effort short git SHA for the repo containing this module."""
    repo_root = _PROJECT_ROOT
    head = repo_root / ".git" / "HEAD"
    if not head.exists():
        return None
    try:
        head_text = head.read_text(encoding="utf-8").strip()
        if head_text.startswith("ref:"):
            ref_path = repo_root / ".git" / head_text.split(" ", 1)[1].strip()
            if ref_path.exists():
                return ref_path.read_text(encoding="utf-8").strip()[:12]
            return None
        return head_text[:12]
    except OSError:
        return None


def save_dmrg_reference(
    metadata: DMRGReferenceMetadata,
    sidecar_path: os.PathLike | str,
) -> Path:
    """Atomically write the sidecar JSON file.

    Writes to a unique ``<name>.<rand>.tmp`` next to the destination then
    ``os.replace``-renames so a concurrent reader never observes a
    partial file and so two writers cannot collide on the tmp name.
    If serialization fails partway through, the tmp artefact is removed
    so it does not survive as orphaned junk in the Hamiltonian directory.

    Permission contract: the final sidecar is chmod'd to a
    umask-respecting ``0o666 & ~umask`` (typically ``0o644``).  This
    matches the historical ``open(path, 'w')`` behaviour and prevents a
    regression that would have made sidecars unreadable to other UIDs
    on shared cluster filesystems — ``tempfile.NamedTemporaryFile``
    defaults to ``0o600`` to protect sensitive temp data, which is the
    wrong policy for a reference-energy artefact meant to be consumed
    by sibling jobs / readers.
    """
    sidecar_path = Path(sidecar_path)
    sidecar_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=sidecar_path.parent,
            prefix=f".{sidecar_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as f:
            tmp_path = Path(f.name)
            json.dump(
                metadata.to_json_dict(),
                f,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            f.write("\n")
        # Restore umask-respecting permissions before the rename so
        # readers see the final mode atomically with the file appearing.
        current_umask = os.umask(0)
        os.umask(current_umask)
        os.chmod(tmp_path, 0o666 & ~current_umask)
        os.replace(tmp_path, sidecar_path)
    except BaseException:
        # Drop any half-written tmp before re-raising; otherwise a future
        # ``compute --force`` leaves behind inscrutable stale files.
        # ``BaseException`` deliberately also catches ``KeyboardInterrupt``
        # mid-write.
        if tmp_path is not None:
            try:
                tmp_path.unlink()
            except FileNotFoundError:
                pass
        raise
    return sidecar_path


def _coerce_path_like(value: Any) -> Optional[Path]:
    """Normalise a sidecar-path config value to ``Path`` or ``None``.

    Accepts ``str`` and ``os.PathLike``; everything else (booleans,
    numbers, dicts, …) returns ``None``.  This is defensive: a config
    loaded from JSON could expose ``dmrg_reference_path=False`` (e.g.
    "feature disabled") and the resolver must not crash on
    ``Path(False)``.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        # bool is os.PathLike-adjacent in some envs via int coercion;
        # explicitly reject so a JSON ``false`` cannot become Path("False").
        return None
    if isinstance(value, (str, os.PathLike)):
        text = os.fspath(value)
        if text == "":
            return None
        return Path(text)
    return None


def _resolve_project_relative_path(path: Path) -> Path:
    """Resolve a config path using the repository root for relative values."""
    if path.is_absolute():
        return path
    return _PROJECT_ROOT / path


def _resolve_hamiltonian_config_path(path: Optional[Path]) -> Optional[Path]:
    """Resolve ``config.hamiltonian_path`` for sidecar hash verification.

    ``run_config.validate_config`` usually canonicalises JSON-loaded paths to
    absolute paths when launched outside the repo. Direct callers can still
    construct ``ExperimentConfig`` with the repo-relative benchmark default, so the
    sidecar resolver mirrors the sidecar path rule and checks the project root.

    If a caller intentionally supplies a relative path that exists under the
    current working directory, keep that value; otherwise prefer the
    repo-relative path so CWD changes do not break hash verification.
    """
    if path is None or path.is_absolute():
        return path
    project_path = _resolve_project_relative_path(path)
    if project_path.exists() or not path.exists():
        return project_path
    return path


def _is_sha256_hex(value: str) -> bool:
    """Return True iff *value* looks like a SHA-256 hex digest."""
    return len(value) == 64 and all(ch in "0123456789abcdefABCDEF" for ch in value)


def _require_payload_key(payload: Dict[str, Any], name: str) -> Any:
    """Return a sidecar payload field or raise if schema-v1 omitted it."""
    if name not in payload:
        raise ValueError(f"{name} is required")
    return payload[name]


def _coerce_int_payload_field(
    payload: Dict[str, Any],
    name: str,
    *,
    default: Optional[int] = 0,
    minimum: Optional[int] = None,
    allow_none: bool = False,
) -> Optional[int]:
    raw = payload.get(name, default)
    if raw is None:
        if allow_none:
            return None
        raise ValueError(f"{name} is required")
    if isinstance(raw, bool):
        raise ValueError(f"{name} must be an integer, not bool")
    if isinstance(raw, (str, bytes)):
        raise ValueError(f"{name} must be a JSON integer, got {raw!r}")
    if isinstance(raw, float) and not raw.is_integer():
        raise ValueError(f"{name} must be an integer, got {raw!r}")
    value = int(raw)
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    return value


def _coerce_required_int_payload_field(
    payload: Dict[str, Any],
    name: str,
    *,
    minimum: Optional[int] = None,
) -> int:
    _require_payload_key(payload, name)
    value = _coerce_int_payload_field(payload, name, minimum=minimum)
    if value is None:
        raise ValueError(f"{name} is required")
    return value


def _coerce_required_nullable_int_payload_field(
    payload: Dict[str, Any],
    name: str,
    *,
    minimum: Optional[int] = None,
) -> Optional[int]:
    _require_payload_key(payload, name)
    return _coerce_int_payload_field(
        payload,
        name,
        default=None,
        minimum=minimum,
        allow_none=True,
    )


def _coerce_float_payload_field(
    payload: Dict[str, Any],
    name: str,
    *,
    default: float = 0.0,
    minimum: Optional[float] = None,
    strict_minimum: bool = False,
) -> float:
    raw = payload.get(name, default)
    if isinstance(raw, (str, bytes)):
        raise ValueError(f"{name} must be a JSON number, got {raw!r}")
    value = _coerce_finite_float(raw)
    if value is None:
        raise ValueError(f"{name} must be a finite float, got {raw!r}")
    if minimum is not None:
        if strict_minimum and value <= minimum:
            raise ValueError(f"{name} must be > {minimum}, got {value}")
        if not strict_minimum and value < minimum:
            raise ValueError(f"{name} must be >= {minimum}, got {value}")
    return value


def _coerce_required_float_payload_field(
    payload: Dict[str, Any],
    name: str,
    *,
    minimum: Optional[float] = None,
    strict_minimum: bool = False,
) -> float:
    _require_payload_key(payload, name)
    return _coerce_float_payload_field(
        payload,
        name,
        minimum=minimum,
        strict_minimum=strict_minimum,
    )


def _coerce_bool_payload_field(
    payload: Dict[str, Any],
    name: str,
    *,
    default: bool = False,
) -> bool:
    raw = payload.get(name, default)
    if not isinstance(raw, bool):
        raise ValueError(f"{name} must be a boolean, got {raw!r}")
    return raw


def _coerce_required_bool_payload_field(payload: Dict[str, Any], name: str) -> bool:
    _require_payload_key(payload, name)
    return _coerce_bool_payload_field(payload, name)


def _coerce_required_string_payload_field(
    payload: Dict[str, Any],
    name: str,
    *,
    allow_empty: bool = False,
) -> str:
    raw = _require_payload_key(payload, name)
    if not isinstance(raw, str):
        raise ValueError(f"{name} must be a string, got {raw!r}")
    if not allow_empty and raw == "":
        raise ValueError(f"{name} must be a non-empty string")
    return raw


def _coerce_optional_string_payload_field(
    payload: Dict[str, Any],
    name: str,
) -> Optional[str]:
    raw = _require_payload_key(payload, name)
    if raw is None:
        return None
    if not isinstance(raw, str):
        raise ValueError(f"{name} must be a string or null, got {raw!r}")
    return raw


def _coerce_energy_history_tail(value: Any) -> Tuple[float, ...]:
    """Validate the optional finite-float convergence-history tail."""
    if value is None:
        return ()
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
        raise ValueError("energy_history_tail must be a list of finite floats")
    if len(value) > 5:
        raise ValueError("energy_history_tail must contain at most 5 entries")
    tail: List[float] = []
    for idx, item in enumerate(value):
        if isinstance(item, (str, bytes)):
            raise ValueError(
                f"energy_history_tail[{idx}] must be a JSON number, got {item!r}"
            )
        energy = _coerce_finite_float(item)
        if energy is None:
            raise ValueError(
                f"energy_history_tail[{idx}] must be a finite float, got {item!r}"
            )
        tail.append(energy)
    return tuple(tail)


def _validate_optional_positive_int(value: Optional[int], name: str) -> Optional[int]:
    if value is None:
        return None
    return _validate_int_minimum(value, name, minimum=1)


def _validate_int_minimum(value: Any, name: str, *, minimum: int) -> int:
    if value is None:
        raise ValueError(f"{name} must be an integer, got None")
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer, not bool")
    if isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must be an integer, got {value!r}")
    if isinstance(value, float) and not value.is_integer():
        raise ValueError(f"{name} must be an integer, got {value!r}")
    coerced = int(value)
    if coerced < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {coerced}")
    return coerced


def _coerce_solver_energy_history_tail(value: Any) -> Tuple[float, ...]:
    """Validate the solver's in-memory convergence history.

    ``compute_ground_state_dmrg`` returns a Python list of floats.  Keep the
    wrapper strict here so malformed provenance from a future solver change
    cannot be serialized as if it were trustworthy sidecar metadata.
    """
    if value is None:
        return ()
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError("DMRG solver returned malformed energy_history")
    tail_values: List[float] = []
    tail_start = max(len(value) - 5, 0)
    for idx, item in enumerate(value[tail_start:], start=tail_start):
        if isinstance(item, (str, bytes)):
            raise ValueError(
                f"DMRG solver returned string energy_history[{idx}]={item!r}"
            )
        item_energy = _coerce_finite_float(item)
        if item_energy is None:
            raise ValueError(
                f"DMRG solver returned non-finite energy_history[{idx}]={item!r}"
            )
        tail_values.append(item_energy)
    return tuple(tail_values)


def _validate_positive_int(value: Any, name: str) -> int:
    """Coerce *value* to an ``int >= 1`` or raise ``ValueError``.

    Inlined rather than wrapping ``_validate_optional_positive_int`` +
    ``assert`` so a ``None`` from the lower-level solver surfaces as a
    clean ``ValueError`` (with the field name) instead of an
    ``AssertionError`` from this module.
    """
    return _validate_int_minimum(value, name, minimum=1)


def _validate_finite_float(
    value: float,
    name: str,
    *,
    minimum: Optional[float] = None,
    strict_minimum: bool = False,
) -> float:
    if isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must be finite, got {value!r}")
    coerced = _coerce_finite_float(value)
    if coerced is None:
        raise ValueError(f"{name} must be finite, got {value!r}")
    if minimum is not None:
        if strict_minimum and coerced <= minimum:
            raise ValueError(f"{name} must be > {minimum}, got {coerced}")
        if not strict_minimum and coerced < minimum:
            raise ValueError(f"{name} must be >= {minimum}, got {coerced}")
    return coerced


def _verify_native_sidecar_metadata(
    metadata: DMRGReferenceMetadata,
    *,
    sidecar_path: Path,
    hamiltonian_path: os.PathLike | str,
) -> bool:
    """Bind native fixed-sector references to adjacent model metadata."""
    declares_native = metadata.notes.startswith(_NATIVE_NOTES_PREFIX)
    try:
        native_route = _load_native_fermion_route(
            hamiltonian_path,
            n_qubits=metadata.n_qubits,
            n_terms=metadata.n_terms,
        )
    except (OSError, TypeError, ValueError) as exc:
        logger.warning(
            "DMRG reference sidecar at %s cannot verify native metadata: %s",
            sidecar_path,
            exc,
        )
        return False

    if native_route is None:
        if declares_native:
            logger.warning(
                "DMRG reference sidecar at %s declares a native fixed-sector "
                "route, but the adjacent metadata is not native-eligible; "
                "refusing to use",
                sidecar_path,
            )
            return False
        return True

    if not declares_native:
        logger.warning(
            "DMRG reference sidecar at %s is missing an unambiguous native "
            "provenance prefix; refusing to use",
            sidecar_path,
        )
        return False
    provenance_text = metadata.notes[len(_NATIVE_NOTES_PREFIX):].split(
        " | ",
        1,
    )[0]
    provenance: Dict[str, str] = {}
    for item in provenance_text.split(";"):
        key, separator, value = item.partition("=")
        if (
            separator != "="
            or not key
            or not value
            or key in provenance
        ):
            logger.warning(
                "DMRG reference sidecar at %s has malformed or duplicate "
                "native provenance tokens; refusing to use",
                sidecar_path,
            )
            return False
        provenance[key] = value

    required_provenance = {
        "route": "fermion_model",
        "model": "Spinless-Hubbard",
        "conserve": "N",
        "n_particles": str(native_route.n_particles),
    }
    if any(
        provenance.get(key) != value
        for key, value in required_provenance.items()
    ):
        logger.warning(
            "DMRG reference sidecar at %s has incorrect native-sector "
            "provenance; refusing to use",
            sidecar_path,
        )
        return False
    recorded_metadata_hash = provenance.get("metadata_sha256")
    if (
        not isinstance(recorded_metadata_hash, str)
        or not _is_sha256_hex(recorded_metadata_hash)
    ):
        logger.warning(
            "DMRG reference sidecar at %s has a missing or invalid "
            "metadata_sha256 provenance token; refusing to use",
            sidecar_path,
        )
        return False

    metadata_path = Path(hamiltonian_path).resolve().parent / "metadata.json"
    try:
        actual_metadata_hash = hashlib.sha256(metadata_path.read_bytes()).hexdigest()
    except OSError as exc:
        logger.warning(
            "Cannot hash native DMRG metadata %s for sidecar verification: %s",
            metadata_path,
            exc,
        )
        return False
    recorded_metadata_hash = recorded_metadata_hash.lower()
    if recorded_metadata_hash != actual_metadata_hash:
        logger.warning(
            "DMRG reference sidecar at %s is stale: metadata hash %s != "
            "current %s; refusing to use a fixed-sector scalar for changed "
            "model metadata",
            sidecar_path,
            recorded_metadata_hash[:12],
            actual_metadata_hash[:12],
        )
        return False

    return True


def load_dmrg_reference(
    sidecar_path: os.PathLike | str,
    *,
    hamiltonian_path: Optional[os.PathLike | str] = None,
    verify_hash: bool = True,
) -> Optional[DMRGReferenceMetadata]:
    """Load a sidecar JSON; return ``None`` if missing or unusable.

    Hash verification policy:

      - If ``hamiltonian_path`` is provided and ``verify_hash=True`` (the
        default), the loader recomputes the source SHA-256 and refuses to
        return the metadata if it disagrees with the sidecar's
        ``hamiltonian_hash``. This is the primary guard against a stale
        sidecar pointing at a regenerated Hamiltonian.
      - A native fixed-particle TeNPy sidecar is additionally bound to the
        adjacent ``metadata.json`` hash and sector provenance carried in its
        existing ``notes`` field. This preserves the locked schema while
        rejecting a scalar computed for stale lattice or filling metadata.
      - If ``hamiltonian_path`` is not provided, the loader still parses
        the sidecar and returns the metadata, but the caller is responsible
        for any cross-check it cares about.

    Schema verification: ``schema_version`` must be an ``int`` in the
    closed range ``[1, DMRG_REFERENCE_SCHEMA_VERSION]``.  Booleans are
    rejected explicitly (``bool`` ⊂ ``int`` in Python, so a JSON
    ``"schema_version": true`` would otherwise be silently coerced to
    ``1``); ``schema_version`` ``<= 0`` is rejected so a sidecar
    written without the field is not quietly treated as compatible.
    """
    sidecar_path = Path(sidecar_path)
    if not sidecar_path.exists():
        return None
    try:
        with open(sidecar_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("DMRG reference sidecar at %s is unreadable: %s", sidecar_path, e)
        return None
    if not isinstance(payload, dict):
        logger.warning("DMRG reference sidecar at %s is not a JSON object", sidecar_path)
        return None

    schema = payload.get("schema_version")
    # Reject bool explicitly: ``bool`` is a subclass of ``int`` in Python,
    # so a JSON ``"schema_version": true`` would otherwise be silently
    # coerced to ``1`` and accepted as the current schema version.  Also
    # reject any value below the lowest known schema (``1``) so a sidecar
    # written without a schema field (defaulted to ``0`` somewhere) is not
    # quietly treated as compatible.
    if (
        isinstance(schema, bool)
        or not isinstance(schema, int)
        or schema < 1
        or schema > DMRG_REFERENCE_SCHEMA_VERSION
    ):
        logger.warning(
            "DMRG reference sidecar at %s has schema_version=%r outside "
            "supported range [1, %d]; refusing to use",
            sidecar_path,
            schema,
            DMRG_REFERENCE_SCHEMA_VERSION,
        )
        return None

    raw_energy = payload.get("energy")
    if isinstance(raw_energy, (str, bytes)):
        logger.warning(
            "DMRG reference sidecar at %s has string 'energy'; ignoring",
            sidecar_path,
        )
        return None
    energy = _coerce_finite_float(raw_energy)
    if energy is None:
        logger.warning(
            "DMRG reference sidecar at %s missing/invalid 'energy'; ignoring",
            sidecar_path,
        )
        return None

    disk_hash = payload.get("hamiltonian_hash")
    if (
        not isinstance(disk_hash, str)
        or not disk_hash
        or not _is_sha256_hex(disk_hash)
    ):
        logger.warning(
            "DMRG reference sidecar at %s missing/invalid 'hamiltonian_hash'; ignoring",
            sidecar_path,
        )
        return None
    disk_hash = disk_hash.lower()

    if verify_hash and hamiltonian_path is not None:
        try:
            actual_hash = hamiltonian_sha256(hamiltonian_path)
        except OSError as e:
            logger.warning(
                "Cannot hash Hamiltonian %s for sidecar verification: %s",
                hamiltonian_path,
                e,
            )
            return None
        if actual_hash != disk_hash:
            logger.warning(
                "DMRG reference sidecar at %s is stale: hash %s != current %s. "
                "Re-run `python -m code.dmrg_reference compute --hamiltonian %s`.",
                sidecar_path,
                disk_hash[:12],
                actual_hash[:12],
                hamiltonian_path,
            )
            return None

    try:
        _require_payload_key(payload, "energy_history_tail")
        history_tail = _coerce_energy_history_tail(
            payload.get("energy_history_tail")
        )
        metadata = DMRGReferenceMetadata(
            energy=energy,
            n_qubits=_coerce_required_int_payload_field(
                payload, "n_qubits", minimum=1
            ),
            n_terms=_coerce_required_int_payload_field(
                payload, "n_terms", minimum=1
            ),
            bond_dim=_coerce_required_int_payload_field(
                payload, "bond_dim", minimum=1
            ),
            requested_bond_dim=_coerce_required_nullable_int_payload_field(
                payload,
                "requested_bond_dim",
                minimum=1,
            ),
            n_sweeps=_coerce_required_int_payload_field(
                payload, "n_sweeps", minimum=0
            ),
            max_sweeps=_coerce_required_int_payload_field(
                payload, "max_sweeps", minimum=1
            ),
            converged=_coerce_required_bool_payload_field(payload, "converged"),
            energy_tol=_coerce_required_float_payload_field(
                payload,
                "energy_tol",
                minimum=0.0,
                strict_minimum=True,
            ),
            final_trunc_err=_coerce_required_float_payload_field(
                payload, "final_trunc_err", minimum=0.0
            ),
            svd_min=_coerce_required_float_payload_field(
                payload, "svd_min", minimum=0.0
            ),
            identity_offset=_coerce_required_float_payload_field(
                payload, "identity_offset"
            ),
            compute_time_seconds=_coerce_required_float_payload_field(
                payload, "compute_time_seconds", minimum=0.0
            ),
            device=_coerce_required_string_payload_field(payload, "device"),
            hamiltonian_hash=disk_hash,
            hamiltonian_path=_coerce_required_string_payload_field(
                payload, "hamiltonian_path"
            ),
            hamiltonian_basename=_coerce_required_string_payload_field(
                payload, "hamiltonian_basename"
            ),
            schema_version=schema,
            generated_at_iso=_coerce_required_string_payload_field(
                payload, "generated_at_iso", allow_empty=True
            ),
            solver_module=_coerce_required_string_payload_field(
                payload, "solver_module"
            ),
            solver_revision=_coerce_optional_string_payload_field(
                payload, "solver_revision"
            ),
            energy_history_tail=history_tail,
            notes=_coerce_required_string_payload_field(
                payload, "notes", allow_empty=True
            ),
        )
    except (TypeError, ValueError) as e:
        logger.warning(
            "DMRG reference sidecar at %s has malformed fields: %s", sidecar_path, e
        )
        return None
    if (
        verify_hash
        and hamiltonian_path is not None
        and not _verify_native_sidecar_metadata(
            metadata,
            sidecar_path=sidecar_path,
            hamiltonian_path=hamiltonian_path,
        )
    ):
        return None
    return metadata


# ---------------------------------------------------------------------------
# Solver resolution
# ---------------------------------------------------------------------------

# Fully-qualified module name of the DMRG backend now in use. M-TENPY.4
# repoints the reference solver from the in-house torch Pauli-MPO DMRG
# (``code.pauli_mpo_dmrg``) to the released physics-tenpy backend
# (``code.tenpy_dmrg``). DMRG here is an OFFLINE precompute path and released
# TeNPy is NumPy/CPU-only — accepted for this workload. The two backends share
# the identical ``compute_ground_state_dmrg`` contract
# ((energy, dense_vector | None, info) with the same ``info`` keys/types), so
# this wrapper's energy-only invariant, identity-offset provenance, strict
# metadata coercions, hash-pinning, and atomic write are all unchanged.
DMRG_SOLVER_MODULE = "code.tenpy_dmrg"


def _tenpy_dmrg_module():
    """Return the ``tenpy_dmrg`` module object (the production DMRG backend).

    Resolved as the module (not a bound function) so the energy-only contract
    tests can ``monkeypatch.setattr`` its ``compute_ground_state_dmrg`` to
    inject malformed solver output and assert the wrapper's coercions fire.
    Production always uses the released physics-tenpy backend.
    """
    try:
        from . import tenpy_dmrg
    except ImportError:  # pragma: no cover - direct (non-package) import
        import tenpy_dmrg
    return tenpy_dmrg


@dataclass(frozen=True)
class _NativeFermionRoute:
    """Validated arguments for the TeNPy-native Spinless-Hubbard solver."""

    metadata_path: Path
    metadata_hash: str
    lattice: str
    lattice_size: Tuple[int, ...]
    L: Optional[int]
    Lx: Optional[int]
    Ly: Optional[int]
    n_sites: int
    n_particles: int
    t: float
    V: float
    mu: float

    @property
    def notes_prefix(self) -> str:
        size = "x".join(str(value) for value in self.lattice_size)
        if self.L is not None:
            dimensions = f"L={self.L};"
        else:
            dimensions = f"Lx={self.Lx};Ly={self.Ly};"
        return (
            "native_fermion_dmrg:"
            "route=fermion_model;"
            "model=Spinless-Hubbard;"
            f"lattice={self.lattice};"
            f"lattice_size={size};"
            f"{dimensions}"
            "boundary=open;"
            f"n_sites={self.n_sites};"
            "conserve=N;"
            f"n_particles={self.n_particles};"
            f"t={self.t:.17g};"
            f"V={self.V:.17g};"
            f"mu={self.mu:.17g};"
            f"metadata_sha256={self.metadata_hash}"
        )


def _spinless_metadata_error(metadata_path: Path, message: str) -> ValueError:
    return ValueError(
        f"Malformed Spinless-Hubbard metadata at {metadata_path}: {message}"
    )


def _load_native_fermion_route(
    hamiltonian_path: os.PathLike | str,
    *,
    n_qubits: int,
    n_terms: int,
) -> Optional[_NativeFermionRoute]:
    """Resolve an adjacent, explicitly eligible Spinless-Hubbard declaration.

    A missing metadata file, or a valid metadata object for another model,
    keeps the generic Pauli-MPO route. Historical and compact encodings also
    keep that route: native TeNPy dispatch requires an explicit Jordan-Wigner
    encoding, a one-site-per-qubit declaration, and an explicit fixed-particle
    sector. Once those eligibility fields are present, malformed native
    metadata is fatal because silently falling back would change the intended
    particle-number sector.
    """
    metadata_path = Path(hamiltonian_path).resolve().parent / "metadata.json"
    if not metadata_path.exists():
        return None

    try:
        metadata_bytes = metadata_path.read_bytes()
        payload = json.loads(metadata_bytes.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"Cannot read Hamiltonian metadata at {metadata_path}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise ValueError(
            f"Hamiltonian metadata at {metadata_path} must be a JSON object"
        )
    if payload.get("model") != "Spinless-Hubbard":
        return None

    encoding = payload.get("encoding")
    if not isinstance(encoding, str):
        return None
    encoding_key = "".join(
        character for character in encoding.casefold() if character.isalnum()
    )
    if encoding_key not in {"jordanwigner", "jw"}:
        return None
    if "half_filling" not in payload or "n_sites" not in payload:
        return None
    try:
        eligibility_n_sites = _validate_positive_int(
            payload.get("n_sites"), "metadata.n_sites"
        )
        eligibility_n_qubits = _validate_positive_int(
            payload.get("n_qubits"), "metadata.n_qubits"
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise _spinless_metadata_error(metadata_path, str(exc)) from exc
    if eligibility_n_sites != eligibility_n_qubits:
        raise _spinless_metadata_error(
            metadata_path,
            f"n_sites={eligibility_n_sites} does not match "
            f"n_qubits={eligibility_n_qubits}",
        )

    boundary = payload.get("boundary_condition")
    if not isinstance(boundary, str) or boundary.lower() != "open":
        raise _spinless_metadata_error(
            metadata_path,
            "boundary_condition must be 'open' for the native TeNPy route, "
            f"got {boundary!r}",
        )

    lattice_type = payload.get("lattice_type")
    if not isinstance(lattice_type, str):
        raise _spinless_metadata_error(
            metadata_path,
            f"lattice_type must be a string, got {lattice_type!r}",
        )
    lattice_key = lattice_type.lower()

    raw_lattice_size = payload.get("lattice_size")
    if not isinstance(raw_lattice_size, (list, tuple)):
        raise _spinless_metadata_error(
            metadata_path,
            f"lattice_size must be a list of positive integers, "
            f"got {raw_lattice_size!r}",
        )
    try:
        lattice_size = tuple(
            _validate_positive_int(value, f"lattice_size[{index}]")
            for index, value in enumerate(raw_lattice_size)
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise _spinless_metadata_error(metadata_path, str(exc)) from exc

    L: Optional[int] = None
    Lx: Optional[int] = None
    Ly: Optional[int] = None
    if lattice_key in {"chain", "line"}:
        if len(lattice_size) != 1:
            raise _spinless_metadata_error(
                metadata_path,
                "a chain/line lattice_size must contain exactly one length, "
                f"got {list(lattice_size)!r}",
            )
        lattice = "Chain"
        L = lattice_size[0]
        expected_sites = L
    elif lattice_key == "ladder":
        if len(lattice_size) != 2 or lattice_size[1] != 2:
            raise _spinless_metadata_error(
                metadata_path,
                "a ladder lattice_size must be [length, 2], "
                f"got {list(lattice_size)!r}",
            )
        lattice = "Ladder"
        L = lattice_size[0]
        expected_sites = 2 * L
    elif lattice_key == "square":
        if len(lattice_size) != 2:
            raise _spinless_metadata_error(
                metadata_path,
                "a square lattice_size must contain exactly two dimensions, "
                f"got {list(lattice_size)!r}",
            )
        lattice = "Square"
        Lx, Ly = lattice_size
        expected_sites = Lx * Ly
    else:
        raise _spinless_metadata_error(
            metadata_path,
            "lattice_type must be chain, line, ladder, or square, "
            f"got {lattice_type!r}",
        )

    try:
        declared_n_qubits = _validate_positive_int(
            payload.get("n_qubits"), "metadata.n_qubits"
        )
        declared_n_terms = _validate_positive_int(
            payload.get("n_terms"), "metadata.n_terms"
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise _spinless_metadata_error(metadata_path, str(exc)) from exc
    if declared_n_qubits != n_qubits:
        raise _spinless_metadata_error(
            metadata_path,
            f"n_qubits={declared_n_qubits} does not match the loaded "
            f"Hamiltonian n_qubits={n_qubits}",
        )
    if declared_n_terms != n_terms:
        raise _spinless_metadata_error(
            metadata_path,
            f"n_terms={declared_n_terms} does not match the loaded "
            f"Hamiltonian term count {n_terms}",
        )
    if expected_sites != n_qubits:
        raise _spinless_metadata_error(
            metadata_path,
            f"lattice_size defines {expected_sites} sites but n_qubits={n_qubits}",
        )

    if "n_sites" in payload:
        try:
            declared_n_sites = _validate_positive_int(
                payload.get("n_sites"), "metadata.n_sites"
            )
        except (TypeError, ValueError, OverflowError) as exc:
            raise _spinless_metadata_error(metadata_path, str(exc)) from exc
        if declared_n_sites != expected_sites:
            raise _spinless_metadata_error(
                metadata_path,
                f"n_sites={declared_n_sites} does not match lattice site "
                f"count {expected_sites}",
            )

    try:
        half_filling = _validate_int_minimum(
            payload.get("half_filling"),
            "metadata.half_filling",
            minimum=0,
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise _spinless_metadata_error(metadata_path, str(exc)) from exc
    if half_filling > expected_sites:
        raise _spinless_metadata_error(
            metadata_path,
            f"half_filling={half_filling} exceeds n_sites={expected_sites}",
        )
    if 2 * half_filling != expected_sites:
        raise _spinless_metadata_error(
            metadata_path,
            f"half_filling={half_filling} is inconsistent with "
            f"n_sites={expected_sites}",
        )

    parameters = payload.get("parameters")
    if not isinstance(parameters, dict):
        raise _spinless_metadata_error(
            metadata_path,
            f"parameters must be a JSON object, got {parameters!r}",
        )
    try:
        hopping = _validate_finite_float(
            parameters.get("hopping"), "parameters.hopping"
        )
        interaction = _validate_finite_float(
            parameters.get("nearest_neighbor_interaction"),
            "parameters.nearest_neighbor_interaction",
        )
        chemical_potential = _validate_finite_float(
            parameters.get("chemical_potential"),
            "parameters.chemical_potential",
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise _spinless_metadata_error(metadata_path, str(exc)) from exc

    return _NativeFermionRoute(
        metadata_path=metadata_path,
        metadata_hash=hashlib.sha256(metadata_bytes).hexdigest(),
        lattice=lattice,
        lattice_size=lattice_size,
        L=L,
        Lx=Lx,
        Ly=Ly,
        n_sites=expected_sites,
        n_particles=half_filling,
        # Generator coefficients are +hopping and +chemical_potential,
        # while FermionModel uses -t and -mu respectively.
        t=-hopping,
        V=interaction,
        mu=-chemical_potential,
    )


def _pauli_identity_offset(
    pauli_strings: Sequence[str],
    coefficients: Sequence[complex],
    n_qubits: int,
) -> float:
    """Validate a Pauli sum and return its real identity-only coefficient."""
    identity = "I" * n_qubits
    offset = 0.0 + 0.0j
    for index, (pauli, coefficient) in enumerate(
        zip(pauli_strings, coefficients)
    ):
        if not isinstance(pauli, str) or len(pauli) != n_qubits:
            raise ValueError(
                f"Pauli string {pauli!r} at index {index} must have "
                f"length n_qubits={n_qubits}"
            )
        if any(char not in "IXYZ" for char in pauli):
            raise ValueError(
                f"Pauli string {pauli!r} at index {index} contains an "
                "invalid character"
            )
        try:
            value = complex(coefficient)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                f"Coefficient {coefficient!r} at index {index} is not numeric"
            ) from exc
        real = _validate_finite_float(value.real, f"coefficients[{index}].real")
        imag = _validate_finite_float(value.imag, f"coefficients[{index}].imag")
        if abs(imag) > 1e-10 * max(abs(value), 1.0):
            raise ValueError(
                f"Coefficient {value!r} on Pauli term {pauli!r} has a "
                "non-negligible imaginary part"
            )
        if pauli == identity:
            offset += complex(real, imag)
    if abs(offset.imag) > 1e-10 * max(abs(offset), 1.0):
        raise ValueError(
            f"Identity-only Pauli coefficient {offset!r} is not real"
        )
    return _validate_finite_float(offset.real, "identity_offset")


# ---------------------------------------------------------------------------
# Computation entry point (energy-only)
# ---------------------------------------------------------------------------


def compute_dmrg_reference_energy(
    pauli_strings: List[str],
    coefficients: List[complex],
    n_qubits: int,
    *,
    hamiltonian_path: os.PathLike | str,
    bond_dim: Optional[int] = None,
    max_sweeps: int = 30,
    energy_tol: float = 1e-9,
    svd_min: float = 1e-10,
    initial_state: str = "neel",
    device: Optional[str] = None,
    seed: int = 0,
    notes: str = "",
    expected_hamiltonian_hash: Optional[str] = None,
) -> DMRGReferenceMetadata:
    """Run DMRG and return an energy-only reference record.

    Spinless-Hubbard Hamiltonians with a valid adjacent ``metadata.json`` are
    dispatched to the charge-conserving native fermion solver in the declared
    half-filling sector. Other Hamiltonians use
    ``tenpy_dmrg.compute_ground_state_dmrg``. Both routes request
    ``return_dense_vector=False`` so the dense state vector is never
    materialised. The MPS payload is discarded — this function does *not
    return or persist any full-state / MPS artefact.

    The returned metadata still needs to be persisted with
    ``save_dmrg_reference`` if the caller wants it to survive a process
    restart. When ``expected_hamiltonian_hash`` is supplied (the prepare
    path does this after loading the Hamiltonian file), the wrapper verifies
    the file still has that hash before running DMRG and verifies it has not
    changed again before returning metadata.
    ``prepare_dmrg_reference_for_hamiltonian`` is the combined one-shot
    entry point.
    """
    if expected_hamiltonian_hash is not None and not _is_sha256_hex(
        expected_hamiltonian_hash
    ):
        raise ValueError(
            "expected_hamiltonian_hash must be a SHA-256 hex digest, "
            f"got {expected_hamiltonian_hash!r}"
        )
    if expected_hamiltonian_hash is not None:
        expected_hamiltonian_hash = expected_hamiltonian_hash.lower()
    n_qubits = _validate_positive_int(n_qubits, "n_qubits")
    if len(pauli_strings) != len(coefficients):
        raise ValueError(
            "pauli_strings and coefficients must have the same length, "
            f"got {len(pauli_strings)} and {len(coefficients)}"
        )
    if not pauli_strings:
        raise ValueError("pauli_strings must contain at least one term")
    bond_dim = _validate_optional_positive_int(bond_dim, "bond_dim")
    max_sweeps = _validate_positive_int(max_sweeps, "max_sweeps")
    energy_tol = _validate_finite_float(
        energy_tol, "energy_tol", minimum=0.0, strict_minimum=True
    )
    svd_min = _validate_finite_float(
        svd_min, "svd_min", minimum=0.0, strict_minimum=False
    )

    hamiltonian_hash_before = hamiltonian_sha256(hamiltonian_path)
    if (
        expected_hamiltonian_hash is not None
        and hamiltonian_hash_before != expected_hamiltonian_hash
    ):
        raise RuntimeError(
            "Hamiltonian changed after it was loaded for DMRG reference "
            f"computation: expected hash {expected_hamiltonian_hash[:12]}, "
            f"current hash {hamiltonian_hash_before[:12]}. Refusing to "
            "write a sidecar whose scalar and source bytes may disagree."
        )

    # Resolve the TeNPy backend via the module object so the energy-only
    # contract tests can monkeypatch ``tenpy_dmrg.compute_ground_state_dmrg``.
    solver_module = DMRG_SOLVER_MODULE
    native_route = _load_native_fermion_route(
        hamiltonian_path,
        n_qubits=n_qubits,
        n_terms=len(pauli_strings),
    )
    native_identity_offset: Optional[float] = None
    effective_notes = notes
    if native_route is not None:
        if initial_state != "neel":
            raise ValueError(
                "Spinless-Hubbard native DMRG currently supports only "
                f"initial_state='neel', got {initial_state!r}"
            )
        native_identity_offset = _pauli_identity_offset(
            pauli_strings,
            coefficients,
            n_qubits,
        )
        effective_notes = native_route.notes_prefix
        if notes:
            effective_notes = f"{effective_notes} | {notes}"

    t0 = time.monotonic()
    tenpy_dmrg = _tenpy_dmrg_module()
    if native_route is None:
        energy, _vec, info = tenpy_dmrg.compute_ground_state_dmrg(
            pauli_strings,
            coefficients,
            n_qubits,
            bond_dim=bond_dim,
            max_sweeps=max_sweeps,
            energy_tol=energy_tol,
            svd_min=svd_min,
            initial_state=initial_state,
            return_dense_vector=False,
            device=device,
            seed=seed,
        )
    else:
        energy, _vec, info = tenpy_dmrg.compute_ground_state_fermion_dmrg(
            n_qubits=n_qubits,
            lattice=native_route.lattice,
            L=native_route.L,
            Lx=native_route.Lx,
            Ly=native_route.Ly,
            t=native_route.t,
            V=native_route.V,
            mu=native_route.mu,
            conserve="N",
            n_particles=native_route.n_particles,
            bond_dim=bond_dim,
            max_sweeps=max_sweeps,
            energy_tol=energy_tol,
            svd_min=svd_min,
            return_dense_vector=False,
            seed=seed,
        )
    compute_time = time.monotonic() - t0
    if _vec is not None:
        raise RuntimeError(
            "DMRG reference wrapper requested return_dense_vector=False but "
            "the solver returned a dense vector; refusing to continue."
        )
    if not isinstance(info, dict):
        raise ValueError(f"DMRG solver returned malformed info={info!r}")
    # Drop the in-memory MPS as soon as we have the metadata. The
    # reference path stores only the scalar; the MPS belongs to the
    # exact-state cache layer in ``PauliHamiltonianHelper``.
    info.pop("mps_numpy", None)

    final_chi = _validate_positive_int(info.get("final_chi", 0), "final_chi")
    n_sweeps = _validate_int_minimum(info.get("n_sweeps", 0), "n_sweeps", minimum=0)
    converged = info.get("converged", False)
    if not isinstance(converged, bool):
        raise ValueError(f"DMRG solver returned non-boolean converged={converged!r}")
    final_trunc_err = _validate_finite_float(
        info.get("final_trunc_err", 0.0),
        "final_trunc_err",
        minimum=0.0,
        strict_minimum=False,
    )
    solver_identity_offset = _validate_finite_float(
        info.get("identity_offset", 0.0),
        "identity_offset",
    )
    if native_identity_offset is None:
        identity_offset = solver_identity_offset
    else:
        if abs(solver_identity_offset) > 1e-12:
            raise ValueError(
                "Native fermion DMRG returned a non-zero identity_offset="
                f"{solver_identity_offset!r}; expected 0.0 before recording "
                "the Pauli representation's identity-only coefficient"
            )
        identity_offset = native_identity_offset

    tail = _coerce_solver_energy_history_tail(info.get("energy_history"))

    if isinstance(energy, (str, bytes)):
        raise ValueError(f"DMRG solver returned string energy={energy!r}")
    energy_value = _coerce_finite_float(energy)
    if energy_value is None:
        raise ValueError(f"DMRG solver returned non-finite energy={energy!r}")

    hamiltonian_hash_after = hamiltonian_sha256(hamiltonian_path)
    if hamiltonian_hash_after != hamiltonian_hash_before:
        raise RuntimeError(
            "Hamiltonian changed during DMRG reference computation "
            f"({hamiltonian_hash_before[:12]} -> {hamiltonian_hash_after[:12]}). "
            "Refusing to write a sidecar whose scalar and source bytes may disagree."
        )
    if native_route is not None:
        try:
            metadata_hash_after = hashlib.sha256(
                native_route.metadata_path.read_bytes()
            ).hexdigest()
        except OSError as exc:
            raise RuntimeError(
                "Spinless-Hubbard metadata became unreadable during DMRG "
                f"reference computation: {native_route.metadata_path}"
            ) from exc
        if metadata_hash_after != native_route.metadata_hash:
            raise RuntimeError(
                "Spinless-Hubbard metadata changed during DMRG reference "
                f"computation ({native_route.metadata_hash[:12]} -> "
                f"{metadata_hash_after[:12]}). Refusing to return provenance "
                "for an unstable native-fermion model declaration."
            )

    return DMRGReferenceMetadata(
        energy=energy_value,
        n_qubits=n_qubits,
        n_terms=int(len(pauli_strings)),
        bond_dim=final_chi,
        requested_bond_dim=bond_dim,
        n_sweeps=n_sweeps,
        max_sweeps=int(max_sweeps),
        converged=converged,
        energy_tol=float(energy_tol),
        final_trunc_err=final_trunc_err,
        svd_min=float(svd_min),
        identity_offset=identity_offset,
        compute_time_seconds=float(compute_time),
        # Released TeNPy is NumPy/CPU-only; ``device`` is an actual-device
        # provenance field, not a copy of the caller's ignored request.
        device="cpu",
        hamiltonian_hash=hamiltonian_hash_before,
        hamiltonian_path=str(hamiltonian_path),
        hamiltonian_basename=Path(hamiltonian_path).name,
        solver_module=str(solver_module),
        generated_at_iso=datetime.now(timezone.utc).isoformat(),
        solver_revision=_git_short_revision(),
        energy_history_tail=tail,
        notes=effective_notes,
    )


def _warn_if_cached_params_looser(
    *,
    cached: DMRGReferenceMetadata,
    requested_bond_dim: Optional[int],
    requested_max_sweeps: int,
    requested_energy_tol: float,
    sidecar_path: Path,
) -> None:
    """Log a single warning when a cached sidecar was produced with looser
    solver parameters than the current call requests.

    Three drift modes are flagged:

      - ``requested_bond_dim`` > cached requested bond dim (or cached used
        memory-autopick): caller asked for a tighter chi than the cache.
      - ``requested_max_sweeps`` > cached ``max_sweeps``: caller granted
        more sweep budget.
      - ``requested_energy_tol`` < cached ``energy_tol``: caller wants
        a stricter convergence threshold.

    The cache is still honoured — the energy is a valid reference
    regardless of the bond dim — but the operator is told how to force
    a tighter recompute.
    """
    drifts: List[str] = []
    cached_req_bd = cached.requested_bond_dim
    # When the cache used memory-autopick (``requested_bond_dim=None``) we
    # have no recorded cap to compare against, so fall back to the realised
    # ``cached.bond_dim`` — the autopicked χ may legitimately have been
    # larger than the caller's new explicit request, in which case we must
    # NOT warn about a "looser" cache.
    cached_effective_bd = (
        cached_req_bd if cached_req_bd is not None else cached.bond_dim
    )
    if (
        requested_bond_dim is not None
        and requested_bond_dim > cached_effective_bd
    ):
        cached_desc = (
            str(cached_req_bd)
            if cached_req_bd is not None
            else f"autopick={cached.bond_dim}"
        )
        drifts.append(
            f"bond_dim {requested_bond_dim} > cached {cached_desc}"
        )
    if requested_max_sweeps > cached.max_sweeps:
        drifts.append(
            f"max_sweeps {requested_max_sweeps} > cached {cached.max_sweeps}"
        )
    if requested_energy_tol < cached.energy_tol:
        drifts.append(
            f"energy_tol {requested_energy_tol:.1e} < cached "
            f"{cached.energy_tol:.1e}"
        )
    if not drifts:
        return
    logger.warning(
        "DMRG reference sidecar at %s was produced with looser solver "
        "parameters than this call requests (%s); honouring the cached "
        "energy (it is still a valid reference). Pass force_recompute=True "
        "(CLI: --force) to recompute with the tighter settings.",
        sidecar_path,
        "; ".join(drifts),
    )


def prepare_dmrg_reference_for_hamiltonian(
    hamiltonian_path: os.PathLike | str,
    *,
    bond_dim: Optional[int] = None,
    max_sweeps: int = 30,
    energy_tol: float = 1e-9,
    svd_min: float = 1e-10,
    initial_state: str = "neel",
    device: Optional[str] = None,
    seed: int = 0,
    notes: str = "",
    force_recompute: bool = False,
) -> Tuple[DMRGReferenceMetadata, Path]:
    """One-shot path: load Hamiltonian → run DMRG → write sidecar.

    Idempotent unless ``force_recompute=True``: if a hash-matching sidecar
    already exists at the canonical location, returns the loaded metadata
    without re-running DMRG. The sidecar path is returned alongside so
    callers can log the artefact location.

    Solver-parameter drift is reported but NOT auto-recomputed: if the
    cached sidecar was produced with looser settings (smaller
    ``requested_bond_dim``, fewer ``max_sweeps``, or larger ``energy_tol``)
    than the current call, a warning is emitted recommending
    ``force_recompute=True``.  The energy is still a valid reference
    regardless, so the cached scalar is honoured to keep the path
    idempotent.
    """
    hamiltonian_path = Path(hamiltonian_path)
    bond_dim = _validate_optional_positive_int(bond_dim, "bond_dim")
    max_sweeps = _validate_positive_int(max_sweeps, "max_sweeps")
    energy_tol = _validate_finite_float(
        energy_tol, "energy_tol", minimum=0.0, strict_minimum=True
    )
    svd_min = _validate_finite_float(
        svd_min, "svd_min", minimum=0.0, strict_minimum=False
    )
    sidecar_path = dmrg_reference_sidecar_path(hamiltonian_path)

    if not force_recompute:
        existing = load_dmrg_reference(
            sidecar_path, hamiltonian_path=hamiltonian_path, verify_hash=True
        )
        if existing is not None:
            _warn_if_cached_params_looser(
                cached=existing,
                requested_bond_dim=bond_dim,
                requested_max_sweeps=max_sweeps,
                requested_energy_tol=energy_tol,
                sidecar_path=sidecar_path,
            )
            logger.info(
                "DMRG reference sidecar already up-to-date at %s "
                "(E=%.10f, chi=%d, sweeps=%d, converged=%s)",
                sidecar_path,
                existing.energy,
                existing.bond_dim,
                existing.n_sweeps,
                existing.converged,
            )
            return existing, sidecar_path

    try:
        from .pauli_hamiltonian_helper import PauliHamiltonianHelper
    except ImportError:
        from pauli_hamiltonian_helper import PauliHamiltonianHelper

    hamiltonian_hash_before_load = hamiltonian_sha256(hamiltonian_path)
    helper = PauliHamiltonianHelper(str(hamiltonian_path))
    hamiltonian_hash_after_load = hamiltonian_sha256(hamiltonian_path)
    if hamiltonian_hash_after_load != hamiltonian_hash_before_load:
        raise RuntimeError(
            "Hamiltonian changed while loading terms for DMRG reference "
            f"computation ({hamiltonian_hash_before_load[:12]} -> "
            f"{hamiltonian_hash_after_load[:12]}). Refusing to compute a "
            "sidecar from unstable source bytes."
        )
    metadata = compute_dmrg_reference_energy(
        helper.pauli_str_list,
        helper.w_list,
        _validate_positive_int(helper.n_qubits, "helper.n_qubits"),
        hamiltonian_path=hamiltonian_path,
        bond_dim=bond_dim,
        max_sweeps=max_sweeps,
        energy_tol=energy_tol,
        svd_min=svd_min,
        initial_state=initial_state,
        device=device,
        seed=seed,
        notes=notes,
        expected_hamiltonian_hash=hamiltonian_hash_after_load,
    )
    save_dmrg_reference(metadata, sidecar_path)
    logger.info(
        "Wrote DMRG reference sidecar to %s (E=%.10f, chi=%d, sweeps=%d, "
        "converged=%s, trunc=%.3e, %.1fs)",
        sidecar_path,
        metadata.energy,
        metadata.bond_dim,
        metadata.n_sweeps,
        metadata.converged,
        metadata.final_trunc_err,
        metadata.compute_time_seconds,
    )
    return metadata, sidecar_path


def prepare_benchmark_dmrg_reference(
    *,
    base_dir: Optional[os.PathLike | str] = None,
    bond_dim: Optional[int] = None,
    max_sweeps: int = 30,
    energy_tol: float = 1e-9,
    svd_min: float = 1e-10,
    device: Optional[str] = None,
    seed: int = 0,
    notes: str = "",
    force_recompute: bool = False,
) -> Tuple[DMRGReferenceMetadata, Path]:
    """Convenience wrapper that targets the canonical 52-qubit benchmark Hamiltonian.

    Resolves the benchmark Hamiltonian path via ``hubbard_loader`` so the caller
    does not need to thread the directory layout through the CLI.
    """
    try:
        from .hubbard_loader import BENCHMARK_DIR_NAME, BENCHMARK_MAPPER
    except ImportError:
        from hubbard_loader import BENCHMARK_DIR_NAME, BENCHMARK_MAPPER  # type: ignore[no-redef]

    if base_dir is not None:
        hamiltonians_dir = Path(base_dir)
    else:
        # Mirror ``hubbard_loader._resolve_hamiltonians_dir`` — FlowMeas
        # repo root is ``Path(__file__).resolve().parent.parent``.
        hamiltonians_dir = _PROJECT_ROOT / "Hamiltonians"
    hamiltonian_path = hamiltonians_dir / BENCHMARK_DIR_NAME / f"{BENCHMARK_MAPPER}.txt"
    if not hamiltonian_path.exists():
        raise FileNotFoundError(
            f"benchmark Hamiltonian not found at {hamiltonian_path}. "
            "Run `python -m code.hubbard_loader generate-benchmark` first."
        )
    return prepare_dmrg_reference_for_hamiltonian(
        hamiltonian_path,
        bond_dim=bond_dim,
        max_sweeps=max_sweeps,
        energy_tol=energy_tol,
        svd_min=svd_min,
        device=device,
        seed=seed,
        notes=notes,
        force_recompute=force_recompute,
    )


# ---------------------------------------------------------------------------
# Config resolution
# ---------------------------------------------------------------------------


def resolve_dmrg_reference_energy(config: Any) -> Optional[float]:
    """Return the DMRG reference scalar to attach to *config*, or ``None``.

    Resolution order:

      1. ``config.dmrg_reference_energy`` — explicit scalar override.
         Wins if it normalises to a finite float (booleans / NaN / inf
         are rejected by ``coerce_dmrg_reference_energy``).
      2. ``config.dmrg_reference_path`` — explicit sidecar JSON. Loaded
         and hash-verified against ``config.hamiltonian_path`` if both
         are present, else loaded unverified.
      3. Otherwise ``None``.

    Auto-discovery from ``<hamiltonian_dir>/dmrg_reference.json`` is
    intentionally *not* performed by default: a config that wants the
    sidecar must opt in by setting ``dmrg_reference_path``. This keeps
    the resolver explicit; the benchmark overrides supplied by
    ``hubbard_loader.get_benchmark_run_config_overrides`` populate the path
    automatically for the canonical workload.
    """
    # ``_coerce_finite_float`` at module top is an alias for
    # ``validation_tier.coerce_dmrg_reference_energy`` — same coercion
    # contract (rejects bool, NaN, inf, non-numeric).  No lazy import
    # needed here because the alias is already resolved at module load.
    explicit = _coerce_finite_float(
        getattr(config, "dmrg_reference_energy", None)
    )
    if explicit is not None:
        return explicit

    sidecar_path = _coerce_path_like(
        getattr(config, "dmrg_reference_path", None)
    )
    if sidecar_path is None:
        return None
    if not sidecar_path.is_absolute():
        # Resolve relative to the project root so the same config string
        # works whether the trainer launches from the repo root or from
        # a SLURM scratch dir.
        sidecar_path = _resolve_project_relative_path(sidecar_path)

    hamiltonian_attr = _coerce_path_like(
        getattr(config, "hamiltonian_path", None)
    )
    hamiltonian_attr = _resolve_hamiltonian_config_path(hamiltonian_attr)
    metadata = load_dmrg_reference(
        sidecar_path,
        hamiltonian_path=hamiltonian_attr,
        verify_hash=hamiltonian_attr is not None,
    )
    if metadata is None:
        return None
    return metadata.energy


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _cli_main(argv: Optional[List[str]] = None) -> int:
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(
        prog="python -m code.dmrg_reference",
        description="Produce / inspect energy-only DMRG reference sidecars.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    compute_p = sub.add_parser(
        "compute",
        help="Run DMRG on a Hamiltonian file and write the sidecar.",
    )
    compute_p.add_argument("--hamiltonian", required=True, type=str)
    compute_p.add_argument("--bond-dim", type=int, default=None)
    compute_p.add_argument("--max-sweeps", type=int, default=30)
    compute_p.add_argument("--energy-tol", type=float, default=1e-9)
    compute_p.add_argument("--svd-min", type=float, default=1e-10)
    compute_p.add_argument("--device", type=str, default=None)
    compute_p.add_argument("--seed", type=int, default=0)
    compute_p.add_argument("--notes", type=str, default="")
    compute_p.add_argument("--force", action="store_true",
                           help="Recompute even if a hash-matching sidecar exists.")

    benchmark_p = sub.add_parser(
        "compute-benchmark",
        help="Run DMRG on the canonical 52-qubit benchmark Hamiltonian.",
    )
    benchmark_p.add_argument("--bond-dim", type=int, default=None)
    benchmark_p.add_argument("--max-sweeps", type=int, default=30)
    benchmark_p.add_argument("--energy-tol", type=float, default=1e-9)
    benchmark_p.add_argument("--svd-min", type=float, default=1e-10)
    benchmark_p.add_argument("--device", type=str, default=None)
    benchmark_p.add_argument("--seed", type=int, default=0)
    benchmark_p.add_argument("--notes", type=str, default="")
    benchmark_p.add_argument("--force", action="store_true")

    show_p = sub.add_parser(
        "show",
        help="Print an existing sidecar; optionally verify against a Hamiltonian.",
    )
    show_p.add_argument("sidecar", type=str)
    show_p.add_argument("--hamiltonian", type=str, default=None,
                        help="Optional Hamiltonian path for hash verification.")

    args = parser.parse_args(argv)

    if args.command == "compute":
        metadata, sidecar_path = prepare_dmrg_reference_for_hamiltonian(
            args.hamiltonian,
            bond_dim=args.bond_dim,
            max_sweeps=args.max_sweeps,
            energy_tol=args.energy_tol,
            svd_min=args.svd_min,
            device=args.device,
            seed=args.seed,
            notes=args.notes,
            force_recompute=args.force,
        )
        print(f"Sidecar: {sidecar_path}")
        print(json.dumps(metadata.to_json_dict(), indent=2, sort_keys=True))
        return 0

    if args.command == "compute-benchmark":
        metadata, sidecar_path = prepare_benchmark_dmrg_reference(
            bond_dim=args.bond_dim,
            max_sweeps=args.max_sweeps,
            energy_tol=args.energy_tol,
            svd_min=args.svd_min,
            device=args.device,
            seed=args.seed,
            notes=args.notes,
            force_recompute=args.force,
        )
        print(f"Sidecar: {sidecar_path}")
        print(json.dumps(metadata.to_json_dict(), indent=2, sort_keys=True))
        return 0

    if args.command == "show":
        metadata = load_dmrg_reference(
            args.sidecar,
            hamiltonian_path=args.hamiltonian,
            verify_hash=args.hamiltonian is not None,
        )
        if metadata is None:
            print(f"No usable sidecar at {args.sidecar}", flush=True)
            return 1
        print(json.dumps(metadata.to_json_dict(), indent=2, sort_keys=True))
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(_cli_main())
