"""Standardized loading and validation for 50+ qubit Hubbard Hamiltonians.

This module is the single entry point for loading the frozen benchmark workload
(52-qubit spin-less 26×2-ladder Hubbard) and any other large Hubbard
Hamiltonians that share the same on-disk format.

Canonical metadata for the benchmark is recorded as constants so that every
consumer in the training and evaluation stack sees exactly the same
workload definition without re-deriving it from Qiskit Nature.

Design rules:
  - No O(2^n) full-state evaluation on the default large-system path.
  - No exact diagonalization for 50+ qubit systems.
  - Loading must be reproducible and fast; generation is a separate step.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from .pauli_hamiltonian_helper import PauliHamiltonianHelper


# ---------------------------------------------------------------------------
# Canonical benchmark workload definition
# ---------------------------------------------------------------------------

BENCHMARK_LATTICE_ROWS = 26
BENCHMARK_LATTICE_COLS = 2
BENCHMARK_N_SITES = BENCHMARK_LATTICE_ROWS * BENCHMARK_LATTICE_COLS  # 52
BENCHMARK_N_QUBITS = BENCHMARK_N_SITES  # spinless → 1 qubit per site
BENCHMARK_MODEL = "Spinless-Hubbard"
BENCHMARK_LATTICE_TYPE = "ladder"
BENCHMARK_LATTICE_SIZE: Tuple[int, int] = (BENCHMARK_LATTICE_ROWS, BENCHMARK_LATTICE_COLS)
BENCHMARK_BOUNDARY_CONDITION = "open"
BENCHMARK_HOPPING = -1.0
BENCHMARK_NEAREST_NEIGHBOR_INTERACTION = 1.0
BENCHMARK_CHEMICAL_POTENTIAL = 0.0
BENCHMARK_MAPPER = "jw"
BENCHMARK_ENCODING = "Jordan-Wigner"
BENCHMARK_HALF_FILLING = BENCHMARK_N_SITES // 2  # 26 fermions

BENCHMARK_DIR_NAME = (
    f"SpinlessHubbard_{BENCHMARK_LATTICE_ROWS}x{BENCHMARK_LATTICE_COLS}"
    f"_{BENCHMARK_LATTICE_TYPE}_{BENCHMARK_N_QUBITS}qubits"
)

LARGE_SYSTEM_QUBIT_THRESHOLD = 26
"""Aligned with ``main.EXACT_FULL_STATE_QUBIT_LIMIT``.  Systems at or
above this boundary must not use O(2^n) full-state evaluation."""

# ---------------------------------------------------------------------------
# Canonical run-config defaults
# ---------------------------------------------------------------------------
#
# These defaults must be loadable without importing ``code.main`` (avoiding the
# import cycle) and must always force the scalable-large evaluator path, so a
# caller that forgets to override ``large_hubbard_mode`` still gets a safe run.
# Keep ``BENCHMARK_DEFAULT_EVALUATOR_MODE`` in sync with
# ``code.main.EVALUATOR_MODE_SCALABLE_LARGE``.

BENCHMARK_HAMILTONIAN_RELATIVE_PATH = f"Hamiltonians/{BENCHMARK_DIR_NAME}/{BENCHMARK_MAPPER}.txt"
BENCHMARK_DEFAULT_RESULTS_DIR = f"results_hubbard/{BENCHMARK_DIR_NAME}"
BENCHMARK_DEFAULT_LARGE_HUBBARD_MODE = True
BENCHMARK_DEFAULT_ASYNC_EVAL = True
BENCHMARK_DEFAULT_MEASUREMENT_BACKEND = "auto"
BENCHMARK_DEFAULT_EVALUATOR_MODE = "scalable_large"

# Canonical sidecar location for the benchmark DMRG-backed reference
# energy.  Lives next to the Hamiltonian source so it is colocated with
# the workload it describes, and explicitly *not* under
# ``cache/ground_states/`` (which is reserved for exact-state /
# MPS payloads written by ``PauliHamiltonianHelper``).  Populated by
# ``python -m code.dmrg_reference compute-benchmark`` and read by
# ``dmrg_reference.resolve_dmrg_reference_energy`` at training startup.
BENCHMARK_DEFAULT_DMRG_REFERENCE_PATH = (
    f"Hamiltonians/{BENCHMARK_DIR_NAME}/dmrg_reference.json"
)

# Cache-key discipline.
#
# ``PauliHamiltonianHelper._get_cache_path()`` builds cache namespaces as
# ``{molecule}_{n_qubits}q_{hamiltonian_hash}_{transformation}`` where
# ``molecule`` is the Hamiltonian directory name and ``transformation``
# is the file stem.  For the benchmark workload that produces the stable
# prefix below; the hash portion changes only if the on-disk Hamiltonian
# terms or coefficients change, so cache hits are reproducible across
# reruns of the same frozen workload.  Above ``LARGE_SYSTEM_QUBIT_THRESHOLD``
# ``_save_ground_state`` refuses to write qubit-statevector / exact-vector
# artefacts unless the caller explicitly opts in via
# ``allow_exact_solver=True``, so the canonical large-system load path
# cannot accidentally write full-state cache entries under this
# namespace.
BENCHMARK_CACHE_NAMESPACE_PREFIX = f"{BENCHMARK_DIR_NAME}_{BENCHMARK_N_QUBITS}q"

_VALID_PAULI_CHARS = frozenset("IXYZ")


@dataclass(frozen=True)
class HubbardWorkloadMetadata:
    """Immutable record of all reproducibility-critical parameters."""

    model: str
    lattice_type: str
    lattice_size: Tuple[int, int]
    boundary_condition: str
    hopping: float
    nearest_neighbor_interaction: float
    chemical_potential: float
    mapper: str
    encoding: str
    n_qubits: int
    n_sites: int
    half_filling: int
    n_terms: Optional[int] = None
    dir_name: Optional[str] = None
    hamiltonian_path: Optional[str] = None
    load_time_seconds: Optional[float] = None
    generation_timestamp: Optional[str] = None
    cache_namespace_prefix: Optional[str] = None
    default_evaluator_mode: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "model": self.model,
            "lattice_type": self.lattice_type,
            "lattice_size": list(self.lattice_size),
            "boundary_condition": self.boundary_condition,
            "parameters": {
                "hopping": self.hopping,
                "nearest_neighbor_interaction": self.nearest_neighbor_interaction,
                "chemical_potential": self.chemical_potential,
            },
            "mapper": self.mapper,
            "encoding": self.encoding,
            "n_qubits": self.n_qubits,
            "n_sites": self.n_sites,
            "half_filling": self.half_filling,
        }
        if self.n_terms is not None:
            d["n_terms"] = self.n_terms
        if self.dir_name is not None:
            d["dir_name"] = self.dir_name
        if self.hamiltonian_path is not None:
            d["hamiltonian_path"] = self.hamiltonian_path
        if self.load_time_seconds is not None:
            d["load_time_seconds"] = self.load_time_seconds
        if self.generation_timestamp is not None:
            d["generation_timestamp"] = self.generation_timestamp
        if self.cache_namespace_prefix is not None:
            d["cache_namespace_prefix"] = self.cache_namespace_prefix
        if self.default_evaluator_mode is not None:
            d["default_evaluator_mode"] = self.default_evaluator_mode
        return d


def get_benchmark_metadata() -> HubbardWorkloadMetadata:
    """Return the canonical metadata for the frozen 52-qubit benchmark workload."""
    return HubbardWorkloadMetadata(
        model=BENCHMARK_MODEL,
        lattice_type=BENCHMARK_LATTICE_TYPE,
        lattice_size=BENCHMARK_LATTICE_SIZE,
        boundary_condition=BENCHMARK_BOUNDARY_CONDITION,
        hopping=BENCHMARK_HOPPING,
        nearest_neighbor_interaction=BENCHMARK_NEAREST_NEIGHBOR_INTERACTION,
        chemical_potential=BENCHMARK_CHEMICAL_POTENTIAL,
        mapper=BENCHMARK_MAPPER,
        encoding=BENCHMARK_ENCODING,
        n_qubits=BENCHMARK_N_QUBITS,
        n_sites=BENCHMARK_N_SITES,
        half_filling=BENCHMARK_HALF_FILLING,
        dir_name=BENCHMARK_DIR_NAME,
        cache_namespace_prefix=BENCHMARK_CACHE_NAMESPACE_PREFIX,
        default_evaluator_mode=BENCHMARK_DEFAULT_EVALUATOR_MODE,
    )


def _dmrg_reference_path_for_hamiltonian(hamiltonian_path: str) -> str:
    """Return the sidecar path colocated with a Hamiltonian source file."""
    return str(Path(hamiltonian_path).with_name("dmrg_reference.json"))


def get_benchmark_run_config_overrides(
    *,
    hamiltonian_path: Optional[str] = None,
    results_dir: Optional[str] = None,
    dmrg_reference_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Return canonical ``ExperimentConfig`` overrides for the benchmark workload.

    The returned dict is suitable for ``ExperimentConfig(**overrides)``
    and pins the cross-cutting workload defaults that must hold
    regardless of any individual JSON config:

      - ``large_hubbard_mode=True``  → ``evaluator_mode == "scalable_large"``
      - ``async_eval=True``          → keeps the training loop unblocked
      - ``measurement_backend="auto"`` → CT backend on CUDA, legacy on CPU
      - ``hamiltonian_path``         → frozen on-disk benchmark Hamiltonian
      - ``results_dir``              → ``results_hubbard/<dir>`` by default
      - ``dmrg_reference_path``      → ``Hamiltonians/<dir>/dmrg_reference.json``
                                        sidecar by default.  If the
                                        sidecar is missing (or hash-mismatched)
                                        the training run stays on the
                                        ``structural`` validation tier and
                                        reports flag the missing scalar.

    These fields are the minimum set required to keep the workload on
    the scalable-large path; callers may layer additional non-conflicting
    fields (max_depth, beta, lr, …) on top.

    A caller MUST NOT override ``large_hubbard_mode`` to False — the
    52-qubit workload cannot run safely on the exact-small path
    (defense-in-depth ``assert_full_state_eval_safe`` would reject it,
    but the canonical config refuses to even produce that combination).
    """
    resolved_hamiltonian_path = hamiltonian_path or BENCHMARK_HAMILTONIAN_RELATIVE_PATH
    resolved_dmrg_reference_path = (
        dmrg_reference_path
        if dmrg_reference_path is not None
        else _dmrg_reference_path_for_hamiltonian(resolved_hamiltonian_path)
    )
    return {
        "hamiltonian_path": resolved_hamiltonian_path,
        "results_dir": results_dir or BENCHMARK_DEFAULT_RESULTS_DIR,
        "large_hubbard_mode": BENCHMARK_DEFAULT_LARGE_HUBBARD_MODE,
        "async_eval": BENCHMARK_DEFAULT_ASYNC_EVAL,
        "measurement_backend": BENCHMARK_DEFAULT_MEASUREMENT_BACKEND,
        "dmrg_reference_path": resolved_dmrg_reference_path,
    }


def get_benchmark_cache_namespace_prefix() -> str:
    """Return the canonical cache-namespace prefix for the benchmark workload.

    Cache paths produced by ``PauliHamiltonianHelper._get_cache_path()``
    for the benchmark workload begin with this prefix; the trailing portion
    (``_{hash}_{mapper}``) tracks term-level Hamiltonian content.
    """
    return BENCHMARK_CACHE_NAMESPACE_PREFIX


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

@dataclass
class LoadValidation:
    """Result of post-load validation checks."""

    ok: bool = True
    n_terms: int = 0
    n_qubits: int = 0
    load_time_seconds: float = 0.0
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "n_terms": self.n_terms,
            "n_qubits": self.n_qubits,
            "load_time_seconds": self.load_time_seconds,
            "warnings": self.warnings,
            "errors": self.errors,
        }


def _expected_term_count_range(n_rows: int, n_cols: int) -> Tuple[int, int]:
    """Return the (min, max) expected Pauli term count for a spinless
    Hubbard ladder / square lattice with JW mapping.

    For an open-boundary rows×cols lattice the fermionic Hamiltonian
    has O(sites) hopping + NN-interaction + on-site terms.  After JW
    mapping each fermionic term becomes a short Pauli string (≤4 body)
    but simplification can merge or cancel some.  The range here is
    deliberately generous so it catches catastrophic generation errors
    without rejecting valid Qiskit output across versions.
    """
    n_sites = n_rows * n_cols
    lower = n_sites  # at least one term per site
    upper = 12 * n_sites  # generous ceiling
    return lower, upper


def validate_loaded_hamiltonian(
    pauli_str_list: List[str],
    w_list: List[Any],
    n_qubits: int,
    expected_metadata: Optional[HubbardWorkloadMetadata] = None,
    load_time_seconds: float = 0.0,
) -> LoadValidation:
    """Run structural checks on a freshly-loaded Hamiltonian."""
    v = LoadValidation(
        n_terms=len(pauli_str_list),
        n_qubits=n_qubits,
        load_time_seconds=load_time_seconds,
    )

    if len(pauli_str_list) == 0:
        v.errors.append("Hamiltonian has zero terms")
        v.ok = False
        return v

    if len(pauli_str_list) != len(w_list):
        v.errors.append(
            f"Pauli list length ({len(pauli_str_list)}) != weight list "
            f"length ({len(w_list)})"
        )
        v.ok = False

    lengths = {len(p) for p in pauli_str_list}
    if len(lengths) != 1:
        v.errors.append(f"Inconsistent Pauli string lengths: {lengths}")
        v.ok = False
    else:
        actual_len = next(iter(lengths))
        if actual_len != n_qubits:
            v.errors.append(
                f"Pauli string length ({actual_len}) != "
                f"n_qubits ({n_qubits})"
            )
            v.ok = False

    bad_labels = [
        p for p in pauli_str_list if not all(c in _VALID_PAULI_CHARS for c in p)
    ]
    if bad_labels:
        sample = bad_labels[:3]
        v.errors.append(
            f"{len(bad_labels)} label(s) contain non-Pauli characters "
            f"(valid: I, X, Y, Z); first bad: {sample}"
        )
        v.ok = False

    if expected_metadata is not None:
        if expected_metadata.n_qubits != n_qubits:
            v.errors.append(
                f"Expected n_qubits={expected_metadata.n_qubits}, "
                f"got {n_qubits}"
            )
            v.ok = False

        if expected_metadata.lattice_size is not None:
            lo, hi = _expected_term_count_range(*expected_metadata.lattice_size)
            if not (lo <= len(pauli_str_list) <= hi):
                v.warnings.append(
                    f"Term count {len(pauli_str_list)} outside expected "
                    f"range [{lo}, {hi}] for lattice "
                    f"{expected_metadata.lattice_size}"
                )

    if load_time_seconds > 60.0:
        v.warnings.append(
            f"Load time ({load_time_seconds:.1f}s) exceeds 60s threshold"
        )

    for msg in v.errors:
        logger.error("Hamiltonian validation FAIL: %s", msg)
    for msg in v.warnings:
        logger.warning("Hamiltonian validation WARN: %s", msg)

    return v


# ---------------------------------------------------------------------------
# Disk-metadata cross-validation
# ---------------------------------------------------------------------------

def _validate_disk_metadata_against_canonical(
    disk_meta: Dict[str, Any],
    helper: Any,
    validation: LoadValidation,
) -> None:
    """Compare on-disk ``metadata.json`` fields against canonical constants
    and the actually-loaded Hamiltonian, appending errors to *validation*.

    Every canonical field that defines the frozen benchmark workload is checked.
    Mismatches are errors (not warnings) because the benchmark loader promises
    reproducible, recorded metadata.
    """
    disk_nq = disk_meta.get("n_qubits")
    if disk_nq is not None and disk_nq != helper.n_qubits:
        validation.errors.append(
            f"metadata.json n_qubits ({disk_nq}) != "
            f"loaded n_qubits ({helper.n_qubits})"
        )
        validation.ok = False

    disk_nt = disk_meta.get("n_terms")
    if disk_nt is not None and disk_nt != len(helper.pauli_str_list):
        validation.errors.append(
            f"metadata.json n_terms ({disk_nt}) != "
            f"loaded n_terms ({len(helper.pauli_str_list)})"
        )
        validation.ok = False

    _canonical_checks: List[Tuple[str, Any, Any]] = [
        ("mapper", disk_meta.get("mapper"), BENCHMARK_MAPPER),
        ("model", disk_meta.get("model"), BENCHMARK_MODEL),
        ("lattice_type", disk_meta.get("lattice_type"), BENCHMARK_LATTICE_TYPE),
        ("boundary_condition", disk_meta.get("boundary_condition"), BENCHMARK_BOUNDARY_CONDITION),
        ("encoding", disk_meta.get("encoding"), BENCHMARK_ENCODING),
    ]
    for field_name, disk_val, canonical_val in _canonical_checks:
        if disk_val is not None and disk_val != canonical_val:
            validation.errors.append(
                f"metadata.json {field_name} ({disk_val!r}) != "
                f"canonical benchmark {field_name} ({canonical_val!r})"
            )
            validation.ok = False

    disk_lattice = disk_meta.get("lattice_size")
    if disk_lattice is not None and tuple(disk_lattice) != BENCHMARK_LATTICE_SIZE:
        validation.errors.append(
            f"metadata.json lattice_size ({disk_lattice}) != "
            f"canonical benchmark lattice_size ({list(BENCHMARK_LATTICE_SIZE)})"
        )
        validation.ok = False

    disk_params = disk_meta.get("parameters", {})
    _param_checks: List[Tuple[str, Any, float]] = [
        ("hopping", disk_params.get("hopping"), BENCHMARK_HOPPING),
        ("nearest_neighbor_interaction",
         disk_params.get("nearest_neighbor_interaction"),
         BENCHMARK_NEAREST_NEIGHBOR_INTERACTION),
        ("chemical_potential",
         disk_params.get("chemical_potential"),
         BENCHMARK_CHEMICAL_POTENTIAL),
    ]
    for pname, disk_val, canonical_val in _param_checks:
        if disk_val is not None and disk_val != canonical_val:
            validation.errors.append(
                f"metadata.json parameters.{pname} ({disk_val}) != "
                f"canonical benchmark ({canonical_val})"
            )
            validation.ok = False

    benchmark_lattice_name = (
        f"Ladder{BENCHMARK_LATTICE_ROWS}x{BENCHMARK_LATTICE_COLS}_{BENCHMARK_BOUNDARY_CONDITION}"
    )
    disk_ln = disk_meta.get("lattice_name")
    if disk_ln is not None and disk_ln != benchmark_lattice_name:
        validation.errors.append(
            f"metadata.json lattice_name ({disk_ln!r}) != "
            f"canonical benchmark lattice_name ({benchmark_lattice_name!r})"
        )
        validation.ok = False


def _validate_disk_metadata_fields(
    disk_meta: Dict[str, Any],
    helper: Any,
    validation: LoadValidation,
    *,
    expected_mapper: Optional[str] = None,
) -> None:
    """Lighter cross-validation for the general-purpose loader.

    Checks ``n_qubits``, ``n_terms``, and ``mapper`` from disk metadata
    against the actually-loaded Hamiltonian, appending warnings.
    """
    disk_nq = disk_meta.get("n_qubits")
    if disk_nq is not None and disk_nq != helper.n_qubits:
        validation.warnings.append(
            f"metadata.json n_qubits ({disk_nq}) != "
            f"loaded n_qubits ({helper.n_qubits})"
        )

    disk_nt = disk_meta.get("n_terms")
    if disk_nt is not None and disk_nt != len(helper.pauli_str_list):
        validation.warnings.append(
            f"metadata.json n_terms ({disk_nt}) != "
            f"loaded n_terms ({len(helper.pauli_str_list)})"
        )

    disk_mapper = disk_meta.get("mapper")
    if (
        expected_mapper is not None
        and disk_mapper is not None
        and disk_mapper != expected_mapper
    ):
        validation.warnings.append(
            f"metadata.json mapper ({disk_mapper!r}) != "
            f"requested mapper ({expected_mapper!r})"
        )


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def _resolve_hamiltonians_dir(base_dir: Optional[Path] = None) -> Path:
    """Return the ``Hamiltonians/`` directory, auto-detecting the project root."""
    if base_dir is not None:
        return Path(base_dir)
    here = Path(__file__).resolve().parent
    project_root = here.parent
    return project_root / "Hamiltonians"


def load_hamiltonian_metadata(hamiltonian_dir: Path) -> Optional[Dict[str, Any]]:
    """Load ``metadata.json`` from a Hamiltonian directory, if present."""
    meta_path = hamiltonian_dir / "metadata.json"
    if not meta_path.exists():
        return None
    with open(meta_path, "r") as f:
        return json.load(f)


def _require_loaded_n_qubits(
    helper: "PauliHamiltonianHelper",
    hamiltonian_path: Path,
) -> int:
    """Return the parsed qubit count, rejecting empty/malformed Hamiltonians."""
    if helper.n_qubits is None:
        raise ValueError(
            f"Hamiltonian file {hamiltonian_path} did not define any Pauli terms"
        )
    return helper.n_qubits


def load_benchmark_hamiltonian(
    base_dir: Optional[Path] = None,
    *,
    validate: bool = True,
) -> Tuple["PauliHamiltonianHelper", HubbardWorkloadMetadata, LoadValidation]:
    """Load the canonical 52-qubit spinless 26×2-ladder benchmark Hamiltonian.

    Returns
    -------
    helper: PauliHamiltonianHelper
        Loaded helper with Pauli list and weights; no ground-state
        vector or exact energy computed.
    metadata: HubbardWorkloadMetadata
        Canonical metadata record.
    validation: LoadValidation
        Structural checks on the loaded data.

    Raises
    ------
    FileNotFoundError
        If the Hamiltonian file has not been generated yet.
    RuntimeError
        If a downstream caller explicitly requests full-state evaluation
        through PauliHamiltonianHelper.
    """
    try:
        from .pauli_hamiltonian_helper import PauliHamiltonianHelper
    except ImportError:
        from pauli_hamiltonian_helper import PauliHamiltonianHelper

    ham_dir = _resolve_hamiltonians_dir(base_dir) / BENCHMARK_DIR_NAME
    jw_path = ham_dir / f"{BENCHMARK_MAPPER}.txt"

    if not jw_path.exists():
        raise FileNotFoundError(
            f"benchmark Hamiltonian not found at {jw_path}. "
            f"Run 'python -m code.hubbard_loader generate-benchmark' to create it."
        )

    t0 = time.monotonic()
    helper = PauliHamiltonianHelper(str(jw_path))
    load_time = time.monotonic() - t0
    n_qubits = _require_loaded_n_qubits(helper, jw_path)

    meta = get_benchmark_metadata()

    validation = LoadValidation(
        n_terms=len(helper.pauli_str_list),
        n_qubits=n_qubits,
        load_time_seconds=load_time,
    )

    if validate:
        validation = validate_loaded_hamiltonian(
            helper.pauli_str_list,
            helper.w_list,
            n_qubits,
            expected_metadata=meta,
            load_time_seconds=load_time,
        )

        disk_meta = load_hamiltonian_metadata(ham_dir)
        if disk_meta is None:
            validation.errors.append(
                f"metadata.json missing from {ham_dir}; "
                f"benchmark workload requires recorded metadata"
            )
            validation.ok = False
        else:
            _validate_disk_metadata_against_canonical(
                disk_meta, helper, validation
            )

        if not validation.ok:
            raise ValueError(
                "benchmark Hamiltonian failed validation: "
                + "; ".join(validation.errors)
            )

    from dataclasses import replace
    meta = replace(
        meta,
        n_qubits=n_qubits,
        n_terms=len(helper.pauli_str_list),
        hamiltonian_path=str(jw_path),
        load_time_seconds=load_time,
        cache_namespace_prefix=BENCHMARK_CACHE_NAMESPACE_PREFIX,
        default_evaluator_mode=BENCHMARK_DEFAULT_EVALUATOR_MODE,
    )

    logger.info(
        "Loaded benchmark Hamiltonian: %d qubits, %d terms in %.3fs",
        n_qubits,
        len(helper.pauli_str_list),
        load_time,
    )

    return helper, meta, validation


def load_hubbard_hamiltonian(
    hamiltonian_dir: Path,
    *,
    mapper: str = "jw",
    validate: bool = True,
) -> Tuple["PauliHamiltonianHelper", Optional[Dict[str, Any]], LoadValidation]:
    """Load any Hubbard Hamiltonian from disk with validation.

    This is the general-purpose loader for non-benchmark Hubbard Hamiltonians
    that follow the same on-disk layout (``{mapper}.txt`` + ``metadata.json``).
    """
    try:
        from .pauli_hamiltonian_helper import PauliHamiltonianHelper
    except ImportError:
        from pauli_hamiltonian_helper import PauliHamiltonianHelper

    hamiltonian_dir = Path(hamiltonian_dir)
    jw_path = hamiltonian_dir / f"{mapper}.txt"

    if not jw_path.exists():
        raise FileNotFoundError(f"Hamiltonian file not found: {jw_path}")

    t0 = time.monotonic()
    helper = PauliHamiltonianHelper(str(jw_path))
    load_time = time.monotonic() - t0
    n_qubits = _require_loaded_n_qubits(helper, jw_path)

    disk_meta = load_hamiltonian_metadata(hamiltonian_dir)

    validation = validate_loaded_hamiltonian(
        helper.pauli_str_list,
        helper.w_list,
        n_qubits,
        load_time_seconds=load_time,
    )

    if validate and disk_meta is not None:
        _validate_disk_metadata_fields(
            disk_meta, helper, validation, expected_mapper=mapper
        )

    if validate and not validation.ok:
        raise ValueError(
            f"Hamiltonian at {jw_path} failed validation: "
            + "; ".join(validation.errors)
        )

    logger.info(
        "Loaded Hubbard Hamiltonian: %d qubits, %d terms in %.3fs from %s",
        n_qubits,
        len(helper.pauli_str_list),
        load_time,
        hamiltonian_dir,
    )

    return helper, disk_meta, validation


# ---------------------------------------------------------------------------
# Generator (one-shot — runs only when the on-disk file is missing)
# ---------------------------------------------------------------------------

def generate_benchmark_hamiltonian(
    base_dir: Optional[Path] = None,
) -> Tuple[Path, HubbardWorkloadMetadata]:
    """Generate and save the 52-qubit benchmark Hamiltonian to disk.

    Requires ``qiskit-nature``.  This is intentionally separate from the
    loader so that CI / training never imports Qiskit at load time.
    """
    try:
        from .generate_hubbard_hamiltonians import (
            generate_spinless_hubbard_hamiltonian,
            save_hamiltonian,
        )
    except ImportError:
        from generate_hubbard_hamiltonians import (
            generate_spinless_hubbard_hamiltonian,
            save_hamiltonian,
        )
    from datetime import datetime, timezone

    ham_dir = _resolve_hamiltonians_dir(base_dir) / BENCHMARK_DIR_NAME
    jw_path = ham_dir / f"{BENCHMARK_MAPPER}.txt"

    if jw_path.exists():
        logger.info("benchmark Hamiltonian already exists at %s — skipping generation", jw_path)
        helper, meta, _ = load_benchmark_hamiltonian(base_dir)
        return jw_path, meta

    logger.info(
        "Generating benchmark Hamiltonian: %dx%d spinless Hubbard ladder (%d qubits)…",
        BENCHMARK_LATTICE_ROWS,
        BENCHMARK_LATTICE_COLS,
        BENCHMARK_N_QUBITS,
    )

    t0 = time.monotonic()
    ham_dict, gen_metadata = generate_spinless_hubbard_hamiltonian(
        lattice_size=BENCHMARK_LATTICE_SIZE,
        hopping=BENCHMARK_HOPPING,
        nearest_neighbor_interaction=BENCHMARK_NEAREST_NEIGHBOR_INTERACTION,
        chemical_potential=BENCHMARK_CHEMICAL_POTENTIAL,
        boundary_condition=BENCHMARK_BOUNDARY_CONDITION,
        mapper=BENCHMARK_MAPPER,
    )
    gen_time = time.monotonic() - t0

    gen_metadata["lattice_type"] = BENCHMARK_LATTICE_TYPE
    gen_metadata["lattice_name"] = (
        f"Ladder{BENCHMARK_LATTICE_ROWS}x{BENCHMARK_LATTICE_COLS}_{BENCHMARK_BOUNDARY_CONDITION}"
    )
    gen_metadata["encoding"] = BENCHMARK_ENCODING
    gen_metadata["n_sites"] = BENCHMARK_N_SITES
    gen_metadata["half_filling"] = BENCHMARK_HALF_FILLING
    gen_metadata["generation_timestamp"] = datetime.now(timezone.utc).isoformat()
    gen_metadata["generation_time_seconds"] = round(gen_time, 3)

    save_hamiltonian(ham_dict, gen_metadata, ham_dir)

    meta = HubbardWorkloadMetadata(
        model=BENCHMARK_MODEL,
        lattice_type=BENCHMARK_LATTICE_TYPE,
        lattice_size=BENCHMARK_LATTICE_SIZE,
        boundary_condition=BENCHMARK_BOUNDARY_CONDITION,
        hopping=BENCHMARK_HOPPING,
        nearest_neighbor_interaction=BENCHMARK_NEAREST_NEIGHBOR_INTERACTION,
        chemical_potential=BENCHMARK_CHEMICAL_POTENTIAL,
        mapper=BENCHMARK_MAPPER,
        encoding=BENCHMARK_ENCODING,
        n_qubits=gen_metadata["n_qubits"],
        n_sites=BENCHMARK_N_SITES,
        half_filling=BENCHMARK_HALF_FILLING,
        n_terms=gen_metadata["n_terms"],
        dir_name=BENCHMARK_DIR_NAME,
        hamiltonian_path=str(jw_path),
        generation_timestamp=gen_metadata["generation_timestamp"],
        cache_namespace_prefix=BENCHMARK_CACHE_NAMESPACE_PREFIX,
        default_evaluator_mode=BENCHMARK_DEFAULT_EVALUATOR_MODE,
    )

    logger.info(
        "Generated benchmark Hamiltonian: %d qubits, %d terms in %.3fs → %s",
        meta.n_qubits,
        meta.n_terms,
        gen_time,
        jw_path,
    )

    return jw_path, meta


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _cli_main() -> None:
    import argparse
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(
        description="Standardized Hubbard Hamiltonian loader / generator"
    )
    sub = parser.add_subparsers(dest="command")

    gen_p = sub.add_parser(
        "generate-benchmark",
        help="Generate the canonical 52-qubit benchmark Hamiltonian on disk",
    )
    gen_p.add_argument("--output-dir", type=str, default=None)

    load_p = sub.add_parser(
        "load-benchmark",
        help="Load and validate the benchmark Hamiltonian",
    )
    load_p.add_argument("--base-dir", type=str, default=None)

    info_p = sub.add_parser(
        "info",
        help="Print canonical benchmark metadata",
    )

    run_p = sub.add_parser(
        "run-config-defaults",
        help="Print canonical run-config overrides for the benchmark workload",
    )

    args = parser.parse_args()

    if args.command == "generate-benchmark":
        base = Path(args.output_dir) if args.output_dir else None
        path, meta = generate_benchmark_hamiltonian(base)
        print(f"Generated: {path}")
        print(json.dumps(meta.to_dict(), indent=2))

    elif args.command == "load-benchmark":
        base = Path(args.base_dir) if args.base_dir else None
        helper, meta, validation = load_benchmark_hamiltonian(base)
        print(json.dumps(meta.to_dict(), indent=2))
        print(f"\nValidation: {'PASS' if validation.ok else 'FAIL'}")
        print(json.dumps(validation.to_dict(), indent=2))

    elif args.command == "info":
        meta = get_benchmark_metadata()
        print(json.dumps(meta.to_dict(), indent=2))

    elif args.command == "run-config-defaults":
        print(json.dumps(get_benchmark_run_config_overrides(), indent=2))

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    _cli_main()
