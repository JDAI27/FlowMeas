#!/usr/bin/env python3
#main.py

"""
Main experiment runner for GFlowNet quantum circuit optimization with energy estimation.
Enhanced with asynchronous evaluation support for improved performance.

ADAPTED: Now supports true asynchronous evaluation using multiprocessing.
"""

import os
import sys
import json
import time
import asyncio
import numpy as np
import torch
import re
import logging
import multiprocessing as mp
from multiprocessing import Queue, Process
import signal
import queue
import fcntl
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import asdict, is_dataclass
import threading
from collections import defaultdict, deque

# Import the GFN modules
try:
    # Try relative imports first (when run as module)
    from .GFNs import EfficientGFNTrainer, get_device, exponential_reward_fn, SamplingMode, default_reward_fn, log_reward_fn, _METRICS_TRIM_SLACK
    from .pauli_hamiltonian_helper import PauliHamiltonianHelper
    from .energy_estimator import EnergyEstimator, BatchElementEnergyResult
    from .cost_computer import detect_stabilizer_terms
    from . import full_state_guard
    from . import validation_tier
    from . import dmrg_reference
except ImportError:
    # Fall back to absolute imports (when run directly)
    from GFNs import EfficientGFNTrainer, get_device, exponential_reward_fn, SamplingMode, default_reward_fn, log_reward_fn, _METRICS_TRIM_SLACK
    from pauli_hamiltonian_helper import PauliHamiltonianHelper
    from energy_estimator import EnergyEstimator, BatchElementEnergyResult
    from cost_computer import detect_stabilizer_terms
    import full_state_guard
    import validation_tier
    import dmrg_reference

# ExperimentConfig + config-coercion helpers live in config.py (a lightweight
# leaf module). Re-exported here so existing ``from (.)main import ExperimentConfig``
# importers keep working. _SAMPLING_MODE_ALIAS_WARNED is re-exported by identity
# (same set object) so the warn-once dedup stays shared.
try:
    from .config import (
        EVALUATOR_MODE_EXACT_SMALL, EVALUATOR_MODE_SCALABLE_LARGE,
        _coerce_bool_config, _coerce_optional_bool_config,
        _SAMPLING_MODE_DYNAMIC_ACTIVE, _SAMPLING_MODE_STATIC_FULL, _SAMPLING_MODE_BUCKETED,
        _SAMPLING_MODE_VALUES, _SAMPLING_MODE_ALIASES, _SAMPLING_MODE_ALIAS_WARNED,
        _coerce_sampling_mode_config, _legacy_sampler_alias, _resolve_sampling_mode_controls,
        resolve_evaluator_mode, ExperimentConfig,
    )
except ImportError:
    from config import (
        EVALUATOR_MODE_EXACT_SMALL, EVALUATOR_MODE_SCALABLE_LARGE,
        _coerce_bool_config, _coerce_optional_bool_config,
        _SAMPLING_MODE_DYNAMIC_ACTIVE, _SAMPLING_MODE_STATIC_FULL, _SAMPLING_MODE_BUCKETED,
        _SAMPLING_MODE_VALUES, _SAMPLING_MODE_ALIASES, _SAMPLING_MODE_ALIAS_WARNED,
        _coerce_sampling_mode_config, _legacy_sampler_alias, _resolve_sampling_mode_controls,
        resolve_evaluator_mode, ExperimentConfig,
    )


# Result-codec types live in result_types.py (a leaf module). Re-exported here
# so existing ``from (.)main import SimulationResult/NumpyEncoder/...`` importers
# keep working; re-export is by identity (same class objects).
try:
    from .result_types import (
        SimulationResult, _migrate_simulation_result_dict, _migrate_result_record_dict,
        _extended_result_from_record, ExtendedBatchElementEnergyResult,
        ScalableLargeEvaluationReport, EvaluationEntryPointResult, NumpyEncoder,
        EVALUATOR_RESULT_TYPE_EXACT, EVALUATOR_RESULT_TYPE_SCALABLE_LARGE_REPORT,
        EVALUATOR_RESULT_TYPE_ERROR, SCALABLE_LARGE_EVALUATION_JSON,
        SCALABLE_LARGE_EVALUATION_JSONL,
    )
except ImportError:
    from result_types import (
        SimulationResult, _migrate_simulation_result_dict, _migrate_result_record_dict,
        _extended_result_from_record, ExtendedBatchElementEnergyResult,
        ScalableLargeEvaluationReport, EvaluationEntryPointResult, NumpyEncoder,
        EVALUATOR_RESULT_TYPE_EXACT, EVALUATOR_RESULT_TYPE_SCALABLE_LARGE_REPORT,
        EVALUATOR_RESULT_TYPE_ERROR, SCALABLE_LARGE_EVALUATION_JSON,
        SCALABLE_LARGE_EVALUATION_JSONL,
    )


# Async plotting + result-JSON reporting lives in reporting.py (report-cadence
# only). Re-exported here so existing main importers keep working.
try:
    from .reporting import (
        AsyncReporter, save_results_safely, _to_report_float, _coefficient_abs_summary,
    )
except ImportError:
    from reporting import (
        AsyncReporter, save_results_safely, _to_report_float, _coefficient_abs_summary,
    )


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


def _verbose_replay_logging_enabled(gfn) -> bool:
    """Whether replay may emit tensor-valued diagnostic log lines."""
    return bool(getattr(gfn, "debug", False)) or logging.getLogger().isEnabledFor(
        logging.DEBUG
    )


# Global reference to trainer for signal handling (preemption checkpoint saving)
# This is set during run_experiment and can be accessed by external signal handlers
_current_trainer = None
_current_results_dir = None

# ``_current_trainer is None`` is overloaded: it holds both *before* the loop
# starts and *after* it ends, so a post-loop SIGUSR1 would take the handler's
# ``scontrol requeue + exit 99`` path and resurrect a finished job at the
# previous checkpoint. ``_loop_finalized`` disambiguates — True once the loop
# has exited normally, so post-loop signals skip requeue.
#
# ORDERING INVARIANT: ``_loop_finalized`` MUST be set True *before*
# ``_current_trainer = None``. Signal handlers run between bytecode boundaries,
# so the reverse order leaves a window that reads as the pre-init branch.
_loop_finalized = False

# ``_loop_finalized`` alone does not make skipping requeue safe: the final
# canonical checkpoint is written *after* the loop returns, so a SIGUSR1 in
# between would exit 143 without requeue and lose the completed training state.
# ``_final_checkpoint_persisted`` closes that gap — post-loop no-requeue is safe
# only once it is True (the final checkpoint was written, or a periodic save
# already covers the final update).
_final_checkpoint_persisted = False

# Graceful-shutdown coordination, owned here in ``code.main`` rather than in
# ``code.run_config``. Reason: the entrypoint is invoked as
# ``python3 -m code.run_config``, which loads ``run_config`` as ``__main__``.
# A separate ``from code import run_config as _rc`` (the previous design)
# imports a *second* module instance under its canonical name, with its own
# ``_shutdown_requested = False`` — the handler in ``__main__`` would flip
# its own copy while the training loop polled the canonical copy and never
# saw the signal. Owning the flags here means both the handler (which
# imports ``code.main`` lazily already) and the loop (same module) operate
# on the same object regardless of how the entrypoint was launched.
_shutdown_requested = False
_shutdown_is_warning_signal = False


def request_shutdown(is_warning_signal: bool) -> None:
    """Flip the shared shutdown flags. Called from the signal handler.

    Centralising the flag mutation here (rather than ``run_config`` setting
    its own module globals) means the handler and the loop always agree on
    state regardless of whether ``run_config`` was loaded as ``__main__``
    or as ``code.run_config``.

    The warning bit is **sticky**: once a SIGUSR1 has
    been seen, a later SIGTERM (often what slurm sends right after the
    USR1@120 warning when walltime actually expires) must not flip the
    bit back to ``False`` — that would route a preempt through the
    no-requeue path. We OR in the new value, so SIGUSR1 → SIGTERM still
    requeues.
    """
    global _shutdown_requested, _shutdown_is_warning_signal
    _shutdown_requested = True
    _shutdown_is_warning_signal = bool(_shutdown_is_warning_signal) or bool(is_warning_signal)


def shutdown_exit_code() -> int:
    """Return the canonical exit code for the current shutdown state.

    99 if a SIGUSR1 was observed at any point (sticky → requeue path),
    143 for SIGTERM/manual-cancel paths. Callers should only invoke
    ``sys.exit`` after running any cleanup they own.
    """
    return 99 if _shutdown_is_warning_signal else 143


# Registry of live async evaluator processes so an
# exception out of ``run_experiment`` between ``evaluator_process.start()``
# and the normal cleanup block doesn't leak a non-daemon child.
# ``run_config.main``'s broad except handler drains this registry, and so
# does the safe-point shutdown branch.
_evaluator_cleanup_registry: list = []


def _register_evaluator_for_cleanup(process, checkpoint_queue) -> None:
    _evaluator_cleanup_registry.append((process, checkpoint_queue))


def _unregister_evaluator(process) -> None:
    """Remove ``process`` from the cleanup registry by identity. Idempotent.

    Removal is by value, not by index. If ``drain_registered_evaluators``
    or another concurrent pop ran between the copy and the removal, an index
    could already be stale, whereas ``remove`` just raises ``ValueError``,
    which we treat as already-removed.
    """
    for entry in list(_evaluator_cleanup_registry):
        p, _q = entry
        if p is process:
            try:
                _evaluator_cleanup_registry.remove(entry)
            except ValueError:
                # Already removed by a concurrent ``drain`` — ok.
                pass
            return


def _drain_one_evaluator(process, checkpoint_queue) -> None:
    """STOP/join/terminate/kill/reap an evaluator. Idempotent and exception-safe."""
    if process is None or not process.is_alive():
        return
    logging.info("Draining live evaluator process before exit")
    try:
        if checkpoint_queue is not None:
            try:
                checkpoint_queue.put('STOP')
            except Exception:
                pass
        process.join(timeout=10.0)
        if process.is_alive():
            logging.warning("Evaluator didn't STOP in 10s; terminating")
            process.terminate()
            process.join(timeout=5.0)
            if process.is_alive():
                logging.error("Evaluator did not respond to terminate; killing")
                process.kill()
                # Always ``join`` after ``kill`` so the OS
                # reaps the child and ``exitcode`` becomes readable.
                # Without this, ``exitcode`` can stay ``None`` and
                # downstream "non-zero exit" detection silently skips.
                process.join(timeout=5.0)
    except Exception as e:
        logging.error(f"_drain_one_evaluator errored: {e}")


def drain_registered_evaluators() -> None:
    """Drain every registered evaluator; safe to call multiple times."""
    while _evaluator_cleanup_registry:
        process, q = _evaluator_cleanup_registry.pop()
        _drain_one_evaluator(process, q)


def complete_shutdown_after_safe_save(is_warning_signal: bool) -> "Optional[int]":
    """Finish the shutdown sequence after the main loop wrote a safe-point save.

    Returns the exit code; the loop is responsible for actually calling
    ``sys.exit`` after running its async-evaluator cleanup
    so we don't bypass non-daemon child shutdown.
    """
    import subprocess
    job_id = os.environ.get('SLURM_JOB_ID')
    if is_warning_signal and job_id:
        try:
            result = subprocess.run(
                ['scontrol', 'requeue', job_id],
                capture_output=True, text=True, timeout=10, check=False,
            )
            if result.returncode == 0:
                logging.info(f"Explicit requeue requested for SLURM_JOB_ID={job_id}")
            else:
                logging.warning(
                    f"scontrol requeue exited {result.returncode}: {result.stderr.strip()}"
                )
        except (FileNotFoundError, subprocess.TimeoutExpired) as e:
            logging.warning(f"Could not invoke scontrol requeue: {e}")
    # SIGUSR1 → 99 (preempt + requeue); SIGTERM/everything else → 143
    # (128 + SIGTERM, no requeue). Returned to the caller so async-evaluator
    # cleanup can run BEFORE ``sys.exit``.
    return 99 if is_warning_signal else 143


# filename token charset excludes ``.``
# so a stray ``signame="USR1.pth"`` cannot produce a double extension
# and confuse downstream tools that match on ``*.pth``. Only alnum,
# underscore, and hyphen are allowed. The same pattern is used to
# sanitize the fallback value so a typo'd fallback
# (e.g. "lo/cal") cannot leak a path separator into the filename.
# (collapsed into a single constant — duplicating
# the regex invited future divergence with no benefit.)
_CHECKPOINT_TOKEN_CHARS = re.compile(r"[^A-Za-z0-9_-]+")
_CHECKPOINT_TOKEN_FALLBACK = _CHECKPOINT_TOKEN_CHARS
# Shared filename-update regex for both ``_pick_best_checkpoint`` and
# ``load_experiment_state``. Matches ``checkpoint_update_<N>...`` for
# the bare canonical name, the legacy ``checkpoint_update_<N>.pth``
# pattern, AND the emergency suffix pattern
# ``checkpoint_update_<N>_emergency_<SIG>_<JOBID>.pth``.
# noted that the stricter ``...(\d+)\.pth`` form in ``load_experiment_state``
# did not match emergency names, leaving ``start_update=0`` for
# emergency-only resumes where the payload lacked an ``update`` field.
_CHECKPOINT_FILENAME_UPDATE_RE = re.compile(r"checkpoint_update_?(\d+)")


def _checkpoint_filename_token(value: Any, fallback: str) -> str:
    """Constrain a value to a conservative filename token.

    Returns ``fallback`` (also sanitized) when ``value`` is ``None``, the
    empty string, or sanitizes to an empty token. The earlier ``value or
    fallback`` form conflated "caller passed nothing meaningful" with
    "caller passed a legitimately falsy value like ``0``"; this is the
    explicit form recommended.
    """
    safe_fallback = _CHECKPOINT_TOKEN_FALLBACK.sub("_", str(fallback)).strip("._-")
    if not safe_fallback:
        # Pathological caller passed an entirely-unsafe fallback. Use a
        # generic constant rather than allowing the empty string through —
        # an empty token would let the f-string compose ``..._.pth``.
        safe_fallback = "x"
    if value is None or value == "":
        return safe_fallback
    token = _CHECKPOINT_TOKEN_CHARS.sub("_", str(value)).strip("._-")
    return token or safe_fallback


def checkpoint_filename_update(name: str) -> int:
    """Return the update number embedded in a checkpoint filename, or 0.

    Shared between ``_pick_best_checkpoint`` and ``load_experiment_state``
: both call sites need to extract the update
    from filenames including the emergency suffix pattern
    ``checkpoint_update_<N>_emergency_<SIG>_<JOBID>.pth``. The previous
    stricter ``(\\d+)\\.pth`` regex in the resume path missed emergency
    names, falling back to ``0`` if the payload also lacked ``update``.
    """
    m = _CHECKPOINT_FILENAME_UPDATE_RE.search(name)
    return int(m.group(1)) if m else 0


def safe_point_checkpoint_path(
    results_dir: Path,
    completed_updates: Optional[int],
    signame: str,
    jobid: Optional[str] = None,
) -> Path:
    """Return a discoverable, non-canonical safe-point checkpoint path.

    Safe-point checkpoints must never overwrite ``checkpoint_update.pth``
    (the canonical periodic-save target). The
    ``checkpoint_update_<N>_emergency_<SIG>_<JOBID>.pth`` prefix keeps
    them discoverable by the resume globs (``checkpoint_update*.pth``)
    and by ``_pick_best_checkpoint`` (which prefers the highest payload
    ``update`` rather than newest mtime).

    Signal-to-token mapping:

    * ``USR1`` — SLURM walltime warning (sbatch ``--signal=USR1@120``);
      pairs with the requeue / exit-99 path so the job resumes from
      this safe point on restart.
    * ``TERM`` — ``scancel`` / operator-initiated cancel; pairs with the
      no-requeue / exit-143 path.
    * ``INT`` — ``KeyboardInterrupt`` / Ctrl+C; pairs with no-requeue /
      exit-130, written by the wrapper's exception handler rather than
      the loop's iteration boundary.
    """
    jobid_value = os.environ.get("SLURM_JOB_ID", "local") if jobid is None else jobid
    safe_jobid = _checkpoint_filename_token(jobid_value, "local")
    safe_signame = _checkpoint_filename_token(signame, "SIGNAL").upper()
    return results_dir / (
        f"checkpoint_update_{int(completed_updates or 0)}"
        f"_emergency_{safe_signame}_{safe_jobid}.pth"
    )


def _readable_candidate_update(path: Path) -> Optional[int]:
    """Return the payload ``update`` of a checkpoint candidate, or ``None``.

    Shared "readable candidate update" helper used by both
:func:`_pick_best_checkpoint` and:func:`highest_existing_checkpoint_update`
. Matching semantics keeps the resume picker
    and the refuse-stale guard from disagreeing about which file is
    "advanced" — a divergence that would let an unreadable or
    payload-stale-but-filename-high emergency file block a fresh
    safe-point save while resume silently ignored that same file.

    Contract:

    * ``torch.load`` succeeds and payload is a dict with an explicit
      integer ``update`` (or ``epoch`` if ``update`` is absent):
      returns the payload value AS-IS, including ``0`` ( -M2 — a legitimate early-training checkpoint's payload is
      trusted, not silently demoted to filename hint).
    * ``torch.load`` succeeds but payload has neither ``update`` nor
      ``epoch`` (or both are explicit ``None``): returns the
      filename-encoded update (or ``0``).
    * ``torch.load`` fails: returns ``None`` (unreadable — caller should
      treat as ineligible, NOT as filename-update). This mirrors
      ``_pick_best_checkpoint``'s behavior.

    Higher-level filtering (e.g., "drop ``update == 0`` candidates when
    any candidate has ``update > 0``") lives in ``_pick_best_checkpoint``;
    this helper just reports what each file's payload claims.
    """
    try:
        payload = torch.load(path, map_location='cpu', weights_only=False)
    except Exception:
        return None
    # Explicit None check rather than ``or``
    # so a legitimate ``{"update": 0}`` payload is not silently demoted
    # to filename fallback. Only fall through to filename when both
    # ``update`` and ``epoch`` are absent or explicit ``None``.
    raw_payload_update: Optional[int] = None
    if isinstance(payload, dict):
        raw = payload.get('update')
        if raw is None:
            raw = payload.get('epoch')
        if raw is not None:
            try:
                raw_payload_update = int(raw)
            except (TypeError, ValueError):
                raw_payload_update = None
    if raw_payload_update is not None:
        # Trust the payload value AS-IS, including ``0``. The picker's
        # higher-level logic ("ignore update=0 when any candidate has
        # update > 0") still drops zero-update partials from selection;
        # this helper just reports what the file *says*.
        return raw_payload_update
    # No meaningful payload update — fall back to filename hint.
    return checkpoint_filename_update(path.name)


def highest_existing_checkpoint_filename_hint(
    results_dir: Path,
    proposed_update: Optional[int] = None,
) -> int:
    """Refuse-stale guard with filename-hint fast path + verify-on-block.

    The guard runs inside the SIGUSR1 grace window, where a full
    ``torch.load(weights_only=False)`` of every emergency file — each carrying
    multi-MB optimizer state — just to read one int can exhaust the budget
    before the emergency write starts. Trusting filename hints unconditionally
    is wrong too: a corrupt high-numbered file would block a fresh save that
    the resume picker would have ignored. Hence verify-on-block:

    * Emergency file whose filename hint <= ``proposed_update``: trusted via
      filename — it would not cause refuse-stale anyway.
    * Emergency file whose filename hint > ``proposed_update``: verified via
      ``_readable_candidate_update``. Unreadable → skip, matching
      ``_pick_best_checkpoint``.
    * Canonical ``checkpoint_update.pth``: filename hint is always 0, so its
      payload is always read (one ``torch.load`` per directory).

    ``proposed_update=None`` means "fully verify": every emergency candidate's
    payload is read. Production callers always pass a concrete value.

    Once a verified candidate exceeds ``proposed_update`` the scan returns
    early — any higher value already settles the caller's decision.

    Returns ``-1`` when no candidates exist.
    """
    if not results_dir.exists():
        return -1
    best = -1
    canonical_path: Optional[Path] = None
    for p in results_dir.glob("checkpoint_update*.pth"):
        if p.name == "checkpoint_update.pth":
            # Canonical's filename hint is 0; defer to payload.
            canonical_path = p
            continue
        hint = checkpoint_filename_update(p.name)
        # ``proposed_update is None`` means full-verify;
        # otherwise verify only candidates that would actually block
        # the caller's write. Trust filename hints for non-blocking
        # emergencies (fast path).
        must_verify = proposed_update is None or hint > int(proposed_update)
        if must_verify:
            payload_or_filename = _readable_candidate_update(p)
            if payload_or_filename is None:
                # Unreadable — ineligible (matches ``_pick_best_checkpoint``).
                continue
            if payload_or_filename > best:
                best = payload_or_filename
            # Break early once we have a verified
            # candidate strictly above ``proposed_update`` — the
            # caller's refuse-stale decision is already sealed.
            if proposed_update is not None and best > int(proposed_update):
                return best
        else:
            if hint > best:
                best = hint
    if canonical_path is not None:
        canonical_update = _readable_candidate_update(canonical_path)
        if canonical_update is not None and canonical_update > best:
            best = canonical_update
    return best


def highest_existing_checkpoint_update(results_dir: Path) -> int:
    """Return the highest readable on-disk checkpoint update, or ``-1``.

    Defense-in-depth for the rollback class: before writing a
    safe-point or interrupt checkpoint, the caller can check whether
    any existing readable checkpoint in the directory already advances
    past the proposed update. If yes, the new write would be a
    regression and should be skipped — logged loudly so future
    post-mortems have a paper trail.

:

    * **N1.2 / HIGH 2**: shares ``_readable_candidate_update`` with
      ``_pick_best_checkpoint`` so a corrupt or payload-stale emergency
      file with a high filename number cannot block a fresh save even
      though resume would have ignored it.
    * **N1.2**: dropped the unused ``current_update`` parameter. The
      old signature implied "at least X" semantics; the body just
      returned the dir's max. Callers now compare the return value
      against their proposed update at the call site.
    The filename-first short-circuit was intentionally dropped (
): a high-numbered filename whose payload reports a
    lower update is a legitimate scenario the picker handles by
    preferring payload, and short-circuiting on filename would skip
    reading the canonical lower-numbered file that legitimately wins.
    Correctness over a small I/O optimization.
    """
    if not results_dir.exists():
        return -1
    best = -1
    for p in results_dir.glob("checkpoint_update*.pth"):
        readable = _readable_candidate_update(p)
        if readable is None:
            # Unreadable — ineligible (matches ``_pick_best_checkpoint``).
            continue
        if readable > best:
            best = readable
    return best


# keep the old name as a deprecated alias so any
# external import keeps working in this revision. Drops the ``current_update``
# param via *args/**kwargs to remain forward-compatible. New code should
# call:func:`highest_existing_checkpoint_update` directly.
def existing_checkpoint_update_at_least(results_dir: Path, *_args: Any, **_kwargs: Any) -> int:
    """Deprecated alias for:func:`highest_existing_checkpoint_update`."""
    return highest_existing_checkpoint_update(results_dir)


def _safe_point_checkpoint_path(
    results_dir: Path,
    completed_updates: Optional[int],
    is_warning_signal: bool,
) -> Path:
    """Return the non-destructive SIGTERM/SIGUSR1 safe-point checkpoint path.

    Thin wrapper around:func:`safe_point_checkpoint_path` that maps the
    boolean ``is_warning_signal`` flag (the form the loop carries) to the
    signal label embedded in the filename. Kept as a separate symbol so
    the loop call site stays terse and so callers that already track the
    boolean form don't have to translate inline.

    signature aligned with the public sibling
    (``Optional[int]``); both accept ``None`` via the ``int(... or 0)``
    cast inside:func:`safe_point_checkpoint_path`.
    """
    signame = "USR1" if is_warning_signal else "TERM"
    return safe_point_checkpoint_path(results_dir, completed_updates, signame)


def resolve_sampler_metadata(
    config: "ExperimentConfig",
    device: torch.device,
) -> Dict[str, Any]:
    """Return requested and effective sampler controls for run artifacts."""
    sampler_controls = _resolve_sampling_mode_controls(
        config.static_shape_sampling,
        config.cuda_graph_sampling,
        getattr(
            config,
            "_requested_sampling_mode_raw",
            getattr(config, "sampling_mode", None),
        ),
        device,
        warn_inconsistent=True,
    )

    # Metadata honesty for the bucketed graph opt-in: graph capture ALSO requires
    # the CT tableau backend (the GFlowNet enforces this and otherwise falls back
    # to the eager bucketed sampler, forcing its own cuda_graph_sampling=False).
    # _resolve_sampling_mode_controls cannot see measurement_backend, so its
    # bucketed 'effective' value is request+mode/device only. Mirror the backend
    # gate here, using the SAME resolver the GFlowNet uses (handles
    # measurement_backend=None auto-selection), so recorded run metadata is not
    # mislabeled as graph-captured on clifford_map. requested_cuda_graph_sampling
    # is preserved (it reflects the user's request).
    if (
        sampler_controls["effective_sampling_mode"] == _SAMPLING_MODE_BUCKETED
        and sampler_controls.get("cuda_graph_sampling")
    ):
        # Mirror resolve_tableau_backend's NAME resolution WITHOUT importing the
        # backend class (avoids a cupy import at config-resolution time / on
        # CPU hosts). Per backends.py, None/'auto' resolves to the CT adapter on
        # CUDA (and this branch already implies device.type=='cuda' via the
        # graph-eligible gate); any explicit non-CT backend (e.g. 'clifford_map')
        # cannot run the graph path, so demote the effective value to match the
        # GFlowNet (which falls back to the eager bucketed sampler).
        mb = getattr(config, "measurement_backend", None)
        backend_name = "tableau_batch_adapter" if mb in (None, "auto") else str(mb)
        # mps_native resolves to TableauBatchAdapter (has reset_inplace_with_mask,
        # the capability the graph path requires), so GFlowNet enables graph capture
        # for it — treat it as graph-capable here.
        CT_CAPABLE_BACKENDS = {"tableau_batch_adapter", "mps_native"}
        # The GFNs call-time gate (_effective_bucketed_graph) also hard-requires
        # the fused apply + fused mask/counts kernels. Their config knobs are
        # static, so mirror them here for the same metadata honesty; runtime
        # latch-offs (NVRTC/launch failures mid-run) are call-time-only and are
        # deliberately not modeled in resolved metadata.
        fused_ok = bool(getattr(config, "use_fused_apply_kernel", True)) and bool(
            getattr(config, "use_fused_mask_counts_kernel", True)
        )
        # Graph capture also requires the DEFAULT flattened-W feature mode.
        # The trainer derives the feature mode from model_type (GFNs.py
        # EfficientGFNTrainer): packed_w_rowtoken / packed_w_split set
        # packed_w_input=True and hit_mlp / hit_deepsets build a
        # feature_extractor — _effective_bucketed_graph() refuses both and the
        # GFlowNet falls back to the eager bucketed sampler. model_type is
        # config-static, so mirror it here too.
        NON_FLAT_W_MODEL_TYPES = {
            "packed_w_rowtoken",
            "packed_w_split",
            "hit_mlp",
            "hit_deepsets",
        }
        flat_w_ok = (
            str(getattr(config, "model_type", "clifford_mlp"))
            not in NON_FLAT_W_MODEL_TYPES
        )
        if backend_name not in CT_CAPABLE_BACKENDS or not fused_ok or not flat_w_ok:
            sampler_controls = {**sampler_controls, "cuda_graph_sampling": False}

    return {
        **sampler_controls,
        "use_cuda_graph_policy": bool(
            getattr(config, "use_cuda_graph_policy", False)
        ),
        "cuda_graph_policy_eligible": (
            bool(getattr(config, "use_cuda_graph_policy", False))
            and device.type == "cuda"
            and sampler_controls["effective_sampling_mode"]
            == _SAMPLING_MODE_DYNAMIC_ACTIVE
        ),
        "cuda_graph_policy_max_rows": int(
            getattr(config, "cuda_graph_policy_max_rows", 2048)
        ),
        "use_fused_metadata_kernel": bool(config.use_fused_metadata_kernel),
        "use_fused_sampling_kernel": bool(
            getattr(config, "use_fused_sampling_kernel", True)
        ),
        "use_fused_mask_counts_kernel": bool(
            getattr(config, "use_fused_mask_counts_kernel", True)
        ),
        "use_fused_counter_rng_kernel": bool(
            getattr(config, "use_fused_counter_rng_kernel", True)
        ),
        "use_fused_partition_update_kernel": bool(
            getattr(config, "use_fused_partition_update_kernel", True)
        ),
        "use_fused_apply_kernel": bool(
            getattr(config, "use_fused_apply_kernel", True)
        ),
        "use_bf16_sampling": bool(
            getattr(config, "use_bf16_sampling", True)
        ),
        # NOTE: ``use_bf16_backward`` is deliberately NOT here — like
        # use_activation_checkpointing it is a BACKWARD-only knob, not a sampler
        # control, so it lives in the ``computational`` dict.
        # NOTE: ``FLOWMEAS_FLOW_DEDUP`` is intentionally absent — an env-var
        # off-switch rather than a config field, read lazily per call-site, so
        # snapshotting it here would record only the init-time value. The GFN
        # __init__ log line records its construction-time state.
    }


def get_evaluator_mode_metadata(config: Any) -> Dict[str, Any]:
    """Return visible evaluator-mode metadata for logs and reports.

    The returned dict is the union of:
      - the historical ``mode`` / ``allows_full_state_evaluation`` fields
        that downstream code already keys on;
      - the validation-tier fields (``validation_tier``,
        ``provides_scalar_energy``, ``sufficient_for_final_readiness_claim``,
        ``tier_description``, ``dmrg_reference_available``,
        ``dmrg_reference_energy``) that record the quality of reference the
        run produces.  The ``dmrg_reference_energy`` value in the returned
        dict is the *resolved* scalar (post-coercion ``Optional[float]``)
        and is the source of truth for tier promotion;
        ``dmrg_reference_available`` is derived from it.

    Tier promotion to ``dmrg_reference`` is driven by the presence of an
    actual scalar reference resolved from either ``config.dmrg_reference_energy``
    (explicit scalar override) or ``config.dmrg_reference_path`` (hash-verified
    sidecar produced by ``code.dmrg_reference``).  ``resolve_dmrg_reference_energy``
    enforces the precedence (explicit float > hash-verified sidecar > None)
    and runs ``coerce_dmrg_reference_energy`` so a bare capability flag, a
    JSON ``false``, a ``NaN``, or a stale-hash sidecar are all rejected
    (finding).  Until one of those sources yields a finite
    scalar, the tier stays ``structural``.
    """
    large_hubbard_mode = _coerce_bool_config(
        getattr(config, "large_hubbard_mode", False),
        "large_hubbard_mode",
    )
    mode = resolve_evaluator_mode(large_hubbard_mode)
    allows_full_state_evaluation = mode == EVALUATOR_MODE_EXACT_SMALL

    if allows_full_state_evaluation:
        description = (
            "Exact-small validation mode; exact ground-state metadata and "
            "state-vector evaluation paths may be used."
        )
    else:
        description = (
            "Scalable-large Hubbard mode; exact full-state metadata and "
            "state-vector evaluation paths are not allowed."
        )

    # Tier promotion is gated on an actual DMRG scalar
    # reference, not on a bare capability flag.  ``coerce_dmrg_reference_energy``
    # normalises ``None`` / strings like ``"false"`` / NaN to ``None`` so
    # the tier cannot silently move to ``dmrg_reference`` without
    # measurable scalar evidence.: the scalar may come from an
    # explicit ``dmrg_reference_energy`` *or* from a sidecar pointed at by
    # ``dmrg_reference_path``; ``resolve_dmrg_reference_energy`` enforces
    # both the coercion and the hash-verified sidecar load.
    dmrg_reference_energy = dmrg_reference.resolve_dmrg_reference_energy(config)
    tier_metadata = validation_tier.get_validation_tier_metadata(
        evaluator_mode=mode,
        dmrg_reference_energy=dmrg_reference_energy,
    )

    return {
        "large_hubbard_mode": large_hubbard_mode,
        "mode": mode,
        "allows_full_state_evaluation": allows_full_state_evaluation,
        "description": description,
        **tier_metadata,
    }


def ensure_exact_small_evaluation_allowed(config: Any, context: str) -> None:
    """Fail loudly before exact-only evaluation runs in scalable-large mode."""
    evaluator_mode_metadata = get_evaluator_mode_metadata(config)
    if evaluator_mode_metadata["allows_full_state_evaluation"]:
        return

    raise RuntimeError(
        f"{context} requires the exact-small EnergyEstimator path, but "
        f"large_hubbard_mode=True configured evaluator mode "
        f"{evaluator_mode_metadata['mode']!r}. Route this call through the "
        "scalable-large structural reporter or run with large_hubbard_mode=False."
    )


# Defense-in-depth size guard. ``large_hubbard_mode`` is a *config* flag that
# defaults to False — a user who forgets to set it on a 52-qubit Hubbard
# config can still enter ``exact_small`` and trigger ``ground_state_energy``
# materialisation, which is the very thing the no-full-state rule forbids.
# ``EXACT_FULL_STATE_QUBIT_LIMIT`` is the boundary above which we refuse full-state
# evaluation regardless of the config flag. 26 qubits sits a safe distance
# above the largest exact target (n=20 for the H2O/HCl/C2 sweep) and well
# below any Hubbard benchmark (n=52); pick a number small enough that it fits in
# memory but large enough not to interfere with the canonical sweep.


def assert_full_state_eval_safe(
    hamiltonian_helper: Any, config: Any, context: str
) -> None:
    """Refuse exact full-state evaluation for systems beyond the safe limit.

    Even when ``large_hubbard_mode=False`` (the default), a system with
    ``n_qubits >= EXACT_FULL_STATE_QUBIT_LIMIT`` must not touch
    ``ground_state_energy`` / ``_apply_circuits_to_states`` / any other
    full-state path — those allocate ``O(2^n)`` tensors and are the
    exact regression the no-full-state rule calls out.

    Failing here is loud-by-design: the operator needs to either
    explicitly set ``large_hubbard_mode=True`` (routes to the scalable
    reporter) or pick a smaller Hamiltonian. Silent fallback would hide
    the misconfiguration until OOM kills the job.
    """
    n_qubits = int(getattr(hamiltonian_helper, "n_qubits", 0) or 0)
    qubit_limit = full_state_guard.EXACT_FULL_STATE_QUBIT_LIMIT
    if n_qubits < qubit_limit:
        return
    evaluator_mode_metadata = get_evaluator_mode_metadata(config)
    if not evaluator_mode_metadata["allows_full_state_evaluation"]:
        # Already on the scalable path; nothing to refuse.
        return
    raise full_state_guard.ExactFullStateGuardError(
        f"{context}: n_qubits={n_qubits} >= {qubit_limit} "
        f"would trigger O(2^n) full-state evaluation but "
        f"large_hubbard_mode is False. Set large_hubbard_mode=True in the "
        f"config to route this through the scalable-large structural "
        f"reporter, or use a Hamiltonian with n_qubits < "
        f"{qubit_limit} if exact evaluation is required."
    )


def create_large_mode_hamiltonian_summary(
    hamiltonian_helper: PauliHamiltonianHelper,
    evaluator_mode_metadata: Dict[str, Any],
) -> Dict[str, Any]:
    """Create Hamiltonian metadata without triggering exact ground-state solves."""
    exact_energies = hamiltonian_helper.get_exact_energy_from_file()
    exact_total_energy = None
    if exact_energies:
        exact_total_energy = exact_energies.get(
            "total_energy",
            exact_energies.get("electronic_energy"),
        )

    hf_bitstrings = hamiltonian_helper.get_hartree_fock_bitstring()
    hf_bitstring = None
    if hf_bitstrings:
        hf_bitstring = hf_bitstrings.get(hamiltonian_helper.filepath.stem)

    return {
        "molecule": hamiltonian_helper.filepath.parent.name,
        "transformation": hamiltonian_helper.filepath.stem,
        "n_qubits": hamiltonian_helper.n_qubits,
        "n_terms": len(hamiltonian_helper.pauli_str_list),
        "ground_state_energy": None,
        "ground_state_energy_status": "not_computed_scalable_large_mode",
        "exact_energies": exact_energies,
        "exact_total_energy": exact_total_energy,
        "energy_difference": None,
        "hf_bitstring": hf_bitstring,
        "largest_coefficient": max(abs(w) for w in hamiltonian_helper.w_list),
        "smallest_coefficient": min(
            abs(w) for w in hamiltonian_helper.w_list if abs(w) > 1e-10
        ),
        "evaluator_mode": evaluator_mode_metadata["mode"],
        "large_hubbard_mode": evaluator_mode_metadata["large_hubbard_mode"],
        "full_state_evaluation_allowed": evaluator_mode_metadata[
            "allows_full_state_evaluation"
        ],
    }


def _config_to_dict(config: Any) -> Dict[str, Any]:
    """Serialize ExperimentConfig or legacy config-like objects."""
    if is_dataclass(config):
        return asdict(config)
    if isinstance(config, dict):
        return dict(config)
    if hasattr(config, "__dict__"):
        return dict(vars(config))
    raise TypeError(
        "config must be an ExperimentConfig, dict, or object with __dict__; "
        f"got {type(config).__name__}"
    )


def convert_metrics_to_cpu(metrics: Dict[str, Any]) -> Dict[str, float]:
    """
    Convert metric dictionary with potential GPU tensors to CPU floats/numpy arrays.
    Optimized to minimize GPU synchronization points.

    Args:
        metrics: Dictionary potentially containing torch.Tensor values

    Returns:
        Dictionary with all tensors converted to float/numpy for JSON serialization
    """
    converted = {}
    for k, v in metrics.items():
        if torch.is_tensor(v):
            # Single GPU sync point per tensor
            converted[k] = v.item() if v.numel() == 1 else v.cpu().numpy()
        else:
            converted[k] = v
    return converted


def convert_metrics_history_to_cpu(metrics_history: Dict[str, List[Any]]) -> Dict[str, List[float]]:
    """
    Convert metrics history (which may contain lists of GPU tensors) to CPU for checkpoint saving.

    Args:
        metrics_history: Dictionary of metric names to lists of values (may be tensors)

    Returns:
        Dictionary with all tensors in lists converted to CPU floats/numpy
    """
    converted_history = {}
    for metric_name, value_list in metrics_history.items():
        converted_list = []
        for v in value_list:
            if torch.is_tensor(v):
                converted_list.append(v.item() if v.numel() == 1 else v.cpu().numpy())
            else:
                converted_list.append(v)
        converted_history[metric_name] = converted_list
    return converted_history


class DiskBackedResultStore:
    """Append-only store for evaluation results backed by a JSONL file."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()

    def is_empty(self) -> bool:
        return not self.path.exists() or self.path.stat().st_size == 0

    def append(self, results: List[ExtendedBatchElementEnergyResult]) -> None:
        if not results:
            return

        with self.lock:
            with self.path.open('a', encoding='utf-8') as f:
                for result in results:
                    record = asdict(result)
                    f.write(json.dumps(record, cls=NumpyEncoder))
                    f.write('\n')

    def load_all(self) -> List[ExtendedBatchElementEnergyResult]:
        if not self.path.exists():
            return []

        # Skip-and-warn on per-row failures so
        # one bad JSONL line doesn't poison resume of the whole
        # experiment. Mirror of the three sibling loaders in
        #:func:`load_experiment_state`.
        results: List[ExtendedBatchElementEnergyResult] = []
        with self.lock:
            with self.path.open('r', encoding='utf-8') as f:
                for line_number, line in enumerate(f, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                        results.append(self._to_result(record))
                    except Exception as exc:
                        logging.warning(
                            "Could not load result from %s:%d: %s",
                            self.path, line_number, exc,
                        )
        return results

    def export_json(self, export_path: Path) -> None:
        export_path = Path(export_path)
        export_path.parent.mkdir(parents=True, exist_ok=True)

        if self.is_empty():
            export_path.write_text('[]\n', encoding='utf-8')
            return

        with self.lock:
            with self.path.open('r', encoding='utf-8') as src, \
                    export_path.open('w', encoding='utf-8') as dst:
                dst.write('[\n')
                first = True
                for line in src:
                    line = line.strip()
                    if not line:
                        continue
                    if not first:
                        dst.write(',\n')
                    parsed = json.loads(line)
                    dst.write(json.dumps(parsed, indent=2, cls=NumpyEncoder))
                    first = False
                dst.write('\n]\n')

    def _to_result(self, record: Dict) -> ExtendedBatchElementEnergyResult:
        # Routed through the shared reconstruction helper so
        # this loader and the three in:func:`load_experiment_state`
        # apply the same migrations and field-threading rules.
        return _extended_result_from_record(record)


class DiskBackedMetricStore:
    """Append-only metric logger backed by JSONL, with bounded incremental reading.

    Key features:
    - Incremental reading: only reads newly appended lines (tracks file offset)
    - Bounded memory: uses deques with maxlen to cap RAM usage
    - Truncation detection: resets cache if file is rotated/replaced
    """

    def __init__(self, path: Path, max_buffer_size: int = 1000, cache_window: int = 1024):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        # Buffer for accumulating metrics before writing to disk
        self.buffer: List[Tuple[int, Dict[str, float], Optional[Dict[str, float]]]] = []
        self.max_buffer_size = max_buffer_size  # Prevent unbounded memory growth
        
        # Incremental reading cache (bounded by cache_window)
        self.cache_window = cache_window
        self._cached_updates: deque = deque(maxlen=cache_window)
        self._cached_metrics: Dict[str, deque] = {}  # key -> deque(maxlen=cache_window)
        self._cached_timing: Dict[str, deque] = {}   # key -> deque(maxlen=cache_window)
        self._read_offset: int = 0  # byte offset of last read position
        self._file_inode: Optional[int] = None  # to detect file replacement
        self._file_size: int = 0  # to detect truncation

    def _get_or_create_deque(self, cache_dict: Dict[str, deque], key: str) -> deque:
        """Get or create a bounded deque for a metric key."""
        if key not in cache_dict:
            cache_dict[key] = deque(maxlen=self.cache_window)
        return cache_dict[key]

    def _reset_cache(self) -> None:
        """Reset all cached data (called on file truncation/rotation)."""
        self._cached_updates.clear()
        self._cached_metrics.clear()
        self._cached_timing.clear()
        self._read_offset = 0
        self._file_inode = None
        self._file_size = 0

    def _check_file_identity(self) -> bool:
        """Check if file was replaced or truncated. Returns True if cache is still valid."""
        if not self.path.exists():
            if self._file_inode is not None:
                # File was deleted
                self._reset_cache()
            return False
        
        stat = self.path.stat()
        current_inode = stat.st_ino
        current_size = stat.st_size
        
        # Check for file replacement (different inode)
        if self._file_inode is not None and current_inode != self._file_inode:
            self._reset_cache()
            return True
        
        # Check for truncation (size decreased)
        if current_size < self._read_offset:
            self._reset_cache()
            return True
        
        # Update tracked identity
        self._file_inode = current_inode
        self._file_size = current_size
        return True

    def is_empty(self) -> bool:
        return not self.path.exists() or self.path.stat().st_size == 0

    def append(self, update: int, metrics: Dict[str, float], timing: Optional[Dict[str, float]] = None) -> None:
        if metrics is None:
            return

        record: Dict[str, Dict] = {
            'update': update,
            'metrics': metrics,
        }
        if timing:
            record['timing'] = timing

        line = json.dumps(record, cls=NumpyEncoder)
        with self.lock:
            with self.path.open('a', encoding='utf-8') as f:
                f.write(line)
                f.write('\n')

    def append_to_buffer(self, update: int, metrics: Dict[str, float], timing: Optional[Dict[str, float]] = None) -> None:
        """Add metrics to buffer without writing to disk (avoids GPU-CPU transfer).

        Important: Detaches tensors to prevent memory leaks from gradient tracking.
        Auto-flushes if buffer exceeds max_buffer_size to prevent unbounded growth.
        """
        if metrics is None:
            return

        # Detach tensors to prevent memory leaks while keeping them on GPU
        metrics_detached = {}
        for k, v in metrics.items():
            if torch.is_tensor(v):
                metrics_detached[k] = v.detach()
            else:
                metrics_detached[k] = v

        timing_detached = None
        if timing:
            timing_detached = {}
            for k, v in timing.items():
                if torch.is_tensor(v):
                    timing_detached[k] = v.detach()
                else:
                    timing_detached[k] = v

        # Store detached metrics in buffer (still on GPU, but no gradient tracking)
        self.buffer.append((update, metrics_detached, timing_detached))

        # Auto-flush if buffer is too large (safety against memory leaks)
        if len(self.buffer) >= self.max_buffer_size:
            logging.warning(f"Metric buffer exceeded {self.max_buffer_size} entries, auto-flushing to prevent memory leak")
            self.flush_buffer()

    def flush_buffer(self) -> None:
        """Flush buffered metrics to disk (performs GPU-CPU transfer and disk write).

        Explicitly deletes GPU tensors after transfer to prevent memory leaks.
        Also updates internal cache to avoid needing disk re-reads for plotting.
        """
        if not self.buffer:
            return

        num_flushed = len(self.buffer)

        with self.lock:
            with self.path.open('a', encoding='utf-8') as f:
                for update, metrics, timing in self.buffer:
                    # Convert GPU tensors to CPU now
                    metrics_cpu = convert_metrics_to_cpu(metrics)
                    timing_cpu = convert_metrics_to_cpu(timing) if timing else None

                    record: Dict[str, Dict] = {
                        'update': update,
                        'metrics': metrics_cpu,
                    }
                    if timing_cpu:
                        record['timing'] = timing_cpu

                    line = json.dumps(record, cls=NumpyEncoder)
                    f.write(line)
                    f.write('\n')
                    
                    # Update cache directly (avoids disk re-read for plotting)
                    self._cached_updates.append(update)
                    for key, value in metrics_cpu.items():
                        self._get_or_create_deque(self._cached_metrics, key).append(value)
                    if timing_cpu:
                        for key, value in timing_cpu.items():
                            self._get_or_create_deque(self._cached_timing, key).append(value)

                    # Explicitly delete GPU tensors to free memory
                    del metrics
                    del timing
                
                # Update read offset to end of file (we've cached these entries)
                # Must be inside the with block while f is still open
                self._read_offset = f.tell()
            
            # Update file identity after writing
            stat = self.path.stat()
            self._file_inode = stat.st_ino
            self._file_size = stat.st_size

        # Clear buffer after writing and force garbage collection
        self.buffer.clear()

        # Optional: Force GPU cache cleanup if using CUDA
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return num_flushed

    def replace(self, metrics_history: Dict[str, List], timing_history: Optional[Dict[str, List]] = None) -> None:
        """Replace entire file contents (used for migration from checkpoint)."""
        max_len = max((len(values) for values in metrics_history.values()), default=0)
        timing_history = timing_history or {}

        with self.lock:
            # Reset cache since we're replacing the file
            self._reset_cache()
            
            with self.path.open('w', encoding='utf-8') as f:
                for idx in range(max_len):
                    metrics_record = {
                        key: values[idx]
                        for key, values in metrics_history.items()
                        if idx < len(values)
                    }
                    timing_record = {
                        key: values[idx]
                        for key, values in timing_history.items()
                        if idx < len(values)
                    }

                    record: Dict[str, Dict] = {
                        'update': idx + 1,
                        'metrics': metrics_record,
                    }
                    if timing_record:
                        record['timing'] = timing_record

                    f.write(json.dumps(record, cls=NumpyEncoder))
                    f.write('\n')

    def load_series(self, window: Optional[int] = None) -> Tuple[List[int], Dict[str, List[float]], Dict[str, List[float]]]:
        """Load metric series with bounded memory and incremental disk reads.

        Args:
            window: If provided, return only the last `window` entries.
                    Uses internal cache_window by default for bounded memory.

        Returns:
            Tuple of (updates, metrics_series, timing_series) as lists.

        Note:
            This method uses incremental reading - it only reads newly appended
            lines from disk (after _read_offset) and caches them in bounded deques.
            Subsequent calls with unchanged file return cached data with no I/O.
        """
        effective_window = window if window is not None else self.cache_window
        
        with self.lock:
            # Check if file was replaced/truncated
            if not self._check_file_identity():
                # File doesn't exist
                return [], defaultdict(list), defaultdict(list)
            
            # Read only newly appended data
            stat = self.path.stat()
            if stat.st_size > self._read_offset:
                with self.path.open('r', encoding='utf-8') as f:
                    f.seek(self._read_offset)
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            record = json.loads(line)
                            self._cached_updates.append(record.get('update', len(self._cached_updates) + 1))
                            for key, value in record.get('metrics', {}).items():
                                self._get_or_create_deque(self._cached_metrics, key).append(value)
                            for key, value in record.get('timing', {}).items():
                                self._get_or_create_deque(self._cached_timing, key).append(value)
                        except json.JSONDecodeError:
                            # Partial line at end of file (concurrent write)
                            break
                    self._read_offset = f.tell()
            
            # Convert deques to lists, applying window limit
            updates = list(self._cached_updates)[-effective_window:] if effective_window else list(self._cached_updates)
            
            metrics_series: Dict[str, List[float]] = {}
            for key, dq in self._cached_metrics.items():
                metrics_series[key] = list(dq)[-effective_window:] if effective_window else list(dq)
            
            timing_series: Dict[str, List[float]] = {}
            for key, dq in self._cached_timing.items():
                timing_series[key] = list(dq)[-effective_window:] if effective_window else list(dq)
        
        return updates, defaultdict(list, metrics_series), defaultdict(list, timing_series)

def _scalable_large_hamiltonian_metadata(
    *,
    config: Optional[ExperimentConfig] = None,
    trainer: Optional[Any] = None,
    hamiltonian_helper: Optional[Any] = None,
    checkpoint: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build Hamiltonian metadata for structural reports without exact solves."""
    pauli_strings = None
    weights = None
    n_qubits = None
    filepath = None

    if hamiltonian_helper is not None:
        pauli_strings = getattr(hamiltonian_helper, "pauli_str_list", None)
        weights = getattr(hamiltonian_helper, "w_list", None)
        n_qubits = getattr(hamiltonian_helper, "n_qubits", None)
        filepath = getattr(hamiltonian_helper, "filepath", None)
    elif trainer is not None:
        pauli_strings = getattr(trainer, "pauli_str_list", None)
        weights = getattr(trainer, "w_list", None)
        n_qubits = getattr(trainer, "n_qubits", None)

    if checkpoint is not None:
        n_qubits = n_qubits if n_qubits is not None else checkpoint.get("n_qubits")

    if config is not None:
        filepath = filepath if filepath is not None else getattr(
            config,
            "hamiltonian_path",
            None,
        )

    return {
        "filepath": str(filepath) if filepath is not None else None,
        "n_qubits": n_qubits,
        "n_terms": len(pauli_strings) if pauli_strings is not None else None,
        "coefficient_abs": _coefficient_abs_summary(weights),
        "ground_state_energy": None,
        "ground_state_energy_status": "not_computed_scalable_large_mode",
        "full_state_evaluation_allowed": False,
    }


def _summarize_structural_batch_element(
    actions: Any,
    lengths: Any,
    *,
    batch_element_rank: int,
    cost: Any = None,
    terminal_index: Optional[int] = None,
) -> Dict[str, Any]:
    """Summarize a sampled circuit batch without simulating amplitudes."""
    actions_tensor = (
        actions.detach().cpu()
        if torch.is_tensor(actions)
        else torch.as_tensor(actions)
    )
    lengths_tensor = (
        lengths.detach().cpu()
        if torch.is_tensor(lengths)
        else torch.as_tensor(lengths)
    )

    if actions_tensor.dim() == 1:
        actions_tensor = actions_tensor.unsqueeze(0)
    if lengths_tensor.dim() == 0:
        lengths_tensor = lengths_tensor.unsqueeze(0)

    length_values = lengths_tensor.reshape(-1).to(torch.long)
    length_float = length_values.to(torch.float32)
    n_circuits = int(length_values.numel())
    total_actions = int(length_values.sum().item()) if n_circuits else 0

    terminal_action_count = None
    if terminal_index is not None and actions_tensor.numel() > 0:
        terminal_action_count = int(
            (actions_tensor == int(terminal_index)).sum().item()
        )

    stored_action_width = (
        int(actions_tensor.shape[-1])
        if actions_tensor.dim() > 1 and actions_tensor.numel() > 0
        else 0
    )

    return {
        "batch_element_rank": batch_element_rank,
        "n_circuits": n_circuits,
        "total_recorded_actions": total_actions,
        "mean_circuit_length": (
            float(length_float.mean().item()) if n_circuits else 0.0
        ),
        "min_circuit_length": int(length_values.min().item()) if n_circuits else 0,
        "max_circuit_length": int(length_values.max().item()) if n_circuits else 0,
        "stored_action_width": stored_action_width,
        "terminal_index": terminal_index,
        "terminal_action_count": terminal_action_count,
        "batch_cost": _to_report_float(cost),
    }


def create_scalable_large_evaluation_report(
    *,
    batch_actions_list: List[Any],
    batch_lengths_list: List[Any],
    batch_costs: Optional[List[Any]],
    update: int,
    config: ExperimentConfig,
    source: str,
    hamiltonian_metadata: Dict[str, Any],
    terminal_index: Optional[int] = None,
    checkpoint_path: Optional[Path] = None,
    measurement_backend: Optional[str] = None,
) -> Dict[str, Any]:
    """Create a large-mode structural report with no state-vector evaluation."""
    evaluator_mode_metadata = get_evaluator_mode_metadata(config)
    if evaluator_mode_metadata["allows_full_state_evaluation"]:
        raise RuntimeError(
            "Scalable-large structural reporting requires scalable_large mode. "
            "Use the exact-small EnergyEstimator path for exact_small mode."
        )

    num_to_eval = min(
        int(getattr(config, "n_eval_top_k_batch_elements", len(batch_actions_list))),
        len(batch_actions_list),
    )

    batch_summaries = []
    batch_costs = batch_costs or []
    for batch_idx in range(num_to_eval):
        cost = batch_costs[batch_idx] if batch_idx < len(batch_costs) else None
        batch_summaries.append(
            _summarize_structural_batch_element(
                batch_actions_list[batch_idx],
                batch_lengths_list[batch_idx],
                batch_element_rank=batch_idx,
                cost=cost,
                terminal_index=terminal_index,
            )
        )

    total_circuits = sum(b["n_circuits"] for b in batch_summaries)
    total_recorded_actions = sum(b["total_recorded_actions"] for b in batch_summaries)
    mean_circuit_length = (
        total_recorded_actions / total_circuits if total_circuits else 0.0
    )
    max_circuit_length = max(
        (b["max_circuit_length"] for b in batch_summaries),
        default=0,
    )

    # ``energy_estimate`` is
    # reserved for the policy/evaluator estimate (or None if not computed
    # in this mode).  The DMRG reference scalar lives ONLY under
    # ``dmrg_reference_energy`` so the artifact cannot be misread as the
    # trained policy's energy.  ``energy_status`` distinguishes the
    # two states: structural-only vs. structural-with-DMRG-reference.
    dmrg_reference_energy = evaluator_mode_metadata.get("dmrg_reference_energy")
    has_dmrg_scalar = dmrg_reference_energy is not None
    energy_status_field = (
        "structural_report_with_dmrg_reference"
        if has_dmrg_scalar
        else "not_computed_structural_report_only"
    )

    report = {
        "schema_version": 1,
        "mode": EVALUATOR_MODE_SCALABLE_LARGE,
        "update": update,
        "source": source,
        "timestamp": datetime.now().isoformat(),
        "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
        "large_hubbard_mode": True,
        "full_state_evaluation_allowed": False,
        "exact_energy_estimator_enabled": False,
        "statevector_access": "not_used",
        # No policy/evaluator estimate is computed in scalable_large mode,
        # regardless of whether a DMRG reference is attached.
        "energy_estimate": None,
        "energy_difference": None,
        "energy_status": energy_status_field,
        "dss_metrics_status": "not_computed_without_measurement_backend_samples",
        "measurement_backend": measurement_backend,
        "hamiltonian": hamiltonian_metadata,
        "n_batch_elements": len(batch_summaries),
        "n_circuits_total": total_circuits,
        "total_recorded_actions": total_recorded_actions,
        "mean_circuit_length": mean_circuit_length,
        "max_circuit_length": max_circuit_length,
        "batch_elements": batch_summaries,
        # ``dmrg_reference`` is selected
        # iff an actual scalar reference is attached.
        "validation_tier": evaluator_mode_metadata["validation_tier"],
        "provides_scalar_energy": evaluator_mode_metadata[
            "provides_scalar_energy"
        ],
        "sufficient_for_final_readiness_claim": evaluator_mode_metadata[
            "sufficient_for_final_readiness_claim"
        ],
        "tier_description": evaluator_mode_metadata["tier_description"],
        "dmrg_reference_available": evaluator_mode_metadata[
            "dmrg_reference_available"
        ],
        "dmrg_reference_energy": dmrg_reference_energy,
    }

    # Defense-in-depth invariant: a report that advertises
    # final-readiness sufficiency must carry an actual scalar reference.
    # Rekeyed onto ``dmrg_reference_energy`` (the source of truth) so the
    # check no longer requires overloading ``energy_estimate``.
    if (
        report["sufficient_for_final_readiness_claim"]
        and report["dmrg_reference_energy"] is None
    ):
        raise RuntimeError(
            "Internal invariant violated in create_scalable_large_evaluation_report: "
            "validation_tier advertises final-readiness sufficiency but "
            f"dmrg_reference_energy is None (status={report['energy_status']!r}, "
            f"tier={report['validation_tier']!r})."
        )

    return report


def _write_json_atomic(path: Path, payload: Any) -> None:
    """Write JSON atomically so readers do not observe partial artifacts."""
    temp_path = str(path) + ".tmp"
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, cls=NumpyEncoder)
    os.rename(temp_path, str(path))


def _structural_report_sort_key(report: Dict[str, Any]) -> Tuple[int, str, str]:
    """Stable sort key for aggregate structural report artifacts."""
    try:
        update = int(report.get("update", 0))
    except (TypeError, ValueError):
        update = 0
    return (
        update,
        str(report.get("source", "")),
        str(report.get("checkpoint_path", "")),
    )


def _structural_report_identity(report: Dict[str, Any]) -> Tuple[str, str, str]:
    """Dedupe reports loaded from per-update, JSONL, and aggregate artifacts."""
    return (
        str(report.get("update", "")),
        str(report.get("source", "")),
        str(report.get("checkpoint_path", "")),
    )


def _normalize_loaded_report_invariants(
    report: Dict[str, Any],
    *,
    source_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Reconcile a loaded structural report against the scalar source of truth.

    The writer path (``create_scalable_large_evaluation_report``) enforces an
    invariant: ``sufficient_for_final_readiness_claim`` implies a non-None
    ``dmrg_reference_energy``.  A stale or hand-authored sidecar may break
    this — e.g. ``validation_tier="dmrg_reference"`` /
    ``sufficient_for_final_readiness_claim=True`` /
    ``dmrg_reference_energy=false``.  Re-derive every tier-related field
    from the coerced scalar so downstream consumers (markdown rendering,
    aggregation, wiring) cannot observe a contradictory state
    (fourth-round review finding).

    Mutates and returns the input dict for callers that want the
    normalized value back; the same dict is returned for chaining.
    """
    coerced_energy = validation_tier.coerce_dmrg_reference_energy(
        report.get("dmrg_reference_energy")
    )
    tier_metadata = validation_tier.get_validation_tier_metadata(
        evaluator_mode=EVALUATOR_MODE_SCALABLE_LARGE,
        dmrg_reference_energy=coerced_energy,
    )

    # Detect inconsistency before overwriting so a stale-sidecar warning
    # surfaces in logs.  Comparing the existing sidecar fields against the
    # derived state catches the "claims readiness, missing scalar" class.
    inconsistent_fields = [
        key
        for key, derived in (
            ("validation_tier", tier_metadata["validation_tier"]),
            (
                "sufficient_for_final_readiness_claim",
                tier_metadata["sufficient_for_final_readiness_claim"],
            ),
            (
                "dmrg_reference_available",
                tier_metadata["dmrg_reference_available"],
            ),
        )
        if key in report and report[key] != derived
    ]
    if inconsistent_fields:
        logging.warning(
            "Normalising structural report %s: sidecar tier fields %s did "
            "not match the coerced dmrg_reference_energy=%r; using the "
            "scalar-derived values.",
            source_path or "<unknown>",
            inconsistent_fields,
            coerced_energy,
        )

    report["validation_tier"] = tier_metadata["validation_tier"]
    report["sufficient_for_final_readiness_claim"] = tier_metadata[
        "sufficient_for_final_readiness_claim"
    ]
    report["tier_description"] = tier_metadata["tier_description"]
    report["dmrg_reference_available"] = tier_metadata[
        "dmrg_reference_available"
    ]
    report["dmrg_reference_energy"] = coerced_energy
    report["provides_scalar_energy"] = tier_metadata["provides_scalar_energy"]
    # ``energy_status`` is rederived so it cannot lag behind the scalar.
    report["energy_status"] = (
        "structural_report_with_dmrg_reference"
        if coerced_energy is not None
        else "not_computed_structural_report_only"
    )
    # ``energy_estimate`` stays the policy/evaluator field; never the scalar.
    report["energy_estimate"] = None
    return report


def _add_structural_report(
    reports_by_key: Dict[Tuple[str, str, str], ScalableLargeEvaluationReport],
    report: Any,
    *,
    source_path: Path,
) -> None:
    """Add one structural report payload, skipping malformed records."""
    if not isinstance(report, dict):
        logging.warning(
            f"Skipping malformed scalable-large report in {source_path}: "
            f"expected object, got {type(report).__name__}"
        )
        return
    if "update" not in report:
        logging.warning(
            f"Skipping malformed scalable-large report in {source_path}: "
            "missing update"
        )
        return

    # Enforce the readiness/scalar invariant
    # at the load boundary so every downstream consumer (markdown, aggregate,
    # Putting the helper here means
    # all three load paths (aggregate file, JSONL, per-update sidecar) get
    # the same treatment in one place.
    _normalize_loaded_report_invariants(report, source_path=source_path)

    reports_by_key[_structural_report_identity(report)] = report


def save_scalable_large_report_safely(
    report: ScalableLargeEvaluationReport,
    results_dir: Path,
) -> Path:
    """Persist a scalable-large structural report as per-update JSON, JSONL, and aggregate JSON."""
    results_dir.mkdir(parents=True, exist_ok=True)
    update = report.get("update", "unknown")
    report_path = results_dir / f"scalable_large_eval_update_{update}.json"
    _write_json_atomic(report_path, report)

    jsonl_path = results_dir / SCALABLE_LARGE_EVALUATION_JSONL
    with open(jsonl_path, "a", encoding="utf-8") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write(json.dumps(report, cls=NumpyEncoder))
        f.write("\n")
        fcntl.flock(f, fcntl.LOCK_UN)

    aggregate_path = results_dir / SCALABLE_LARGE_EVALUATION_JSON
    _write_json_atomic(
        aggregate_path,
        load_scalable_large_reports(results_dir),
    )

    return report_path


def load_scalable_large_reports(results_dir: Path) -> List[ScalableLargeEvaluationReport]:
    """Load structural reports produced by scalable-large evaluation."""
    reports_by_key = {}

    aggregate_path = results_dir / SCALABLE_LARGE_EVALUATION_JSON
    if aggregate_path.exists():
        try:
            with open(aggregate_path, "r", encoding="utf-8") as f:
                aggregate_payload = json.load(f)
            if isinstance(aggregate_payload, list):
                for report in aggregate_payload:
                    _add_structural_report(
                        reports_by_key,
                        report,
                        source_path=aggregate_path,
                    )
            else:
                _add_structural_report(
                    reports_by_key,
                    aggregate_payload,
                    source_path=aggregate_path,
                )
        except Exception as exc:
            logging.warning(
                f"Could not load scalable-large aggregate {aggregate_path}: {exc}"
            )

    jsonl_path = results_dir / SCALABLE_LARGE_EVALUATION_JSONL
    if jsonl_path.exists():
        try:
            with open(jsonl_path, "r", encoding="utf-8") as f:
                for line_number, line in enumerate(f, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        _add_structural_report(
                            reports_by_key,
                            json.loads(line),
                            source_path=jsonl_path,
                        )
                    except Exception as exc:
                        logging.warning(
                            "Could not load scalable-large JSONL report "
                            f"{jsonl_path}:{line_number}: {exc}"
                        )
        except Exception as exc:
            logging.warning(f"Could not read scalable-large JSONL {jsonl_path}: {exc}")

    for report_path in sorted(results_dir.glob("scalable_large_eval_update_*.json")):
        try:
            with open(report_path, "r", encoding="utf-8") as f:
                _add_structural_report(
                    reports_by_key,
                    json.load(f),
                    source_path=report_path,
                )
        except Exception as exc:
            logging.warning(f"Could not load scalable-large report {report_path}: {exc}")
    return sorted(reports_by_key.values(), key=_structural_report_sort_key)


async def evaluate_top_batch_elements_from_checkpoint_scalable_large(
    checkpoint_path: Path,
    update: int,
    config: ExperimentConfig,
    device: torch.device,
    hamiltonian_helper: Optional[PauliHamiltonianHelper] = None,
    preloaded_checkpoint: Optional[Dict] = None,
) -> ScalableLargeEvaluationReport:
    """Build a structural checkpoint report without constructing EnergyEstimator."""
    logging.info(f"\n=== Structural scalable-large report for update {update} ===")

    checkpoint = preloaded_checkpoint if preloaded_checkpoint is not None \
        else torch.load(checkpoint_path, map_location=device, weights_only=False)
    top_trajectories_data = checkpoint.get("top_trajectories", [])

    if not top_trajectories_data:
        logging.info("  No trajectories found in checkpoint. Skipping report.")
        return {}

    num_to_eval = min(config.n_eval_top_k_batch_elements, len(top_trajectories_data))
    logging.info(f"  Reporting top {num_to_eval} batch elements from checkpoint")

    batch_actions_list = []
    batch_lengths_list = []
    batch_costs = []

    for batch_idx in range(num_to_eval):
        traj_data = top_trajectories_data[batch_idx]
        batch_actions_list.append(traj_data["actions"])
        batch_lengths_list.append(traj_data["lengths"])
        batch_costs.append(traj_data.get("cost", None))

    report = create_scalable_large_evaluation_report(
        batch_actions_list=batch_actions_list,
        batch_lengths_list=batch_lengths_list,
        batch_costs=batch_costs,
        update=update,
        config=config,
        source="checkpoint",
        hamiltonian_metadata=_scalable_large_hamiltonian_metadata(
            config=config,
            hamiltonian_helper=hamiltonian_helper,
            checkpoint=checkpoint,
        ),
        terminal_index=checkpoint.get("terminal_index"),
        checkpoint_path=checkpoint_path,
        measurement_backend=checkpoint.get("measurement_backend"),
    )

    logging.info(
        "  Structural report: "
        f"{report['n_batch_elements']} batch elements, "
        f"{report['n_circuits_total']} circuits, "
        f"mean length {report['mean_circuit_length']:.1f}"
    )
    return report


async def evaluate_top_batch_elements_from_checkpoint(checkpoint_path: Path,
                                                    energy_estimator: Optional[EnergyEstimator],
                                                    update: int,
                                                    config: ExperimentConfig,
                                                    device: torch.device,
                                                    hamiltonian_helper: Optional[PauliHamiltonianHelper] = None,
                                                    preloaded_checkpoint: Optional[Dict] = None) -> EvaluationEntryPointResult:
    """
    Evaluate top-k batch elements from checkpoint without needing trainer instance.

    ``preloaded_checkpoint`` lets the caller pass a checkpoint
    dict that was already validated against a queue's ``checkpoint_id`` —
    avoiding a second open of the mutable ``checkpoint_update.pth`` between
    validation and read (training could have overwritten it).

    Returns:
        In exact_small mode, a list of ExtendedBatchElementEnergyResult rows.
        In scalable_large mode, a structural report dict. Empty structural
        inputs return an empty dict.
    """

    if not get_evaluator_mode_metadata(config)["allows_full_state_evaluation"]:
        return await evaluate_top_batch_elements_from_checkpoint_scalable_large(
            checkpoint_path,
            update,
            config,
            device,
            hamiltonian_helper=hamiltonian_helper,
            preloaded_checkpoint=preloaded_checkpoint,
        )

    ensure_exact_small_evaluation_allowed(
        config,
        "Checkpoint batch-element evaluation",
    )
    if energy_estimator is None:
        raise RuntimeError(
            "Checkpoint batch-element evaluation in exact_small mode requires "
            "an EnergyEstimator instance."
        )

    logging.info(f"\n=== Evaluating checkpoint at update {update} ===")

    # Use the caller-validated checkpoint dict when supplied; only fall
    # through to ``torch.load`` for legacy direct invocations. Without
    # this, the async evaluator's pre-validate-then-reopen pattern would
    # still race against a concurrent training save.
    checkpoint = preloaded_checkpoint if preloaded_checkpoint is not None \
        else torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Extract top trajectories
    top_trajectories_data = checkpoint.get('top_trajectories', [])
    
    if not top_trajectories_data:
        logging.info("  No trajectories found in checkpoint. Skipping evaluation.")
        return []
    
    # Determine how many batch elements to evaluate
    num_to_eval = min(config.n_eval_top_k_batch_elements, len(top_trajectories_data))
    logging.info(f"  Evaluating top {num_to_eval} batch elements from checkpoint")
    
    n_simulations = getattr(config, 'n_simulations', 1)
    if n_simulations > 1:
        logging.info(f"  Running {n_simulations} simulations per batch element")

    # Prepare batch data
    batch_actions_list = []
    batch_lengths_list = []
    batch_costs = []
    
    for batch_idx in range(num_to_eval):
        traj_data = top_trajectories_data[batch_idx]
        
        # Extract actions and lengths
        batch_actions = traj_data['actions'].to(device)
        batch_lengths = traj_data['lengths'].to(device)
        
        # Handle different possible storage formats
        if len(batch_actions.shape) == 1:
            # Single circuit stored - expand to batch
            batch_actions = batch_actions.unsqueeze(0)
            batch_lengths = batch_lengths.unsqueeze(0) if isinstance(batch_lengths, torch.Tensor) else torch.tensor([batch_lengths])
        
        batch_actions_list.append(batch_actions)
        batch_lengths_list.append(batch_lengths)
        batch_costs.append(traj_data.get('cost', 0.0))
    
    # Stack all batch elements
    all_batch_actions = torch.stack(batch_actions_list, dim=0)
    all_batch_lengths = torch.stack(batch_lengths_list, dim=0)
    
    logging.info(f"  Combined batch shape: {all_batch_actions.shape}")
    
    # Run energy estimation (not async - runs synchronously)
    summaries = energy_estimator.estimate_energy_with_simulations(
        all_batch_actions,
        all_batch_lengths,
        M=n_simulations,
    )
    
    # Process results
    results = []
    
    for batch_idx, summary in enumerate(summaries):
        # Extract information from summary
        final_result_obj = summary['final_results_object']
        
        # Create simulation result if we ran multiple simulations
        simulation_result = None
        if n_simulations > 1 and 'energy_variance' in summary:
            mean_energy = summary['mean_energy']
            variance = summary['energy_variance']

            # Only create simulation result if we have actual individual energies
            if 'individual_energies' in summary and 'individual_absolute_errors' in summary:
                simulation_result = SimulationResult(
                    energy_estimates=summary['individual_energies'],
                    absolute_errors=summary['individual_absolute_errors'],
                    rmse=summary['rmse'],
                    std_absolute_error=summary.get('std_absolute_error', 0.0),
                    mean_energy_estimate=mean_energy,
                    std_energy_estimate=np.sqrt(variance) if variance > 0 else 0.0
                )
            else:
                # If individual energies aren't available, don't create fake data
                logging.warning(f"Individual simulation energies not available for batch {batch_idx}")
                simulation_result = None
        
        # Create extended result
        # ``rmse`` / ``mae`` are now first-class fields on the
        # result; ``energy_difference`` carries the absolute-error
        # quantity its name implies (=MAE at M>1).
        result = ExtendedBatchElementEnergyResult(
            batch_element_rank=batch_idx,
            energy_estimate=summary['mean_energy'],
            energy_difference=summary['energy_difference'],
            rmse=summary.get('rmse'),
            mae=summary.get('mae'),
            hitting_counts=final_result_obj.hitting_counts,
            pauli_estimates=summary['mean_pauli_estimates'],
            circuit_lengths=final_result_obj.circuit_lengths,
            mean_circuit_length=final_result_obj.mean_circuit_length,
            n_circuits=final_result_obj.n_circuits,
            total_measurements=final_result_obj.total_measurements,
            convergence_metrics=final_result_obj.convergence_metrics,
            batch_cost=batch_costs[batch_idx],
            update=update,
            simulation_result=simulation_result
        )

        results.append(result)

        # Print summary
        logging.info(f"\n  Batch element rank {batch_idx}:")
        logging.info(f"    Number of circuits: {result.n_circuits}")
        logging.info(f"    Energy estimate: {result.energy_estimate:.6f}")
        # Label the aggregate by what it actually is. At M=1 the
        # field is a per-sim |E - E*|; at M>1 it is the MAE, and the RMSE
        # gets its own log line so the two aggregates can be compared.
        if simulation_result and n_simulations > 1:
            logging.info(f"    MAE (energy_difference): {result.energy_difference:.6e}")
            if result.rmse is not None:
                logging.info(f"    RMSE: {result.rmse:.6e}")
            logging.info(f"    Std absolute error: {simulation_result.std_absolute_error:.6e}")
        else:
            logging.info(f"    Energy difference: {result.energy_difference:.6e}")
        logging.info(f"    Pauli coverage: {result.convergence_metrics['coverage']:.1%}")
        logging.info(f"    Mean circuit length: {result.mean_circuit_length:.1f}")
    
    # Print overall summary
    if results:
        energy_diffs = [r.energy_difference for r in results]
        best_result = min(results, key=lambda r: r.energy_difference)

        # At M>1 each ``energy_difference`` is a MAE, so the
        # aggregates are best-of-MAE / mean-of-MAE. Disambiguate the
        # labels under the M>1 branch.
        diff_label = "MAE" if n_simulations > 1 else "Energy difference"
        logging.info(f"\n  Evaluation Summary:")
        logging.info(
            f"    Best {diff_label} (Batch rank {best_result.batch_element_rank}): "
            f"{best_result.energy_difference:.6e}"
        )
        logging.info(f"    Mean {diff_label}: {np.mean(energy_diffs):.6e}")
        logging.info(f"    Total circuits evaluated: {sum(r.n_circuits for r in results)}")
        if n_simulations > 1:
            logging.info(f"    Total simulation runs: {len(results) * n_simulations}")
        logging.info(f"    Success rate (< 1.6e-3): {sum(1 for e in energy_diffs if e < 1.6e-3) / len(energy_diffs) * 100:.1f}%")

    return results


def evaluator_loop(config: ExperimentConfig,
                   results_dir: Path,
                   hamiltonian_helper: PauliHamiltonianHelper,
                   checkpoint_queue: Queue,
                   results_queue: Queue):
    """
    Evaluator process loop that runs independently of training.
    Polls for new checkpoints and runs energy estimation on CPU.
    """
    evaluator_mode_metadata = get_evaluator_mode_metadata(config)
    logging.info("\n=== Starting Evaluator Process ===")
    logging.info(f"Results directory: {results_dir}")
    logging.info(f"Poll interval: {config.eval_poll_interval}s")
    logging.info(f"Evaluator mode: {evaluator_mode_metadata['mode']}")
    
    # Force CPU for evaluation
    device = torch.device('cpu')
    logging.info(f"Evaluator using device: {device}")
    
    # Initialize exact estimator only for exact-small mode.
    energy_estimator = None
    if evaluator_mode_metadata["allows_full_state_evaluation"]:
        # Defense-in-depth: the ``run_experiment`` guard catches the bulk
        # of misconfiguration upstream, but ``evaluator_loop`` is reachable
        # from direct callers too. Refuse exact-size construction for
        # systems beyond ``EXACT_FULL_STATE_QUBIT_LIMIT`` regardless of
        # the config flag — ``EnergyEstimator(...)`` reads
        # ``hamiltonian_helper.ground_state_energy`` during init, which is
        # the O(2^n) path the no-full-state rule forbids.
        assert_full_state_eval_safe(
            hamiltonian_helper, config,
            "evaluator_loop.EnergyEstimator",
        )
        energy_estimator = EnergyEstimator(
            hamiltonian_helper,
            hamiltonian_helper.n_qubits,
            device=device,
            force_cpu=True
        )
    else:
        logging.info(
            "Scalable-large evaluator will emit structural reports only; "
            "EnergyEstimator and state-vector paths are not constructed."
        )
    
    # Track processed checkpoints
    processed_checkpoints = set()
    
    # Create event loop for async operations
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    # Main evaluation loop
    try:
        while True:
            try:
                # Check for checkpoint from queue (non-blocking)
                checkpoint_info = checkpoint_queue.get_nowait()
                
                if checkpoint_info == 'STOP':
                    logging.info("Received STOP signal. Shutting down evaluator.")
                    break
                    
                checkpoint_path, update, checkpoint_id = checkpoint_info

                # Skip if already processed
                if checkpoint_id in processed_checkpoints:
                    logging.info(f"Skipping already processed checkpoint {checkpoint_id}")
                    continue

                # Validate + load atomically into ``preloaded`` so the
                # downstream evaluator never reopens a potentially-mutated
                # path.: on preload failure for the
                # mutable ``checkpoint_update.pth`` alias, SKIP this queue
                # entry instead of falling back to an unchecked
                # on-demand load — that fallback would reintroduce the
                # stale-queue-label race the preload was meant to close.
                preloaded = None
                try:
                    on_disk = torch.load(
                        checkpoint_path,
                        map_location=device,
                        weights_only=False,
                    )
                    on_disk_id = on_disk.get('checkpoint_id') if isinstance(on_disk, dict) else None
                    on_disk_update = on_disk.get('update') if isinstance(on_disk, dict) else None
                    if on_disk_id is not None and on_disk_id != checkpoint_id:
                        logging.warning(
                            f"Checkpoint at {checkpoint_path} now has id={on_disk_id} "
                            f"(update={on_disk_update}) but the queue payload expects "
                            f"id={checkpoint_id} (update={update}). Skipping stale entry."
                        )
                        processed_checkpoints.add(checkpoint_id)
                        continue
                    preloaded = on_disk  # threaded into the evaluator below
                except Exception as e:
                    logging.warning(
                        f"Could not pre-load mutable checkpoint at {checkpoint_path} "
                        f"(id={checkpoint_id}): {e}. Skipping this queue entry — "
                        f"the next entry will pick up newer model bytes."
                    )
                    # Surface the failure with an error
                    # sentinel. Without this, a preload failure on the
                    # *final* queued checkpoint silently disappeared:
                    # next queue item was ``STOP``, evaluator exited 0,
                    # parent reported clean success even though the
                    # final eval artifact never got written.
                    try:
                        results_queue.put(
                            (EVALUATOR_RESULT_TYPE_ERROR, update, f"preload failure: {e}")
                        )
                    except Exception as put_err:
                        logging.error(f"Could not send preload-error sentinel: {put_err}")
                    # Mark as processed so we don't retry the same id;
                    # newer training updates will queue fresh ids.
                    processed_checkpoints.add(checkpoint_id)
                    continue

                logging.info(f"\n{'='*60}")
                logging.info(f"Processing checkpoint {checkpoint_path} (ID: {checkpoint_id})")
                logging.info(f"Update: {update}")
                logging.info(f"{'='*60}")

                # Run evaluation — pass the validated payload so the
                # evaluator never reopens a potentially-mutated path.
                try:
                    results = loop.run_until_complete(
                        evaluate_top_batch_elements_from_checkpoint(
                            Path(checkpoint_path),
                            energy_estimator,
                            update,
                            config,
                            device,
                            hamiltonian_helper=hamiltonian_helper,
                            preloaded_checkpoint=preloaded,
                        )
                    )
                    
                    if results and evaluator_mode_metadata["allows_full_state_evaluation"]:
                        logging.info(f"\nEvaluation complete for update {update}")
                        logging.info(f"Number of results: {len(results)}")

                        # Save results
                        eval_results_path = results_dir / f'eval_results_update_{update}.json'
                        save_results_safely(results, eval_results_path)
                        logging.info(f"Saved evaluation results to: {eval_results_path}")

                        # Send results back via queue
                        results_queue.put((EVALUATOR_RESULT_TYPE_EXACT, update, results))
                        logging.info(f"Sent {len(results)} results back to main process")

                        # Log summary
                        energy_diffs = [r.energy_difference for r in results]
                        best_energy = min(energy_diffs)
                        logging.info(f"Best energy difference: {best_energy:.6e}")
                    elif results:
                        report_path = save_scalable_large_report_safely(
                            results,
                            results_dir,
                        )
                        results_queue.put(
                            (
                                EVALUATOR_RESULT_TYPE_SCALABLE_LARGE_REPORT,
                                update,
                                results,
                            )
                        )
                        logging.info(
                            f"Saved scalable-large structural report to: {report_path}"
                        )
                    else:
                        logging.warning(f"No results generated for update {update}")

                    processed_checkpoints.add(checkpoint_id)
                        
                except Exception as e:
                    logging.error(f"Error evaluating checkpoint: {e}")
                    import traceback
                    traceback.print_exc()
                    # Surface to the parent so training doesn't report
                    # success with missing/stale eval artifacts. The
                    # outer loop continues processing further queue
                    # entries; the parent's ``check_evaluation_results``
                    # logs the error and records the failure.
                    try:
                        results_queue.put(
                            (EVALUATOR_RESULT_TYPE_ERROR, update, f"per-checkpoint: {e}")
                        )
                    except Exception as put_err:
                        logging.error(f"Could not send per-checkpoint error sentinel: {put_err}")
                    processed_checkpoints.add(checkpoint_id)
                    
            except queue.Empty:
                # No new checkpoints, wait
                time.sleep(config.eval_poll_interval)
                
            except KeyboardInterrupt:
                logging.info("Evaluator interrupted by user.")
                break
                
    except Exception as e:
        logging.error(f"Evaluator process error: {e}")
        import traceback
        traceback.print_exc()
        # Surface the failure to the parent so a broken evaluator can't
        # leave training silently marked successful with missing artifacts
        #. Best-effort: ``results_queue.put`` may itself
        # fail if the queue/process is in a bad state.
        try:
            results_queue.put((EVALUATOR_RESULT_TYPE_ERROR, -1, str(e)))
        except Exception as put_err:
            logging.error(f"Could not send error sentinel to parent: {put_err}")
        # Non-zero exit code so the parent can also detect via
        # ``evaluator_process.exitcode`` even if the queue was unusable.
        try:
            sys.exit(2)
        except SystemExit:
            raise
    finally:
        loop.close()
        logging.info("Evaluator process finished.")


def create_hyperparameters_dict(config: ExperimentConfig, 
                               hamiltonian_helper: PauliHamiltonianHelper,
                               training_pauli_strings: List[str],
                               identity_weight: float) -> Dict:
    """Create a comprehensive hyperparameters dictionary"""

    evaluator_mode_metadata = get_evaluator_mode_metadata(config)
    # Defense-in-depth: refuse exact full-state evaluation for systems
    # beyond the safe qubit limit even when the config defaults
    # ``large_hubbard_mode=False`` ( — a 52-qubit Hubbard
    # config that forgets the flag must not reach ``ground_state_energy``).
    if evaluator_mode_metadata["allows_full_state_evaluation"]:
        assert_full_state_eval_safe(
            hamiltonian_helper, config,
            "create_hyperparameters_dict.exact_ground_state_energy",
        )
    exact_ground_state_energy = (
        hamiltonian_helper.ground_state_energy
        if evaluator_mode_metadata["allows_full_state_evaluation"]
        else None
    )
    exact_energy_estimator_enabled = (
        config.eval_every is not None
        and config.eval_every > 0
        and evaluator_mode_metadata["allows_full_state_evaluation"]
    )
    scalable_large_structural_reporting_enabled = (
        config.eval_every is not None
        and config.eval_every > 0
        and not evaluator_mode_metadata["allows_full_state_evaluation"]
    )
    if exact_energy_estimator_enabled:
        async_mode = (
            "multiprocessing with CPU-only evaluation"
            if config.async_eval
            else "synchronous"
        )
    elif scalable_large_structural_reporting_enabled:
        async_mode = (
            "multiprocessing with CPU-only structural reporting"
            if config.async_eval
            else "synchronous structural reporting"
        )
    else:
        async_mode = "disabled"
    if evaluator_mode_metadata["allows_full_state_evaluation"]:
        energy_estimation_method = "batched_clifford_map_with_state_vector"
        energy_estimation_implementation = "energy_estimator.EnergyEstimator"
        energy_computation = "circuit_based_expectation_value"
    elif evaluator_mode_metadata.get("dmrg_reference_energy") is not None:
        # When a DMRG scalar
        # reference is attached, the run is still structurally evaluated
        # (no policy/evaluator energy estimate) but the reference status
        # is no longer "not computed".  Distinguish the case explicitly so
        # the hyperparameter artifact cannot say structural-only/no-energy
        # while ``validation_tier=dmrg_reference`` claims otherwise.
        energy_estimation_method = "scalable_large_structural_report_with_dmrg_reference"
        energy_estimation_implementation = (
            "main.evaluate_top_batch_elements_scalable_large+dmrg_reference"
        )
        energy_computation = "dmrg_reference_scalar_no_policy_estimate"
    else:
        energy_estimation_method = "scalable_large_structural_report"
        energy_estimation_implementation = "main.evaluate_top_batch_elements_scalable_large"
        energy_computation = "not_computed_structural_report_only"

    # Get device info
    device = get_device(config.device_preference)
    device_info = {
        "type": str(device),
        "preference": config.device_preference
    }
    if device.type == "cuda":
        device_info["cuda_device_name"] = torch.cuda.get_device_name(device)
        device_info["cuda_device_count"] = torch.cuda.device_count()
    sampler_metadata = resolve_sampler_metadata(config, device)
    
    hyperparameters = {
        "experiment": {
            "eval_every": config.eval_every,
            "n_updates": config.n_updates,
            "n_eval_top_k_batch_elements": config.n_eval_top_k_batch_elements,
            "n_simulations": config.n_simulations,
            "results_dir": config.results_dir,
            "timestamp": datetime.now().isoformat(),
            "async_eval": config.async_eval,
            "eval_poll_interval": config.eval_poll_interval,
            "eval_process_timeout": config.eval_process_timeout,
            "large_hubbard_mode": evaluator_mode_metadata["large_hubbard_mode"],
            "evaluator_mode": evaluator_mode_metadata["mode"],
            "measurement_backend": config.measurement_backend,
            "sampler": sampler_metadata["effective_sampler"],
            # validation-tier surface (orthogonal to evaluator mode).
            "validation_tier": evaluator_mode_metadata["validation_tier"],
            "dmrg_reference_available": evaluator_mode_metadata[
                "dmrg_reference_available"
            ],
            "dmrg_reference_energy": evaluator_mode_metadata.get(
                "dmrg_reference_energy"
            ),
        },

        "hamiltonian": {
            "filepath": str(config.hamiltonian_path),
            "n_qubits": hamiltonian_helper.n_qubits,
            "n_terms": len(hamiltonian_helper.pauli_str_list),
            "n_training_terms": len(training_pauli_strings),
            "identity_weight": identity_weight,
            "exact_ground_state_energy": exact_ground_state_energy,
            "exact_ground_state_energy_status": (
                "computed_exact"
                if exact_ground_state_energy is not None
                else "not_computed_scalable_large_mode"
            ),
            "molecule": hamiltonian_helper.filepath.parent.name,
            "transformation": hamiltonian_helper.filepath.stem,
            "excluded_from_training": ["Identity term (I^n)"] if identity_weight != 0 else []
        },
        
        "gfn_model": {
            "model_type": config.model_type,
            "measurement_backend": config.measurement_backend,
            "hidden_dim": config.hidden_dim,
            "num_hidden_layers": config.num_hidden_layers,
            "lr": config.lr,
            "weight_decay": config.weight_decay,
            "grad_clip_value": 100.0,
            "weight_init": "xavier_uniform",
            "logZ_init": config.beta * 0.5,  # Initial logZ value
            "logZ_lr_multiplier": 10
        },

        "sampler": sampler_metadata,
        
        "gfn_objective": {
            "type": config.objective_type,
            "kwargs": config.objective_kwargs,
            "reward_function": config.reward_type,
            "reward_kwargs": config.reward_kwargs
        },
        
        "training": {
            "beta": config.beta,
            "n_measurements": config.n_measurements,
            "update_freq": config.update_freq,
            "max_depth": config.max_depth,
            "K_top_trajectories": config.n_eval_top_k_batch_elements,
            "replay_every": config.replay_every,
            "offpolicy_every": config.offpolicy_every,
            "checkpoint_every": config.checkpoint_every,
            "sampling_modes": ["on_policy", "off_policy", "replay"]
        },
        
        "batch_structure": {
            "batch_element_size": config.n_measurements,
            "batch_elements_per_update": config.update_freq,
            "total_circuits_per_update": config.n_measurements * config.update_freq,
            "description": "Each batch element represents a distribution over circuits"
        },
        
        "cost_function": {
            "type": config.cost_type,
            "kwargs": config.cost_kwargs,
            "zero_stabilizer_cost_weights": config.zero_stabilizer_cost_weights,
            "available_types": ["exponential", "linear_bias", "logarithmic", "ogm", "l1", "confidence"]
        },
        
        "quantum_gates": {
            "single_qubit": ["H", "S", "HS", "SH", "HSH"],
            "two_qubit": ["CNOT"],
            "connectivity": "nearest_neighbor",
            "total_actions": 5 * hamiltonian_helper.n_qubits + 2 * (hamiltonian_helper.n_qubits - 1) + 1,
            "training_note": "Identity Pauli string excluded from cost function"
        },
        
        "energy_estimation": {
            "method": energy_estimation_method,
            "implementation": energy_estimation_implementation,
            "equation": "ô(P) = (1/N_P) ∑_i ⟨b_i|U_i†PU_i|b_i⟩",
            "reference": "Equation (3) from DSS paper",
            "simulations": config.n_simulations if hasattr(config, 'n_simulations') else 1,
            "error_formula": "(1/S) ∑_{s=1}^S |⟨H⟩ - ô_N^(s)(H)|",
            "async_mode": async_mode,
            "exact_energy_estimator_enabled": exact_energy_estimator_enabled,
            "scalable_large_structural_reporting_enabled": scalable_large_structural_reporting_enabled,
            "configured_evaluator_mode": evaluator_mode_metadata["mode"],
            "large_hubbard_mode": evaluator_mode_metadata["large_hubbard_mode"],
            "allows_full_state_evaluation": evaluator_mode_metadata["allows_full_state_evaluation"],
            "mode_description": evaluator_mode_metadata["description"],
            # validation-tier surface (orthogonal to evaluator mode):
            # records the quality of reference this run claims to produce.
            "validation_tier": evaluator_mode_metadata["validation_tier"],
            "provides_scalar_energy": evaluator_mode_metadata["provides_scalar_energy"],
            "sufficient_for_final_readiness_claim": evaluator_mode_metadata[
                "sufficient_for_final_readiness_claim"
            ],
            "tier_description": evaluator_mode_metadata["tier_description"],
            "dmrg_reference_available": evaluator_mode_metadata[
                "dmrg_reference_available"
            ],
            "dmrg_reference_energy": evaluator_mode_metadata.get(
                "dmrg_reference_energy"
            ),
        },

        "computational": {
            "device": device_info,
            "measurement_backend": config.measurement_backend,
            "batch_processing": True,
            "gpu_enabled": True,
            "sparse_matrices": True,
            "async_evaluation": (
                config.async_eval
                and (
                    exact_energy_estimator_enabled
                    or scalable_large_structural_reporting_enabled
                )
            ),
            "configured_async_evaluation": config.async_eval,
            "full_state_evaluation_allowed": evaluator_mode_metadata["allows_full_state_evaluation"],
            "sampler": sampler_metadata["effective_sampler"],
            # P0.5: backward-only knob, NOT a sampler control, so it is
            # deliberately kept out of resolve_sampler_metadata()/the
            # ``sampler`` dict. The value here is the post-__post_init__
            # RESOLVED bool (request default ``None``/auto resolves to
            # ``bool(large_hubbard_mode)``), which is load-bearing: it
            # selects the cached-flow backward recompute path / memory
            # envelope and is not inferable from the request alone.
            "use_activation_checkpointing": bool(
                config.use_activation_checkpointing
            ),
            # uint8 cached-state compression: flow-cache memory knob, NOT a
            # sampler control — sits here next to its checkpointing sibling.
            "use_uint8_state_cache": bool(
                getattr(config, "use_uint8_state_cache", True)
            ),
            # P2.3: bf16 autocast on the gradient-path GEMMs — a BACKWARD-only knob,
            # NOT a sampler control, so it sits here alongside
            # use_activation_checkpointing (kept out of resolve_sampler_metadata).
            "use_bf16_backward": bool(config.use_bf16_backward),
        },

        "evaluation": {
            "mode": evaluator_mode_metadata["mode"],
            "large_hubbard_mode": evaluator_mode_metadata["large_hubbard_mode"],
            "allows_full_state_evaluation": evaluator_mode_metadata["allows_full_state_evaluation"],
            "exact_energy_estimator_enabled": exact_energy_estimator_enabled,
            "scalable_large_structural_reporting_enabled": scalable_large_structural_reporting_enabled,
            "mode_description": evaluator_mode_metadata["description"],
            "energy_computation": energy_computation,
            "convergence_threshold": 1e-3,
            "success_thresholds": [1.6e-3, 1e-2],
            # validation-tier surface (orthogonal to evaluator mode).
            "validation_tier": evaluator_mode_metadata["validation_tier"],
            "provides_scalar_energy": evaluator_mode_metadata["provides_scalar_energy"],
            "sufficient_for_final_readiness_claim": evaluator_mode_metadata[
                "sufficient_for_final_readiness_claim"
            ],
            "tier_description": evaluator_mode_metadata["tier_description"],
            "dmrg_reference_available": evaluator_mode_metadata[
                "dmrg_reference_available"
            ],
            "dmrg_reference_energy": evaluator_mode_metadata.get(
                "dmrg_reference_energy"
            ),
        }
    }
    
    return hyperparameters


def refresh_evaluator_mode_hyperparameters(
    hyperparameters: Dict[str, Any],
    current_hyperparameters: Dict[str, Any],
) -> bool:
    """Refresh mode-sensitive hyperparameter sections loaded from older runs.

    Resume flows intentionally preserve most training metadata, but evaluator
    mode and sampler fields must reflect the current config so reports do not
    describe current execution with stale exact-only or sampler metadata.
    """
    section_keys = {
        "experiment": [
            "eval_every",
            "n_updates",
            "n_eval_top_k_batch_elements",
            "n_simulations",
            "results_dir",
            "async_eval",
            "eval_poll_interval",
            "eval_process_timeout",
            "large_hubbard_mode",
            "evaluator_mode",
            "measurement_backend",
            "sampler",
            "validation_tier",
            "dmrg_reference_available",
            "dmrg_reference_energy",
        ],
        "hamiltonian": [
            "filepath",
            "n_qubits",
            "n_terms",
            "n_training_terms",
            "identity_weight",
            "exact_ground_state_energy",
            "exact_ground_state_energy_status",
            "molecule",
            "transformation",
            "excluded_from_training",
        ],
        "energy_estimation": list(
            current_hyperparameters.get("energy_estimation", {}).keys()
        ),
        "computational": [
            "device",
            "measurement_backend",
            "async_evaluation",
            "configured_async_evaluation",
            "full_state_evaluation_allowed",
            "sampler",
        ],
        # Refresh keys currently emitted by this version. We intentionally do
        # not prune unknown saved keys; older reports may contain diagnostic
        # fields that are harmless to preserve across resume.
        "sampler": list(current_hyperparameters.get("sampler", {}).keys()),
        "evaluation": list(current_hyperparameters.get("evaluation", {}).keys()),
    }

    changed = False
    for section, keys in section_keys.items():
        current_section = current_hyperparameters.get(section, {})
        target_section = hyperparameters.setdefault(section, {})
        for key in keys:
            if key not in current_section:
                continue
            new_value = current_section[key]
            if target_section.get(key) != new_value:
                target_section[key] = new_value
                changed = True

    return changed


def find_latest_experiment(results_base_dir: str) -> Optional[Path]:
    """Find the most recent experiment directory"""
    results_base = Path(results_base_dir)
    if not results_base.exists():
        return None
    
    experiment_dirs = [d for d in results_base.iterdir() 
                      if d.is_dir() and d.name.startswith('experiment_')]
    
    if not experiment_dirs:
        return None
    
    # Sort by modification time and return the most recent
    return max(experiment_dirs, key=lambda d: d.stat().st_mtime)


def _pick_best_checkpoint(checkpoint_files: List[Path]) -> Path:
    """Return the most advanced checkpoint by payload ``update``, mtime as tie-break.

    Filename pattern is ``checkpoint_update[_<N>][_emergency_<sig>_<jobid>].pth``.
    The discovery glob picks them all up, including ``checkpoint_update_0.pth``
    written when an async run crashes at update 0; that file would mtime-shadow
    an older legitimate higher-update checkpoint, rolling back progress on
    resume. Here we:
      1. Read each file's payload ``update`` field via the shared
:func:`_readable_candidate_update` helper, which also handles
         emergency-name filename fallback and ignores unreadable files.
      2. Prefer higher updates. Mtime is only the tie-break (the typical case
         is a periodic save shadowing an emergency save at the same update).
      3. Ignore ``update == 0`` candidates entirely when any candidate has
         ``update > 0`` — a zero-update partial cannot beat real progress.

    Precedence ( -3):

    * Highest payload ``update`` wins.
    * On equal payload ``update``, the **newer mtime** wins. This
      handles "fresh emergency save written after the canonical periodic
      save at the same step" — the picker prefers the most recently
      written file when both contain the same training state.
    * Pinned by ``test_pick_best_checkpoint_tie_breaks_on_mtime``.
    """
    # Candidates that fail to torch.load are marked
    # *ineligible* rather than added with a filename-only update. A
    # corrupt ``checkpoint_update_500_emergency_*.pth`` would otherwise
    # win the high-update sort and then abort the immediately-following
    # ``torch.load`` in ``load_experiment_state``; preferring the next
    # best readable checkpoint is more graceful than crashing resume.
    #
    # Candidate update extraction routed through
    # the shared ``_readable_candidate_update`` helper so this picker
    # and ``highest_existing_checkpoint_update`` (the refuse-stale guard)
    # agree on which file is "advanced".
    readable: List[Tuple[int, float, Path]] = []
    unreadable: List[Path] = []
    for p in checkpoint_files:
        update = _readable_candidate_update(p)
        if update is None:
            logging.warning(
                f"_pick_best_checkpoint: could not read {p.name}. "
                f"Treating as ineligible; will not be selected even if its "
                f"filename suggests a higher update."
            )
            unreadable.append(p)
            continue
        readable.append((update, p.stat().st_mtime, p))

    if not readable:
        # All candidates were unreadable. Re-raise via filename-based
        # selection on the unreadables and let ``torch.load`` fail
        # downstream with a clear message — better than returning None.
        logging.error(
            f"_pick_best_checkpoint: ALL {len(unreadable)} candidate(s) "
            f"failed to torch.load. Falling back to filename-newest; "
            f"resume will likely fail and require manual intervention."
        )
        return max(unreadable, key=lambda p: p.stat().st_mtime)

    max_update = max(u for u, _, _ in readable)
    if max_update > 0:
        # Drop update=0 candidates so a crashed partial can't win.
        readable = [c for c in readable if c[0] > 0]
    # Highest update, then newest mtime
    readable.sort(key=lambda t: (t[0], t[1]))
    chosen = readable[-1][2]
    if len(readable) > 1 or unreadable:
        logging.info(
            f"Picked {chosen.name} (update={readable[-1][0]}) over "
            f"{len(readable)-1} other readable candidate(s) and "
            f"{len(unreadable)} unreadable candidate(s); "
            f"highest-update + mtime tie-break"
        )
    return chosen


def load_experiment_state(experiment_dir: Path) -> Tuple[int, Dict, List[ExtendedBatchElementEnergyResult], Dict]:
    """Load the state of a previous experiment"""
    logging.info(f"Loading experiment state from {experiment_dir}")
    
    # Load configuration
    with open(experiment_dir / 'config.json', 'r') as f:
        saved_config = json.load(f)
    
    # Load hyperparameters
    with open(experiment_dir / 'hyperparameters.json', 'r') as f:
        hyperparameters = json.load(f)
    
    # Find the most advanced checkpoint, not the most recently modified one.
    # mtime-based selection was prone to two bugs:
    #   1. ``checkpoint_update_0.pth`` (partial async crash at update 0)
    #      could mtime-shadow an older valid high-update checkpoint.
    #   2. An emergency save written *before* a periodic save with the
    #      same name could rollback progress on resume.
    # Pick by payload ``update`` (or filename update as fallback), mtime
    # only as tie-break — and prefer non-zero updates so a crashed
    # partial doesn't outvote a real checkpoint.
    checkpoint_files = list(experiment_dir.glob('checkpoint_update*.pth'))
    if not checkpoint_files:
        start_update = 0
        metrics_history = defaultdict(list)
    else:
        latest_checkpoint = _pick_best_checkpoint(checkpoint_files)
        # Use the same ``torch.load`` semantics as
        # ``_pick_best_checkpoint`` so a candidate the picker ranked
        # successfully cannot fail or behave differently on the real
        # resume load. ``weights_only=False`` is the canonical training
        # checkpoint contract (we own these writes).
        checkpoint = torch.load(latest_checkpoint, map_location='cpu', weights_only=False)

        # Shared filename helper (matches emergency suffix names too).
        # The previous stricter
        # ``(\d+)\.pth`` regex did not match
        # ``checkpoint_update_<N>_emergency_<SIG>_<JOBID>.pth``, so
        # emergency-only resumes whose payload lacked ``update`` fell
        # back to ``0`` and threw away progress.
        filename_update = checkpoint_filename_update(latest_checkpoint.name)

        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            # Explicit ``None`` check instead
            # of ``or`` chain. An ``or`` chain conflates legitimate ``0``
            # with ``None``: if both ``update`` and ``epoch`` are
            # explicit-zero but a filename hint exists, the chain falls
            # through to the filename. Treat ``None``-absent as
            # "no metadata", but trust ``0`` as a real value.
            payload_update = checkpoint.get('update')
            if payload_update is None:
                payload_update = checkpoint.get('epoch')
            if payload_update is None:
                start_update = int(filename_update)
            else:
                try:
                    start_update = int(payload_update)
                except (TypeError, ValueError):
                    start_update = int(filename_update)
            
            # Try to get metrics history, handle different key names
            if 'metrics_history' in checkpoint:
                metrics_history = defaultdict(list, checkpoint['metrics_history'])
            elif 'metrics' in checkpoint:
                metrics_history = defaultdict(list, checkpoint['metrics'])
            else:
                # No metrics found, start fresh
                metrics_history = defaultdict(list)
                logging.warning("No metrics history found in checkpoint, starting fresh")
        else:
            # Old format - just the state dict
            start_update = filename_update
            metrics_history = defaultdict(list)
            logging.warning(f"Old checkpoint format detected, using update number from filename: {start_update}")
        
        logging.info(f"Found checkpoint at update {start_update}")
    
    # Load existing evaluation results
    evaluation_results = []
    
    # Check for individual eval result files (from async evaluation).
    # Routed through ``_extended_result_from_record`` so the
    # legacy-RMSE migration and ``rmse`` / ``mae`` field passthrough
    # match the JSONL store loader exactly.
    eval_files = list(experiment_dir.glob('eval_results_update_*.json'))
    for eval_file in eval_files:
        with open(eval_file, 'r') as f:
            results_data = json.load(f)

        for r_dict in results_data:
            try:
                evaluation_results.append(_extended_result_from_record(r_dict))
            except Exception as e:
                logging.warning(
                    f"Could not load result from {eval_file}: {e}"
                )

    jsonl_file = experiment_dir / 'evaluation_results.jsonl'
    if jsonl_file.exists():
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r_dict = json.loads(line)
                    evaluation_results.append(
                        _extended_result_from_record(r_dict)
                    )
                except Exception as e:
                    logging.warning(
                        f"Could not load result from {jsonl_file}: {e}"
                    )
    else:
        # Legacy evaluation_results.json support
        eval_file = experiment_dir / 'evaluation_results.json'
        if eval_file.exists():
            with open(eval_file, 'r') as f:
                results_data = json.load(f)

            for r_dict in results_data:
                if any(r.update == r_dict['update'] and r.batch_element_rank == r_dict['batch_element_rank']
                       for r in evaluation_results):
                    continue
                try:
                    evaluation_results.append(
                        _extended_result_from_record(r_dict)
                    )
                except Exception as e:
                    logging.warning(
                        f"Could not load result from legacy "
                        f"evaluation_results.json: {e}"
                    )
    
    if evaluation_results:
        logging.info(f"Loaded {len(evaluation_results)} existing evaluation results")
        # Note: ``start_update`` is *only* advanced from real model
        # checkpoints (``checkpoint_update*.pth``) — never from
        # evaluation reports or structural reports, even when the
        # checkpoint glob came back empty. Reports are reporting history;
        # advancing the training cursor past freshly-initialised weights
        # would skip updates silently. The previous behaviour ("if
        # start_update == 0, use last eval update") rolled the loop
        # forward without loading any policy state.

    structural_reports = load_scalable_large_reports(experiment_dir)
    if structural_reports:
        logging.info(
            "Loaded "
            f"{len(structural_reports)} existing scalable-large structural reports"
        )
        # Same rule: structural reports describe *what was evaluated*, not
        # *what was trained*. ``start_update`` stays 0 here if no real
        # checkpoint is on disk so the next ``run_experiment`` either
        # starts from scratch on freshly-initialised weights (correct) or
        # fails loudly when ``check_config_compatibility`` notices the
        # mismatch — not silently skipping updates.

    # Fail-loud sanity check: if we found reporting history but the
    # checkpoint glob produced nothing, leave ``start_update == 0`` and
    # log a warning so the operator can decide whether to delete the
    # stale reports or recover a checkpoint from elsewhere.:
    # *additionally* quarantine the stale reports into a
    # ``stale_reports_<ts>/`` subdirectory before training restarts so
    # final summaries / aggregations don't pick them up as if they
    # described the fresh run.
    if start_update == 0 and (evaluation_results or structural_reports):
        n_eval = len(evaluation_results)
        n_struct = len(structural_reports) if structural_reports else 0
        logging.warning(
            "Found %d evaluation result(s) and %d structural report(s) "
            "but no loadable checkpoint. Training will start from update 0 "
            "with freshly-initialised weights. Stale reports will be moved "
            "to a quarantine subdirectory so they don't pollute fresh-run "
            "summaries.",
            n_eval, n_struct,
        )
        try:
            # Match the *actual* writer filenames:
            #   - exact-small per-update: ``eval_results_update_<N>.json``
            #   - exact-small aggregate JSONL: ``evaluation_results.jsonl``
            #     (written by ``DiskBackedResultStore``)
            #   - exact-small legacy aggregate JSON: ``evaluation_results.json``
            #     and ``all_evaluation_results.json``
            #     (reporter exports)
            #   - scalable-large per-update: ``scalable_large_eval_update_<N>.json``
            #   - scalable-large aggregates: module constants
            #     ``SCALABLE_LARGE_EVALUATION_JSON`` / ``..._JSONL``.
            # The exact-side aggregate JSONL/JSON must be included: without it a
            # fresh run after a checkpoint-less restart still sees stale
            # ``EnergyEstimator`` rows in final summaries.
            patterns = [
                "eval_results_update_*.json",
                "evaluation_results.jsonl",
                "evaluation_results.json",
                "all_evaluation_results.json",
                "scalable_large_eval_update_*.json",
                SCALABLE_LARGE_EVALUATION_JSON,
                SCALABLE_LARGE_EVALUATION_JSONL,
            ]
            to_move = []
            for pattern in patterns:
                to_move.extend(experiment_dir.glob(pattern))
            if not to_move:
                # Don't create an empty quarantine subdir — there's
                # nothing to put in it.
                logging.warning(
                    "No matching report files found to quarantine "
                    "(structural artifacts may have already been cleaned up)."
                )
            else:
                ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
                quarantine = experiment_dir / f"stale_reports_{ts}"
                quarantine.mkdir(parents=True, exist_ok=True)
                moved = 0
                for p in to_move:
                    p.rename(quarantine / p.name)
                    moved += 1
                logging.warning(
                    "Quarantined %d stale report file(s) → %s. Training "
                    "starts fresh; aggregations and final summaries will "
                    "see only the new run's artifacts.",
                    moved, quarantine,
                )
                # Drop the now-quarantined entries from the return so the
                # caller doesn't replay them into the fresh run's pipeline.
                evaluation_results = []
        except Exception as quarantine_err:
            logging.error(
                "Failed to quarantine stale reports under %s: %s. "
                "Delete them manually or rename the experiment dir.",
                experiment_dir, quarantine_err,
            )

    # Filter + PERSIST. Just dropping
    # ``evaluation_results`` in memory wasn't enough — the
    # ``DiskBackedResultStore`` re-reads ``evaluation_results.jsonl``
    # later, and ``load_scalable_large_reports`` re-globs the per-update
    # JSONs at final-report time, so stale future rows would still
    # re-enter the resumed run's summaries. Quarantine the
    # offending on-disk files into ``stale_future_reports_<ts>/`` so
    # downstream re-reads only see entries ≤ start_update.
    if start_update > 0:
        before_eval = len(evaluation_results)
        evaluation_results = [
            r for r in evaluation_results if int(getattr(r, "update", 0) or 0) <= start_update
        ]
        dropped_eval = before_eval - len(evaluation_results)

        try:
            future_files = []
            # Per-update exact eval files (found these
            # in memory but not on disk).
            for p in experiment_dir.glob("eval_results_update_*.json"):
                m = re.search(r"eval_results_update_(\d+)\.json", p.name)
                if m and int(m.group(1)) > start_update:
                    future_files.append(p)
            # Per-update structural reports.
            for p in experiment_dir.glob("scalable_large_eval_update_*.json"):
                m = re.search(r"scalable_large_eval_update_(\d+)\.json", p.name)
                if m and int(m.group(1)) > start_update:
                    future_files.append(p)

            # Helper: atomic filter+rewrite of a JSONL file, keeping only
            # rows whose ``update`` field is ``<= start_update``.
            def _rewrite_jsonl_filtered(path: Path) -> int:
                if not path.exists():
                    return 0
                kept_lines = []
                dropped = 0
                for line in path.read_text(encoding="utf-8").splitlines():
                    s = line.strip()
                    if not s:
                        continue
                    try:
                        row = json.loads(s)
                        if int(row.get("update", 0) or 0) <= start_update:
                            kept_lines.append(line)
                        else:
                            dropped += 1
                    except Exception:
                        kept_lines.append(line)  # keep malformed rows
                if dropped:
                    tmp = path.with_suffix(path.suffix + ".tmp")
                    tmp.write_text("\n".join(kept_lines) + ("\n" if kept_lines else ""), encoding="utf-8")
                    tmp.rename(path)
                return dropped

            # Exact-side JSONL (DiskBackedResultStore source).
            jsonl_path = experiment_dir / "evaluation_results.jsonl"
            rewrote_jsonl_rows = _rewrite_jsonl_filtered(jsonl_path)

            # Also rewrite the scalable-large aggregate
            # JSON/JSONL. ``load_scalable_large_reports`` reads the
            # aggregate JSON FIRST (it has all rows in one file); even
            # with per-update files quarantined, a stale aggregate row
            # past ``start_update`` would re-enter the final report.
            sl_jsonl_path = experiment_dir / SCALABLE_LARGE_EVALUATION_JSONL
            sl_rewrote_jsonl_rows = _rewrite_jsonl_filtered(sl_jsonl_path)

            sl_json_path = experiment_dir / SCALABLE_LARGE_EVALUATION_JSON
            sl_dropped_json_rows = 0
            if sl_json_path.exists():
                try:
                    arr = json.loads(sl_json_path.read_text(encoding="utf-8"))
                    if isinstance(arr, list):
                        kept = [r for r in arr if isinstance(r, dict)
                                and int(r.get("update", 0) or 0) <= start_update]
                        sl_dropped_json_rows = len(arr) - len(kept)
                        if sl_dropped_json_rows:
                            tmp = sl_json_path.with_suffix(".json.tmp")
                            tmp.write_text(json.dumps(kept, cls=NumpyEncoder), encoding="utf-8")
                            tmp.rename(sl_json_path)
                except Exception as sl_err:
                    logging.warning(
                        f"Could not filter scalable-large aggregate JSON: {sl_err}"
                    )

            if future_files or rewrote_jsonl_rows or sl_rewrote_jsonl_rows or sl_dropped_json_rows:
                ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
                quarantine = experiment_dir / f"stale_future_reports_{ts}"
                quarantine.mkdir(parents=True, exist_ok=True)
                for p in future_files:
                    p.rename(quarantine / p.name)
                logging.warning(
                    "Filtered/quarantined stale post-resume artifacts: "
                    "%d in-memory evaluation row(s), %d exact JSONL row(s), "
                    "%d scalable-large JSONL row(s), %d scalable-large aggregate JSON row(s), "
                    "%d per-update file(s) moved → %s",
                    dropped_eval, rewrote_jsonl_rows,
                    sl_rewrote_jsonl_rows, sl_dropped_json_rows,
                    len(future_files), quarantine,
                )
            elif dropped_eval:
                logging.warning(
                    "Filtered %d in-memory evaluation result(s) with update > "
                    "start_update=%d; no on-disk files needed quarantine.",
                    dropped_eval, start_update,
                )
        except Exception as quarantine_err:
            logging.error(
                "Failed to quarantine post-resume stale reports: %s. "
                "In-memory filter still applied; manual cleanup recommended.",
                quarantine_err,
            )

    return start_update, metrics_history, evaluation_results, hyperparameters


# List of fused performance
# knobs whose values are bit-identical (parity-tested) so changing
# them between runs is resume-safe and does not count as a
# hyperparameter mismatch. ``check_config_compatibility`` excludes
# these from ``all_params``; the resume paths log their deltas
# separately via ``_log_fused_knob_resume_deltas``.
_FUSED_RESUME_SAFE_KNOBS: Tuple[str, ...] = (
    # MEMBERSHIP NOTE: the resume-delta loop below reads `.get(knob, True)`, so a
    # member ideally has ExperimentConfig default True. The fused-kernel knobs do.
    # ``use_activation_checkpointing`` is a tolerated pre-existing exception: its
    # default is None -> bool(large_hubbard_mode), so a pre-P0.5 config.json missing
    # the key can log one benign spurious True->False delta on a non-large-hubbard
    # resume (log-only, not a fork). Do NOT add a new False-default knob here (e.g.
    # use_bf16_backward) — handle it in its own block in _log_fused_knob_resume_deltas
    # with the correct default (that is exactly what P2.3 does).
    'use_fused_metadata_kernel',
    'use_fused_sampling_kernel',
    'use_fused_mask_counts_kernel',
    'use_fused_counter_rng_kernel',
    'use_fused_partition_update_kernel',
    'use_fused_apply_kernel',
    'use_activation_checkpointing',
    'use_uint8_state_cache',
    # NOTE: ``use_bf16_sampling`` is intentionally absent: it is NOT bit-identical
    # (bf16 perturbs the gumbel-argmax), so it gets its own logging path in
    # ``_log_fused_knob_resume_deltas`` rather than the "non-training-affecting" message.
    # NOTE: ``use_bf16_backward`` is ALSO absent for the same reason: bf16 on the
    # gradient-path GEMMs perturbs the gradient, but within seed variance, so it is
    # resume-safe and gets its own resume-diff log line rather than all_params.
    # NOTE: ``FLOWMEAS_FLOW_DEDUP`` is also absent, but because it is an env-var-only
    # emergency off-switch rather than an ``ExperimentConfig`` field — it never
    # appears in saved config.json files so there is nothing to compare across
    # resumes. If it is ever promoted to a config field it should be added here
    # (it is bit-identical / exact by construction).
)

# Bit-identical perf knobs that travel inside ``model_kwargs`` (not top-level
# ExperimentConfig fields). They must be stripped before the resume
# compatibility comparison so toggling them — parity test, broken-CuPy
# rollback, GPU upgrade — never forks a fresh experiment dir, matching the
# treatment of the top-level fused kernels above.
# 'embed_impl' is also bit-identical (dense vs bag are exact) — same
# rationale as use_fused_unpack.
_PERF_ONLY_MODEL_KWARGS: frozenset = frozenset({'use_fused_unpack', 'embed_impl'})


def _normalize_model_kwargs(d: Optional[Dict]) -> Dict:
    """Drop bit-identical perf-only keys so resume compares only the
    architecture-affecting model_kwargs (e.g. row_embed_dim, pool, and the
    stripped perf-only keys use_fused_unpack / embed_impl)."""
    return {k: v for k, v in (d or {}).items() if k not in _PERF_ONLY_MODEL_KWARGS}


def _log_fused_knob_resume_deltas(
    saved_config: Dict,
    current_config: "ExperimentConfig",
) -> None:
    """Log any fused-knob deltas between ``saved_config`` and
    ``current_config`` so operators can correlate new sampler-metadata
    values in ``hyperparameters.json`` with the actual config change.

    Called from the two resume branches that would otherwise hide a fused-knob
    change (the plain ``elif nn_match`` weight-transfer branch is NOT a caller —
    it already dumps every differing param, including the fused knobs and
    ``use_bf16_sampling``, via its own ``diff_params`` list):

    * ``all_match=True`` (normal resume): operator sees "All
      hyperparameters match" but a fused knob may still have
      changed since ``check_config_compatibility`` excludes those
      from ``all_params``..
    * ``is_requeued_job and nn_match`` (forced requeue resume):
      operator may not even see the diff_params dump because that
      runs only on the plain ``elif nn_match`` branch (skipped when
      a requeue short-circuits to weight reuse). — without this call here, a requeued run that flips a
      fused knob still logged only the generic "Some config
      parameters differ…" line without the detail.

    Resume-safe contract is stated explicitly in the log so a future
    maintainer doesn't mistake the diff for a hyperparameter mismatch.
    """
    deltas: List[Tuple[str, bool, bool]] = []
    for knob in _FUSED_RESUME_SAFE_KNOBS:
        saved_v = _coerce_bool_config(saved_config.get(knob, True), knob)
        current_v = _coerce_bool_config(
            getattr(current_config, knob, True), knob
        )
        if saved_v != current_v:
            deltas.append((knob, saved_v, current_v))
    # Also surface bit-identical perf knobs that live INSIDE model_kwargs (e.g.
    # use_fused_unpack). _normalize_model_kwargs strips them from the compat
    # check, so without this they'd change silently on resume — mirror the
    # top-level fused-knob "omitted from all_params but logged in the diff"
    # treatment. Default missing to True (the knob's model default), matching
    # the _FUSED_RESUME_SAFE_KNOBS handling above.
    saved_mk = saved_config.get("model_kwargs", {}) or {}
    current_mk = getattr(current_config, "model_kwargs", {}) or {}
    for k in _PERF_ONLY_MODEL_KWARGS:
        s_v, c_v = saved_mk.get(k, True), current_mk.get(k, True)
        if s_v != c_v:
            deltas.append((f"model_kwargs.{k}", s_v, c_v))
    # ``use_bf16_sampling`` is intentionally NOT in _FUSED_RESUME_SAFE_KNOBS:
    # unlike the fused kernels it is NOT bit-identical (bf16 perturbs the
    # gumbel-argmax, so sampled trajectories differ). But its effect is within
    # run-to-run exploration variance, so it is likewise omitted from
    # ``all_params`` (resume-safe, non-forking). Log it on its own honest line —
    # not lumped into the "non-training-affecting" fused message — defaulting a
    # missing key to the config default (True) so pre-bf16 configs don't spam.
    saved_bf16 = _coerce_bool_config(saved_config.get("use_bf16_sampling", True), "use_bf16_sampling")
    current_bf16 = _coerce_bool_config(getattr(current_config, "use_bf16_sampling", True), "use_bf16_sampling")
    if saved_bf16 != current_bf16:
        logging.info(
            f"Note: use_bf16_sampling changed on resume ({saved_bf16} → {current_bf16}); "
            "sampler-only (within exploration variance), resume-safe — not a "
            "hyperparameter mismatch."
        )
    # P2.3: ``use_bf16_backward`` (gradient-path bf16 GEMMs) follows the same
    # treatment — NOT bit-identical (perturbs the gradient) but within seed
    # variance (resume-safe, non-forking), so log it honestly here instead of
    # all_params. Default missing key to False (its ExperimentConfig default).
    saved_bf16b = _coerce_bool_config(saved_config.get("use_bf16_backward", False), "use_bf16_backward")
    current_bf16b = _coerce_bool_config(getattr(current_config, "use_bf16_backward", False), "use_bf16_backward")
    if saved_bf16b != current_bf16b:
        logging.info(
            f"Note: use_bf16_backward changed on resume ({saved_bf16b} → {current_bf16b}); "
            "gradient-path precision (within seed variance), resume-safe — not a "
            "hyperparameter mismatch."
        )
    device = get_device(getattr(current_config, "device_preference", "auto"))
    saved_sampling = _resolve_sampling_mode_controls(
        saved_config.get("static_shape_sampling"),
        saved_config.get("cuda_graph_sampling"),
        saved_config.get("sampling_mode"),
        device,
        warn_inconsistent=False,
    )
    current_sampling = _resolve_sampling_mode_controls(
        getattr(current_config, "static_shape_sampling", None),
        getattr(current_config, "cuda_graph_sampling", None),
        getattr(
            current_config,
            "_requested_sampling_mode_raw",
            getattr(current_config, "sampling_mode", None),
        ),
        device,
        warn_inconsistent=False,
    )
    sampling_delta_keys = (
        "requested_sampling_mode",
        "effective_sampling_mode",
        "cuda_graph_sampling",
    )
    saved_sampling_for_compare = dict(saved_sampling)
    current_sampling_for_compare = dict(current_sampling)
    for sampling in (saved_sampling_for_compare, current_sampling_for_compare):
        sampling["requested_sampling_mode"] = _coerce_sampling_mode_config(
            sampling["requested_sampling_mode"],
            warn_alias=False,
        )
    if any(
        saved_sampling_for_compare[key] != current_sampling_for_compare[key]
        for key in sampling_delta_keys
    ):
        logging.info(
            "Note: sampling_mode changed on resume "
            f"(requested: {saved_sampling_for_compare['requested_sampling_mode']} → "
            f"{current_sampling_for_compare['requested_sampling_mode']}; effective: "
            f"{saved_sampling['effective_sampling_mode']} → "
            f"{current_sampling['effective_sampling_mode']}; "
            f"cuda_graph_sampling: {saved_sampling['cuda_graph_sampling']} → "
            f"{current_sampling['cuda_graph_sampling']}); sampler perf knob, "
            "resume-safe — not a hyperparameter mismatch."
        )
    # The use_bf16_sampling, use_bf16_backward, and sampling_mode blocks above
    # fire UNCONDITIONALLY (they are not in ``deltas``) — do NOT hoist this
    # early return above them.
    # Dynamic-active CUDA graph replay is a default-off sampler-only
    # performance knob. It captures the eager inner module while the default
    # no-replay path may use torch.compile, so log it honestly outside the
    # bit-identical fused-knob bucket and default missing keys to False.
    saved_dyn_graph = _coerce_bool_config(
        saved_config.get("use_cuda_graph_policy", False),
        "use_cuda_graph_policy",
    )
    current_dyn_graph = _coerce_bool_config(
        getattr(current_config, "use_cuda_graph_policy", False),
        "use_cuda_graph_policy",
    )
    if saved_dyn_graph != current_dyn_graph:
        logging.info(
            f"Note: use_cuda_graph_policy changed on resume ({saved_dyn_graph} → {current_dyn_graph}); "
            "sampler-only performance knob, resume-safe — not a "
            "hyperparameter mismatch."
        )
    saved_dyn_cap = int(saved_config.get("cuda_graph_policy_max_rows", 2048))
    current_dyn_cap = int(getattr(current_config, "cuda_graph_policy_max_rows", 2048))
    if saved_dyn_cap != current_dyn_cap:
        logging.info(
            f"Note: cuda_graph_policy_max_rows changed on resume ({saved_dyn_cap} → {current_dyn_cap}); "
            "sampler-only performance knob, resume-safe — not a "
            "hyperparameter mismatch."
        )
    # The use_bf16_sampling/use_bf16_backward/sampling-mode/dynamic-graph blocks
    # above fire UNCONDITIONALLY (they are not in ``deltas``) — do NOT hoist this
    # early return above them.
    # test_resume_delta_logs_bf16_backward_change exercises the empty-deltas path.
    if not deltas:
        return
    logging.info(
        "Note: fused performance knobs changed on resume "
        "(non-training-affecting; resume-safe — not a hyperparameter "
        "mismatch):"
    )
    for knob, saved_v, current_v in deltas:
        logging.info(f"  {knob}: {saved_v} → {current_v}")


def check_config_compatibility(saved_config: Dict, current_config: ExperimentConfig) -> Tuple[bool, bool, bool]:
    """
    Check compatibility between saved and current configurations.

    Returns:
        (nn_params_match, all_params_match, max_depth_changed): Tuple indicating compatibility levels
    """
    def saved_value(param: str) -> Any:
        if param == "large_hubbard_mode":
            return _coerce_bool_config(
                saved_config.get(param, False),
                "large_hubbard_mode",
            )
        if param == "zero_stabilizer_cost_weights":
            # Default-False bool; a pre-existing config.json that predates the key
            # must compare equal to a current default-False run (no spurious fork).
            return _coerce_bool_config(
                saved_config.get(param, False),
                "zero_stabilizer_cost_weights",
            )
        if param == "measurement_backend":
            value = saved_config.get(param)
            return "auto" if value is None or value == "auto" else value
        # Default to the ExperimentConfig defaults so a pre-existing config.json
        # that predates these keys still resumes cleanly (saved-missing == current
        # default), rather than spuriously forking a fresh experiment dir.
        if param == "model_type":
            return saved_config.get(param, "clifford_mlp")
        if param == "model_kwargs":
            return _normalize_model_kwargs(saved_config.get(param, {}))
        return saved_config.get(param)

    def current_value(param: str) -> Any:
        if param == "large_hubbard_mode":
            return _coerce_bool_config(
                getattr(current_config, param, False),
                "large_hubbard_mode",
            )
        if param == "zero_stabilizer_cost_weights":
            return _coerce_bool_config(
                getattr(current_config, param, False),
                "zero_stabilizer_cost_weights",
            )
        if param == "measurement_backend":
            value = getattr(current_config, param, None)
            return "auto" if value is None or value == "auto" else value
        if param == "model_kwargs":
            return _normalize_model_kwargs(getattr(current_config, param, {}))
        return getattr(current_config, param)

    # NN-architecture identity. model_type / model_kwargs join the classic NN
    # dims here so an architecture switch (clifford_mlp -> packed_w_rowtoken, or a
    # changed row_embed_dim) makes nn_params_match False and forks a FRESH
    # experiment dir instead of transferring mismatched weights into the new
    # model. Bit-identical perf knobs inside model_kwargs (use_fused_unpack) are
    # stripped by _normalize_model_kwargs so toggling them never forks. Compared
    # through saved_value/current_value so the normalization + defaults apply.
    nn_params = ['hidden_dim', 'num_hidden_layers', 'model_type', 'model_kwargs']
    nn_params_match = all(
        saved_value(param) == current_value(param)
        for param in nn_params
    )

    # Check if max_depth changed
    max_depth_changed = saved_config.get('max_depth') != current_config.max_depth
    
    # Define all critical hyperparameters for full compatibility.
    #
    # Deliberately omitted: the fused-kernel knobs (``use_fused_*_kernel``),
    # ``use_activation_checkpointing`` and ``use_uint8_state_cache``. These are
    # bit-identical performance knobs, not training-affecting hyperparameters.
    # Also omitted, for a slightly different reason: ``use_bf16_sampling``,
    # ``use_bf16_backward``, ``use_cuda_graph_policy``,
    # ``cuda_graph_policy_max_rows``, ``static_shape_sampling``,
    # ``cuda_graph_sampling`` and ``sampling_mode``. Not all of those are
    # bit-identical (bf16 perturbs the sampling gumbel-argmax and the gradient
    # path), but they select a sampler/precision implementation rather than the
    # objective, and forking on them would break resume for every pre-knob config
    # while defeating their purpose as toggleable knobs. Every one of them is
    # logged in the resume diff (``_log_fused_knob_resume_deltas``), so a stale
    # ``config.json`` is still flagged without creating a fresh experiment dir.
    #
    # ``dmrg_reference_energy`` / ``dmrg_reference_path`` are omitted too: they
    # report the quality of the reference the run produces but never enter the
    # loss, so recomputing a tighter reference between resumes is resume-safe.
    #
    # ``FLOWMEAS_FLOW_DEDUP`` and ``FLOWMEAS_FLOW_CHUNK_SIZE`` have no entry
    # because they are env-var-only escape hatches that never appear in a saved
    # config.json, and both are numerically equivalent by design.
    all_params = [
        'hamiltonian_path', 'n_measurements', 'max_depth', 'beta',
        'hidden_dim', 'num_hidden_layers', 'lr', 'weight_decay',
        'reward_type', 'reward_kwargs', 'cost_type', 'cost_kwargs',
        'objective_type', 'objective_kwargs',
        'update_freq', 'n_eval_top_k_batch_elements',
        'large_hubbard_mode', 'measurement_backend',
        # ``zero_stabilizer_cost_weights`` IS training-affecting (it changes the cost's
        # weight vector and hence the reward landscape the policy optimizes), so unlike
        # the bit-identical fused perf knobs it MUST be in all_params: toggling it forks
        # a fresh experiment dir instead of resuming a policy trained on a different
        # objective. Missing-key legacy configs compare as False (saved_value handler).
        'zero_stabilizer_cost_weights',
        # Architecture identity: a different policy network (or its kwargs, e.g.
        # row_embed_dim/pool for packed_w_rowtoken) must fork a fresh experiment
        # dir rather than resume into mismatched weights.
        'model_type', 'model_kwargs',
    ]
    
    # Check full compatibility
    all_params_match = all(
        saved_value(param) == current_value(param)
        for param in all_params
    )
    
    return nn_params_match, all_params_match, max_depth_changed


def extract_logZ_from_checkpoint(checkpoint_path: Path) -> Optional[float]:
    """
    Extract logZ value from a checkpoint file.

    Returns:
        logZ value if found, None otherwise
    """
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        if isinstance(checkpoint, dict):
            # Try to get logZ from model state dict
            state_dict = None
            if 'pf_model_state_dict' in checkpoint:
                state_dict = checkpoint['pf_model_state_dict']
            elif 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            
            if state_dict and 'logZ' in state_dict:
                logZ_tensor = state_dict['logZ']
                if torch.is_tensor(logZ_tensor):
                    return logZ_tensor.item() if logZ_tensor.numel() == 1 else logZ_tensor[0].item()
            
            # Try to get from metrics
            if 'metrics' in checkpoint:
                metrics = checkpoint['metrics']
                if isinstance(metrics, dict) and 'logZ' in metrics:
                    logZ_val = metrics['logZ']
                    if torch.is_tensor(logZ_val):
                        return logZ_val.item() if logZ_val.numel() == 1 else logZ_val[0].item()
                    return float(logZ_val)
        
        return None
    except Exception as e:
        logging.warning(f"Failed to extract logZ from checkpoint: {e}")
        return None


def load_checkpoint_weights_only(trainer: EfficientGFNTrainer, checkpoint_path: Path, device) -> bool:
    """
    Load only the neural network weights from a checkpoint, including logZ.

    Returns:
        True if weights were loaded successfully, False otherwise
    """
    try:
        logging.info(f"Loading neural network weights from {checkpoint_path}")
        # Match ``_pick_best_checkpoint`` /
        # ``load_experiment_state`` and pass ``weights_only=False``
        # explicitly. FlowMeas training checkpoints are trusted; on
        # PyTorch versions where the default tightens, the warm-start
        # / weights-only transfer path would otherwise fail while the
        # same file loads cleanly elsewhere.
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        
        # After ``torch.compile``, ``logZ`` lives on ``_orig_mod`` rather
        # than the OptimizedModule wrapper, so the previous direct
        # ``trainer.gfn.pf_model.logZ`` access read/wrote a non-existent
        # attribute on the wrapper and silently kept the freshly-init
        # value. Resolve the inner module once and use it for every logZ
        # touch in this function.
        pf_inner = getattr(trainer.gfn.pf_model, "_orig_mod", trainer.gfn.pf_model)

        # Extract logZ BEFORE loading state dict to ensure we have it
        logZ_val = None
        if hasattr(pf_inner, 'logZ') and isinstance(checkpoint, dict):
            # Try to get from state dict first (check for torch.compile prefix)
            state_dict = checkpoint.get('pf_model_state_dict') or checkpoint.get('model_state_dict') or checkpoint.get('state_dict')
            if state_dict:
                # Check for logZ with different possible keys (torch.compile adds _orig_mod. prefix)
                logZ_keys = ['logZ', '_orig_mod.logZ', 'module.logZ']
                for key in logZ_keys:
                    if key in state_dict:
                        logZ_tensor = state_dict[key]
                        if torch.is_tensor(logZ_tensor):
                            # Handle both CPU and device tensors
                            logZ_val = logZ_tensor.item() if logZ_tensor.numel() == 1 else logZ_tensor[0].item()
                            break
                        elif isinstance(logZ_tensor, (int, float)):
                            logZ_val = float(logZ_tensor)
                            break
            
            # Fallback: try to get from metrics (use latest value if it's a list)
            if logZ_val is None and 'metrics' in checkpoint:
                metrics = checkpoint['metrics']
                if isinstance(metrics, dict) and 'logZ' in metrics:
                    logZ_metric = metrics['logZ']
                    if isinstance(logZ_metric, list) and len(logZ_metric) > 0:
                        # Get the latest logZ value from the list
                        logZ_val = float(logZ_metric[-1])
                    elif torch.is_tensor(logZ_metric):
                        logZ_val = logZ_metric.item() if logZ_metric.numel() == 1 else logZ_metric[-1].item() if logZ_metric.numel() > 0 else None
                    elif isinstance(logZ_metric, (int, float)):
                        logZ_val = float(logZ_metric)
        
        # Normalize ``_orig_mod.`` keys before loading. The target may be a
        # ``torch.compile``'d ``OptimizedModule`` (CUDA path), while
        # ``GFlowNet.save_checkpoint`` saves through the
        # inner module — without unwrap-on-target + strip-on-source, the
        # ``strict=False`` path would silently skip the policy weights and
        # report success.
        def _strip_orig_mod(sd):
            if isinstance(sd, dict) and any(
                isinstance(k, str) and k.startswith("_orig_mod.") for k in sd.keys()
            ):
                return {
                    (k.removeprefix("_orig_mod.") if isinstance(k, str) else k): v
                    for k, v in sd.items()
                }
            return sd

        def _load_into(target_module, source_sd):
            inner = getattr(target_module, "_orig_mod", target_module)
            normalized = _strip_orig_mod(source_sd)
            result = inner.load_state_dict(normalized, strict=False)
            missing = getattr(result, "missing_keys", []) or []
            unexpected = getattr(result, "unexpected_keys", []) or []
            if missing:
                logging.warning(
                    f"  load_checkpoint_weights_only: {len(missing)} missing keys "
                    f"(first 5: {missing[:5]})"
                )
            if unexpected:
                logging.warning(
                    f"  load_checkpoint_weights_only: {len(unexpected)} unexpected keys "
                    f"(first 5: {unexpected[:5]})"
                )

        # Load the state dict
        if isinstance(checkpoint, dict):
            # Handle different checkpoint formats
            if 'pf_model_state_dict' in checkpoint:
                # New checkpoint format with pf_model
                _load_into(trainer.gfn.pf_model, checkpoint['pf_model_state_dict'])
                # Also load pb_model if needed (though it's usually uniform)
                if 'pb_model_state_dict' in checkpoint:
                    _load_into(trainer.gfn.pb_model, checkpoint['pb_model_state_dict'])
            elif 'model_state_dict' in checkpoint:
                # Old checkpoint format - try to load into pf_model
                _load_into(trainer.gfn.pf_model, checkpoint['model_state_dict'])
            elif 'state_dict' in checkpoint:
                # Another old format
                _load_into(trainer.gfn.pf_model, checkpoint['state_dict'])
            else:
                # Check if checkpoint contains the state dict directly
                # This is unlikely but handle it
                logging.warning("Checkpoint format not recognized, attempting direct load")
                _load_into(trainer.gfn.pf_model, checkpoint)
        else:
            # Assume the checkpoint is the state dict itself
            _load_into(trainer.gfn.pf_model, checkpoint)
        
        # Explicitly set logZ if we found it (this ensures it's set even if
        # load_state_dict failed to load it). Touch the inner module so a
        # ``torch.compile``'d wrapper doesn't shadow the real ``logZ``.
        if hasattr(pf_inner, 'logZ') and logZ_val is not None:
            with torch.no_grad():
                pf_inner.logZ.data.fill_(logZ_val)
            logging.info(f"Transferred logZ from source checkpoint: {logZ_val:.6f}")
        elif hasattr(pf_inner, 'logZ'):
            # Check current value for debugging
            current_logZ = pf_inner.logZ.data.item()
            logging.warning(f"Could not extract logZ from checkpoint, current value: {current_logZ:.6f}")
            if isinstance(checkpoint, dict):
                logging.debug(f"Checkpoint keys: {list(checkpoint.keys())[:10]}")
                state_dict = checkpoint.get('pf_model_state_dict') or checkpoint.get('model_state_dict') or checkpoint.get('state_dict')
                if state_dict:
                    logging.debug(f"State dict has logZ: {'logZ' in state_dict}")
                    if 'logZ' in state_dict:
                        logging.debug(f"logZ type: {type(state_dict['logZ'])}")
        
        logging.info("Successfully loaded neural network weights only")
        return True
        
    except Exception as e:
        logging.error(f"Failed to load weights: {e}")
        return False


async def run_experiment(config: ExperimentConfig):
    """Main experiment runner with optional asynchronous evaluation"""
    
    # Initialize variables
    start_update = 0
    existing_results = []
    existing_metrics = defaultdict(list)
    results_dir = None
    hyperparameters = None
    load_weights_only = False
    
    # Ensure backward compatibility
    if not hasattr(config, 'n_simulations'):
        config.n_simulations = 1
    if not hasattr(config, 'resume'):
        config.resume = True
    if not hasattr(config, 'experiment_dir'):
        config.experiment_dir = None
    if not hasattr(config, 'cost_kwargs'):
        config.cost_kwargs = {}
    if not hasattr(config, 'async_eval'):
        config.async_eval = False
    if not hasattr(config, 'eval_poll_interval'):
        config.eval_poll_interval = 30
    if not hasattr(config, 'eval_process_timeout'):
        config.eval_process_timeout = 300
    if not hasattr(config, 'transfer_weights_on_depth_change'):
        config.transfer_weights_on_depth_change = False
    if not hasattr(config, 'warm_start_from'):
        config.warm_start_from = None
    if not hasattr(config, 'large_hubbard_mode'):
        config.large_hubbard_mode = False
    if not hasattr(config, 'measurement_backend'):
        config.measurement_backend = None
    if not hasattr(config, 'use_cuda_graph_policy'):
        config.use_cuda_graph_policy = False
    if not hasattr(config, 'cuda_graph_policy_max_rows'):
        config.cuda_graph_policy_max_rows = 2048
    config.large_hubbard_mode = _coerce_bool_config(
        config.large_hubbard_mode,
        "large_hubbard_mode",
    )
    config.use_cuda_graph_policy = _coerce_bool_config(
        config.use_cuda_graph_policy,
        "use_cuda_graph_policy",
    )
    try:
        config.cuda_graph_policy_max_rows = int(config.cuda_graph_policy_max_rows)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "cuda_graph_policy_max_rows must be a positive integer; "
            f"got {config.cuda_graph_policy_max_rows!r}"
        ) from exc
    if config.cuda_graph_policy_max_rows < 1:
        raise ValueError(
            "cuda_graph_policy_max_rows must be a positive integer; "
            f"got {config.cuda_graph_policy_max_rows!r}"
        )
    evaluator_mode_metadata = get_evaluator_mode_metadata(config)
    evaluation_enabled = config.eval_every is not None and config.eval_every > 0
    exact_energy_estimator_enabled = (
        evaluation_enabled
        and evaluator_mode_metadata["allows_full_state_evaluation"]
    )
    scalable_large_structural_reporting_enabled = (
        evaluation_enabled
        and not evaluator_mode_metadata["allows_full_state_evaluation"]
    )
    
    # Handle warm start (load NN weights from another experiment)
    if config.warm_start_from:
        warm_start_path = Path(config.warm_start_from)
        if not warm_start_path.is_absolute() and not config.warm_start_from.startswith('results'):
            warm_start_path = Path(config.results_dir).parent / config.warm_start_from
        
        if warm_start_path.exists():
            logging.info(f"Warm start from: {warm_start_path}")
            logging.info("Will load NN weights and logZ from this checkpoint")
            load_weights_only = True
            checkpoint_to_load = warm_start_path
        else:
            logging.warning(f"Warm start path not found: {warm_start_path}")
            logging.warning("Starting from scratch without warm start")
    
    # Check if this is a requeued SLURM job
    is_requeued_job = int(os.environ.get('SLURM_RESTART_COUNT', '0')) > 0
    
    # Handle resumption/checkpoint loading
    if config.resume or config.experiment_dir:
        experiment_to_check = None
        
        if config.experiment_dir:
            # Check specific experiment directory
            # Handle both absolute paths and paths relative to project root
            exp_dir_path = Path(config.experiment_dir)
            if exp_dir_path.is_absolute() or config.experiment_dir.startswith('results'):
                # It's a full path from project root (e.g., "results_water/angle109p0_bond0p881/experiment_...")
                experiment_to_check = exp_dir_path
            else:
                # It's just the experiment folder name within results_dir
                experiment_to_check = Path(config.results_dir) / config.experiment_dir
            
            if not experiment_to_check.exists():
                logging.info(f"Specified experiment directory {experiment_to_check} not found.")
                experiment_to_check = None
            else:
                logging.info(f"Found source experiment at {experiment_to_check}")
        else:
            # Find most recent experiment
            experiment_to_check = find_latest_experiment(config.results_dir)
        
        if experiment_to_check:
            # Load saved configuration
            with open(experiment_to_check / 'config.json', 'r') as f:
                saved_config = json.load(f)
            
            # Check compatibility levels
            nn_match, all_match, max_depth_changed = check_config_compatibility(saved_config, config)
            
            # CRITICAL: For requeued jobs, ALWAYS resume the latest experiment
            # This prevents creating new experiments when a job is preempted and restarted
            if is_requeued_job and nn_match:
                logging.info(f"\n{'='*60}")
                logging.info(f"REQUEUED JOB: Forcing resume of {experiment_to_check}")
                logging.info(f"{'='*60}\n")
                results_dir = experiment_to_check
                start_update, existing_metrics, existing_results, hyperparameters = load_experiment_state(results_dir)
                logging.info(f"Resuming from update {start_update + 1}")
                if not all_match:
                    logging.info("Note: Some config parameters differ but resuming anyway (requeued job)")
                # Also surface fused-knob
                # deltas on the requeued-resume branch — without this
                # call, operators see only "Some config parameters
                # differ" with no detail about which knobs changed.
                _log_fused_knob_resume_deltas(saved_config, config)
            elif all_match:
                # Full compatibility - resume training.
                # Fused-knob deltas are
                # logged separately via ``_log_fused_knob_resume_deltas``
                # — ``check_config_compatibility`` intentionally
                # excludes them from ``all_params`` (resume-safe perf
                # knobs), but the operator still needs to see when
                # they changed so the new sampler-metadata values in
                # ``hyperparameters.json`` are explainable.
                logging.info(f"All hyperparameters match. Resuming training from {experiment_to_check}")
                _log_fused_knob_resume_deltas(saved_config, config)
                results_dir = experiment_to_check
                start_update, existing_metrics, existing_results, hyperparameters = load_experiment_state(results_dir)
                logging.info(f"Starting from update {start_update + 1}")
                
            elif nn_match:
                # Only NN parameters match - consider loading weights
                # But check if max_depth changed and transfer is disabled
                if max_depth_changed and not config.transfer_weights_on_depth_change:
                    logging.info(f"max_depth changed ({saved_config.get('max_depth')} → {config.max_depth}) "
                                 f"and transfer_weights_on_depth_change=False")
                    logging.info("Starting completely new experiment WITHOUT weight transfer.")
                else:
                    logging.info(f"Neural network hyperparameters match. Will load weights from {experiment_to_check}")
                    logging.info("Other hyperparameters differ - starting new experiment with transferred weights")
                    load_weights_only = True
                    checkpoint_to_load = experiment_to_check
                
                # Print what parameters differ
                logging.info("\nDiffering parameters:")
                diff_params = [
                    'hamiltonian_path', 'n_measurements', 'max_depth', 'beta',
                    'lr', 'weight_decay', 'reward_type', 'cost_type',
                    'cost_kwargs', 'zero_stabilizer_cost_weights',
                    'large_hubbard_mode', 'measurement_backend',
                    'static_shape_sampling', 'cuda_graph_sampling',
                    'sampling_mode', 'use_cuda_graph_policy',
                    'cuda_graph_policy_max_rows',
                    'use_fused_metadata_kernel', 'use_fused_sampling_kernel',
                    'use_fused_mask_counts_kernel', 'use_fused_counter_rng_kernel',
                    'use_fused_partition_update_kernel',
                    'use_fused_apply_kernel',
                    'use_activation_checkpointing', 'use_uint8_state_cache',
                    'use_bf16_sampling',
                    'use_bf16_backward',
                ]
                # NOTE: ``FLOWMEAS_FLOW_DEDUP`` is intentionally absent — it is an
                # env-var-only gate, not a config field, so it never appears in
                # saved config.json files.

                # Both closures default a MISSING key to the param's ExperimentConfig
                # default, so a pre-knob config.json (no key) compares equal and does
                # not fork: False-default knobs (large_hubbard_mode, use_bf16_backward)
                # vs True-default knobs (the fused/bf16-sampling/checkpointing set).
                def _saved_diff_value(param: str) -> Any:
                    if param in {'large_hubbard_mode', 'use_bf16_backward', 'use_cuda_graph_policy', 'zero_stabilizer_cost_weights'}:  # default False
                        return _coerce_bool_config(saved_config.get(param, False), param)
                    if param in {'static_shape_sampling', 'cuda_graph_sampling'}:
                        return _coerce_optional_bool_config(saved_config.get(param), param)
                    if param == 'sampling_mode':
                        return _coerce_sampling_mode_config(
                            saved_config.get(param),
                            param,
                            warn_alias=False,
                        )
                    if param == 'cuda_graph_policy_max_rows':
                        return int(saved_config.get(param, 2048))
                    if param in {'use_fused_metadata_kernel', 'use_fused_sampling_kernel', 'use_fused_mask_counts_kernel', 'use_fused_counter_rng_kernel', 'use_fused_partition_update_kernel', 'use_fused_apply_kernel', 'use_activation_checkpointing', 'use_uint8_state_cache', 'use_bf16_sampling'}:  # default True
                        return _coerce_bool_config(saved_config.get(param, True), param)
                    return saved_config.get(param)

                def _current_diff_value(param: str) -> Any:
                    if param in {'large_hubbard_mode', 'use_bf16_backward', 'use_cuda_graph_policy', 'zero_stabilizer_cost_weights'}:  # default False
                        return _coerce_bool_config(getattr(config, param, False), param)
                    if param in {'static_shape_sampling', 'cuda_graph_sampling'}:
                        return _coerce_optional_bool_config(getattr(config, param), param)
                    if param == 'sampling_mode':
                        return _coerce_sampling_mode_config(
                            getattr(config, param),
                            param,
                            warn_alias=False,
                        )
                    if param == 'cuda_graph_policy_max_rows':
                        return int(getattr(config, param, 2048))
                    if param in {'use_fused_metadata_kernel', 'use_fused_sampling_kernel', 'use_fused_mask_counts_kernel', 'use_fused_counter_rng_kernel', 'use_fused_partition_update_kernel', 'use_fused_apply_kernel', 'use_activation_checkpointing', 'use_uint8_state_cache', 'use_bf16_sampling'}:  # default True
                        return _coerce_bool_config(getattr(config, param, True), param)
                    return getattr(config, param)

                for param in diff_params:
                    previous_value = _saved_diff_value(param)
                    current_value = _current_diff_value(param)
                    if previous_value != current_value:
                        logging.info(f"  {param}: {previous_value} → {current_value}")
            else:
                logging.info("Neural network architecture differs. Starting completely new experiment.")
                # Include model_type / model_kwargs: an arch switch can leave
                # hidden_dim/num_hidden_layers identical (e.g. clifford_mlp ->
                # packed_w_rowtoken, or a changed row_embed_dim), so naming them
                # tells the operator which field forced the fresh fork.
                logging.info(f"  Previous: hidden_dim={saved_config.get('hidden_dim')}, "
                      f"num_hidden_layers={saved_config.get('num_hidden_layers')}, "
                      f"model_type={saved_config.get('model_type')}, "
                      f"model_kwargs={saved_config.get('model_kwargs')}")
                logging.info(f"  Current: hidden_dim={config.hidden_dim}, "
                      f"num_hidden_layers={config.num_hidden_layers}, "
                      f"model_type={config.model_type}, "
                      f"model_kwargs={config.model_kwargs}")
    
    # Create new experiment directory if needed
    if results_dir is None:
        results_dir = Path(config.results_dir) / f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        results_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"Starting new experiment in {results_dir}")
        
        # Save configuration
        with open(results_dir / 'config.json', 'w') as f:
            json.dump(_config_to_dict(config), f, indent=2, cls=NumpyEncoder)
    
    # Print experiment details
    logging.info(f"\nExperiment details:")
    logging.info(f"Hamiltonian: {config.hamiltonian_path}")
    logging.info(f"Batch structure: {config.n_measurements} circuits per batch element")
    logging.info(f"Evaluation: Top {config.n_eval_top_k_batch_elements} batch elements")
    logging.info(
        f"Large Hubbard mode: "
        f"{'ENABLED' if evaluator_mode_metadata['large_hubbard_mode'] else 'DISABLED'}"
    )
    logging.info(f"Configured evaluator mode: {evaluator_mode_metadata['mode']}")
    if not evaluation_enabled:
        logging.info("Evaluator scheduling: DISABLED (eval_every <= 0)")
    elif exact_energy_estimator_enabled:
        logging.info("Exact EnergyEstimator evaluation: ENABLED")
    else:
        logging.info(
            "Scalable-large structural reporting: ENABLED "
            "(no EnergyEstimator or state-vector paths)"
        )
    if config.n_simulations > 1:
        logging.info(f"Simulations: {config.n_simulations} runs per batch element")
    if config.cost_kwargs:
        logging.info(f"Cost function: {config.cost_type} with kwargs: {config.cost_kwargs}")
    if config.zero_stabilizer_cost_weights:
        logging.info(
            "Stabilizer cost-weight zeroing: ENABLED "
            "(stabilizer penalty terms get weight 0 before normalization)"
        )
    if config.async_eval:
        logging.info(f"Asynchronous evaluation: ENABLED")
    if start_update > 0:
        logging.info(f"Resuming from update: {start_update + 1}/{config.n_updates}")
    elif load_weights_only:
        logging.info(f"Starting fresh training with weights from: {checkpoint_to_load.name}")
    
    # Load Hamiltonian
    hamiltonian_helper = PauliHamiltonianHelper(config.hamiltonian_path)
    logging.info(f"Hamiltonian: {hamiltonian_helper}")
    if evaluator_mode_metadata["allows_full_state_evaluation"]:
        # Defense-in-depth size guard before the log-line access.
        assert_full_state_eval_safe(
            hamiltonian_helper, config,
            "run_experiment.exact_ground_state_energy_logline",
        )
        logging.info(f"Exact ground state energy: {hamiltonian_helper.ground_state_energy:.10f}")
    else:
        logging.info(
            "Exact ground state energy: not computed "
            "(scalable_large mode disallows full-state evaluation)"
        )
    
    # Save Hamiltonian info (only for new experiments)
    if start_update == 0:
        with open(results_dir / 'hamiltonian_info.json', 'w') as f:
            hamiltonian_summary = (
                hamiltonian_helper.summary()
                if evaluator_mode_metadata["allows_full_state_evaluation"]
                else create_large_mode_hamiltonian_summary(
                    hamiltonian_helper,
                    evaluator_mode_metadata,
                )
            )
            hamiltonian_summary.update({
                "evaluator_mode": evaluator_mode_metadata["mode"],
                "large_hubbard_mode": evaluator_mode_metadata["large_hubbard_mode"],
                "full_state_evaluation_allowed": evaluator_mode_metadata[
                    "allows_full_state_evaluation"
                ],
            })
            json.dump(hamiltonian_summary, f, indent=2, cls=NumpyEncoder)

    # Filter out identity terms for training
    identity_term = "I" * hamiltonian_helper.n_qubits
    training_pauli_strings = []
    training_weights = []
    identity_weight = 0.0
    
    for pauli_str, weight in zip(hamiltonian_helper.pauli_str_list, hamiltonian_helper.w_list):
        if pauli_str == identity_term:
            identity_weight = weight.real
        else:
            training_pauli_strings.append(pauli_str)
            training_weights.append(weight.real)
    
    # Create or refresh comprehensive hyperparameters. Resumed experiments may
    # carry stale evaluator-mode fields from older runs, so refresh the
    # mode-sensitive sections from the current config before reporting.
    current_hyperparameters = create_hyperparameters_dict(
        config,
        hamiltonian_helper,
        training_pauli_strings,
        identity_weight,
    )
    if hyperparameters is None:
        hyperparameters = current_hyperparameters
        with open(results_dir / 'hyperparameters.json', 'w') as f:
            json.dump(hyperparameters, f, indent=2, cls=NumpyEncoder)
    elif refresh_evaluator_mode_hyperparameters(
        hyperparameters,
        current_hyperparameters,
    ):
        logging.info("Refreshed evaluator-mode metadata in hyperparameters.json")
        with open(results_dir / 'hyperparameters.json', 'w') as f:
            json.dump(hyperparameters, f, indent=2, cls=NumpyEncoder)
    
    # Initialize disk-backed stores for metrics and results
    result_store = DiskBackedResultStore(results_dir / 'evaluation_results.jsonl')
    metric_store = DiskBackedMetricStore(results_dir / 'metrics_history.jsonl')

    if existing_metrics and metric_store.is_empty():
        metric_store.replace(existing_metrics)

    # Initialize reporter
    reporter = AsyncReporter(results_dir, hyperparameters, result_store)
    if existing_results and result_store.is_empty():
        reporter.add_results(existing_results)
        existing_results = []
    
    # Create GFN configuration
    logging.info(f"Training with {len(training_pauli_strings)} non-identity Pauli terms")
    if identity_weight != 0:
        logging.info(f"Note: Identity term contributes a constant energy offset of {identity_weight:.6f}")
    
    # Resolve the stabilizer zero-weight mask up front (fail-fast): a bad or
    # missing metadata.json should kill the run at startup, not inside a
    # sampler worker. detect_stabilizer_terms raises RuntimeError with a
    # precise message when the Hamiltonian lacks a stabilizer_penalty block.
    stabilizer_zero_mask = None
    if config.zero_stabilizer_cost_weights:
        stabilizer_zero_mask = detect_stabilizer_terms(
            training_pauli_strings,
            training_weights,
            config.hamiltonian_path,
        )
        logging.info(
            f"Stabilizer cost-weight zeroing: detected {sum(stabilizer_zero_mask)} "
            f"stabilizer penalty terms out of {len(training_pauli_strings)} training terms"
        )

    gfn_config = {
        "model": {
            "model_type": config.model_type,
            "hidden_dim": config.hidden_dim,
            "num_hidden_layers": config.num_hidden_layers,
            "lr": config.lr,
            "weight_decay": config.weight_decay,
            "model_dir": str(results_dir),
            "model_kwargs": dict(config.model_kwargs),
            "objective_type": config.objective_type,
            "objective_kwargs": config.objective_kwargs,
            "measurement_backend": config.measurement_backend,
            "static_shape_sampling": config.static_shape_sampling,
            "cuda_graph_sampling": config.cuda_graph_sampling,
            "sampling_mode": config.sampling_mode,
            "use_cuda_graph_policy": config.use_cuda_graph_policy,
            "cuda_graph_policy_max_rows": config.cuda_graph_policy_max_rows,
            "use_fused_metadata_kernel": config.use_fused_metadata_kernel,
            "use_fused_sampling_kernel": config.use_fused_sampling_kernel,
            "use_fused_mask_counts_kernel": config.use_fused_mask_counts_kernel,
            "use_fused_counter_rng_kernel": config.use_fused_counter_rng_kernel,
            "use_fused_partition_update_kernel": config.use_fused_partition_update_kernel,
            "use_fused_apply_kernel": config.use_fused_apply_kernel,
            "use_bf16_sampling": config.use_bf16_sampling,
            "use_bf16_backward": config.use_bf16_backward,
            "use_activation_checkpointing": config.use_activation_checkpointing,
            "use_uint8_state_cache": config.use_uint8_state_cache,
            "debug": False
        },
        "training": {
            "beta": config.beta,
            "n_measurements": config.n_measurements,
            "update_freq": config.update_freq,
            "max_depth": config.max_depth,
            "K": config.n_eval_top_k_batch_elements,
            "reward_kwargs": config.reward_kwargs,
            "cost": {
                "type": config.cost_type,
                **config.cost_kwargs  # Unpack cost_kwargs into the cost config
            }
        },
        "quantum": {
            "pauli_str_list": training_pauli_strings,
            "w_list": training_weights,
            # ``stabilizer_zero_mask`` is resolved HERE (fail-fast at startup, before
            # any worker spawns) so trainer/async construction sites only forward an
            # already-validated mask. None when the knob is off.
            "stabilizer_zero_mask": stabilizer_zero_mask,
        }
    }
    
    # Create trainer
    device = get_device(config.device_preference)
    
    # Select reward function
    reward_fn_map = {
        "exp": exponential_reward_fn,
        "default": default_reward_fn,
        "log": log_reward_fn,
    }
    reward_fn = reward_fn_map.get(config.reward_type)
    
    trainer = EfficientGFNTrainer(
        gfn_config,
        reward_fn=reward_fn,
        device_preference=config.device_preference,
        metric_store=metric_store,
    )
    
    # Initialize the trainer's update trackers *before* publishing it to
    # the signal handler. Without this, a preemption arriving in the
    # window between ``_current_trainer = trainer`` and the loop's first
    # ``trainer.current_update = start_update`` assignment would write an
    # emergency checkpoint tagged with update=0 — overwriting any valid
    # resumed checkpoint at ``start_update > 0``.
    trainer.current_update = int(start_update or 0)
    trainer.completed_updates = int(start_update or 0)

    # Set global reference for signal handlers (preemption checkpoint saving)
    global _current_trainer, _current_results_dir, _loop_finalized, _final_checkpoint_persisted
    _current_trainer = trainer
    _current_results_dir = results_dir
    # Fresh ``run_experiment`` call — clear the "loop already finished"
    # latches from any prior in-process run (e.g., a test that re-enters
    # ``run_experiment``).: ``_final_checkpoint_persisted``
    # is the latch the handler reads to decide whether post-loop
    # no-requeue is safe (i.e., final training progress is durable).
    _loop_finalized = False
    _final_checkpoint_persisted = False
    
    # Handle checkpoint loading based on compatibility
    if start_update > 0:
        # Full resume - load complete checkpoint
        checkpoint_files = list(results_dir.glob('checkpoint_update*.pth'))
        if checkpoint_files:
            latest_checkpoint = _pick_best_checkpoint(checkpoint_files)
            logging.info(f"Loading full checkpoint from update {start_update}")
            try:
                trainer.gfn.load_checkpoint(str(latest_checkpoint))
                trainer.ingest_metrics(existing_metrics)
            except Exception as e:
                logging.warning(f"Failed to load checkpoint: {e}")
                # Try manual loading.: match
                # ``_pick_best_checkpoint`` / ``load_experiment_state``
                # by passing ``weights_only=False`` explicitly so this
                # fallback can't fail on PyTorch versions where the
                # default differs.
                checkpoint = torch.load(
                    latest_checkpoint, map_location=trainer.device, weights_only=False
                )
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    trainer.gfn.model.load_state_dict(checkpoint['model_state_dict'])
                    if 'optimizer_state_dict' in checkpoint:
                        trainer.gfn.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    trainer.ingest_metrics(existing_metrics)
                    
    elif load_weights_only:
        # Load only neural network weights
        checkpoint_files = list(checkpoint_to_load.glob('checkpoint_update*.pth'))
        if checkpoint_files:
            latest_checkpoint = _pick_best_checkpoint(checkpoint_files)
            success = load_checkpoint_weights_only(trainer, latest_checkpoint, trainer.device)
            if success:
                # Note the transfer in hyperparameters
                if 'training' not in hyperparameters:
                    hyperparameters['training'] = {}
                hyperparameters['training']['weights_transferred_from'] = str(checkpoint_to_load.name)
                hyperparameters['training']['transfer_checkpoint'] = str(latest_checkpoint.name)
                
                # Update hyperparameters file
                with open(results_dir / 'hyperparameters.json', 'w') as f:
                    json.dump(hyperparameters, f, indent=2, cls=NumpyEncoder)
    
    # Save training-specific hyperparameters (only for new experiments)
    if start_update == 0:
        training_hyperparams = {
            "actual_device": str(trainer.device),
            "num_actions": trainer.gfn.num_actions,
            "state_dim": trainer.gfn.state_dim,
            "action_mapping_size": len(trainer.gfn.action_mapping),
            "gfn_config": gfn_config,
            "weights_loaded_from": str(checkpoint_to_load.name) if load_weights_only else None,
            "energy_estimator": (
                "EnergyEstimator with batched Clifford map"
                if exact_energy_estimator_enabled
                else (
                    "scalable_large structural reporter"
                    if scalable_large_structural_reporting_enabled
                    else "disabled"
                )
            ),
            "async_eval_enabled": config.async_eval and evaluation_enabled,
            "configured_async_eval": config.async_eval,
            "large_hubbard_mode": evaluator_mode_metadata["large_hubbard_mode"],
            "evaluator_mode": evaluator_mode_metadata["mode"],
            "exact_energy_estimator_enabled": exact_energy_estimator_enabled,
            "scalable_large_structural_reporting_enabled": scalable_large_structural_reporting_enabled,
            "full_state_evaluation_allowed": evaluator_mode_metadata[
                "allows_full_state_evaluation"
            ],
            "measurement_backend": config.measurement_backend,
        }
        
        with open(results_dir / 'training_hyperparameters.json', 'w') as f:
            json.dump(training_hyperparams, f, indent=2, cls=NumpyEncoder)
    
    async def check_evaluation_results():
        """No-op when evaluation is disabled."""
        return False

    # Initialize components for async evaluation
    if config.async_eval and evaluation_enabled:
        checkpoint_queue = mp.Queue()
        results_queue = mp.Queue()
        
        # Start evaluator process
        evaluator_process = mp.Process(
            target=evaluator_loop,
            args=(config, results_dir, hamiltonian_helper, checkpoint_queue, results_queue)
        )
        evaluator_process.start()
        logging.info("Started asynchronous evaluator process")
        
        # Sticky parent-side flag that records whether the
        # async evaluator ever reported a fatal error (either via
        # ``EVALUATOR_RESULT_TYPE_ERROR`` sentinel or a non-zero exit
        # code at final cleanup). Final report consults this and refuses
        # to label the run a clean success when set.
        evaluator_failed = {"value": False, "reasons": []}

        # Create function to handle results from evaluator
        async def check_evaluation_results():
            """Check for results from evaluator process and update visualizations"""
            results_received = False
            plots_updated = False
            while True:
                try:
                    msg_type, update, results = results_queue.get_nowait()
                    if msg_type == EVALUATOR_RESULT_TYPE_EXACT:
                        logging.info(f"Received evaluation results for update {update}")
                        # Persist and update visualizations immediately
                        reporter.add_results(results)
                        await reporter.update_summary_async()
                        reporter.export_results_json(results_dir / 'evaluation_results.json')
                        
                        logging.info(f"Updated plots and saved results for update {update}")
                        results_received = True
                        plots_updated = True
                    elif msg_type == EVALUATOR_RESULT_TYPE_SCALABLE_LARGE_REPORT:
                        logging.info(
                            "Received scalable-large structural report for "
                            f"update {update}: "
                            f"{results.get('n_batch_elements', 0)} batch elements, "
                            f"{results.get('n_circuits_total', 0)} circuits"
                        )
                        results_received = True
                    elif msg_type == EVALUATOR_RESULT_TYPE_ERROR:
                        # Surface evaluator failure to operators so the
                        # run isn't reported successful with stale/missing
                        # eval artifacts.: flip the sticky
                        # ``evaluator_failed`` flag so the final report /
                        # post-loop logic can refuse to label the run a
                        # clean success.
                        logging.error(
                            f"Async evaluator reported a fatal error: {results}. "
                            f"Final evaluation artifacts may be missing or stale."
                        )
                        evaluator_failed["value"] = True
                        evaluator_failed["reasons"].append(f"sentinel at update {update}: {results}")
                        results_received = True
                except queue.Empty:
                    break
                except (EOFError, BrokenPipeError, OSError) as q_err:
                    # Catch pipe-side queue errors so they cannot break the
                    # outer drain-while-join silently: log, let the outer loop
                    # retry, and flip the sticky ``evaluator_failed`` flag
                    # because what the evaluator was sending is lost.
                    logging.warning(
                        f"results_queue.get_nowait errored ({type(q_err).__name__}): "
                        f"{q_err}. Aborting this drain pass."
                    )
                    evaluator_failed["value"] = True
                    evaluator_failed["reasons"].append(
                        f"results_queue I/O error: {type(q_err).__name__}: {q_err}"
                    )
                    break
                except Exception as q_err:
                    # Unexpected (e.g., pickle error on a malformed
                    # result). Log loudly, flag, and exit the drain pass.
                    logging.error(
                        f"results_queue.get_nowait unexpected error: "
                        f"{type(q_err).__name__}: {q_err}"
                    )
                    evaluator_failed["value"] = True
                    evaluator_failed["reasons"].append(
                        f"results_queue unexpected: {type(q_err).__name__}: {q_err}"
                    )
                    break
            
            if plots_updated:
                logging.info("Visualization plots have been updated")
            elif results_received:
                logging.info("Evaluation artifacts have been updated")
            
            return results_received
    else:
        # Initialize synchronous energy estimator only for exact-small evaluation.
        energy_estimator = None
        if exact_energy_estimator_enabled:
            energy_estimator = EnergyEstimator(
                hamiltonian_helper,
                hamiltonian_helper.n_qubits,
                device,
                measurement_backend=config.measurement_backend,
            )
        checkpoint_queue = None
        results_queue = None
        evaluator_process = None

    # Register the evaluator on a module-level cleanup
    # registry so any unhandled exception out of the training loop still
    # tears it down. ``_register_evaluator_for_cleanup`` is idempotent
    # and the un-registration happens at the natural cleanup point
    # below; in between, the run_config wrapper's broad except (or any
    # SystemExit raised by the safe-point branch) can invoke
    # ``_drain_registered_evaluators`` to guarantee no leak.
    if evaluator_process is not None:
        _register_evaluator_for_cleanup(evaluator_process, checkpoint_queue)

    # Store all evaluation results
    logging.info(f"\n=== {'Resuming' if start_update > 0 else 'Starting'} Training ===")
    if load_weights_only:
        logging.info("Note: Using transferred neural network weights")
    
    # Pre-compute cost_kwargs for compute_costs_with_probabilities calls
    # Filter out normalization_type as it's only used at CostComputer init
    cost_compute_kwargs = {k: v for k, v in config.cost_kwargs.items() 
                          if k != "normalization_type"}
    
    # Training loop
    for update in range(start_update, config.n_updates):
        # Mirror the loop counter into ``trainer.current_update`` so the
        # SIGUSR1 preempt handler in run_config.py can read the real
        # update number — deriving from ``len(metrics_history['loss'])``
        # would cap at ``metrics_window`` (512 default) and write the
        # wrong offset to ``checkpoint_update.pth`` on long runs.
        trainer.current_update = update

        update_start = time.time()

        # Sample trajectories (batch elements)
        trajectory_batch = trainer.gfn.sample_trajectories(
            batch_size=config.update_freq,
            n_measurements=config.n_measurements,
            max_depth=config.max_depth,
            mode=SamplingMode.ON_POLICY
        )
        
        # Compute costs for each batch element with cost_kwargs
        costs = trainer.compute_costs_with_probabilities(
            trajectory_batch.batched_tableau, 
            **cost_compute_kwargs
        )
        
        # Compute loss and update. ``metrics_to_cpu=False`` keeps metrics
        # on GPU; ``return_tensor=True`` returns the loss tensor without
        # the.item() that ``update_step`` would otherwise sync on. This
        # matches the hot-path-optimised trainer in ``EfficientGFNTrainer.train``
        # — without these two, every update in the production runner
        # forced a host sync that undercut 's hot-path work
        #. The existing periodic.item() conversions
        # below remain the safe sync boundary.
        loss, metrics = trainer.gfn.compute_loss(
            trajectory_batch, costs, config.beta, max_depth=config.max_depth,
            metrics_to_cpu=False,
            **config.reward_kwargs,
        )
        trainer.gfn.update_step(loss, return_tensor=True)

        # Update top trajectories (batch elements) for replay
        trainer.gfn._update_top_trajectories(trajectory_batch, costs)
        
        # Store metrics - KEEP AS GPU TENSORS to avoid synchronization
        # Only convert to CPU when actually needed (logging/checkpointing)
        for k, v in metrics.items():
            if torch.is_tensor(v):
                # Detach from computation graph but keep on GPU
                v_detached = v.detach()
                metrics[k] = v_detached
                trainer.metrics_history[k].append(v_detached)
            else:
                trainer.metrics_history[k].append(v)

            # Trim history window
            if trainer.metrics_window and len(trainer.metrics_history[k]) > trainer.metrics_window + _METRICS_TRIM_SLACK:
                del trainer.metrics_history[k][:-trainer.metrics_window]

        # Print progress - ONLY sync when logging (every 100 updates)
        if (update + 1) % 100 == 0:
            # Single synchronization point for logging
            loss_val = metrics['loss'].item() if torch.is_tensor(metrics['loss']) else metrics['loss']
            reward_val = metrics['reward'].item() if torch.is_tensor(metrics['reward']) else metrics['reward']
            cost_val = metrics['cost'].item() if torch.is_tensor(metrics['cost']) else metrics['cost']
            logging.info(f"Update {update + 1}/{config.n_updates}: "
                  f"Loss={loss_val:.6f}, "
                  f"Reward={reward_val:.4f}, "
                  f"Cost={cost_val:.4f}")
        
        # Replay training
        if config.replay_every and (update + 1) % config.replay_every == 0 and trainer.gfn.top_trajectories_actions:
            replay_batch = trainer.gfn.sample_trajectories(
                batch_size=min(len(trainer.gfn.top_trajectories_actions), config.update_freq),
                n_measurements=config.n_measurements,
                max_depth=config.max_depth,
                mode=SamplingMode.REPLAY
            )
            logging.info(f"At step {update + 1}, performing replay training:")
            replay_costs = trainer.compute_costs_with_probabilities(
                replay_batch.batched_tableau,
                # ``silence=False`` logged a CUDA tensor (``probs.sum``) plus a
                # ``.cpu()`` costs line on every replay update — a forced host
                # sync every ``replay_every`` steps spent purely on log lines.
                # Keep the verbose replay dump under explicit GFN debug or a
                # DEBUG logging level; the marker above stays either way.
                silence=not _verbose_replay_logging_enabled(trainer.gfn),
                **cost_compute_kwargs  # Pass filtered cost_kwargs (no normalization_type)
            )
            replay_loss, replay_metrics = trainer.gfn.compute_loss(
                replay_batch, replay_costs, config.beta, max_depth=config.max_depth, **config.reward_kwargs
            )
            trainer.gfn.update_step(replay_loss)

            # Store replay metrics - keep as GPU tensors
            for k, v in replay_metrics.items():
                if torch.is_tensor(v):
                    v_detached = v.detach()
                    replay_metrics[k] = v_detached
                    trainer.metrics_history[f'replay_{k}'].append(v_detached)
                else:
                    trainer.metrics_history[f'replay_{k}'].append(v)

                # Trim history
                hist = trainer.metrics_history[f'replay_{k}']
                if trainer.metrics_window and len(hist) > trainer.metrics_window + _METRICS_TRIM_SLACK:
                    del hist[:-trainer.metrics_window]

        # Off-policy training
        if config.offpolicy_every and (update + 1) % config.offpolicy_every == 0:
            offpolicy_batch = trainer.gfn.sample_trajectories(
                batch_size=config.update_freq,
                n_measurements=config.n_measurements,
                max_depth=config.max_depth,
                mode=SamplingMode.OFF_POLICY
            )

            offpolicy_costs = trainer.compute_costs_with_probabilities(
                offpolicy_batch.batched_tableau,
                **cost_compute_kwargs  # Pass filtered cost_kwargs (no normalization_type)
            )
            offpolicy_loss, offpolicy_metrics = trainer.gfn.compute_loss(
                offpolicy_batch, offpolicy_costs, config.beta, max_depth=config.max_depth, **config.reward_kwargs
            )
            trainer.gfn.update_step(offpolicy_loss)

            # Store off-policy metrics - keep as GPU tensors
            for k, v in offpolicy_metrics.items():
                if torch.is_tensor(v):
                    v_detached = v.detach()
                    offpolicy_metrics[k] = v_detached
                    trainer.metrics_history[f'offpolicy_{k}'].append(v_detached)
                else:
                    trainer.metrics_history[f'offpolicy_{k}'].append(v)

                # Trim history
                hist = trainer.metrics_history[f'offpolicy_{k}']
                if trainer.metrics_window and len(hist) > trainer.metrics_window + _METRICS_TRIM_SLACK:
                    del hist[:-trainer.metrics_window]

        # Buffer metrics every iteration (stays on GPU, no transfer overhead)
        if metric_store is not None:
            # Create a shallow copy of metrics dict for buffering
            # This keeps GPU tensors on GPU without transfer
            metrics_to_buffer = dict(metrics)

            # Include replay/offpolicy if they were computed this update
            if config.replay_every and (update + 1) % config.replay_every == 0 and trainer.gfn.top_trajectories_actions:
                for k, v in replay_metrics.items():
                    metrics_to_buffer[f'replay_{k}'] = v
            if config.offpolicy_every and (update + 1) % config.offpolicy_every == 0:
                for k, v in offpolicy_metrics.items():
                    metrics_to_buffer[f'offpolicy_{k}'] = v

            # Add to buffer without GPU-CPU transfer
            metric_store.append_to_buffer(update + 1, metrics_to_buffer)

        # Flush buffered metrics to disk at checkpoint intervals
        # This is when GPU-CPU transfer and disk I/O happen
        if (update + 1) % config.checkpoint_every == 0 and metric_store is not None:
            num_flushed = metric_store.flush_buffer()
            if num_flushed:
                logging.info(f"Flushed {num_flushed} iterations of metrics to disk")
        
        # Checkpoint saving. The ``update + 1`` argument MUST equal the
        # iteration's completed update number; ``update`` is the live loop
        # variable, so this is correct by construction. The refuse-stale guard at
        # the safe-point save sites is the second line of defence against a
        # non-local source for the update number creeping in here.
        if (update + 1) % config.checkpoint_every == 0:
            checkpoint_path = results_dir / f'checkpoint_update.pth'
            # Convert metrics history from GPU tensors to CPU for saving
            metrics_history_cpu = convert_metrics_history_to_cpu(trainer.metrics_history)
            trainer.gfn.save_checkpoint(str(checkpoint_path), update + 1, metrics_history_cpu)
            
            # Only queue for evaluation if it's also an evaluation update
            if (
                evaluation_enabled
                and config.async_eval
                and checkpoint_queue
                and (update + 1) % config.eval_every == 0
            ):
                checkpoint_data = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
                checkpoint_id = checkpoint_data.get('checkpoint_id', time.time())
                checkpoint_queue.put((str(checkpoint_path), update + 1, checkpoint_id))
                logging.info(f"Checkpoint saved and queued for evaluation at update {update + 1}")
            else:
                logging.info(f"Checkpoint saved at update {update + 1}")
        
        # Evaluation - handle both sync and async modes
        if evaluation_enabled and (update + 1) % config.eval_every == 0:
            if config.async_eval:
                # For async mode, ensure we have a checkpoint to evaluate
                # If checkpoint_every doesn't align with eval_every, save a special checkpoint
                if (update + 1) % config.checkpoint_every != 0:
                    checkpoint_path = results_dir / f'checkpoint_update.pth'
                    # Convert metrics history from GPU tensors to CPU for saving
                    metrics_history_cpu = convert_metrics_history_to_cpu(trainer.metrics_history)
                    trainer.gfn.save_checkpoint(str(checkpoint_path), update + 1, metrics_history_cpu)
                    
                    checkpoint_data = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
                    checkpoint_id = checkpoint_data.get('checkpoint_id', time.time())
                    checkpoint_queue.put((str(checkpoint_path), update + 1, checkpoint_id))
                    logging.info(f"Special evaluation checkpoint saved and queued at update {update + 1}")
                
                # Check for any completed evaluation results
                await check_evaluation_results()
                logging.info(f"\nAsync evaluation in progress for update {update + 1}")
            else:
                # Synchronous evaluation
                logging.info(f"\n{'='*60}")
                logging.info(f"Checkpoint evaluation at update {update + 1}")
                
                # Evaluate top batch elements
                evaluation_results = await evaluate_top_batch_elements(
                    trainer, energy_estimator, update + 1, config
                )
                
                if (
                    evaluation_results
                    and evaluator_mode_metadata["allows_full_state_evaluation"]
                ):
                    reporter.add_results(evaluation_results)
                    await reporter.update_summary_async()
                    reporter.export_results_json(results_dir / 'evaluation_results.json')
                elif evaluation_results:
                    report_path = save_scalable_large_report_safely(
                        evaluation_results,
                        results_dir,
                    )
                    logging.info(
                        "Saved scalable-large structural report to: "
                        f"{report_path}"
                    )

                logging.info(f"{'='*60}\n")
        
        # In async mode, periodically check for evaluation results
        # This ensures visualizations are updated as soon as results are available
        if evaluation_enabled and config.async_eval and (update + 1) % 100 == 0:  # Check every 100 updates
            await check_evaluation_results()

        # Bump the *completed* count only after the entire iteration body
        # (incl. periodic checkpoint save + evaluation) is done. The
        # signal handler uses this for emergency saves so an interrupt
        # mid-iteration doesn't tag advanced weights with the previous
        # update number.
        trainer.completed_updates = update + 1

        # Safe-point handoff for SIGUSR1/SIGTERM. Both flags live in this
        # module so the read works regardless of how the entrypoint was
        # launched (``python3 -m code.run_config`` vs ``python3 code/run_config.py``).
        # We explicitly tear down the async evaluator BEFORE
        # ``sys.exit`` so the non-daemon child doesn't leak past interpreter
        # shutdown.
        if _shutdown_requested:
            is_warning = bool(_shutdown_is_warning_signal)
            emergency_path = _safe_point_checkpoint_path(
                results_dir, trainer.completed_updates, is_warning
            )
            logging.warning(
                f"Shutdown flag set after update {trainer.completed_updates}; "
                f"saving safe-point checkpoint to {emergency_path} and exiting "
                f"({'requeue' if is_warning else 'no-requeue'})."
            )
            # Refuse-stale guard: defence-in-depth against ``completed_updates``
            # decoupling from live training state. ``_pick_best_checkpoint``
            # already mitigates on resume by ranking on payload ``update``;
            # logging at the write site leaves a trail for a post-mortem.
            #
            # Two distinct boolean states, so the downstream exit-code branches
            # read true to their semantics:
            #   - ``safe_to_proceed`` — either we wrote a fresh emergency
            #     checkpoint OR the on-disk state was already advanced
            #     enough that no write was needed. The shutdown path
            #     (including ``scontrol requeue`` for SIGUSR1) is safe.
            #   - ``write_succeeded`` — we actually wrote a new file.
            #     Logged for post-mortem clarity; not a control-flow input.
            safe_to_proceed = False
            write_succeeded = False
            try:
                # Use the filename-hint fast path so the guard stays inside the
                # SIGUSR1 grace budget even with many emergency files present.
                # Hints written by ``safe_point_checkpoint_path`` faithfully
                # encode the payload update; only canonical
                # ``checkpoint_update.pth`` always reads 0. Passing the proposed
                # ``completed_updates`` makes the helper verify any high-hint
                # file that would actually block the write, so a corrupt or
                # payload-stale one cannot poison the guard.
                proposed_update = int(trainer.completed_updates or 0)
                existing_update = highest_existing_checkpoint_filename_hint(
                    results_dir, proposed_update=proposed_update
                )
                if existing_update > proposed_update:
                    logging.warning(
                        f"Refusing safe-point write: an existing verified "
                        f"checkpoint in {results_dir} reports update="
                        f"{existing_update}, ahead of in-memory completed_updates="
                        f"{proposed_update}. Skipping write to avoid the "
                        f"rollback regression. Resume will still pick "
                        f"the on-disk advanced checkpoint; requeue is safe."
                    )
                    safe_to_proceed = True  # On-disk state wins; nothing to write.
                else:
                    metrics_history_cpu = convert_metrics_history_to_cpu(trainer.metrics_history)
                    trainer.gfn.save_checkpoint(
                        str(emergency_path), trainer.completed_updates, metrics_history_cpu
                    )
                    write_succeeded = True
                    safe_to_proceed = True
            except Exception as save_err:
                logging.error(
                    f"Safe-point checkpoint save failed: {save_err}. "
                    f"Will skip scontrol requeue to prevent restarting from "
                    f"an older checkpoint with progress lost."
                )

            # Flush buffered metrics before the safe
            # exit so the sidecar JSONL stays in sync with the emergency
            # checkpoint we just wrote. Without this, model weights would
            # be preserved while the metrics buffer accumulated since the
            # last periodic flush is dropped. Treat a flush failure as
            # best-effort (the model state is the load-bearing artifact);
            # log but don't escalate to ``sys.exit(1)`` since the
            # checkpoint itself wrote successfully.
            if metric_store is not None:
                try:
                    num_flushed = metric_store.flush_buffer()
                    if num_flushed:
                        logging.info(
                            f"Flushed {num_flushed} buffered metrics before safe-point exit"
                        )
                except Exception as flush_err:
                    logging.warning(
                        f"metric_store flush failed on safe-point exit: {flush_err}. "
                        f"Sidecar JSONL may lag the emergency checkpoint."
                    )

            # Tear down the async evaluator cleanly before exiting.
            # ``checkpoint_queue.put('STOP')`` is the polite signal; we
            # join with a bounded timeout and fall back to terminate/kill
            # so we don't hang the requeue path on a stuck evaluator.
            try:
                if (
                    config.async_eval
                    and 'evaluator_process' in locals()
                    and evaluator_process is not None
                    and evaluator_process.is_alive()
                    and checkpoint_queue is not None
                ):
                    logging.info("Sending STOP to async evaluator before safe-point exit")
                    try:
                        checkpoint_queue.put('STOP')
                    except Exception:
                        pass
                    evaluator_process.join(timeout=10.0)
                    if evaluator_process.is_alive():
                        logging.warning("Evaluator didn't STOP in 10s; terminating")
                        evaluator_process.terminate()
                        evaluator_process.join(timeout=5.0)
                        if evaluator_process.is_alive():
                            evaluator_process.kill()
            except Exception as cleanup_err:
                logging.warning(f"Async evaluator cleanup errored on safe-point exit: {cleanup_err}")

            # Drain any other registered evaluators too (idempotent if the
            # inline cleanup above already terminated this one).
            drain_registered_evaluators()

            # Flip the
            # post-loop latches BEFORE the save-failure ``sys.exit(1)``
            # branch as well as before the normal exit. Defensive
            # consistency: the "loop is done from the runtime's POV"
            # contract holds for *any* exit path out of the safe-point
            # block. ``_final_checkpoint_persisted`` is False on save
            # failure (the on-disk state was not advanced AND the new
            # write didn't succeed), True when ``safe_to_proceed`` was
            # confirmed (write succeeded OR refuse-stale found on-disk
            # ahead).
            _loop_finalized = True
            _final_checkpoint_persisted = bool(safe_to_proceed)
            _current_trainer = None

            # If the safe-point save failed AND the
            # on-disk state was not already advanced, do NOT take the
            # requeue path — restarting from an older checkpoint would
            # silently roll progress back. Force exit 1 so slurm marks
            # the job failed, and skip ``complete_shutdown_after_safe_save``
            # (which would have called ``scontrol requeue``).
            #
            # ``safe_to_proceed`` is the
            # requeue-eligibility predicate. It's True both when the
            # write succeeded and when the refuse-stale branch found
            # on-disk state ahead. ``write_succeeded`` is preserved for
            # post-mortem logs but not part of the requeue decision.
            if not safe_to_proceed:
                logging.error(
                    "Safe-point save failed and no on-disk checkpoint was ahead; "
                    "exiting non-99 / no-requeue so the job is marked failed "
                    "instead of restarting at an older checkpoint. Investigate "
                    "disk / I/O state."
                )
                sys.exit(1)

            exit_code = complete_shutdown_after_safe_save(is_warning)
            logging.info(
                f"Safe-point save + evaluator cleanup complete "
                f"(write_succeeded={write_succeeded}). Exiting (code {exit_code})."
            )
            sys.exit(exit_code)

    # Training complete - handle cleanup.
    #
    # Assignment ORDER matters: set ``_loop_finalized = True`` FIRST, then clear
    # ``_current_trainer``. Signal handlers run between bytecode boundaries, so
    # the reverse order leaves a window where a signal sees
    # ``_current_trainer is None`` and ``_loop_finalized is False``, takes the
    # pre-init branch, and wrongly calls ``scontrol requeue``.
    #
    # ``_loop_finalized`` only marks "training loop returned";
    # ``_final_checkpoint_persisted`` (flipped after the final canonical write)
    # is what authorizes the no-requeue post-loop path. Before that latch a
    # post-loop SIGUSR1 still requeues — losing cluster time beats terminating
    # before the final checkpoint reaches disk.
    _loop_finalized = True
    _current_trainer = None

    # Flush any remaining buffered metrics to disk
    if metric_store is not None:
        num_remaining = metric_store.flush_buffer()
        if num_remaining:
            logging.info(f"Flushed {num_remaining} remaining metrics at end of training")

    # Final evaluation (if not already done)
    if evaluation_enabled and config.n_updates % config.eval_every != 0:
        if config.async_eval:
            # Save final checkpoint for evaluation (convert GPU tensors to CPU first)
            checkpoint_path = results_dir / f'checkpoint_update.pth'
            metrics_history_cpu = convert_metrics_history_to_cpu(trainer.metrics_history)
            trainer.gfn.save_checkpoint(str(checkpoint_path), config.n_updates, metrics_history_cpu)

            if checkpoint_queue:
                checkpoint_data = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
                checkpoint_id = checkpoint_data.get('checkpoint_id', time.time())
                checkpoint_queue.put((str(checkpoint_path), config.n_updates, checkpoint_id))
                logging.info(f"Final evaluation checkpoint queued at update {config.n_updates}")

            # Wait a bit for final evaluation
            logging.info("Waiting for final async evaluation...")
            await asyncio.sleep(config.eval_poll_interval * 2)
            await check_evaluation_results()
        else:
            logging.info("\n=== Final Evaluation ===")
            final_results = await evaluate_top_batch_elements(
                trainer, energy_estimator, config.n_updates, config
            )

            if final_results and evaluator_mode_metadata["allows_full_state_evaluation"]:
                reporter.add_results(final_results)
            elif final_results:
                report_path = save_scalable_large_report_safely(
                    final_results,
                    results_dir,
                )
                logging.info(
                    "Saved final scalable-large structural report to: "
                    f"{report_path}"
                )

    if evaluation_enabled and config.async_eval and evaluator_process:
        logging.info("\nTraining complete. Waiting for evaluator to finish...")

        # Signal evaluator to stop
        checkpoint_queue.put('STOP')

        # Drain ``results_queue`` *concurrently* with the
        # ``join`` instead of blocking on ``join`` first. ``Queue.put``
        # in the evaluator can stall at process shutdown if the parent
        # isn't reading the queue — the feeder thread holds the data
        # until consumed. A blocked feeder then makes ``join`` hit the
        # eval_process_timeout, we ``terminate``, and the last (possibly
        # valid) result payload is lost. Drain in a tight poll loop so
        # large exact-eval payloads make it through before we give up.
        timeout_deadline = time.time() + config.eval_process_timeout
        while evaluator_process.is_alive() and time.time() < timeout_deadline:
            try:
                await check_evaluation_results()
            except Exception as drain_err:
                logging.warning(f"Drain-while-join errored (continuing): {drain_err}")
            evaluator_process.join(timeout=0.5)
        # One final drain pass after the child exited or we hit the deadline,
        # in case there's still data sitting in the feeder.
        try:
            await check_evaluation_results()
        except Exception as drain_err:
            logging.warning(f"Final drain after join errored (continuing): {drain_err}")

        if evaluator_process.is_alive():
            logging.warning("Evaluator process did not finish within timeout. Terminating...")
            evaluator_process.terminate()
            evaluator_process.join(timeout=10)

            if evaluator_process.is_alive():
                logging.error("Evaluator process could not be terminated. Killing...")
                evaluator_process.kill()

        # Surface a non-zero evaluator exitcode. The
        # evaluator's own try/except now ``sys.exit(2)``s on fatal errors;
        # log loudly here so operators don't miss the failure when the
        # parent reports training success.
        # Ensure the process is fully reaped before reading
        # ``exitcode``. ``kill()`` returns immediately and ``exitcode``
        # can still be ``None`` until the OS reaps the child; without a
        # final ``join()`` we'd silently skip the non-zero-exit failure
        # path (and the sticky-flag set below).
        if not evaluator_process.is_alive():
            pass  # already reaped via prior join
        else:
            evaluator_process.join(timeout=5.0)
        eval_exitcode = evaluator_process.exitcode
        if eval_exitcode is not None and eval_exitcode not in (0, None):
            logging.error(
                f"Async evaluator process exited with non-zero code "
                f"{eval_exitcode}. Final evaluation artifacts may be missing "
                f"or stale; training will still report results from the in-process drain."
            )
            evaluator_failed["value"] = True
            evaluator_failed["reasons"].append(f"process exited code={eval_exitcode}")
        elif eval_exitcode is None:
            # Process was force-killed and never reaped, OR still running.
            # Both are signs of pathology — flag as failed.
            logging.error(
                f"Async evaluator process exitcode is None after final join — "
                f"likely force-killed or stuck. Marking evaluator as failed."
            )
            evaluator_failed["value"] = True
            evaluator_failed["reasons"].append("exitcode=None after final join (force-kill or stuck)")

        # Normal cleanup path reached — pop the evaluator from the
        # cleanup registry so a later ``drain_registered_evaluators`` is a no-op.
        _unregister_evaluator(evaluator_process)

        # Collect any remaining results
        await check_evaluation_results()

    # Unconditional final canonical checkpoint. The main loop only saves on
    # ``checkpoint_every`` boundaries and the post-loop save above only fires in
    # the async-final-eval branch, so a completed run could otherwise leave
    # ``checkpoint_update.pth`` behind the final weights. Write one here whenever
    # the trainer reached the target and the canonical file is stale or missing.
    #
    # ``_final_checkpoint_persisted`` flips only once the on-disk canonical
    # checkpoint covers the completed loop state. Until then, post-loop signals
    # still requeue, so the next run resumes from periodic state rather than
    # terminating before final progress reached disk.
    try:
        target_updates = int(getattr(config, "n_updates", 0) or 0)
        if (
            getattr(trainer, "completed_updates", 0) >= target_updates > 0
        ):
            canonical_path = results_dir / "checkpoint_update.pth"
            existing_update = -1
            if canonical_path.exists():
                try:
                    payload = torch.load(canonical_path, map_location='cpu', weights_only=False)
                    if isinstance(payload, dict):
                        existing_update = int(payload.get('update', 0) or 0)
                except Exception:
                    existing_update = -1
            if existing_update < target_updates:
                logging.info(
                    f"Writing final canonical checkpoint at update {target_updates} "
                    f"(existing canonical was at update={existing_update})"
                )
                metrics_history_cpu = convert_metrics_history_to_cpu(trainer.metrics_history)
                trainer.gfn.save_checkpoint(
                    str(canonical_path), target_updates, metrics_history_cpu
                )
                _final_checkpoint_persisted = True
            else:
                # On-disk canonical already at >= target. Either a
                # checkpoint-aligned final periodic save covered it,
                # or a previous run completed the final write. Either
                # way, post-loop no-requeue is safe.
                _final_checkpoint_persisted = True
        else:
            # No target / loop ended early — there is no "final" state
            # to persist beyond what periodic saves already wrote, so
            # the post-loop no-requeue path is safe.
            _final_checkpoint_persisted = True
    except Exception as final_save_err:
        logging.error(
            f"Final canonical checkpoint write failed: {final_save_err}. "
            f"Resume may fall back to an older checkpoint."
        )
        # IMPORTANT: leave ``_final_checkpoint_persisted = False``.
        #
        # This only triggers a requeue for SIGUSR1
        # (preempt warning) — the defer-branch helper fires
        # ``scontrol requeue`` immediately, and the wrapper's
        # post-return path then routes to ``shutdown_exit_code() == 99``
        # so ``RequeueExit=99`` policy applies as a second line of
        # defense. SIGTERM/manual cancel still exits 143 with no
        # requeue (per the operator-cancel contract); the failed
        # final-write progress IS dropped in that case, but that's
        # consistent with the explicit cancel semantics.

    # Final visualization update and exports
    await reporter.update_summary_async()
    reporter.export_results_json(results_dir / 'evaluation_results.json')
    reporter.export_results_json(results_dir / 'all_evaluation_results.json')
    
    full_results = reporter.load_all_results()

    # Pull the sticky evaluator failure state if async eval was enabled.
    # ``evaluator_failed`` lives in the async branch's local scope; it
    # only exists when the async eval setup ran.
    evaluator_failure_info = locals().get("evaluator_failed")

    # Generate final report
    generate_final_report(
        full_results, hamiltonian_helper, results_dir, hyperparameters, trainer,
        evaluator_failure_info=evaluator_failure_info,
    )

    # Refuse to silently report a clean success when the
    # async evaluator hit a fatal error at any point. Operators / job
    # orchestration need to see a non-zero exit so missing/stale eval
    # artifacts aren't masked. The training itself completed (we got
    # here), so we DON'T re-raise — we just surface the partial state.
    if evaluator_failure_info and evaluator_failure_info.get("value"):
        logging.error(
            "Async evaluator reported one or more fatal errors during this run "
            "(reasons: %s). Marking run as PARTIAL — training completed but "
            "evaluation artifacts may be missing or stale.",
            evaluator_failure_info.get("reasons", []),
        )
        # Caller (``run_config.main``) sees this via the structured return.
        # Stash on the returned trainer so external callers can detect it
        # without parsing the report.
        try:
            setattr(trainer, "_evaluator_failure_info", evaluator_failure_info)
        except Exception:
            pass
        logging.warning(f"\nExperiment completed (PARTIAL — eval failures): {results_dir}")
    else:
        logging.info(f"\nExperiment completed! Results saved to {results_dir}")

    return full_results, trainer


def generate_final_report(results: List[ExtendedBatchElementEnergyResult],
                         hamiltonian_helper: PauliHamiltonianHelper,
                         results_dir: Path,
                         hyperparameters: Dict,
                         trainer: EfficientGFNTrainer,
                         evaluator_failure_info: Optional[Dict] = None):
    """Generate a final summary report.

    ``evaluator_failure_info`` is the sticky parent-side
    record of any async evaluator failures observed during the run. When
    set with ``value=True``, the report is prefixed with a clear
    "PARTIAL — evaluator failed" banner so the artifact reflects reality.
    """

    evaluation_metadata = hyperparameters.get("evaluation", {})
    energy_metadata = hyperparameters.get("energy_estimation", {})
    sampler_metadata = hyperparameters.get("sampler", {})
    evaluator_mode = evaluation_metadata.get(
        "mode",
        energy_metadata.get("configured_evaluator_mode", EVALUATOR_MODE_EXACT_SMALL),
    )
    large_hubbard_mode = evaluation_metadata.get("large_hubbard_mode", False)
    full_state_allowed = evaluation_metadata.get(
        "allows_full_state_evaluation",
        evaluator_mode == EVALUATOR_MODE_EXACT_SMALL,
    )
    structural_reporting_enabled = energy_metadata.get(
        "scalable_large_structural_reporting_enabled",
        evaluation_metadata.get(
            "scalable_large_structural_reporting_enabled",
            False,
        ),
    )
    structural_reports = load_scalable_large_reports(results_dir)
    latest_structural_update = max(
        (int(r.get("update", 0)) for r in structural_reports),
        default=0,
    )
    total_structural_batch_elements = sum(
        int(r.get("n_batch_elements", 0) or 0)
        for r in structural_reports
    )
    total_structural_circuits = sum(
        int(r.get("n_circuits_total", 0) or 0)
        for r in structural_reports
    )
    exact_ground_state_energy = hyperparameters["hamiltonian"].get(
        "exact_ground_state_energy"
    )
    if exact_ground_state_energy is None:
        exact_ground_state_line = (
            "- Exact ground state energy: not computed "
            f"({evaluator_mode} mode)"
        )
    else:
        exact_ground_state_line = (
            f"- Exact ground state energy: {exact_ground_state_energy:.10f}"
        )
    
    report_lines = [
        "# GFlowNet Quantum Circuit Optimization Experiment Report",
        f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
    ]
    # If the sticky evaluator-failure flag was set,
    # prefix the report with a clear "PARTIAL — evaluator failed" banner.
    if evaluator_failure_info and evaluator_failure_info.get("value"):
        reasons = evaluator_failure_info.get("reasons", []) or ["(no reasons recorded)"]
        report_lines.append(
            "\n> **PARTIAL RUN — async evaluator reported fatal errors.** "
            "Training completed, but evaluation artifacts may be missing or stale. "
            "Reasons: " + "; ".join(str(r) for r in reasons)
        )
    report_lines.extend([
        f"\n## Hamiltonian Information",
        f"- File: {hamiltonian_helper.filepath}",
        f"- Number of qubits: {hamiltonian_helper.n_qubits}",
        f"- Number of Pauli terms: {len(hamiltonian_helper.pauli_str_list)}",
        f"- Number of training terms: {hyperparameters['hamiltonian']['n_training_terms']} (excluding identity)",
        f"- Identity term weight: {hyperparameters['hamiltonian']['identity_weight']:.6f}",
        exact_ground_state_line,
        
        f"\n## Key Hyperparameters",
        f"- Model type: {hyperparameters['gfn_model']['model_type']}",
        f"- Hidden dimensions: {hyperparameters['gfn_model']['hidden_dim']}",
        f"- Hidden layers: {hyperparameters['gfn_model']['num_hidden_layers']}",
        f"- Learning rate: {hyperparameters['gfn_model']['lr']}",
        f"- Beta (temperature): {hyperparameters['training']['beta']}",
        f"- Circuits per batch element: {hyperparameters['batch_structure']['batch_element_size']}",
        f"- Batch elements per update: {hyperparameters['batch_structure']['batch_elements_per_update']}",
        f"- Max circuit length: {hyperparameters['training']['max_depth']}",
        f"- Cost function: {hyperparameters['cost_function']['type']} with kwargs: {hyperparameters['cost_function'].get('kwargs', {})}",
        f"- Zero stabilizer cost weights: {hyperparameters['cost_function'].get('zero_stabilizer_cost_weights', False)}",
        f"- Device: {hyperparameters['computational']['device']['type']}",
        f"- Sampler: {sampler_metadata.get('effective_sampler', 'not recorded')}",
        f"- Requested sampling_mode: "
        f"{sampler_metadata.get('requested_sampling_mode', 'not recorded')}",
        f"- Effective sampling_mode: "
        f"{sampler_metadata.get('effective_sampling_mode', 'not recorded')}",
        f"- Static-shape sampling: "
        f"{sampler_metadata.get('effective_static_shape_sampling', 'not recorded')}",
        f"- CUDA graph sampling: "
        f"{sampler_metadata.get('cuda_graph_sampling', 'not recorded')}",
        f"- Dynamic CUDA graph policy (requested): "
        f"{sampler_metadata.get('use_cuda_graph_policy', 'not recorded')}",
        f"- Dynamic CUDA graph policy (eligible): "
        f"{sampler_metadata.get('cuda_graph_policy_eligible', 'not recorded')}",
        f"- Dynamic CUDA graph policy max rows: "
        f"{sampler_metadata.get('cuda_graph_policy_max_rows', 'not recorded')}",
        # These values come from the ExperimentConfig (the REQUEST), not from
        # observed runtime state: a request of ``True`` does not mean the kernel
        # ran, since CuPy/NVRTC issues, a missing CT install, or a latch trip can
        # silently degrade to the fallback. They are labelled "requested"
        # accordingly. Runtime-used telemetry exists only for some of the family:
        # ``fused_apply_used`` / ``fused_apply_call_count`` (call-site counter),
        # ``fused_mask_counts_used`` (request AND latch-clear AND
        # effective_sampling_mode in {"dynamic_active", "bucketed"}), and
        # ``fused_sampling_effective`` (request AND module latch clear).
        # fused_metadata has module-latch telemetry in the sampler benchmark
        # only, and none of it is surfaced in this report.
        f"- Fused metadata kernel (requested): "
        f"{sampler_metadata.get('use_fused_metadata_kernel', 'not recorded')}",
        f"- Fused sampling kernel (requested): "
        f"{sampler_metadata.get('use_fused_sampling_kernel', 'not recorded')}",
        f"- Fused mask+counts kernel (requested): "
        f"{sampler_metadata.get('use_fused_mask_counts_kernel', 'not recorded')}",
        f"- Fused counter-RNG kernel (requested): "
        f"{sampler_metadata.get('use_fused_counter_rng_kernel', 'not recorded')}",
        f"- Fused partition-update kernel (requested): "
        f"{sampler_metadata.get('use_fused_partition_update_kernel', 'not recorded')}",
        f"- Fused apply kernel (requested): "
        f"{sampler_metadata.get('use_fused_apply_kernel', 'not recorded')}",
        f"- bf16 sampling autocast (requested): "
        f"{sampler_metadata.get('use_bf16_sampling', 'not recorded')}",
        # P2.3: bf16 autocast on the gradient-path GEMMs (training-affecting but
        # within seed variance; default OFF). BACKWARD-only knob -> read from the
        # ``computational`` dict (like use_activation_checkpointing), not sampler.
        f"- bf16 backward autocast (requested): "
        f"{hyperparameters['computational'].get('use_bf16_backward', 'not recorded')}",
        # P0.5: backward-only knob (not a sampler control). Resolved bool;
        # request default ``None``/auto resolves to ``bool(large_hubbard_mode)``
        # in __post_init__, so the effective value is not inferable from the
        # request. Selects the cached-flow backward recompute path / memory
        # envelope, hence surfaced here as a genuine run control.
        f"- Activation checkpointing (cached-flow backward): "
        f"{hyperparameters['computational'].get('use_activation_checkpointing', 'not recorded')}",
        f"- uint8 state cache (requested): "
        f"{hyperparameters['computational'].get('use_uint8_state_cache', 'not recorded')}",
        f"- Async evaluation: {hyperparameters['computational'].get('async_evaluation', False)}",
        f"- Large Hubbard mode: {large_hubbard_mode}",
        f"- Configured evaluator mode: {evaluator_mode}",
        # Surface the validation-tier in the human-facing report
        # so a reader can distinguish structural-only from DMRG-backed
        # readiness without parsing the JSON sidecars (fix).
        f"- Validation tier: {energy_metadata.get('validation_tier', 'unknown')}",
        f"- Sufficient for final readiness claim: "
        f"{energy_metadata.get('sufficient_for_final_readiness_claim', False)}",
        f"- DMRG reference available: "
        f"{energy_metadata.get('dmrg_reference_available', False)}",

        f"\n## Energy Estimation Method",
        f"- Method: {hyperparameters['energy_estimation']['method']}",
        f"- Implementation: {hyperparameters['energy_estimation'].get('implementation', 'N/A')}",
        f"- Equation: {hyperparameters['energy_estimation']['equation']}",
        f"- Reference: {hyperparameters['energy_estimation']['reference']}",
        f"- Full-state evaluation allowed by mode: {full_state_allowed}",
        f"- Exact EnergyEstimator enabled: {energy_metadata.get('exact_energy_estimator_enabled', True)}",
        f"- Scalable-large structural reporting enabled: {structural_reporting_enabled}",
        # tier description gives the reader the why behind the tier.
        f"- Tier description: {energy_metadata.get('tier_description', 'n/a')}",
    ])

    if 'simulations' in hyperparameters['energy_estimation'] and hyperparameters['energy_estimation']['simulations'] > 1:
        report_lines.extend([
            f"- Simulations per batch element: {hyperparameters['energy_estimation']['simulations']}",
            f"- Error formula: {hyperparameters['energy_estimation']['error_formula']}",
        ])
    
    if hyperparameters['computational'].get('async_evaluation', False):
        report_lines.append(
            "- Evaluation scheduling: "
            f"{hyperparameters['energy_estimation'].get('async_mode', 'asynchronous')}"
        )
    
    latest_exact_update = max((r.update for r in results), default=0)
    # "Total training updates" must reflect actual training
    # progress, not the latest evaluation update. Eval reports lag training
    # and can also be filtered, so labeling them as the
    # training total under-reported scalable-large progress whenever
    # structural reporting was sparse or async final-eval timed out.
    # Preferred order: trainer's ``completed_updates`` → hyperparameters
    # ``n_updates`` → fall back to max(exact, structural) only if neither
    # is available.
    completed_from_trainer = int(getattr(trainer, "completed_updates", 0) or 0)
    configured_target = int(
        hyperparameters.get("experiment", {}).get("n_updates", 0) or 0
    )
    if completed_from_trainer > 0:
        total_training_updates = completed_from_trainer
        progress_note = f"trainer.completed_updates={completed_from_trainer}"
    elif configured_target > 0:
        total_training_updates = configured_target
        progress_note = f"hyperparameters.experiment.n_updates={configured_target}"
    else:
        total_training_updates = max(latest_exact_update, latest_structural_update)
        progress_note = "max(latest_exact_update, latest_structural_update) — no trainer state"
    report_lines.extend(["\n## Experiment Summary",
        f"- Total training updates: {total_training_updates}  ({progress_note})",
        f"- Latest evaluated update: {latest_exact_update}",
        f"- Latest structural report update: {latest_structural_update}",
        f"- Total batch elements evaluated: {len(results) + total_structural_batch_elements}",
        f"- Exact batch elements evaluated: {len(results)}",
    ])
    if structural_reports:
        report_lines.extend([
            f"- Structural reports generated: {len(structural_reports)}",
            f"- Structural batch elements reported: {total_structural_batch_elements}",
            f"- Structural circuits reported: {total_structural_circuits}",
        ])
    
    # Add simulation info if available
    if 'simulations' in hyperparameters.get('energy_estimation', {}) and hyperparameters['energy_estimation']['simulations'] > 1:
        n_sims = hyperparameters['energy_estimation']['simulations']
        report_lines.append(f"- Total simulations run: {len(results) * n_sims}")
    
    exact_evaluator_enabled = energy_metadata.get(
        "exact_energy_estimator_enabled",
        True,
    )
    if exact_evaluator_enabled:
        report_lines.extend([
            f"- Evaluation frequency: every {hyperparameters['experiment']['eval_every']} updates",
            f"- Batch elements per evaluation: {hyperparameters['experiment']['n_eval_top_k_batch_elements']}",
        ])
    else:
        report_lines.append("- Exact EnergyEstimator evaluation: disabled")

    if structural_reports:
        latest_structural_report = max(
            structural_reports,
            key=lambda r: int(r.get("update", 0)),
        )
        # Surface validation-tier fields in the markdown so a reader sees at once
        # whether the run is structural-only or DMRG-backed. ``Energy estimate``
        # is the policy/evaluator output (never computed in scalable_large) and
        # the DMRG reference scalar gets its own line, so neither can be mistaken
        # for the other. Invariant normalization happens at the load boundary in
        # ``_normalize_loaded_report_invariants``, so the tier fields and the
        # coerced scalar are already in sync here; the ``.get(..., default)``
        # calls are defensive only, for a report passed in without the loader.
        latest_dmrg_energy = latest_structural_report.get(
            "dmrg_reference_energy"
        )
        latest_tier = latest_structural_report.get(
            "validation_tier", "structural"
        )
        latest_readiness = latest_structural_report.get(
            "sufficient_for_final_readiness_claim", False
        )
        latest_tier_desc = latest_structural_report.get(
            "tier_description", ""
        )
        latest_energy_status = latest_structural_report.get(
            "energy_status", "not_computed_structural_report_only"
        )
        if latest_dmrg_energy is None:
            dmrg_line = "- DMRG reference energy: none attached"
        else:
            dmrg_line = (
                f"- DMRG reference energy: {latest_dmrg_energy:.10f}"
            )
        report_lines.extend([
            "\n## Scalable-Large Structural Reports",
            f"- Reports generated: {len(structural_reports)}",
            f"- Latest report update: {latest_structural_report.get('update', 0)}",
            f"- Latest reported batch elements: {latest_structural_report.get('n_batch_elements', 0)}",
            f"- Latest reported circuits: {latest_structural_report.get('n_circuits_total', 0)}",
            f"- Latest mean circuit length: {latest_structural_report.get('mean_circuit_length', 0.0):.1f}",
            f"- Validation tier: {latest_tier}",
            f"- Sufficient for final readiness claim: {latest_readiness}",
            f"- Energy estimate: not computed (status: {latest_energy_status})",
            dmrg_line,
        ])
        if latest_tier_desc:
            report_lines.append(f"- Tier description: {latest_tier_desc}")
    
    if not results:
        if structural_reports:
            report_lines.append("\nNo exact EnergyEstimator results were generated.")
        else:
            report_lines.append("\nNo evaluation results were generated.")
        with open(results_dir / 'experiment_report.md', 'w') as f:
            f.write('\n'.join(report_lines))
        return

    # Get final results
    final_update = max(r.update for r in results)
    final_results = [r for r in results if r.update == final_update]
    
    if final_results:
        energy_diffs = [r.energy_difference for r in final_results]
        best_result = min(final_results, key=lambda r: r.energy_difference)
        
        total_circuits = sum(r.n_circuits for r in final_results)
        total_measurements = sum(r.total_measurements for r in final_results)
        mean_coverage = np.mean([r.convergence_metrics['coverage'] for r in final_results])
        
        # Get MAE statistics
        mae_results = [r for r in final_results if hasattr(r, 'simulation_result') and r.simulation_result is not None]
        if mae_results:
            mae_values = [r.simulation_result.mean_absolute_error for r in mae_results]
            mae_stds = [r.simulation_result.std_absolute_error for r in mae_results]
            best_mae = min(mae_values)
            mean_mae = np.mean(mae_values)
        
        # Under M>1 each ``r.energy_difference`` is a MAE
        # aggregate (post-), so the best/mean aggregates here are
        # best-MAE / mean-MAE. Disambiguate in the markdown by detecting
        # whether any result carries a simulation_result block, which is
        # the same M>1 signal used by the logger and the per-batch loop.
        diff_label = "MAE" if mae_results else "Energy difference"
        report_lines.extend([
            f"\n## Final Results (Update {final_update})",
            f"- Batch elements evaluated: {len(final_results)}",
            f"- Total circuits: {total_circuits}",
            f"- Total measurements: {total_measurements}",
            f"- Best {diff_label} (Batch rank {best_result.batch_element_rank}): {best_result.energy_difference:.6e}",
            f"- Mean {diff_label}: {np.mean(energy_diffs):.6e}",
            f"- Success rate (< 1.6e-3): {sum(1 for e in energy_diffs if e < 1.6e-3) / len(energy_diffs) * 100:.1f}%",
            f"- Success rate (< 1e-2): {sum(1 for e in energy_diffs if e < 1e-2) / len(energy_diffs) * 100:.1f}%",
            f"- Mean Pauli coverage: {mean_coverage:.1%}",
            f"- Mean circuit length: {np.mean([r.mean_circuit_length for r in final_results]):.1f}",
        ])
        
        if mae_results:
            report_lines.extend([
                f"\n### Simulation Statistics",
                f"- Best mean absolute error: {best_mae:.6e}",
                f"- Average mean absolute error: {mean_mae:.6e}",
                f"- Standard deviation of MAE (best batch): {mae_stds[mae_values.index(best_mae)]:.6e}",
            ])
        
        # Training metrics summary
        if hasattr(trainer, 'metrics_history') and trainer.metrics_history:
            # Convert final metrics from GPU tensors to CPU floats for formatting
            final_metrics_raw = {k: v[-1] if v else 0 for k, v in trainer.metrics_history.items()}
            final_metrics = {}
            for k, v in final_metrics_raw.items():
                if torch.is_tensor(v):
                    final_metrics[k] = v.item() if v.numel() == 1 else v.cpu().numpy()
                else:
                    final_metrics[k] = v

            report_lines.extend([
                f"\n## Final Training Metrics",
                f"- Final loss: {final_metrics.get('loss', 0):.6f}",
                f"- Final reward: {final_metrics.get('reward', 0):.4f}",
                f"- Final cost: {final_metrics.get('cost', 0):.4f}",
                f"- Final logZ: {final_metrics.get('logZ', 0):.3f}",
            ])
    
    # Progress summary
    updates = sorted(set(r.update for r in results))
    if len(updates) > 1:
        report_lines.append(f"\n## Training Progress")
        report_lines.append("| Update | Best Energy Diff | Mean Energy Diff | Best MAE | Total Circuits | Pauli Coverage |")
        report_lines.append("|--------|-----------------|------------------|----------|----------------|----------------|")
        
        for update in updates:
            update_results = [r for r in results if r.update == update]
            if update_results:
                energy_diffs = [r.energy_difference for r in update_results]
                best_energy = min(energy_diffs)
                mean_energy = np.mean(energy_diffs)
                total_circuits = sum(r.n_circuits for r in update_results)
                mean_coverage = np.mean([r.convergence_metrics['coverage'] for r in update_results])
                
                # Get MAE for this update
                mae_results = [r for r in update_results if hasattr(r, 'simulation_result') and r.simulation_result is not None]
                if mae_results:
                    best_mae = min(r.simulation_result.mean_absolute_error for r in mae_results)
                    mae_str = f"{best_mae:.2e}"
                else:
                    mae_str = "N/A"
                
                report_lines.append(f"| {update:6d} | {best_energy:15.6e} | {mean_energy:16.6e} | {mae_str:8s} | {total_circuits:14d} | {mean_coverage:14.1%} |")
    
    # Write report
    with open(results_dir / 'experiment_report.md', 'w') as f:
        f.write('\n'.join(report_lines))
    
    logging.info(f"\nFinal report saved to {results_dir / 'experiment_report.md'}")


async def evaluate_top_batch_elements_scalable_large(
    trainer: EfficientGFNTrainer,
    update: int,
    config: ExperimentConfig,
) -> ScalableLargeEvaluationReport:
    """Build a structural replay-buffer report for scalable-large mode."""
    logging.info(f"\n=== Structural scalable-large report at update {update} ===")

    top_actions = trainer.gfn.top_trajectories_actions
    top_lengths = trainer.gfn.top_trajectories_lengths
    top_costs = (
        trainer.gfn.top_trajectories_costs
        if hasattr(trainer.gfn, "top_trajectories_costs")
        else []
    )

    if not top_actions:
        logging.info("  Replay buffer is empty. Skipping structural report.")
        return {}

    report = create_scalable_large_evaluation_report(
        batch_actions_list=top_actions,
        batch_lengths_list=top_lengths,
        batch_costs=top_costs,
        update=update,
        config=config,
        source="trainer_replay_buffer",
        hamiltonian_metadata=_scalable_large_hamiltonian_metadata(
            config=config,
            trainer=trainer,
        ),
        terminal_index=getattr(trainer.gfn, "terminal_index", None),
        measurement_backend=getattr(trainer.gfn, "measurement_backend", None),
    )

    logging.info(
        "  Structural report: "
        f"{report['n_batch_elements']} batch elements, "
        f"{report['n_circuits_total']} circuits, "
        f"mean length {report['mean_circuit_length']:.1f}"
    )
    return report


# For backward compatibility with existing code
async def evaluate_top_batch_elements(trainer: EfficientGFNTrainer,
                                    energy_estimator: Optional[EnergyEstimator],
                                    update: int,
                                    config: ExperimentConfig) -> EvaluationEntryPointResult:
    """
    Evaluate top-k batch elements from replay buffer using energy estimation.
    Wrapper for synchronous evaluation mode.

    Returns:
        In exact_small mode, a list of ExtendedBatchElementEnergyResult rows.
        In scalable_large mode, a structural report dict. Empty structural
        replay buffers return an empty dict.
    """

    if not get_evaluator_mode_metadata(config)["allows_full_state_evaluation"]:
        return await evaluate_top_batch_elements_scalable_large(
            trainer,
            update,
            config,
        )

    ensure_exact_small_evaluation_allowed(
        config,
        "Synchronous top-batch evaluation",
    )
    if energy_estimator is None:
        raise RuntimeError(
            "Synchronous top-batch evaluation in exact_small mode requires "
            "an EnergyEstimator instance."
        )
    
    logging.info(f"\n=== Evaluating top batch elements at update {update} ===")
    
    # Get top batch elements from the replay buffer
    top_actions = trainer.gfn.top_trajectories_actions
    top_lengths = trainer.gfn.top_trajectories_lengths
    top_costs = trainer.gfn.top_trajectories_costs if hasattr(trainer.gfn, 'top_trajectories_costs') else None
    
    if not top_actions:
        logging.info("  Replay buffer is empty. Skipping evaluation.")
        return []
    
    # Determine how many batch elements to evaluate
    num_to_eval = len(top_actions) #min(config.n_eval_top_k_batch_elements, len(top_actions))
    logging.info(f"  Evaluating top {num_to_eval} batch elements from replay buffer")
    
    n_simulations = getattr(config, 'n_simulations', 1)

    if n_simulations > 1:
        logging.info(f"  Running {n_simulations} simulations per batch element")

    # Prepare batch data for all batch elements
    batch_actions_list = []
    batch_lengths_list = []
    
    for batch_idx in range(num_to_eval):
        # Get the batch data
        batch_actions = top_actions[batch_idx]
        batch_lengths = top_lengths[batch_idx]
        
        # Handle different possible storage formats
        if len(batch_actions.shape) == 1:
            # Single circuit stored - expand to batch
            batch_actions = batch_actions.unsqueeze(0)
            batch_lengths = batch_lengths.unsqueeze(0) if isinstance(batch_lengths, torch.Tensor) else torch.tensor([batch_lengths])
        
        batch_actions_list.append(batch_actions)
        batch_lengths_list.append(batch_lengths)
    
    # Stack all batch elements together
    # Each element in batch_actions_list has shape (n_circuits, max_depth)
    # We want to create shape (batch_size, n_circuits, max_depth)
    all_batch_actions = torch.stack(batch_actions_list, dim=0)
    all_batch_lengths = torch.stack(batch_lengths_list, dim=0)
    
    logging.info(f"  Combined batch shape: {all_batch_actions.shape}")
    
    # Run energy estimation for all batch elements at once.
    # ``estimate_energy_with_simulations`` is synchronous (plain ``def``);
    # awaiting it raised ``TypeError: object list can't be used in 'await'
    # expression`` and crashed the live final eval. The checkpoint-replay
    # path at ``evaluate_top_batch_elements_from_checkpoint`` already calls
    # it without ``await`` — mirror that.
    summaries = energy_estimator.estimate_energy_with_simulations(
        all_batch_actions,
        all_batch_lengths,
        M=n_simulations,
    )
    
    # Process results
    results = []
    
    for batch_idx, summary in enumerate(summaries):
        # Extract information from summary
        final_result_obj = summary['final_results_object']
        
        # Create simulation result if we ran multiple simulations
        simulation_result = None
        if n_simulations > 1 and 'energy_variance' in summary:
            mean_energy = summary['mean_energy']
            variance = summary['energy_variance']

            # Only create simulation result if we have actual individual energies
            if 'individual_energies' in summary and 'individual_absolute_errors' in summary:
                simulation_result = SimulationResult(
                    energy_estimates=summary['individual_energies'],
                    absolute_errors=summary['individual_absolute_errors'],
                    rmse=summary['rmse'],
                    std_absolute_error=summary.get('std_absolute_error', 0.0),
                    mean_energy_estimate=mean_energy,
                    std_energy_estimate=np.sqrt(variance) if variance > 0 else 0.0
                )
            else:
                # If individual energies aren't available, don't create fake data
                logging.warning(f"Individual simulation energies not available for batch {batch_idx}")
                simulation_result = None
        
        # Create extended result.
        # ``rmse`` / ``mae`` are first-class fields populated from
        # the summary; ``energy_difference`` is now the absolute-error
        # quantity its name implies (=MAE at M>1). Mirror of the fix at
        # ``evaluate_top_batch_elements_from_checkpoint``.
        result = ExtendedBatchElementEnergyResult(
            batch_element_rank=batch_idx,
            energy_estimate=summary['mean_energy'],
            energy_difference=summary['energy_difference'],
            rmse=summary.get('rmse'),
            mae=summary.get('mae'),
            hitting_counts=final_result_obj.hitting_counts,
            pauli_estimates=summary['mean_pauli_estimates'],
            circuit_lengths=final_result_obj.circuit_lengths,
            mean_circuit_length=final_result_obj.mean_circuit_length,
            n_circuits=final_result_obj.n_circuits,
            total_measurements=final_result_obj.total_measurements,
            convergence_metrics=final_result_obj.convergence_metrics,
            batch_cost=final_result_obj.batch_cost,
            update=update,
            simulation_result=simulation_result
        )

        # Add cost if available
        if top_costs is not None and batch_idx < len(top_costs):
            result.batch_cost = top_costs[batch_idx].item() if torch.is_tensor(top_costs[batch_idx]) else top_costs[batch_idx]

        results.append(result)

        # Print summary for this batch element.
        # Label the aggregate by what it actually is. Mirror of
        # the fix in ``evaluate_top_batch_elements_from_checkpoint``.
        logging.info(f"\n  Batch element rank {batch_idx}:")
        logging.info(f"    Number of circuits: {result.n_circuits}")
        logging.info(f"    Energy estimate: {result.energy_estimate:.6f}")
        if simulation_result and n_simulations > 1:
            logging.info(f"    MAE (energy_difference): {result.energy_difference:.6e}")
            if result.rmse is not None:
                logging.info(f"    RMSE: {result.rmse:.6e}")
            logging.info(f"    Std absolute error: {simulation_result.std_absolute_error:.6e}")
        else:
            logging.info(f"    Energy difference: {result.energy_difference:.6e}")
        logging.info(f"    Pauli coverage: {result.convergence_metrics['coverage']:.1%}")
        logging.info(f"    Mean circuit length: {result.mean_circuit_length:.1f}")
    
    # Print overall summary
    if results:
        energy_diffs = [r.energy_difference for r in results]
        best_result = min(results, key=lambda r: r.energy_difference)

        # Mirror of the disambiguation in
        # ``evaluate_top_batch_elements_from_checkpoint``.
        diff_label = "MAE" if n_simulations > 1 else "Energy difference"
        logging.info(f"\n  Evaluation Summary:")
        logging.info(
            f"    Best {diff_label} (Batch rank {best_result.batch_element_rank}): "
            f"{best_result.energy_difference:.6e}"
        )
        logging.info(f"    Mean {diff_label}: {np.mean(energy_diffs):.6e}")
        logging.info(f"    Total circuits evaluated: {sum(r.n_circuits for r in results)}")
        if n_simulations > 1:
            logging.info(f"    Total simulation runs: {len(results) * n_simulations}")
        logging.info(f"    Success rate (< 1.6e-3): {sum(1 for e in energy_diffs if e < 1.6e-3) / len(energy_diffs) * 100:.1f}%")

    return results


if __name__ == "__main__":
    # Example configuration
    config = ExperimentConfig(
        model_type="clifford_mlp",                # Model architecture
        hamiltonian_path="Hamiltonians/H2_6-31G_8qubits/jw.txt",
        eval_every=200,                    # Evaluate every N training updates
        n_updates=100000,                  # Total training updates
        n_eval_top_k_batch_elements=5,     # Number of top batch elements to evaluate
        n_measurements=1000,                 # Circuits per batch element
        update_freq=100,                     # Batch elements per training update
        max_depth=2,                       # Maximum circuit depth
        beta=1e2,                          # Temperature parameter
        hidden_dim=512,                   # Neural network hidden dimension
        num_hidden_layers=2,               # Number of hidden layers
        lr=1e-3,                           # Learning rate
        weight_decay=1e-5,                 # Weight decay
        device_preference="auto",           # "cuda", "mps", "cpu", or "auto"
        results_dir="results/H2_6-31G_8qubits",
        replay_every=50,                   # Replay training frequency
        offpolicy_every=50,                # Off-policy training frequency
        checkpoint_every=50,              # Model checkpoint frequency
        reward_type="log",             # Reward function type
        reward_kwargs={"alpha": 1.},     # Reward function parameters
        cost_type="linear_bias",           # Cost function type
        cost_kwargs={"epsilon": 0.9},      # Cost function parameters
        objective_type="tb",               # GFlowNet objective
        objective_kwargs={"loss_type": "squared"},
        n_simulations=500,                 # Number of simulation runs (set to 1 for single run)
        resume=True,                       # Auto-resume from latest matching experiment
        experiment_dir=None,               # Or specify: "experiment_20241210_143022"
        async_eval=True,                   # Enable asynchronous evaluation
        eval_poll_interval=30,             # Seconds between checkpoint polls
        eval_process_timeout=300           # Timeout for evaluator shutdown
    )
    
    # Run experiment (will resume automatically if matching experiment exists)
    asyncio.run(run_experiment(config))
