#!/usr/bin/env python3
"""
Wrapper script to run GFlowNet experiments from JSON configuration files.
Designed for SLURM cluster submission (checkpoint / requeue aware).

This script:
1. Loads configuration from JSON file
2. Creates ExperimentConfig with proper defaults
3. Runs the experiment using the main.py infrastructure
4. Handles preemption signals (SIGTERM, SIGUSR1) for graceful shutdown
"""

import os
import sys
import json
import asyncio
import argparse
import logging
import signal
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

# Add parent directory to path to import from code module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from main.py
from code.main import ExperimentConfig, run_experiment

# Shutdown state lives in ``code.main``, which is always loaded under its canonical
# name regardless of how ``run_config`` was launched. The module-level names here
# are a thin proxy so legacy callers/tests reading
# ``run_config._shutdown_requested`` still work; internal paths go through
# ``code.main``.
def _shutdown_requested_proxy() -> bool:
    try:
        from code import main as _main
        return bool(getattr(_main, "_shutdown_requested", False))
    except Exception:
        return False


def setup_logging(log_file: Optional[str] = None):
    """Setup logging configuration."""
    log_format = "%(asctime)s - %(levelname)s - %(message)s"
    
    if log_file:
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
    else:
        logging.basicConfig(
            level=logging.INFO,
            format=log_format
        )


def validate_config(config_dict: Dict[str, Any]) -> None:
    """Validate required configuration parameters."""
    required_fields = ['hamiltonian_path']
    
    for field in required_fields:
        if field not in config_dict:
            raise ValueError(f"Missing required field: {field}")
    
    # Check if hamiltonian file exists
    hamiltonian_path = config_dict['hamiltonian_path']
    if not Path(hamiltonian_path).exists():
        # Try relative to project root
        project_root = Path(__file__).parent.parent
        full_path = project_root / hamiltonian_path
        if full_path.exists():
            config_dict['hamiltonian_path'] = str(full_path)
        else:
            raise FileNotFoundError(f"Hamiltonian file not found: {hamiltonian_path}")


def load_config_from_json(json_path: str) -> ExperimentConfig:
    """Load ExperimentConfig from JSON file with proper defaults."""
    with open(json_path, 'r') as f:
        config_dict = json.load(f)

    # Strip the documented set of underscore-prefixed keys (inline documentation /
    # workload metadata in canonical configs). ``ExperimentConfig`` is a dataclass
    # without ``**kwargs``, so leaving these in would crash the constructor. The
    # allowlist is intentionally tight so a typo like ``_n_updates`` is NOT silently
    # swallowed — it surfaces as an unexpected-keyword TypeError.
    _DOC_KEYS = ("_comment", "_workload")
    config_dict = {k: v for k, v in config_dict.items() if k not in _DOC_KEYS}

    # Validate configuration
    validate_config(config_dict)

    # Set default model_type if not specified
    if 'model_type' not in config_dict:
        config_dict['model_type'] = 'clifford_mlp'

    # Handle async evaluation settings
    if 'async_eval' not in config_dict:
        config_dict['async_eval'] = True  # Enable by default for cluster

    # Create ExperimentConfig from dictionary
    # The dataclass will use its default values for any missing fields
    config = ExperimentConfig(**config_dict)

    return config


def setup_cluster_environment():
    """Setup environment variables for the SLURM cluster."""
    # Use available cores efficiently, but do NOT clobber explicit per-job
    # overrides: the full-run sbatches export OMP_NUM_THREADS so the trainer and the
    # async-eval subprocess each get their share, and auto-setting to
    # SLURM_CPUS_PER_TASK reintroduces the oversubscription those exports prevent.
    if 'SLURM_CPUS_PER_TASK' in os.environ:
        n_cpus = os.environ['SLURM_CPUS_PER_TASK']
        for var in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS',
                    'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
            if var in os.environ:
                logging.info(
                    f"Keeping explicit {var}={os.environ[var]} "
                    f"(SLURM_CPUS_PER_TASK={n_cpus})"
                )
            else:
                os.environ[var] = n_cpus
                logging.info(f"Set {var}={n_cpus} from SLURM_CPUS_PER_TASK")
    
    # Set PyTorch settings for deterministic behavior
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    
    # Log cluster information if available
    if 'SLURM_JOB_ID' in os.environ:
        logging.info(f"Running on SLURM cluster - Job ID: {os.environ['SLURM_JOB_ID']}")
        if 'SLURM_NODELIST' in os.environ:
            logging.info(f"Node: {os.environ['SLURM_NODELIST']}")
        
        # Log if this is a requeued job
        restart_count = os.environ.get('SLURM_RESTART_COUNT', '0')
        if int(restart_count) > 0:
            logging.info(f"This is a REQUEUED job (restart #{restart_count})")


def _fire_immediate_requeue(signal_name: str, context_label: str) -> None:
    """Fire ``scontrol requeue`` immediately, idempotent best-effort.

    Shared between the defer-branch (`defer_to_post_loop_finalization`) and
    the probe-exception branch. In both cases the handler is about to return
    without ``sys.exit`` from a SIGUSR1 path that could overrun the grace
    window or has no live loop to service the
    eventual exit. Firing requeue here makes the job recoverable
    regardless of how the process actually terminates — SLURM-level
    requeue persists across SIGKILL.

    Idempotent at the controller; safe to call from multiple paths in
    the same job.
    """
    job_id = os.environ.get('SLURM_JOB_ID')
    if not job_id:
        return
    try:
        result = subprocess.run(
            ['scontrol', 'requeue', job_id],
            capture_output=True, text=True, timeout=10, check=False,
        )
        if result.returncode == 0:
            logging.info(
                f"Pre-emptively requested requeue for SLURM_JOB_ID={job_id} "
                f"({context_label}, signal={signal_name}). SLURM-level "
                f"requeue persists across SIGKILL."
            )
        else:
            logging.warning(
                f"Pre-emptive scontrol requeue ({context_label}) "
                f"exited {result.returncode}: {result.stderr.strip()}"
            )
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        logging.warning(f"Could not invoke pre-emptive scontrol requeue ({context_label}): {e}")


def handle_preemption_signal(signum, frame):
    """
    Handle preemption signals (SIGTERM, SIGUSR1) for graceful shutdown.

    Strategy:
    1. Flip ``_shutdown_requested`` so the training loop saves at a safe
       iteration boundary instead of racing the periodic checkpoint save
       — both write to the same atomic-rename ``temp + os.rename`` path,
       so a handler-side save during a normal save can clobber the
       in-flight ``.tmp`` file.
    2. Exit code is signal-specific. SIGUSR1 (the sbatch
       ``--signal=USR1@120`` warning) uses exit 99 + ``scontrol requeue``
       so the job can resume from the safe-point checkpoint. SIGTERM
       (``scancel`` or any other operator-initiated cancel) uses a
       non-99 exit code so clusters configured with ``RequeueExit=99``
       can't resurrect a manually-cancelled job.
    """
    signal_name = signal.Signals(signum).name
    is_warning_signal = (signum == signal.SIGUSR1)
    logging.warning(f"\n{'='*60}")
    if is_warning_signal:
        logging.warning(f"Received {signal_name} — Slurm walltime/preempt warning, will requeue after safe-point save")
    else:
        logging.warning(f"Received {signal_name} — flagging shutdown for safe-point save (no requeue)")
    logging.warning(f"Time: {datetime.now()}")
    logging.warning(f"{'='*60}")

    # Mark for graceful shutdown via the *canonical* ``code.main`` module
    #Setting a global in this file would only flip the
    # local module's copy, which can be a separate object from
    # ``code.main`` when ``run_config`` is loaded as ``__main__``.
    from code import main as _main
    _main.request_shutdown(is_warning_signal)

    # Two cases:
    #   (a) Trainer is initialised (loop running). Don't save inside the handler —
    #       a SIGUSR1 between ``update_step`` and the end-of-iteration
    #       ``completed_updates += 1`` would tag advanced weights with the previous
    #       update number. Just set the shutdown flag; the loop saves at its
    #       safe-point barrier and exits itself.
    #   (b) Trainer isn't published yet. No loop exists to defer to, so the handler
    #       MUST exit itself or the process hangs. Saving is impossible, so log+exit.
    loop_will_handle_save = False
    skip_requeue_post_loop = False
    defer_to_post_loop_finalization = False
    try:
        from code.main import _current_trainer, _current_results_dir, convert_metrics_history_to_cpu

        if _current_trainer is not None and _current_results_dir is not None:
            # Loop is running; let it handle the save + exit. We just
            # flipped ``_shutdown_requested`` so the next iteration boundary
            # catches it. Return from the handler so the in-flight Python
            # statement continues to a safe interruption point.
            loop_will_handle_save = True
            logging.info(
                f"Deferring save to training loop's safe-point handler "
                f"(current completed_updates="
                f"{int(getattr(_current_trainer, 'completed_updates', 0) or 0)})"
            )
        else:
            # ``_current_trainer is None`` is overloaded: pre-init and post-finalize
            # look the same. Post-loop no-requeue is only safe once BOTH
            # ``_loop_finalized`` AND ``_final_checkpoint_persisted`` are True.
            # The window in between is a durability hole: a direct ``sys.exit`` here
            # would terminate before the post-loop code writes the final canonical
            # checkpoint. For both SIGUSR1 and SIGTERM, return from the handler
            # instead so that write can complete; the wrapper's post-return check
            # then exits 143 once persistence is durable.
            from code import main as _main
            loop_finalized = bool(getattr(_main, "_loop_finalized", False))
            final_persisted = bool(getattr(_main, "_final_checkpoint_persisted", False))
            skip_requeue_post_loop = loop_finalized and final_persisted
            defer_to_post_loop_finalization = loop_finalized and not final_persisted
            if skip_requeue_post_loop:
                logging.warning(
                    f"Training loop already finalized and final checkpoint "
                    f"persisted; {signal_name} arrived during post-loop work. "
                    f"Skipping scontrol requeue to avoid resurrecting a finished job."
                )
            elif defer_to_post_loop_finalization:
                logging.warning(
                    f"Training loop finalized but final canonical checkpoint "
                    f"NOT yet persisted; deferring {signal_name} to let "
                    f"post-loop code finish the final write before exit. "
                    f"The wrapper's post-return check will exit with the "
                    f"correct code once persistence completes."
                )
            else:
                logging.warning("Trainer not yet initialized - no checkpoint to save")
    except Exception as e:
        # If the probe raises we cannot determine whether a live loop exists. The
        # risky direction is falling through to a direct ``sys.exit``, which would
        # kill a live loop before its safe-point save. Prefer
        # ``loop_will_handle_save = True`` so the handler returns; the sticky
        # shutdown flag is already set, so a live loop services it at the next
        # iteration boundary, and a dead one exits via the wrapper's post-return
        # check.
        logging.error(
            f"Failed to consult trainer state in signal handler: {e}. "
            f"Defaulting to defer-to-loop to avoid killing a live loop "
            f"mid-iteration; ``_shutdown_requested`` is already set so "
            f"the loop will exit at its next safe-point check."
        )
        import traceback
        traceback.print_exc()
        loop_will_handle_save = True
        # If the probe raised on a SIGUSR1
        # path, no live loop might exist to fire the eventual
        # ``sys.exit(99)`` from the direct branch below. Pre-emptive
        # requeue here closes the durability hole — SLURM-level requeue
        # persists across any termination mode.
        if is_warning_signal:
            _fire_immediate_requeue(signal_name, "probe-exception fallback")

    # When the loop is going to handle save+exit, return from the handler
    # so the next iteration boundary picks up ``code.main._shutdown_requested``
    # and closes out cleanly. The signal context was already carried into
    # ``code.main`` via ``request_shutdown(is_warning_signal)`` above; no
    # local state to stash here.
    if loop_will_handle_save:
        return

    # Same return-without-exit pattern for the post-loop pre-final-write window:
    # terminating here would drop the completed-but-not-yet-persisted tail.
    # ``scontrol requeue`` fires IMMEDIATELY before returning for SIGUSR1, because
    # post-loop work can include long evaluator drains that overrun the grace window
    # — SLURM-level requeue persists across the follow-up SIGKILL, so the job stays
    # recoverable regardless of whether the final write completes.
    if defer_to_post_loop_finalization:
        if is_warning_signal:
            _fire_immediate_requeue(signal_name, "post-loop pre-final-write defer")
        return

    # Request requeue ONLY on the warning signal (SIGUSR1, via ``--signal=USR1@120``).
    # ``scancel`` sends SIGTERM, so requeuing on every signal would let a manually
    # cancelled job resurrect itself. ``scontrol requeue`` is idempotent and needs the
    # job to have been submitted with ``--requeue``. ``skip_requeue_post_loop``
    # short-circuits it for SIGUSR1 too: once training is finished and the final
    # checkpoint is durable, restarting wastes a slot and obscures completion.
    job_id = os.environ.get('SLURM_JOB_ID')
    if is_warning_signal and job_id and not skip_requeue_post_loop:
        try:
            result = subprocess.run(
                ['scontrol', 'requeue', job_id],
                capture_output=True, text=True, timeout=10, check=False,
            )
            if result.returncode == 0:
                logging.info(f"Explicit requeue requested for SLURM_JOB_ID={job_id}")
            else:
                logging.warning(
                    f"scontrol requeue exited {result.returncode} "
                    f"(falling back to RequeueExit policy if any): {result.stderr.strip()}"
                )
        except (FileNotFoundError, subprocess.TimeoutExpired) as e:
            logging.warning(f"Could not invoke scontrol requeue: {e}")
    elif not is_warning_signal:
        logging.info(f"Skipping scontrol requeue for {signal_name} (only SIGUSR1 triggers requeue)")

    # If we're about to take the DIRECT ``sys.exit`` path (the loop exited or never
    # started), drain any registered evaluators so a non-daemon child doesn't leak
    # past the SystemExit boundary — the wrapper's ``except Exception`` does not
    # catch ``SystemExit``.
    try:
        from code import main as _main
        _main.drain_registered_evaluators()
    except Exception as drain_err:
        logging.error(f"drain_registered_evaluators on signal-direct-exit failed: {drain_err}")

    # Distinct exit codes prevent ``RequeueExit=99`` from resurrecting a manually
    # cancelled job: 99 is only for the SIGUSR1 warning path, SIGTERM uses 143.
    # When post-loop no-requeue is authorized, even SIGUSR1 exits 143 — so neither
    # the explicit requeue nor a cluster ``RequeueExit=99`` policy can restart a
    # finished job.
    if is_warning_signal and not skip_requeue_post_loop:
        logging.info("Exiting (code 99 — preemption; scontrol requeue + RequeueExit fallback).")
        sys.exit(99)
    else:
        if is_warning_signal and skip_requeue_post_loop:
            logging.info(
                "Exiting (code 143 — SIGUSR1 after loop finalized + final "
                "checkpoint persisted; no requeue)."
            )
        else:
            logging.info("Exiting (code 143 — SIGTERM; no requeue).")
        sys.exit(143)  # 128 + SIGTERM (15) — conventional "terminated by signal"


def setup_signal_handlers():
    """Setup signal handlers for graceful preemption handling."""
    # SIGTERM: Sent when job is being terminated/preempted
    signal.signal(signal.SIGTERM, handle_preemption_signal)
    
    # SIGUSR1: Can be configured in SBATCH to be sent before timeout
    signal.signal(signal.SIGUSR1, handle_preemption_signal)
    
    logging.info("Signal handlers installed for graceful preemption handling")


async def main():
    parser = argparse.ArgumentParser(
        description="Run GFlowNet experiment from JSON configuration (SLURM compatible)"
    )
    parser.add_argument(
        "config",
        type=str,
        help="Path to JSON configuration file"
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Path to log file (in addition to stdout)"
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate configuration without running experiment"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_file)
    
    # Setup signal handlers for graceful preemption
    setup_signal_handlers()
    
    # Setup SLURM environment if on cluster
    setup_cluster_environment()
    
    # Log start time
    start_time = datetime.now()
    logging.info("=" * 60)
    logging.info("GFlowNet Experiment Runner")
    logging.info("=" * 60)
    logging.info(f"Start time: {start_time}")
    logging.info(f"Configuration file: {args.config}")
    
    # Verify config file exists
    config_path = Path(args.config)
    if not config_path.exists():
        logging.error(f"Configuration file not found: {args.config}")
        sys.exit(1)
    
    # Load configuration
    try:
        config = load_config_from_json(args.config)
        logging.info("Configuration loaded successfully")
        
        # CRITICAL: Force resume on requeued jobs to prevent creating new experiments
        restart_count = int(os.environ.get('SLURM_RESTART_COUNT', '0'))
        if restart_count > 0 and not config.resume:
            logging.info(f"\n{'='*60}")
            logging.info(f"REQUEUED JOB DETECTED (restart #{restart_count})")
            logging.info(f"Forcing resume=True to continue existing experiment")
            logging.info(f"{'='*60}\n")
            config.resume = True
        
        # Log key parameters
        logging.info("\nExperiment Parameters:")
        logging.info(f"  Hamiltonian: {config.hamiltonian_path}")
        logging.info(f"  Results directory: {config.results_dir}")
        logging.info(f"  Device preference: {config.device_preference}")
        logging.info(f"  Number of updates: {config.n_updates}")
        logging.info(f"  Measurements per batch: {config.n_measurements}")
        logging.info(f"  Max circuit depth: {config.max_depth}")
        logging.info(f"  Beta (temperature): {config.beta}")
        logging.info(f"  Learning rate: {config.lr}")
        logging.info(f"  Hidden dimension: {config.hidden_dim}")
        logging.info(f"  Hidden layers: {config.num_hidden_layers}")
        logging.info(f"  Reward type: {config.reward_type}")
        logging.info(f"  Objective: {config.objective_type}")
        logging.info(f"  Resume from checkpoint: {config.resume}")
        logging.info(f"  Async evaluation: {config.async_eval}")
        logging.info(f"  Large Hubbard mode: {config.large_hubbard_mode}")
        logging.info(f"  Evaluator mode: {config.evaluator_mode}")
        
    except Exception as e:
        logging.error(f"Error loading configuration: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Validate only mode
    if args.validate_only:
        logging.info("\nConfiguration validation successful!")
        sys.exit(0)
    
    # Run the experiment
    logging.info("\nStarting experiment...")
    try:
        results, trainer = await run_experiment(config)

        # Check if shutdown was requested during training. The canonical flag lives in
        # ``code.main``; a bare local reference would raise ``NameError``, be caught
        # by the broad except below, and turn a successful run into ``sys.exit(1)``.
        # Use the proxy so both module-load forms read the same value, and take the
        # exit code from the canonical helper so SIGTERM/manual-cancel paths aren't
        # routed to the requeue code.
        if _shutdown_requested_proxy():
            from code import main as _main
            loop_finalized = bool(getattr(_main, "_loop_finalized", False))
            final_persisted = bool(getattr(_main, "_final_checkpoint_persisted", False))
            if loop_finalized and final_persisted:
                # Shutdown was
                # requested but the loop already completed AND the final
                # checkpoint is durable. Don't take the requeue exit
                # code — restarting from the canonical checkpoint would
                # repeat completed work for no gain.
                logging.info(
                    "\nShutdown was requested AFTER the training loop finalized "
                    "AND the final checkpoint was persisted. Exiting 143 "
                    "(no requeue) — restart would only repeat completed work."
                )
                sys.exit(143)
            exit_code = _main.shutdown_exit_code()
            logging.info(
                f"\nShutdown was requested during training (exit code {exit_code}). "
                f"Checkpoint should have been saved by the loop's safe-point handler."
            )
            sys.exit(exit_code)

        # If the async evaluator hit a fatal error during training,
        # ``run_experiment`` records ``_evaluator_failure_info`` and writes a PARTIAL
        # report. Exit non-zero so cluster orchestration sees the run as
        # failed/partial. Exit code 4 (distinct from 99/143/130/1) so callers can
        # tell "training completed but eval was broken" from "training failed".
        evaluator_failure = getattr(trainer, "_evaluator_failure_info", None)
        if evaluator_failure and isinstance(evaluator_failure, dict) and evaluator_failure.get("value"):
            reasons = evaluator_failure.get("reasons", []) or ["(no reasons recorded)"]
            logging.error(
                "Run completed but async evaluator reported fatal errors. "
                "Reasons: %s. Exiting code 4 (PARTIAL).",
                "; ".join(str(r) for r in reasons),
            )
            sys.exit(4)
        
        # Log completion
        end_time = datetime.now()
        duration = end_time - start_time
        logging.info("\n" + "=" * 60)
        logging.info("Experiment completed successfully!")
        logging.info(f"End time: {end_time}")
        logging.info(f"Total duration: {duration}")
        logging.info("=" * 60)
        
    except KeyboardInterrupt:
        logging.info("\nExperiment interrupted by user")
        # Try to save a checkpoint using main.py's global reference. Use
        # ``completed_updates`` (bumped at end-of-iteration) — ``current_update``
        # would undercount when interrupted mid-iteration. SIGINT is intentionally
        # asymmetric with SIGTERM/SIGUSR1: no ``complete_shutdown_after_safe_save``
        # call, because Ctrl+C is a manual cancel that must never requeue. It is also
        # best-effort — a mid-iteration interrupt snapshots the previous label.
        try:
            from code.main import (
                _current_trainer,
                _current_results_dir,
                convert_metrics_history_to_cpu,
                safe_point_checkpoint_path,
                highest_existing_checkpoint_filename_hint,
            )
            if _current_trainer is not None and _current_results_dir is not None:
                # Single-pass lookup: a nested ``getattr(..., getattr)`` would always
                # evaluate the inner call, so a trainer ``__getattr__`` raising on
                # ``current_update`` would raise even when the outer name resolves.
                # Explicit ``is None`` checks rather than ``or`` so a legitimate
                # ``completed_updates=0`` is not demoted to ``current_update``,
                # mirroring ``_readable_candidate_update``.
                _cu = getattr(_current_trainer, 'completed_updates', None)
                if _cu is None:
                    _cu = getattr(_current_trainer, 'current_update', None)
                current_update = int(_cu if _cu is not None else 0)
                # Refuse-stale guard (sibling to the loop-side check). If an on-disk
                # checkpoint already advances past ``current_update``, writing this
                # interrupt snapshot is at best a no-op and at worst a confusing
                # post-mortem artifact — skip and log so resume takes the better file.
                # Filename-hint fast path with verify-on-block: trust filenames for
                # non-blocking emergencies, verify the payload only for those whose
                # hint would actually refuse this save.
                existing_update = highest_existing_checkpoint_filename_hint(
                    _current_results_dir, proposed_update=current_update
                )
                if existing_update > current_update:
                    logging.warning(
                        f"Skipping interrupt checkpoint write: an existing checkpoint "
                        f"reports update={existing_update}, ahead of current "
                        f"completed_updates={current_update}. Resume will pick the "
                        f"on-disk advanced checkpoint."
                    )
                else:
                    checkpoint_path = safe_point_checkpoint_path(
                        _current_results_dir, current_update, "INT"
                    )
                    metrics_history_cpu = convert_metrics_history_to_cpu(_current_trainer.metrics_history)
                    _current_trainer.gfn.save_checkpoint(str(checkpoint_path), current_update, metrics_history_cpu)
                    logging.info(
                        f"Interrupt checkpoint saved at update {current_update} "
                        f"to {checkpoint_path} before exit"
                    )
        except Exception as e:
            logging.error(f"Failed to save checkpoint on interrupt: {e}")
        # Drain any live evaluator processes before exit.
        try:
            from code import main as _main
            _main.drain_registered_evaluators()
        except Exception as drain_err:
            logging.error(f"Evaluator drain on interrupt failed: {drain_err}")
        # KeyboardInterrupt is a manual cancel — never trigger
        # ``RequeueExit=99``. Exit code 130 = 128 + SIGINT (2).
        sys.exit(130)
    except Exception as e:
        logging.error(f"\nError running experiment: {e}")
        import traceback
        traceback.print_exc()
        # Drain any live evaluator processes so a
        # ``run_experiment`` failure between evaluator start and the
        # normal cleanup block doesn't leak a non-daemon child.
        try:
            from code import main as _main
            _main.drain_registered_evaluators()
        except Exception as drain_err:
            logging.error(f"Evaluator drain on exception failed: {drain_err}")
        sys.exit(1)


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())
