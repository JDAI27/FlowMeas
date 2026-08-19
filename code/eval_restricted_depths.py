#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Restricted-depth evaluator.

Loads a FlowMeas checkpoint trained at ``max_depth=D_train`` and resamples
under restricted eval depths ``D_eval <= D_train``. Per eval depth it reports:

* training cost,
* canonical per-batch-element energy metrics: per-batch-element RMSE / MAE /
  mean energy / energy variance / bias arrays (length B), plus derived
  cross-batch aggregates (rmse_best, rmse_mean, rmse_top1, mae_best,
  top1_energy, best_of_K, mean_energy),
* circuit diagnostics (gate counts, actual 2q-depth distribution, unique
  circuit rate),
* a masked-policy-mass diagnostic that quantifies how much probability the
  unrestricted policy would have placed on actions that are illegal under the
  restricted eval cap.

The RMSE convention follows the repo-wide canonical (and the
 algorithm memo):

    rmse_b = sqrt((1/M) * sum_m (E_{b,m} - E_0)^2)

per outer batch element b over M shot simulations. This matches the value
``EnergyEstimator.estimate_energy_with_simulations`` returns as
``summary['rmse']``. The per-batch-element array is always saved; cross-batch
scalars (min / mean / rank-0) are derived from it and never replace it.

Direct CLI example:

    python -m code.eval_restricted_depths \\
        --experiment-dir results/profile_rorqual/H2O_STO3g_14qubits_md2/experiment_... \\
        --eval-max-depths 2 \\
        --batch-size 16 --n-measurements 1000 --n-simulations 500 \\
        --baseline policy \\
        --output restricted_depths_summary.json

The runner refuses ``--eval-max-depths`` values greater than the training
``max_depth`` (would extrapolate the policy beyond its support).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

try:
    from .quantum_action_mapping import build_action_mapping
except ImportError:
    from quantum_action_mapping import build_action_mapping

try:
    from .rmse_reporting import (
        canonical_metrics_from_energies,
        empty_canonical_metrics,
    )
except ImportError:
    from rmse_reporting import (
        canonical_metrics_from_energies,
        empty_canonical_metrics,
    )


# ----------------------------------------------------------------------------
# Action-index layout (mirrors quantum_action_mapping.build_action_mapping)
# ----------------------------------------------------------------------------
#   [0, 5n)               single-qubit Clifford gates
#   [5n, 5n + 2(n-1))     CNOT gates (forward + reverse per nearest-neighbor)
#   5n + 2(n-1)           terminal sentinel


def _action_layout(n_qubits: int) -> Tuple[int, int, int]:
    """Return ``(single_qubit_end, two_qubit_end, terminal_index)``.

    The three integers partition action indices into single-qubit / two-qubit /
    terminal regions. Keeping this derivation in one place avoids drifting away
    from ``quantum_action_mapping`` when the layout is extended later.
    """
    single_end = 5 * n_qubits
    two_end = single_end + 2 * max(n_qubits - 1, 0)
    terminal = two_end
    return single_end, two_end, terminal


# ----------------------------------------------------------------------------
# Run-metadata capture
# ----------------------------------------------------------------------------


def _capture_git_commit() -> Optional[str]:
    """Best-effort short git commit hash; ``None`` if not in a git tree."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short=12", "HEAD"],
            stderr=subprocess.DEVNULL,
        )
        commit = out.decode().strip()
        return commit or None
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _capture_job_id() -> Optional[str]:
    """SLURM job id from env if running under sbatch; ``None`` otherwise."""
    return os.environ.get("SLURM_JOB_ID")


def _infer_molecule_tag(experiment_dir: Path, hamiltonian_path: str) -> str:
    """Extract a short molecule tag from the experiment-dir or Hamiltonian path.

    Examples:

      results/profile_rorqual/H2O_STO3g_14qubits_md2/experiment_... → "H2O"
      results/full_run/NH3_STO3g_16qubits_md2/...                   → "NH3"
      Hamiltonians/H2_4qubits_pauli.txt                             → "H2"

    Falls back to the experiment-dir basename if no obvious tag is found, so
    the field is always populated for downstream readers.
    """
    candidates = [experiment_dir.parent.name, experiment_dir.name, hamiltonian_path]
    for c in candidates:
        m = re.match(
            r"^([A-Z][A-Za-z0-9]*?)"
            r"(?:_STO3g|_STO|_\d+qubits|_md\d|_\d+q)",
            Path(c).name,
        )
        if m:
            return m.group(1)
    return experiment_dir.name


# ----------------------------------------------------------------------------
# Diagnostic computations
# ----------------------------------------------------------------------------


def compute_circuit_diagnostics(
    actions: torch.Tensor,
    lengths: torch.Tensor,
    circuit_depths: torch.Tensor,
    n_qubits: int,
) -> Dict[str, Any]:
    """Summarize gate-type counts, 2q-depth histogram, and unique-circuit rate.

    Args:
        actions: ``(B, M, T)`` int64 action indices. Entries past ``lengths`` and
            terminal entries are ignored.
        lengths: ``(B, M)`` int64 trajectory lengths (terminal step inclusive).
        circuit_depths: ``(B, M)`` int64 actual 2q layer depth reached.
        n_qubits: System size.

    The "actual 2q depth" reported here is the depth tracked during sampling
    (``TrajectoryBatch.circuit_depths``) — i.e. the number of opened 2q layers,
    which is the depth semantic used by ``max_depth`` masking.
    """
    single_end, two_end, terminal = _action_layout(n_qubits)

    actions_cpu = actions.detach().to("cpu")
    lengths_cpu = lengths.detach().to("cpu")
    depths_cpu = circuit_depths.detach().to("cpu")

    B, M, T = actions_cpu.shape
    step_idx = torch.arange(T).view(1, 1, T).expand(B, M, T)
    pre_terminal = step_idx < (lengths_cpu - 1).unsqueeze(-1).clamp(min=0)
    is_terminal = actions_cpu == terminal
    real_gates = pre_terminal & ~is_terminal

    is_single = (actions_cpu >= 0) & (actions_cpu < single_end) & real_gates
    is_two = (actions_cpu >= single_end) & (actions_cpu < two_end) & real_gates

    single_count = int(is_single.sum().item())
    two_count = int(is_two.sum().item())
    total_gates = single_count + two_count

    depth_values = depths_cpu.reshape(-1).tolist()
    depth_hist = Counter(depth_values)
    depth_hist_sorted = {str(k): depth_hist[k] for k in sorted(depth_hist)}

    seen = set()
    n_circuits = B * M
    for b in range(B):
        for m in range(M):
            L = int(lengths_cpu[b, m].item())
            seq = tuple(actions_cpu[b, m, :L].tolist())
            seen.add(seq)
    unique_rate = (len(seen) / n_circuits) if n_circuits > 0 else 0.0

    return {
        "total_gates": total_gates,
        "single_qubit_gate_count": single_count,
        "cnot_count": two_count,
        "actual_2q_depth_histogram": depth_hist_sorted,
        "actual_2q_depth_mean": float(np.mean(depth_values)) if depth_values else 0.0,
        "actual_2q_depth_max": int(max(depth_values)) if depth_values else 0,
        "n_circuits": n_circuits,
        "n_unique_circuits": len(seen),
        "unique_circuit_rate": float(unique_rate),
    }


def compute_masked_policy_mass(
    gfn,
    eval_max_depth: int,
    training_max_depth: int,
    batch_size: int,
    n_measurements: int,
) -> Dict[str, Any]:
    """Estimate the policy-mass diversion onto eval-illegal actions at step 0.

    The diagnostic is policy-based, not sampler-based: it reads the trained
    ``pf_model`` distribution and asks how much mass the eval mask removes.
    The result is independent of which sampling baseline is run (policy or
    random), and matches the memo's mask-before-sample contract.

    Returns 0 by construction when ``eval_max_depth >= training_max_depth``
    (no action is "newly illegal").
    """
    try:
        from .GFNs import TrajectoryBatch
    except ImportError:
        from GFNs import TrajectoryBatch  # type: ignore[no-redef]

    if eval_max_depth >= training_max_depth:
        return {
            "eval_max_depth": eval_max_depth,
            "training_max_depth": training_max_depth,
            "initial_state_mass_on_illegal_actions": 0.0,
            "max_initial_state_mass_on_illegal_actions": 0.0,
            "note": (
                "eval_max_depth >= training_max_depth — no actions become "
                "newly-illegal under the eval cap; mass is 0 by construction."
            ),
        }

    device = gfn.device
    n_qubits = gfn.n_qubits
    max_length = gfn.determine_buffer_size(training_max_depth)

    traj = TrajectoryBatch(
        batch_size=batch_size,
        n_measurements=n_measurements,
        max_length=max_length,
        n_qubits=n_qubits,
        device=device,
    )
    batched_tableau = gfn._tableau_cls(
        n_qubits=n_qubits,
        batch_size=batch_size,
        n_measurements=n_measurements,
        device=str(device),
    )
    gfn._configure_sampling_tableau(batched_tableau)
    traj.batched_tableau = batched_tableau

    with torch.no_grad():
        states_tensor, indices = batched_tableau.to_flat_tensors_active_only()
        if states_tensor.shape[0] == 0:
            return {
                "eval_max_depth": eval_max_depth,
                "training_max_depth": training_max_depth,
                "initial_state_mass_on_illegal_actions": 0.0,
                "max_initial_state_mass_on_illegal_actions": 0.0,
                "note": "No active rows at step 0 — diagnostic skipped.",
            }
        if isinstance(indices, torch.Tensor):
            indices_tensor = indices.to(device)
        else:
            indices_tensor = torch.as_tensor(indices, dtype=torch.long, device=device)

        mask_train = gfn.masking_engine.compute_action_masks_active_gpu(
            traj, indices_tensor, max_depth=training_max_depth
        )
        mask_eval = gfn.masking_engine.compute_action_masks_active_gpu(
            traj, indices_tensor, max_depth=eval_max_depth
        )

        logits = gfn.pf_model(states_tensor)
        neg_inf = torch.finfo(logits.dtype).min
        masked_logits_train = logits.masked_fill(~mask_train, neg_inf)
        # Fail-loud guard: if any row has all -inf logits (no valid action),
        # softmax would produce NaN. Surface that immediately instead of
        # propagating a corrupted distribution into the diagnostic.
        if not torch.isfinite(masked_logits_train.max(dim=-1).values).all():
            raise RuntimeError(
                "compute_masked_policy_mass: at least one row has no valid "
                "actions under the training mask. This should never happen "
                "at the empty initial state — check the masking engine."
            )
        probs_train = torch.softmax(masked_logits_train, dim=-1)
        # "Newly illegal" = legal at training cap but blocked at eval cap.
        newly_illegal = mask_train & ~mask_eval
        diverted = (probs_train * newly_illegal.float()).sum(dim=-1)
        mean_diverted = float(diverted.mean().item())
        max_diverted = float(diverted.max().item())

    return {
        "eval_max_depth": eval_max_depth,
        "training_max_depth": training_max_depth,
        "initial_state_mass_on_illegal_actions": mean_diverted,
        "max_initial_state_mass_on_illegal_actions": max_diverted,
        "note": (
            "Probability the unrestricted (training-cap) policy assigns to "
            "actions blocked by the eval cap, averaged over active rows at "
            "the initial empty-circuit state."
        ),
    }


# ----------------------------------------------------------------------------
# Trainer + checkpoint loading
# ----------------------------------------------------------------------------


def _load_experiment_config(experiment_dir: Path):
    """Reconstruct an ``ExperimentConfig`` from a saved experiment directory."""
    try:
        from .main import ExperimentConfig
    except ImportError:
        from main import ExperimentConfig

    config_path = experiment_dir / "config.json"
    with open(config_path, "r") as f:
        config_dict = json.load(f)
    return ExperimentConfig(**config_dict)


def _build_trainer(config, hamiltonian_helper, device_preference: Optional[str]):
    """Build an ``EfficientGFNTrainer`` whose architecture matches ``config``.

    Mirrors the gfn_config construction in ``main.run_experiment`` so the loaded
    checkpoint weights are guaranteed to map onto a layout-compatible network.
    """
    try:
        from .main import EfficientGFNTrainer  # noqa: F401
        from .GFNs import (  # noqa: F401
            default_reward_fn,
            exponential_reward_fn,
            log_reward_fn,
            EfficientGFNTrainer as _Trainer,
        )
    except ImportError:
        from main import EfficientGFNTrainer as _Trainer  # noqa: F401
        from GFNs import (  # noqa: F401
            default_reward_fn,
            exponential_reward_fn,
            log_reward_fn,
        )

    reward_fn_map = {
        "exp": exponential_reward_fn,
        "default": default_reward_fn,
        "log": log_reward_fn,
    }
    reward_fn = reward_fn_map.get(config.reward_type)

    identity_term = "I" * hamiltonian_helper.n_qubits
    training_pauli_strings: List[str] = []
    training_weights: List[float] = []
    identity_weight = 0.0
    for pauli_str, weight in zip(hamiltonian_helper.pauli_str_list, hamiltonian_helper.w_list):
        if pauli_str == identity_term:
            identity_weight = weight.real
        else:
            training_pauli_strings.append(pauli_str)
            training_weights.append(weight.real)

    gfn_config = {
        "model": {
            "model_type": config.model_type,
            "hidden_dim": config.hidden_dim,
            "num_hidden_layers": config.num_hidden_layers,
            "lr": config.lr,
            "weight_decay": config.weight_decay,
            "model_kwargs": {},
            "objective_type": config.objective_type,
            "objective_kwargs": config.objective_kwargs,
            "measurement_backend": config.measurement_backend,
            # sampling_mode must be forwarded too, else a bucketed-trained
            # checkpoint silently evaluates with the default dynamic sampler
            # (review: missed sibling site).
            "sampling_mode": getattr(config, "sampling_mode", None),
            "static_shape_sampling": config.static_shape_sampling,
            "cuda_graph_sampling": config.cuda_graph_sampling,
            "use_fused_metadata_kernel": config.use_fused_metadata_kernel,
            "use_fused_sampling_kernel": config.use_fused_sampling_kernel,
            "use_fused_mask_counts_kernel": config.use_fused_mask_counts_kernel,
            "use_fused_counter_rng_kernel": getattr(
                config, "use_fused_counter_rng_kernel", True
            ),
            "use_fused_partition_update_kernel": getattr(
                config, "use_fused_partition_update_kernel", True
            ),
            "use_fused_apply_kernel": config.use_fused_apply_kernel,
            "debug": False,
        },
        "training": {
            "beta": config.beta,
            "n_measurements": config.n_measurements,
            "update_freq": config.update_freq,
            "max_depth": config.max_depth,
            "K": config.n_eval_top_k_batch_elements,
            "reward_kwargs": config.reward_kwargs,
            "cost": {"type": config.cost_type, **config.cost_kwargs},
        },
        "quantum": {
            "pauli_str_list": training_pauli_strings,
            "w_list": training_weights,
        },
    }

    trainer = _Trainer(
        gfn_config,
        reward_fn=reward_fn,
        device_preference=device_preference or config.device_preference,
        metric_store=None,
    )
    return trainer, training_pauli_strings, identity_weight


def _pick_checkpoint(experiment_dir: Path) -> Path:
    candidates = sorted(experiment_dir.glob("checkpoint_update*.pth"))
    if not candidates:
        raise FileNotFoundError(f"No checkpoint_update*.pth in {experiment_dir}")
    canonical = experiment_dir / "checkpoint_update.pth"
    if canonical.exists():
        return canonical
    return candidates[-1]


# ----------------------------------------------------------------------------
# Per-depth evaluation
# ----------------------------------------------------------------------------


def _sample_at_depth(
    trainer,
    eval_max_depth: int,
    batch_size: int,
    n_measurements: int,
    sampling_mode: str = "policy",
):
    """Sample one batch at the given depth cap (mask-before-sample inside GFN)."""
    try:
        from .GFNs import SamplingMode
    except ImportError:
        from GFNs import SamplingMode

    mode_map = {"policy": SamplingMode.ON_POLICY, "random": SamplingMode.OFF_POLICY}
    if sampling_mode not in mode_map:
        raise ValueError(
            f"sampling_mode must be one of {list(mode_map)}, got {sampling_mode!r}"
        )

    traj = trainer.gfn.sample_trajectories(
        batch_size=batch_size,
        n_measurements=n_measurements,
        max_depth=eval_max_depth,
        mode=mode_map[sampling_mode],
        cache_for_flows=False,
    )
    return traj


def _compute_training_cost(trainer, traj) -> torch.Tensor:
    return trainer.compute_costs_with_probabilities(traj.batched_tableau)


def _summarize_training_cost(costs: torch.Tensor) -> Dict[str, float]:
    c = costs.detach().to("cpu")
    return {
        "cost_mean": float(c.mean().item()),
        "cost_min": float(c.min().item()),
        "cost_max": float(c.max().item()),
    }


def _compute_canonical_energy_metrics(
    energy_estimator,
    traj,
    n_simulations: int,
) -> Dict[str, Any]:
    """Per-batch-element energy metrics in the canonical schema.

    Thin wrapper: computes the ``(B, M)`` energy tensor via
    ``EnergyEstimator.estimate_energy_tensor`` and delegates the schema (field
    names + formulas) to:mod:`code.rmse_reporting`, the single repo-wide
    definition. When ``energy_estimator`` is ``None`` (scalable_large mode),
    returns the empty/None canonical dict — consumers must treat that as "not
    measured", never as zero.
    """
    if energy_estimator is None:
        return empty_canonical_metrics()

    M = max(int(n_simulations), 1)
    energies = energy_estimator.estimate_energy_tensor(
        traj.actions, traj.lengths, M=M
    )  # (B, M)
    return canonical_metrics_from_energies(
        energies, energy_estimator.ground_state_energy
    )


def evaluate_one_depth(
    trainer,
    energy_estimator,
    eval_max_depth: int,
    training_max_depth: int,
    batch_size: int,
    n_measurements: int,
    n_simulations: int,
    sampling_mode: str,
    molecule: str,
    n_qubits: int,
) -> Dict[str, Any]:
    """Sample at ``eval_max_depth`` and return a flat per-depth result dict.

    Schema matches the algorithm-memo "Required output schema" minus
    the ``source_checkpoint`` block (added by the caller).
    """
    import time as _time
    if eval_max_depth > training_max_depth:
        raise ValueError(
            f"eval_max_depth={eval_max_depth} exceeds training_max_depth="
            f"{training_max_depth}; the policy was never trained against deeper "
            "circuits and the resulting distribution is undefined."
        )

    peak_mem_bytes = 0
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    _t = _time.time()
    traj = _sample_at_depth(
        trainer, eval_max_depth, batch_size, n_measurements, sampling_mode=sampling_mode
    )
    logging.info(
        f"  [step] sample_trajectories ({sampling_mode}): {_time.time()-_t:.1f}s"
    )

    _t = _time.time()
    cost_metrics = _summarize_training_cost(_compute_training_cost(trainer, traj))
    logging.info(f"  [step] compute_costs: {_time.time()-_t:.1f}s")

    _t = _time.time()
    energy_metrics = _compute_canonical_energy_metrics(
        energy_estimator, traj, n_simulations
    )
    logging.info(f"  [step] estimate_energy (M={n_simulations}): {_time.time()-_t:.1f}s")

    _t = _time.time()
    circuit_diag = compute_circuit_diagnostics(
        traj.actions, traj.lengths, traj.circuit_depths, n_qubits=trainer.gfn.n_qubits
    )
    logging.info(f"  [step] circuit_diagnostics: {_time.time()-_t:.1f}s")

    _t = _time.time()
    masked_mass = compute_masked_policy_mass(
        trainer.gfn,
        eval_max_depth=eval_max_depth,
        training_max_depth=training_max_depth,
        batch_size=batch_size,
        n_measurements=n_measurements,
    )
    logging.info(f"  [step] masked_policy_mass: {_time.time()-_t:.1f}s")

    if torch.cuda.is_available():
        peak_mem_bytes = int(torch.cuda.max_memory_allocated())
        logging.info(
            f"  [mem] peak_gpu_alloc={peak_mem_bytes / 1024**3:.2f} GiB "
            f"(B={batch_size} n_meas={n_measurements} n_sims={n_simulations} "
            f"eval_md={eval_max_depth})"
        )

    return {
        # Run identity
        "molecule": molecule,
        "n_qubits": int(n_qubits),
        "train_max_depth": int(training_max_depth),
        "eval_max_depth": int(eval_max_depth),
        "baseline": sampling_mode,
        "B": int(batch_size),
        "n_measurements": int(n_measurements),
        "n_simulations": int(n_simulations),
        # Training cost
        "cost_mean": cost_metrics["cost_mean"],
        "cost_min": cost_metrics["cost_min"],
        "cost_max": cost_metrics["cost_max"],
        # Energy (canonical per-batch-element + derived aggregates)
        **energy_metrics,
        # Circuit diagnostics
        "total_gate_count": circuit_diag["total_gates"],
        "single_qubit_gate_count": circuit_diag["single_qubit_gate_count"],
        "cnot_count": circuit_diag["cnot_count"],
        "actual_2q_depth_mean": circuit_diag["actual_2q_depth_mean"],
        "actual_2q_depth_max": circuit_diag["actual_2q_depth_max"],
        "actual_2q_depth_hist": circuit_diag["actual_2q_depth_histogram"],
        "n_circuits": circuit_diag["n_circuits"],
        "n_unique_circuits": circuit_diag["n_unique_circuits"],
        "unique_circuit_rate": circuit_diag["unique_circuit_rate"],
        # Mask diagnostics
        "initial_illegal_policy_mass": masked_mass["initial_state_mass_on_illegal_actions"],
        "initial_illegal_policy_mass_max": masked_mass.get(
            "max_initial_state_mass_on_illegal_actions"
        ),
        # Resources
        "peak_gpu_alloc_gib": (
            peak_mem_bytes / 1024**3 if torch.cuda.is_available() else None
        ),
    }


# ----------------------------------------------------------------------------
# Top-level entrypoint
# ----------------------------------------------------------------------------


def evaluate_restricted_depths(
    experiment_dir: Path,
    eval_max_depths: List[int],
    batch_size: Optional[int] = None,
    n_measurements: Optional[int] = None,
    n_simulations: Optional[int] = None,
    output_path: Optional[Path] = None,
    device_preference: Optional[str] = None,
    sampling_mode: str = "policy",
    seed: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Evaluate one checkpoint at multiple eval depths.

    Returns a list of flat per-depth result dicts (one per requested depth).
    Each dict carries the full memo schema, with ``source_checkpoint``
    nested for traceability.

    When ``output_path`` is given:

    * one eval depth (canonical sbatch usage): writes one flat JSON to
      ``output_path``.
    * multiple eval depths: writes one flat JSON per depth, named
      ``<output_stem>__md{D}.json`` next to ``output_path`` if it has a
      ``.json`` suffix, or inside ``output_path`` if it has no suffix
      (treated as a directory).
    """
    try:
        from .pauli_hamiltonian_helper import PauliHamiltonianHelper
        from .energy_estimator import EnergyEstimator
        from .main import get_evaluator_mode_metadata
    except ImportError:
        from pauli_hamiltonian_helper import PauliHamiltonianHelper
        from energy_estimator import EnergyEstimator
        from main import get_evaluator_mode_metadata

    experiment_dir = Path(experiment_dir)
    config = _load_experiment_config(experiment_dir)
    training_max_depth = config.max_depth

    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    if n_measurements is None:
        n_measurements = int(config.n_measurements)
    if n_simulations is None:
        n_simulations = int(getattr(config, "n_simulations", 1))
    if batch_size is None:
        batch_size = int(getattr(config, "update_freq", 4))
    logging.info(
        f"[knobs] batch_size={batch_size} n_measurements={n_measurements} "
        f"n_simulations={n_simulations} (defaults filled from training config; "
        f"override via CLI flags if needed)"
    )

    for d in eval_max_depths:
        if d < 0:
            raise ValueError(f"eval_max_depth must be >= 0, got {d}")
        if d > training_max_depth:
            raise ValueError(
                f"eval_max_depth={d} exceeds training max_depth={training_max_depth}"
            )

    import time as _time
    _t0 = _time.time()
    logging.info(f"[phase] loading hamiltonian: {config.hamiltonian_path}")
    hamiltonian_helper = PauliHamiltonianHelper(config.hamiltonian_path)
    logging.info(
        f"[phase] hamiltonian loaded in {_time.time()-_t0:.1f}s "
        f"({len(hamiltonian_helper.pauli_str_list)} paulis)"
    )

    _t1 = _time.time()
    logging.info("[phase] building trainer")
    trainer, _, _ = _build_trainer(config, hamiltonian_helper, device_preference)
    logging.info(f"[phase] trainer built in {_time.time()-_t1:.1f}s")

    _t2 = _time.time()
    checkpoint_path = _pick_checkpoint(experiment_dir)
    logging.info(f"[phase] loading checkpoint {checkpoint_path.name}")
    update, _ = trainer.gfn.load_checkpoint(str(checkpoint_path))
    logging.info(f"[phase] checkpoint loaded in {_time.time()-_t2:.1f}s at update {update}")

    energy_estimator: Optional[EnergyEstimator] = None
    if get_evaluator_mode_metadata(config)["allows_full_state_evaluation"]:
        _t3 = _time.time()
        logging.info("[phase] building EnergyEstimator")
        energy_estimator = EnergyEstimator(
            hamiltonian_helper=hamiltonian_helper,
            n_qubits=hamiltonian_helper.n_qubits,
            device=trainer.device,
            measurement_backend=config.measurement_backend,
        )
        logging.info(f"[phase] EnergyEstimator built in {_time.time()-_t3:.1f}s")

    molecule = _infer_molecule_tag(experiment_dir, config.hamiltonian_path)
    n_qubits = int(hamiltonian_helper.n_qubits)

    source_checkpoint = {
        "experiment_dir": str(experiment_dir),
        "checkpoint_path": str(checkpoint_path),
        "update": int(update),
        "training_max_depth": int(training_max_depth),
        "hamiltonian_path": config.hamiltonian_path,
        "n_qubits": n_qubits,
        "measurement_backend": config.measurement_backend,
        "molecule": molecule,
        "git_commit": _capture_git_commit(),
        "job_id": _capture_job_id(),
        "seed": seed,
    }

    results: List[Dict[str, Any]] = []
    for eval_depth in eval_max_depths:
        _td = _time.time()
        logging.info(
            f"[phase] evaluating eval_max_depth={eval_depth} mode={sampling_mode}"
        )
        result = evaluate_one_depth(
            trainer=trainer,
            energy_estimator=energy_estimator,
            eval_max_depth=eval_depth,
            training_max_depth=training_max_depth,
            batch_size=batch_size,
            n_measurements=n_measurements,
            n_simulations=n_simulations,
            sampling_mode=sampling_mode,
            molecule=molecule,
            n_qubits=n_qubits,
        )
        result["source_checkpoint"] = source_checkpoint
        results.append(result)
        logging.info(f"[phase] eval_max_depth={eval_depth} done in {_time.time()-_td:.1f}s")

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if len(results) == 1:
            with open(output_path, "w") as f:
                json.dump(results[0], f, indent=2)
            logging.info(f"Wrote {output_path}")
        elif output_path.suffix == ".json":
            stem = output_path.with_suffix("")
            for r in results:
                p = Path(f"{stem}__md{r['eval_max_depth']}.json")
                with open(p, "w") as f:
                    json.dump(r, f, indent=2)
                logging.info(f"Wrote {p}")
        else:
            output_path.mkdir(parents=True, exist_ok=True)
            for r in results:
                p = output_path / f"eval_md{r['eval_max_depth']}__{r['baseline']}.json"
                with open(p, "w") as f:
                    json.dump(r, f, indent=2)
                logging.info(f"Wrote {p}")

    return results


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        required=True,
        help="Path to a FlowMeas experiment directory containing config.json "
        "and checkpoint_update*.pth",
    )
    parser.add_argument(
        "--eval-max-depths",
        type=int,
        nargs="+",
        default=[2, 1, 0],
        help="Eval max_depth values to sweep (each must be <= training max_depth)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Defaults to the training config's ``update_freq`` if omitted",
    )
    parser.add_argument(
        "--n-measurements",
        type=int,
        default=None,
        help="Defaults to the training config's ``n_measurements`` if omitted",
    )
    parser.add_argument(
        "--n-simulations",
        type=int,
        default=None,
        help="Defaults to the training config's ``n_simulations`` if omitted",
    )
    parser.add_argument("--output", type=Path, default=None, help="JSON output path")
    parser.add_argument(
        "--device-preference",
        type=str,
        default=None,
        help="Override device (e.g. 'cpu', 'cuda:0'); default uses config",
    )
    parser.add_argument(
        "--baseline",
        choices=["policy", "random"],
        default="policy",
        help="'policy' samples from the trained pf_model (mask-before-sample); "
        "'random' samples uniformly over valid actions at each step. The eval "
        "validity mask is identical in both modes.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional RNG seed for torch / numpy; persisted in source_checkpoint.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    evaluate_restricted_depths(
        experiment_dir=args.experiment_dir,
        eval_max_depths=args.eval_max_depths,
        batch_size=args.batch_size,
        n_measurements=args.n_measurements,
        n_simulations=args.n_simulations,
        output_path=args.output,
        device_preference=args.device_preference,
        sampling_mode=args.baseline,
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
