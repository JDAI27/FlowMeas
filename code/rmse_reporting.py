"""Canonical per-batch-element RMSE reporting.

ONE definition of the energy-RMSE reporting schema for every code path in
FlowMeas. Any code that reports RMSE must build its output through this
module so the field names and formulas live in a single place.

Canonical convention (see the algorithm memo)::

    rmse_b = sqrt((1/M) * sum_{m=0..M-1} (E_{b,m} - E0)^2)

per batch element ``b``, over ``M`` independent shot simulations, versus the
exact / DMRG reference ``E0``. This is the value
``EnergyEstimator.estimate_energy_with_simulations`` returns as
``summary['rmse']`` for batch element ``b``.

Cross-batch aggregates are **derived** and must never replace the per-batch-element
arrays in the output::

    rmse_best = min_b rmse_b
    rmse_mean = (1/B) sum_b rmse_b
    rmse_top1 = rmse_0     # only meaningful when the outer batch is ranked

Reporting rule: every writer persists all five per-batch-element arrays, each of
length ``B``. Scalar aggregates may be saved alongside, never alone.

Two builders cover the two ways callers already have the data:

*:func:`canonical_metrics_from_energies` — from the raw ``(B, M)`` energy tensor
  (the ``estimate_energy_tensor`` path; computes everything incl. ``best_of_K``).
*:func:`canonical_metrics_from_summaries` — from the per-batch-element summary
  dicts returned by ``estimate_energy_with_simulations`` (rmse/mae/mean/var per
  element already computed; individual-shot energies are not retained, so
  ``best_of_K`` is reported as ``None``).

:func:`assert_canonical_rmse_schema` validates a metrics dict and is used by both
the writers and the tests so a future drift fails loudly.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

# The five per-batch-element arrays every RMSE-reporting code path MUST persist,
# each of length B. Change a name here and every writer/reader follows.
PER_BATCH_FIELDS = (
    "per_batch_element_rmse",
    "per_batch_element_mae",
    "per_batch_element_mean_energy",
    "per_batch_element_energy_variance",
    "per_batch_element_bias",
)

# Derived cross-batch aggregates. ``rmse_best``/``rmse_mean``/``rmse_top1`` and
# ``reference_energy`` are mandatory; the energy aggregates are best-effort and
# may be ``None`` (e.g. ``best_of_K`` is unavailable from per-element summaries).
MANDATORY_AGGREGATE_FIELDS = ("rmse_best", "rmse_mean", "rmse_top1", "reference_energy")
OPTIONAL_AGGREGATE_FIELDS = ("mean_energy", "mae_best", "top1_energy", "best_of_K")

_ABS_TOL = 1e-6
_REL_TOL = 1e-6


def empty_canonical_metrics() -> Dict[str, Any]:
    """The canonical dict for the no-measurement case (e.g. scalable_large mode).

    All per-batch arrays empty and aggregates ``None`` — downstream consumers
    must treat these as "not measured", never as zero.
    """
    out: Dict[str, Any] = {f: [] for f in PER_BATCH_FIELDS}
    for f in MANDATORY_AGGREGATE_FIELDS + OPTIONAL_AGGREGATE_FIELDS:
        out[f] = None
    return out


def _to_cpu_tensor(energies_bm):
    """Accept a torch tensor (preferred) or array-like; return a CPU float tensor."""
    import torch

    if isinstance(energies_bm, torch.Tensor):
        return energies_bm.detach().to("cpu")
    return torch.as_tensor(energies_bm, dtype=torch.float64).to("cpu")


def canonical_metrics_from_energies(energies_bm, reference_energy: float) -> Dict[str, Any]:
    """Canonical metrics from the raw ``(B, M)`` per-shot energy tensor.

    ``energies_bm[b, m]`` is the energy of shot ``m`` for batch element ``b``.
    Formulas match ``EnergyEstimator.estimate_energy_with_simulations`` exactly
    (population mean of squared errors for rmse; ``ddof=1`` for the variance).
    """
    import torch

    energies = _to_cpu_tensor(energies_bm)
    if energies.ndim != 2:
        raise ValueError(f"energies must be (B, M); got shape {tuple(energies.shape)}")
    B, M = int(energies.shape[0]), int(energies.shape[1])
    E0 = float(reference_energy)
    if B == 0:
        out = empty_canonical_metrics()
        out["reference_energy"] = E0
        return out

    sq_err = (energies - E0) ** 2
    per_rmse = torch.sqrt(sq_err.mean(dim=-1))           # (B,)
    per_mae = (energies - E0).abs().mean(dim=-1)         # (B,)
    per_mean = energies.mean(dim=-1)                     # (B,)
    per_bias = per_mean - E0                             # (B,)
    if M > 1:
        per_var = energies.var(dim=-1, unbiased=True)    # ddof=1
    else:
        per_var = torch.zeros_like(per_mean)

    return {
        "per_batch_element_rmse": per_rmse.tolist(),
        "per_batch_element_mae": per_mae.tolist(),
        "per_batch_element_mean_energy": per_mean.tolist(),
        "per_batch_element_energy_variance": per_var.tolist(),
        "per_batch_element_bias": per_bias.tolist(),
        "mean_energy": float(per_mean.mean().item()),
        "rmse_best": float(per_rmse.min().item()),
        "rmse_mean": float(per_rmse.mean().item()),
        # rmse_top1: rank-0 batch element. Meaningful only when the outer batch is
        # ranked (e.g. top_trajectories[:K]); for independent draws it is simply
        # "the first sampled element", saved for schema parity.
        "rmse_top1": float(per_rmse[0].item()),
        "mae_best": float(per_mae.min().item()),
        "top1_energy": float(per_mean.min().item()),
        "best_of_K": float(energies.min().item()),
        "reference_energy": E0,
    }


def canonical_metrics_from_summaries(
    summaries: Sequence[Dict[str, Any]], reference_energy: float
) -> Dict[str, Any]:
    """Canonical metrics from per-batch-element summary dicts.

    ``summaries`` is the list returned by
    ``EnergyEstimator.estimate_energy_with_simulations`` (one dict per batch
    element, each carrying ``rmse``, ``mae``, ``mean_energy``,
    ``energy_variance``). The individual-shot energies are not retained in the
    summaries, so ``best_of_K`` is ``None`` here — use
:func:`canonical_metrics_from_energies` when the raw ``(B, M)`` tensor is
    available and the global shot minimum is needed.
    """
    E0 = float(reference_energy)
    if not summaries:
        out = empty_canonical_metrics()
        out["reference_energy"] = E0
        return out

    per_rmse: List[float] = [float(s["rmse"]) for s in summaries]
    per_mae: List[float] = [float(s["mae"]) for s in summaries]
    per_mean: List[float] = [float(s["mean_energy"]) for s in summaries]
    per_var: List[float] = [float(s["energy_variance"]) for s in summaries]
    per_bias: List[float] = [m - E0 for m in per_mean]

    return {
        "per_batch_element_rmse": per_rmse,
        "per_batch_element_mae": per_mae,
        "per_batch_element_mean_energy": per_mean,
        "per_batch_element_energy_variance": per_var,
        "per_batch_element_bias": per_bias,
        "mean_energy": sum(per_mean) / len(per_mean),
        "rmse_best": min(per_rmse),
        "rmse_mean": sum(per_rmse) / len(per_rmse),
        "rmse_top1": per_rmse[0],
        "mae_best": min(per_mae),
        "top1_energy": min(per_mean),
        "best_of_K": None,  # individual-shot energies not retained in summaries
        "reference_energy": E0,
    }


def assert_canonical_rmse_schema(
    metrics: Dict[str, Any], *, expected_B: Optional[int] = None
) -> None:
    """Validate a metrics dict against the canonical schema. Raises on violation.

    Checks: all five per-batch arrays present and equal length (== ``expected_B``
    if given); mandatory aggregates present; and when non-empty, ``rmse_best`` /
    ``rmse_mean`` / ``rmse_top1`` equal min / mean / element-0 of the array.
    Used by writers (as a self-check) and by the test suite.
    """
    missing = [f for f in PER_BATCH_FIELDS if f not in metrics]
    if missing:
        raise ValueError(
            f"canonical RMSE schema violation: missing per-batch arrays {missing}. "
            f"Build output via code.rmse_reporting."
        )
    lengths = {f: len(metrics[f]) for f in PER_BATCH_FIELDS}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"per-batch arrays have unequal lengths: {lengths}")
    B = next(iter(lengths.values()))
    if expected_B is not None and B != expected_B:
        raise ValueError(f"per-batch arrays have length {B}, expected {expected_B}")

    for f in MANDATORY_AGGREGATE_FIELDS:
        if f not in metrics:
            raise ValueError(f"canonical RMSE schema violation: missing aggregate '{f}'")

    if B == 0:
        return  # no-measurement case: aggregates may be None

    arr = metrics["per_batch_element_rmse"]
    for name, expected in (
        ("rmse_best", min(arr)),
        ("rmse_mean", sum(arr) / len(arr)),
        ("rmse_top1", arr[0]),
    ):
        got = metrics[name]
        if got is None or abs(float(got) - expected) > _ABS_TOL + _REL_TOL * abs(expected):
            raise ValueError(
                f"canonical RMSE schema violation: {name}={got} but expected "
                f"{expected} (= derived from per_batch_element_rmse)."
            )
