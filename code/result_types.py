# -*- coding: utf-8 -*-
"""Shared result-codec types for FlowMeas experiment results.

Extracted from main.py (which keeps a backward-compatible facade re-export).
A LEAF module: it imports only stdlib + numpy + energy_estimator (for the
BatchElementEnergyResult base class) and NEVER imports main/config, so the
COLD reporting/evaluator extractions can source these shared types here without
creating an import cycle with main.
"""
import json
import logging
import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

try:
    from .energy_estimator import BatchElementEnergyResult
except ImportError:  # direct-execution mode (python code/X.py)
    from energy_estimator import BatchElementEnergyResult


@dataclass
class SimulationResult:
    """Results from multiple simulation runs.

    ``mean_absolute_error`` (MAE) and ``rmse`` are *different*
    aggregates — MAE is the L1 mean, RMSE is the L2 mean. Not drop-in
    replacements. Pick whichever matches what you are reporting.
    """
    energy_estimates: List[float]
    absolute_errors: List[float]
    rmse: float  # Root Mean Squared Error: sqrt(mean((E_m - E*)^2))
    std_absolute_error: float  # Sample std of |E_m - E*| across the M sims (ddof=1; 0 at M=1)
    mean_energy_estimate: float
    std_energy_estimate: float

    @property
    def mean_absolute_error(self) -> float:
        """Mean Absolute Error: mean(|E_m - E*|) across the M simulations.

        Different from ``rmse`` whenever the per-sim errors are not all
        equal. Use ``rmse`` for the L2 aggregate and this property for L1.
        """
        return float(np.mean(self.absolute_errors)) if self.absolute_errors else 0.0


def _migrate_simulation_result_dict(data: dict) -> dict:
    """
    Migrate old SimulationResult dict format to new format.
    Handles backward compatibility for loading old checkpoints.

    Old format had 'mean_absolute_error', new format uses 'rmse'.
    """
    if data is None:
        return None

    # Create a copy to avoid modifying the original
    migrated = data.copy()

    # If old format has mean_absolute_error, remove the legacy key before
    # constructing SimulationResult. Some hand-written or partially migrated
    # records may contain both fields; in that case rmse is authoritative and
    # the legacy key is only an unknown-constructor hazard.
    if 'mean_absolute_error' in migrated:
        legacy_value = migrated.pop('mean_absolute_error')
        if 'rmse' not in migrated:
            migrated['rmse'] = legacy_value
            logging.debug(
                "Migrated old SimulationResult format: mean_absolute_error -> rmse"
            )

    return migrated


def _migrate_result_record_dict(record: dict) -> dict:
    """Rewrite legacy outer-result records where ``energy_difference``
    was silently RMSE instead of an absolute error.

    Detection rule: the legacy bug always set
    ``record['energy_difference'] == simulation_result['rmse']`` exactly
    (both came from the same ``np.sqrt(np.mean)`` value). When we
    see that match and also have ``simulation_result.absolute_errors``,
    rewrite ``energy_difference`` (and the synced ``absolute_error``) to
    the true MAE so plots labeled "Energy difference" / "Absolute error"
    show absolute-error values, not RMSE.

    The migration is value-preserving in the M=1 case (RMSE = MAE = |E - E*|).
    """
    if not isinstance(record, dict):
        return record

    sim = record.get('simulation_result')
    if not isinstance(sim, dict):
        return record

    sim_for_detection = _migrate_simulation_result_dict(sim)
    legacy_diff = record.get('energy_difference')
    legacy_rmse = sim_for_detection.get('rmse')
    abs_errs = sim_for_detection.get('absolute_errors')

    if legacy_diff is None or legacy_rmse is None or abs_errs is None:
        return record
    try:
        if len(abs_errs) == 0:
            return record
    except TypeError:
        return record

    # Only rewrite when the legacy bug actually applied: energy_difference
    # was the same value as the RMSE field. Any other relation means a
    # downstream consumer already cleaned the field up; leave it alone.
    if abs(float(legacy_diff) - float(legacy_rmse)) > 1e-12:
        return record

    mae = float(np.mean(abs_errs))
    if abs(mae - float(legacy_rmse)) < 1e-12:
        # M=1 (or numerically equal). Nothing to fix.
        return record

    migrated = dict(record)
    migrated['energy_difference'] = mae
    migrated['absolute_error'] = mae
    migrated['mae'] = mae
    migrated['rmse'] = float(legacy_rmse)
    logging.info(
        " migration: rewrote legacy energy_difference (=RMSE %.6e) "
        "to MAE %.6e on result record (batch_element_rank=%s, update=%s)",
        float(legacy_rmse), mae,
        record.get('batch_element_rank'),
        record.get('update'),
    )
    return migrated


def _extended_result_from_record(record: dict) -> "ExtendedBatchElementEnergyResult":
    """Single point of truth for reconstructing
    ``ExtendedBatchElementEnergyResult`` from a deserialized JSON record.

    Applies the legacy-RMSE migration via
:func:`_migrate_result_record_dict`, the per-simulation field
    migration via:func:`_migrate_simulation_result_dict`, then threads
    every base field present in the record through the constructor — so
    new optional fields (``rmse``, ``mae``,...) round-trip without each
    loader site having to remember to add a keyword argument.

    Use this helper from every JSON loader instead of constructing
    ``ExtendedBatchElementEnergyResult`` by hand; review-history included four loader sites (one in
:class:`DiskBackedResultStore` and three in
:func:`load_experiment_state`) where missing a single keyword
    silently dropped a field.

    Recovery semantics: a corrupt nested ``simulation_result`` payload
    is logged at WARNING and the outer row is reconstructed with
    ``simulation_result=None`` — mirroring the pre- loaders so a
    single bad SimulationResult schema change can't poison an entire
    JSONL file.

    Raises:
        TypeError: ``record`` is not a ``dict``. The previous unguarded
            ``None.get`` traceback gave loaders a vague
            ``AttributeError`` to log; this contract is clearer.
    """
    if not isinstance(record, dict):
        raise TypeError(
            f"_extended_result_from_record expected dict, got {type(record).__name__}"
        )
    record = _migrate_result_record_dict(record)
    sim_data = record.get('simulation_result')
    simulation_result = None
    if isinstance(sim_data, dict):
        sim_data = _migrate_simulation_result_dict(sim_data)
        try:
            simulation_result = SimulationResult(**sim_data)
        except Exception as exc:
            logging.warning(
                "Could not load nested simulation_result "
                "(batch_element_rank=%s, update=%s): %s",
                record.get('batch_element_rank'),
                record.get('update'),
                exc,
            )
            simulation_result = None
    base_kwargs = {
        fld: record[fld]
        for fld in BatchElementEnergyResult.__dataclass_fields__
        if fld in record
    }
    return ExtendedBatchElementEnergyResult(
        simulation_result=simulation_result, **base_kwargs
    )


EVALUATOR_RESULT_TYPE_EXACT = "RESULTS"
EVALUATOR_RESULT_TYPE_SCALABLE_LARGE_REPORT = "SCALABLE_LARGE_REPORT"
EVALUATOR_RESULT_TYPE_ERROR = "EVAL_ERROR"
SCALABLE_LARGE_EVALUATION_JSON = "scalable_large_evaluation_results.json"
SCALABLE_LARGE_EVALUATION_JSONL = "scalable_large_evaluation_results.jsonl"


@dataclass
class ExtendedBatchElementEnergyResult(BatchElementEnergyResult):
    """Extended result including simulation statistics"""
    simulation_result: Optional[SimulationResult] = None


ScalableLargeEvaluationReport = Dict[str, Any]
EvaluationEntryPointResult = Union[
    List[ExtendedBatchElementEnergyResult],
    ScalableLargeEvaluationReport,
]


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles NumPy types"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, complex):
            return {'real': obj.real, 'imag': obj.imag, '_type': 'complex'}
        return super().default(obj)
