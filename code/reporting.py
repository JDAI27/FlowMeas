# -*- coding: utf-8 -*-
"""Asynchronous result-JSON reporting for FlowMeas.

Extracted from main.py (which keeps a backward-compatible facade re-export).
Report-cadence only (JSON) — no per-step or per-update training-loop
code. Imports the shared result-codec types from result_types.py and never
imports main/config, so there is no import cycle (main -> reporting only). The
result store is described by a structural protocol, so reporting.py does
not need to import its concrete implementation from main.
"""

import asyncio
import json
import os
import threading
import numpy as np
import torch
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol

try:
    from .result_types import NumpyEncoder, ExtendedBatchElementEnergyResult
except ImportError:  # direct-execution mode
    from result_types import NumpyEncoder, ExtendedBatchElementEnergyResult

class _ResultStoreLike(Protocol):
    """Operations AsyncReporter needs from its disk-backed result store."""

    def append(self, results: List[ExtendedBatchElementEnergyResult]) -> None:
        ...

    def load_all(self) -> List[ExtendedBatchElementEnergyResult]:
        ...

    def export_json(self, export_path: Path) -> None:
        ...


class AsyncReporter:
    """Handle asynchronous result reporting with disk-backed storage."""

    def __init__(
        self,
        results_dir: Path,
        hyperparameters: Optional[Dict],
        result_store: _ResultStoreLike,
    ):
        self.results_dir = results_dir
        self.hyperparameters = hyperparameters
        self.result_store = result_store
        self.lock = threading.Lock()

    def add_results(self, results: List[ExtendedBatchElementEnergyResult]):
        """Persist new results to disk."""
        if not results:
            return
        self.result_store.append(results)

    async def update_summary_async(self):
        """Recompute and persist summary statistics asynchronously."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._write_summary)

    def load_all_results(self) -> List[ExtendedBatchElementEnergyResult]:
        return self.result_store.load_all()

    def export_results_json(self, path: Path) -> None:
        self.result_store.export_json(path)

    def _write_summary(self):
        """Recompute and persist summary statistics."""
        with self.lock:
            results_copy = self.result_store.load_all()

        if not results_copy:
            return

        updates = sorted(set(r.update for r in results_copy))
        self._save_summary_statistics(results_copy, updates)


    def _save_summary_statistics(self, results: List[ExtendedBatchElementEnergyResult], updates: List[int]):
        """Save summary statistics to JSON"""
        summary_stats = {
            'updates': updates,
            'n_batch_elements_per_update': {},
            'best_energy_per_update': {},
            'mean_energy_per_update': {},
            'best_rmse_per_update': {},
            'mean_rmse_per_update': {},
            'total_circuits_per_update': {},
            'total_measurements_per_update': {},
            'success_rate_1.6e-3': {},
            'success_rate_1e-2': {},
            'mean_circuit_length': {},
            'pauli_coverage': {}
        }

        for update in updates:
            update_results = [r for r in results if r.update == update]
            if update_results:
                energy_diffs = [r.energy_difference for r in update_results]

                summary_stats['n_batch_elements_per_update'][str(update)] = len(update_results)
                summary_stats['best_energy_per_update'][str(update)] = min(energy_diffs)
                summary_stats['mean_energy_per_update'][str(update)] = np.mean(energy_diffs)
                summary_stats['total_circuits_per_update'][str(update)] = sum(r.n_circuits for r in update_results)
                summary_stats['total_measurements_per_update'][str(update)] = sum(r.total_measurements for r in update_results)
                summary_stats['success_rate_1.6e-3'][str(update)] = sum(1 for e in energy_diffs if e < 1.6e-3) / len(energy_diffs)
                summary_stats['success_rate_1e-2'][str(update)] = sum(1 for e in energy_diffs if e < 1e-2) / len(energy_diffs)
                summary_stats['mean_circuit_length'][str(update)] = np.mean([r.mean_circuit_length for r in update_results])
                summary_stats['pauli_coverage'][str(update)] = np.mean([r.convergence_metrics['coverage'] for r in update_results])

                # Add RMSE statistics if available
                rmse_results = [r for r in update_results if hasattr(r, 'simulation_result') and r.simulation_result is not None]
                if rmse_results:
                    rmse_values = [r.simulation_result.rmse for r in rmse_results]
                    summary_stats['best_rmse_per_update'][str(update)] = min(rmse_values)
                    summary_stats['mean_rmse_per_update'][str(update)] = np.mean(rmse_values)

        with open(self.results_dir / 'summary_statistics.json', 'w') as f:
            json.dump(summary_stats, f, indent=2, cls=NumpyEncoder)


def save_results_safely(results: List[ExtendedBatchElementEnergyResult], path: Path):
    """Save results with file locking to prevent race conditions."""
    results_data = []
    for r in results:
        r_dict = asdict(r)
        # Handle simulation result separately
        if hasattr(r, 'simulation_result') and r.simulation_result:
            r_dict['simulation_result'] = asdict(r.simulation_result)
        results_data.append(r_dict)

    # Use atomic write
    temp_path = str(path) + '.tmp'
    with open(temp_path, 'w') as f:
        json.dump(results_data, f, indent=2, cls=NumpyEncoder)

    # Atomic rename on most filesystems
    os.rename(temp_path, str(path))


def _to_report_float(value: Any) -> Optional[float]:
    """Convert scalar-ish values to JSON-safe floats without keeping tensors alive."""
    if value is None:
        return None
    if torch.is_tensor(value):
        if value.numel() == 0:
            return None
        return float(value.detach().cpu().reshape(-1)[0].item())
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return None
        return float(value.reshape(-1)[0])
    if isinstance(value, np.generic):
        return float(value)
    if isinstance(value, complex):
        return float(value.real)
    return float(value)


def _coefficient_abs_summary(weights: Optional[List[Any]]) -> Dict[str, Any]:
    """Return coefficient metadata without touching exact-energy properties."""
    if not weights:
        return {
            "count": 0,
            "largest_abs": None,
            "smallest_nonzero_abs": None,
            "mean_abs": None,
        }

    abs_values = [float(abs(w)) for w in weights]
    nonzero_abs_values = [w for w in abs_values if w > 1e-10]
    return {
        "count": len(abs_values),
        "largest_abs": max(abs_values) if abs_values else None,
        "smallest_nonzero_abs": (
            min(nonzero_abs_values) if nonzero_abs_values else None
        ),
        "mean_abs": float(np.mean(abs_values)) if abs_values else None,
    }
