#!/usr/bin/env python3
"""
Main experiment runner for GFlowNet quantum circuit optimization with energy estimation.
Supports asynchronous evaluation using multiprocessing for improved performance.
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
from dataclasses import dataclass, asdict, field
import threading
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    from .GFNs import EfficientGFNTrainer, get_device, exponential_reward_fn, SamplingMode, default_reward_fn, log_reward_fn
    from .pauli_hamiltonian_helper import PauliHamiltonianHelper
    from .energy_estimator import EnergyEstimator, BatchElementEnergyResult
    from .reward_functions import log_space_reward_fn
except ImportError:
    from GFNs import EfficientGFNTrainer, get_device, exponential_reward_fn, SamplingMode, default_reward_fn, log_reward_fn
    from pauli_hamiltonian_helper import PauliHamiltonianHelper
    from energy_estimator import EnergyEstimator, BatchElementEnergyResult

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

_current_trainer = None
_current_results_dir = None


@dataclass
class SimulationResult:
    """Results from multiple simulation runs."""
    energy_estimates: List[float]
    absolute_errors: List[float]
    rmse: float
    std_absolute_error: float
    mean_energy_estimate: float
    std_energy_estimate: float

    @property
    def mean_absolute_error(self) -> float:
        """Deprecated: Use rmse instead."""
        return np.mean(self.absolute_errors) if self.absolute_errors else 0.0


def _migrate_simulation_result_dict(data: dict) -> dict:
    """Migrate old SimulationResult dict format to new format for backward compatibility."""
    if data is None:
        return None

    migrated = data.copy()
    if 'mean_absolute_error' in migrated and 'rmse' not in migrated:
        migrated['rmse'] = migrated.pop('mean_absolute_error')
    return migrated
    

@dataclass
class ExtendedBatchElementEnergyResult(BatchElementEnergyResult):
    """Extended result including simulation statistics"""
    simulation_result: Optional[SimulationResult] = None


@dataclass
class ExperimentConfig:
    """Configuration for experiments"""
    hamiltonian_path: str
    eval_every: int = 1000
    n_updates: int = 10000
    n_measurements: int = 1000  # Circuits per batch element
    update_freq: int = 5  # Number of batch elements per update
    max_depth: int = 8
    beta: float = 1e3
    hidden_dim: int = 1024
    num_hidden_layers: int = 3
    lr: float = 1e-3
    weight_decay: float = 1e-5
    device_preference: str = "auto"
    results_dir: str = "./experiment_results"
    n_eval_top_k_batch_elements: int = 5  # Number of top batch elements to evaluate
    replay_every: int = 25
    offpolicy_every: int = 20
    checkpoint_every: int = 100
    reward_type: str = "log"  # Options: "default", "exponential", "threshold"
    reward_kwargs: Dict = field(default_factory=lambda: {"alpha": 1.0})
    cost_type: str = "exponential"
    cost_kwargs: Dict = field(default_factory=dict)  # Arguments for cost computation
    objective_type: str = "tb"
    objective_kwargs: Dict = field(default_factory=lambda: {"loss_type": "squared"})
    n_simulations: int = 10  # Number of simulation runs for error estimation
    resume: bool = True  # Whether to resume from existing experiment
    experiment_dir: Optional[str] = None  # Specific experiment directory to resume from
    warm_start_from: Optional[str] = None  # Load NN weights and logZ from this experiment (no hyperparameter check)
    model_type: str = "clifford_mlp"
    async_eval: bool = False  # Enable asynchronous evaluation
    eval_poll_interval: int = 30  # Seconds between checkpoint polls
    eval_process_timeout: int = 300  # Timeout for evaluator process shutdown
    transfer_weights_on_depth_change: bool = False  # If False, don't transfer NN weights when max_depth changes


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles NumPy types."""
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


def convert_metrics_to_cpu(metrics: Dict[str, Any]) -> Dict[str, float]:
    """Convert metric dictionary with GPU tensors to CPU floats/numpy arrays."""
    converted = {}
    for k, v in metrics.items():
        if torch.is_tensor(v):
            converted[k] = v.item() if v.numel() == 1 else v.cpu().numpy()
        else:
            converted[k] = v
    return converted


def convert_metrics_history_to_cpu(metrics_history: Dict[str, List[Any]]) -> Dict[str, List[float]]:
    """Convert metrics history with GPU tensors to CPU for checkpoint saving."""
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
        self._base_fields = list(BatchElementEnergyResult.__dataclass_fields__.keys())

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

        results: List[ExtendedBatchElementEnergyResult] = []
        with self.lock:
            with self.path.open('r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    results.append(self._to_result(record))
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
        base_kwargs = {field: record[field] for field in self._base_fields if field in record}
        sim_data = record.get('simulation_result')
        sim_data = _migrate_simulation_result_dict(sim_data) if isinstance(sim_data, dict) else None
        simulation_result = SimulationResult(**sim_data) if sim_data is not None else None
        return ExtendedBatchElementEnergyResult(simulation_result=simulation_result, **base_kwargs)


class DiskBackedMetricStore:
    """Append-only metric logger backed by JSONL, with helpers for aggregation."""

    def __init__(self, path: Path, max_buffer_size: int = 1000):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.buffer: List[Tuple[int, Dict[str, float], Optional[Dict[str, float]]]] = []
        self.max_buffer_size = max_buffer_size

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
        """Add metrics to buffer without writing to disk. Detaches tensors to prevent memory leaks."""
        if metrics is None:
            return

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

        self.buffer.append((update, metrics_detached, timing_detached))

        if len(self.buffer) >= self.max_buffer_size:
            logging.warning(f"Metric buffer exceeded {self.max_buffer_size} entries, auto-flushing")
            self.flush_buffer()

    def flush_buffer(self) -> None:
        """Flush buffered metrics to disk with GPU-CPU transfer."""
        if not self.buffer:
            return

        num_flushed = len(self.buffer)

        with self.lock:
            with self.path.open('a', encoding='utf-8') as f:
                for update, metrics, timing in self.buffer:
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

                    del metrics
                    del timing

        self.buffer.clear()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return num_flushed

    def replace(self, metrics_history: Dict[str, List], timing_history: Optional[Dict[str, List]] = None) -> None:
        max_len = max((len(values) for values in metrics_history.values()), default=0)
        timing_history = timing_history or {}

        with self.lock:
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

    def load_series(self) -> Tuple[List[int], Dict[str, List[float]], Dict[str, List[float]]]:
        updates: List[int] = []
        metrics_series: Dict[str, List[float]] = defaultdict(list)
        timing_series: Dict[str, List[float]] = defaultdict(list)

        if not self.path.exists():
            return updates, metrics_series, timing_series

        with self.lock:
            with self.path.open('r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    updates.append(record.get('update', len(updates) + 1))
                    for key, value in record.get('metrics', {}).items():
                        metrics_series[key].append(value)
                    for key, value in record.get('timing', {}).items():
                        timing_series[key].append(value)

        return updates, metrics_series, timing_series

class AsyncVisualizer:
    """Handle asynchronous visualization updates with disk-backed storage."""
    
    def __init__(self, results_dir: Path, hyperparameters: Optional[Dict], result_store: DiskBackedResultStore):
        self.results_dir = results_dir
        self.fig_dir = results_dir / "figures"
        self.fig_dir.mkdir(exist_ok=True)
        self.hyperparameters = hyperparameters
        self.result_store = result_store
        self.lock = threading.Lock()
    
    def add_results(self, results: List[ExtendedBatchElementEnergyResult]):
        """Persist new results to disk."""
        if not results:
            return
        self.result_store.append(results)
    
    async def update_plots_async(self):
        """Update plots asynchronously."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._update_plots)
    
    def load_all_results(self) -> List[ExtendedBatchElementEnergyResult]:
        return self.result_store.load_all()

    def export_results_json(self, path: Path) -> None:
        self.result_store.export_json(path)
    
    def _update_plots(self):
        """Generate all plots."""
        with self.lock:
            results_copy = self.result_store.load_all()
        
        if not results_copy:
            return
        
        updates = sorted(set(r.update for r in results_copy))
        
        plt.figure(figsize=(14, 10))
        
        for update in updates:
            update_results = [r for r in results_copy if r.update == update]
            batch_ranks = [r.batch_element_rank for r in update_results]
            energy_diffs = [r.energy_difference for r in update_results]
            
            error_bars = []
            for r in update_results:
                if hasattr(r, 'simulation_result') and r.simulation_result:
                    error_bars.append(r.simulation_result.std_absolute_error)
                else:
                    error_bars.append(0)
            
            plt.errorbar(batch_ranks, energy_diffs, yerr=error_bars, 
                        fmt='o', alpha=0.6, capsize=5, markersize=8,
                        label=f'Update {update}')
        
        plt.xlabel('Batch Element Rank (0 = best)')
        plt.ylabel('Energy Difference from Ground State')
        plt.title('Energy Estimation Results by Batch Element (with simulation error bars)')
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
        plt.text(0.02, 0.02, 'Error bars show std dev across simulations', 
                transform=plt.gca().transAxes, fontsize=10)
        plt.tight_layout()
        plt.savefig(self.fig_dir / 'batch_energy_convergence.png', dpi=300)
        plt.close()
        
        plt.figure(figsize=(12, 8))
        
        best_energies_per_update = []
        best_rmse_per_update = []
        update_labels = []
        total_measurements_per_update = []
        
        for update in updates:
            update_results = [r for r in results_copy if r.update == update]
            if update_results:
                best_result = min(update_results, key=lambda r: r.energy_difference)
                best_energies_per_update.append(best_result.energy_difference)
                update_labels.append(update)
                
                if hasattr(best_result, 'simulation_result') and best_result.simulation_result:
                    best_rmse_per_update.append(best_result.simulation_result.rmse)
                else:
                    best_rmse_per_update.append(None)
                
                total_measurements = sum(r.total_measurements for r in update_results)
                total_measurements_per_update.append(total_measurements)
        
        plt.plot(update_labels, best_energies_per_update, 'o-', linewidth=2, markersize=8, label='Energy difference')
        
        rmse_values = [rmse for rmse in best_rmse_per_update if rmse is not None]
        rmse_updates = [u for u, rmse in zip(update_labels, best_rmse_per_update) if rmse is not None]
        if rmse_values:
            plt.plot(rmse_updates, rmse_values, 's--', linewidth=2, markersize=6,
                    label='RMSE (simulations)', alpha=0.7)
        
        plt.xlabel('Training Update')
        plt.ylabel('Energy Difference / Error')
        plt.title('Best Batch Element Energy vs Training Progress')
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
        plt.axhline(y=1.6e-3, color='g', linestyle='--', alpha=0.5, label='Chemical accuracy (1.6e-3)')
        plt.axhline(y=1e-2, color='orange', linestyle='--', alpha=0.5, label='1e-2 threshold')
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.fig_dir / 'best_energy_vs_training.png', dpi=300)
        plt.close()
        
        plt.figure(figsize=(12, 8))
        
        if updates:
            latest_results = [r for r in results_copy if r.update == updates[-1] and hasattr(r, 'simulation_result') and r.simulation_result is not None]
            
            if latest_results:
                mae_values = [r.simulation_result.mean_absolute_error for r in latest_results]
                batch_ranks = [r.batch_element_rank for r in latest_results]
                
                plt.bar(batch_ranks, mae_values, alpha=0.7)
                plt.xlabel('Batch Element Rank')
                plt.ylabel('Mean Absolute Error (across simulations)')
                plt.title(f'Mean Absolute Estimation Error by Batch Element (Update {updates[-1]})')
                plt.yscale('log')
                plt.grid(True, alpha=0.3, axis='y')
                
                mean_mae = np.mean(mae_values)
                plt.axhline(y=mean_mae, color='r', linestyle='--', alpha=0.7, 
                           label=f'Mean MAE: {mean_mae:.2e}')
                plt.legend()
        
        plt.tight_layout()
        plt.savefig(self.fig_dir / 'mae_distribution.png', dpi=300)
        plt.close()
        
        if results_copy and updates:
            latest_results = [r for r in results_copy if r.update == updates[-1]]
            
            if latest_results:
                n_batch_elements = len(latest_results)
                n_cols = min(3, n_batch_elements)
                n_rows = (n_batch_elements + n_cols - 1) // n_cols
                
                fig, axes = plt.subplots(n_rows, n_cols, figsize=(8*n_cols, 6*n_rows))
                
                if n_batch_elements == 1:
                    axes = [axes]
                else:
                    axes = axes.flatten() if n_rows > 1 or n_cols > 1 else [axes]
                
                all_paulis = set()
                for result in latest_results:
                    all_paulis.update(result.hitting_counts.keys())
                
                ordered_paulis = list(all_paulis)
                max_paulis_shown = 30
                if len(ordered_paulis) > max_paulis_shown:
                    top_paulis = ordered_paulis[:max_paulis_shown]
                    x_label = f'First {max_paulis_shown} Pauli Strings'
                else:
                    top_paulis = ordered_paulis
                    x_label = 'Pauli Strings'
                
                for idx, result in enumerate(latest_results):
                    ax = axes[idx]
                    hit_counts = [result.hitting_counts.get(p, 0) for p in top_paulis]
                    pauli_indices = list(range(len(top_paulis)))
                    
                    bars = ax.bar(pauli_indices, hit_counts, alpha=0.7)
                    
                    max_count = max(hit_counts) if hit_counts else 1
                    for bar, count in zip(bars, hit_counts):
                        if count > 0:
                            bar.set_color(plt.cm.viridis(count / max_count))
                    
                    ax.set_xticks(pauli_indices)
                    ax.set_xticklabels(top_paulis, rotation=90, ha='right', fontsize=8)
                    ax.set_xlabel(x_label, fontsize=10)
                    ax.set_ylabel('Hitting Count')
                    ax.set_title(f'Batch Element Rank {result.batch_element_rank}')
                    ax.grid(True, alpha=0.3, axis='y')
                    
                    total_paulis_with_hits = sum(1 for p in ordered_paulis if result.hitting_counts.get(p, 0) > 0)
                    coverage = total_paulis_with_hits / len(ordered_paulis) if ordered_paulis else 0
                    ax.text(0.98, 0.98, f'Coverage: {coverage:.1%}', 
                           transform=ax.transAxes, ha='right', va='top',
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                    ax.text(0.02, 0.98, f'Circuits: {result.n_circuits}', 
                           transform=ax.transAxes, ha='left', va='top',
                           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
                
                for idx in range(n_batch_elements, len(axes)):
                    axes[idx].set_visible(False)
                
                plt.suptitle(f'Pauli Measurement Coverage by Batch Element (Update {updates[-1]})', fontsize=16)
                plt.tight_layout()
                plt.savefig(self.fig_dir / 'pauli_coverage_per_batch.png', dpi=300)
                plt.close()
                
                plt.figure(figsize=(16, 10))
                heatmap_paulis = ordered_paulis[:50]
                
                hit_matrix = np.zeros((n_batch_elements, len(heatmap_paulis)))
                for i, result in enumerate(latest_results):
                    for j, pauli in enumerate(heatmap_paulis):
                        hit_matrix[i, j] = result.hitting_counts.get(pauli, 0)
                
                im = plt.imshow(hit_matrix, aspect='auto', cmap='YlOrRd', interpolation='nearest')
                plt.colorbar(im, label='Hitting Count')
                plt.xlabel('Pauli Strings')
                plt.ylabel('Batch Element Rank')
                plt.title(f'Pauli Hitting Count Heatmap (Update {updates[-1]})')
                plt.yticks(range(n_batch_elements), [r.batch_element_rank for r in latest_results])
                plt.xticks(range(len(heatmap_paulis)), heatmap_paulis, rotation=90, ha='right', fontsize=8)
                plt.grid(True, alpha=0.3, which='both', linestyle='-', linewidth=0.5)
                
                for i, result in enumerate(latest_results):
                    total_paulis_with_hits = sum(1 for p in ordered_paulis if result.hitting_counts.get(p, 0) > 0)
                    coverage = total_paulis_with_hits / len(ordered_paulis) if ordered_paulis else 0
                    plt.text(len(heatmap_paulis) + 0.5, i, f'{coverage:.1%}', va='center', ha='left')
                
                if len(ordered_paulis) > len(heatmap_paulis):
                    plt.text(0.5, -0.15, f'Note: Showing {len(heatmap_paulis)} of {len(ordered_paulis)} total Pauli strings', 
                            transform=plt.gca().transAxes, ha='center', fontsize=10, style='italic')
                
                plt.tight_layout()
                plt.savefig(self.fig_dir / 'pauli_coverage_heatmap.png', dpi=300, bbox_inches='tight')
                plt.close()
                
                plt.figure(figsize=(14, 8))
                top_paulis = ordered_paulis[:20]
                
                total_hitting_counts = defaultdict(int)
                for result in latest_results:
                    for pauli, count in result.hitting_counts.items():
                        if pauli in top_paulis:
                            total_hitting_counts[pauli] += count
                
                pauli_labels = top_paulis
                hit_counts = [total_hitting_counts[p] for p in pauli_labels]
                x_pos = np.arange(len(pauli_labels))
                
                bars = plt.bar(x_pos, hit_counts, alpha=0.7)
                
                if latest_results[0].pauli_estimates:
                    coeffs = [abs(latest_results[0].pauli_estimates.get(p, 0)) for p in pauli_labels]
                    max_coeff = max(coeffs) if coeffs else 1
                    for bar, coeff in zip(bars, coeffs):
                        bar.set_color(plt.cm.plasma(coeff / max_coeff))
                    
                    sm = plt.cm.ScalarMappable(cmap=plt.cm.plasma, norm=plt.Normalize(vmin=0, vmax=max_coeff))
                    sm.set_array([])
                    cbar = plt.colorbar(sm, ax=plt.gca(), pad=0.01)
                    cbar.set_label('Coefficient Magnitude', rotation=270, labelpad=15)
                
                plt.xlabel('Pauli Strings')
                plt.ylabel('Total Hitting Count (all batch elements)')
                plt.title(f'Hitting Counts for First 20 Pauli Strings (Update {updates[-1]})')
                plt.xticks(x_pos, pauli_labels, rotation=45, ha='right')
                plt.grid(True, alpha=0.3, axis='y')
                plt.tight_layout()
                plt.savefig(self.fig_dir / 'top_pauli_coverage.png', dpi=300)
                plt.close()
        
        plt.figure(figsize=(10, 6))
        
        for update in updates[-3:]:
            all_lengths = []
            update_results = [r for r in results_copy if r.update == update]
            
            for result in update_results:
                all_lengths.extend(result.circuit_lengths)
            
            if all_lengths:
                plt.hist(all_lengths, bins=range(1, max(all_lengths)+2), 
                        alpha=0.5, label=f'Update {update}', density=True)
        
        plt.xlabel('Circuit Length')
        plt.ylabel('Density')
        plt.title('Distribution of Circuit Lengths')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.fig_dir / 'circuit_length_distribution.png', dpi=300)
        plt.close()
        
        self._save_summary_statistics(results_copy, updates)
    
    def _save_summary_statistics(self, results: List[ExtendedBatchElementEnergyResult], updates: List[int]):
        """Save summary statistics to JSON."""
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
                
                rmse_results = [r for r in update_results if hasattr(r, 'simulation_result') and r.simulation_result is not None]
                if rmse_results:
                    rmse_values = [r.simulation_result.rmse for r in rmse_results]
                    summary_stats['best_rmse_per_update'][str(update)] = min(rmse_values)
                    summary_stats['mean_rmse_per_update'][str(update)] = np.mean(rmse_values)
        
        with open(self.results_dir / 'summary_statistics.json', 'w') as f:
            json.dump(summary_stats, f, indent=2, cls=NumpyEncoder)


def save_results_safely(results: List[ExtendedBatchElementEnergyResult], path: Path):
    """Save results with atomic write to prevent corruption."""
    results_data = []
    for r in results:
        r_dict = asdict(r)
        if hasattr(r, 'simulation_result') and r.simulation_result:
            r_dict['simulation_result'] = asdict(r.simulation_result)
        results_data.append(r_dict)
    
    temp_path = str(path) + '.tmp'
    with open(temp_path, 'w') as f:
        json.dump(results_data, f, indent=2, cls=NumpyEncoder)
    os.rename(temp_path, str(path))


async def evaluate_top_batch_elements_from_checkpoint(checkpoint_path: Path,
                                                    energy_estimator: EnergyEstimator,
                                                    update: int,
                                                    config: ExperimentConfig,
                                                    device: torch.device) -> List[ExtendedBatchElementEnergyResult]:
    """Evaluate top-k batch elements from checkpoint without needing trainer instance."""
    logging.info(f"\n=== Evaluating checkpoint at update {update} ===")
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    top_trajectories_data = checkpoint.get('top_trajectories', [])
    
    if not top_trajectories_data:
        logging.info("  No trajectories found in checkpoint. Skipping evaluation.")
        return []
    
    num_to_eval = min(config.n_eval_top_k_batch_elements, len(top_trajectories_data))
    logging.info(f"  Evaluating top {num_to_eval} batch elements from checkpoint")
    
    n_simulations = getattr(config, 'n_simulations', 1)
    if n_simulations > 1:
        logging.info(f"  Running {n_simulations} simulations per batch element")

    batch_actions_list = []
    batch_lengths_list = []
    batch_costs = []
    
    for batch_idx in range(num_to_eval):
        traj_data = top_trajectories_data[batch_idx]
        batch_actions = traj_data['actions'].to(device)
        batch_lengths = traj_data['lengths'].to(device)
        
        if len(batch_actions.shape) == 1:
            batch_actions = batch_actions.unsqueeze(0)
            batch_lengths = batch_lengths.unsqueeze(0) if isinstance(batch_lengths, torch.Tensor) else torch.tensor([batch_lengths])
        
        batch_actions_list.append(batch_actions)
        batch_lengths_list.append(batch_lengths)
        batch_costs.append(traj_data.get('cost', 0.0))
    
    all_batch_actions = torch.stack(batch_actions_list, dim=0)
    all_batch_lengths = torch.stack(batch_lengths_list, dim=0)
    
    logging.info(f"  Combined batch shape: {all_batch_actions.shape}")
    
    summaries = energy_estimator.estimate_energy_with_simulations(
        all_batch_actions,
        all_batch_lengths,
        M=n_simulations,
    )
    
    results = []
    
    for batch_idx, summary in enumerate(summaries):
        final_result_obj = summary['final_results_object']
        
        simulation_result = None
        if n_simulations > 1 and 'energy_variance' in summary:
            mean_energy = summary['mean_energy']
            variance = summary['energy_variance']

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
                logging.warning(f"Individual simulation energies not available for batch {batch_idx}")
                simulation_result = None
        
        result = ExtendedBatchElementEnergyResult(
            batch_element_rank=batch_idx,
            energy_estimate=summary['mean_energy'],
            energy_difference=summary['energy_difference'],
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
        
        logging.info(f"\n  Batch element rank {batch_idx}:")
        logging.info(f"    Number of circuits: {result.n_circuits}")
        logging.info(f"    Energy estimate: {result.energy_estimate:.6f}")
        logging.info(f"    Energy difference: {result.energy_difference:.6e}")
        if simulation_result and n_simulations > 1:
            logging.info(f"    Standard deviation: {simulation_result.std_absolute_error:.6e}")
        logging.info(f"    Pauli coverage: {result.convergence_metrics['coverage']:.1%}")
        logging.info(f"    Mean circuit length: {result.mean_circuit_length:.1f}")
    
    if results:
        energy_diffs = [r.energy_difference for r in results]
        best_result = min(results, key=lambda r: r.energy_difference)
        
        logging.info(f"\n  Evaluation Summary:")
        logging.info(f"    Best energy difference (Batch rank {best_result.batch_element_rank}): {best_result.energy_difference:.6e}")
        logging.info(f"    Mean energy difference: {np.mean(energy_diffs):.6e}")
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
    """Evaluator process loop that runs independently of training on CPU."""
    logging.info("\n=== Starting Evaluator Process ===")
    logging.info(f"Results directory: {results_dir}")
    logging.info(f"Poll interval: {config.eval_poll_interval}s")
    
    device = torch.device('cpu')
    logging.info(f"Evaluator using device: {device}")
    
    energy_estimator = EnergyEstimator(
        hamiltonian_helper,
        hamiltonian_helper.n_qubits,
        device=device,
        force_cpu=True
    )
    
    processed_checkpoints = set()
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        while True:
            try:
                checkpoint_info = checkpoint_queue.get_nowait()
                
                if checkpoint_info == 'STOP':
                    logging.info("Received STOP signal. Shutting down evaluator.")
                    break
                    
                checkpoint_path, update, checkpoint_id = checkpoint_info
                
                if checkpoint_id in processed_checkpoints:
                    logging.info(f"Skipping already processed checkpoint {checkpoint_id}")
                    continue
                    
                processed_checkpoints.add(checkpoint_id)
                logging.info(f"\n{'='*60}")
                logging.info(f"Processing checkpoint {checkpoint_path} (ID: {checkpoint_id})")
                logging.info(f"Update: {update}")
                logging.info(f"{'='*60}")
                
                try:
                    results = loop.run_until_complete(
                        evaluate_top_batch_elements_from_checkpoint(
                            Path(checkpoint_path),
                            energy_estimator,
                            update,
                            config,
                            device
                        )
                    )
                    
                    if results:
                        logging.info(f"\nEvaluation complete for update {update}")
                        logging.info(f"Number of results: {len(results)}")
                        
                        eval_results_path = results_dir / f'eval_results_update_{update}.json'
                        save_results_safely(results, eval_results_path)
                        logging.info(f"Saved evaluation results to: {eval_results_path}")
                        
                        results_queue.put(('RESULTS', update, results))
                        logging.info(f"Sent {len(results)} results back to main process")
                        
                        energy_diffs = [r.energy_difference for r in results]
                        best_energy = min(energy_diffs)
                        logging.info(f"Best energy difference: {best_energy:.6e}")
                    else:
                        logging.warning(f"No results generated for update {update}")
                        
                except Exception as e:
                    logging.error(f"Error evaluating checkpoint: {e}")
                    import traceback
                    traceback.print_exc()
                    
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
    finally:
        loop.close()
        logging.info("Evaluator process finished.")


def create_hyperparameters_dict(config: ExperimentConfig, 
                               hamiltonian_helper: PauliHamiltonianHelper,
                               training_pauli_strings: List[str],
                               identity_weight: float) -> Dict:
    """Create a comprehensive hyperparameters dictionary."""
    device = get_device(config.device_preference)
    device_info = {
        "type": str(device),
        "preference": config.device_preference
    }
    if device.type == "cuda":
        device_info["cuda_device_name"] = torch.cuda.get_device_name(device)
        device_info["cuda_device_count"] = torch.cuda.device_count()
    
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
            "eval_process_timeout": config.eval_process_timeout
        },
        
        "hamiltonian": {
            "filepath": str(config.hamiltonian_path),
            "n_qubits": hamiltonian_helper.n_qubits,
            "n_terms": len(hamiltonian_helper.pauli_str_list),
            "n_training_terms": len(training_pauli_strings),
            "identity_weight": identity_weight,
            "exact_ground_state_energy": hamiltonian_helper.ground_state_energy,
            "molecule": hamiltonian_helper.filepath.parent.name,
            "transformation": hamiltonian_helper.filepath.stem,
            "excluded_from_training": ["Identity term (I^n)"] if identity_weight != 0 else []
        },
        
        "gfn_model": {
            "model_type": config.model_type,
            "hidden_dim": config.hidden_dim,
            "num_hidden_layers": config.num_hidden_layers,
            "lr": config.lr,
            "weight_decay": config.weight_decay,
            "grad_clip_value": 100.0,
            "weight_init": "xavier_uniform",
            "logZ_init": config.beta * 0.5,  # Initial logZ value
            "logZ_lr_multiplier": 10
        },
        
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
            "available_types": ["exponential", "linear", "linear_bias", "logarithmic", "quadratic", "threshold"]
        },
        
        "quantum_gates": {
            "single_qubit": ["H", "S", "HS", "SH", "HSH"],
            "two_qubit": ["CNOT"],
            "connectivity": "nearest_neighbor",
            "total_actions": 5 * hamiltonian_helper.n_qubits + 2 * (hamiltonian_helper.n_qubits - 1) + 1,
            "training_note": "Identity Pauli string excluded from cost function"
        },
        
        "energy_estimation": {
            "method": "batched_clifford_map_with_state_vector",
            "implementation": "energy_estimator.EnergyEstimator",
            "equation": "ô(P) = (1/N_P) ∑_i ⟨b_i|U_i†PU_i|b_i⟩",
            "reference": "Equation (3) from DSS paper",
            "simulations": config.n_simulations if hasattr(config, 'n_simulations') else 1,
            "error_formula": "(1/S) ∑_{s=1}^S |⟨H⟩ - ô_N^(s)(H)|",
            "async_mode": "multiprocessing with CPU-only evaluation" if config.async_eval else "synchronous"
        },
        
        "computational": {
            "device": device_info,
            "batch_processing": True,
            "gpu_enabled": True,
            "sparse_matrices": True,
            "async_evaluation": config.async_eval
        },
        
        "evaluation": {
            "energy_computation": "circuit_based_expectation_value",
            "convergence_threshold": 1e-3,
            "success_thresholds": [1.6e-3, 1e-2]
        }
    }
    
    return hyperparameters


def find_latest_experiment(results_base_dir: str) -> Optional[Path]:
    """Find the most recent experiment directory."""
    results_base = Path(results_base_dir)
    if not results_base.exists():
        return None
    
    experiment_dirs = [d for d in results_base.iterdir() 
                      if d.is_dir() and d.name.startswith('experiment_')]
    
    if not experiment_dirs:
        return None
    
    return max(experiment_dirs, key=lambda d: d.stat().st_mtime)


def load_experiment_state(experiment_dir: Path) -> Tuple[int, Dict, List[ExtendedBatchElementEnergyResult], Dict]:
    """Load the state of a previous experiment."""
    logging.info(f"Loading experiment state from {experiment_dir}")
    
    with open(experiment_dir / 'config.json', 'r') as f:
        saved_config = json.load(f)
    
    with open(experiment_dir / 'hyperparameters.json', 'r') as f:
        hyperparameters = json.load(f)
    
    checkpoint_files = list(experiment_dir.glob('checkpoint_update*.pth'))
    if not checkpoint_files:
        start_update = 0
        metrics_history = defaultdict(list)
    else:
        latest_checkpoint = max(checkpoint_files, key=lambda f: f.stat().st_mtime)
        checkpoint = torch.load(latest_checkpoint, map_location='cpu')
        
        filename_match = re.search(r'checkpoint_update_?(\d+)\.pth', latest_checkpoint.name)
        filename_update = int(filename_match.group(1)) if filename_match else 0
        
        if isinstance(checkpoint, dict):
            start_update = checkpoint.get('update', checkpoint.get('epoch', filename_update))
            
            if 'metrics_history' in checkpoint:
                metrics_history = defaultdict(list, checkpoint['metrics_history'])
            elif 'metrics' in checkpoint:
                metrics_history = defaultdict(list, checkpoint['metrics'])
            else:
                metrics_history = defaultdict(list)
                logging.warning("No metrics history found in checkpoint, starting fresh")
        else:
            start_update = filename_update
            metrics_history = defaultdict(list)
            logging.warning(f"Old checkpoint format detected, using update number from filename: {start_update}")
        
        logging.info(f"Found checkpoint at update {start_update}")
    
    evaluation_results = []
    eval_files = list(experiment_dir.glob('eval_results_update_*.json'))
    for eval_file in eval_files:
        with open(eval_file, 'r') as f:
            results_data = json.load(f)
        
        for r_dict in results_data:
            sim_result = None
            if 'simulation_result' in r_dict and r_dict['simulation_result'] and isinstance(r_dict['simulation_result'], dict):
                sim_data = _migrate_simulation_result_dict(r_dict['simulation_result'])
                try:
                    sim_result = SimulationResult(**sim_data)
                except Exception as e:
                    logging.warning(f"Could not load simulation result: {e}")
                    sim_result = None
            
            result = ExtendedBatchElementEnergyResult(
                batch_element_rank=r_dict['batch_element_rank'],
                energy_estimate=r_dict['energy_estimate'],
                energy_difference=r_dict['energy_difference'],
                hitting_counts=r_dict['hitting_counts'],
                pauli_estimates=r_dict['pauli_estimates'],
                circuit_lengths=r_dict['circuit_lengths'],
                mean_circuit_length=r_dict['mean_circuit_length'],
                n_circuits=r_dict['n_circuits'],
                total_measurements=r_dict['total_measurements'],
                convergence_metrics=r_dict['convergence_metrics'],
                batch_cost=r_dict.get('batch_cost'),
                update=r_dict['update'],
                simulation_result=sim_result
            )
            evaluation_results.append(result)
    
    jsonl_file = experiment_dir / 'evaluation_results.jsonl'
    if jsonl_file.exists():
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r_dict = json.loads(line)

                sim_result = None
                sim_data = r_dict.get('simulation_result')
                if isinstance(sim_data, dict):
                    sim_data = _migrate_simulation_result_dict(sim_data)
                    try:
                        sim_result = SimulationResult(**sim_data)
                    except Exception as e:
                        logging.warning(f"Could not load simulation result: {e}")
                        sim_result = None

                evaluation_results.append(ExtendedBatchElementEnergyResult(
                    batch_element_rank=r_dict['batch_element_rank'],
                    energy_estimate=r_dict['energy_estimate'],
                    energy_difference=r_dict['energy_difference'],
                    hitting_counts=r_dict['hitting_counts'],
                    pauli_estimates=r_dict['pauli_estimates'],
                    circuit_lengths=r_dict['circuit_lengths'],
                    mean_circuit_length=r_dict['mean_circuit_length'],
                    n_circuits=r_dict['n_circuits'],
                    total_measurements=r_dict['total_measurements'],
                    convergence_metrics=r_dict['convergence_metrics'],
                    batch_cost=r_dict.get('batch_cost'),
                    update=r_dict['update'],
                    simulation_result=sim_result
                ))
    else:
        eval_file = experiment_dir / 'evaluation_results.json'
        if eval_file.exists():
            with open(eval_file, 'r') as f:
                results_data = json.load(f)

            for r_dict in results_data:
                if any(r.update == r_dict['update'] and r.batch_element_rank == r_dict['batch_element_rank']
                       for r in evaluation_results):
                    continue

                sim_result = None
                if 'simulation_result' in r_dict and r_dict['simulation_result'] and isinstance(r_dict['simulation_result'], dict):
                    sim_data = _migrate_simulation_result_dict(r_dict['simulation_result'])
                    try:
                        sim_result = SimulationResult(**sim_data)
                    except Exception as e:
                        logging.warning(f"Could not load simulation result: {e}")
                        sim_result = None

                evaluation_results.append(ExtendedBatchElementEnergyResult(
                    batch_element_rank=r_dict['batch_element_rank'],
                    energy_estimate=r_dict['energy_estimate'],
                    energy_difference=r_dict['energy_difference'],
                    hitting_counts=r_dict['hitting_counts'],
                    pauli_estimates=r_dict['pauli_estimates'],
                    circuit_lengths=r_dict['circuit_lengths'],
                    mean_circuit_length=r_dict['mean_circuit_length'],
                    n_circuits=r_dict['n_circuits'],
                    total_measurements=r_dict['total_measurements'],
                    convergence_metrics=r_dict['convergence_metrics'],
                    batch_cost=r_dict.get('batch_cost'),
                    update=r_dict['update'],
                    simulation_result=sim_result
                ))
    
    if evaluation_results:
        logging.info(f"Loaded {len(evaluation_results)} existing evaluation results")
        if start_update == 0:
            last_eval_update = max(r.update for r in evaluation_results)
            logging.info(f"No checkpoint update found, using last evaluation update: {last_eval_update}")
            start_update = last_eval_update
    
    return start_update, metrics_history, evaluation_results, hyperparameters


def check_config_compatibility(saved_config: Dict, current_config: ExperimentConfig) -> Tuple[bool, bool, bool]:
    """Check compatibility between saved and current configurations.
    
    Returns:
        Tuple of (nn_params_match, all_params_match, max_depth_changed)
    """
    nn_params = ['hidden_dim', 'num_hidden_layers']
    nn_params_match = all(
        saved_config.get(param) == getattr(current_config, param)
        for param in nn_params
    )
    
    max_depth_changed = saved_config.get('max_depth') != current_config.max_depth
    
    all_params = [
        'hamiltonian_path', 'n_measurements', 'max_depth', 'beta',
        'hidden_dim', 'num_hidden_layers', 'lr', 'weight_decay',
        'reward_type', 'reward_kwargs', 'cost_type', 'cost_kwargs',
        'objective_type', 'objective_kwargs',
        'update_freq', 'n_eval_top_k_batch_elements'
    ]
    all_params_match = all(
        saved_config.get(param) == getattr(current_config, param)
        for param in all_params
    )
    
    return nn_params_match, all_params_match, max_depth_changed


def extract_logZ_from_checkpoint(checkpoint_path: Path) -> Optional[float]:
    """Extract logZ value from a checkpoint file."""
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        if isinstance(checkpoint, dict):
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
    """Load only the neural network weights from a checkpoint, including logZ."""
    try:
        logging.info(f"Loading neural network weights from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        logZ_val = None
        if hasattr(trainer.gfn.pf_model, 'logZ') and isinstance(checkpoint, dict):
            state_dict = checkpoint.get('pf_model_state_dict') or checkpoint.get('model_state_dict') or checkpoint.get('state_dict')
            if state_dict:
                logZ_keys = ['logZ', '_orig_mod.logZ', 'module.logZ']
                for key in logZ_keys:
                    if key in state_dict:
                        logZ_tensor = state_dict[key]
                        if torch.is_tensor(logZ_tensor):
                            logZ_val = logZ_tensor.item() if logZ_tensor.numel() == 1 else logZ_tensor[0].item()
                            break
                        elif isinstance(logZ_tensor, (int, float)):
                            logZ_val = float(logZ_tensor)
                            break
            
            if logZ_val is None and 'metrics' in checkpoint:
                metrics = checkpoint['metrics']
                if isinstance(metrics, dict) and 'logZ' in metrics:
                    logZ_metric = metrics['logZ']
                    if isinstance(logZ_metric, list) and len(logZ_metric) > 0:
                        logZ_val = float(logZ_metric[-1])
                    elif torch.is_tensor(logZ_metric):
                        logZ_val = logZ_metric.item() if logZ_metric.numel() == 1 else logZ_metric[-1].item() if logZ_metric.numel() > 0 else None
                    elif isinstance(logZ_metric, (int, float)):
                        logZ_val = float(logZ_metric)
        
        if isinstance(checkpoint, dict):
            if 'pf_model_state_dict' in checkpoint:
                trainer.gfn.pf_model.load_state_dict(checkpoint['pf_model_state_dict'], strict=False)
                if 'pb_model_state_dict' in checkpoint:
                    trainer.gfn.pb_model.load_state_dict(checkpoint['pb_model_state_dict'])
            elif 'model_state_dict' in checkpoint:
                trainer.gfn.pf_model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            elif 'state_dict' in checkpoint:
                trainer.gfn.pf_model.load_state_dict(checkpoint['state_dict'], strict=False)
            else:
                logging.warning("Checkpoint format not recognized, attempting direct load")
                trainer.gfn.pf_model.load_state_dict(checkpoint, strict=False)
        else:
            trainer.gfn.pf_model.load_state_dict(checkpoint, strict=False)
        
        if hasattr(trainer.gfn.pf_model, 'logZ') and logZ_val is not None:
            with torch.no_grad():
                trainer.gfn.pf_model.logZ.data.fill_(logZ_val)
            logging.info(f"✓ Transferred logZ from source checkpoint: {logZ_val:.6f}")
        elif hasattr(trainer.gfn.pf_model, 'logZ'):
            current_logZ = trainer.gfn.pf_model.logZ.data.item()
            logging.warning(f"⚠ Could not extract logZ from checkpoint, current value: {current_logZ:.6f}")
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
    """Main experiment runner with optional asynchronous evaluation."""
    start_update = 0
    existing_results = []
    existing_metrics = defaultdict(list)
    results_dir = None
    hyperparameters = None
    load_weights_only = False
    
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
    
    is_requeued_job = int(os.environ.get('SLURM_RESTART_COUNT', '0')) > 0
    
    if config.resume or config.experiment_dir:
        experiment_to_check = None
        
        if config.experiment_dir:
            exp_dir_path = Path(config.experiment_dir)
            if exp_dir_path.is_absolute() or config.experiment_dir.startswith('results'):
                experiment_to_check = exp_dir_path
            else:
                experiment_to_check = Path(config.results_dir) / config.experiment_dir
            
            if not experiment_to_check.exists():
                logging.info(f"Specified experiment directory {experiment_to_check} not found.")
                experiment_to_check = None
            else:
                logging.info(f"Found source experiment at {experiment_to_check}")
        else:
            experiment_to_check = find_latest_experiment(config.results_dir)
        
        if experiment_to_check:
            with open(experiment_to_check / 'config.json', 'r') as f:
                saved_config = json.load(f)
            
            nn_match, all_match, max_depth_changed = check_config_compatibility(saved_config, config)
            
            # For requeued SLURM jobs, always resume to prevent duplicate experiments
            if is_requeued_job and nn_match:
                logging.info(f"\n{'='*60}")
                logging.info(f"REQUEUED JOB: Forcing resume of {experiment_to_check}")
                logging.info(f"{'='*60}\n")
                results_dir = experiment_to_check
                start_update, existing_metrics, existing_results, hyperparameters = load_experiment_state(results_dir)
                logging.info(f"Resuming from update {start_update + 1}")
                if not all_match:
                    logging.info("Note: Some config parameters differ but resuming anyway (requeued job)")
            elif all_match:
                # Full compatibility - resume training
                logging.info(f"All hyperparameters match. Resuming training from {experiment_to_check}")
                results_dir = experiment_to_check
                start_update, existing_metrics, existing_results, hyperparameters = load_experiment_state(results_dir)
                logging.info(f"Starting from update {start_update + 1}")
                
            elif nn_match:
                if max_depth_changed and not config.transfer_weights_on_depth_change:
                    logging.info(f"max_depth changed ({saved_config.get('max_depth')} → {config.max_depth}) "
                                 f"and transfer_weights_on_depth_change=False")
                    logging.info("Starting completely new experiment WITHOUT weight transfer.")
                else:
                    logging.info(f"Neural network hyperparameters match. Will load weights from {experiment_to_check}")
                    logging.info("Other hyperparameters differ - starting new experiment with transferred weights")
                    load_weights_only = True
                    checkpoint_to_load = experiment_to_check
                
                logging.info("\nDiffering parameters:")
                for param in ['hamiltonian_path', 'n_measurements', 'max_depth', 'beta',
                             'lr', 'weight_decay', 'reward_type', 'cost_type', 'cost_kwargs']:
                    if saved_config.get(param) != getattr(config, param):
                        logging.info(f"  {param}: {saved_config.get(param)} → {getattr(config, param)}")
            else:
                logging.info("Neural network architecture differs. Starting completely new experiment.")
                logging.info(f"  Previous: hidden_dim={saved_config.get('hidden_dim')}, "
                      f"num_hidden_layers={saved_config.get('num_hidden_layers')}")
                logging.info(f"  Current: hidden_dim={config.hidden_dim}, "
                      f"num_hidden_layers={config.num_hidden_layers}")
    
    if results_dir is None:
        results_dir = Path(config.results_dir) / f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        results_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"Starting new experiment in {results_dir}")
        
        with open(results_dir / 'config.json', 'w') as f:
            json.dump(asdict(config), f, indent=2, cls=NumpyEncoder)
    
    logging.info(f"\nExperiment details:")
    logging.info(f"Hamiltonian: {config.hamiltonian_path}")
    logging.info(f"Batch structure: {config.n_measurements} circuits per batch element")
    logging.info(f"Evaluation: Top {config.n_eval_top_k_batch_elements} batch elements")
    if config.n_simulations > 1:
        logging.info(f"Simulations: {config.n_simulations} runs per batch element")
    if config.cost_kwargs:
        logging.info(f"Cost function: {config.cost_type} with kwargs: {config.cost_kwargs}")
    if config.async_eval:
        logging.info(f"Asynchronous evaluation: ENABLED")
    if start_update > 0:
        logging.info(f"Resuming from update: {start_update + 1}/{config.n_updates}")
    elif load_weights_only:
        logging.info(f"Starting fresh training with weights from: {checkpoint_to_load.name}")
    
    hamiltonian_helper = PauliHamiltonianHelper(config.hamiltonian_path)
    logging.info(f"Hamiltonian: {hamiltonian_helper}")
    logging.info(f"Exact ground state energy: {hamiltonian_helper.ground_state_energy:.10f}")
    
    if start_update == 0:
        with open(results_dir / 'hamiltonian_info.json', 'w') as f:
            json.dump(hamiltonian_helper.summary(), f, indent=2, cls=NumpyEncoder)

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
    
    if hyperparameters is None:
        hyperparameters = create_hyperparameters_dict(config, hamiltonian_helper, 
                                                      training_pauli_strings, identity_weight)
        with open(results_dir / 'hyperparameters.json', 'w') as f:
            json.dump(hyperparameters, f, indent=2, cls=NumpyEncoder)
    
    result_store = DiskBackedResultStore(results_dir / 'evaluation_results.jsonl')
    metric_store = DiskBackedMetricStore(results_dir / 'metrics_history.jsonl')

    if existing_metrics and metric_store.is_empty():
        metric_store.replace(existing_metrics)

    visualizer = AsyncVisualizer(results_dir, hyperparameters, result_store)
    if existing_results and result_store.is_empty():
        visualizer.add_results(existing_results)
        existing_results = []
    
    logging.info(f"Training with {len(training_pauli_strings)} non-identity Pauli terms")
    if identity_weight != 0:
        logging.info(f"Note: Identity term contributes a constant energy offset of {identity_weight:.6f}")
    
    gfn_config = {
        "model": {
            "model_type": config.model_type,
            "hidden_dim": config.hidden_dim,
            "num_hidden_layers": config.num_hidden_layers,
            "lr": config.lr,
            "weight_decay": config.weight_decay,
            "model_dir": str(results_dir),
            "model_kwargs": {},
            "objective_type": config.objective_type,
            "objective_kwargs": config.objective_kwargs,
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
            "w_list": training_weights
        }
    }
    
    device = get_device(config.device_preference)
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
    
    global _current_trainer, _current_results_dir
    _current_trainer = trainer
    _current_results_dir = results_dir
    
    if start_update > 0:
        checkpoint_files = list(results_dir.glob('checkpoint_update*.pth'))
        if checkpoint_files:
            latest_checkpoint = max(checkpoint_files, key=lambda f: f.stat().st_mtime)
            logging.info(f"Loading full checkpoint from update {start_update}")
            try:
                trainer.gfn.load_checkpoint(str(latest_checkpoint))
                trainer.ingest_metrics(existing_metrics)
            except Exception as e:
                logging.warning(f"Failed to load checkpoint: {e}")
                # Try manual loading
                checkpoint = torch.load(latest_checkpoint, map_location=trainer.device)
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    trainer.gfn.model.load_state_dict(checkpoint['model_state_dict'])
                    if 'optimizer_state_dict' in checkpoint:
                        trainer.gfn.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    trainer.ingest_metrics(existing_metrics)
                    
    elif load_weights_only:
        checkpoint_files = list(checkpoint_to_load.glob('checkpoint_update*.pth'))
        if checkpoint_files:
            latest_checkpoint = max(checkpoint_files, key=lambda f: f.stat().st_mtime)
            success = load_checkpoint_weights_only(trainer, latest_checkpoint, trainer.device)
            if success:
                if 'training' not in hyperparameters:
                    hyperparameters['training'] = {}
                hyperparameters['training']['weights_transferred_from'] = str(checkpoint_to_load.name)
                hyperparameters['training']['transfer_checkpoint'] = str(latest_checkpoint.name)
                
                with open(results_dir / 'hyperparameters.json', 'w') as f:
                    json.dump(hyperparameters, f, indent=2, cls=NumpyEncoder)
    
    if start_update == 0:
        training_hyperparams = {
            "actual_device": str(trainer.device),
            "num_actions": trainer.gfn.num_actions,
            "state_dim": trainer.gfn.state_dim,
            "action_mapping_size": len(trainer.gfn.action_mapping),
            "gfn_config": gfn_config,
            "weights_loaded_from": str(checkpoint_to_load.name) if load_weights_only else None,
            "energy_estimator": "EnergyEstimator with batched Clifford map",
            "async_eval_enabled": config.async_eval
        }
        
        with open(results_dir / 'training_hyperparameters.json', 'w') as f:
            json.dump(training_hyperparams, f, indent=2, cls=NumpyEncoder)
    
    if config.async_eval:
        checkpoint_queue = mp.Queue()
        results_queue = mp.Queue()
        
        evaluator_process = mp.Process(
            target=evaluator_loop,
            args=(config, results_dir, hamiltonian_helper, checkpoint_queue, results_queue)
        )
        evaluator_process.start()
        logging.info("Started asynchronous evaluator process")
        
        async def check_evaluation_results():
            """Check for results from evaluator process."""
            results_received = False
            while True:
                try:
                    msg_type, update, results = results_queue.get_nowait()
                    if msg_type == 'RESULTS':
                        logging.info(f"Received evaluation results for update {update}")
                        # Persist and update visualizations immediately
                        visualizer.add_results(results)
                        await visualizer.update_plots_async()
                        visualizer.export_results_json(results_dir / 'evaluation_results.json')
                        
                        logging.info(f"Updated plots and saved results for update {update}")
                        results_received = True
                except queue.Empty:
                    break
            
            if results_received:
                logging.info("Visualization plots have been updated")
            
            return results_received
    else:
        energy_estimator = EnergyEstimator(
            hamiltonian_helper, 
            hamiltonian_helper.n_qubits, 
            device
        )
        checkpoint_queue = None
        results_queue = None
        evaluator_process = None
    
    logging.info(f"\n=== {'Resuming' if start_update > 0 else 'Starting'} Training ===")
    if load_weights_only:
        logging.info("Note: Using transferred neural network weights")
    
    cost_compute_kwargs = {k: v for k, v in config.cost_kwargs.items() 
                          if k != "normalization_type"}
    
    for update in range(start_update, config.n_updates):
        update_start = time.time()
        
        trajectory_batch = trainer.gfn.sample_trajectories(
            batch_size=config.update_freq,
            n_measurements=config.n_measurements,
            max_depth=config.max_depth,
            mode=SamplingMode.ON_POLICY
        )
        
        costs = trainer.compute_costs_with_probabilities(
            trajectory_batch.batched_tableau, 
            **cost_compute_kwargs
        )
        
        loss, metrics = trainer.gfn.compute_loss(
            trajectory_batch, costs, config.beta, max_depth=config.max_depth, **config.reward_kwargs
        )
        trainer.gfn.update_step(loss)
        trainer.gfn._update_top_trajectories(trajectory_batch, costs)
        
        for k, v in metrics.items():
            if torch.is_tensor(v):
                v_detached = v.detach()
                metrics[k] = v_detached
                trainer.metrics_history[k].append(v_detached)
            else:
                trainer.metrics_history[k].append(v)

            if trainer.metrics_window and len(trainer.metrics_history[k]) > trainer.metrics_window:
                del trainer.metrics_history[k][:-trainer.metrics_window]

        if (update + 1) % 100 == 0:
            loss_val = metrics['loss'].item() if torch.is_tensor(metrics['loss']) else metrics['loss']
            reward_val = metrics['reward'].item() if torch.is_tensor(metrics['reward']) else metrics['reward']
            cost_val = metrics['cost'].item() if torch.is_tensor(metrics['cost']) else metrics['cost']
            logging.info(f"Update {update + 1}/{config.n_updates}: "
                  f"Loss={loss_val:.6f}, "
                  f"Reward={reward_val:.4f}, "
                  f"Cost={cost_val:.4f}")
        
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
                silence=False,
                **cost_compute_kwargs
            )
            replay_loss, replay_metrics = trainer.gfn.compute_loss(
                replay_batch, replay_costs, config.beta, max_depth=config.max_depth, **config.reward_kwargs
            )
            trainer.gfn.update_step(replay_loss)

            for k, v in replay_metrics.items():
                if torch.is_tensor(v):
                    v_detached = v.detach()
                    replay_metrics[k] = v_detached
                    trainer.metrics_history[f'replay_{k}'].append(v_detached)
                else:
                    trainer.metrics_history[f'replay_{k}'].append(v)

                hist = trainer.metrics_history[f'replay_{k}']
                if trainer.metrics_window and len(hist) > trainer.metrics_window:
                    del hist[:-trainer.metrics_window]

        if config.offpolicy_every and (update + 1) % config.offpolicy_every == 0:
            offpolicy_batch = trainer.gfn.sample_trajectories(
                batch_size=config.update_freq,
                n_measurements=config.n_measurements,
                max_depth=config.max_depth,
                mode=SamplingMode.OFF_POLICY
            )

            offpolicy_costs = trainer.compute_costs_with_probabilities(
                offpolicy_batch.batched_tableau,
                **cost_compute_kwargs
            )
            offpolicy_loss, offpolicy_metrics = trainer.gfn.compute_loss(
                offpolicy_batch, offpolicy_costs, config.beta, max_depth=config.max_depth, **config.reward_kwargs
            )
            trainer.gfn.update_step(offpolicy_loss)

            for k, v in offpolicy_metrics.items():
                if torch.is_tensor(v):
                    v_detached = v.detach()
                    offpolicy_metrics[k] = v_detached
                    trainer.metrics_history[f'offpolicy_{k}'].append(v_detached)
                else:
                    trainer.metrics_history[f'offpolicy_{k}'].append(v)

                hist = trainer.metrics_history[f'offpolicy_{k}']
                if trainer.metrics_window and len(hist) > trainer.metrics_window:
                    del hist[:-trainer.metrics_window]

        if metric_store is not None:
            metrics_to_buffer = dict(metrics)

            if config.replay_every and (update + 1) % config.replay_every == 0 and trainer.gfn.top_trajectories_actions:
                for k, v in replay_metrics.items():
                    metrics_to_buffer[f'replay_{k}'] = v
            if config.offpolicy_every and (update + 1) % config.offpolicy_every == 0:
                for k, v in offpolicy_metrics.items():
                    metrics_to_buffer[f'offpolicy_{k}'] = v

            metric_store.append_to_buffer(update + 1, metrics_to_buffer)

        if (update + 1) % config.checkpoint_every == 0 and metric_store is not None:
            num_flushed = metric_store.flush_buffer()
            if num_flushed:
                logging.info(f"Flushed {num_flushed} iterations of metrics to disk")
        
        if (update + 1) % config.checkpoint_every == 0:
            checkpoint_path = results_dir / f'checkpoint_update.pth'
            metrics_history_cpu = convert_metrics_history_to_cpu(trainer.metrics_history)
            trainer.gfn.save_checkpoint(str(checkpoint_path), update + 1, metrics_history_cpu)
            trainer.plot_metrics(update + 1)
            
            if config.async_eval and checkpoint_queue and (update + 1) % config.eval_every == 0:
                checkpoint_data = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
                checkpoint_id = checkpoint_data.get('checkpoint_id', time.time())
                checkpoint_queue.put((str(checkpoint_path), update + 1, checkpoint_id))
                logging.info(f"Checkpoint saved and queued for evaluation at update {update + 1}")
            else:
                logging.info(f"Checkpoint saved at update {update + 1}")
        
        if (update + 1) % config.eval_every == 0:
            if config.async_eval:
                if (update + 1) % config.checkpoint_every != 0:
                    checkpoint_path = results_dir / f'checkpoint_update.pth'
                    metrics_history_cpu = convert_metrics_history_to_cpu(trainer.metrics_history)
                    trainer.gfn.save_checkpoint(str(checkpoint_path), update + 1, metrics_history_cpu)
                    
                    checkpoint_data = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
                    checkpoint_id = checkpoint_data.get('checkpoint_id', time.time())
                    checkpoint_queue.put((str(checkpoint_path), update + 1, checkpoint_id))
                    logging.info(f"Special evaluation checkpoint saved and queued at update {update + 1}")
                
                await check_evaluation_results()
                logging.info(f"\nAsync evaluation in progress for update {update + 1}")
            else:
                logging.info(f"\n{'='*60}")
                logging.info(f"Checkpoint evaluation at update {update + 1}")
                
                from main import evaluate_top_batch_elements
                
                evaluation_results = await evaluate_top_batch_elements(
                    trainer, energy_estimator, update + 1, config
                )
                
                if evaluation_results:
                    visualizer.add_results(evaluation_results)
                    await visualizer.update_plots_async()
                    visualizer.export_results_json(results_dir / 'evaluation_results.json')

                logging.info(f"{'='*60}\n")
        
        if config.async_eval and (update + 1) % 100 == 0:
            await check_evaluation_results()
    
    if metric_store is not None:
        num_remaining = metric_store.flush_buffer()
        if num_remaining:
            logging.info(f"Flushed {num_remaining} remaining metrics at end of training")

    if config.async_eval and evaluator_process:
        logging.info("\nTraining complete. Waiting for evaluator to finish...")
        checkpoint_queue.put('STOP')
        evaluator_process.join(timeout=config.eval_process_timeout)
        
        if evaluator_process.is_alive():
            logging.warning("Evaluator process did not finish within timeout. Terminating...")
            evaluator_process.terminate()
            evaluator_process.join(timeout=10)
            
            if evaluator_process.is_alive():
                logging.error("Evaluator process could not be terminated. Killing...")
                evaluator_process.kill()
        
        await check_evaluation_results()
    
    if config.n_updates % config.eval_every != 0:
        if config.async_eval:
            checkpoint_path = results_dir / f'checkpoint_update.pth'
            metrics_history_cpu = convert_metrics_history_to_cpu(trainer.metrics_history)
            trainer.gfn.save_checkpoint(str(checkpoint_path), config.n_updates, metrics_history_cpu)

            logging.info("Waiting for final async evaluation...")
            await asyncio.sleep(config.eval_poll_interval * 2)
            await check_evaluation_results()
        else:
            logging.info("\n=== Final Evaluation ===")
            final_results = await evaluate_top_batch_elements(
                trainer, energy_estimator, config.n_updates, config
            )

            if final_results:
                visualizer.add_results(final_results)

    await visualizer.update_plots_async()
    visualizer.export_results_json(results_dir / 'evaluation_results.json')
    visualizer.export_results_json(results_dir / 'all_evaluation_results.json')
    
    full_results = visualizer.load_all_results()
    generate_final_report(full_results, hamiltonian_helper, results_dir, hyperparameters, trainer)
    
    logging.info(f"\nExperiment completed! Results saved to {results_dir}")
    
    return full_results, trainer


def generate_final_report(results: List[ExtendedBatchElementEnergyResult], 
                         hamiltonian_helper: PauliHamiltonianHelper,
                         results_dir: Path,
                         hyperparameters: Dict,
                         trainer: EfficientGFNTrainer):
    """Generate a final summary report."""
    report_lines = [
        "# GFlowNet Quantum Circuit Optimization Experiment Report",
        f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        
        f"\n## Hamiltonian Information",
        f"- File: {hamiltonian_helper.filepath}",
        f"- Number of qubits: {hamiltonian_helper.n_qubits}",
        f"- Number of Pauli terms: {len(hamiltonian_helper.pauli_str_list)}",
        f"- Number of training terms: {hyperparameters['hamiltonian']['n_training_terms']} (excluding identity)",
        f"- Identity term weight: {hyperparameters['hamiltonian']['identity_weight']:.6f}",
        f"- Exact ground state energy: {hamiltonian_helper.ground_state_energy:.10f}",
        
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
        f"- Device: {hyperparameters['computational']['device']['type']}",
        f"- Async evaluation: {hyperparameters['computational'].get('async_evaluation', False)}",
        
        f"\n## Energy Estimation Method",
        f"- Method: {hyperparameters['energy_estimation']['method']}",
        f"- Implementation: {hyperparameters['energy_estimation'].get('implementation', 'N/A')}",
        f"- Equation: {hyperparameters['energy_estimation']['equation']}",
        f"- Reference: {hyperparameters['energy_estimation']['reference']}",
    ]
    
    if 'simulations' in hyperparameters['energy_estimation'] and hyperparameters['energy_estimation']['simulations'] > 1:
        report_lines.extend([
            f"- Simulations per batch element: {hyperparameters['energy_estimation']['simulations']}",
            f"- Error formula: {hyperparameters['energy_estimation']['error_formula']}",
        ])
    
    if hyperparameters['computational'].get('async_evaluation', False):
        report_lines.append(f"- Evaluation mode: Asynchronous (CPU-based parallel evaluation)")
    
    report_lines.extend([
        "\n## Experiment Summary",
        f"- Total training updates: {max(r.update for r in results) if results else 0}",
        f"- Total batch elements evaluated: {len(results)}",
    ])
    
    # Add simulation info if available
    if 'simulations' in hyperparameters.get('energy_estimation', {}) and hyperparameters['energy_estimation']['simulations'] > 1:
        n_sims = hyperparameters['energy_estimation']['simulations']
        report_lines.append(f"- Total simulations run: {len(results) * n_sims}")
    
    report_lines.extend([
        f"- Evaluation frequency: every {hyperparameters['experiment']['eval_every']} updates",
        f"- Batch elements per evaluation: {hyperparameters['experiment']['n_eval_top_k_batch_elements']}",
    ])
    
    if not results:
        report_lines.append("\nNo evaluation results were generated.")
        with open(results_dir / 'experiment_report.md', 'w') as f:
            f.write('\n'.join(report_lines))
        return

    final_update = max(r.update for r in results)
    final_results = [r for r in results if r.update == final_update]
    
    if final_results:
        energy_diffs = [r.energy_difference for r in final_results]
        best_result = min(final_results, key=lambda r: r.energy_difference)
        
        total_circuits = sum(r.n_circuits for r in final_results)
        total_measurements = sum(r.total_measurements for r in final_results)
        mean_coverage = np.mean([r.convergence_metrics['coverage'] for r in final_results])
        
        mae_results = [r for r in final_results if hasattr(r, 'simulation_result') and r.simulation_result is not None]
        if mae_results:
            mae_values = [r.simulation_result.mean_absolute_error for r in mae_results]
            mae_stds = [r.simulation_result.std_absolute_error for r in mae_results]
            best_mae = min(mae_values)
            mean_mae = np.mean(mae_values)
        
        report_lines.extend([
            f"\n## Final Results (Update {final_update})",
            f"- Batch elements evaluated: {len(final_results)}",
            f"- Total circuits: {total_circuits}",
            f"- Total measurements: {total_measurements}",
            f"- Best energy difference (Batch rank {best_result.batch_element_rank}): {best_result.energy_difference:.6e}",
            f"- Mean energy difference: {np.mean(energy_diffs):.6e}",
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
        
        if hasattr(trainer, 'metrics_history') and trainer.metrics_history:
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
                
                mae_results = [r for r in update_results if hasattr(r, 'simulation_result') and r.simulation_result is not None]
                if mae_results:
                    best_mae = min(r.simulation_result.mean_absolute_error for r in mae_results)
                    mae_str = f"{best_mae:.2e}"
                else:
                    mae_str = "N/A"
                
                report_lines.append(f"| {update:6d} | {best_energy:15.6e} | {mean_energy:16.6e} | {mae_str:8s} | {total_circuits:14d} | {mean_coverage:14.1%} |")
    
    with open(results_dir / 'experiment_report.md', 'w') as f:
        f.write('\n'.join(report_lines))
    
    logging.info(f"\nFinal report saved to {results_dir / 'experiment_report.md'}")


async def evaluate_top_batch_elements(trainer: EfficientGFNTrainer,
                                    energy_estimator: EnergyEstimator,
                                    update: int,
                                    config: ExperimentConfig) -> List[ExtendedBatchElementEnergyResult]:
    """Evaluate top-k batch elements from replay buffer using energy estimation."""
    logging.info(f"\n=== Evaluating top batch elements at update {update} ===")
    
    top_actions = trainer.gfn.top_trajectories_actions
    top_lengths = trainer.gfn.top_trajectories_lengths
    top_costs = trainer.gfn.top_trajectories_costs if hasattr(trainer.gfn, 'top_trajectories_costs') else None
    
    if not top_actions:
        logging.info("  Replay buffer is empty. Skipping evaluation.")
        return []
    
    num_to_eval = len(top_actions)
    logging.info(f"  Evaluating top {num_to_eval} batch elements from replay buffer")
    
    n_simulations = getattr(config, 'n_simulations', 1)
    if n_simulations > 1:
        logging.info(f"  Running {n_simulations} simulations per batch element")

    batch_actions_list = []
    batch_lengths_list = []
    
    for batch_idx in range(num_to_eval):
        batch_actions = top_actions[batch_idx]
        batch_lengths = top_lengths[batch_idx]
        
        if len(batch_actions.shape) == 1:
            batch_actions = batch_actions.unsqueeze(0)
            batch_lengths = batch_lengths.unsqueeze(0) if isinstance(batch_lengths, torch.Tensor) else torch.tensor([batch_lengths])
        
        batch_actions_list.append(batch_actions)
        batch_lengths_list.append(batch_lengths)
    
    all_batch_actions = torch.stack(batch_actions_list, dim=0)
    all_batch_lengths = torch.stack(batch_lengths_list, dim=0)
    
    logging.info(f"  Combined batch shape: {all_batch_actions.shape}")
    
    summaries = await energy_estimator.estimate_energy_with_simulations(
        all_batch_actions,
        all_batch_lengths,
        M=n_simulations,
    )
    
    results = []
    
    for batch_idx, summary in enumerate(summaries):
        final_result_obj = summary['final_results_object']
        
        simulation_result = None
        if n_simulations > 1 and 'energy_variance' in summary:
            mean_energy = summary['mean_energy']
            variance = summary['energy_variance']

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
                logging.warning(f"Individual simulation energies not available for batch {batch_idx}")
                simulation_result = None
        
        result = ExtendedBatchElementEnergyResult(
            batch_element_rank=batch_idx,
            energy_estimate=summary['mean_energy'],
            energy_difference=summary['energy_difference'],
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
        
        if top_costs is not None and batch_idx < len(top_costs):
            result.batch_cost = top_costs[batch_idx].item() if torch.is_tensor(top_costs[batch_idx]) else top_costs[batch_idx]
        
        results.append(result)
        
        logging.info(f"\n  Batch element rank {batch_idx}:")
        logging.info(f"    Number of circuits: {result.n_circuits}")
        logging.info(f"    Energy estimate: {result.energy_estimate:.6f}")
        logging.info(f"    Energy difference: {result.energy_difference:.6e}")
        if simulation_result and n_simulations > 1:
            logging.info(f"    Standard deviation: {simulation_result.std_absolute_error:.6e}")
        logging.info(f"    Pauli coverage: {result.convergence_metrics['coverage']:.1%}")
        logging.info(f"    Mean circuit length: {result.mean_circuit_length:.1f}")
    
    if results:
        energy_diffs = [r.energy_difference for r in results]
        best_result = min(results, key=lambda r: r.energy_difference)
        
        logging.info(f"\n  Evaluation Summary:")
        logging.info(f"    Best energy difference (Batch rank {best_result.batch_element_rank}): {best_result.energy_difference:.6e}")
        logging.info(f"    Mean energy difference: {np.mean(energy_diffs):.6e}")
        logging.info(f"    Total circuits evaluated: {sum(r.n_circuits for r in results)}")
        if n_simulations > 1:
            logging.info(f"    Total simulation runs: {len(results) * n_simulations}")
        logging.info(f"    Success rate (< 1.6e-3): {sum(1 for e in energy_diffs if e < 1.6e-3) / len(energy_diffs) * 100:.1f}%")
    
    return results


if __name__ == "__main__":
    # Example configuration
    config = ExperimentConfig(
        model_type= "clifford_deepsets",          # Model architecture
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
