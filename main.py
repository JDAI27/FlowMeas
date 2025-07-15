#!/usr/bin/env python3
#main.py

"""
Main experiment runner for GFlowNet quantum circuit optimization with energy estimation.
Trains a single GFlowNet and performs energy estimations using derandomized circuits.
Now includes multiple simulation runs to compute average estimation error.

ADAPTED: Now uses the new EnergyEstimator API with estimate_energy_with_simulations.
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

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend - must be before pyplot import
import matplotlib.pyplot as plt


from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, asdict, field
from concurrent.futures import ThreadPoolExecutor
import threading
from collections import defaultdict

# Import the GFN modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from GFNs import EfficientGFNTrainer, get_device, exponential_reward_fn, SamplingMode, default_reward_fn, threshold_reward_fn
from clifford_map import CliffordMap
from pauli_hamiltonian_helper import PauliHamiltonianHelper
from quantum_action_mapping import build_action_mapping

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Import from energy_estimator
from energy_estimator import EnergyEstimator, BatchElementEnergyResult


@dataclass
class SimulationResult:
    """Results from multiple simulation runs"""
    energy_estimates: List[float]
    absolute_errors: List[float]
    mean_absolute_error: float
    std_absolute_error: float
    mean_energy_estimate: float
    std_energy_estimate: float
    

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
    beta: float = 2e4
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
    reward_type: str = "default"  # Options: "default", "exponential", "threshold"
    reward_kwargs: Dict = field(default_factory=lambda: {"alpha": 2.0})
    cost_type: str = "exponential"
    cost_kwargs: Dict = field(default_factory=dict)  # Arguments for cost computation
    objective_type: str = "tb"
    objective_kwargs: Dict = field(default_factory=lambda: {"loss_type": "squared"})
    n_simulations: int = 10  # Number of simulation runs for error estimation
    resume: bool = True  # Whether to resume from existing experiment
    experiment_dir: Optional[str] = None  # Specific experiment directory to resume from

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


class AsyncVisualizer:
    """Handle asynchronous visualization updates"""
    
    def __init__(self, results_dir: Path, hyperparameters: Dict = None):
        self.results_dir = results_dir
        self.fig_dir = results_dir / "figures"
        self.fig_dir.mkdir(exist_ok=True)
        self.hyperparameters = hyperparameters
        
        # Data storage
        self.all_results = []
        self.lock = threading.Lock()
        
    def add_results(self, results: List[ExtendedBatchElementEnergyResult]):
        """Thread-safe addition of results"""
        with self.lock:
            self.all_results.extend(results)
    
    async def update_plots_async(self):
        """Update plots asynchronously"""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._update_plots)
    
    def _update_plots(self):
        """Generate all plots"""
        with self.lock:
            if not self.all_results:
                return
            
            results_copy = self.all_results.copy()
        
        # Get unique updates
        updates = sorted(set(r.update for r in results_copy))
        
        # Plot 1: Energy convergence across batch elements with error bars
        plt.figure(figsize=(14, 10))
        
        for update in updates:
            update_results = [r for r in results_copy if r.update == update]
            
            # Plot batch element energies
            batch_ranks = [r.batch_element_rank for r in update_results]
            energy_diffs = [r.energy_difference for r in update_results]
            n_circuits = [r.n_circuits for r in update_results]
            
            # Get error bars from simulations if available
            error_bars = []
            for r in update_results:
                if hasattr(r, 'simulation_result') and r.simulation_result:
                    error_bars.append(r.simulation_result.std_absolute_error)
                else:
                    error_bars.append(0)
            
            # Size points by number of circuits
            sizes = [n * 10 for n in n_circuits]
            
            plt.errorbar(batch_ranks, energy_diffs, yerr=error_bars, 
                        fmt='o', alpha=0.6, capsize=5, markersize=8,
                        label=f'Update {update}')
        
        plt.xlabel('Batch Element Rank (0 = best)')
        plt.ylabel('Energy Difference from Ground State')
        plt.title('Energy Estimation Results by Batch Element (with simulation error bars)')
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
        #plt.legend()
        
        # Add text explaining error bars
        plt.text(0.02, 0.02, 'Error bars show std dev across simulations', 
                transform=plt.gca().transAxes, fontsize=10)
        
        plt.tight_layout()
        plt.savefig(self.fig_dir / 'batch_energy_convergence.png', dpi=300)
        plt.close()
        
        # Plot 2: Best energy vs training progress
        plt.figure(figsize=(12, 8))
        
        best_energies_per_update = []
        best_mae_per_update = []
        update_labels = []
        total_measurements_per_update = []
        
        for update in updates:
            update_results = [r for r in results_copy if r.update == update]
            if update_results:
                # Find best batch element
                best_result = min(update_results, key=lambda r: r.energy_difference)
                best_energies_per_update.append(best_result.energy_difference)
                update_labels.append(update)
                
                # Get mean absolute error if available
                if hasattr(best_result, 'simulation_result') and best_result.simulation_result:
                    best_mae_per_update.append(best_result.simulation_result.mean_absolute_error)
                else:
                    best_mae_per_update.append(None)
                
                # Total measurements across all evaluated batch elements
                total_measurements = sum(r.total_measurements for r in update_results)
                total_measurements_per_update.append(total_measurements)
        
        plt.plot(update_labels, best_energies_per_update, 'o-', linewidth=2, markersize=8, label='Energy difference')
        
        # Plot MAE if available
        mae_values = [mae for mae in best_mae_per_update if mae is not None]
        mae_updates = [u for u, mae in zip(update_labels, best_mae_per_update) if mae is not None]
        if mae_values:
            plt.plot(mae_updates, mae_values, 's--', linewidth=2, markersize=6, 
                    label='Mean absolute error (simulations)', alpha=0.7)
        
        plt.xlabel('Training Update')
        plt.ylabel('Energy Difference / Error')
        plt.title('Best Batch Element Energy vs Training Progress')
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
        
        # Add success threshold lines
        plt.axhline(y=1.6e-3, color='g', linestyle='--', alpha=0.5, label='Chemical accuracy (1.6e-3)')
        plt.axhline(y=1e-2, color='orange', linestyle='--', alpha=0.5, label='1e-2 threshold')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(self.fig_dir / 'best_energy_vs_training.png', dpi=300)
        plt.close()
        
        # Plot 3: Mean Absolute Error distribution
        plt.figure(figsize=(12, 8))
        
        # Get MAE values from latest update
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
                
                # Add mean MAE line
                mean_mae = np.mean(mae_values)
                plt.axhline(y=mean_mae, color='r', linestyle='--', alpha=0.7, 
                           label=f'Mean MAE: {mean_mae:.2e}')
                plt.legend()
        
        plt.tight_layout()
        plt.savefig(self.fig_dir / 'mae_distribution.png', dpi=300)
        plt.close()
        
        # Plot 4: Pauli coverage analysis per batch element
        if results_copy and updates:
            latest_results = [r for r in results_copy if r.update == updates[-1]]
            
            if latest_results:
                # Determine number of subplots needed
                n_batch_elements = len(latest_results)
                n_cols = min(3, n_batch_elements)  # Max 3 columns
                n_rows = (n_batch_elements + n_cols - 1) // n_cols
                
                fig, axes = plt.subplots(n_rows, n_cols, figsize=(8*n_cols, 6*n_rows))
                
                # Handle single subplot case
                if n_batch_elements == 1:
                    axes = [axes]
                else:
                    axes = axes.flatten() if n_rows > 1 or n_cols > 1 else [axes]
                
                # Get all Pauli strings in consistent order
                all_paulis = set()
                for result in latest_results:
                    all_paulis.update(result.hitting_counts.keys())
                
                # Convert to list to maintain consistent order
                ordered_paulis = list(all_paulis)
                
                # Limit number of Paulis shown if too many
                max_paulis_shown = 30  # Adjust as needed
                if len(ordered_paulis) > max_paulis_shown:
                    top_paulis = ordered_paulis[:max_paulis_shown]
                    x_label = f'First {max_paulis_shown} Pauli Strings'
                else:
                    top_paulis = ordered_paulis
                    x_label = 'Pauli Strings'
                
                # Plot for each batch element
                for idx, result in enumerate(latest_results):
                    ax = axes[idx]
                    
                    # Get hitting counts for this batch element
                    hit_counts = [result.hitting_counts.get(p, 0) for p in top_paulis]
                    pauli_indices = list(range(len(top_paulis)))
                    
                    # Create bar plot
                    bars = ax.bar(pauli_indices, hit_counts, alpha=0.7)
                    
                    # Color bars by hit count
                    max_count = max(hit_counts) if hit_counts else 1
                    for bar, count in zip(bars, hit_counts):
                        if count > 0:
                            bar.set_color(plt.cm.viridis(count / max_count))
                    
                    # Set x-axis labels as Pauli strings
                    ax.set_xticks(pauli_indices)
                    ax.set_xticklabels(top_paulis, rotation=90, ha='right', fontsize=8)
                    
                    ax.set_xlabel(x_label, fontsize=10)
                    ax.set_ylabel('Hitting Count')
                    ax.set_title(f'Batch Element Rank {result.batch_element_rank}')
                    ax.grid(True, alpha=0.3, axis='y')
                    
                    # Add coverage text (for all Paulis, not just shown)
                    total_paulis_with_hits = sum(1 for p in ordered_paulis if result.hitting_counts.get(p, 0) > 0)
                    coverage = total_paulis_with_hits / len(ordered_paulis) if ordered_paulis else 0
                    ax.text(0.98, 0.98, f'Coverage: {coverage:.1%}', 
                           transform=ax.transAxes, ha='right', va='top',
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                    
                    # Add total circuits info
                    ax.text(0.02, 0.98, f'Circuits: {result.n_circuits}', 
                           transform=ax.transAxes, ha='left', va='top',
                           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
                
                # Hide extra subplots
                for idx in range(n_batch_elements, len(axes)):
                    axes[idx].set_visible(False)
                
                plt.suptitle(f'Pauli Measurement Coverage by Batch Element (Update {updates[-1]})', fontsize=16)
                plt.tight_layout()
                plt.savefig(self.fig_dir / 'pauli_coverage_per_batch.png', dpi=300)
                plt.close()
                
                # Also create a heatmap view with Pauli labels
                plt.figure(figsize=(16, 10))
                
                # Use subset of Paulis for heatmap too
                heatmap_paulis = ordered_paulis[:50]  # Show more in heatmap
                
                # Create hit count matrix
                hit_matrix = np.zeros((n_batch_elements, len(heatmap_paulis)))
                for i, result in enumerate(latest_results):
                    for j, pauli in enumerate(heatmap_paulis):
                        hit_matrix[i, j] = result.hitting_counts.get(pauli, 0)
                
                # Create heatmap
                im = plt.imshow(hit_matrix, aspect='auto', cmap='YlOrRd', interpolation='nearest')
                plt.colorbar(im, label='Hitting Count')
                
                # Set labels
                plt.xlabel('Pauli Strings')
                plt.ylabel('Batch Element Rank')
                plt.title(f'Pauli Hitting Count Heatmap (Update {updates[-1]})')
                
                # Add batch element ranks as y-tick labels
                plt.yticks(range(n_batch_elements), [r.batch_element_rank for r in latest_results])
                
                # Add Pauli strings as x-tick labels
                plt.xticks(range(len(heatmap_paulis)), heatmap_paulis, rotation=90, ha='right', fontsize=8)
                
                # Add grid
                plt.grid(True, alpha=0.3, which='both', linestyle='-', linewidth=0.5)
                
                # Add coverage annotations on the right
                for i, result in enumerate(latest_results):
                    total_paulis_with_hits = sum(1 for p in ordered_paulis if result.hitting_counts.get(p, 0) > 0)
                    coverage = total_paulis_with_hits / len(ordered_paulis) if ordered_paulis else 0
                    plt.text(len(heatmap_paulis) + 0.5, i, f'{coverage:.1%}', 
                            va='center', ha='left')
                
                # Add note if not all Paulis are shown
                if len(ordered_paulis) > len(heatmap_paulis):
                    plt.text(0.5, -0.15, f'Note: Showing {len(heatmap_paulis)} of {len(ordered_paulis)} total Pauli strings', 
                            transform=plt.gca().transAxes, ha='center', fontsize=10, style='italic')
                
                plt.tight_layout()
                plt.savefig(self.fig_dir / 'pauli_coverage_heatmap.png', dpi=300, bbox_inches='tight')
                plt.close()
                
                # Create a focused plot for top Paulis (by position in list, not sorted)
                plt.figure(figsize=(14, 8))
                
                # Show aggregated hits for first 20 Paulis
                top_paulis = ordered_paulis[:20]
                
                # Aggregate hitting counts across all batch elements
                total_hitting_counts = defaultdict(int)
                for result in latest_results:
                    for pauli, count in result.hitting_counts.items():
                        if pauli in top_paulis:
                            total_hitting_counts[pauli] += count
                
                # Create bar plot
                pauli_labels = top_paulis
                hit_counts = [total_hitting_counts[p] for p in pauli_labels]
                x_pos = np.arange(len(pauli_labels))
                
                bars = plt.bar(x_pos, hit_counts, alpha=0.7)
                
                # Color bars by coefficient magnitude if available
                if latest_results[0].pauli_estimates:
                    coeffs = [abs(latest_results[0].pauli_estimates.get(p, 0)) for p in pauli_labels]
                    max_coeff = max(coeffs) if coeffs else 1
                    for bar, coeff in zip(bars, coeffs):
                        bar.set_color(plt.cm.plasma(coeff / max_coeff))
                    
                    # Add colorbar for coefficient magnitude
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
        
        # Plot 5: Circuit length distribution
        plt.figure(figsize=(10, 6))
        
        for update in updates[-3:]:  # Last 3 updates
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
        
        # Save summary statistics
        self._save_summary_statistics(results_copy, updates)
    
    def _save_summary_statistics(self, results: List[ExtendedBatchElementEnergyResult], updates: List[int]):
        """Save summary statistics to JSON"""
        summary_stats = {
            'updates': updates,
            'n_batch_elements_per_update': {},
            'best_energy_per_update': {},
            'mean_energy_per_update': {},
            'best_mae_per_update': {},
            'mean_mae_per_update': {},
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
                
                # Add MAE statistics if available
                mae_results = [r for r in update_results if hasattr(r, 'simulation_result') and r.simulation_result is not None]
                if mae_results:
                    mae_values = [r.simulation_result.mean_absolute_error for r in mae_results]
                    summary_stats['best_mae_per_update'][str(update)] = min(mae_values)
                    summary_stats['mean_mae_per_update'][str(update)] = np.mean(mae_values)
        
        with open(self.results_dir / 'summary_statistics.json', 'w') as f:
            json.dump(summary_stats, f, indent=2, cls=NumpyEncoder)


async def evaluate_top_batch_elements(trainer: EfficientGFNTrainer,
                                    energy_estimator: EnergyEstimator,
                                    update: int,
                                    config: ExperimentConfig) -> List[ExtendedBatchElementEnergyResult]:
    """
    Evaluate top-k batch elements from replay buffer using energy estimation.
    Now adapted to use the new EnergyEstimator API.
    """
    
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
    
    # Run energy estimation for all batch elements at once
    summaries = await energy_estimator.estimate_energy_with_simulations(
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
            # Estimate individual simulation results from the variance
            # This is approximate since we don't have individual run data
            mean_energy = summary['mean_energy']
            variance = summary['energy_variance']
            
            # Generate approximate simulation results
            energy_estimates = [mean_energy] * n_simulations  # Simplified
            absolute_errors = [abs(mean_energy - energy_estimator.ground_state_energy)] * n_simulations
            
            simulation_result = SimulationResult(
                energy_estimates=energy_estimates,
                absolute_errors=absolute_errors,
                mean_absolute_error=summary['energy_difference'],
                std_absolute_error=np.sqrt(variance) if variance > 0 else 0.0,
                mean_energy_estimate=mean_energy,
                std_energy_estimate=np.sqrt(variance) if variance > 0 else 0.0
            )
        
        # Create extended result
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
        
        # Add cost if available
        if top_costs is not None and batch_idx < len(top_costs):
            result.batch_cost = top_costs[batch_idx].item() if torch.is_tensor(top_costs[batch_idx]) else top_costs[batch_idx]
        
        results.append(result)
        
        # Print summary for this batch element
        logging.info(f"\n  Batch element rank {batch_idx}:")
        logging.info(f"    Number of circuits: {result.n_circuits}")
        logging.info(f"    Energy estimate: {result.energy_estimate:.6f}")
        logging.info(f"    Energy difference: {result.energy_difference:.6e}")
        if simulation_result and n_simulations > 1:
            logging.info(f"    Standard deviation: {simulation_result.std_absolute_error:.6e}")
        logging.info(f"    Pauli coverage: {result.convergence_metrics['coverage']:.1%}")
        logging.info(f"    Mean circuit length: {result.mean_circuit_length:.1f}")
    
    # Print overall summary
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


def create_hyperparameters_dict(config: ExperimentConfig, 
                               hamiltonian_helper: PauliHamiltonianHelper,
                               training_pauli_strings: List[str],
                               identity_weight: float) -> Dict:
    """Create a comprehensive hyperparameters dictionary"""
    
    # Get device info
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
            "timestamp": datetime.now().isoformat()
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
            "model_type": "clifford_mlp",
            "hidden_dim": config.hidden_dim,
            "num_hidden_layers": config.num_hidden_layers,
            "lr": config.lr,
            "weight_decay": config.weight_decay,
            "grad_clip_value": 10.0,
            "weight_init": "xavier_uniform",
            "logZ_init": 0.0,
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
            "error_formula": "(1/S) ∑_{s=1}^S |⟨H⟩ - ô_N^(s)(H)|"
        },
        
        "computational": {
            "device": device_info,
            "batch_processing": True,
            "gpu_optimized": True,
            "sparse_matrices": True,
            "async_evaluation": True
        },
        
        "evaluation": {
            "energy_computation": "circuit_based_expectation_value",
            "convergence_threshold": 1e-3,
            "success_thresholds": [1.6e-3, 1e-2]
        }
    }
    
    return hyperparameters


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


def load_experiment_state(experiment_dir: Path) -> Tuple[int, Dict, List[ExtendedBatchElementEnergyResult], Dict]:
    """Load the state of a previous experiment"""
    logging.info(f"Loading experiment state from {experiment_dir}")
    
    # Load configuration
    with open(experiment_dir / 'config.json', 'r') as f:
        saved_config = json.load(f)
    
    # Load hyperparameters
    with open(experiment_dir / 'hyperparameters.json', 'r') as f:
        hyperparameters = json.load(f)
    
    # Find latest checkpoint
    checkpoint_files = list(experiment_dir.glob('checkpoint_update*.pth'))
    if not checkpoint_files:
        start_update = 0
        metrics_history = defaultdict(list)
    else:
        # Load the latest checkpoint
        latest_checkpoint = max(checkpoint_files, key=lambda f: f.stat().st_mtime)
        checkpoint = torch.load(latest_checkpoint, map_location='cpu')
        
        # Try to extract update number from filename as fallback
        filename_match = re.search(r'checkpoint_update_?(\d+)\.pth', latest_checkpoint.name)
        filename_update = int(filename_match.group(1)) if filename_match else 0
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            # New format with metadata
            start_update = checkpoint.get('update', checkpoint.get('epoch', filename_update))
            
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
    eval_file = experiment_dir / 'evaluation_results.json'
    if eval_file.exists():
        with open(eval_file, 'r') as f:
            results_data = json.load(f)
            
        for r_dict in results_data:
            # Reconstruct simulation result if present
            sim_result = None
            if 'simulation_result' in r_dict and r_dict['simulation_result'] and isinstance(r_dict['simulation_result'], dict):
                sim_data = r_dict['simulation_result']
                try:
                    sim_result = SimulationResult(**sim_data)
                except Exception as e:
                    logging.warning(f"Could not load simulation result: {e}")
                    sim_result = None
            
            # Create the result object
            # Check if this is an old result without simulation data
            if 'simulation_result' not in r_dict:
                # Convert to extended result with None simulation
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
                    simulation_result=None
                )
            else:
                # New format with simulation result
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
        
        logging.info(f"Loaded {len(evaluation_results)} existing evaluation results")
        
        # If we have evaluation results but no checkpoint update, use the last evaluation update
        if start_update == 0 and evaluation_results:
            last_eval_update = max(r.update for r in evaluation_results)
            logging.info(f"No checkpoint update found, using last evaluation update: {last_eval_update}")
            start_update = last_eval_update
    
    return start_update, metrics_history, evaluation_results, hyperparameters


def check_config_compatibility(saved_config: Dict, current_config: ExperimentConfig) -> Tuple[bool, bool]:
    """
    Check compatibility between saved and current configurations.
    
    Returns:
        (nn_params_match, all_params_match): Tuple indicating compatibility levels
    """
    # Define neural network hyperparameters
    nn_params = ['hidden_dim', 'num_hidden_layers']
    
    # Check NN parameter compatibility
    nn_params_match = all(
        saved_config.get(param) == getattr(current_config, param)
        for param in nn_params
    )
    
    # Define all critical hyperparameters for full compatibility
    all_params = [
        'hamiltonian_path', 'n_measurements', 'max_depth', 'beta',
        'hidden_dim', 'num_hidden_layers', 'lr', 'weight_decay',
        'reward_type', 'reward_kwargs', 'cost_type', 'cost_kwargs',
        'objective_type', 'objective_kwargs',
        'update_freq', 'n_eval_top_k_batch_elements'
    ]
    
    # Check full compatibility
    all_params_match = all(
        saved_config.get(param) == getattr(current_config, param)
        for param in all_params
    )
    
    return nn_params_match, all_params_match


def load_checkpoint_weights_only(trainer: EfficientGFNTrainer, checkpoint_path: Path, device) -> bool:
    """
    Load only the neural network weights from a checkpoint.
    
    Returns:
        True if weights were loaded successfully, False otherwise
    """
    try:
        logging.info(f"Loading neural network weights from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        if isinstance(checkpoint, dict):
            # Handle different checkpoint formats
            if 'pf_model_state_dict' in checkpoint:
                # New checkpoint format with pf_model
                trainer.gfn.pf_model.load_state_dict(checkpoint['pf_model_state_dict'])
                # Also load pb_model if needed (though it's usually uniform)
                if 'pb_model_state_dict' in checkpoint:
                    trainer.gfn.pb_model.load_state_dict(checkpoint['pb_model_state_dict'])
            elif 'model_state_dict' in checkpoint:
                # Old checkpoint format - try to load into pf_model
                trainer.gfn.pf_model.load_state_dict(checkpoint['model_state_dict'])
            elif 'state_dict' in checkpoint:
                # Another old format
                trainer.gfn.pf_model.load_state_dict(checkpoint['state_dict'])
            else:
                # Check if checkpoint contains the state dict directly
                # This is unlikely but handle it
                logging.warning("Checkpoint format not recognized, attempting direct load")
                trainer.gfn.pf_model.load_state_dict(checkpoint)
        else:
            # Assume the checkpoint is the state dict itself
            trainer.gfn.pf_model.load_state_dict(checkpoint)
        
        logging.info("Successfully loaded neural network weights only")
        return True
        
    except Exception as e:
        logging.error(f"Failed to load weights: {e}")
        return False


async def run_experiment(config: ExperimentConfig):
    """Main experiment runner with enhanced checkpoint loading"""
    
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
    
    # Handle resumption/checkpoint loading
    if config.resume or config.experiment_dir:
        experiment_to_check = None
        
        if config.experiment_dir:
            # Check specific experiment directory
            experiment_to_check = Path(config.results_dir) / config.experiment_dir
            if not experiment_to_check.exists():
                logging.info(f"Specified experiment directory {experiment_to_check} not found.")
                experiment_to_check = None
        else:
            # Find most recent experiment
            experiment_to_check = find_latest_experiment(config.results_dir)
        
        if experiment_to_check:
            # Load saved configuration
            with open(experiment_to_check / 'config.json', 'r') as f:
                saved_config = json.load(f)
            
            # Check compatibility levels
            nn_match, all_match = check_config_compatibility(saved_config, config)
            
            if all_match:
                # Full compatibility - resume training
                logging.info(f"All hyperparameters match. Resuming training from {experiment_to_check}")
                results_dir = experiment_to_check
                start_update, existing_metrics, existing_results, hyperparameters = load_experiment_state(results_dir)
                logging.info(f"Starting from update {start_update + 1}")
                
            elif nn_match:
                # Only NN parameters match - load weights but start new experiment
                logging.info(f"Neural network hyperparameters match. Will load weights from {experiment_to_check}")
                logging.info("Other hyperparameters differ - starting new experiment with transferred weights")
                load_weights_only = True
                checkpoint_to_load = experiment_to_check
                
                # Print what parameters differ
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
    
    # Create new experiment directory if needed
    if results_dir is None:
        results_dir = Path(config.results_dir) / f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        results_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"Starting new experiment in {results_dir}")
        
        # Save configuration
        with open(results_dir / 'config.json', 'w') as f:
            json.dump(asdict(config), f, indent=2, cls=NumpyEncoder)
    
    # Print experiment details
    logging.info(f"\nExperiment details:")
    logging.info(f"Hamiltonian: {config.hamiltonian_path}")
    logging.info(f"Batch structure: {config.n_measurements} circuits per batch element")
    logging.info(f"Evaluation: Top {config.n_eval_top_k_batch_elements} batch elements")
    logging.info(f"Energy Estimation: Using new EnergyEstimator API")
    if config.n_simulations > 1:
        logging.info(f"Simulations: {config.n_simulations} runs per batch element")
    if config.cost_kwargs:
        logging.info(f"Cost function: {config.cost_type} with kwargs: {config.cost_kwargs}")
    if start_update > 0:
        logging.info(f"Resuming from update: {start_update + 1}/{config.n_updates}")
    elif load_weights_only:
        logging.info(f"Starting fresh training with weights from: {checkpoint_to_load.name}")
    
    # Load Hamiltonian
    hamiltonian_helper = PauliHamiltonianHelper(config.hamiltonian_path)
    logging.info(f"Hamiltonian: {hamiltonian_helper}")
    logging.info(f"Exact ground state energy: {hamiltonian_helper.ground_state_energy:.10f}")
    
    # Save Hamiltonian info (only for new experiments)
    if start_update == 0:
        with open(results_dir / 'hamiltonian_info.json', 'w') as f:
            json.dump(hamiltonian_helper.summary(), f, indent=2, cls=NumpyEncoder)

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
    
    # Create comprehensive hyperparameters file
    if hyperparameters is None:
        hyperparameters = create_hyperparameters_dict(config, hamiltonian_helper, 
                                                      training_pauli_strings, identity_weight)
        with open(results_dir / 'hyperparameters.json', 'w') as f:
            json.dump(hyperparameters, f, indent=2, cls=NumpyEncoder)
    
    # Initialize visualizer
    visualizer = AsyncVisualizer(results_dir, hyperparameters)
    if existing_results:
        visualizer.add_results(existing_results)
    
    # Create GFN configuration
    logging.info(f"Training with {len(training_pauli_strings)} non-identity Pauli terms")
    if identity_weight != 0:
        logging.info(f"Note: Identity term contributes a constant energy offset of {identity_weight:.6f}")
    
    gfn_config = {
        "model": {
            "model_type": "clifford_mlp",
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
    
    # Create trainer
    device = get_device(config.device_preference)
    
    # Select reward function
    reward_fn_map = {
        "exponential": exponential_reward_fn,
        "default": default_reward_fn,
    }
    reward_fn = reward_fn_map.get(config.reward_type, default_reward_fn)
    
    trainer = EfficientGFNTrainer(gfn_config, reward_fn=reward_fn, 
                                device_preference=config.device_preference)
    
    # Handle checkpoint loading based on compatibility
    if start_update > 0:
        # Full resume - load complete checkpoint
        checkpoint_files = list(results_dir.glob('checkpoint_update*.pth'))
        if checkpoint_files:
            latest_checkpoint = max(checkpoint_files, key=lambda f: f.stat().st_mtime)
            logging.info(f"Loading full checkpoint from update {start_update}")
            try:
                trainer.gfn.load_checkpoint(str(latest_checkpoint))
                trainer.metrics_history = existing_metrics
            except Exception as e:
                logging.warning(f"Failed to load checkpoint: {e}")
                # Try manual loading
                checkpoint = torch.load(latest_checkpoint, map_location=trainer.device)
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    trainer.gfn.model.load_state_dict(checkpoint['model_state_dict'])
                    if 'optimizer_state_dict' in checkpoint:
                        trainer.gfn.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    trainer.metrics_history = existing_metrics
                    
    elif load_weights_only:
        # Load only neural network weights
        checkpoint_files = list(checkpoint_to_load.glob('checkpoint_update*.pth'))
        if checkpoint_files:
            latest_checkpoint = max(checkpoint_files, key=lambda f: f.stat().st_mtime)
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
    
    # Initialize energy estimator with new API
    energy_estimator = EnergyEstimator(
        hamiltonian_helper, 
        hamiltonian_helper.n_qubits, 
        device
    )
    
    # Save training-specific hyperparameters (only for new experiments)
    if start_update == 0:
        training_hyperparams = {
            "actual_device": str(trainer.device),
            "num_actions": trainer.gfn.num_actions,
            "state_dim": trainer.gfn.state_dim,
            "action_mapping_size": len(trainer.gfn.action_mapping),
            "gfn_config": gfn_config,
            "weights_loaded_from": str(checkpoint_to_load.name) if load_weights_only else None,
            "energy_estimator": "EnergyEstimator with batched Clifford map"
        }
        
        with open(results_dir / 'training_hyperparameters.json', 'w') as f:
            json.dump(training_hyperparams, f, indent=2, cls=NumpyEncoder)
    
    # Rest of the training loop remains the same...
    all_evaluation_results = existing_results.copy() if existing_results else []
    
    logging.info(f"\n=== {'Resuming' if start_update > 0 else 'Starting'} Training ===")
    if load_weights_only:
        logging.info("Note: Using transferred neural network weights")
    
    # Training loop
    for update in range(start_update, config.n_updates):
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
            **config.cost_kwargs  # Pass cost_kwargs here
        )
        
        # Compute loss and update
        loss, metrics = trainer.gfn.compute_loss(
            trajectory_batch, costs, config.beta, max_depth=config.max_depth, **config.reward_kwargs
        )
        trainer.gfn.update_step(loss)
        
        # Update top trajectories (batch elements) for replay
        trainer.gfn._update_top_trajectories(trajectory_batch, costs)
        
        # Store metrics
        for k, v in metrics.items():
            trainer.metrics_history[k].append(v)
        
        # Print progress
        if (update + 1) % 100 == 0:
            logging.info(f"Update {update + 1}/{config.n_updates}: "
                  f"Loss={metrics['loss']:.6f}, "
                  f"Reward={metrics['reward']:.4f}, "
                  f"Cost={metrics['cost']:.4f}")
        
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
                silence=False,
                **config.cost_kwargs  # Pass cost_kwargs here too
            )
            replay_loss, replay_metrics = trainer.gfn.compute_loss(
                replay_batch, replay_costs, config.beta, max_depth=config.max_depth, **config.reward_kwargs
            )
            trainer.gfn.update_step(replay_loss)
            
            # Store replay metrics
            for k, v in replay_metrics.items():
                trainer.metrics_history[f'replay_{k}'].append(v)
        
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
                **config.cost_kwargs  # Pass cost_kwargs here too
            )
            offpolicy_loss, offpolicy_metrics = trainer.gfn.compute_loss(
                offpolicy_batch, offpolicy_costs, config.beta, max_depth=config.max_depth, **config.reward_kwargs
            )
            trainer.gfn.update_step(offpolicy_loss)
            
            # Store off-policy metrics
            for k, v in offpolicy_metrics.items():
                trainer.metrics_history[f'offpolicy_{k}'].append(v)
        
        # Evaluation with energy estimation
        if (update + 1) % config.eval_every == 0:
            logging.info(f"\n{'='*60}")
            logging.info(f"Checkpoint evaluation at update {update + 1}")
            
            # Evaluate top batch elements
            evaluation_results = await evaluate_top_batch_elements(
                trainer, energy_estimator, update + 1, config
            )
            
            # Store results
            all_evaluation_results.extend(evaluation_results)
            
            # Update visualizations
            visualizer.add_results(evaluation_results)
            await visualizer.update_plots_async()
            
            # Save intermediate results
            results_data = []
            for r in all_evaluation_results:
                r_dict = asdict(r)
                # Handle simulation result separately
                if hasattr(r, 'simulation_result') and r.simulation_result:
                    r_dict['simulation_result'] = asdict(r.simulation_result)
                results_data.append(r_dict)
            
            with open(results_dir / 'evaluation_results.json', 'w') as f:
                json.dump(results_data, f, indent=2, cls=NumpyEncoder)
            
            logging.info(f"{'='*60}\n")
        
        # Checkpoint saving
        if (update + 1) % config.checkpoint_every == 0:
            checkpoint_path = results_dir / f'checkpoint_update.pth'
            trainer.gfn.save_checkpoint(str(checkpoint_path), update + 1, dict(trainer.metrics_history))
            
            # Save training metrics plot
            trainer.plot_metrics(update + 1)
    
    # Final evaluation
    logging.info("\n=== Final Evaluation ===")
    final_results = await evaluate_top_batch_elements(
        trainer, energy_estimator, config.n_updates, config
    )
    
    all_evaluation_results.extend(final_results)
    
    # Final visualization update
    visualizer.add_results(final_results)
    await visualizer.update_plots_async()
    
    # Save all results
    results_data = []
    for r in all_evaluation_results:
        r_dict = asdict(r)
        if hasattr(r, 'simulation_result') and r.simulation_result:
            r_dict['simulation_result'] = asdict(r.simulation_result)
        results_data.append(r_dict)
    
    with open(results_dir / 'all_evaluation_results.json', 'w') as f:
        json.dump(results_data, f, indent=2, cls=NumpyEncoder)
    
    # Generate final report
    generate_final_report(all_evaluation_results, hamiltonian_helper, results_dir, hyperparameters, trainer)
    
    logging.info(f"\nExperiment completed! Results saved to {results_dir}")
    
    return all_evaluation_results, trainer


def generate_final_report(results: List[ExtendedBatchElementEnergyResult], 
                         hamiltonian_helper: PauliHamiltonianHelper,
                         results_dir: Path,
                         hyperparameters: Dict,
                         trainer: EfficientGFNTrainer):
    """Generate a final summary report"""
    
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
    
    report_lines.extend(["\n## Experiment Summary",
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
        
        # Training metrics summary
        if hasattr(trainer, 'metrics_history') and trainer.metrics_history:
            final_metrics = {k: v[-1] if v else 0 for k, v in trainer.metrics_history.items()}
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


if __name__ == "__main__":
    # Example configuration
    config = ExperimentConfig(
        hamiltonian_path="../Hamiltonians/H2_6-31G_8qubits/jw.txt",
        eval_every=1000,                    # Evaluate every N training updates
        n_updates=100000,                  # Total training updates
        n_eval_top_k_batch_elements=5,     # Number of top batch elements to evaluate
        n_measurements=1000,                 # Circuits per batch element
        update_freq=4,                     # Batch elements per training update
        max_depth=3,                       # Maximum circuit depth
        beta=100.0,                          # Temperature parameter
        hidden_dim=1024,                   # Neural network hidden dimension
        num_hidden_layers=3,               # Number of hidden layers
        lr=1e-3,                           # Learning rate
        weight_decay=1e-5,                 # Weight decay
        device_preference="cpu",           # "cuda", "mps", "cpu", or "auto"
        results_dir="../results/H2_6-31G_8qubits",
        replay_every=40,                   # Replay training frequency
        offpolicy_every=40,                # Off-policy training frequency
        checkpoint_every=40,              # Model checkpoint frequency
        reward_type="default",             # Reward function type
        reward_kwargs={"alpha": 1.0},     # Reward function parameters
        cost_type="linear_bias",           # Cost function type
        cost_kwargs={"epsilon": 0.9},      # Cost function parameters
        objective_type="tb",               # GFlowNet objective
        objective_kwargs={"loss_type": "squared"},
        n_simulations=100,                 # Number of simulation runs (set to 1 for single run)
        resume=True,                       # Auto-resume from latest matching experiment
        experiment_dir=None                # Or specify: "experiment_20241210_143022"
    )
    
    # Run experiment (will resume automatically if matching experiment exists)
    asyncio.run(run_experiment(config))
