"""CPU-side result-bundle assembly for the energy estimator.

Pure post-processing helpers. They take the already-sampled tensors (hits,
per-Pauli estimates, energy estimates) plus the Hamiltonian metadata (Pauli
strings / coefficients / ground-state energy) and assemble the
``BatchElementEnergyResult`` bundles and the per-batch-element summary dicts.

No sampling, no full-state materialization, and no ``EnergyEstimator`` state
beyond the explicit arguments — so the estimator god-class shrinks without
touching the tensor pipeline or the 26-qubit safety guard.

``BatchElementEnergyResult`` is passed in as ``result_cls`` (it is defined in
``energy_estimator.py``); this keeps ``energy_reporting`` a leaf that never
imports ``energy_estimator``, so there is no import cycle.
"""
from __future__ import annotations

import warnings
from typing import Dict, List

import numpy as np


def assemble_batch_element_results(
    *,
    pauli_strings,
    pauli_to_coeff,
    ground_state_energy,
    batch_lengths,
    hits,
    estimates_all,
    energy_estimates,
    batch_size,
    n_circuits,
    M,
    result_cls,
):
    """Build the per-(batch element, simulation) result objects.

    Returns:
        Tuple ``(batch_simulation_results, mean_pauli_estimates_per_b)``:
          - ``batch_simulation_results``: list (batch elements) of lists
            (M simulations) of ``BatchElementEnergyResult``. Only the
            final simulation per batch element carries a populated
            ``pauli_estimates`` dict (it becomes ``final_results_object``);
            intermediate sims carry ``{}``.
          - ``mean_pauli_estimates_per_b``: list of per-batch-element dicts
            mapping each Pauli string to its mean estimate across the M
            simulations.

    Sole caller is ``EnergyEstimator._estimate_energy`` (``hits`` is
    ``prepared.hits``; ``result_cls`` is ``BatchElementEnergyResult``).
    """
    n_paulis = len(pauli_strings)

    # Convert tensors to CPU once for efficient result construction
    batch_lengths_cpu = batch_lengths.cpu().numpy()
    hits_cpu = hits.cpu().numpy()
    estimates_all_cpu = estimates_all.cpu().numpy()
    energy_estimates_cpu = energy_estimates.cpu().numpy()

    # Per-Pauli mean across the M simulations, computed once on-array in
    # float64. A per-Pauli dict is built only for the *reported* simulation
    # (the last m, which becomes ``final_results_object``) plus one mean dict
    # per batch element; intermediate sims carry an empty ``pauli_estimates``.
    estimates_mean_cpu = estimates_all_cpu.astype(np.float64).mean(axis=1)  # (B, K)

    batch_simulation_results = []
    mean_pauli_estimates_per_b: List[Dict[str, float]] = []
    for b_idx in range(batch_size):
        c_lens = batch_lengths_cpu[b_idx].tolist()
        hitting_counts_dict = {
            pauli_strings[p_idx]: int(hits_cpu[b_idx, p_idx])
            for p_idx in range(n_paulis)
        }

        # Emit warning for unmeasured Paulis (only once per batch element)
        for p_idx, p_str in enumerate(pauli_strings):
            if hits_cpu[b_idx, p_idx] == 0 and p_str in pauli_to_coeff:
                warnings.warn(f"Pauli {p_str} was never measured (N_P = 0)")

        measured_paulis = [p for p, n in hitting_counts_dict.items() if n > 0]
        hitting_values = [n for n in hitting_counts_dict.values() if n > 0]

        mean_pauli_estimates_per_b.append({
            pauli_strings[p_idx]: float(estimates_mean_cpu[b_idx, p_idx])
            for p_idx in range(n_paulis)
        })

        convergence = {
            'coverage': len(measured_paulis) / len(pauli_strings) if pauli_strings else 0,
            'avg_hitting_count': np.mean(hitting_values) if hitting_values else 0,
        }
        mean_circuit_length = np.mean(c_lens) if c_lens else 0

        results_for_batch = []
        for m in range(M):
            # Only the final simulation's object is consumed downstream
            # (``final_results_object``); building the full per-Pauli dict
            # for every intermediate m was pure O(M*K) overhead.
            if m == M - 1:
                pauli_estimates_dict = {
                    pauli_strings[p_idx]: float(estimates_all_cpu[b_idx, m, p_idx])
                    for p_idx in range(n_paulis)
                }
            else:
                pauli_estimates_dict = {}

            result = result_cls(
                update=0,
                batch_element_rank=b_idx,
                n_circuits=n_circuits,
                # Total shots = circuits * sims. Equals n_circuits when M=1.
                total_measurements=n_circuits * M,
                energy_estimate=float(energy_estimates_cpu[b_idx, m]),
                energy_difference=abs(float(energy_estimates_cpu[b_idx, m]) - ground_state_energy),
                pauli_estimates=pauli_estimates_dict,
                hitting_counts=hitting_counts_dict,
                circuit_lengths=c_lens,
                mean_circuit_length=mean_circuit_length,
                batch_cost=0.0,
                convergence_metrics=convergence,
            )
            results_for_batch.append(result)

        batch_simulation_results.append(results_for_batch)

    return batch_simulation_results, mean_pauli_estimates_per_b


def summarize_simulations(
    *,
    batch_simulation_results,
    mean_pauli_estimates_per_b,
    batch_size,
    M,
    ground_state_energy,
):
    """Aggregate the per-(batch element, simulation) results into one summary
    dict per batch element.

    Sole caller is ``EnergyEstimator.estimate_energy_with_simulations``.
    """
    # Aggregate results
    final_summaries = []
    for b_idx in range(batch_size):
        results_for_el = batch_simulation_results[b_idx]
        energies = [res.energy_estimate for res in results_for_el]
        mean_energy = sum(energies) / M

        # Per-Pauli mean across the M sims is precomputed on-array inside
        # ``_estimate_energy`` (float64), so no per-result dict re-walk here.
        mean_pauli_estimates = mean_pauli_estimates_per_b[b_idx]

        # Calculate individual errors
        individual_absolute_errors = [abs(e - ground_state_energy) for e in energies]
        individual_squared_errors = [(e - ground_state_energy) ** 2 for e in energies]

        # Keep the L1 (MAE) and L2 (RMSE) aggregates as distinct
        # fields so callers don't have to guess what the field name means.
        mae = float(np.mean(individual_absolute_errors))
        rmse = float(np.sqrt(np.mean(individual_squared_errors)))

        summary = {
            'batch_index': b_idx,
            'mean_energy': mean_energy,
            'energy_variance': np.var(energies, ddof=1) if M > 1 else 0.0,
            'rmse': rmse,    # L2 aggregate
            'mae': mae,      # L1 aggregate
            # ``energy_difference`` now always means an
            # absolute-error value (per-sim |E-E*| at M=1, MAE at M>1)
            # so the field name matches the content. Previously it was
            # silently RMSE at M>1, which broke downstream readers.
            'energy_difference': mae,
            'std_absolute_error': np.std(individual_absolute_errors, ddof=1) if M > 1 else 0.0,  # Std of absolute errors
            'num_simulations': M,
            'mean_pauli_estimates': mean_pauli_estimates,
            'final_results_object': results_for_el[-1],
            'individual_energies': energies,  # Individual energies for each simulation
            'individual_absolute_errors': individual_absolute_errors,  # Individual absolute errors for each simulation
            'individual_squared_errors': individual_squared_errors  # Individual squared errors for each simulation
        }
        final_summaries.append(summary)

    return final_summaries
