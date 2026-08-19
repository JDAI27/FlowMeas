# -*- coding: utf-8 -*-
"""
Cost computation for circuit quality evaluation.

This module computes costs (negative rewards) for measurement circuits
based on their ability to measure Hamiltonian Pauli terms.

Cost Functions:
===============
- CoverageVarianceCost: Penalizes low coverage and high variance
- ThresholdCost: Penalizes Paulis measured fewer than threshold times
- UniformVarianceCost: Targets uniform hitting counts
- ConfidenceCost: Union-Hoeffding bound on per-term estimation failure probability

Key Quantities:
    hitting_counts: (B, K) - how many circuits measure each of K Paulis
    coverage: fraction of Paulis with hitting_count > 0
    variance: variance of hitting counts across Paulis (lower = more uniform)
    cost: scalar measuring circuit set quality (lower = better)

Normalization:
    Pauli weights can be normalized by sum or max for scale invariance.
    Identity term is typically excluded from normalization.
"""
import torch
from typing import List, Tuple, Optional, Literal, Callable, Union
from abc import ABC, abstractmethod


NormalizationType = Literal['sum', 'max']


def normalize_pauli_weights(weights: torch.Tensor, 
                            pauli_strings: Optional[List[str]] = None,
                            n_qubits: Optional[int] = None,
                            non_identity_mask: Optional[torch.Tensor] = None,
                            normalization_type: NormalizationType = 'sum') -> torch.Tensor:
    """
    Normalize Pauli weights, excluding identity.
    GPU-optimized: avoids Python loops and CPU-GPU sync.

    Args:
        weights: Tensor of Pauli weights
        pauli_strings: Optional list of Pauli strings to identify identity terms
        n_qubits: Optional number of qubits (used to construct identity string if pauli_strings provided)
        non_identity_mask: Pre-computed mask for non-identity terms (GPU optimization)
        normalization_type: 'sum' divides by sum(|w_i|), 'max' divides by max(|w_i|)
            - 'sum': Traditional normalization where sum of |w_i| = 1 (excluding identity)
            - 'max': Normalize by largest weight, preserving relative scale better
                     Makes largest weight = 1, useful for better bias term interpretation

    Returns:
        Normalized weights tensor (excluding identity terms)
    """
    weights_abs = weights.abs()
    
    # If pre-computed mask provided, use it directly (GPU-optimized path)
    if non_identity_mask is not None:
        # Fully GPU-native: no Python conditionals, no CPU-GPU sync
        masked_weights = weights_abs * non_identity_mask.float()
        if normalization_type == 'max':
            norm_factor = masked_weights.max().clamp(min=1e-10)
        else:  # 'sum'
            norm_factor = masked_weights.sum().clamp(min=1e-10)
        normalized = masked_weights / norm_factor
        return normalized
    
    # If pauli_strings provided, compute mask (slower path, but only called once if cached)
    if pauli_strings is not None:
        if n_qubits is None:
            n_qubits = len(pauli_strings[0]) if pauli_strings else 0
        identity_str = "I" * n_qubits
        
        # Create mask for non-identity terms
        non_identity_mask = torch.tensor(
            [p != identity_str for p in pauli_strings],
            dtype=torch.bool,
            device=weights.device
        )
        
        # GPU-optimized: use clamp instead of conditional
        masked_weights = weights_abs * non_identity_mask.float()
        if normalization_type == 'max':
            norm_factor = masked_weights.max().clamp(min=1e-10)
        else:  # 'sum'
            norm_factor = masked_weights.sum().clamp(min=1e-10)
        normalized = masked_weights / norm_factor
        return normalized
    
    # No pauli_strings provided - normalize all weights (GPU-optimized)
    if normalization_type == 'max':
        norm_factor = weights_abs.max().clamp(min=1e-10)
    else:  # 'sum'
        norm_factor = weights_abs.sum().clamp(min=1e-10)
    normalized = weights_abs / norm_factor
    return normalized


def create_non_identity_mask(pauli_strings: List[str], 
                              n_qubits: int,
                              device: torch.device) -> torch.Tensor:
    """
    Pre-compute non-identity mask once during initialization.
    This should be called once, not on every cost computation.

    Args:
        pauli_strings: List of Pauli strings
        n_qubits: Number of qubits
        device: Device to create tensor on

    Returns:
        Boolean mask tensor where True = non-identity term
    """
    identity_str = "I" * n_qubits
    return torch.tensor(
        [p != identity_str for p in pauli_strings],
        dtype=torch.bool,
        device=device
    )


def detect_stabilizer_terms(pauli_strings: List[str],
                            coeffs: List[Union[float, complex]],
                            hamiltonian_path: Union[str, "os.PathLike"]) -> List[bool]:
    """Identify the stabilizer PENALTY terms of a compact-encoded Hamiltonian (fail-fast).

    Compact (Derby-Klassen) Hamiltonian files carry a stabilizer penalty
    ``H_enc + lambda * sum_i (I - S_i)/2``: each stabilizer word appears as a Pauli
    term with coefficient ``-lambda/2``. On the code space these terms have
    ``Gamma == 0`` (zero measurement information), yet at
    ``|c| = lambda/2`` they dominate the cost's weight incentive — the
    ``zero_stabilizer_cost_weights`` knob zeroes them pre-normalization.

    Detection is metadata-driven, never guessed:
      * Read the ``metadata.json`` sibling of ``hamiltonian_path``. REQUIRE a
        ``stabilizer_penalty`` block with ``lambda`` and ``n_stabilizers``; a
        missing file/block raises a precise ``RuntimeError`` (no silent no-op).
      * Mask = terms with Pauli weight >= 4 AND ``isclose(coeff, -lambda/2)``.
      * ASSERT the match count equals ``n_stabilizers`` (18 at 6x6, 8 at 4x4);
        anything else raises, listing what matched.
      * ASSERT every matched word commutes with every Hamiltonian term. Stabilizer
        penalties are central constraints; same-count coefficient impostors are not.

    Returns a ``List[bool]`` aligned with ``pauli_strings`` (True = stabilizer term).
    """
    import json
    import math
    import os

    ham_path = os.fspath(hamiltonian_path)
    meta_path = os.path.join(os.path.dirname(os.path.abspath(ham_path)), "metadata.json")
    if not os.path.exists(meta_path):
        raise RuntimeError(
            f"zero_stabilizer_cost_weights: no metadata.json next to {ham_path!r} "
            f"(looked for {meta_path!r}). Stabilizer detection is metadata-driven and "
            "never guessed; disable the flag or supply the compact-encoding metadata."
        )
    with open(meta_path) as f:
        meta = json.load(f)
    block = meta.get("stabilizer_penalty")
    if not isinstance(block, dict) or "lambda" not in block or "n_stabilizers" not in block:
        raise RuntimeError(
            f"zero_stabilizer_cost_weights: {meta_path!r} has no usable "
            f"'stabilizer_penalty' block (need 'lambda' and 'n_stabilizers'; got "
            f"{block!r}). This Hamiltonian is not a compact-encoded file with a "
            "stabilizer penalty; disable the flag for it."
        )
    lam = float(block["lambda"])
    n_stab = int(block["n_stabilizers"])
    target = -lam / 2.0

    if len(pauli_strings) != len(coeffs):
        raise ValueError(
            f"pauli_strings ({len(pauli_strings)}) and coeffs ({len(coeffs)}) length mismatch."
        )
    mask = []
    matched = []
    for i, (p, c) in enumerate(zip(pauli_strings, coeffs)):
        weight = sum(1 for ch in p if ch != "I")
        c_real = complex(c).real
        is_stab = weight >= 4 and math.isclose(c_real, target, rel_tol=1e-9, abs_tol=1e-12)
        mask.append(is_stab)
        if is_stab:
            matched.append((i, weight, c_real))
    if len(matched) != n_stab:
        raise RuntimeError(
            f"zero_stabilizer_cost_weights: matched {len(matched)} candidate stabilizer "
            f"terms (weight>=4 and coeff=={target}) but metadata says n_stabilizers="
            f"{n_stab}. Matches: {matched}. Refusing to zero an ambiguous set."
        )
    def pauli_commutes(a: str, b: str) -> bool:
        anti = 0
        for ca, cb in zip(a, b):
            if ca == "I" or cb == "I" or ca == cb:
                continue
            anti += 1
        return anti % 2 == 0

    noncentral = []
    matched_indices = [i for i, _, _ in matched]
    for i in matched_indices:
        p = pauli_strings[i]
        for j, q in enumerate(pauli_strings):
            if not pauli_commutes(p, q):
                noncentral.append((i, j, p, q))
                break
    if noncentral:
        raise RuntimeError(
            "zero_stabilizer_cost_weights: matched coefficient/weight candidates "
            "that do not commute with the Hamiltonian. Stabilizer penalties must be "
            f"central constraints. Examples: {noncentral[:5]}. Refusing to zero an "
            "ambiguous set."
        )
    return mask


class CostFunction(ABC):
    """Abstract base class for cost functions"""
    
    @abstractmethod
    def compute(self, probs: torch.Tensor, weights: torch.Tensor, epsilon: float) -> torch.Tensor:
        """Compute cost given probabilities, weights, and epsilon"""
        pass
    
    @abstractmethod
    def compute_batch(self, probs: torch.Tensor, weights: torch.Tensor, epsilon: float) -> torch.Tensor:
        """Compute batch cost (summed over measurements)"""
        pass


class ExponentialCost(CostFunction):
    """Outer-weight coverage proxy: sum_i |w_i| * exp(-p_i * epsilon^2 / 2).

    This is the existing ``exponential`` cost family, called the exponential
    confidence cost in the mathematical contract. It is distinct from
:class:`ConfidenceCost`, whose coefficient magnitude appears in the
    exponent denominator to target a fixed per-term estimation tolerance.
    """
    
    def compute(self, probs: torch.Tensor, weights: torch.Tensor, epsilon: float) -> torch.Tensor:
        """
        Compute exponential cost for each measurement trajectory

        Args:
            probs: Tensor of shape (batch_size, n_measurements, n_paulis)
            weights: Tensor of shape (n_paulis,) - assumed already non-negative from CostComputer
            epsilon: Error parameter

        Returns:
            costs: Tensor of shape (batch_size, n_measurements)
        """
        # Note: weights already normalized/abs from CostComputer._prepare_weights
        exp_term = torch.exp(-probs * epsilon**2 / 2)
        costs = torch.einsum('bmp,p->bm', exp_term, weights)
        return costs
    
    def compute_batch(self, probs: torch.Tensor, weights: torch.Tensor, epsilon: float) -> torch.Tensor:
        """
        Compute batch cost summed over measurements

        Args:
            probs: Tensor of shape (batch_size, n_measurements, n_paulis)
            weights: Tensor of shape (n_paulis,) - assumed already non-negative from CostComputer
            epsilon: Error parameter

        Returns:
            costs: Tensor of shape (batch_size,)
        """
        # Note: weights already normalized/abs from CostComputer._prepare_weights
        exponent_terms = torch.sum(-probs * epsilon**2 / 2, dim=1)
        exp_terms = torch.exp(exponent_terms)
        costs = torch.matmul(exp_terms, weights)
        return costs

class LinearBiasCost(CostFunction):
    """Linear cost function: sum_{i,p_i != 0} w_i^2 /(\\sum_j^n_measurements p_{i,j}) + sum_{i,j, p_i = 0,p_j = 0} w_i w_j"""
    
    def __init__(self, epsilon_zero: float = 1e-4):
        self.epsilon_zero = epsilon_zero  # Threshold for considering probability as zero
    
    def compute(self, probs: torch.Tensor, weights: torch.Tensor, epsilon: float) -> torch.Tensor:
        """
        Compute linear cost for each measurement trajectory
        Args:
            probs: Tensor of shape (batch_size, n_measurements, n_paulis)
            weights: Tensor of shape (n_paulis,)
            epsilon: Error parameter (not used in this formulation)
        Returns:
            costs: Tensor of shape (batch_size, n_measurements)
        """
        # Identify zero and non-zero probabilities
        probs_sum = probs.sum(dim=1)  # Sum over measurements, probs_sum in shape (batch_size, n_paulis)
        is_zero = probs_sum < self.epsilon_zero
        is_nonzero = ~is_zero
        
        # Compute w_i^2 / p_i for non-zero probabilities
        weights_squared = weights.pow(2)
        nonzero_cost = torch.where(
            is_nonzero,
            weights_squared.unsqueeze(0) / (probs_sum),
            torch.zeros_like(probs_sum)
        )
        
        # Sum over Pauli dimension for non-zero term
        nonzero_term = nonzero_cost.sum(dim=1) 
        
        # Compute sum_{i,j, p_i = 0, p_j = 0} w_i * w_j
        # This equals (sum_{i, p_i = 0} w_i)^2
        weights_masked = is_zero * weights.abs().unsqueeze(0)
        sum_zero_weights = weights_masked.sum(dim=1) 
        zero_term = sum_zero_weights.pow(2)  # Square the sum of zero weights
        
        # Total cost per measurement
        costs = nonzero_term + zero_term
        return costs
    
    def compute_batch(self, probs: torch.Tensor, weights: torch.Tensor, epsilon: float) -> torch.Tensor:
        """
        Compute batch cost summed over measurements
        Args:
            probs: Tensor of shape (batch_size, n_measurements, n_paulis)
            weights: Tensor of shape (n_paulis,)
            epsilon: Error parameter (not used in this formulation)
        Returns:
            costs: Tensor of shape (batch_size,)
        """
        n_measurements = probs.shape[1]
        
        # Sum probabilities over measurements for each Pauli
        probs_sum = probs.sum(dim=1)  # Shape: (batch_size, n_paulis)
        is_zero = probs_sum < self.epsilon_zero
        is_nonzero = ~is_zero
        
        # Compute w_i^2 / p_i for non-zero probabilities
        weights_squared = weights.pow(2)
        nonzero_cost = torch.where(
            is_nonzero,
            weights_squared.unsqueeze(0) / probs_sum,
            torch.zeros_like(probs_sum)
        )
        
        # Sum over Pauli dimension
        nonzero_term = nonzero_cost.sum(dim=1)
        
        # Compute sum_{i,j, p_i = 0, p_j = 0} w_i * w_j
        # This equals (sum_{i, p_i = 0} w_i)^2
        weights_masked = is_zero * weights.abs().unsqueeze(0)
        sum_zero_weights = weights_masked.sum(dim=1)  # Shape: (batch_size,)
        zero_term = sum_zero_weights ** 2
        
        # Total batch cost
        costs = nonzero_term + zero_term
        return costs


class OGMCost(CostFunction):
    """
    Overlapped Grouping Measurement (OGM) cost function from arXiv:2105.13091.

    Cost = sum_{i: p_i > 0} w_i^2 / p_i + sum_{i: p_i = 0} w_i^2 * T

    Key difference from LinearBiasCost:
    - LinearBiasCost: unmeasured penalty = (sum |w_i|)^2 (squared L1 norm, has cross terms)
    - OGMCost: unmeasured penalty = sum(w_i^2 * T) (L2 squared * T, no cross terms)

    The OGM penalty scales with T (number of measurements) based on Chebyshev inequality,
    providing a principled trade-off for unmeasured terms.

    Reference: Wu et al., "Overlapped grouping measurement: A unified framework for
    measuring quantum states", arXiv:2105.13091, Equation (15)
    """
    
    def __init__(self, epsilon_zero: float = 1e-4):
        self.epsilon_zero = epsilon_zero  # Threshold for considering probability as zero
    
    def compute(self, probs: torch.Tensor, weights: torch.Tensor, epsilon: float) -> torch.Tensor:
        """
        Compute OGM cost for each measurement trajectory.

        Args:
            probs: Tensor of shape (batch_size, n_measurements, n_paulis)
            weights: Tensor of shape (n_paulis,)
            epsilon: Error parameter (not used directly, T derived from n_measurements)

        Returns:
            costs: Tensor of shape (batch_size, n_measurements)
        """
        n_measurements = probs.shape[1]
        
        # Sum probabilities over measurements for each Pauli
        probs_sum = probs.sum(dim=1)  # Shape: (batch_size, n_paulis)
        is_zero = probs_sum < self.epsilon_zero
        is_nonzero = ~is_zero
        
        # Compute w_i^2 / p_i for non-zero probabilities (measured terms)
        weights_squared = weights.pow(2)
        nonzero_cost = torch.where(
            is_nonzero,
            weights_squared.unsqueeze(0) / probs_sum,
            torch.zeros_like(probs_sum)
        )
        
        # Sum over Pauli dimension for measured term
        nonzero_term = nonzero_cost.sum(dim=1)
        
        # OGM penalty for unmeasured terms: sum_{i: p_i = 0} w_i^2 * T
        # No cross terms, scales with T
        zero_cost = torch.where(
            is_zero,
            weights_squared.unsqueeze(0) * n_measurements,
            torch.zeros_like(probs_sum)
        )
        zero_term = zero_cost.sum(dim=1)
        
        # Total cost
        costs = nonzero_term + zero_term
        return costs
    
    def compute_batch(self, probs: torch.Tensor, weights: torch.Tensor, epsilon: float) -> torch.Tensor:
        """
        Compute batch OGM cost summed over measurements.

        Args:
            probs: Tensor of shape (batch_size, n_measurements, n_paulis)
            weights: Tensor of shape (n_paulis,)
            epsilon: Error parameter (not used directly, T derived from n_measurements)

        Returns:
            costs: Tensor of shape (batch_size,)
        """
        n_measurements = probs.shape[1]
        
        # Sum probabilities over measurements for each Pauli
        probs_sum = probs.sum(dim=1)  # Shape: (batch_size, n_paulis)
        is_zero = probs_sum < self.epsilon_zero
        is_nonzero = ~is_zero
        
        # Compute w_i^2 / p_i for non-zero probabilities (measured terms)
        weights_squared = weights.pow(2)
        nonzero_cost = torch.where(
            is_nonzero,
            weights_squared.unsqueeze(0) / probs_sum,
            torch.zeros_like(probs_sum)
        )
        
        # Sum over Pauli dimension for measured term
        nonzero_term = nonzero_cost.sum(dim=1)
        
        # OGM penalty for unmeasured terms: sum_{i: p_i = 0} w_i^2 * T
        # No cross terms, scales with T
        zero_cost = torch.where(
            is_zero,
            weights_squared.unsqueeze(0) * n_measurements,
            torch.zeros_like(probs_sum)
        )
        zero_term = zero_cost.sum(dim=1)
        
        # Total batch cost
        costs = nonzero_term + zero_term
        return costs

class L1Cost(CostFunction):
    """
    L1-based cost function: sum_{P: p(P) > 0} |w_P| / p(P) + sum_{P: p(P) = 0} |w_P|

    Key differences from LinearBiasCost (L2-like):
    - LinearBiasCost: w² / p for measured, (Σ|w|)² for unmeasured (variance-based)
    - L1Cost: |w| / p for measured, Σ|w| for unmeasured (mean absolute error-based)

    Properties:
    - Linear in weights (vs quadratic in LinearBiasCost)
    - No cross terms for unmeasured Paulis
    - More robust to large weight outliers
    - Gentler gradients during optimization
    - Better for Hamiltonians with many similar-magnitude weights

    Mathematical interpretation:
    - Minimizes expected absolute error rather than variance
    - More uniform penalty across Pauli terms
    """
    
    def __init__(self, epsilon_zero: float = 1e-4):
        self.epsilon_zero = epsilon_zero  # Threshold for considering probability as zero
    
    def compute(self, probs: torch.Tensor, weights: torch.Tensor, epsilon: float) -> torch.Tensor:
        """
        Compute L1 cost for each measurement trajectory.

        Args:
            probs: Tensor of shape (batch_size, n_measurements, n_paulis)
            weights: Tensor of shape (n_paulis,) - already |w| from CostComputer
            epsilon: Error parameter (not used in this formulation)

        Returns:
            costs: Tensor of shape (batch_size, n_measurements)
        """
        # Sum probabilities over measurements for each Pauli
        probs_sum = probs.sum(dim=1)  # Shape: (batch_size, n_paulis)
        is_zero = probs_sum < self.epsilon_zero
        is_nonzero = ~is_zero
        
        # Term 1: |w| / p for measured terms (linear in weight)
        weights_abs = weights.abs()
        nonzero_cost = torch.where(
            is_nonzero,
            weights_abs.unsqueeze(0) / probs_sum,
            torch.zeros_like(probs_sum)
        )
        nonzero_term = nonzero_cost.sum(dim=1)
        
        # Term 2: Σ|w| for unmeasured terms (linear sum, no squaring)
        zero_cost = torch.where(
            is_zero,
            weights_abs.unsqueeze(0),
            torch.zeros_like(probs_sum)
        )
        zero_term = zero_cost.sum(dim=1)
        
        return nonzero_term + zero_term
    
    def compute_batch(self, probs: torch.Tensor, weights: torch.Tensor, epsilon: float) -> torch.Tensor:
        """
        Compute batch L1 cost summed over measurements.

        Args:
            probs: Tensor of shape (batch_size, n_measurements, n_paulis)
            weights: Tensor of shape (n_paulis,) - already |w| from CostComputer
            epsilon: Error parameter (not used in this formulation)

        Returns:
            costs: Tensor of shape (batch_size,)
        """
        # Sum probabilities over measurements for each Pauli
        probs_sum = probs.sum(dim=1)  # Shape: (batch_size, n_paulis)
        is_zero = probs_sum < self.epsilon_zero
        is_nonzero = ~is_zero
        
        # Term 1: |w| / p for measured terms (linear in weight)
        weights_abs = weights.abs()
        nonzero_cost = torch.where(
            is_nonzero,
            weights_abs.unsqueeze(0) / probs_sum,
            torch.zeros_like(probs_sum)
        )
        nonzero_term = nonzero_cost.sum(dim=1)
        
        # Term 2: Σ|w| for unmeasured terms (linear sum, no squaring)
        zero_cost = torch.where(
            is_zero,
            weights_abs.unsqueeze(0),
            torch.zeros_like(probs_sum)
        )
        zero_term = zero_cost.sum(dim=1)
        
        return nonzero_term + zero_term


class ConfidenceCost(CostFunction):
    """
    Confidence cost function: Cost_CONF(U) = 2 * sum_P exp(-epsilon^2 / (2 * alpha_P^2) * h_U(P))

    where alpha_P are the (prepared) Pauli weights, h_U(P) is the expected hit
    count of Pauli P under the measurement ensemble U, and epsilon is a
    hyperparameter (the target per-term estimation error).

    Interpretation: applying the two-sided Hoeffding inequality to ``h_U(P)``
    independent samples in ``[-alpha_P, alpha_P]`` gives the per-term bound
        P(|est_P - true_P| >= epsilon) <= 2 * exp(-h_U(P) * epsilon^2 / (2 * alpha_P^2))
    because the single-shot range is ``2 * alpha_P``. Summing over P is a
    union bound, so this cost is a state-independent upper-bound surrogate for
    the probability that any Hamiltonian-term estimate misses its target by
    more than epsilon; minimizing the bound does not claim to minimize the
    unknown true failure probability itself.

    Key differences from ExponentialCost (sum_P w_P * exp(-p * eps^2 / 2)):
    - Weights enter the exponent denominator, not as an outer factor: heavier
      terms decay slower and thus demand more hits before their bound shrinks.
    - Every unmeasured non-zero-weight Pauli contributes the full bound of 2,
      regardless of its weight.

    Zero-weight terms (e.g. the identity after normalization, which zeroes it)
    are excluded: a term absent from the Hamiltonian has zero failure
    probability, and naively dividing by alpha_P^2 = 0 would yield NaN at
    h_U(P) = 0.

    Note: when CostComputer weight normalization is enabled (the trainer
    default), alpha_P are the normalized weights, so epsilon is expressed on
    the normalized scale — consistent with how epsilon is treated by the other
    cost families.

    Float16 probability tensors are supported for inference. Training with
    float16 probabilities is rejected because valid small coefficients can
    produce gradients outside the float16 representable range.
    """

    # Clamp floor for alpha_P^2 keeps the exponent rate finite so that masked
    # near-zero weights at h=0 give exp(-0 * huge) = 1 instead of -inf * 0 = NaN.
    _ALPHA_SQ_FLOOR = 1e-30

    @staticmethod
    def _validate_probability_dtype(probs: torch.Tensor) -> None:
        """Reject reduced-precision autograd before its steep gradients overflow."""
        if (
            torch.is_grad_enabled()
            and probs.requires_grad
            and probs.dtype == torch.float16
        ):
            raise ValueError(
                "ConfidenceCost requires float32 or float64 probability tensors "
                "when float16 gradients are enabled; float16 gradients can overflow"
            )

    def _exponent_rate_and_mask(self, weights: torch.Tensor,
                                epsilon: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return the exponent rate and non-zero mask in a numerically safe dtype."""
        # ``1e-30`` underflows to zero in float16. Promote reduced-precision
        # inputs before clamping so an unmeasured zero-weight term evaluates as
        # ``exp(-0 * finite) * 0 == 0`` rather than ``exp(-0 * inf) * 0 == NaN``.
        compute_dtype = (
            torch.float32
            if weights.dtype in (torch.float16, torch.bfloat16)
            else weights.dtype
        )
        compute_weights = weights.to(dtype=compute_dtype)
        alpha_sq = compute_weights.pow(2).clamp(min=self._ALPHA_SQ_FLOOR)
        rate = epsilon**2 / (2 * alpha_sq)
        mask = (compute_weights > 0).to(compute_dtype)
        return rate, mask

    def compute(self, probs: torch.Tensor, weights: torch.Tensor, epsilon: float) -> torch.Tensor:
        """
        Compute confidence cost for each measurement trajectory

        Args:
            probs: Tensor of shape (batch_size, n_measurements, n_paulis)
            weights: Tensor of shape (n_paulis,) - assumed already non-negative from CostComputer
            epsilon: Target per-term estimation error

        Returns:
            costs: Tensor of shape (batch_size, n_measurements)
        """
        self._validate_probability_dtype(probs)
        rate, mask = self._exponent_rate_and_mask(weights, epsilon)
        exp_term = torch.exp(-probs * rate)
        costs = 2 * torch.einsum('bmp,p->bm', exp_term, mask)
        return costs

    def compute_batch(self, probs: torch.Tensor, weights: torch.Tensor, epsilon: float) -> torch.Tensor:
        """
        Compute batch cost using the full ensemble: h_U(P) = sum over measurements

        Args:
            probs: Tensor of shape (batch_size, n_measurements, n_paulis)
            weights: Tensor of shape (n_paulis,) - assumed already non-negative from CostComputer
            epsilon: Target per-term estimation error

        Returns:
            costs: Tensor of shape (batch_size,)
        """
        self._validate_probability_dtype(probs)
        rate, mask = self._exponent_rate_and_mask(weights, epsilon)
        hit_counts = probs.sum(dim=1)  # h_U(P), shape (batch_size, n_paulis)
        exp_terms = torch.exp(-hit_counts * rate)
        costs = 2 * torch.matmul(exp_terms, mask)
        return costs


class LogCost(CostFunction):
    """Logarithmic cost function: sum_i w_i * log(1 + epsilon / (p_i + delta))"""
    
    def __init__(self, delta: float = 1e-6):
        self.delta = delta
    
    def compute(self, probs: torch.Tensor, weights: torch.Tensor, epsilon: float) -> torch.Tensor:
        """
        Compute logarithmic cost for each measurement trajectory

        Args:
            probs: Tensor of shape (batch_size, n_measurements, n_paulis)
            weights: Tensor of shape (n_paulis,)
            epsilon: Error parameter

        Returns:
            costs: Tensor of shape (batch_size, n_measurements)
        """
        log_term = torch.log(1 + epsilon / (probs + self.delta))
        costs = torch.einsum('bmp,p->bm', log_term, weights)
        return costs
    
    def compute_batch(self, probs: torch.Tensor, weights: torch.Tensor, epsilon: float) -> torch.Tensor:
        """
        Compute batch cost summed over measurements

        Args:
            probs: Tensor of shape (batch_size, n_measurements, n_paulis)
            weights: Tensor of shape (n_paulis,)
            epsilon: Error parameter

        Returns:
            costs: Tensor of shape (batch_size,)
        """
        log_term = torch.log(1 + epsilon / (probs + self.delta))
        summed_log = log_term.sum(dim=1)
        costs = torch.matmul(summed_log, weights)
        return costs


# Single source of truth for cost-type dispatch (used by CostComputer.__init__ and
# set_cost_type). Values are zero-argument constructors. The lambdas intentionally
# resolve the public class symbols at call time, preserving the legacy dispatch's
# monkeypatch/subclass hooks as well as its bit-identical default instances.
# These are the supported config/API families (linear_bias / ogm / l1 /
# exponential / logarithmic / confidence) — do not remove without a config, schedule,
# job-script, and public-API usage check.
_COST_FACTORY = {
    'exponential': lambda: ExponentialCost(),
    'linear_bias': lambda: LinearBiasCost(),
    'logarithmic': lambda: LogCost(),
    'ogm': lambda: OGMCost(),
    'l1': lambda: L1Cost(),
    'confidence': lambda: ConfidenceCost(),
}


def _make_cost_fn(cost_type: str) -> CostFunction:
    """Instantiate the cost function for ``cost_type``; raise ValueError if unknown."""
    try:
        factory = _COST_FACTORY[cost_type]
    except (KeyError, TypeError):
        # Raise outside the exception handler so callers retain the legacy
        # ValueError contract without a leaked dictionary-lookup traceback.
        pass
    else:
        return factory()
    raise ValueError(f"Unknown cost type: {cost_type}")


class CostComputer:
    """
    Helper class for computing various cost functions for quantum circuits

    This class provides a unified interface for different cost functions
    that can be used with Clifford tableau simulations.
    """
    
    CostType = Literal['exponential', 'linear_bias', 'logarithmic', 'ogm', 'l1', 'confidence']
    cost_fn: CostFunction
    
    def __init__(self, cost_type: CostType = 'exponential',
                 n_measurements: Optional[int] = None,
                 device: torch.device = torch.device('cpu'),
                 normalize_weights: bool = True,
                 normalization_type: NormalizationType = 'sum',
                 pauli_strings: Optional[List[str]] = None,
                 n_qubits: Optional[int] = None,
                 zero_weight_mask: Optional[List[bool]] = None):
        """
        Initialize the cost computer

        Args:
            cost_type: Type of cost function to use
            n_measurements: Number of measurements
            device: Device to perform computations on
            normalize_weights: If True (default), normalize weights (excluding identity)
            normalization_type: 'sum' or 'max' - how to normalize weights
                - 'sum': divide by sum of |w_i|, makes sum(w_norm) = 1
                - 'max': divide by max of |w_i|, makes max(w_norm) = 1
                  Better preserves relative scale for bias term interpretation
            pauli_strings: List of Pauli strings (needed for identity exclusion in normalization)
            n_qubits: Number of qubits (inferred from pauli_strings if not provided)
            zero_weight_mask: Optional bool sequence aligned with pauli_strings; True entries
                get weight FORCED TO ZERO before normalization, so sum-normalization
                redistributes over the remaining terms only. Used by
                ``zero_stabilizer_cost_weights`` to strip compact-encoding stabilizer
                penalty terms (zero-information on the code space) from the cost.
        """
        self.cost_type = cost_type
        self.device = device
        self.n_measurements = n_measurements
        self.normalize_weights = normalize_weights
        self.normalization_type = normalization_type
        self.pauli_strings = pauli_strings
        self.n_qubits = n_qubits

        # GPU OPTIMIZATION: Pre-compute non-identity mask once during init
        self._non_identity_mask: Optional[torch.Tensor] = None
        self._cached_normalized_weights: Optional[torch.Tensor] = None
        self._cached_weights_hash: Optional[int] = None

        # Optional pre-normalization zero mask (True = force weight to 0). Stored as a
        # float KEEP-mask so the hot path is a single elementwise multiply.
        self._zero_weight_keep: Optional[torch.Tensor] = None
        if zero_weight_mask is not None:
            zmask = torch.as_tensor(list(zero_weight_mask), dtype=torch.bool, device=device)
            if pauli_strings is not None and len(zmask) != len(pauli_strings):
                raise ValueError(
                    f"zero_weight_mask has {len(zmask)} entries but pauli_strings has "
                    f"{len(pauli_strings)} — they must be aligned."
                )
            self._zero_weight_keep = (~zmask).float()
        
        if normalize_weights and pauli_strings is not None:
            if n_qubits is None:
                n_qubits = len(pauli_strings[0]) if pauli_strings else 0
            self._non_identity_mask = create_non_identity_mask(
                pauli_strings, n_qubits, device
            )
        
        # Initialize the appropriate cost function (single dispatch -> _COST_FACTORY)
        self.cost_fn = _make_cost_fn(cost_type)

    def precompute_weights(self, pauli_weights: List[float]) -> None:
        """
        GPU OPTIMIZATION: Pre-compute and cache normalized weights during initialization.
        Call this once after creating the CostComputer to eliminate all per-call overhead.

        Args:
            pauli_weights: List of weights for each Pauli string (typically hamiltonian_helper.w_list)
        """
        weights = torch.tensor(pauli_weights, dtype=torch.float32, device=self.device)

        # Zero masked terms BEFORE normalization so sum-normalization redistributes
        # over the remaining (physical) terms only (zero_stabilizer_cost_weights).
        if self._zero_weight_keep is not None:
            weights = weights * self._zero_weight_keep

        if self.normalize_weights:
            weights = normalize_pauli_weights(
                weights,
                pauli_strings=self.pauli_strings,
                n_qubits=self.n_qubits,
                non_identity_mask=self._non_identity_mask,
                normalization_type=self.normalization_type
            )
        else:
            weights = weights.abs()

        self._cached_normalized_weights = weights
        self._cached_weights_hash = id(pauli_weights)
    
    def _prepare_weights(self, pauli_weights: List[float]) -> torch.Tensor:
        """
        Prepare weights tensor, applying normalization if enabled.
        GPU-optimized: uses cached mask and weights when possible.

        Args:
            pauli_weights: List of weights for each Pauli string

        Returns:
            weights: Tensor of (potentially normalized) weights
        """
        # GPU OPTIMIZATION: Use object ID for fast cache check
        # In typical training, pauli_weights is the same list object every call
        weights_id = id(pauli_weights)
        if (self._cached_normalized_weights is not None and 
            self._cached_weights_hash == weights_id):
            return self._cached_normalized_weights
        
        # Cache miss - create weights tensor on device
        weights = torch.tensor(pauli_weights, dtype=torch.float32, device=self.device)

        # Zero masked terms BEFORE normalization (see precompute_weights /
        # zero_stabilizer_cost_weights): normalization then redistributes over
        # the remaining terms only.
        if self._zero_weight_keep is not None:
            weights = weights * self._zero_weight_keep

        if self.normalize_weights:
            # GPU-optimized path: use pre-computed mask
            weights = normalize_pauli_weights(
                weights,
                pauli_strings=self.pauli_strings,
                n_qubits=self.n_qubits,
                non_identity_mask=self._non_identity_mask,
                normalization_type=self.normalization_type
            )
        else:
            weights = weights.abs()
        
        # Cache the result using object ID (fast O(1) check)
        self._cached_normalized_weights = weights
        self._cached_weights_hash = weights_id
        
        return weights
    
    def compute_costs(self, probs: torch.Tensor, pauli_weights: List[float], 
                     epsilon: float) -> torch.Tensor:
        """
        Compute costs for each measurement trajectory

        Args:
            probs: Probability tensor from tableau.prob_P_batch_multi()
                   Shape: (batch_size, n_measurements, n_paulis)
            pauli_weights: List of weights for each Pauli string
            epsilon: Error parameter

        Returns:
            costs: Tensor of shape (batch_size, n_measurements)
        """
        # Convert weights to tensor (with optional normalization)
        weights = self._prepare_weights(pauli_weights)
        
        # Ensure probs is on the correct device
        if probs.device != self.device:
            probs = probs.to(self.device)
        
        return self.cost_fn.compute(probs, weights, epsilon)
    
    def compute_batch_cost(self, probs: torch.Tensor, pauli_weights: List[float], 
                          epsilon: float) -> torch.Tensor:
        """
        Compute batch cost (summed over measurements)

        Args:
            probs: Probability tensor from tableau.prob_P_batch_multi()
                   Shape: (batch_size, n_measurements, n_paulis)
            pauli_weights: List of weights for each Pauli string
            epsilon: Error parameter

        Returns:
            costs: Tensor of shape (batch_size,)
        """
        # Convert weights to tensor (with optional normalization)
        weights = self._prepare_weights(pauli_weights)
        
        # Ensure probs is on the correct device
        if probs.device != self.device:
            probs = probs.to(self.device)
        
        return self.cost_fn.compute_batch(probs, weights, epsilon)
    
    def compute_mean_cost(self, probs: torch.Tensor, pauli_weights: List[float], 
                         epsilon: float,) -> torch.Tensor:
        """
        Compute mean cost across all trajectories

        Args:
            probs: Probability tensor from tableau.prob_P_batch_multi()
                   Shape: (batch_size, n_measurements, n_paulis)
            pauli_weights: List of weights for each Pauli string
            epsilon: Error parameter

        Returns:
            mean_cost: Scalar tensor with mean cost
        """
        costs = self.compute_batch_cost(probs, pauli_weights, epsilon)
        return costs.mean()

    def set_cost_type(self, cost_type: CostType, n_measurements: Optional[int] = None):
        """
        Change the cost function type

        Args:
            cost_type: New cost function type
            n_measurements: Number of measurements
        """
        self.cost_type = cost_type
        self.cost_fn = _make_cost_fn(cost_type)

        if n_measurements is not None:
            self.n_measurements = n_measurements
    
    def add_custom_cost_function(self, name: str, cost_fn: CostFunction):
        """
        Add a custom cost function

        Args:
            name: Name for the custom cost function
            cost_fn: Instance of CostFunction
        """
        # Store custom functions in a registry
        if not hasattr(self, '_custom_functions'):
            self._custom_functions = {}
        
        self._custom_functions[name] = cost_fn
        
        # If setting as current, use it
        if self.cost_type == name:
            self.cost_fn = cost_fn


# Example custom cost function
class ThresholdCost(CostFunction):
    """Custom cost function with threshold behavior"""
    
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
    
    def compute(self, probs: torch.Tensor, weights: torch.Tensor, epsilon: float) -> torch.Tensor:
        """Cost is weight if probability < threshold, 0 otherwise"""
        below_threshold = (probs < self.threshold).float()
        costs = torch.einsum('bmp,p->bm', below_threshold, weights)
        return costs
    
    def compute_batch(self, probs: torch.Tensor, weights: torch.Tensor, epsilon: float) -> torch.Tensor:
        """Batch version of threshold cost"""
        below_threshold = (probs < self.threshold).float()
        summed = below_threshold.sum(dim=1)
        costs = torch.matmul(summed, weights)
        return costs
