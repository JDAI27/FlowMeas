# -*- coding: utf-8 -*-
# cost_computer.py
import torch
from typing import List, Tuple, Optional, Literal, Callable, Union
from abc import ABC, abstractmethod


NormalizationType = Literal['sum', 'max']


def normalize_pauli_weights(weights: torch.Tensor, 
                            pauli_strings: Optional[List[str]] = None,
                            n_qubits: Optional[int] = None,
                            non_identity_mask: Optional[torch.Tensor] = None,
                            normalization_type: NormalizationType = 'sum') -> torch.Tensor:
    """Normalize Pauli weights (excluding identity). GPU-optimized."""
    weights_abs = weights.abs()
    
    if non_identity_mask is not None:
        masked_weights = weights_abs * non_identity_mask.float()
        if normalization_type == 'max':
            norm_factor = masked_weights.max().clamp(min=1e-10)
        else:  # 'sum'
            norm_factor = masked_weights.sum().clamp(min=1e-10)
        normalized = masked_weights / norm_factor
        return normalized
    
    if pauli_strings is not None:
        if n_qubits is None:
            n_qubits = len(pauli_strings[0]) if pauli_strings else 0
        identity_str = "I" * n_qubits
        
        non_identity_mask = torch.tensor(
            [p != identity_str for p in pauli_strings],
            dtype=torch.bool,
            device=weights.device
        )
        
        masked_weights = weights_abs * non_identity_mask.float()
        if normalization_type == 'max':
            norm_factor = masked_weights.max().clamp(min=1e-10)
        else:  # 'sum'
            norm_factor = masked_weights.sum().clamp(min=1e-10)
        normalized = masked_weights / norm_factor
        return normalized
    
    if normalization_type == 'max':
        norm_factor = weights_abs.max().clamp(min=1e-10)
    else:  # 'sum'
        norm_factor = weights_abs.sum().clamp(min=1e-10)
    normalized = weights_abs / norm_factor
    return normalized


def create_non_identity_mask(pauli_strings: List[str], 
                              n_qubits: int,
                              device: torch.device) -> torch.Tensor:
    """Pre-compute non-identity mask (True = non-identity term)."""
    identity_str = "I" * n_qubits
    return torch.tensor(
        [p != identity_str for p in pauli_strings],
        dtype=torch.bool,
        device=device
    )


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
    """Exponential cost: sum_i w_i^2 * exp(-p_i * epsilon^2 / 2)"""
    
    def compute(self, probs: torch.Tensor, weights: torch.Tensor, epsilon: float) -> torch.Tensor:
        exp_term = torch.exp(-probs * epsilon**2 / 2)
        costs = torch.einsum('bmp,p->bm', exp_term, weights)
        return costs
    
    def compute_batch(self, probs: torch.Tensor, weights: torch.Tensor, epsilon: float) -> torch.Tensor:
        exponent_terms = torch.sum(-probs * epsilon**2 / 2, dim=1)
        exp_terms = torch.exp(exponent_terms)
        costs = torch.matmul(exp_terms, weights)
        return costs

class LinearBiasCost(CostFunction):
    """Linear cost: sum_{p_i != 0} w_i^2/p_i + (sum_{p_i = 0} |w_i|)^2"""
    
    def __init__(self, epsilon_zero: float = 1e-4):
        self.epsilon_zero = epsilon_zero
    
    def compute(self, probs: torch.Tensor, weights: torch.Tensor, epsilon: float) -> torch.Tensor:
        probs_sum = probs.sum(dim=1)
        is_zero = probs_sum < self.epsilon_zero
        is_nonzero = ~is_zero
        
        # Compute w_i^2 / p_i for non-zero probabilities
        weights_squared = weights.pow(2)
        nonzero_cost = torch.where(
            is_nonzero,
            weights_squared.unsqueeze(0) / (probs_sum),
            torch.zeros_like(probs_sum)
        )
        
        nonzero_term = nonzero_cost.sum(dim=1)
        weights_masked = is_zero * weights.abs().unsqueeze(0)
        sum_zero_weights = weights_masked.sum(dim=1)
        zero_term = sum_zero_weights.pow(2)
        
        costs = nonzero_term + zero_term
        return costs
    
    def compute_batch(self, probs: torch.Tensor, weights: torch.Tensor, epsilon: float) -> torch.Tensor:
        n_measurements = probs.shape[1]
        
        probs_sum = probs.sum(dim=1)
        is_zero = probs_sum < self.epsilon_zero
        is_nonzero = ~is_zero
        
        weights_squared = weights.pow(2)
        nonzero_cost = torch.where(
            is_nonzero,
            weights_squared.unsqueeze(0) / probs_sum,
            torch.zeros_like(probs_sum)
        )
        
        nonzero_term = nonzero_cost.sum(dim=1)
        weights_masked = is_zero * weights.abs().unsqueeze(0)
        sum_zero_weights = weights_masked.sum(dim=1)
        zero_term = sum_zero_weights ** 2
        
        costs = nonzero_term + zero_term
        return costs


class OGMCost(CostFunction):
    """OGM cost (arXiv:2105.13091): sum_{p>0} w²/p + sum_{p=0} w²*T"""
    
    def __init__(self, epsilon_zero: float = 1e-4):
        self.epsilon_zero = epsilon_zero
    
    def compute(self, probs: torch.Tensor, weights: torch.Tensor, epsilon: float) -> torch.Tensor:
        n_measurements = probs.shape[1]
        
        probs_sum = probs.sum(dim=1)
        is_zero = probs_sum < self.epsilon_zero
        is_nonzero = ~is_zero
        
        weights_squared = weights.pow(2)
        nonzero_cost = torch.where(
            is_nonzero,
            weights_squared.unsqueeze(0) / probs_sum,
            torch.zeros_like(probs_sum)
        )
        
        nonzero_term = nonzero_cost.sum(dim=1)
        zero_cost = torch.where(
            is_zero,
            weights_squared.unsqueeze(0) * n_measurements,
            torch.zeros_like(probs_sum)
        )
        zero_term = zero_cost.sum(dim=1)
        
        costs = nonzero_term + zero_term
        return costs
    
    def compute_batch(self, probs: torch.Tensor, weights: torch.Tensor, epsilon: float) -> torch.Tensor:
        n_measurements = probs.shape[1]
        
        probs_sum = probs.sum(dim=1)
        is_zero = probs_sum < self.epsilon_zero
        is_nonzero = ~is_zero
        
        weights_squared = weights.pow(2)
        nonzero_cost = torch.where(
            is_nonzero,
            weights_squared.unsqueeze(0) / probs_sum,
            torch.zeros_like(probs_sum)
        )
        
        nonzero_term = nonzero_cost.sum(dim=1)
        zero_cost = torch.where(
            is_zero,
            weights_squared.unsqueeze(0) * n_measurements,
            torch.zeros_like(probs_sum)
        )
        zero_term = zero_cost.sum(dim=1)
        
        costs = nonzero_term + zero_term
        return costs


class L1Cost(CostFunction):
    """L1 cost: sum_{p>0} |w|/p + sum_{p=0} |w|"""
    
    def __init__(self, epsilon_zero: float = 1e-4):
        self.epsilon_zero = epsilon_zero
    
    def compute(self, probs: torch.Tensor, weights: torch.Tensor, epsilon: float) -> torch.Tensor:
        probs_sum = probs.sum(dim=1)
        is_zero = probs_sum < self.epsilon_zero
        is_nonzero = ~is_zero
        
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
        probs_sum = probs.sum(dim=1)
        is_zero = probs_sum < self.epsilon_zero
        is_nonzero = ~is_zero
        
        weights_abs = weights.abs()
        nonzero_cost = torch.where(
            is_nonzero,
            weights_abs.unsqueeze(0) / probs_sum,
            torch.zeros_like(probs_sum)
        )
        nonzero_term = nonzero_cost.sum(dim=1)
        
        zero_cost = torch.where(
            is_zero,
            weights_abs.unsqueeze(0),
            torch.zeros_like(probs_sum)
        )
        zero_term = zero_cost.sum(dim=1)
        
        return nonzero_term + zero_term


class LinearCost(CostFunction):
    """Linear cost: sum w²/p if all p>0, else (sum w)²"""
    
    def __init__(self, n_measurements: int, epsilon_zero: float = 1e-10):
        self.n_measurements = n_measurements
        self.epsilon_zero = epsilon_zero
    
    def compute(self, probs: torch.Tensor, weights: torch.Tensor, epsilon: float) -> torch.Tensor:
        is_zero = probs < self.epsilon_zero
        is_nonzero = ~is_zero
        
        weights_squared = weights.pow(2)
        nonzero_cost = torch.where(
            is_nonzero,
            weights_squared.unsqueeze(0).unsqueeze(0) / (probs + self.epsilon_zero),
            torch.zeros_like(probs)
        )
        
        nonzero_term = nonzero_cost.sum(dim=2)
        weights_masked = is_zero * weights.unsqueeze(0).unsqueeze(0)
        sum_zero_weights = weights_masked.sum(dim=2)
        zero_term = sum_zero_weights ** 2
        
        costs = nonzero_term + zero_term
        return costs
    
    def compute_batch(self, probs: torch.Tensor, weights: torch.Tensor, epsilon: float) -> torch.Tensor:
        is_zero = probs < self.epsilon_zero
        is_nonzero = ~is_zero
        
        weights_squared = weights ** 2
        nonzero_cost = torch.where(
            is_nonzero,
            weights_squared.unsqueeze(0).unsqueeze(0) / (probs + self.epsilon_zero),
            torch.zeros_like(probs)
        )
        
        nonzero_term = nonzero_cost.sum(dim=(1, 2))

        weights_expanded = weights.unsqueeze(0).unsqueeze(0)
        weights_masked = is_zero * weights_expanded
        sum_zero_weights_per_measurement = weights_masked.sum(dim=2)
        zero_term = (sum_zero_weights_per_measurement ** 2).sum(dim=1)
        
        costs = nonzero_term + zero_term
        return costs


class LogCost(CostFunction):
    """Logarithmic cost: sum_i w_i * log(1 + epsilon / (p_i + delta))"""
    
    def __init__(self, delta: float = 1e-6):
        self.delta = delta
    
    def compute(self, probs: torch.Tensor, weights: torch.Tensor, epsilon: float) -> torch.Tensor:
        log_term = torch.log(1 + epsilon / (probs + self.delta))
        costs = torch.einsum('bmp,p->bm', log_term, weights)
        return costs
    
    def compute_batch(self, probs: torch.Tensor, weights: torch.Tensor, epsilon: float) -> torch.Tensor:
        log_term = torch.log(1 + epsilon / (probs + self.delta))
        summed_log = log_term.sum(dim=1)
        costs = torch.matmul(summed_log, weights)
        return costs


class CostComputer:
    """Unified interface for quantum circuit cost functions."""
    
    CostType = Literal['exponential', 'linear', 'linear_bias', 'logarithmic', 'ogm', 'l1']
    cost_fn: CostFunction
    
    def __init__(self, cost_type: CostType = 'exponential', 
                 n_measurements: Optional[int] = None,
                 device: torch.device = torch.device('cpu'),
                 normalize_weights: bool = True,
                 normalization_type: NormalizationType = 'sum',
                 pauli_strings: Optional[List[str]] = None,
                 n_qubits: Optional[int] = None):
        self.cost_type = cost_type
        self.device = device
        self.n_measurements = n_measurements
        self.normalize_weights = normalize_weights
        self.normalization_type = normalization_type
        self.pauli_strings = pauli_strings
        self.n_qubits = n_qubits
        
        self._non_identity_mask: Optional[torch.Tensor] = None
        self._cached_normalized_weights: Optional[torch.Tensor] = None
        self._cached_weights_hash: Optional[int] = None
        
        if normalize_weights and pauli_strings is not None:
            if n_qubits is None:
                n_qubits = len(pauli_strings[0]) if pauli_strings else 0
            self._non_identity_mask = create_non_identity_mask(
                pauli_strings, n_qubits, device
            )
        
        if cost_type == 'exponential':
            self.cost_fn = ExponentialCost()
        elif cost_type == 'linear':
            if n_measurements is None:
                raise ValueError("n_measurements required for linear cost")
            self.cost_fn = LinearCost(n_measurements)
        elif cost_type == 'linear_bias':
            self.cost_fn = LinearBiasCost()
        elif cost_type == 'logarithmic':
            self.cost_fn = LogCost()
        elif cost_type == 'ogm':
            self.cost_fn = OGMCost()
        elif cost_type == 'l1':
            self.cost_fn = L1Cost()
        else:
            raise ValueError(f"Unknown cost type: {cost_type}")
    
    def precompute_weights(self, pauli_weights: List[float]) -> None:
        """Pre-compute and cache normalized weights."""
        weights = torch.tensor(pauli_weights, dtype=torch.float32, device=self.device)
        
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
        """Prepare weights tensor with optional normalization (cached)."""
        weights_id = id(pauli_weights)
        if (self._cached_normalized_weights is not None and 
            self._cached_weights_hash == weights_id):
            return self._cached_normalized_weights
        
        weights = torch.tensor(pauli_weights, dtype=torch.float32, device=self.device)
        
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
        self._cached_weights_hash = weights_id
        
        return weights
    
    def compute_costs(self, probs: torch.Tensor, pauli_weights: List[float], 
                     epsilon: float) -> torch.Tensor:
        """Compute costs for each measurement trajectory."""
        weights = self._prepare_weights(pauli_weights)
        
        if probs.device != self.device:
            probs = probs.to(self.device)
        
        return self.cost_fn.compute(probs, weights, epsilon)
    
    def compute_batch_cost(self, probs: torch.Tensor, pauli_weights: List[float], 
                          epsilon: float) -> torch.Tensor:
        """Compute batch cost (summed over measurements)."""
        weights = self._prepare_weights(pauli_weights)
        
        if probs.device != self.device:
            probs = probs.to(self.device)
        
        return self.cost_fn.compute_batch(probs, weights, epsilon)
    
    def compute_mean_cost(self, probs: torch.Tensor, pauli_weights: List[float], 
                         epsilon: float,) -> torch.Tensor:
        """Compute mean cost across all trajectories."""
        costs = self.compute_batch_cost(probs, pauli_weights, epsilon)
        return costs.mean()

    def set_cost_type(self, cost_type: CostType, n_measurements: Optional[int] = None):
        """Change the cost function type."""
        self.cost_type = cost_type
        
        if cost_type == 'exponential':
            self.cost_fn = ExponentialCost()
        elif cost_type == 'linear':
            if n_measurements is None and self.n_measurements is None:
                raise ValueError("n_measurements required for linear cost")
            n_meas = n_measurements if n_measurements is not None else self.n_measurements
            if n_meas is None:
                raise ValueError("n_measurements required for linear cost")
            self.cost_fn = LinearCost(n_meas)
        elif cost_type == 'linear_bias':
            self.cost_fn = LinearBiasCost()
        elif cost_type == 'logarithmic':
            self.cost_fn = LogCost()
        elif cost_type == 'ogm':
            self.cost_fn = OGMCost()
        elif cost_type == 'l1':
            self.cost_fn = L1Cost()
        else:
            raise ValueError(f"Unknown cost type: {cost_type}")
        
        if n_measurements is not None:
            self.n_measurements = n_measurements
    
    def add_custom_cost_function(self, name: str, cost_fn: CostFunction):
        """Add a custom cost function."""
        if not hasattr(self, '_custom_functions'):
            self._custom_functions = {}
        
        self._custom_functions[name] = cost_fn
        
        if self.cost_type == name:
            self.cost_fn = cost_fn


class ThresholdCost(CostFunction):
    """Threshold cost: weight if probability < threshold, 0 otherwise."""
    
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
