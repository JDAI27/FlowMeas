# -*- coding: utf-8 -*-
# cost_computer.py
import torch
from typing import List, Tuple, Optional, Literal, Callable, Union
from abc import ABC, abstractmethod


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
    """Exponential cost function: sum_i w_i^2 * exp(-p_i * epsilon^2 / 2)"""
    
    def compute(self, probs: torch.Tensor, weights: torch.Tensor, epsilon: float) -> torch.Tensor:
        """
        Compute exponential cost for each measurement trajectory
        
        Args:
            probs: Tensor of shape (batch_size, n_measurements, n_paulis)
            weights: Tensor of shape (n_paulis,)
            epsilon: Error parameter
            
        Returns:
            costs: Tensor of shape (batch_size, n_measurements)
        """
        weights = weights.abs() #.pow(2)  # Ensure weights are non-negative
        exp_term = torch.exp(-probs * epsilon**2 / 2)
        costs = torch.einsum('bmp,p->bm', exp_term, weights)
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
        weights = weights.abs() #.pow(2)  # Ensure weights are non-negative
        exponent_terms = torch.sum(-probs * epsilon**2 / 2, dim=1)
        exp_terms = torch.exp(exponent_terms)
        costs = torch.matmul(exp_terms, weights)
        return costs

class LinearBiasCost(CostFunction):
    """Linear cost function: sum_{i,p_i != 0} w_i^2 /(\sum_j^n_measurements p_{i,j}) + sum_{i,j, p_i = 0,p_j = 0} w_i w_j"""
    
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

class LinearCost(CostFunction):
    """Linear cost function: sum_{i} w_i^2 /p_i if all p_i != 0, else (sum_{i} w_i)^2"""
    
    def __init__(self, n_measurements: int, epsilon_zero: float = 1e-10):
        self.n_measurements = n_measurements
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
        is_zero = probs < self.epsilon_zero
        is_nonzero = ~is_zero
        
        # Compute w_i^2 / p_i for non-zero probabilities
        # Add small epsilon to avoid division by zero
        weights_squared = weights.pow(2)
        nonzero_cost = torch.where(
            is_nonzero,
            weights_squared.unsqueeze(0).unsqueeze(0) / (probs + self.epsilon_zero),
            torch.zeros_like(probs)
        )
        
        # Sum over Pauli dimension for non-zero term
        nonzero_term = nonzero_cost.sum(dim=2)
        
        # Compute sum_{i,j, p_i = 0, p_j = 0} w_i * w_j
        # This equals (sum_{i, p_i = 0} w_i)^2
        weights_masked = is_zero * weights.unsqueeze(0).unsqueeze(0)
        sum_zero_weights = weights_masked.sum(dim=2)
        zero_term = sum_zero_weights ** 2
        
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
        is_zero = probs < self.epsilon_zero
        is_nonzero = ~is_zero
        
        # Compute w_i^2 / p_i for non-zero probabilities
        weights_squared = weights ** 2
        nonzero_cost = torch.where(
            is_nonzero,
            weights_squared.unsqueeze(0).unsqueeze(0) / (probs + self.epsilon_zero),
            torch.zeros_like(probs)
        )
        
        # Sum over both measurements and Pauli dimensions
        nonzero_term = nonzero_cost.sum(dim=(1, 2))
        

        weights_expanded = weights.unsqueeze(0).unsqueeze(0)
        weights_masked = is_zero * weights_expanded
        sum_zero_weights_per_measurement = weights_masked.sum(dim=2)  # Shape: (batch_size, n_measurements)
        zero_term = (sum_zero_weights_per_measurement ** 2).sum(dim=1)
        
        # Total batch cost
        costs = nonzero_term + zero_term
        
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


class CostComputer:
    """
    Helper class for computing various cost functions for quantum circuits
    
    This class provides a unified interface for different cost functions
    that can be used with Clifford tableau simulations.
    """
    
    CostType = Literal['exponential', 'linear', 'linear_bias', 'logarithmic']
    
    def __init__(self, cost_type: CostType = 'exponential', 
                 n_measurements: Optional[int] = None,
                 device: torch.device = torch.device('cpu')):
        """
        Initialize the cost computer
        
        Args:
            cost_type: Type of cost function to use
            n_measurements: Number of measurements (required for linear cost)
            device: Device to perform computations on
        """
        self.cost_type = cost_type
        self.device = device
        self.n_measurements = n_measurements
        
        # Initialize the appropriate cost function
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
        else:
            raise ValueError(f"Unknown cost type: {cost_type}")
    
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
        # Convert weights to tensor
        weights = torch.tensor(pauli_weights, 
                             dtype=torch.float32, device=self.device).abs()
        
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
        # Convert weights to tensor
        weights = torch.tensor(pauli_weights, 
                             dtype=torch.float32, device=self.device)#.abs()
        
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
            n_measurements: Number of measurements (required for linear cost)
        """
        self.cost_type = cost_type
        
        if cost_type == 'exponential':
            self.cost_fn = ExponentialCost()
        elif cost_type == 'linear':
            if n_measurements is None and self.n_measurements is None:
                raise ValueError("n_measurements required for linear cost")
            n_meas = n_measurements if n_measurements is not None else self.n_measurements
            self.cost_fn = LinearCost(n_meas)
        elif cost_type == 'linear_bias':
            self.cost_fn = LinearBiasCost()
        elif cost_type == 'logarithmic':
            self.cost_fn = LogCost()
        else:
            raise ValueError(f"Unknown cost type: {cost_type}")
        
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
