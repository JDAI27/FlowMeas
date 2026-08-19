"""
Modular GFlowNet Training Objectives

This module provides different training objectives for GFlowNets that can be easily swapped.

IMPORTANT: All rewards passed to these objectives are expected to be in log space (log rewards).
"""

import logging

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple, Callable
import torch.nn.functional as F


class GFlowNetObjective(ABC):
    """Abstract base class for GFlowNet training objectives"""

    # Objectives that accept a static-shape ``valid_mask`` kwarg (masked
    # mean over valid rows) set this True so the trainer can skip the
    # boolean-index gather host sync. Default False → legacy boolean-filter path.
    supports_valid_mask = False

    @abstractmethod
    def compute_loss(self, 
                    forward_flows: torch.Tensor,
                    backward_flows: torch.Tensor,
                    rewards: torch.Tensor,
                    logZ: torch.nn.Parameter,
                    **kwargs) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute the loss for the given flows and rewards.

        Args:
            forward_flows: Log probabilities of forward trajectories [batch_size]
            backward_flows: Log probabilities of backward trajectories [batch_size]
            rewards: Log rewards for each trajectory [batch_size] (already in log space)
            logZ: Learned log partition function parameter
            **kwargs: Additional arguments specific to the objective

        Returns:
            loss: Scalar loss tensor
            metrics: Dictionary of metrics to log
        """
        pass


class TrajectoryBalance(GFlowNetObjective):
    """
    Trajectory Balance (TB) objective - the original GFlowNet objective.

    Loss = E[(log Z + log P_F(τ) - log R(τ) - log P_B(τ))²]

    Note: Expects rewards to already be in log space.
    """
    
    # This objective supports a static-
    # shape ``valid_mask`` reduction, so ``EfficientGFNTrainer.compute_loss`` can
    # skip the boolean-index gather (a host sync) on the gradient path.
    supports_valid_mask = True

    def __init__(self, loss_type: str = "squared"):
        """
        Args:
            loss_type: Type of loss - "squared", "abs", or "huber"
        """
        self.loss_type = loss_type

    def compute_loss(self,
                    forward_flows: torch.Tensor,
                    backward_flows: torch.Tensor,
                    rewards: torch.Tensor,
                    logZ: torch.nn.Parameter,
                    valid_mask: Optional[torch.Tensor] = None,
                    **kwargs) -> Tuple[torch.Tensor, Dict[str, float]]:

        # Rewards are already in log space
        log_rewards = rewards

        # Compute trajectory balance error
        # log Z + log P_F(τ) = log R(τ) + log P_B(τ)
        tb_error = logZ + forward_flows - log_rewards - backward_flows

        # Check for NaN/Inf in tb_error.: gate this debug probe behind
        # DEBUG-level logging so the ``.any`` host sync (a device→host copy)
        # does not run on the default training path — the masked branch below
        # is sync-free, and ``EfficientGFNTrainer.update_step`` already performs
        # the single intended fused finiteness guard that skips the optimizer
        # step on any non-finite loss/grad. This is a pure logging side-effect.
        if logging.getLogger().isEnabledFor(logging.DEBUG) and (
                torch.isnan(tb_error).any() or torch.isinf(tb_error).any()):
            logging.warning("NaN/Inf in TB error! Components:")
            logging.warning(f"  logZ: {logZ.item()}")
            logging.warning(f"  forward_flows: min={forward_flows.min().item():.4f}, max={forward_flows.max().item():.4f}, has_nan={torch.isnan(forward_flows).any()}")
            logging.warning(f"  log_rewards: min={log_rewards.min().item():.4f}, max={log_rewards.max().item():.4f}, has_nan={torch.isnan(log_rewards).any()}")
            logging.warning(f"  backward_flows: min={backward_flows.min().item():.4f}, max={backward_flows.max().item():.4f}, has_nan={torch.isnan(backward_flows).any()}")

        if valid_mask is not None:
            # Masked static-shape reduction. ``loss`` is the mean of the
            # per-row error over the valid rows ONLY (invalid rows zero-weighted
            # in the numerator AND excluded from the denominator) — exactly equal
            # to the legacy ``tb_error[valid_mask]`` reductions for ALL inputs
            # (including non-finite invalid rows, which the ``torch.where`` guard
            # below drops before weighting so inf*0=nan cannot leak in), with a
            # fixed (B,) shape and no boolean-index host sync. The metrics are
            # detached logging values (population, not Bessel-corrected, std).
            w = valid_mask.to(tb_error.dtype)
            denom = w.sum().clamp(min=1.0)
            # Mask tb_error to a finite 0 on invalid rows BEFORE the nonlinear op,
            # and use ``tb_safe`` as the sole source for the loss and every metric.
            # Masking after the square/abs/huber is not enough: a non-finite
            # tb_error on a masked-off row poisons the forward (inf*0 = NaN) and the
            # backward (``pow``'s local grad is inf there, and the weight zero makes
            # 0*inf = NaN), which reaches logZ / forward_flows and trips
            # update_step's finiteness guard as a silently skipped update.
            valid_bool = valid_mask.bool()
            tb_safe = torch.where(valid_bool, tb_error, torch.zeros_like(tb_error))
            if self.loss_type == "squared":
                per_row = tb_safe.pow(2)
            elif self.loss_type == "abs":
                per_row = tb_safe.abs()
            elif self.loss_type == "huber":
                per_row = F.huber_loss(
                    tb_safe, torch.zeros_like(tb_safe), reduction='none'
                )
            else:
                raise ValueError(f"Unknown loss type: {self.loss_type}")
            loss = (per_row * w).sum() / denom
            err_mean = (tb_safe * w).sum() / denom
            # ``err_mean``/``err_var`` feed only detached logging metrics (never
            # ``loss``). Detach ``err_mean`` before the variance so autograd does
            # not build (and immediately discard) the unused subgraph through
            # ``(tb_safe - err_mean)`` on every masked step. Values are unchanged
            # (detach is value-preserving); ``loss`` is independent of both.
            err_mean_d = err_mean.detach()
            err_var = (((tb_safe - err_mean_d) ** 2) * w).sum() / denom
            metrics = {
                'tb_error_mean': err_mean.detach(),
                'tb_error_std': err_var.sqrt().detach(),
                'tb_error_abs_mean': (tb_safe.abs() * w).sum().detach() / denom,
            }
            return loss, metrics

        # Apply loss function
        if self.loss_type == "squared":
            loss = tb_error.pow(2).mean()
        elif self.loss_type == "abs":
            loss = tb_error.abs().mean()
        elif self.loss_type == "huber":
            loss = F.huber_loss(tb_error, torch.zeros_like(tb_error), reduction='mean')
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        # Return GPU tensors - caller will convert for logging/serialization
        metrics = {
            'tb_error_mean': tb_error.mean().detach(),
            'tb_error_std': tb_error.std().detach(),
            'tb_error_abs_mean': tb_error.abs().mean().detach(),
        }

        return loss, metrics


class DetailedBalance(GFlowNetObjective):
    """
    Detailed Balance (DB) objective - enforces balance at each state.

    This requires state-level information and is more complex to implement.
    Note: Expects rewards to already be in log space.
    """
    
    def __init__(self, epsilon: float = 1e-8):
        self.epsilon = epsilon
    
    def compute_loss(self,
                    forward_flows: torch.Tensor,
                    backward_flows: torch.Tensor,
                    rewards: torch.Tensor,
                    logZ: torch.nn.Parameter,
                    states_info: Optional[Dict] = None,
                    **kwargs) -> Tuple[torch.Tensor, Dict[str, float]]:
        
        if states_info is None:
            raise ValueError("Detailed Balance requires states_info with transition information")
        
        # Extract state transition information
        state_pairs = states_info['state_pairs']  # List of (s, s') pairs
        forward_probs = states_info['forward_probs']  # P_F(s'|s)
        backward_probs = states_info['backward_probs']  # P_B(s|s')
        state_flows = states_info['state_flows']  # F(s) values
        
        # Detailed balance: F(s)P_F(s'|s) = F(s')P_B(s|s')
        losses = []
        
        for i, (s, s_prime) in enumerate(state_pairs):
            if s is None:  # Initial state
                flow_s = logZ
            else:
                flow_s = state_flows[s]
            
            flow_s_prime = state_flows[s_prime]
            
            # log F(s) + log P_F(s'|s) = log F(s') + log P_B(s|s')
            lhs = flow_s + forward_probs[i]
            rhs = flow_s_prime + backward_probs[i]
            
            db_error = lhs - rhs
            losses.append(db_error.pow(2))
        
        loss = torch.stack(losses).mean()
        
        # Return GPU tensors - caller will convert for logging/serialization
        metrics = {
            'db_error_mean': torch.stack(losses).sqrt().mean().detach(),
            'num_transitions': len(state_pairs),
        }
        
        return loss, metrics


class SubTrajectoryBalance(GFlowNetObjective):
    """
    Sub-Trajectory Balance (SubTB) - enforces balance for all sub-trajectories.

    More sample-efficient than TB by using all partial trajectories.
    Note: Expects partial_rewards to already be in log space.
    """
    
    def __init__(self, lambda_coef: float = 0.9):
        """
        Args:
            lambda_coef: Geometric weighting coefficient for sub-trajectories
        """
        self.lambda_coef = lambda_coef
    
    def compute_loss(self,
                    forward_flows: torch.Tensor,
                    backward_flows: torch.Tensor,
                    rewards: torch.Tensor,
                    logZ: torch.nn.Parameter,
                    trajectories_info: Optional[Dict] = None,
                    **kwargs) -> Tuple[torch.Tensor, Dict[str, float]]:
        
        if trajectories_info is None:
            # Fall back to regular TB if no trajectory info provided
            return TrajectoryBalance().compute_loss(
                forward_flows, backward_flows, rewards, logZ, **kwargs
            )
        
        # Extract sub-trajectory information
        partial_forward_flows = trajectories_info['partial_forward_flows']  # [batch, max_len]
        partial_backward_flows = trajectories_info['partial_backward_flows']  # [batch, max_len]
        partial_rewards = trajectories_info['partial_rewards']  # [batch, max_len] - already in log space
        masks = trajectories_info['masks']  # [batch, max_len] validity mask
        
        batch_size, max_len = partial_forward_flows.shape
        
        losses = []
        weights = []
        
        for t in range(max_len):
            # Get flows up to time t
            fwd_t = partial_forward_flows[:, t]
            bwd_t = partial_backward_flows[:, t]
            log_reward_t = partial_rewards[:, t]  # Already in log space
            mask_t = masks[:, t]
            
            if mask_t.sum() == 0:
                continue
            
            # Compute sub-trajectory balance
            subtb_error = logZ + fwd_t - log_reward_t - bwd_t
            
            # Apply mask and weight
            weight = self.lambda_coef ** (max_len - t - 1)
            masked_error = subtb_error * mask_t
            
            losses.append((masked_error.pow(2) * mask_t).sum() / mask_t.sum())
            weights.append(weight)
        
        # Weighted average of losses
        if losses:
            losses = torch.stack(losses)
            weights = torch.tensor(weights, device=losses.device)
            loss = (losses * weights).sum() / weights.sum()
        else:
            loss = torch.tensor(0.0, device=forward_flows.device)
        
        # Return GPU tensors - caller will convert for logging/serialization
        metrics = {
            'subtb_num_terms': len(losses),
            'subtb_avg_length': (masks.sum() / batch_size).detach(),
        }
        
        return loss, metrics


class ForwardLookingObjective(GFlowNetObjective):
    """
    Forward-Looking (FL) objective - uses estimated future rewards.

    Useful when rewards are expensive to compute or for credit assignment.
    Note: The value network should predict log rewards if rewards are in log space.
    """
    
    def __init__(self, value_network: Optional[nn.Module] = None, 
                 value_weight: float = 0.1):
        """
        Args:
            value_network: Network to estimate state values V(s) (should predict log rewards)
            value_weight: Weight for value prediction loss
        """
        self.value_network = value_network
        self.value_weight = value_weight
    
    def compute_loss(self,
                    forward_flows: torch.Tensor,
                    backward_flows: torch.Tensor,
                    rewards: torch.Tensor,
                    logZ: torch.nn.Parameter,
                    states: Optional[torch.Tensor] = None,
                    **kwargs) -> Tuple[torch.Tensor, Dict[str, float]]:
        
        # Standard TB loss (rewards already in log space)
        tb_loss, tb_metrics = TrajectoryBalance().compute_loss(
            forward_flows, backward_flows, rewards, logZ, **kwargs
        )
        
        if self.value_network is None or states is None:
            return tb_loss, tb_metrics
        
        # Value prediction loss (predicting log rewards)
        predicted_log_values = self.value_network(states).squeeze()
        value_loss = F.mse_loss(predicted_log_values, rewards.detach())
        
        # Combined loss
        total_loss = tb_loss + self.value_weight * value_loss
        
        # Return GPU tensors - caller will convert for logging/serialization
        metrics = {
            **tb_metrics,
            'value_loss': value_loss.detach(),
            'value_mae': (predicted_log_values - rewards).abs().mean().detach(),
        }
        
        return total_loss, metrics


class EntropyRegularizedObjective(GFlowNetObjective):
    """
    Entropy-regularized objective to encourage exploration.

    Adds entropy bonus to the standard objective.
    """
    
    def __init__(self, base_objective: Optional[GFlowNetObjective] = None,
                 entropy_weight: float = 0.01):
        """
        Args:
            base_objective: Base objective to add entropy to
            entropy_weight: Weight for entropy regularization
        """
        self.base_objective = base_objective or TrajectoryBalance()
        self.entropy_weight = entropy_weight
    
    def compute_loss(self,
                    forward_flows: torch.Tensor,
                    backward_flows: torch.Tensor,
                    rewards: torch.Tensor,
                    logZ: torch.nn.Parameter,
                    action_logits: Optional[torch.Tensor] = None,
                    action_masks: Optional[torch.Tensor] = None,
                    **kwargs) -> Tuple[torch.Tensor, Dict[str, float]]:
        
        # Base objective loss (rewards already in log space)
        base_loss, base_metrics = self.base_objective.compute_loss(
            forward_flows, backward_flows, rewards, logZ, **kwargs
        )
        
        if action_logits is None:
            return base_loss, base_metrics
        
        # Compute entropy of action distributions
        if action_masks is not None:
            # Apply masks
            masked_logits = action_logits.clone()
            masked_logits[~action_masks] = float('-inf')
        else:
            masked_logits = action_logits
        
        # Compute entropy
        probs = F.softmax(masked_logits, dim=-1)
        log_probs = F.log_softmax(masked_logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1)
        
        # Filter out invalid entries
        valid_entropy = entropy[~entropy.isnan()]
        if len(valid_entropy) > 0:
            entropy_loss = -valid_entropy.mean()  # Negative because we want to maximize
        else:
            entropy_loss = torch.tensor(0.0, device=entropy.device)
        
        # Combined loss
        total_loss = base_loss + self.entropy_weight * entropy_loss
        
        # Return GPU tensors - caller will convert for logging/serialization
        metrics = {
            **base_metrics,
            'entropy': valid_entropy.mean().detach() if len(valid_entropy) > 0 else 0.0,
            'entropy_loss': entropy_loss.detach(),
        }
        
        return total_loss, metrics


class MultiObjective(GFlowNetObjective):
    """
    Combines multiple objectives with different weights.
    All objectives will receive rewards in log space.
    """
    
    def __init__(self, objectives: Dict[str, Tuple[GFlowNetObjective, float]]):
        """
        Args:
            objectives: Dict mapping names to (objective, weight) tuples
        """
        self.objectives = objectives
    
    def compute_loss(self,
                    forward_flows: torch.Tensor,
                    backward_flows: torch.Tensor,
                    rewards: torch.Tensor,
                    logZ: torch.nn.Parameter,
                    **kwargs) -> Tuple[torch.Tensor, Dict[str, float]]:
        
        total_loss = 0.0
        all_metrics = {}
        
        for name, (objective, weight) in self.objectives.items():
            loss, metrics = objective.compute_loss(
                forward_flows, backward_flows, rewards, logZ, **kwargs
            )
            
            total_loss = total_loss + weight * loss
            
            # Prefix metrics with objective name
            for k, v in metrics.items():
                all_metrics[f"{name}_{k}"] = v
            # Return GPU tensors - caller will convert for logging/serialization
            all_metrics[f"{name}_loss"] = loss.detach()
            all_metrics[f"{name}_weight"] = weight
        
        return total_loss, all_metrics


# Factory function to create objectives
def create_gfn_objective(objective_type: str, **kwargs) -> GFlowNetObjective:
    """
    Create a GFlowNet objective based on type string.

    Note: All objectives expect rewards to be in log space.

    Args:
        objective_type: One of 'tb', 'db', 'subtb', 'fl', 'entropy', 'multi'
        **kwargs: Arguments specific to each objective type

    Returns:
        GFlowNetObjective instance
    """
    if objective_type == 'tb' or objective_type == 'trajectory_balance':
        return TrajectoryBalance(**kwargs)
    elif objective_type == 'db' or objective_type == 'detailed_balance':
        return DetailedBalance(**kwargs)
    elif objective_type == 'subtb' or objective_type == 'subtrajectory_balance':
        return SubTrajectoryBalance(**kwargs)
    elif objective_type == 'fl' or objective_type == 'forward_looking':
        return ForwardLookingObjective(**kwargs)
    elif objective_type == 'entropy':
        base_obj = create_gfn_objective(kwargs.pop('base_objective', 'tb'), **kwargs)
        return EntropyRegularizedObjective(base_obj, **kwargs)
    elif objective_type == 'multi':
        objectives_config = kwargs.get('objectives', {})
        objectives = {}
        for name, config in objectives_config.items():
            obj_type = config['type']
            weight = config.get('weight', 1.0)
            obj_kwargs = config.get('kwargs', {})
            objectives[name] = (create_gfn_objective(obj_type, **obj_kwargs), weight)
        return MultiObjective(objectives)
    else:
        raise ValueError(f"Unknown objective type: {objective_type}")


# Example configurations
OBJECTIVE_CONFIGS = {
    'default': {
        'type': 'tb',
        'kwargs': {'loss_type': 'squared'}
    },
    
    'robust_tb': {
        'type': 'tb',
        'kwargs': {'loss_type': 'huber'}
    },
    
    'exploratory': {
        'type': 'entropy',
        'kwargs': {
            'base_objective': 'tb',
            'entropy_weight': 0.01
        }
    },
    
    'efficient_subtb': {
        'type': 'subtb',
        'kwargs': {'lambda_coef': 0.9}
    },
    
    'multi_balanced': {
        'type': 'multi',
        'kwargs': {
            'objectives': {
                'tb': {'type': 'tb', 'weight': 1.0},
                'entropy': {'type': 'entropy', 'weight': 0.01, 'kwargs': {'base_objective': 'tb'}}
            }
        }
    }
}
