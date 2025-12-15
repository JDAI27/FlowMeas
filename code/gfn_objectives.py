"""
Modular GFlowNet Training Objectives

This module provides different training objectives for GFlowNets that can be easily swapped.

IMPORTANT: All rewards passed to these objectives are expected to be in log space (log rewards).
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple, Callable
import torch.nn.functional as F


class GFlowNetObjective(ABC):
    """Abstract base class for GFlowNet training objectives"""
    
    @abstractmethod
    def compute_loss(self, 
                    forward_flows: torch.Tensor,
                    backward_flows: torch.Tensor,
                    rewards: torch.Tensor,
                    logZ: torch.nn.Parameter,
                    **kwargs) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute loss for given flows and log rewards. Returns (loss, metrics)."""
        pass


class TrajectoryBalance(GFlowNetObjective):
    """TB objective: Loss = E[(log Z + log P_F(τ) - log R(τ) - log P_B(τ))²]"""
    
    def __init__(self, loss_type: str = "squared"):
        self.loss_type = loss_type
    
    def compute_loss(self, 
                    forward_flows: torch.Tensor,
                    backward_flows: torch.Tensor,
                    rewards: torch.Tensor,
                    logZ: torch.nn.Parameter,
                    **kwargs) -> Tuple[torch.Tensor, Dict[str, float]]:
        log_rewards = rewards
        tb_error = logZ + forward_flows - log_rewards - backward_flows
        
        if torch.isnan(tb_error).any() or torch.isinf(tb_error).any():
            import logging
            logging.warning(f"NaN/Inf in TB error! Components:")
            logging.warning(f"  logZ: {logZ.item()}")
            logging.warning(f"  forward_flows: min={forward_flows.min().item():.4f}, max={forward_flows.max().item():.4f}, has_nan={torch.isnan(forward_flows).any()}")
            logging.warning(f"  log_rewards: min={log_rewards.min().item():.4f}, max={log_rewards.max().item():.4f}, has_nan={torch.isnan(log_rewards).any()}")
            logging.warning(f"  backward_flows: min={backward_flows.min().item():.4f}, max={backward_flows.max().item():.4f}, has_nan={torch.isnan(backward_flows).any()}")
        
        if self.loss_type == "squared":
            loss = tb_error.pow(2).mean()
        elif self.loss_type == "abs":
            loss = tb_error.abs().mean()
        elif self.loss_type == "huber":
            loss = F.huber_loss(tb_error, torch.zeros_like(tb_error), reduction='mean')
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
        
        metrics = {
            'tb_error_mean': tb_error.mean().detach(),
            'tb_error_std': tb_error.std().detach(),
            'tb_error_abs_mean': tb_error.abs().mean().detach(),
        }
        
        return loss, metrics


class DetailedBalance(GFlowNetObjective):
    """DB objective - enforces balance at each state transition."""
    
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
        
        state_pairs = states_info['state_pairs']
        forward_probs = states_info['forward_probs']
        backward_probs = states_info['backward_probs']
        state_flows = states_info['state_flows']
        
        losses = []
        
        for i, (s, s_prime) in enumerate(state_pairs):
            if s is None:  # Initial state
                flow_s = logZ
            else:
                flow_s = state_flows[s]
            
            flow_s_prime = state_flows[s_prime]
            
            lhs = flow_s + forward_probs[i]
            rhs = flow_s_prime + backward_probs[i]
            
            db_error = lhs - rhs
            losses.append(db_error.pow(2))
        
        loss = torch.stack(losses).mean()
        
        metrics = {
            'db_error_mean': torch.stack(losses).sqrt().mean().detach(),
            'num_transitions': len(state_pairs),
        }
        
        return loss, metrics


class SubTrajectoryBalance(GFlowNetObjective):
    """SubTB - enforces balance for all sub-trajectories (more sample-efficient)."""
    
    def __init__(self, lambda_coef: float = 0.9):
        self.lambda_coef = lambda_coef
    
    def compute_loss(self,
                    forward_flows: torch.Tensor,
                    backward_flows: torch.Tensor,
                    rewards: torch.Tensor,
                    logZ: torch.nn.Parameter,
                    trajectories_info: Optional[Dict] = None,
                    **kwargs) -> Tuple[torch.Tensor, Dict[str, float]]:
        if trajectories_info is None:
            return TrajectoryBalance().compute_loss(
                forward_flows, backward_flows, rewards, logZ, **kwargs
            )
        
        partial_forward_flows = trajectories_info['partial_forward_flows']
        partial_backward_flows = trajectories_info['partial_backward_flows']
        partial_rewards = trajectories_info['partial_rewards']
        masks = trajectories_info['masks']
        
        batch_size, max_len = partial_forward_flows.shape
        
        losses = []
        weights = []
        
        for t in range(max_len):
            fwd_t = partial_forward_flows[:, t]
            bwd_t = partial_backward_flows[:, t]
            log_reward_t = partial_rewards[:, t]
            mask_t = masks[:, t]
            
            if mask_t.sum() == 0:
                continue
            
            subtb_error = logZ + fwd_t - log_reward_t - bwd_t
            weight = self.lambda_coef ** (max_len - t - 1)
            masked_error = subtb_error * mask_t
            
            losses.append((masked_error.pow(2) * mask_t).sum() / mask_t.sum())
            weights.append(weight)
        
        if losses:
            losses = torch.stack(losses)
            weights = torch.tensor(weights, device=losses.device)
            loss = (losses * weights).sum() / weights.sum()
        else:
            loss = torch.tensor(0.0, device=forward_flows.device)
        
        metrics = {
            'subtb_num_terms': len(losses),
            'subtb_avg_length': (masks.sum() / batch_size).detach(),
        }
        
        return loss, metrics


class ForwardLookingObjective(GFlowNetObjective):
    """FL objective - uses estimated future rewards for credit assignment."""
    
    def __init__(self, value_network: Optional[nn.Module] = None, 
                 value_weight: float = 0.1):
        self.value_network = value_network
        self.value_weight = value_weight
    
    def compute_loss(self,
                    forward_flows: torch.Tensor,
                    backward_flows: torch.Tensor,
                    rewards: torch.Tensor,
                    logZ: torch.nn.Parameter,
                    states: Optional[torch.Tensor] = None,
                    **kwargs) -> Tuple[torch.Tensor, Dict[str, float]]:
        tb_loss, tb_metrics = TrajectoryBalance().compute_loss(
            forward_flows, backward_flows, rewards, logZ, **kwargs
        )
        
        if self.value_network is None or states is None:
            return tb_loss, tb_metrics
        
        predicted_log_values = self.value_network(states).squeeze()
        value_loss = F.mse_loss(predicted_log_values, rewards.detach())
        total_loss = tb_loss + self.value_weight * value_loss
        
        metrics = {
            **tb_metrics,
            'value_loss': value_loss.detach(),
            'value_mae': (predicted_log_values - rewards).abs().mean().detach(),
        }
        
        return total_loss, metrics


class EntropyRegularizedObjective(GFlowNetObjective):
    """Entropy-regularized objective to encourage exploration."""
    
    def __init__(self, base_objective: Optional[GFlowNetObjective] = None,
                 entropy_weight: float = 0.01):
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
        base_loss, base_metrics = self.base_objective.compute_loss(
            forward_flows, backward_flows, rewards, logZ, **kwargs
        )
        
        if action_logits is None:
            return base_loss, base_metrics
        
        if action_masks is not None:
            masked_logits = action_logits.clone()
            masked_logits[~action_masks] = float('-inf')
        else:
            masked_logits = action_logits
        
        probs = F.softmax(masked_logits, dim=-1)
        log_probs = F.log_softmax(masked_logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1)
        
        valid_entropy = entropy[~entropy.isnan()]
        if len(valid_entropy) > 0:
            entropy_loss = -valid_entropy.mean()
        else:
            entropy_loss = torch.tensor(0.0, device=entropy.device)
        
        total_loss = base_loss + self.entropy_weight * entropy_loss
        
        metrics = {
            **base_metrics,
            'entropy': valid_entropy.mean().detach() if len(valid_entropy) > 0 else 0.0,
            'entropy_loss': entropy_loss.detach(),
        }
        
        return total_loss, metrics


class MultiObjective(GFlowNetObjective):
    """Combines multiple objectives with different weights."""
    
    def __init__(self, objectives: Dict[str, Tuple[GFlowNetObjective, float]]):
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
            
            for k, v in metrics.items():
                all_metrics[f"{name}_{k}"] = v
            all_metrics[f"{name}_loss"] = loss.detach()
            all_metrics[f"{name}_weight"] = weight
        
        return total_loss, all_metrics


def create_gfn_objective(objective_type: str, **kwargs) -> GFlowNetObjective:
    """Factory function to create GFlowNet objectives. Rewards expected in log space."""
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
