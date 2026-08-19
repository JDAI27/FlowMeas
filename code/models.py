# -*- coding: utf-8 -*-
"""
Neural network models for GFlowNet policy learning.

The forward policy is an MLP over the flattened Clifford tableau; the backward
policy is a fixed uniform distribution over valid actions.

Input Representation:
    Clifford tableau W: (B, C, 2n, 2n) flattened to (B, C, 4n²)
    The W matrix encodes the current measurement basis via Heisenberg picture.

Output:
    Action logits: (B, C, n_actions) for policy distribution P_F(a|s)
"""
import torch
import torch.nn as nn
from typing import Tuple, Optional


class DiscreteUniform(nn.Module):
    """
    Implements a uniform distribution over discrete actions.

    It uses a zero function approximator (a function that always outputs 0) to be used as
    logits by a DiscretePBEstimator. Now properly handles batch inputs including measurements.

    Attributes:
        output_dim: The size of the output space.
    """
    
    def __init__(self, output_dim: int) -> None:
        """
        Initializes the uniform function approximator.

        Args:
            output_dim (int): Output dimension. This is typically n_actions if it
                implements a Uniform PF, or n_actions-1 if it implements a Uniform PB.
        """
        super().__init__()
        self.output_dim = output_dim
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass that properly handles batch inputs with optional masking.

        Args:
            x: Input tensor of shape [batch_size, input_dim], [batch_size, n_measurements, input_dim],
               or [input_dim]
            mask: Optional boolean mask for active elements

        Returns:
            Tensor of zeros with appropriate shape
        """
        if x.dim() == 1:
            # Single input
            return torch.zeros(self.output_dim, device=x.device)
        elif x.dim() == 2:
            # Batch input
            batch_size = x.shape[0]
            return torch.zeros(batch_size, self.output_dim, device=x.device)
        elif x.dim() == 3:
            # Batch with measurements
            batch_size, n_measurements = x.shape[:2]
            return torch.zeros(batch_size, n_measurements, self.output_dim, device=x.device)
        else:
            raise ValueError(f"Unsupported input dimension: {x.dim()}")


class CliffordMLP(nn.Module):
    """
    A PyTorch MLP model specifically designed for Clifford tableau inputs.
    Handles batch processing with measurements and active masking.

    Parameters:
        n_qubits (int): Number of qubits in the quantum system.
        hidden_dim (int): Number of neurons in each hidden layer.
        num_hidden_layers (int): Number of hidden layers in the network.
        output_dim (int): Size of the output layer (number of actions or predictions).
        use_layer_norm (bool): Whether to use layer normalization.
    """
    
    def __init__(self, n_qubits: int, hidden_dim: int, num_hidden_layers: int, 
                 output_dim: int, use_layer_norm: bool = True):
        super(CliffordMLP, self).__init__()
        
        self.n_qubits = n_qubits
        self.input_dim = (2 * n_qubits) ** 2  # Only W matrix (2nx2n Clifford tableau), no phase vector
        self.output_dim = output_dim
        
        layers = []
        
        # Input layer with optional input normalization
        layers.append(nn.Linear(self.input_dim, hidden_dim))
        if use_layer_norm:
            layers.append(nn.LayerNorm(hidden_dim))
        layers.append(nn.LeakyReLU())
        
        # Hidden layers
        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            if use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.LeakyReLU())
        
        # Output layer
        layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        self.logZ = nn.Parameter(torch.zeros(1))
        
        # Initialize weights
        self.init_weights()
    
    def forward(self, x: torch.Tensor, indices: Optional[torch.Tensor] = None,
                batch_shape: Optional[Tuple[int, int]] = None) -> torch.Tensor:
        """
        Forward pass of the MLP with support for active-only processing.

        Args:
            x: Input tensor of shape [n_active, input_dim] (from to_flat_tensors_active_only)
               or [batch_size * n_measurements, input_dim]
            indices: Optional tensor of shape [n_active, 2] with (batch_idx, meas_idx)
            batch_shape: Optional tuple (batch_size, n_measurements) for reconstruction

        Returns:
            If indices and batch_shape provided: [batch_size, n_measurements, output_dim]
            Otherwise: same batch dimensions as input with output_dim
        """
        if x.shape[0] == 0:  # No active tableaus
            if batch_shape is not None:
                return torch.zeros(*batch_shape, self.output_dim, device=x.device)
            else:
                return torch.zeros(0, self.output_dim, device=x.device)
        
        # Process through network
        output = self.network(x)
        
        # Reconstruct full batch shape if indices provided
        if indices is not None and batch_shape is not None:
            batch_size, n_measurements = batch_shape
            full_output = torch.zeros(batch_size, n_measurements, self.output_dim, 
                                    device=output.device)
            
            batch_indices = indices[:, 0]
            meas_indices = indices[:, 1]
            full_output[batch_indices, meas_indices] = output
            
            return full_output
        
        return output
    
    def init_weights(self):
        """Initializes the weights using Xavier (Glorot) initialization."""
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)


def create_clifford_model(model_type: str, n_qubits: int, hidden_dim: int,
                         num_hidden_layers: int, output_dim: int, **kwargs) -> nn.Module:
    """
    Create a model for Clifford tableau processing.

    Args:
        model_type: 'clifford_mlp' (forward policy) or 'uniform' (backward policy)
        n_qubits: Number of qubits in the quantum system
        hidden_dim: Hidden layer dimension
        num_hidden_layers: Number of hidden layers
        output_dim: Output dimension (number of actions)
        **kwargs: Additional model-specific parameters
            - use_layer_norm (bool): For CliffordMLP (default: True)

    Returns:
        The created model

    Examples:
        >>> model = create_clifford_model('clifford_mlp', n_qubits=10, hidden_dim=512,
        ...                              num_hidden_layers=3, output_dim=51)
    """
    if model_type == 'clifford_mlp':
        return CliffordMLP(n_qubits, hidden_dim, num_hidden_layers, output_dim,
                          use_layer_norm=kwargs.get('use_layer_norm', True))

    if model_type == 'uniform':
        return DiscreteUniform(output_dim)

    available_types = ['clifford_mlp', 'uniform']
    raise ValueError(f"Unknown model type: {model_type}. Available types: {available_types}")
