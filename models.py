# -*- coding: utf-8 -*-
# models.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Union


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
        self.input_dim = (2 * n_qubits) ** 2 + (2 * n_qubits)  # Clifford tableau flat size
        self.output_dim = output_dim
        
        layers = []
        
        # Input layer with optional input normalization
        layers.append(nn.Linear(self.input_dim, hidden_dim))
        if use_layer_norm:
            layers.append(nn.LayerNorm(hidden_dim))
        layers.append(nn.LeakyReLU())
        
        # Hidden layers
        for _ in range(num_hidden_layers - 1):
            #add dropout
            #layers.append(nn.Dropout(0.3))
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


class QuantumAwareMLP(nn.Module):
    """
    MLP with quantum-specific features for Clifford tableau processing.
    Includes separate processing for stabilizer and phase information.
    """
    
    def __init__(self, n_qubits: int, hidden_dim: int, num_hidden_layers: int,
                 output_dim: int, separate_phase_processing: bool = True):
        super(QuantumAwareMLP, self).__init__()
        
        self.n_qubits = n_qubits
        self.n2 = 2 * n_qubits
        self.matrix_size = self.n2 ** 2
        self.phase_size = self.n2
        self.input_dim = self.matrix_size + self.phase_size
        self.separate_phase = separate_phase_processing
        
        if separate_phase_processing:
            # Separate encoders for matrix and phase
            self.matrix_encoder = nn.Sequential(
                nn.Linear(self.matrix_size, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.LeakyReLU()
            )
            
            self.phase_encoder = nn.Sequential(
                nn.Linear(self.phase_size, hidden_dim // 4),
                nn.LayerNorm(hidden_dim // 4),
                nn.LeakyReLU()
            )
            
            # Combined processing
            combined_dim = hidden_dim + hidden_dim // 4
            layers = []
            layers.append(nn.Linear(combined_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.LeakyReLU())
            
        else:
            # Standard processing
            layers = []
            layers.append(nn.Linear(self.input_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.LeakyReLU())
        
        # Hidden layers
        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.LeakyReLU())
        
        # Output layer
        layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        self.logZ = nn.Parameter(torch.zeros(1))
        
        self.init_weights()
    
    def forward(self, x: torch.Tensor, indices: Optional[torch.Tensor] = None,
                batch_shape: Optional[Tuple[int, int]] = None) -> torch.Tensor:
        """Forward pass with quantum-aware processing."""
        if x.shape[0] == 0:  # No active tableaus
            if batch_shape is not None:
                return torch.zeros(*batch_shape, self.network[-1].out_features, device=x.device)
            else:
                return torch.zeros(0, self.network[-1].out_features, device=x.device)
        
        if self.separate_phase:
            # Split input into matrix and phase parts
            matrix_part = x[:, :self.matrix_size]
            phase_part = x[:, self.matrix_size:]
            
            # Encode separately
            matrix_features = self.matrix_encoder(matrix_part)
            phase_features = self.phase_encoder(phase_part)
            
            # Combine
            combined = torch.cat([matrix_features, phase_features], dim=1)
            output = self.network(combined)
        else:
            output = self.network(x)
        
        # Reconstruct full batch shape if needed
        if indices is not None and batch_shape is not None:
            batch_size, n_measurements = batch_shape
            full_output = torch.zeros(batch_size, n_measurements, output.shape[-1], 
                                    device=output.device)
            
            batch_indices = indices[:, 0]
            meas_indices = indices[:, 1]
            full_output[batch_indices, meas_indices] = output
            
            return full_output
        
        return output
    
    def init_weights(self):
        """Initialize all linear layers."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)


class AttentionMLP(nn.Module):
    """
    MLP with self-attention for capturing qubit correlations in Clifford tableaus.
    """
    
    def __init__(self, n_qubits: int, hidden_dim: int, num_hidden_layers: int,
                 output_dim: int, num_heads: int = 4):
        super(AttentionMLP, self).__init__()
        
        self.n_qubits = n_qubits
        self.input_dim = (2 * n_qubits) ** 2 + (2 * n_qubits)
        
        # Input projection
        self.input_proj = nn.Linear(self.input_dim, hidden_dim)
        
        # Self-attention layer
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.attn_norm = nn.LayerNorm(hidden_dim)
        
        # MLP layers
        layers = []
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.LeakyReLU())
        
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.network = nn.Sequential(*layers)
        
        self.logZ = nn.Parameter(torch.zeros(1))
        self.init_weights()
    
    def forward(self, x: torch.Tensor, indices: Optional[torch.Tensor] = None,
                batch_shape: Optional[Tuple[int, int]] = None) -> torch.Tensor:
        """Forward pass with attention mechanism."""
        if x.shape[0] == 0:
            if batch_shape is not None:
                return torch.zeros(*batch_shape, self.network[-1].out_features, device=x.device)
            else:
                return torch.zeros(0, self.network[-1].out_features, device=x.device)
        
        # Project input
        h = self.input_proj(x)
        h = F.leaky_relu(h)
        
        # Add sequence dimension for attention
        h_seq = h.unsqueeze(1)  # [batch, 1, hidden_dim]
        
        # Self-attention
        attn_out, _ = self.attention(h_seq, h_seq, h_seq)
        h = self.attn_norm(h + attn_out.squeeze(1))
        
        # MLP processing
        output = self.network(h)
        
        # Reconstruct if needed
        if indices is not None and batch_shape is not None:
            batch_size, n_measurements = batch_shape
            full_output = torch.zeros(batch_size, n_measurements, output.shape[-1], 
                                    device=output.device)
            
            batch_indices = indices[:, 0]
            meas_indices = indices[:, 1]
            full_output[batch_indices, meas_indices] = output
            
            return full_output
        
        return output
    
    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)


# Utility function to create appropriate model based on config
def create_clifford_model(model_type: str, n_qubits: int, hidden_dim: int, 
                         num_hidden_layers: int, output_dim: int, **kwargs) -> nn.Module:
    """
    Create a model for Clifford tableau processing.
    
    Args:
        model_type: One of 'clifford_mlp', 'quantum_aware', 'attention'
        n_qubits: Number of qubits in the quantum system
        hidden_dim: Hidden layer dimension
        num_hidden_layers: Number of hidden layers
        output_dim: Output dimension (number of actions)
        **kwargs: Additional model-specific parameters
        
    Returns:
        The created model
    """
    if model_type == 'clifford_mlp':
        return CliffordMLP(n_qubits, hidden_dim, num_hidden_layers, output_dim,
                            use_layer_norm=kwargs.get('use_layer_norm', True))
    elif model_type == 'quantum_aware':
        return QuantumAwareMLP(n_qubits, hidden_dim, num_hidden_layers, output_dim,
                                separate_phase_processing=kwargs.get('separate_phase_processing', True))
    elif model_type == 'attention':
        return AttentionMLP(n_qubits, hidden_dim, num_hidden_layers, output_dim,
                            num_heads=kwargs.get('num_heads', 4))
    elif model_type == 'uniform':
        return DiscreteUniform(output_dim)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


# Wrapper for efficient batch processing with BatchedCliffordTableau
class CliffordTableauProcessor:
    """Helper class to process BatchedCliffordTableau through models efficiently."""
    
    def __init__(self, model: nn.Module):
        self.model = model
    
    def process_batch(self, batch_tableau, return_full: bool = True) -> torch.Tensor:
        """
        Process a BatchedCliffordTableau through the model.
        
        Args:
            batch_tableau: BatchedCliffordTableau instance
            return_full: If True, returns full [batch_size, n_measurements, output_dim]
                        If False, returns only active tableaus
        
        Returns:
            Model output with appropriate shape
        """
        # Get active-only flat tensors
        flat_tensors, indices = batch_tableau.to_flat_tensors_active_only()
        
        if return_full:
            # Process and reconstruct
            batch_shape = (batch_tableau.batch_size, batch_tableau.n_measurements)
            return self.model(flat_tensors, indices=indices, batch_shape=batch_shape)
        else:
            # Return only active outputs
            return self.model(flat_tensors)
