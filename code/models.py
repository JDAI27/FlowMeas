# -*- coding: utf-8 -*-
# models.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Union


class DiscreteUniform(nn.Module):
    """Uniform distribution over discrete actions (outputs zeros for logits)."""

    def __init__(self, output_dim: int) -> None:
        super().__init__()
        self.output_dim = output_dim

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Returns zeros with shape matching input batch dimensions."""
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

from typing import Optional, Tuple
import torch
import torch.nn as nn

class CliffordDeepSets(nn.Module):
    """
    Deep Sets over n qubit pairs (permutation-invariant to qubit relabeling).
    Elements: u_i = [W[:, X_i], W[:, Z_i]] per qubit. Drop-in replacement for CliffordMLP.
    """
    def __init__(self, n_qubits: int, hidden_dim: int, num_hidden_layers: int,
                 output_dim: int, use_layer_norm: bool = True):
        super().__init__()

        self.n_qubits = n_qubits
        self.M = 2 * n_qubits                     # rows == cols == 2n
        self.input_dim = self.M * self.M          # Only W matrix (2nx2n Clifford tableau), no phase vector
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.use_layer_norm = use_layer_norm

        width = max(1, hidden_dim // n_qubits)
        elem_dim = 4 * n_qubits

        phi_layers = [nn.Linear(elem_dim, width)]
        if use_layer_norm:
            phi_layers.append(nn.LayerNorm(width))
        phi_layers.append(nn.LeakyReLU())
        phi_layers += [nn.Linear(width, width)]
        if use_layer_norm:
            phi_layers.append(nn.LayerNorm(width))
        phi_layers.append(nn.LeakyReLU())
        self.phi = nn.Sequential(*phi_layers)

        rho_layers = []
        rho_layers.append(nn.Linear(n_qubits * width, hidden_dim))
        if use_layer_norm:
            rho_layers.append(nn.LayerNorm(hidden_dim))
        rho_layers.append(nn.LeakyReLU())
        for _ in range(max(0, num_hidden_layers - 1)):
            rho_layers.append(nn.Linear(hidden_dim, hidden_dim))
            if use_layer_norm:
                rho_layers.append(nn.LayerNorm(hidden_dim))
            rho_layers.append(nn.LeakyReLU())
        rho_layers.append(nn.Linear(hidden_dim, output_dim))
        self.rho = nn.Sequential(*rho_layers)

        self.logZ = nn.Parameter(torch.zeros(1))

        self.init_weights()

    def _pairs(self, x: torch.Tensor) -> torch.Tensor:
        """Extract per-qubit elements: [B, n, 4n] with u_i = [W[:, X_i], W[:, Z_i]]."""
        B = x.shape[0]
        n = self.n_qubits
        M = self.M

        W = x.view(B, M, M)
        cols = W.transpose(1, 2)
        colX = cols[:, :n, :]
        colZ = cols[:, n:, :]
        return torch.cat([colX.float(), colZ.float()], dim=-1)

    def forward(self, x: torch.Tensor,
                indices: Optional[torch.Tensor] = None,
                batch_shape: Optional[Tuple[int, int]] = None) -> torch.Tensor:
        """Forward pass with optional reconstruction to full batch shape."""
        if x.shape[0] == 0:
            if batch_shape is not None:
                return torch.zeros(*batch_shape, self.output_dim, device=x.device)
            return torch.zeros(0, self.output_dim, device=x.device)

        if x.shape[-1] != self.input_dim:
            raise ValueError(f"Expected input_dim={self.input_dim}, got {x.shape[-1]}")

        B = x.shape[0]
        elems = self._pairs(x)
        n = self.n_qubits
        width = max(1, self.hidden_dim // 8)

        h = self.phi(elems.reshape(B * n, -1)).view(B, n, width)
        s = h.flatten(start_dim=1)
        out = self.rho(s)

        if indices is not None and batch_shape is not None:
            batch_size, n_measurements = batch_shape
            full_output = torch.zeros(batch_size, n_measurements, self.output_dim,
                                      device=out.device)
            full_output[indices[:, 0], indices[:, 1]] = out
            return full_output
        return out

    def init_weights(self):
        for module in list(self.phi) + list(self.rho):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)


class CliffordMLP(nn.Module):
    """MLP for Clifford tableau inputs with batch processing support."""

    def __init__(self, n_qubits: int, hidden_dim: int, num_hidden_layers: int,
                 output_dim: int, use_layer_norm: bool = True):
        super(CliffordMLP, self).__init__()

        self.n_qubits = n_qubits
        self.input_dim = (2 * n_qubits) ** 2  # Only W matrix (2nx2n Clifford tableau), no phase vector
        self.output_dim = output_dim

        layers = []
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
        self.init_weights()

    def forward(self, x: torch.Tensor, indices: Optional[torch.Tensor] = None,
                batch_shape: Optional[Tuple[int, int]] = None) -> torch.Tensor:
        """Forward pass with optional reconstruction to full batch shape."""
        if x.shape[0] == 0:
            if batch_shape is not None:
                return torch.zeros(*batch_shape, self.output_dim, device=x.device)
            else:
                return torch.zeros(0, self.output_dim, device=x.device)

        output = self.network(x)

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
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)


class QuantumAwareMLP(nn.Module):
    """MLP with quantum-specific features for Clifford tableau processing."""

    def __init__(self, n_qubits: int, hidden_dim: int, num_hidden_layers: int,
                 output_dim: int, separate_phase_processing: bool = True):
        super(QuantumAwareMLP, self).__init__()

        self.n_qubits = n_qubits
        self.n2 = 2 * n_qubits
        self.matrix_size = self.n2 ** 2
        self.input_dim = self.matrix_size

        layers = []
        layers.append(nn.Linear(self.input_dim, hidden_dim))
        layers.append(nn.LayerNorm(hidden_dim))
        layers.append(nn.LeakyReLU())

        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.LeakyReLU())

        layers.append(nn.Linear(hidden_dim, output_dim))

        self.network = nn.Sequential(*layers)
        self.logZ = nn.Parameter(torch.zeros(1))

        self.init_weights()

    def forward(self, x: torch.Tensor, indices: Optional[torch.Tensor] = None,
                batch_shape: Optional[Tuple[int, int]] = None) -> torch.Tensor:
        """Forward pass with optional reconstruction to full batch shape."""
        if x.shape[0] == 0:
            if batch_shape is not None:
                return torch.zeros(*batch_shape, self.network[-1].out_features, device=x.device)
            else:
                return torch.zeros(0, self.network[-1].out_features, device=x.device)

        output = self.network(x)

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


class AttentionMLP(nn.Module):
    """
    MLP with one self‑attention block that first converts the flattened
    Clifford tableau into per‑generator (token‑wise) embeddings:
        • sequence length  = 2 n       (X₁ … Xₙ Z₁ … Zₙ)
        • token dimension  = 2 n       (2 n symplectic bits, no phase)

    The attention layer captures correlations between different generators.
    After attention we average‑pool the tokens to obtain a single vector
    fed into a standard position‑wise MLP.
    Uses only W matrix (2nx2n Clifford tableau), no phase vector.
    """

    def __init__(self,
                 n_qubits: int,
                 hidden_dim: int,
                 num_hidden_layers: int,
                 output_dim: int,
                 num_heads: int = 4,
                 dropout: float = 0.0):
        super().__init__()

        self.n_qubits = n_qubits
        self.seq_len = 2 * n_qubits
        self.token_dim = self.seq_len
        self.flat_dim = self.seq_len * self.seq_len

        self.token_proj = nn.Linear(self.token_dim, hidden_dim)

        self.attn = nn.MultiheadAttention(hidden_dim,
                                          num_heads,
                                          dropout=dropout,
                                          batch_first=True)
        self.attn_norm = nn.LayerNorm(hidden_dim)

        layers = []
        for _ in range(1):
            layers.extend([
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.LayerNorm(hidden_dim // 2),
                nn.LeakyReLU()
            ])
        layers.append(nn.Linear(hidden_dim // 2, output_dim))
        self.ff = nn.Sequential(*layers)

        self.logZ = nn.Parameter(torch.zeros(1))

        self._init_weights()

    def forward(self,
                x: torch.Tensor,
                indices: Optional[torch.Tensor] = None,
                batch_shape: Optional[Tuple[int, int]] = None) -> torch.Tensor:
        """Forward pass with optional reconstruction to full batch shape."""
        if x.numel() == 0:
            out_shape = (*batch_shape, self.ff[-1].out_features) if batch_shape \
                        else (0, self.ff[-1].out_features)
            return torch.zeros(*out_shape, device=x.device, dtype=x.dtype)

        B = x.size(0)
        n2 = self.seq_len       # 2 n

        W_flat = x.view(B, n2, n2)
        tokens = W_flat.transpose(1, 2)


        h = self.token_proj(tokens)
        h = F.leaky_relu(h)
        attn_out, _ = self.attn(h, h, h)
        h = self.attn_norm(h + attn_out)
        h_pooled = h.mean(dim=1)
        out = self.ff(h_pooled)

        if indices is not None and batch_shape is not None:
            batch_size, n_meas = batch_shape
            full = torch.zeros(batch_size, n_meas, out.size(-1),
                               device=out.device, dtype=out.dtype)
            full[indices[:, 0], indices[:, 1]] = out
            return full

        return out

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


class LocalQuantumAttention(nn.Module):
    """Memory-efficient attention exploiting quantum circuit locality."""

    def __init__(self, hidden_dim: int, n_qubits: int, window_size: int = 3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.window_size = window_size

        self.qk_proj = nn.Linear(hidden_dim, hidden_dim // 2)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.generator_importance = nn.Parameter(torch.ones(2))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """Apply local attention based on quantum structure."""
        B, seq_len, _ = h.shape
        n = self.n_qubits

        h_x = h[:, :n] * self.generator_importance[0]
        h_z = h[:, n:] * self.generator_importance[1]

        h_x_att = self._local_attention(h_x)
        h_z_att = self._local_attention(h_z)
        h_cross = self._cross_attention(h_x, h_z)

        h_combined = torch.cat([h_x_att + h_cross[:, :n],
                               h_z_att + h_cross[:, n:]], dim=1)

        return self.out_proj(h_combined)

    def _local_attention(self, h: torch.Tensor) -> torch.Tensor:
        """Apply attention only to nearby qubits (local window)."""
        B, n, d = h.shape

        qk = self.qk_proj(h)
        v = self.v_proj(h)
        scores = torch.zeros(B, n, n, device=h.device)

        for i in range(n):
            start = max(0, i - self.window_size)
            end = min(n, i + self.window_size + 1)
            q_i = qk[:, i:i+1]
            k_local = qk[:, start:end]
            local_scores = torch.matmul(q_i, k_local.transpose(-2, -1)) / torch.sqrt(qk.size(-1))
            scores[:, i, start:end] = local_scores.squeeze(1)

        attn_weights = F.softmax(scores, dim=-1)
        return torch.matmul(attn_weights, v)

    def _cross_attention(self, h_x: torch.Tensor, h_z: torch.Tensor) -> torch.Tensor:
        """Cross-attention between X and Z generators of the same qubit."""
        B, n, d = h_x.shape

        qk_x = self.qk_proj(h_x)
        qk_z = self.qk_proj(h_z)
        v_x = self.v_proj(h_x)
        v_z = self.v_proj(h_z)

        scores_x_to_z = (qk_x * qk_z).sum(dim=-1, keepdim=True) / torch.sqrt(qk_x.size(-1))
        attn_x_to_z = torch.sigmoid(scores_x_to_z)
        out_x = h_x + attn_x_to_z * v_z

        scores_z_to_x = (qk_z * qk_x).sum(dim=-1, keepdim=True) / torch.sqrt(qk_z.size(-1))
        attn_z_to_x = torch.sigmoid(scores_z_to_x)
        out_z = h_z + attn_z_to_x * v_x

        return torch.cat([out_x, out_z], dim=1)


class MemoryEfficientQuantumMLP(nn.Module):
    """Memory-efficient model combining local attention with compressed representations."""

    def __init__(self,
                 n_qubits: int,
                 hidden_dim: int,
                 num_hidden_layers: int,
                 output_dim: int,
                 compression_ratio: int = 4,
                 window_size: int = 3):
        super().__init__()

        self.n_qubits = n_qubits
        self.seq_len = 2 * n_qubits
        self.token_dim = self.seq_len
        self.flat_dim = self.seq_len * self.seq_len

        self.compressed_dim = hidden_dim // compression_ratio

        self.token_proj = nn.Sequential(
            nn.Linear(self.token_dim, self.compressed_dim),
            nn.LayerNorm(self.compressed_dim),
            nn.GELU()
        )

        self.local_attn = LocalQuantumAttention(self.compressed_dim, n_qubits, window_size)
        self.pool_proj = nn.Linear(self.compressed_dim, hidden_dim)

        mlp_layers = []
        for i in range(num_hidden_layers):
            if i == 0:
                mlp_layers.extend([
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.LayerNorm(hidden_dim // 2),
                    nn.GELU()
                ])
            else:
                mlp_layers.extend([
                    nn.Linear(hidden_dim // 2, hidden_dim // 2),
                    nn.LayerNorm(hidden_dim // 2),
                    nn.GELU()
                ])

        mlp_layers.append(nn.Linear(hidden_dim // 2, output_dim))
        self.mlp = nn.Sequential(*mlp_layers)

        self.logZ = nn.Parameter(torch.zeros(1))
        self._init_weights()

    def forward(self,
                x: torch.Tensor,
                indices: Optional[torch.Tensor] = None,
                batch_shape: Optional[Tuple[int, int]] = None) -> torch.Tensor:
        """Forward pass with optional reconstruction to full batch shape."""
        if x.numel() == 0:
            out_shape = (*batch_shape, self.mlp[-1].out_features) if batch_shape \
                        else (0, self.mlp[-1].out_features)
            return torch.zeros(*out_shape, device=x.device, dtype=x.dtype)

        B = x.size(0)
        n2 = self.seq_len

        W_cols = x.view(B, n2, n2)
        tokens = W_cols.transpose(1, 2)

        h = self.token_proj(tokens)
        h = self.local_attn(h)

        h_x = h[:, :self.n_qubits].mean(dim=1)
        h_z = h[:, self.n_qubits:].mean(dim=1)
        h_pooled = self.pool_proj(h_x + h_z)
        out = self.mlp(h_pooled)

        if indices is not None and batch_shape is not None:
            batch_size, n_meas = batch_shape
            full = torch.zeros(batch_size, n_meas, out.size(-1),
                              device=out.device, dtype=out.dtype)
            full[indices[:, 0], indices[:, 1]] = out
            return full

        return out

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


class HybridQuantumMLP(nn.Module):
    """Hybrid model routing to MLP or attention based on circuit complexity."""

    def __init__(self,
                 n_qubits: int,
                 hidden_dim: int,
                 num_hidden_layers: int,
                 output_dim: int,
                 depth_threshold: int = 10):
        super().__init__()

        self.n_qubits = n_qubits
        self.depth_threshold = depth_threshold
        self.flat_dim = (2 * n_qubits) ** 2

        self.shallow_net = CliffordMLP(n_qubits, hidden_dim, num_hidden_layers, output_dim)
        self.deep_net = MemoryEfficientQuantumMLP(
            n_qubits, hidden_dim, num_hidden_layers, output_dim,
            compression_ratio=4, window_size=3
        )

        self.depth_estimator = nn.Sequential(
            nn.Linear(self.flat_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        self.logZ = nn.Parameter(torch.zeros(1))

    def forward(self,
                x: torch.Tensor,
                indices: Optional[torch.Tensor] = None,
                batch_shape: Optional[Tuple[int, int]] = None) -> torch.Tensor:
        """Route to appropriate processor based on estimated circuit depth."""
        if x.numel() == 0:
            return torch.zeros(0, self.shallow_net.output_dim, device=x.device)

        depth_scores = self.depth_estimator(x).squeeze(-1)
        shallow_mask = depth_scores < self.depth_threshold
        deep_mask = ~shallow_mask

        out = torch.zeros(x.size(0), self.shallow_net.output_dim, device=x.device)

        if shallow_mask.any():
            shallow_out = self.shallow_net(x[shallow_mask])
            out[shallow_mask] = shallow_out

        if deep_mask.any():
            deep_out = self.deep_net(x[deep_mask])
            out[deep_mask] = deep_out

        if indices is not None and batch_shape is not None:
            batch_size, n_meas = batch_shape
            full = torch.zeros(batch_size, n_meas, out.size(-1),
                              device=out.device, dtype=out.dtype)
            full[indices[:, 0], indices[:, 1]] = out
            return full

        return out


def create_clifford_model(model_type: str, n_qubits: int, hidden_dim: int,
                         num_hidden_layers: int, output_dim: int, **kwargs) -> nn.Module:
    """Factory function to create Clifford tableau models by type."""
    if model_type == 'clifford_mlp':
        return CliffordMLP(n_qubits, hidden_dim, num_hidden_layers, output_dim,
                          use_layer_norm=kwargs.get('use_layer_norm', True))

    if model_type == 'clifford_deepsets':
        return CliffordDeepSets(n_qubits, hidden_dim, num_hidden_layers, output_dim,
                               use_layer_norm=kwargs.get('use_layer_norm', True))

    elif model_type == 'quantum_aware':
        return QuantumAwareMLP(n_qubits, hidden_dim, num_hidden_layers, output_dim,
                              separate_phase_processing=kwargs.get('separate_phase_processing', True))

    elif model_type == 'attention':
        return AttentionMLP(n_qubits, hidden_dim, num_hidden_layers, output_dim,
                           num_heads=kwargs.get('num_heads', 4),
                           dropout=kwargs.get('dropout', 0.0))

    elif model_type == 'memory_efficient':
        return MemoryEfficientQuantumMLP(n_qubits, hidden_dim, num_hidden_layers, output_dim,
                                        compression_ratio=kwargs.get('compression_ratio', 4),
                                        window_size=kwargs.get('window_size', 3))

    elif model_type == 'hybrid':
        return HybridQuantumMLP(n_qubits, hidden_dim, num_hidden_layers, output_dim,
                               depth_threshold=kwargs.get('depth_threshold', 10))

    elif model_type == 'uniform':
        return DiscreteUniform(output_dim)

    else:
        available_types = ['clifford_mlp', 'quantum_aware', 'attention',
                          'memory_efficient', 'hybrid', 'uniform']
        raise ValueError(f"Unknown model type: {model_type}. Available types: {available_types}")


class CliffordTableauProcessor:
    """Helper class to process BatchedCliffordTableau through models efficiently."""

    def __init__(self, model: nn.Module, device: Optional[torch.device] = None):
        self.model = model
        self.device = device or next(model.parameters()).device
        self.model_type = self._detect_model_type()

    def _detect_model_type(self) -> str:
        model_class = type(self.model).__name__
        type_map = {
            'DiscreteUniform': 'uniform',
            'CliffordMLP': 'clifford_mlp',
            'QuantumAwareMLP': 'quantum_aware',
            'AttentionMLP': 'attention',
            'MemoryEfficientQuantumMLP': 'memory_efficient',
            'HybridQuantumMLP': 'hybrid'
        }
        return type_map.get(model_class, 'unknown')

    def process_batch(self, batch_tableau, return_full: bool = True) -> torch.Tensor:
        """Process BatchedCliffordTableau through the model."""
        flat_tensors, indices = batch_tableau.to_flat_tensors_active_only()

        flat_tensors = flat_tensors.to(self.device)
        if indices is not None:
            indices = indices.to(self.device)

        if return_full:
            batch_shape = (batch_tableau.batch_size, batch_tableau.n_measurements)
            return self.model(flat_tensors, indices=indices, batch_shape=batch_shape)
        else:
            return self.model(flat_tensors)

    def get_memory_usage(self, batch_size: int, n_measurements: int, n_qubits: int) -> dict:
        """Estimate memory usage for given batch configuration."""
        n = 2 * n_qubits
        input_size = n * n

        param_memory = sum(p.numel() * 4 for p in self.model.parameters()) / (1024 * 1024)
        if self.model_type == 'uniform':
            activation_memory = 0
        elif self.model_type in ['clifford_mlp', 'quantum_aware']:
            hidden_dim = getattr(self.model, 'network')[0].out_features
            activation_memory = (batch_size * n_measurements * hidden_dim * 4) / (1024 * 1024)
        elif self.model_type == 'attention':
            hidden_dim = self.model.token_proj.out_features
            seq_len = n
            # Attention matrix + embeddings
            activation_memory = (batch_size * n_measurements * (seq_len * seq_len + seq_len * hidden_dim) * 4) / (1024 * 1024)
        elif self.model_type == 'memory_efficient':
            compressed_dim = self.model.compressed_dim
            window_size = self.model.local_attn.window_size
            activation_memory = (batch_size * n_measurements * n * (window_size + compressed_dim) * 4) / (1024 * 1024)
        else:
            activation_memory = param_memory  # Conservative estimate

        return {
            'parameter_memory_mb': param_memory,
            'activation_memory_mb': activation_memory,
            'total_memory_mb': param_memory + activation_memory,
            'model_type': self.model_type
        }

    def benchmark(self, batch_tableau, num_runs: int = 10) -> dict:
        """Benchmark model forward pass timing."""
        import time

        _ = self.process_batch(batch_tableau)  # Warmup
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_time = time.time()

        for _ in range(num_runs):
            _ = self.process_batch(batch_tableau)

        torch.cuda.synchronize() if torch.cuda.is_available() else None
        total_time = time.time() - start_time

        avg_time = total_time / num_runs
        throughput = batch_tableau.batch_size * batch_tableau.n_measurements / avg_time

        return {
            'avg_forward_time_ms': avg_time * 1000,
            'throughput_samples_per_sec': throughput,
            'total_samples': batch_tableau.batch_size * batch_tableau.n_measurements,
            'active_samples': len(batch_tableau.to_flat_tensors_active_only()[0])
        }

