import torch
import torch.nn as nn
import torch.nn.functional as F



class DiscreteUniform(nn.Module):
    """Implements a uniform distribution over discrete actions.

    It uses a zero function approximator (a function that always outputs 0) to be used as
    logits by a DiscretePBEstimator.

    Attributes:
        output_dim: The size of the output space.
    """

    def __init__(self, output_dim: int) -> None:
        """Initializes the uniform function approximiator.

        Args:
            output_dim (int): Output dimension. This is typically n_actions if it
                implements a Uniform PF, or n_actions-1 if it implements a Uniform PB.
        """
        super().__init__()
        self.output_dim = output_dim

    def forward(self,x) :
        out = torch.zeros(self.output_dim).to(x.device)
        return out

class MLP(nn.Module):
    """
    A PyTorch MLP model for inputs representing a flat Clifford tableau.

    Parameters:
        input_dim (int): Size of the input feature vector (flat Clifford tableau length).
        hidden_dim (int): Number of neurons in each hidden layer.
        num_hidden_layers (int): Number of hidden layers in the network.
        output_dim (int): Size of the output layer (number of predictions `p`).
    """
    def __init__(self, input_dim: int, hidden_dim: int, num_hidden_layers: int, output_dim: int):
        super(MLP, self).__init__()
        layers = []
        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        # Additional hidden layers
        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Dropout(p=0.5))
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Dropout(p=0.5))
        
        # Output layer
        layers.append(nn.Linear(hidden_dim, output_dim))
        
        #layers.append(nn.LogSigmoid())
        self.network = nn.Sequential(*layers)
        self.logZ = nn.Parameter(torch.zeros(1))
        # Initialize weights
        self.init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the MLP.
        """
        return self.network(x)

    def init_weights(self):
        """
        Initializes the weights of the network using Xavier (Glorot) initialization.
        """
        nn.init.zeros_(self.logZ)
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)  # Xavier initialization
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)  # Initialize biases to zero

class GNNLayer(nn.Module):
    """
    A simple Graph Convolution-like layer that aggregates neighbor features
    via matrix multiplication with the adjacency, then transforms them linearly.
    """
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=True)

    def forward(self, x, adj):
        """
        Args:
            x:   (num_nodes, in_features) - node feature matrix
            adj: (num_nodes, num_nodes)   - adjacency matrix (0/1 from W)

        Returns:
            (num_nodes, out_features) updated node feature matrix
        """
        # Aggregate neighbor features: multiply adjacency by x (sum of neighbors)
        agg = torch.matmul(adj.float(), x)  # shape (num_nodes, in_features)
        
        # Linear transform of aggregated features
        out = self.linear(agg)
        return F.relu(out)


class EquivariantHeisenbergNet(nn.Module):
    """
    A GNN-based network to process:
      - W: (2N x 2N) binary matrix
      - phase_vec: length 2N
    and output `output_dim` logits.

    Pipeline:
        1) Each row of W (length 2N) -> node feature embedding
        2) W -> adjacency (i-j edge if W[i,j] == 1)
        3) Embed the 2N-dim phase vector and add/concat to node features
        4) Pass through multiple GNN layers
        5) Pool node embeddings -> graph embedding
        6) Linear readout -> (output_dim) logits
    """
    def __init__(self, 
                 N: int, 
                 hidden_dim: int, 
                 num_gnn_layers: int, 
                 output_dim: int):
        """
        Args:
            N (int): 
                Defines the size of W: (2N x 2N),
                and phase_vec: length 2N.
            hidden_dim (int): 
                Dimension for node embeddings in GNN.
            num_gnn_layers (int): 
                How many GNN layers to stack.
            output_dim (int): 
                Number of logits for the final output.
        """
        super().__init__()

        self.N = N
        self.hidden_dim = hidden_dim
        self.num_gnn_layers = num_gnn_layers
        self.output_dim = output_dim
        
        # 1) Node embedding for each row of W: 
        #    in_features = 2N (row length), out_features = hidden_dim
        self.node_embed = nn.Linear(2*N, hidden_dim)
        
        # 2) Phase embedding: in_features = 2N, out_features = hidden_dim
        self.phase_embed = nn.Linear(2*N, hidden_dim)

        # Create GNN layers
        self.gnn_layers = nn.ModuleList([
            GNNLayer(hidden_dim, hidden_dim) for _ in range(num_gnn_layers)
        ])

        # Final readout: from pooled embedding -> output_dim
        self.readout = nn.Linear(hidden_dim, output_dim)

        # logZ parameter for the model
        self.logZ = nn.Parameter(torch.zeros(1))
        
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, W, phase_vec):
        """
        Args:
            W:         (2N, 2N) binary matrix (0/1 in float or bool)
            phase_vec: (2N,) binary vector (0/1)

        Returns:
            logits of shape (output_dim,).
        """
        num_nodes = W.size(0)  # should be 2N

        # Create node features from each row of W
        # shape: (2N, 2N) -> node_feats: (2N, hidden_dim)
        node_feats = self.node_embed(W.float())

        # Embed the phase vector: shape (2N,) -> (1, 2N) -> (1, hidden_dim)
        phase_emb = self.phase_embed(phase_vec.float().unsqueeze(0))
        # Expand to match node dimension: (2N, hidden_dim)
        phase_emb_expanded = phase_emb.repeat(num_nodes, 1)

        # Combine W-based embedding + phase-based embedding
        h = node_feats + phase_emb_expanded  # shape: (2N, hidden_dim)

        # Adjacency from W (float):
        adjacency = W.float()  # shape: (2N, 2N)

        # Pass through GNN layers
        for gnn_layer in self.gnn_layers:
            h = gnn_layer(h, adjacency)  # shape stays (2N, hidden_dim)

        # Pool node embeddings -> single graph embedding
        # e.g., mean over all nodes
        h_graph = torch.mean(h, dim=0)  # shape (hidden_dim,)

        # Final linear layer -> logits
        logits = self.readout(h_graph)  # shape (output_dim,)
        return logits

