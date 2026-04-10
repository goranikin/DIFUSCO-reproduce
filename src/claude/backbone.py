"""
AGNN Backbone with Timestep Conditioning for DIFUSCO
=====================================================
Based on Section 3.4 of the paper.

Key modification from your existing AGNN:
  - Sinusoidal timestep embedding (t) is injected into the edge update
    via MLP_t(t), as in the paper's equation:
        e_{ij}^{l+1} = e_{ij}^l + MLP_e(BN(ê_{ij}^{l+1})) + MLP_t(t)
  - Initial edge features include the noisy solution x_t (not just distance)
  - Node features use sinusoidal position encoding of (x,y) coordinates

Paper hyperparameters (Section 3.4):
  - 12-layer AGNN, width 256 for full DIFUSCO
  - We use configurable layers/width for experimentation
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Sinusoidal Embeddings
# ============================================================

def sinusoidal_embedding(values: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Create sinusoidal positional embeddings.

    Used for:
    1. Timestep t → embed the diffusion step
    2. Node coordinates (x, y) → embed spatial position
    3. Edge distances → embed edge features

    This is the same scheme from "Attention Is All You Need" (Vaswani et al.):
        PE(pos, 2i)   = sin(pos / 10000^(2i/d))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d))

    Args:
        values: (...) tensor of scalar values to embed
        dim: embedding dimension (must be even)
    Returns:
        (..., dim) tensor of sinusoidal embeddings
    """
    assert dim % 2 == 0, f"Embedding dim must be even, got {dim}"
    half_dim = dim // 2

    # Frequency bands: 10000^(-2i/d) for i = 0, 1, ..., d/2-1
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half_dim, device=values.device, dtype=torch.float32) / half_dim
    )

    # Outer product: values × frequencies
    # values shape: (...), freqs shape: (half_dim,)
    args = values.unsqueeze(-1).float() * freqs
    embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    return embedding


def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Embed a single timestep into a vector.

    Args:
        t: (1,) or scalar timestep
        dim: embedding dimension
    Returns:
        (1, dim) or (dim,) embedding vector
    """
    return sinusoidal_embedding(t.float(), dim)


class PositionEmbeddingSine2D(nn.Module):
    """
    Sinusoidal position embedding for 2D coordinates.
    Embeds (x, y) coordinates into a high-dimensional space.

    For each coordinate, we create dim//2 sinusoidal features,
    then concatenate x and y embeddings → total dim features per node.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        assert dim % 2 == 0
        self.half_dim = dim // 2

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Args:
            coords: (N, 2) node coordinates in [0, 1]
        Returns:
            (N, dim) sinusoidal position embeddings
        """
        x_embed = sinusoidal_embedding(coords[:, 0], self.half_dim)
        y_embed = sinusoidal_embedding(coords[:, 1], self.half_dim)
        return torch.cat([x_embed, y_embed], dim=-1)


class ScalarEmbeddingSine(nn.Module):
    """
    Sinusoidal embedding for scalar edge features (distances, noisy labels).
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        """
        Args:
            values: (E,) or (E, 1) scalar values
        Returns:
            (E, dim) sinusoidal embeddings
        """
        if values.dim() > 1:
            values = values.squeeze(-1)
        return sinusoidal_embedding(values, self.dim)


# ============================================================
# AGNN Layer with Timestep Conditioning
# ============================================================

class AGNNLayerWithTime(nn.Module):
    """
    AGNN layer from DIFUSCO Section 3.4, with timestep conditioning.

    The key equations:
        ê_{ij}^{l+1} = P^l e_{ij}^l + Q^l h_i^l + R^l h_j^l
        e_{ij}^{l+1} = e_{ij}^l + MLP_e(LN(ê_{ij}^{l+1})) + MLP_t(t)
        h_i^{l+1}   = h_i^l + ReLU(LN(U^l h_i^l + Σ_{j∈N(i)} σ(ê_{ij}^{l+1}) ⊙ V^l h_j^l))

    The MLP_t(t) term is the timestep conditioning — it tells each layer
    what noise level the input corresponds to.
    """

    def __init__(self, node_dim: int, edge_dim: int):
        super().__init__()
        # Edge update projections
        self.P = nn.Linear(edge_dim, edge_dim, bias=False)
        self.Q = nn.Linear(node_dim, edge_dim, bias=False)
        self.R = nn.Linear(node_dim, edge_dim, bias=False)

        # Edge MLP with residual
        self.edge_norm = nn.LayerNorm(edge_dim)
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_dim, edge_dim),
            nn.ReLU(),
            nn.Linear(edge_dim, edge_dim),
        )

        # Timestep conditioning MLP for edges
        self.time_mlp = nn.Sequential(
            nn.Linear(edge_dim, edge_dim),
            nn.ReLU(),
            nn.Linear(edge_dim, edge_dim),
        )

        # Node update projections
        self.U = nn.Linear(node_dim, node_dim, bias=False)
        self.V = nn.Linear(node_dim, node_dim, bias=False)
        self.node_norm = nn.LayerNorm(node_dim)

    def forward(self, h, e, edge_index, t_emb):
        """
        Args:
            h:          (N, node_dim) node features
            e:          (E, edge_dim) edge features
            edge_index: (2, E) edge indices
            t_emb:      (1, edge_dim) timestep embedding (broadcast to all edges)
        Returns:
            h_new, e_new
        """
        src, dst = edge_index[0], edge_index[1]

        # Edge update: ê = P*e + Q*h_i + R*h_j
        e_hat = self.P(e) + self.Q(h[src]) + self.R(h[dst])

        # Residual + MLP + timestep conditioning
        e_new = e + self.edge_mlp(self.edge_norm(e_hat)) + self.time_mlp(t_emb)

        # Node update with gating
        gate = torch.sigmoid(e_hat)
        Vh = self.V(h)
        msg = gate * Vh[dst]

        agg = torch.zeros_like(h)
        agg.index_add_(0, src, msg)

        Uh = self.U(h)
        h_new = h + F.relu(self.node_norm(Uh + agg))

        return h_new, e_new


# ============================================================
# Full DIFUSCO Backbone
# ============================================================

class DifuscoBackbone(nn.Module):
    """
    Complete AGNN backbone for DIFUSCO.

    Input processing:
      - Node features: sinusoidal encoding of (x, y) coordinates
      - Edge features: sinusoidal encoding of distance + noisy solution x_t
      - Timestep: sinusoidal embedding of diffusion step t

    Output:
      - For discrete diffusion: 2-class logits per edge (prob of 0 vs 1)
      - For continuous diffusion: predicted noise ε per edge (1 scalar)
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_layers: int = 12,
        dropout: float = 0.0,
        diffusion_type: str = "categorical",  # "categorical" or "gaussian"
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.diffusion_type = diffusion_type

        # Input embeddings
        self.node_pos_embed = PositionEmbeddingSine2D(hidden_dim)
        self.edge_dist_embed = ScalarEmbeddingSine(hidden_dim // 2)
        self.edge_noise_embed = ScalarEmbeddingSine(hidden_dim // 2)

        # Timestep embedding projection
        self.time_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # AGNN layers
        self.layers = nn.ModuleList([
            AGNNLayerWithTime(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])

        # Output head
        if diffusion_type == "categorical":
            # 2-class classification per edge: [prob_0, prob_1]
            self.edge_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 2),
            )
        else:
            # Predict noise ε (1 scalar per edge)
            self.edge_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )

    def forward(
        self,
        node_coords: torch.Tensor,     # (N, 2)
        edge_index: torch.Tensor,       # (2, E)
        edge_distances: torch.Tensor,   # (E,)
        x_t: torch.Tensor,              # (E,) noisy edge labels (current diffusion state)
        t: torch.Tensor,                # scalar or (1,) timestep
    ):
        """
        Predict clean solution (categorical) or noise (gaussian) from noisy input.

        Returns:
            For categorical: (E, 2) logits
            For gaussian: (E, 1) predicted noise
        """
        # 1. Embed node positions
        h = self.node_pos_embed(node_coords)  # (N, hidden_dim)

        # 2. Embed edge features: distance + noisy solution
        e_dist = self.edge_dist_embed(edge_distances)       # (E, hidden_dim//2)
        e_noise = self.edge_noise_embed(x_t)                 # (E, hidden_dim//2)
        e = torch.cat([e_dist, e_noise], dim=-1)             # (E, hidden_dim)

        # 3. Embed timestep
        t_emb = timestep_embedding(t, self.hidden_dim)        # (1, hidden_dim) or (hidden_dim,)
        t_emb = self.time_proj(t_emb.unsqueeze(0) if t_emb.dim() == 1 else t_emb)
        # (1, hidden_dim) — will broadcast to all edges

        # 4. Forward through AGNN layers
        for layer in self.layers:
            h, e = layer(h, e, edge_index, t_emb)
            if self.dropout > 0 and self.training:
                h = F.dropout(h, p=self.dropout, training=True)
                e = F.dropout(e, p=self.dropout, training=True)

        # 5. Output head
        out = self.edge_head(e)  # (E, 2) or (E, 1)
        return out
