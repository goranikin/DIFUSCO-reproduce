import torch
import torch.nn as nn
import torch.nn.functional as F

from src.embeddings import (
    PositionEmbeddingSine2D,
    ScalarEmbeddingSine,
    timestep_embedding,
)


class AGNNLayer(nn.Module):
    """
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


class DifuscoBackbone(nn.Module):
    def __init__(
        self, hidden_dim: int, num_layers: int, dropout: float, diffusion_type: str
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

        self.time_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.layers: nn.ModuleList[AGNNLayer] = nn.ModuleList(
            [AGNNLayer(hidden_dim, hidden_dim) for _ in range(num_layers)]
        )

        if diffusion_type == "categorical":
            self.edge_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 2),
            )
        else:
            self.edge_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )

    def forward(
        self,
        node_coords: torch.Tensor,
        edge_index: torch.Tensor,
        edge_distances: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
    ):
        """
        node_coords: (N, 2)
        edge_index: (2, E)
        edge_distances: (E,)
        x_t: (E,) current noisy edge labels
        t: (1,) timestep (scalar)
        """
        # (N, hidden_dim)
        h = self.node_pos_embed(node_coords)

        # (N*(N-1), hidden_dim//2)
        e_dist = self.edge_dist_embed(edge_distances)
        # (N*(N-1), hidden_dim//2)
        e_noise = self.edge_noise_embed(x_t)
        # (N*(N-1), hidden_dim)
        e = torch.cat([e_dist, e_noise], dim=-1)

        # (1, hidden_dim)
        t_emb = timestep_embedding(t, self.hidden_dim)
        t_emb = self.time_proj(t_emb.unsqueeze(0) if t_emb.dim() == 1 else t_emb)

        for layer in self.layers:
            h, e = layer(h, e, edge_index, t_emb)
            if self.dropout > 0 and self.training:
                h = F.dropout(h, p=self.dropout, training=True)
                e = F.dropout(e, p=self.dropout, training=True)

        out = self.edge_head(e)
        return out
