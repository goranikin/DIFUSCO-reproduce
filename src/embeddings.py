import math

import torch
import torch.nn as nn

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
        -math.log(10000.0)
        * torch.arange(half_dim, device=values.device, dtype=torch.float32)
        / half_dim
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
