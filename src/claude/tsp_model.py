"""
DIFUSCO TSP Model — Training & Inference
==========================================
Combines the AGNN backbone with the diffusion process for TSP.

Training (Section 3.2):
  1. Sample a random timestep t ~ Uniform(0, T)
  2. Add noise to the ground-truth tour edges: x_t ~ q(x_t | x_0)
  3. Feed (graph, x_t, t) to the backbone
  4. Compute loss:
     - Categorical: cross-entropy between predicted p(x_0 | x_t) and true x_0
     - Gaussian: MSE between predicted noise and true noise

Inference (Sections 3.3, 3.5):
  1. Start from pure noise x_T
  2. For each step in the inference schedule:
     - Predict x_0 (or noise) using the backbone
     - Sample x_{t-1} from the posterior q(x_{t-1} | x_t, x_0_pred)
  3. Generate heatmap from final prediction
  4. Apply greedy decoding + 2-opt to extract a valid tour
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import DifuscoBackbone
from .diffusion import CategoricalDiffusion, GaussianDiffusion, InferenceSchedule


class DifuscoTSP(nn.Module):
    """
    Full DIFUSCO model for TSP.
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_layers: int = 12,
        T: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        diffusion_type: str = "categorical",
        dropout: float = 0.0,
    ):
        super().__init__()
        self.T = T
        self.diffusion_type = diffusion_type

        # Backbone (the AGNN denoising network)
        self.backbone = DifuscoBackbone(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            diffusion_type=diffusion_type,
        )

        # Diffusion process
        if diffusion_type == "categorical":
            self.diffusion = CategoricalDiffusion(T, beta_start, beta_end)
        else:
            self.diffusion = GaussianDiffusion(T, beta_start, beta_end)

    def training_step(self, batch, device):
        """
        One training step.

        Args:
            batch: (node_feat, edge_index, edge_dist, edge_label) from DataLoader
            device: torch device
        Returns:
            loss: scalar tensor
        """
        node_feat, edge_index, edge_dist, edge_label = batch
        node_feat = node_feat.squeeze(0).to(device)
        edge_index = edge_index.squeeze(0).to(device)
        edge_dist = edge_dist.squeeze(0).to(device)
        edge_label = edge_label.squeeze(0).to(device)

        # 1. Sample random timestep t ~ Uniform(0, T-1)
        t = torch.randint(0, self.T, (1,), device=device).long()

        if self.diffusion_type == "categorical":
            return self._categorical_training_step(
                node_feat, edge_index, edge_dist, edge_label, t
            )
        else:
            return self._gaussian_training_step(
                node_feat, edge_index, edge_dist, edge_label, t
            )

    def _categorical_training_step(
        self, node_feat, edge_index, edge_dist, edge_label, t
    ):
        """
        Categorical diffusion training.

        The network predicts p_θ(x̃_0 | x_t), i.e., the clean solution
        from the noisy input. Loss is cross-entropy.
        """
        # 2. Add Bernoulli noise: sample x_t ~ q(x_t | x_0)
        x_t = self.diffusion.q_sample(edge_label, t.item())

        # 3. Forward pass: predict p(x_0 = 1 | x_t, graph, t)
        logits = self.backbone(node_feat, edge_index, edge_dist, x_t, t.float())
        # logits: (E, 2) — class 0 and class 1 logits

        # 4. Loss: cross-entropy with the true labels
        targets = edge_label.long()  # (E,) with values {0, 1}
        loss = F.cross_entropy(logits, targets)

        return loss

    def _gaussian_training_step(self, node_feat, edge_index, edge_dist, edge_label, t):
        """
        Gaussian diffusion training.

        The network predicts the noise ε that was added.
        Loss is MSE between predicted and true noise.
        """
        # 2. Add Gaussian noise: sample x̃_t ~ q(x̃_t | x̃_0)
        noise = torch.randn_like(edge_label)
        x_t = self.diffusion.q_sample(edge_label, t.item(), noise=noise)

        # 3. Forward pass: predict the noise ε
        noise_pred = self.backbone(node_feat, edge_index, edge_dist, x_t, t.float())
        noise_pred = noise_pred.squeeze(-1)  # (E,)

        # 4. Loss: MSE between predicted and true noise
        loss = F.mse_loss(noise_pred, noise)

        return loss

    @torch.no_grad()
    def generate(
        self,
        node_feat: torch.Tensor,
        edge_index: torch.Tensor,
        edge_dist: torch.Tensor,
        num_inference_steps: int = 50,
        schedule_type: str = "cosine",
        device: torch.device = None,
    ) -> torch.Tensor:
        """
        Generate a TSP solution using iterative denoising.

        Args:
            node_feat:  (N, 2) node coordinates
            edge_index: (2, E) edge indices
            edge_dist:  (E,) edge distances
            num_inference_steps: M (number of denoising steps)
            schedule_type: "linear" or "cosine"
            device: torch device
        Returns:
            heatmap: (E,) edge probabilities (confidence scores)
        """
        self.eval()
        if device is None:
            device = next(self.parameters()).device

        node_feat = node_feat.to(device)
        edge_index = edge_index.to(device)
        edge_dist = edge_dist.to(device)
        E = edge_index.shape[1]

        # Get inference timestep schedule
        timesteps = InferenceSchedule.get_schedule(
            schedule_type, num_inference_steps, self.T
        )

        if self.diffusion_type == "categorical":
            return self._categorical_inference(
                node_feat, edge_index, edge_dist, E, timesteps, device
            )
        else:
            return self._gaussian_inference(
                node_feat, edge_index, edge_dist, E, timesteps, device
            )

    def _categorical_inference(
        self, node_feat, edge_index, edge_dist, E, timesteps, device
    ):
        """
        Reverse process for categorical diffusion.

        Start from x_T ~ Uniform({0,1}), iteratively denoise.
        """
        # Start from pure noise: x_T ~ Bernoulli(0.5) = Uniform({0,1})
        x_t = torch.bernoulli(torch.ones(E, device=device) * 0.5)

        for i, t in enumerate(timesteps):
            t_tensor = torch.tensor([t], device=device, dtype=torch.float32)

            # Predict p(x_0 | x_t)
            logits = self.backbone(node_feat, edge_index, edge_dist, x_t, t_tensor)
            probs = F.softmax(logits, dim=-1)  # (E, 2)
            x_0_pred = probs[:, 1]  # probability of class 1

            if i == len(timesteps) - 1:
                # Last step: return heatmap (don't sample)
                return x_0_pred
            else:
                # Sample x_{t-1} from posterior
                next_t = timesteps[i + 1] if i + 1 < len(timesteps) else 0
                x_t = self.diffusion.q_posterior(x_t, x_0_pred, t)

        return x_0_pred

    def _gaussian_inference(
        self, node_feat, edge_index, edge_dist, E, timesteps, device
    ):
        """
        Reverse process for Gaussian diffusion.

        Start from x_T ~ N(0, I), iteratively denoise.
        """
        # Start from pure Gaussian noise
        x_t = torch.randn(E, device=device)

        for i, t in enumerate(timesteps):
            t_tensor = torch.tensor([t], device=device, dtype=torch.float32)

            # Predict noise
            noise_pred = self.backbone(node_feat, edge_index, edge_dist, x_t, t_tensor)
            noise_pred = noise_pred.squeeze(-1)

            # Recover x̃_0 prediction
            x_0_pred = self.diffusion.predict_x0_from_noise(x_t, t, noise_pred)

            if i == len(timesteps) - 1:
                # Last step: convert to heatmap
                # From paper: use 0.5*(x̃_0 + 1) as heatmap scores
                heatmap = 0.5 * (x_0_pred + 1.0)
                return heatmap.clamp(0.0, 1.0)
            else:
                # Sample x̃_{t-1}
                x_t = self.diffusion.q_posterior(x_t, x_0_pred, t)

        heatmap = 0.5 * (x_0_pred + 1.0)
        return heatmap.clamp(0.0, 1.0)


# ============================================================
# TSP Decoding — Section 3.5
# ============================================================


def greedy_decode_tsp(
    heatmap: torch.Tensor, edge_index: torch.Tensor, node_coords: torch.Tensor
) -> list:
    """
    Greedy decoding from heatmap scores (Section 3.5).

    All edges are ranked by (A_ij + A_ji) / ||c_i - c_j|| and inserted
    into the partial solution if there are no conflicts.

    A valid TSP tour requires:
    - Each node has exactly degree 2
    - No subtours (must be a single Hamiltonian cycle)

    Args:
        heatmap:     (E,) edge confidence scores from diffusion model
        edge_index:  (2, E) edge indices
        node_coords: (N, 2) coordinates
    Returns:
        tour: list of node indices forming the tour
    """
    N = node_coords.shape[0]
    E = edge_index.shape[1]

    # Combine scores for both directions of each undirected edge
    # and normalize by distance
    edge_scores = {}
    for k in range(E):
        u, v = edge_index[0, k].item(), edge_index[1, k].item()
        key = (min(u, v), max(u, v))
        score = heatmap[k].item()
        if key in edge_scores:
            edge_scores[key] += score
        else:
            edge_scores[key] = score

    # Normalize by distance
    for u, v in edge_scores:
        dist = torch.norm(node_coords[u] - node_coords[v]).item()
        edge_scores[(u, v)] /= dist + 1e-8

    # Sort edges by score (descending)
    sorted_edges = sorted(edge_scores.items(), key=lambda x: x[1], reverse=True)

    # Greedy insertion
    adj = {i: [] for i in range(N)}
    selected_edges = []

    for (u, v), score in sorted_edges:
        # Check degree constraint: each node can have at most 2 edges
        if len(adj[u]) >= 2 or len(adj[v]) >= 2:
            continue

        # Check subtour constraint: don't close a cycle unless it would be the full tour
        if len(adj[u]) == 1 and len(adj[v]) == 1:
            # Would this create a cycle? Trace the path from u
            if _would_create_subtour(adj, u, v, N):
                continue

        adj[u].append(v)
        adj[v].append(u)
        selected_edges.append((u, v))

        if len(selected_edges) == N:
            break

    # If we don't have enough edges, add remaining greedily
    if len(selected_edges) < N:
        # Find endpoints (nodes with degree < 2) and connect them
        endpoints = [i for i in range(N) if len(adj[i]) < 2]
        while len(selected_edges) < N and len(endpoints) >= 2:
            u = endpoints[0]
            # Find closest available endpoint
            best_v = None
            best_dist = float("inf")
            for v in endpoints[1:]:
                if len(adj[u]) < 2 and len(adj[v]) < 2:
                    dist = torch.norm(node_coords[u] - node_coords[v]).item()
                    if dist < best_dist and not _would_create_subtour(adj, u, v, N):
                        best_dist = dist
                        best_v = v
            if best_v is None:
                # Allow closing subtour as last resort
                for v in endpoints[1:]:
                    if len(adj[u]) < 2 and len(adj[v]) < 2:
                        best_v = v
                        break
            if best_v is not None:
                adj[u].append(best_v)
                adj[best_v].append(u)
                selected_edges.append((u, best_v))
            endpoints = [i for i in range(N) if len(adj[i]) < 2]

    # Extract tour from adjacency list
    tour = _extract_tour(adj, N)
    return tour


def _would_create_subtour(adj, u, v, N):
    """Check if adding edge (u, v) would create a cycle shorter than N."""
    if not adj[u] or not adj[v]:
        return False

    # Trace path from u (not going through v)
    path_length = 1
    current = u
    prev = v  # pretend we came from v
    while True:
        neighbors = [n for n in adj[current] if n != prev]
        if not neighbors:
            break
        prev = current
        current = neighbors[0]
        path_length += 1
        if current == v:
            # Found a cycle — only allow if it's the full tour
            return path_length < N

    return False


def _extract_tour(adj, N):
    """Extract a tour from the adjacency list."""
    if not all(len(adj[i]) == 2 for i in range(N)):
        # Incomplete tour — return best-effort ordering
        visited = [False] * N
        tour = [0]
        visited[0] = True
        current = 0
        while len(tour) < N:
            neighbors = [n for n in adj[current] if not visited[n]]
            if neighbors:
                current = neighbors[0]
            else:
                # Jump to nearest unvisited
                unvisited = [i for i in range(N) if not visited[i]]
                if unvisited:
                    current = unvisited[0]
                else:
                    break
            visited[current] = True
            tour.append(current)
        return tour

    # Complete tour — trace it
    tour = [0]
    prev = -1
    current = 0
    for _ in range(N - 1):
        neighbors = [n for n in adj[current] if n != prev]
        prev = current
        current = neighbors[0]
        tour.append(current)
    return tour


def two_opt(tour: list, node_coords: torch.Tensor, max_iterations: int = 100) -> list:
    """
    2-opt local search to improve tour quality.

    Repeatedly tries reversing segments of the tour to reduce total distance.
    """
    coords = node_coords.cpu().numpy()
    tour = list(tour)
    N = len(tour)
    improved = True
    iteration = 0

    def tour_dist(t):
        d = 0
        for i in range(len(t)):
            d += np.sqrt(((coords[t[i]] - coords[t[(i + 1) % len(t)]]) ** 2).sum())
        return d

    best_distance = tour_dist(tour)

    while improved and iteration < max_iterations:
        improved = False
        iteration += 1
        for i in range(1, N - 1):
            for j in range(i + 1, N):
                # Try reversing segment [i, j]
                new_tour = tour[:i] + tour[i : j + 1][::-1] + tour[j + 1 :]
                new_distance = tour_dist(new_tour)
                if new_distance < best_distance - 1e-10:
                    tour = new_tour
                    best_distance = new_distance
                    improved = True
                    break
            if improved:
                break

    return tour


def compute_tour_length(tour: list, node_coords: torch.Tensor) -> float:
    """Compute total Euclidean distance of a tour."""
    total = 0.0
    coords = node_coords.cpu()
    for i in range(len(tour)):
        u, v = tour[i], tour[(i + 1) % len(tour)]
        total += torch.norm(coords[u] - coords[v]).item()
    return total
