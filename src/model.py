import torch
import torch.nn as nn
import torch.nn.functional as F

from src.backbone import DifuscoBackbone
from src.diffusion import CategoricalDiffusion, GaussianDiffusion, InferenceSchedule


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
        One training step. Accepts a batched super-graph from collate_tsp.
        """
        node_feat, edge_index, edge_dist, edge_label = batch
        node_feat = node_feat.to(device)
        edge_index = edge_index.to(device)
        edge_dist = edge_dist.to(device)
        edge_label = edge_label.to(device)

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
        """
        assert isinstance(self.diffusion, GaussianDiffusion)
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
        device: torch.device,
        node_feat: torch.Tensor,
        edge_index: torch.Tensor,
        edge_dist: torch.Tensor,
        num_inference_steps: int = 50,
        schedule_type: str = "cosine",
    ) -> torch.Tensor:
        """
        Generate a TSP solution using iterative denoising.

        Args:
            device: torch device
            node_feat:  (N, 2) node coordinates
            edge_index: (2, E) edge indices
            edge_dist:  (E,) edge distances
            num_inference_steps: M (number of denoising steps)
            schedule_type: "linear" or "cosine"
        Returns:
            heatmap: (E,) edge probabilities (confidence scores)
        """
        self.eval()

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
