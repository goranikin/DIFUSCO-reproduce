import math

import numpy as np
import torch


class CategoricalDiffusion:
    """
    Forward process:
        q(x_t | x_{t-1}) = Cat(x_t; p = x̃_{t-1} Q_t)

    where Q_t = [(1-β_t, β_t), (β_t, 1-β_t)] is the transition matrix,
    and x̃ is the one-hot encoding of x.

    The t-step marginal is:
        q(x_t | x_0) = Cat(x_t; p = x̃_0 Q̄_t)
    where Q̄_t = Q_1 Q_2 ... Q_t
    """

    def __init__(self, T: int, beta_start: float = 1e-4, beta_end: float = 0.02):
        self.T = T

        # Linear noise schedule: β_t linearly interpolated from β_1 to β_T
        betas = np.linspace(beta_start, beta_end, T)
        self.betas = torch.tensor(betas, dtype=torch.float64)

        # For Bernoulli diffusion, the cumulative transition probability
        # that a bit has been flipped by time t is:
        #   Q̄_t = Π_{s=1}^{t} Q_s
        #
        # For 2-state system, Q̄_t can be parameterized by a single value ᾱ_t
        # where ᾱ_t = Π_{s=1}^{t} (1 - β_s)
        #
        # Q̄_t = [(1-ᾱ_t)/2 + ᾱ_t/2 + 1/2,  (1-ᾱ_t)/2]
        #        [(1-ᾱ_t)/2,                  (1-ᾱ_t)/2 + ᾱ_t/2 + 1/2]
        # Simplifies to: p(x_t = x_0 | x_0) = (1 + ᾱ_t) / 2
        #                p(x_t ≠ x_0 | x_0) = (1 - ᾱ_t) / 2
        alphas = 1.0 - betas
        self.alphas_cumprod = torch.tensor(np.cumprod(alphas), dtype=torch.float64)

    def q_sample(self, x_0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Sample x_t from q(x_t | x_0) — the forward process at timestep t.

        For Bernoulli diffusion:
            p(x_t = 1 | x_0) = x_0 * (1+ᾱ_t)/2 + (1-x_0) * (1-ᾱ_t)/2
                              = x_0 * ᾱ_t + (1 - ᾱ_t) / 2

        Args:
            x_0: (E,) or (E, 1) binary edge labels {0, 1}
            t:   (1,) or scalar, timestep index (0-indexed)
        Returns:
            x_t: (E,) sampled noisy edge labels {0, 1}
        """
        alpha_bar = self.alphas_cumprod[t].float().to(x_0.device)

        # Probability that x_t = 1
        # If x_0 = 1: prob = (1 + ᾱ_t) / 2
        # If x_0 = 0: prob = (1 - ᾱ_t) / 2
        prob_one = x_0 * (1.0 + alpha_bar) / 2.0 + (1.0 - x_0) * (1.0 - alpha_bar) / 2.0

        # Sample from Bernoulli
        x_t = torch.bernoulli(prob_one)
        return x_t

    def q_posterior(
        self, x_t: torch.Tensor, x_0_pred: torch.Tensor, t: int
    ) -> torch.Tensor:
        """
        Compute q(x_{t-1} | x_t, x_0) — the posterior for one reverse step.

        From Eq. 5 in the paper:
            q(x_{t-1} | x_t, x_0) = Cat(x_{t-1}; p = (x̃_t Q_t^T ⊙ x̃_0 Q̄_{t-1}^T) / Z)

        For Bernoulli, this simplifies to computing the probability that
        x_{t-1} = 1 given x_t and predicted x_0.

        Args:
            x_t:      (E,) current noisy state
            x_0_pred: (E,) predicted clean probabilities p(x_0 = 1 | x_t)
            t:        current timestep (must be > 0)
        Returns:
            x_{t-1}: (E,) sampled state one step closer to clean
        """
        if t == 0:
            # At t=0, just threshold the prediction
            return (x_0_pred > 0.5).float()

        beta_t = self.betas[t].float().to(x_t.device)
        alpha_bar_t_minus_1 = self.alphas_cumprod[t - 1].float().to(x_t.device)

        # Likelihood term: p(x_t | x_{t-1}=1) and p(x_t | x_{t-1}=0)
        # From Q_t: p(x_t=v | x_{t-1}=v) = 1-β_t, p(x_t=v | x_{t-1}≠v) = β_t
        p_xt_given_xtm1_is_1 = x_t * (1 - beta_t) + (1 - x_t) * beta_t
        p_xt_given_xtm1_is_0 = x_t * beta_t + (1 - x_t) * (1 - beta_t)

        # Prior term: p(x_{t-1}=1 | x_0) and p(x_{t-1}=0 | x_0)
        # Using predicted x_0 probabilities
        p_xtm1_is_1 = (
            x_0_pred * (1 + alpha_bar_t_minus_1) / 2
            + (1 - x_0_pred) * (1 - alpha_bar_t_minus_1) / 2
        )
        p_xtm1_is_0 = 1.0 - p_xtm1_is_1

        # Posterior: p(x_{t-1}=1 | x_t, x_0) ∝ p(x_t | x_{t-1}=1) * p(x_{t-1}=1 | x_0)
        unnorm_1 = p_xt_given_xtm1_is_1 * p_xtm1_is_1
        unnorm_0 = p_xt_given_xtm1_is_0 * p_xtm1_is_0

        prob_1 = unnorm_1 / (unnorm_1 + unnorm_0 + 1e-8)

        return torch.bernoulli(prob_1)


class GaussianDiffusion:
    """
    Continuous diffusion for discrete data by lifting to {-1, 1} space.

    Following Chen et al. (2022), we rescale {0,1} variables to {-1,1}:
        x̃_0 = 2 * x_0 - 1

    Forward process:
        q(x̃_t | x̃_{t-1}) = N(x̃_t; √(1-β_t) x̃_{t-1}, β_t I)
    """

    def __init__(self, T: int, beta_start: float = 1e-4, beta_end: float = 0.02):
        self.T = T

        betas = np.linspace(beta_start, beta_end, T)
        self.betas = torch.tensor(betas, dtype=torch.float64)

        alphas = 1.0 - betas
        self.alphas_cumprod = torch.tensor(np.cumprod(alphas), dtype=torch.float64)

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

    def q_sample(
        self, x_0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Sample x̃_t from q(x̃_t | x̃_0).

        x̃_t = √(ᾱ_t) * x̃_0 + √(1-ᾱ_t) * ε,  where ε ~ N(0, I)

        Args:
            x_0: (E,) binary edge labels {0, 1} — will be rescaled to {-1, 1}
            t:   timestep index
            noise: optional pre-generated noise
        Returns:
            x_t: (E,) noisy continuous values
        """
        # Rescale {0,1} → {-1,1}
        x_0_scaled = 2.0 * x_0 - 1.0

        if noise is None:
            noise = torch.randn_like(x_0_scaled)

        sqrt_alpha_bar = self.sqrt_alphas_cumprod[t].float().to(x_0.device)
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t].float().to(x_0.device)

        x_t = sqrt_alpha_bar * x_0_scaled + sqrt_one_minus * noise
        return x_t

    def predict_x0_from_noise(
        self, x_t: torch.Tensor, t: int, noise_pred: torch.Tensor
    ) -> torch.Tensor:
        """
        Recover x̃_0 from the predicted noise:
            x̃_0 = (x̃_t - √(1-ᾱ_t) * ε_pred) / √(ᾱ_t)
        """
        sqrt_alpha_bar = self.sqrt_alphas_cumprod[t].float().to(x_t.device)
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t].float().to(x_t.device)

        x_0_pred = (x_t - sqrt_one_minus * noise_pred) / (sqrt_alpha_bar + 1e-8)
        return x_0_pred

    def q_posterior(
        self, x_t: torch.Tensor, x_0_pred: torch.Tensor, t: int
    ) -> torch.Tensor:
        """
        Sample x̃_{t-1} from q(x̃_{t-1} | x̃_t, x̃_0) — Eq. 7 & 8 in the paper.

        This is a closed-form Gaussian:
            p_θ(x̃_{t-1} | x̃_t) = q(x̃_{t-1} | x̃_t, x̃_0_pred)

        Using Eq. 8:
            x̃_{t-1} = (x̃_t - √(1-ᾱ_t) * f_θ(x̃_t,t)) / √(ᾱ_t)
            then sample with appropriate variance
        """
        if t == 0:
            # Convert back to {0,1} by thresholding at 0
            return (x_0_pred > 0.0).float()

        alpha_bar_t = self.alphas_cumprod[t].float().to(x_t.device)
        alpha_bar_t_minus_1 = self.alphas_cumprod[t - 1].float().to(x_t.device)
        beta_t = self.betas[t].float().to(x_t.device)

        # Posterior mean (DDPM formula)
        posterior_mean = (
            torch.sqrt(alpha_bar_t_minus_1) * beta_t / (1.0 - alpha_bar_t) * x_0_pred
            + torch.sqrt(1.0 - beta_t)
            * (1.0 - alpha_bar_t_minus_1)
            / (1.0 - alpha_bar_t)
            * x_t
        )

        # Posterior variance
        posterior_var = beta_t * (1.0 - alpha_bar_t_minus_1) / (1.0 - alpha_bar_t)

        noise = torch.randn_like(x_t)
        x_t_minus_1 = posterior_mean + torch.sqrt(posterior_var + 1e-8) * noise

        return x_t_minus_1


###


class InferenceSchedule:
    """
    Maps M inference steps to a subset of T training timesteps.

    Two schedule types from the paper:
      - linear:  τ_i = ⌊c·i⌋
      - cosine:  τ_i = ⌊cos((1-c·i)/2 · π) · T⌋

    The cosine schedule spends more steps in the low-noise regime,
    which improves generation quality.
    """

    @staticmethod
    def get_schedule(schedule_type: str, num_inference_steps: int, T: int):
        """
        Args:
            schedule_type: "linear" or "cosine"
            num_inference_steps: M (number of actual denoising steps)
            T: total training timesteps
        Returns:
            list of timestep indices (descending, from high noise to low noise)
        """
        if schedule_type == "linear":
            c = T / num_inference_steps
            timesteps = [int(c * i) for i in range(num_inference_steps, 0, -1)]
        elif schedule_type == "cosine":
            timesteps = []
            for i in range(num_inference_steps, 0, -1):
                c = i / num_inference_steps
                t = int(math.cos((1 - c) / 2 * math.pi) * T)
                t = max(0, min(T - 1, t))
                timesteps.append(t)
        else:
            raise ValueError(f"Unknown schedule type: {schedule_type}")

        return timesteps
