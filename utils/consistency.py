import torch
import torch.nn as nn
import math
import numpy as np

# ---------------------------------
# Consistency Policy Wrapper
# ---------------------------------
class ConsistencyPolicy(nn.Module):
    """
    Wrapper for DiffusionPolicy to implement Consistency Models.
    f_theta(x, t) = c_skip(t) * x + c_out(t) * F_theta(x, t)
    """
    def __init__(
        self,
        model: nn.Module,
        sigma_data: float = 0.1, # Lowered from 0.5 to reduce noise leakage
        sigma_min: float = 0.002,
        sigma_max: float = 80.0
    ):
        super().__init__()
        self.model = model
        self.sigma_data = sigma_data
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def get_scalings(self, sigma):
        """
        Scale-invariant scalings for consistency models.
        f_theta = c_skip * x + (1 - c_skip) * F_theta
        """
        # Map discrete sigma (0-99) to [0, 1] range
        t = sigma / 100.0
        c_skip = self.sigma_data**2 / (t**2 + self.sigma_data**2)
        c_out = 1.0 - c_skip
        
        return c_skip, c_out

    def forward(
        self,
        observations: torch.Tensor,
        sigma: torch.Tensor,
        noised_actions: torch.Tensor,
        images: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Forward pass for consistency mapping.
        """
        sigma_bc = sigma.view(-1, 1, 1)
        c_skip, c_out = self.get_scalings(sigma_bc)
        
        # Internal model prediction
        model_output = self.model(
            observations=observations,
            diffusion_steps=sigma.long(), 
            noised_actions=noised_actions,
            images=images
        )
        
        return c_skip * noised_actions + c_out * model_output

# ---------------------------------
# Consistency Sampler
# ---------------------------------
class ConsistencySampler:
    def __init__(self, model, sigma_min=0.002, sigma_max=80.0):
        self.model = model
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    @torch.no_grad()
    def sample(
        self,
        observations,
        pred_horizon,
        action_dim,
        images=None,
        steps=1,
        sigma_start=1.0,
        discrete_sigma=99.0
    ):
        """
        Sample actions in 'steps' iterations.
        steps=1: Direct consistency mapping (extremely fast)
        steps=2: Two-step sampling (much smoother, recommended)
        """
        device = observations.device
        B = observations.shape[0]
        
        # 1. Initial noise
        x = torch.randn((B, pred_horizon, action_dim), device=device) * sigma_start
        
        if steps == 1:
            sigma = torch.ones((B,), device=device) * discrete_sigma
            return self.model(observations, sigma, x, images=images)
        
        # 2-step sampling logic
        # First step from T to 0
        sigma = torch.ones((B,), device=device) * discrete_sigma
        x = self.model(observations, sigma, x, images=images)
        
        # Intermediate step (re-noising)
        # We pick an intermediate step like T/2
        tau_idx = discrete_sigma / 2.0
        # Add noise proportional to tau_idx (simplified scale)
        z = torch.randn_like(x)
        x = x + z * (tau_idx / 100.0) # Scale noise to tau
        
        # Second step from tau to 0
        sigma = torch.ones((B,), device=device) * tau_idx
        x = self.model(observations, sigma, x, images=images)
        
        return x
