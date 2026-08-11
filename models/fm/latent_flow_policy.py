"""Flow-matching loss and ODE sampling for LIP interaction latents."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from .latent_dit import LatentDiT


class LatentFlowMatchingPolicy(nn.Module):
    """Thin training/sampling wrapper around :class:`LatentDiT`."""

    def __init__(
        self,
        *,
        latent_dim: int = 256,
        latent_horizon: int = 16,
        visual_dim: int = 256,
        embodiment_dim: int = 256,
        diffusion_step_embed_dim: int = 256,
        hidden_dim: int = 512,
        depth: int = 14,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        cross_attn_layers: Sequence[int] = (2, 5, 8, 11),
        num_inference_steps: int = 32,
        solver: str = "heun",
    ) -> None:
        super().__init__()
        if num_inference_steps < 1:
            raise ValueError("num_inference_steps must be positive")
        solver = str(solver).lower()
        if solver not in {"euler", "heun"}:
            raise ValueError("solver must be 'euler' or 'heun'")
        self.latent_dim = int(latent_dim)
        self.latent_horizon = int(latent_horizon)
        self.num_inference_steps = int(num_inference_steps)
        self.solver = solver
        self.model = LatentDiT(
            latent_dim=self.latent_dim,
            latent_horizon=self.latent_horizon,
            visual_dim=visual_dim,
            embodiment_dim=embodiment_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            hidden_dim=hidden_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            cross_attn_layers=cross_attn_layers,
        )

    def predict_velocity(
        self,
        noisy_latent: torch.Tensor,
        flow_time: torch.Tensor,
        *,
        visual_tokens: torch.Tensor | None,
        embodiment_tokens: torch.Tensor | None,
        latent_valid: torch.Tensor | None = None,
        visual_valid: torch.Tensor | None = None,
        embodiment_valid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.model(
            noisy_latent,
            flow_time,
            visual_tokens=visual_tokens,
            embodiment_tokens=embodiment_tokens,
            latent_valid=latent_valid,
            visual_valid=visual_valid,
            embodiment_valid=embodiment_valid,
        )

    def compute_loss(
        self,
        target_latent: torch.Tensor,
        *,
        visual_tokens: torch.Tensor | None,
        embodiment_tokens: torch.Tensor | None,
        latent_valid: torch.Tensor | None = None,
        visual_valid: torch.Tensor | None = None,
        embodiment_valid: torch.Tensor | None = None,
    ) -> Mapping[str, torch.Tensor]:
        expected = (self.latent_horizon, self.latent_dim)
        if target_latent.ndim != 3 or target_latent.shape[1:] != expected:
            raise ValueError(
                f"target_latent must be [B,{expected[0]},{expected[1]}], "
                f"got {tuple(target_latent.shape)}"
            )
        noise = torch.randn_like(target_latent)
        flow_time = torch.rand(
            target_latent.shape[0],
            device=target_latent.device,
            dtype=target_latent.dtype,
        )
        alpha = flow_time[:, None, None]
        noisy_latent = (1.0 - alpha) * noise + alpha * target_latent
        target_velocity = target_latent - noise
        predicted_velocity = self.predict_velocity(
            noisy_latent,
            flow_time,
            visual_tokens=visual_tokens,
            embodiment_tokens=embodiment_tokens,
            latent_valid=latent_valid,
            visual_valid=visual_valid,
            embodiment_valid=embodiment_valid,
        )
        error = F.mse_loss(
            predicted_velocity, target_velocity, reduction="none"
        )
        if latent_valid is None:
            loss = error.mean()
        else:
            mask = latent_valid[:, :, None].to(error.dtype)
            denominator = mask.sum().clamp_min(1.0) * target_latent.shape[-1]
            loss = (error * mask).sum() / denominator
        return {
            "loss": loss,
            "flow_matching_loss": loss.detach(),
            "flow_time_mean": flow_time.detach().mean(),
        }

    @torch.no_grad()
    def sample(
        self,
        *,
        batch_size: int,
        visual_tokens: torch.Tensor | None,
        embodiment_tokens: torch.Tensor | None,
        latent_valid: torch.Tensor | None = None,
        visual_valid: torch.Tensor | None = None,
        embodiment_valid: torch.Tensor | None = None,
        num_inference_steps: int | None = None,
        solver: str | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        reference = visual_tokens if visual_tokens is not None else embodiment_tokens
        if reference is None:
            raise ValueError("at least one condition group is required for sampling")
        steps = (
            self.num_inference_steps
            if num_inference_steps is None
            else int(num_inference_steps)
        )
        if steps < 1:
            raise ValueError("num_inference_steps must be positive")
        selected_solver = self.solver if solver is None else str(solver).lower()
        if selected_solver not in {"euler", "heun"}:
            raise ValueError("solver must be 'euler' or 'heun'")

        trajectory = torch.randn(
            batch_size,
            self.latent_horizon,
            self.latent_dim,
            device=reference.device,
            dtype=reference.dtype,
            generator=generator,
        )
        times = torch.linspace(
            0.0, 1.0, steps + 1, device=reference.device, dtype=reference.dtype
        )
        for step_idx in range(steps):
            t0, t1 = times[step_idx], times[step_idx + 1]
            dt = t1 - t0
            velocity = self.predict_velocity(
                trajectory,
                t0.expand(batch_size),
                visual_tokens=visual_tokens,
                embodiment_tokens=embodiment_tokens,
                latent_valid=latent_valid,
                visual_valid=visual_valid,
                embodiment_valid=embodiment_valid,
            )
            if selected_solver == "heun" and step_idx < steps - 1:
                euler = trajectory + dt * velocity
                velocity_next = self.predict_velocity(
                    euler,
                    t1.expand(batch_size),
                    visual_tokens=visual_tokens,
                    embodiment_tokens=embodiment_tokens,
                    latent_valid=latent_valid,
                    visual_valid=visual_valid,
                    embodiment_valid=embodiment_valid,
                )
                trajectory = trajectory + 0.5 * dt * (velocity + velocity_next)
            else:
                trajectory = trajectory + dt * velocity
        return trajectory
