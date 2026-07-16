from __future__ import annotations

from typing import Any, Dict, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.diffusion.conditional_unet1d import ConditionalUnet1D
from models.diffusion.mask_generator import LowdimMaskGenerator

from .action_dit import ActionDiT
from .condition_encoder import ConditionEncoder


class FlowMatchingPolicy(nn.Module):
    def __init__(
        self,
        *,
        action_dim: int,
        state_dim: int,
        cond_steps: int = 8,
        cond_dim: int = 256,
        use_tactile: bool = True,
        tactile_channels: int = 12,
        action_horizon: int = 32,
        n_action_steps: int = 32,
        image_encoder_name: str = "dinov2",
        dino_model_name: str = "vit_small_patch14_dinov2.lvd142m",
        freeze_image_encoder: bool = True,
        image_pretrained: bool = True,
        image_feat_dim: int = 256,
        n_image_views: int = 3,
        tactile_feat_dim: int = 256,
        tactile_temporal_pool: str = "conv1d",
        state_feat_dim: int = 256,
        state_pool: str = "flatten",
        diffusion_step_embed_dim: int = 256,
        down_dims=(256, 512, 1024),
        kernel_size: int = 5,
        n_groups: int = 8,
        velocity_model: str = "unet",
        dit_hidden_dim: int = 512,
        dit_depth: int = 14,
        dit_num_heads: int = 8,
        dit_mlp_ratio: float = 4.0,
        dit_dropout: float = 0.1,
        num_inference_steps: int = 16,
        solver: str = "euler",
    ):
        super().__init__()
        self.action_dim = int(action_dim)
        self.state_dim = int(state_dim)
        self.cond_steps = int(cond_steps)
        self.cond_dim = int(cond_dim)
        self.use_tactile = bool(use_tactile)
        self.action_horizon = int(action_horizon)
        self.n_action_steps = int(n_action_steps)
        self.num_inference_steps = int(num_inference_steps)
        self.solver = str(solver).lower()
        if self.solver not in {"euler", "heun"}:
            raise ValueError(f"unsupported solver={solver!r}")

        self.condition_encoder = ConditionEncoder(
            state_dim=self.state_dim,
            cond_dim=self.cond_dim,
            cond_steps=self.cond_steps,
            use_tactile=self.use_tactile,
            tactile_channels=tactile_channels,
            image_encoder_name=image_encoder_name,
            dino_model_name=dino_model_name,
            freeze_image_encoder=freeze_image_encoder,
            image_pretrained=image_pretrained,
            image_feat_dim=image_feat_dim,
            n_image_views=n_image_views,
            tactile_feat_dim=tactile_feat_dim,
            tactile_temporal_pool=tactile_temporal_pool,
            state_feat_dim=state_feat_dim,
            state_pool=state_pool,
        )

        self.velocity_model = str(velocity_model).lower()
        if self.velocity_model == "unet":
            self.model = ConditionalUnet1D(
                input_dim=self.action_dim,
                local_cond_dim=None,
                global_cond_dim=self.cond_dim,
                diffusion_step_embed_dim=diffusion_step_embed_dim,
                down_dims=tuple(down_dims),
                kernel_size=kernel_size,
                n_groups=n_groups,
                cond_predict_scale=True,
            )
        elif self.velocity_model == "dit":
            self.model = ActionDiT(
                input_dim=self.action_dim,
                action_horizon=self.action_horizon,
                global_cond_dim=self.cond_dim,
                diffusion_step_embed_dim=diffusion_step_embed_dim,
                hidden_dim=dit_hidden_dim,
                depth=dit_depth,
                num_heads=dit_num_heads,
                mlp_ratio=dit_mlp_ratio,
                dropout=dit_dropout,
            )
        else:
            raise ValueError(f"unsupported velocity_model={velocity_model!r}")

        self.mask_generator = LowdimMaskGenerator(
            action_dim=self.action_dim,
            obs_dim=0,
            max_n_obs_steps=1,
            fix_obs_steps=True,
            action_visible=False,
        )

    @classmethod
    def from_config(
        cls,
        cfg: Mapping[str, Any],
        *,
        action_dim: int,
        state_dim: int,
        cond_steps: int,
        tactile_channels: int = 12,
    ) -> "FlowMatchingPolicy":
        fm_cfg = dict(cfg.get("models", {}).get("fm", cfg))
        return cls(
            action_dim=action_dim,
            state_dim=state_dim,
            cond_steps=cond_steps,
            tactile_channels=tactile_channels,
            **fm_cfg,
        )

    @staticmethod
    def _pad_or_trim_time(x: torch.Tensor, target_t: int) -> torch.Tensor:
        t = x.shape[1]
        if t == target_t:
            return x
        if t > target_t:
            return x[:, :target_t]
        pad = x[:, -1:].expand(-1, target_t - t, -1)
        return torch.cat([x, pad], dim=1)

    def _build_condition(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        image = obs.get("image")
        image_backbone_feat = obs.get("image_backbone_feat")
        state = obs["state"]
        tactile = obs.get("tactile") if self.use_tactile else None
        return self.condition_encoder(
            state=state,
            image=image,
            image_backbone_feat=image_backbone_feat,
            tactile=tactile,
        )

    def compute_loss(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        obs = batch["obs"]
        actions = self._pad_or_trim_time(batch["action"], self.action_horizon)
        if actions.shape[-1] != self.action_dim:
            raise ValueError(
                f"action dim mismatch: got {actions.shape[-1]}, expected {self.action_dim}"
            )

        global_cond = self._build_condition(obs)

        condition_mask = self.mask_generator(actions.shape)
        loss_mask = ~condition_mask

        x1 = actions
        x0 = torch.randn_like(x1)
        bsz = actions.shape[0]
        t = torch.rand(bsz, device=actions.device, dtype=actions.dtype)
        t_broadcast = t.view(bsz, 1, 1)
        xt = (1.0 - t_broadcast) * x0 + t_broadcast * x1
        target_velocity = x1 - x0
        xt = torch.where(condition_mask, x1, xt)

        pred_velocity = self.model(xt, t, local_cond=None, global_cond=global_cond)
        loss = F.mse_loss(pred_velocity, target_velocity, reduction="none")
        loss = loss * loss_mask.to(loss.dtype)
        loss = loss.reshape(loss.shape[0], -1).mean(dim=1).mean()

        return {
            "loss": loss,
            "metrics": {"flow_matching_loss": loss.detach()},
        }

    @torch.no_grad()
    def conditional_sample(
        self,
        obs: Dict[str, torch.Tensor],
        num_inference_steps: int | None = None,
        solver: str | None = None,
    ) -> torch.Tensor:
        global_cond = self._build_condition(obs)
        steps = self.num_inference_steps if num_inference_steps is None else int(num_inference_steps)
        solver = self.solver if solver is None else str(solver).lower()

        bsz = global_cond.shape[0]
        device = global_cond.device
        dtype = global_cond.dtype
        trajectory = torch.randn(
            bsz,
            self.action_horizon,
            self.action_dim,
            device=device,
            dtype=dtype,
        )
        times = torch.linspace(0.0, 1.0, steps + 1, device=device, dtype=dtype)

        for i in range(steps):
            t0 = times[i]
            t1 = times[i + 1]
            dt = t1 - t0
            t_batch = t0.expand(bsz)
            velocity = self.model(trajectory, t_batch, local_cond=None, global_cond=global_cond)
            if solver == "heun" and i < steps - 1:
                x_euler = trajectory + dt * velocity
                t_batch_next = t1.expand(bsz)
                velocity_next = self.model(
                    x_euler, t_batch_next, local_cond=None, global_cond=global_cond
                )
                trajectory = trajectory + 0.5 * dt * (velocity + velocity_next)
            else:
                trajectory = trajectory + dt * velocity
        return trajectory

    @torch.no_grad()
    def predict_action(
        self,
        obs: Dict[str, torch.Tensor],
        num_inference_steps: int | None = None,
        solver: str | None = None,
    ) -> Dict[str, torch.Tensor]:
        action_norm = self.conditional_sample(
            obs=obs,
            num_inference_steps=num_inference_steps,
            solver=solver,
        )
        return {
            "action_normalized": action_norm[:, : self.n_action_steps],
            "action_pred_normalized": action_norm,
        }

    def forward(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        return self.compute_loss(batch)


def build_flow_policy(
    cfg: Mapping[str, Any],
    *,
    action_dim: int,
    state_dim: int,
    cond_steps: int,
    tactile_channels: int = 12,
) -> FlowMatchingPolicy:
    return FlowMatchingPolicy.from_config(
        cfg,
        action_dim=action_dim,
        state_dim=state_dim,
        cond_steps=cond_steps,
        tactile_channels=tactile_channels,
    )
