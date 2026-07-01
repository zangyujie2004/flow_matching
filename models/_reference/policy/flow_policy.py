from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.diffusion.conditional_unet1d import ConditionalUnet1D
from models.diffusion.mask_generator import LowdimMaskGenerator
from models.dynamics import BaseDynamics
from models.policy.condition_encoder import ConditionEncoder
from tools.action_util import relative_actions_to_absolute_tensor
from utils.normalizer import MultiFieldNormalizer


class FlowMatchingPolicy(nn.Module):
    def __init__(
        self,
        dynamics: Optional[BaseDynamics] = None,
        latent_dim: Optional[int] = None,
        action_dim: int = 10,
        force_dim: int = 6,
        state_dim: int = 9,
        cond_dim: int = 256,
        curr_steps: int = 1,
        future_steps: int = 4,
        n_action_steps: int = 8,
        action_horizon: Optional[int] = None,
        action_representation: str = "absolute",
        down_dims=(256, 512, 1024),
        diffusion_step_embed_dim: int = 256,
        kernel_size: int = 5,
        n_groups: int = 8,
        image_encoder_name: str = "dinov2_small",
        freeze_image_encoder: bool = True,
        image_pretrained: bool = True,
        dino_model_name: str = "vit_small_patch14_dinov2.lvd142m",
        tactile_future_slice: str = "front",
        tactile_cross_heads: int = 2,
    ):
        super().__init__()

        self.action_dim = int(action_dim)
        self.force_dim = int(force_dim)
        self.state_dim = int(state_dim)
        self.cond_dim = int(cond_dim)
        self.curr_steps = int(curr_steps)
        self.future_steps = int(future_steps)
        self.n_action_steps = int(n_action_steps)
        self.action_horizon = int(action_horizon if action_horizon is not None else n_action_steps)
        self.action_representation = str(action_representation).lower()
        self.tactile_future_slice = str(tactile_future_slice).lower()
        if self.curr_steps < 1:
            raise ValueError(f"curr_steps must be >=1, got {self.curr_steps}")
        if self.future_steps < 1:
            raise ValueError(f"future_steps must be >=1, got {self.future_steps}")
        if self.action_horizon < 1:
            raise ValueError(f"action_horizon must be >=1, got {self.action_horizon}")
        if self.n_action_steps < 1 or self.n_action_steps > self.action_horizon:
            raise ValueError(
                f"n_action_steps must be in [1, action_horizon], got n_action_steps={self.n_action_steps}, "
                f"action_horizon={self.action_horizon}"
            )
        if self.action_representation not in {"absolute", "chunk_relative"}:
            raise ValueError(
                f"Unsupported action_representation={action_representation}. "
                "Choose from ['absolute', 'chunk_relative']."
            )
        if self.tactile_future_slice not in {"front", "back"}:
            raise ValueError(
                f"Unsupported tactile_future_slice={self.tactile_future_slice}. Choose from ['front', 'back']."
            )
        self.tactile_cross_heads = int(tactile_cross_heads)

        self.dynamics = dynamics
        if self.dynamics is not None:
            self.dynamics.eval()
            for p in self.dynamics.parameters():
                p.requires_grad = False
            latent_dim = self.dynamics.latent_dim
        elif latent_dim is None:
            raise ValueError("latent_dim must be provided when dynamics is None.")

        self.condition_encoder = ConditionEncoder(
            latent_dim=int(latent_dim),
            force_dim=self.force_dim,
            state_dim=self.state_dim,
            cond_dim=self.cond_dim,
            image_encoder_name=image_encoder_name,
            freeze_image_encoder=freeze_image_encoder,
            image_pretrained=image_pretrained,
            dino_model_name=dino_model_name,
            cond_steps=self.curr_steps,
            tactile_cross_heads=self.tactile_cross_heads,
        )

        self.model = ConditionalUnet1D(
            input_dim=self.action_dim,
            local_cond_dim=None,
            global_cond_dim=self.cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=list(down_dims),
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=True,
        )

        self.mask_generator = LowdimMaskGenerator(
            action_dim=self.action_dim,
            obs_dim=0,
            max_n_obs_steps=1,
            fix_obs_steps=True,
            action_visible=False,
        )

        self.normalizer = MultiFieldNormalizer()

    def set_normalizer(self, normalizer: MultiFieldNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    @staticmethod
    def _pad_or_trim_time(x: torch.Tensor, target_t: int) -> torch.Tensor:
        t = x.shape[1]
        if t == target_t:
            return x
        if t > target_t:
            return x[:, :target_t]
        pad = x[:, -1:].expand(-1, target_t - t, *x.shape[2:])
        return torch.cat([x, pad], dim=1)

    def _normalize_obs(self, obs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if len(self.normalizer.fields) == 0:
            return obs
        return self.normalizer.normalize_obs(obs)

    def _normalize_action(self, action: torch.Tensor) -> torch.Tensor:
        if "action" in self.normalizer:
            return self.normalizer["action"].normalize(action)
        return action

    def _unnormalize_action(self, action: torch.Tensor) -> torch.Tensor:
        if "action" in self.normalizer:
            action = self.normalizer["action"].unnormalize(action)
        return action

    def _action_reference(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor | None:
        if self.action_representation == "absolute":
            return None
        if "state" not in obs:
            raise KeyError("chunk_relative action representation requires obs['state'].")
        state = obs["state"]
        if state.shape[-1] < self.action_dim:
            raise ValueError(
                "chunk_relative action representation requires state_dim >= action_dim, "
                f"got state_dim={state.shape[-1]}, action_dim={self.action_dim}"
            )
        return state[:, -1, : self.action_dim]

    def _action_to_absolute(self, action: torch.Tensor, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        ref = self._action_reference(obs)
        if ref is None:
            return action
        base_absolute_action = obs["state"][:, -1, : self.action_dim]
        base = base_absolute_action.unsqueeze(1).expand(-1, action.shape[1], -1)
        return relative_actions_to_absolute_tensor(action, base)

    def _build_condition(
        self,
        obs_raw: Dict[str, torch.Tensor],
        obs_norm: Dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if "tactile_latent_curr" in obs_raw and "tactile_latent_future" in obs_raw:
            tactile_curr = obs_raw["tactile_latent_curr"]
            tactile_fut = obs_raw["tactile_latent_future"]
        else:
            if self.dynamics is None:
                raise KeyError(
                    "Batch is missing cached tactile latents and no dynamics module is available."
                )
            with torch.no_grad():
                dynamics_out = self.dynamics(obs_raw)
            tactile_curr = dynamics_out["tactile_latent_curr"]
            tactile_fut = dynamics_out["tactile_latent_future"]

        image = obs_norm.get("image")
        image_backbone_feat = obs_raw.get("image_backbone_feat")
        force = obs_norm["force"]
        state = obs_norm["state"]
        if (image is None) == (image_backbone_feat is None):
            raise ValueError("Expected exactly one of image or image_backbone_feat.")

        curr_lengths = [
            tactile_curr.shape[1],
            force.shape[1],
            state.shape[1],
        ]
        image_avail = image_backbone_feat.shape[1] if image_backbone_feat is not None else image.shape[1]

        curr_avail = min(curr_lengths)
        if curr_avail < 1:
            raise ValueError("Current condition sequence is empty.")
        fut_avail = tactile_fut.shape[1]
        if fut_avail < 1:
            raise ValueError("Future tactile sequence is empty.")

        t_curr = min(self.curr_steps, curr_avail)
        t_img = min(self.curr_steps, image_avail)
        t_fut = min(self.future_steps, fut_avail)

        tactile_curr_sel = tactile_curr[:, -t_curr:]
        force_sel = force[:, -t_curr:]
        state_sel = state[:, -t_curr:]
        if image_backbone_feat is not None:
            image_backbone_feat_sel = image_backbone_feat[:, -t_img:]
            image_sel = None
        else:
            image_sel = image[:, -t_img:]
            image_backbone_feat_sel = None

        if self.tactile_future_slice == "back":
            tactile_fut_sel = tactile_fut[:, -t_fut:]
        else:
            tactile_fut_sel = tactile_fut[:, :t_fut]

        global_cond, local_cond = self.condition_encoder(
            image=image_sel,
            image_backbone_feat=image_backbone_feat_sel,
            tactile_latent_curr=tactile_curr_sel,
            tactile_latent_future=tactile_fut_sel,
            force=force_sel,
            state=state_sel,
        )
        return global_cond, local_cond

    def compute_loss(self, batch: Dict[str, Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        obs_raw = batch["obs"]
        obs_norm = self._normalize_obs(obs_raw)

        nactions = self._normalize_action(batch["action"])
        nactions = self._pad_or_trim_time(nactions, self.action_horizon)

        global_cond, local_cond = self._build_condition(obs_raw=obs_raw, obs_norm=obs_norm)

        trajectory = nactions
        cond_data = trajectory

        if trajectory.shape[-1] != self.action_dim:
            raise ValueError(
                "Action dim mismatch before mask generation: "
                f"trajectory D={trajectory.shape[-1]}, policy.action_dim={self.action_dim}. "
                "Please align config data.action_dim and policy.action_dim with the dataset action tensor shape."
            )

        condition_mask = self.mask_generator(trajectory.shape)
        loss_mask = ~condition_mask

        x1 = trajectory
        x0 = torch.randn_like(x1)

        bsz = trajectory.shape[0]
        t = torch.rand(bsz, device=trajectory.device, dtype=trajectory.dtype)
        t_broadcast = t.view(bsz, 1, 1)

        xt = (1.0 - t_broadcast) * x0 + t_broadcast * x1
        target_velocity = x1 - x0

        xt = torch.where(condition_mask, cond_data, xt)

        pred_velocity = self.model(
            xt,
            t,
            local_cond=local_cond,
            global_cond=global_cond,
        )

        loss = F.mse_loss(pred_velocity, target_velocity, reduction="none")
        loss = loss * loss_mask.to(loss.dtype)
        loss = loss.reshape(loss.shape[0], -1).mean(dim=1).mean()

        return {
            "loss": loss,
            "metrics": {
                "flow_matching_loss": loss.detach(),
            },
        }

    @torch.no_grad()
    def conditional_sample(
        self,
        obs: Dict[str, torch.Tensor],
        num_inference_steps: int = 16,
        solver: str = "euler",
    ) -> torch.Tensor:
        obs_norm = self._normalize_obs(obs)
        global_cond, local_cond = self._build_condition(obs_raw=obs, obs_norm=obs_norm)

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

        times = torch.linspace(0.0, 1.0, num_inference_steps + 1, device=device, dtype=dtype)

        for i in range(num_inference_steps):
            t0 = times[i]
            t1 = times[i + 1]
            dt = t1 - t0
            t_batch = t0.expand(bsz)

            velocity = self.model(trajectory, t_batch, local_cond=local_cond, global_cond=global_cond)

            if solver.lower() == "heun" and i < num_inference_steps - 1:
                x_euler = trajectory + dt * velocity
                t_batch_next = t1.expand(bsz)
                velocity_next = self.model(
                    x_euler,
                    t_batch_next,
                    local_cond=local_cond,
                    global_cond=global_cond,
                )
                trajectory = trajectory + 0.5 * dt * (velocity + velocity_next)
            else:
                trajectory = trajectory + dt * velocity

        return trajectory

    @torch.no_grad()
    def predict_action(
        self,
        obs: Dict[str, torch.Tensor],
        num_inference_steps: int = 16,
        solver: str = "euler",
    ) -> Dict[str, torch.Tensor]:
        action_norm = self.conditional_sample(
            obs=obs,
            num_inference_steps=num_inference_steps,
            solver=solver,
        )
        action_pred_model = self._unnormalize_action(action_norm)
        action_model = action_pred_model[:, :self.n_action_steps]
        action_pred = self._action_to_absolute(action_pred_model, obs)
        action = action_pred[:, :self.n_action_steps]
        return {
            "action": action,
            "action_model": action_model,
            "action_pred": action_pred,
            "action_pred_model": action_pred_model,
            "action_normalized": action_norm[:, :self.n_action_steps],
            "action_pred_normalized": action_norm,
        }

    def forward(self, batch: Dict[str, Dict[str, torch.Tensor]]):
        return self.compute_loss(batch)
