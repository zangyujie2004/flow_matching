from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.diffusion.conditional_unet1d import ConditionalUnet1D
from models.diffusion.mask_generator import LowdimMaskGenerator

from .action_dit import ActionDiT
from .condition_encoder import ConditionEncoder
from .memory import build_memory_encoder


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
        memory_enabled: bool = False,
        memory_method: str = "fusion",
        memory_injection: str = "cross_attn",
        memory_dim: int = 256,
        memory_history_frames: int = 64,
        memory_recent_frame: int = 2,
        memory_visual_history_length: int = 64,
        memory_visual_sample_stride: int = 8,
        memory_visual_recent_frame: int = 0,
        memory_visual_layers: int = 2,
        memory_visual_heads: int = 4,
        memory_state_channels: int = 128,
        memory_state_layers: int = 2,
        memory_state_mem_dim: int = 64,
        memory_num_queries: int = 3,
        memory_state_hidden_dim: int = 64,
        memory_state_heads: int = 4,
        memory_dropout: float = 0.1,
        memory_cross_attn_layers: Sequence[int] = (3, 7, 11),
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
        self.memory_enabled = bool(memory_enabled)
        self.memory_method = str(memory_method)
        self.memory_history_frames = int(memory_history_frames)
        self.memory_recent_frame = int(memory_recent_frame)
        self.memory_visual_history_length = int(memory_visual_history_length)
        self.memory_visual_sample_stride = int(memory_visual_sample_stride)
        self.memory_visual_recent_frame = int(memory_visual_recent_frame)
        if self.memory_visual_history_length < 1:
            raise ValueError("memory_visual_history_length must be positive")
        if self.memory_visual_sample_stride < 1:
            raise ValueError("memory_visual_sample_stride must be positive")
        if self.memory_visual_recent_frame < 0:
            raise ValueError("memory_visual_recent_frame must be non-negative")
        self.memory_injection = str(memory_injection).lower()
        if self.memory_injection not in {"cross_attn", "concat_global_cond"}:
            raise ValueError(
                "memory_injection must be one of ['cross_attn', 'concat_global_cond'], "
                f"got {memory_injection!r}"
            )
        self.velocity_model = str(velocity_model).lower()
        if self.memory_enabled and self.memory_injection == "cross_attn" and self.velocity_model != "dit":
            raise ValueError(
                "memory_injection='cross_attn' requires velocity_model='dit'. "
                "Use memory_injection='concat_global_cond' for UNet."
            )
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

        self.memory_encoder = None
        self.memory_token_proj = None
        self.memory_cond_fusion = None
        if self.memory_enabled:
            self.memory_encoder = build_memory_encoder(
                self.memory_method,
                state_dim=self.state_dim,
                visual_dim=image_feat_dim,
                memory_dim=int(memory_dim),
                history_frames=self.memory_history_frames,
                recent_frame=self.memory_recent_frame,
                max_visual_time_offset=(
                    self.memory_visual_recent_frame
                    + self.memory_visual_sample_stride
                    * (self.memory_visual_history_length - 1)
                ),
                visual_layers=memory_visual_layers,
                visual_heads=memory_visual_heads,
                state_channels=memory_state_channels,
                state_layers=memory_state_layers,
                state_mem_dim=memory_state_mem_dim,
                num_queries=memory_num_queries,
                state_hidden_dim=memory_state_hidden_dim,
                state_heads=memory_state_heads,
                n_views=n_image_views,
                dropout=memory_dropout,
            )
            self.memory_token_proj = (
                nn.Identity()
                if int(memory_dim) == self.cond_dim
                else nn.Linear(int(memory_dim), self.cond_dim)
            )
            if self.memory_injection == "concat_global_cond":
                self.memory_cond_fusion = nn.Sequential(
                    nn.LayerNorm(self.cond_dim * 2),
                    nn.Linear(self.cond_dim * 2, self.cond_dim),
                    nn.SiLU(),
                    nn.Dropout(memory_dropout),
                    nn.Linear(self.cond_dim, self.cond_dim),
                )

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
                condition_token_dim=(
                    self.cond_dim
                    if self.memory_enabled and self.memory_injection == "cross_attn"
                    else None
                ),
                cross_attn_layers=(
                    tuple(int(x) for x in memory_cross_attn_layers)
                    if self.memory_enabled and self.memory_injection == "cross_attn"
                    else None
                ),
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
        models = cfg.get("models", {})
        if not isinstance(models, Mapping):
            models = {}
        fm_cfg = dict(models.get("fm", cfg))
        mem_cfg = dict(models.get("memory") or {})
        data = cfg.get("data", {})
        if not isinstance(data, Mapping):
            data = {}
        data_mem = dict(data.get("memory") or {})

        kwargs = dict(fm_cfg)
        if bool(data_mem.get("enabled", False)):
            kwargs["memory_enabled"] = True
            kwargs["memory_method"] = str(mem_cfg.get("method", kwargs.get("memory_method", "fusion")))
            kwargs["memory_injection"] = str(
                mem_cfg.get("injection", kwargs.get("memory_injection", "cross_attn"))
            )
            kwargs["memory_dim"] = int(mem_cfg.get("dim", kwargs.get("memory_dim", 256)))
            kwargs["memory_history_frames"] = int(
                data_mem.get("history_frames", kwargs.get("memory_history_frames", 64))
            )
            kwargs["memory_recent_frame"] = int(
                data_mem.get("recent_frame", kwargs.get("memory_recent_frame", 2))
            )
            kwargs["memory_visual_history_length"] = int(
                data_mem.get(
                    "visual_history_length",
                    kwargs.get("memory_visual_history_length", 64),
                )
            )
            kwargs["memory_visual_sample_stride"] = int(
                data_mem.get(
                    "sample_stride",
                    kwargs.get("memory_visual_sample_stride", 8),
                )
            )
            kwargs["memory_visual_recent_frame"] = int(
                data_mem.get(
                    "visual_recent_frame",
                    kwargs.get("memory_visual_recent_frame", 0),
                )
            )
            kwargs["memory_visual_layers"] = int(
                mem_cfg.get("visual_layers", kwargs.get("memory_visual_layers", 2))
            )
            kwargs["memory_visual_heads"] = int(
                mem_cfg.get("visual_heads", kwargs.get("memory_visual_heads", 4))
            )
            kwargs["memory_state_channels"] = int(
                mem_cfg.get("state_channels", kwargs.get("memory_state_channels", 128))
            )
            kwargs["memory_state_layers"] = int(
                mem_cfg.get("state_layers", kwargs.get("memory_state_layers", 2))
            )
            kwargs["memory_state_mem_dim"] = int(
                mem_cfg.get("state_mem_dim", kwargs.get("memory_state_mem_dim", 64))
            )
            kwargs["memory_num_queries"] = int(
                mem_cfg.get("num_queries", kwargs.get("memory_num_queries", 3))
            )
            kwargs["memory_state_hidden_dim"] = int(
                mem_cfg.get("state_hidden_dim", kwargs.get("memory_state_hidden_dim", 64))
            )
            kwargs["memory_state_heads"] = int(
                mem_cfg.get("state_heads", kwargs.get("memory_state_heads", 4))
            )
            kwargs["memory_dropout"] = float(mem_cfg.get("dropout", kwargs.get("memory_dropout", 0.1)))
            if "cross_attn_layers" in mem_cfg:
                kwargs["memory_cross_attn_layers"] = tuple(int(x) for x in mem_cfg["cross_attn_layers"])

        return cls(
            action_dim=action_dim,
            state_dim=state_dim,
            cond_steps=cond_steps,
            tactile_channels=tactile_channels,
            **kwargs,
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

    def _build_obs_condition(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
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

    def _build_memory(self, obs: Dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        if self.memory_encoder is None or self.memory_token_proj is None:
            raise RuntimeError("memory encoder is not configured")
        required = ("memory_state", "memory_visual_offsets")
        missing = [key for key in required if key not in obs]
        if "memory_visual_tokens" not in obs and "memory_image_backbone_feat" not in obs:
            missing.append("memory_visual_tokens or memory_image_backbone_feat")
        if missing:
            raise KeyError(
                "memory is enabled but obs is missing required keys: " + ", ".join(missing)
            )
        visual_tokens = obs.get("memory_visual_tokens")
        num_views = None
        if visual_tokens is None:
            backbone_feat = obs["memory_image_backbone_feat"]
            if self.memory_method == "fusion":
                if backbone_feat.ndim != 4:
                    raise ValueError(
                        f"memory backbone features must be (B,T,V,C), got {backbone_feat.shape}"
                    )
                num_views = int(backbone_feat.shape[2])
                visual_tokens = (
                    self.condition_encoder.image_encoder
                    .project_view_histories_from_backbone_feat(backbone_feat)
                )
            else:
                visual_tokens = self.condition_encoder.encode_image_sequence_from_backbone_feat(
                    backbone_feat
                )
        memory_kwargs = {"num_views": num_views} if self.memory_method == "fusion" else {}
        mem_out = self.memory_encoder(
            visual_tokens=visual_tokens,
            visual_offsets=obs["memory_visual_offsets"],
            state=obs["memory_state"],
            visual_valid=obs.get("memory_visual_valid"),
            state_valid=obs.get("memory_state_valid"),
            **memory_kwargs,
        )
        tokens = self.memory_token_proj(mem_out.tokens)
        memory_global = self.memory_token_proj(mem_out.memory_global)
        return tokens, memory_global

    def _build_condition(
        self,
        obs: Dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        obs_cond = self._build_obs_condition(obs)
        if not self.memory_enabled:
            return obs_cond, None
        tokens, memory_global = self._build_memory(obs)
        if self.memory_injection == "cross_attn":
            # Locked: memory only in condition_tokens; obs stays in global_cond / AdaLN.
            return obs_cond, tokens
        if self.memory_cond_fusion is None:
            raise RuntimeError("memory_cond_fusion is required for concat_global_cond mode")
        global_cond = self.memory_cond_fusion(torch.cat([obs_cond, memory_global], dim=-1))
        return global_cond, None

    def _model_forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        *,
        global_cond: torch.Tensor,
        condition_tokens: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.velocity_model == "dit":
            return self.model(
                sample,
                timestep,
                local_cond=None,
                global_cond=global_cond,
                condition_tokens=condition_tokens,
            )
        return self.model(sample, timestep, local_cond=None, global_cond=global_cond)

    def compute_loss(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        obs = batch["obs"]
        actions = self._pad_or_trim_time(batch["action"], self.action_horizon)
        if actions.shape[-1] != self.action_dim:
            raise ValueError(
                f"action dim mismatch: got {actions.shape[-1]}, expected {self.action_dim}"
            )

        global_cond, condition_tokens = self._build_condition(obs)

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

        pred_velocity = self._model_forward(
            xt,
            t,
            global_cond=global_cond,
            condition_tokens=condition_tokens,
        )
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
        global_cond, condition_tokens = self._build_condition(obs)
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
            velocity = self._model_forward(
                trajectory,
                t_batch,
                global_cond=global_cond,
                condition_tokens=condition_tokens,
            )
            if solver == "heun" and i < steps - 1:
                x_euler = trajectory + dt * velocity
                t_batch_next = t1.expand(bsz)
                velocity_next = self._model_forward(
                    x_euler,
                    t_batch_next,
                    global_cond=global_cond,
                    condition_tokens=condition_tokens,
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
