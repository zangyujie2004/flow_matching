from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.diffusion.conditional_unet1d import ConditionalUnet1D
from models.diffusion.mask_generator import LowdimMaskGenerator

from .action_dit import ActionDiT
from .condition_encoder import (
    ConditionEncoder,
    resolve_tactile_condition_encoder_type,
)
from .encoders.tactile_autoencoder import (
    TactileAutoencoder,
    build_tactile_autoencoder,
    load_tactile_autoencoder_checkpoint,
)
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
        view_pool: str = "global_concat",
        local_pool_size: int = 2,
        local_attn_heads: int = 4,
        local_attn_dropout: float = 0.0,
        tactile_encoder_type: str = "temporal_cnn",
        tactile_condition_encoder_type: str | None = None,
        tactile_feat_dim: int = 256,
        tactile_temporal_pool: str = "conv1d",
        tactile_num_sensors: int = 4,
        tactile_channels_per_sensor: int = 3,
        tactile_token_dim: int = 16,
        predict_tactile: bool = False,
        tactile_ae_checkpoint: str | None = None,
        tactile_ae_model_config: Mapping[str, Any] | None = None,
        action_loss_weight: float = 1.0,
        tactile_loss_weight: float = 1.0,
        state_feat_dim: int = 256,
        state_pool: str = "flatten",
        fusion_hidden_dim: int = 512,
        dropout: float = 0.1,
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
        memory_history_frames: int = 128,
        memory_recent_frame: int = 0,
        memory_visual_history_length: int = 128,
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
        self.predict_tactile = bool(predict_tactile)
        if self.predict_tactile and not self.use_tactile:
            raise ValueError("predict_tactile=true requires use_tactile=true")
        self.tactile_latent_dim = (
            int(tactile_num_sensors) * int(tactile_token_dim)
        )
        self.tactile_condition_encoder_type = (
            resolve_tactile_condition_encoder_type(
                predict_tactile=self.predict_tactile,
                tactile_encoder_type=tactile_encoder_type,
                tactile_condition_encoder_type=tactile_condition_encoder_type,
            )
        )
        self.trajectory_dim = self.action_dim + (
            self.tactile_latent_dim if self.predict_tactile else 0
        )
        self.action_loss_weight = float(action_loss_weight)
        self.tactile_loss_weight = float(tactile_loss_weight)
        if self.action_loss_weight <= 0:
            raise ValueError("action_loss_weight must be positive")
        if self.predict_tactile and self.tactile_loss_weight <= 0:
            raise ValueError("tactile_loss_weight must be positive")
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
        if self.memory_enabled and self.memory_history_frames != self.memory_visual_history_length:
            raise ValueError(
                "state and visual Memory must share history_length: "
                f"{self.memory_history_frames} != {self.memory_visual_history_length}"
            )
        if self.memory_enabled and self.memory_recent_frame != self.memory_visual_recent_frame:
            raise ValueError(
                "state and visual Memory must share the same anchor/recent_frame: "
                f"{self.memory_recent_frame} != {self.memory_visual_recent_frame}"
            )
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

        self.tactile_autoencoder: TactileAutoencoder | None = None
        if self.predict_tactile:
            checkpoint_exists = bool(
                tactile_ae_checkpoint
                and Path(tactile_ae_checkpoint).expanduser().is_file()
            )
            if checkpoint_exists:
                codec, _ = load_tactile_autoencoder_checkpoint(
                    str(tactile_ae_checkpoint)
                )
            elif isinstance(tactile_ae_model_config, Mapping):
                codec = build_tactile_autoencoder(tactile_ae_model_config)
            else:
                raise ValueError(
                    "predict_tactile=true requires an existing "
                    "models.fm.tactile_ae_checkpoint during Stage 2 training; "
                    "a saved Stage 2 config may instead provide "
                    "tactile_ae_model_config for self-contained loading"
                )
            if codec.num_sensors != int(tactile_num_sensors):
                raise ValueError(
                    f"AE num_sensors={codec.num_sensors} != "
                    f"config={tactile_num_sensors}"
                )
            if codec.token_dim != int(tactile_token_dim):
                raise ValueError(
                    f"AE token_dim={codec.token_dim} != config={tactile_token_dim}"
                )
            if codec.channels_per_sensor != int(tactile_channels_per_sensor):
                raise ValueError(
                    "AE channels_per_sensor="
                    f"{codec.channels_per_sensor} != "
                    f"config={tactile_channels_per_sensor}"
                )
            codec.requires_grad_(False)
            codec.eval()
            self.tactile_autoencoder = codec
            self.register_buffer(
                "tactile_latent_mean",
                torch.zeros(self.tactile_latent_dim, dtype=torch.float32),
            )
            self.register_buffer(
                "tactile_latent_std",
                torch.ones(self.tactile_latent_dim, dtype=torch.float32),
            )

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
            view_pool=view_pool,
            local_pool_size=local_pool_size,
            local_attn_heads=local_attn_heads,
            local_attn_dropout=local_attn_dropout,
            tactile_encoder_type=self.tactile_condition_encoder_type,
            tactile_feat_dim=tactile_feat_dim,
            tactile_temporal_pool=tactile_temporal_pool,
            tactile_num_sensors=tactile_num_sensors,
            tactile_channels_per_sensor=tactile_channels_per_sensor,
            tactile_token_dim=tactile_token_dim,
            state_feat_dim=state_feat_dim,
            state_pool=state_pool,
            fusion_hidden_dim=fusion_hidden_dim,
            dropout=dropout,
        )

        self.memory_encoder = None
        self.memory_token_proj = None
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

        # concat_global_cond keeps [obs_cond ; memory_global] at 2*cond_dim (no
        # compression MLP); every other path stays at cond_dim.
        self.global_cond_dim = self.cond_dim
        if self.memory_enabled and self.memory_injection == "concat_global_cond":
            self.global_cond_dim = self.cond_dim * 2

        if self.velocity_model == "unet":
            self.model = ConditionalUnet1D(
                input_dim=self.trajectory_dim,
                local_cond_dim=None,
                global_cond_dim=self.global_cond_dim,
                diffusion_step_embed_dim=diffusion_step_embed_dim,
                down_dims=tuple(down_dims),
                kernel_size=kernel_size,
                n_groups=n_groups,
                cond_predict_scale=True,
            )
        elif self.velocity_model == "dit":
            self.model = ActionDiT(
                input_dim=self.trajectory_dim,
                action_horizon=self.action_horizon,
                global_cond_dim=self.global_cond_dim,
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
            action_dim=self.trajectory_dim,
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
                data_mem.get("history_frames", kwargs.get("memory_history_frames", 128))
            )
            kwargs["memory_recent_frame"] = int(
                data_mem.get("recent_frame", kwargs.get("memory_recent_frame", 0))
            )
            kwargs["memory_visual_history_length"] = int(
                data_mem.get(
                    "visual_history_length",
                    kwargs.get("memory_visual_history_length", 128),
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

    def train(self, mode: bool = True):
        super().train(mode)
        if self.tactile_autoencoder is not None:
            self.tactile_autoencoder.eval()
        return self

    def set_tactile_latent_stats(
        self,
        mean: torch.Tensor | Sequence[float],
        std: torch.Tensor | Sequence[float],
    ) -> None:
        if not self.predict_tactile:
            raise RuntimeError("tactile latent stats require predict_tactile=true")
        mean_tensor = torch.as_tensor(
            mean,
            dtype=self.tactile_latent_mean.dtype,
            device=self.tactile_latent_mean.device,
        )
        std_tensor = torch.as_tensor(
            std,
            dtype=self.tactile_latent_std.dtype,
            device=self.tactile_latent_std.device,
        )
        expected = (self.tactile_latent_dim,)
        if tuple(mean_tensor.shape) != expected or tuple(std_tensor.shape) != expected:
            raise ValueError(
                f"latent stats must both have shape {expected}, got "
                f"{tuple(mean_tensor.shape)} and {tuple(std_tensor.shape)}"
            )
        if torch.any(std_tensor <= 0):
            raise ValueError("tactile latent std must be strictly positive")
        self.tactile_latent_mean.copy_(mean_tensor)
        self.tactile_latent_std.copy_(std_tensor)

    def _normalize_tactile_latent(self, latent: torch.Tensor) -> torch.Tensor:
        return (latent - self.tactile_latent_mean.to(latent)) / (
            self.tactile_latent_std.to(latent)
        )

    def _unnormalize_tactile_latent(self, latent: torch.Tensor) -> torch.Tensor:
        return latent * self.tactile_latent_std.to(latent) + (
            self.tactile_latent_mean.to(latent)
        )

    def _current_tactile_latent(
        self,
        obs: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        cached = obs.get("tactile_latent")
        if cached is not None:
            if cached.ndim != 2 or cached.shape[-1] != self.tactile_latent_dim:
                raise ValueError(
                    f"expected obs.tactile_latent (B,{self.tactile_latent_dim}), "
                    f"got {tuple(cached.shape)}"
                )
            return cached
        tactile = obs.get("tactile")
        if tactile is None:
            raise KeyError(
                "predict_tactile=true requires obs.tactile_latent or obs.tactile"
            )
        if self.tactile_autoencoder is None:
            raise RuntimeError("tactile autoencoder is not configured")
        with torch.no_grad():
            raw_latent = self.tactile_autoencoder.encode_flattened(tactile)
        return self._normalize_tactile_latent(raw_latent)

    def decode_tactile_latent(
        self,
        normalized_latent: torch.Tensor,
    ) -> torch.Tensor:
        if self.tactile_autoencoder is None:
            raise RuntimeError("tactile autoencoder is not configured")
        if normalized_latent.ndim not in {2, 3}:
            raise ValueError(
                "normalized tactile latent must be (B,64) or (B,T,64), "
                f"got {tuple(normalized_latent.shape)}"
            )
        leading = normalized_latent.shape[:-1]
        flat = normalized_latent.reshape(-1, self.tactile_latent_dim)
        decoded = self.tactile_autoencoder.decode_flattened(
            self._unnormalize_tactile_latent(flat)
        )
        return decoded.reshape(*leading, *decoded.shape[1:])

    def _build_obs_condition(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        image = obs.get("image")
        image_backbone_feat = obs.get("image_backbone_feat")
        state = obs["state"]
        tactile = obs.get("tactile") if self.use_tactile else None
        tactile_latent = None
        if self.use_tactile and self.tactile_condition_encoder_type == "precomputed":
            tactile_latent = self._current_tactile_latent(obs)
        return self.condition_encoder(
            state=state,
            image=image,
            image_backbone_feat=image_backbone_feat,
            tactile=tactile,
            tactile_latent=tactile_latent,
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
        # concat_global_cond: keep the raw [obs_cond ; memory_global] at 2*cond_dim.
        # Order is fixed (obs first, memory second) and must match at train / infer time.
        global_cond = torch.cat([obs_cond, memory_global], dim=-1)
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

        target = actions
        if self.predict_tactile:
            tactile_target = batch.get("tactile_target_latent")
            if tactile_target is None:
                raise KeyError(
                    "predict_tactile=true requires batch.tactile_target_latent"
                )
            tactile_target = self._pad_or_trim_time(
                tactile_target,
                self.action_horizon,
            )
            if tactile_target.shape[-1] != self.tactile_latent_dim:
                raise ValueError(
                    f"tactile target dim {tactile_target.shape[-1]} != "
                    f"{self.tactile_latent_dim}"
                )
            target = torch.cat([actions, tactile_target], dim=-1)

        global_cond, condition_tokens = self._build_condition(obs)

        condition_mask = self.mask_generator(target.shape)
        loss_mask = ~condition_mask

        x1 = target
        x0 = torch.randn_like(x1)
        bsz = target.shape[0]
        t = torch.rand(bsz, device=target.device, dtype=target.dtype)
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
        element_loss = F.mse_loss(
            pred_velocity,
            target_velocity,
            reduction="none",
        )
        element_loss = element_loss * loss_mask.to(element_loss.dtype)
        action_loss = element_loss[..., : self.action_dim]
        action_loss = action_loss.reshape(action_loss.shape[0], -1).mean(dim=1).mean()
        if not self.predict_tactile:
            return {
                "loss": action_loss,
                "metrics": {"flow_matching_loss": action_loss.detach()},
            }

        tactile_loss = element_loss[..., self.action_dim :]
        tactile_loss = tactile_loss.reshape(
            tactile_loss.shape[0],
            -1,
        ).mean(dim=1).mean()
        loss = (
            self.action_loss_weight * action_loss
            + self.tactile_loss_weight * tactile_loss
        )
        return {
            "loss": loss,
            "metrics": {
                "flow_matching_loss": loss.detach(),
                "action_flow_loss": action_loss.detach(),
                "tactile_flow_loss": tactile_loss.detach(),
            },
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
            self.trajectory_dim,
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

    def conditional_sample_rtc(
        self,
        obs: Dict[str, torch.Tensor],
        *,
        prev_actions: torch.Tensor,
        inference_delay: int,
        prefix_attention_horizon: int,
        prefix_attention_schedule: str = "exp",
        max_guidance_weight: float = 5.0,
        num_inference_steps: int | None = None,
        solver: str | None = None,
    ) -> torch.Tensor:
        """Sample actions with RTC guidance from a previously planned prefix."""
        if self.predict_tactile:
            raise RuntimeError("RTC inference does not support predict_tactile=true")
        solver_name = self.solver if solver is None else str(solver).lower()
        if solver_name != "euler":
            raise ValueError(
                "RTC inference currently requires solver='euler', "
                f"got {solver_name!r}"
            )
        steps = (
            self.num_inference_steps
            if num_inference_steps is None
            else int(num_inference_steps)
        )
        if steps <= 0:
            raise ValueError("num_inference_steps must be positive")
        if int(prefix_attention_horizon) <= 0:
            raise ValueError("RTC prefix_attention_horizon must be positive")

        with torch.no_grad():
            global_cond, condition_tokens = self._build_condition(obs)
        global_cond = global_cond.detach()
        if condition_tokens is not None:
            condition_tokens = condition_tokens.detach()

        bsz = global_cond.shape[0]
        device = global_cond.device
        dtype = global_cond.dtype
        prev_actions = prev_actions.to(device=device, dtype=dtype)
        if prev_actions.ndim == 2:
            prev_actions = prev_actions.unsqueeze(0)
        if prev_actions.ndim != 3:
            raise ValueError(
                "prev_actions must be [B,T,A] or [T,A], "
                f"got {tuple(prev_actions.shape)}"
            )
        if prev_actions.shape[0] == 1 and bsz > 1:
            prev_actions = prev_actions.expand(bsz, -1, -1)
        if prev_actions.shape[0] != bsz:
            raise ValueError(
                f"prev_actions batch {prev_actions.shape[0]} != "
                f"observation batch {bsz}"
            )
        if prev_actions.shape[2] != self.action_dim:
            raise ValueError(
                f"prev_actions dim {prev_actions.shape[2]} != "
                f"action_dim {self.action_dim}"
            )

        from infer.rtc import guided_velocity, prefix_weights

        weights = prefix_weights(
            inference_delay=int(inference_delay),
            prefix_attention_horizon=int(prefix_attention_horizon),
            action_horizon=self.action_horizon,
            schedule=prefix_attention_schedule,
            device=device,
            dtype=dtype,
        )
        trajectory = torch.randn(
            bsz,
            self.action_horizon,
            self.action_dim,
            device=device,
            dtype=dtype,
        )
        times = torch.linspace(
            0.0,
            1.0,
            steps + 1,
            device=device,
            dtype=dtype,
        )
        for index in range(steps):
            t0 = times[index]
            dt = times[index + 1] - t0
            t_value = float(t0.item())
            t_batch = t0.expand(bsz)

            def denoise(value: torch.Tensor) -> torch.Tensor:
                return self._model_forward(
                    value,
                    t_batch,
                    global_cond=global_cond,
                    condition_tokens=condition_tokens,
                )

            velocity = guided_velocity(
                x_t=trajectory,
                time=t_value,
                denoise_fn=denoise,
                prev_actions=prev_actions,
                weights=weights,
                max_guidance_weight=max_guidance_weight,
            )
            trajectory = (trajectory + dt * velocity).detach()
        return trajectory

    @torch.no_grad()
    def predict_action(
        self,
        obs: Dict[str, torch.Tensor],
        num_inference_steps: int | None = None,
        solver: str | None = None,
        decode_tactile: bool = False,
    ) -> Dict[str, torch.Tensor]:
        joint_norm = self.conditional_sample(
            obs=obs,
            num_inference_steps=num_inference_steps,
            solver=solver,
        )
        action_norm = joint_norm[..., : self.action_dim]
        result = {
            "action_normalized": action_norm[:, : self.n_action_steps],
            "action_pred_normalized": action_norm,
        }
        if self.predict_tactile:
            tactile_latent = joint_norm[..., self.action_dim :]
            result["tactile_latent_pred_normalized"] = tactile_latent
            if decode_tactile:
                result["tactile_pred_normalized"] = self.decode_tactile_latent(
                    tactile_latent
                )
        return result

    def predict_action_rtc(
        self,
        obs: Dict[str, torch.Tensor],
        *,
        prev_actions: torch.Tensor,
        inference_delay: int,
        prefix_attention_horizon: int,
        prefix_attention_schedule: str = "exp",
        max_guidance_weight: float = 5.0,
        num_inference_steps: int | None = None,
        solver: str | None = None,
    ) -> Dict[str, torch.Tensor]:
        action_norm = self.conditional_sample_rtc(
            obs,
            prev_actions=prev_actions,
            inference_delay=inference_delay,
            prefix_attention_horizon=prefix_attention_horizon,
            prefix_attention_schedule=prefix_attention_schedule,
            max_guidance_weight=max_guidance_weight,
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
