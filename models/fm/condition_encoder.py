from __future__ import annotations

from typing import Mapping

import torch
import torch.nn as nn

from .encoders.dino_v2 import DinoV2Encoder, resolve_dino_model_name
from .encoders.state_mlp import StateMLP
from .encoders.tactile_cnn import TactileCNNEncoder
from .encoders.tactile_token import TactileResidualTokenEncoder


def _normalize_image_encoder_name(name: str | None) -> str:
    key = str(name or "").strip().lower()
    if key in {"dinov2_base", "dinov2-base", "dino_base", "dinobase"}:
        return "dinov2_base"
    if key in {"dinov2_small", "dinov2-small", "dino_small", "dino", "dinov2"}:
        return "dinov2"
    if not key:
        return "dinov2"
    return key


def _normalize_tactile_encoder_type(name: str | None) -> str:
    key = str(name or "temporal_cnn").strip().lower()
    aliases = {
        "temporal_cnn": "temporal_cnn",
        "cnn": "temporal_cnn",
        "legacy": "temporal_cnn",
        "residual_token": "residual_token",
        "res_token": "residual_token",
        "token": "residual_token",
        "precomputed": "precomputed",
        "precomputed_latent": "precomputed",
    }
    if key not in aliases:
        raise ValueError(
            "unsupported tactile_encoder_type="
            f"{name!r}; expected temporal_cnn, residual_token, or precomputed"
        )
    return aliases[key]


def resolve_tactile_condition_encoder_type(
    *,
    predict_tactile: bool,
    tactile_encoder_type: str | None,
    tactile_condition_encoder_type: str | None = None,
) -> str:
    """Resolve the encoder used for tactile *conditioning*.

    ``predict_tactile`` controls the trajectory target, not the observation
    encoder.  The legacy Stage-2 implementation nevertheless forced
    ``precomputed`` whenever tactile prediction was enabled.  Preserve that
    behavior only for configs/checkpoints which do not contain the new
    explicit ``tactile_condition_encoder_type`` field.
    """
    if tactile_condition_encoder_type is None:
        if bool(predict_tactile):
            return "precomputed"
        return _normalize_tactile_encoder_type(tactile_encoder_type)

    resolved = _normalize_tactile_encoder_type(tactile_condition_encoder_type)
    if resolved == "precomputed" and not bool(predict_tactile):
        raise ValueError(
            "tactile_condition_encoder_type='precomputed' requires "
            "predict_tactile=true so the Stage-1 codec/latent cache is available"
        )
    return resolved


class ConditionEncoder(nn.Module):
    """Fuse vision / tactile / state into global condition for Flow Matching."""

    def __init__(
        self,
        *,
        state_dim: int,
        cond_dim: int = 256,
        cond_steps: int = 8,
        use_tactile: bool = True,
        tactile_channels: int = 12,
        image_encoder_name: str = "dinov2",
        dino_model_name: str | None = None,
        freeze_image_encoder: bool = True,
        image_pretrained: bool = True,
        image_feat_dim: int = 256,
        n_image_views: int = 3,
        view_pool: str = "global_concat",
        local_pool_size: int = 2,
        local_attn_heads: int = 4,
        local_attn_dropout: float = 0.0,
        tactile_encoder_type: str = "temporal_cnn",
        tactile_feat_dim: int = 256,
        tactile_temporal_pool: str = "conv1d",
        tactile_num_sensors: int = 4,
        tactile_channels_per_sensor: int = 3,
        tactile_token_dim: int = 16,
        state_feat_dim: int = 256,
        state_pool: str = "flatten",
        fusion_hidden_dim: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.use_tactile = bool(use_tactile)
        self.cond_dim = int(cond_dim)
        self.tactile_encoder_type = _normalize_tactile_encoder_type(
            tactile_encoder_type
        )

        enc_name = _normalize_image_encoder_name(image_encoder_name)
        if enc_name not in {"dinov2", "dinov2_base"}:
            raise ValueError(f"unsupported image_encoder_name={image_encoder_name!r}")
        model_name = resolve_dino_model_name(enc_name, dino_model_name)

        self.image_encoder = DinoV2Encoder(
            out_dim=image_feat_dim,
            n_views=n_image_views,
            view_pool=str(view_pool),
            local_pool_size=int(local_pool_size),
            local_attn_heads=int(local_attn_heads),
            local_attn_dropout=float(local_attn_dropout),
            pretrained=image_pretrained,
            freeze=freeze_image_encoder,
            model_name=model_name,
        )

        self.tactile_encoder = None
        tactile_out = 0
        if self.use_tactile:
            if self.tactile_encoder_type == "temporal_cnn":
                self.tactile_encoder = TactileCNNEncoder(
                    in_channels=tactile_channels,
                    out_dim=tactile_feat_dim,
                    temporal_pool=tactile_temporal_pool,
                    dropout=dropout,
                )
                tactile_out = tactile_feat_dim
            elif self.tactile_encoder_type == "residual_token":
                expected_channels = (
                    int(tactile_num_sensors)
                    * int(tactile_channels_per_sensor)
                )
                if int(tactile_channels) != expected_channels:
                    raise ValueError(
                        f"tactile_channels={tactile_channels} does not match "
                        f"tactile_num_sensors={tactile_num_sensors} x "
                        "tactile_channels_per_sensor="
                        f"{tactile_channels_per_sensor}"
                    )
                self.tactile_encoder = TactileResidualTokenEncoder(
                    num_sensors=tactile_num_sensors,
                    channels_per_sensor=tactile_channels_per_sensor,
                    token_dim=tactile_token_dim,
                )
                tactile_out = (
                    int(tactile_num_sensors) * int(tactile_token_dim)
                )
            else:
                tactile_out = (
                    int(tactile_num_sensors) * int(tactile_token_dim)
                )
        self.tactile_output_dim = tactile_out

        self.state_encoder = StateMLP(
            state_dim=state_dim,
            cond_steps=cond_steps,
            out_dim=state_feat_dim,
            hidden_dim=fusion_hidden_dim,
            pool=state_pool,
            dropout=dropout,
        )

        fuse_in = image_feat_dim + tactile_out + state_feat_dim
        self.fusion = nn.Sequential(
            nn.Linear(fuse_in, fusion_hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_dim, cond_dim),
        )

    def encode_image(
        self,
        *,
        image: torch.Tensor | None = None,
        image_backbone_feat: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if (image is None) == (image_backbone_feat is None):
            raise ValueError("Expected exactly one of image or image_backbone_feat.")
        if image_backbone_feat is not None:
            return self.image_encoder.encode_from_backbone_feat(image_backbone_feat)
        if image is None:
            raise ValueError("image is required when image_backbone_feat is None")
        return self.image_encoder(image)

    def encode_image_sequence_from_backbone_feat(self, image_backbone_feat: torch.Tensor) -> torch.Tensor:
        return self.image_encoder.encode_all_from_backbone_feat(image_backbone_feat)

    def forward(
        self,
        state: torch.Tensor,
        *,
        image: torch.Tensor | None = None,
        image_backbone_feat: torch.Tensor | None = None,
        tactile: torch.Tensor | None = None,
        tactile_latent: torch.Tensor | None = None,
    ) -> torch.Tensor:
        parts = [self.encode_image(image=image, image_backbone_feat=image_backbone_feat), self.state_encoder(state)]
        if self.use_tactile:
            if self.tactile_encoder_type == "precomputed":
                if tactile_latent is None:
                    raise ValueError(
                        "precomputed tactile encoder requires tactile_latent"
                    )
                if tactile_latent.ndim != 2 or (
                    tactile_latent.shape[-1] != self.tactile_output_dim
                ):
                    raise ValueError(
                        "expected tactile_latent "
                        f"(B,{self.tactile_output_dim}), got "
                        f"{tuple(tactile_latent.shape)}"
                    )
                parts.insert(1, tactile_latent)
            else:
                if tactile is None:
                    raise ValueError("use_tactile=True but tactile is None")
                if self.tactile_encoder is None:
                    raise RuntimeError("tactile encoder module is not configured")
                parts.insert(1, self.tactile_encoder(tactile))
        return self.fusion(torch.cat(parts, dim=-1))

    @classmethod
    def from_config(
        cls,
        cfg: Mapping,
        *,
        state_dim: int,
        cond_steps: int,
        tactile_channels: int = 12,
    ) -> "ConditionEncoder":
        return cls(
            state_dim=state_dim,
            cond_steps=cond_steps,
            tactile_channels=tactile_channels,
            **dict(cfg),
        )
