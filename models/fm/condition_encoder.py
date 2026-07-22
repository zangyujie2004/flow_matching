from __future__ import annotations

from typing import Mapping

import torch
import torch.nn as nn

from .encoders.dino_v2 import DinoV2Encoder, resolve_dino_model_name
from .encoders.state_mlp import StateMLP
from .encoders.tactile_cnn import TactileCNNEncoder


def _normalize_image_encoder_name(name: str | None) -> str:
    key = str(name or "").strip().lower()
    if key in {"dinov2_base", "dinov2-base", "dino_base", "dinobase"}:
        return "dinov2_base"
    if key in {"dinov2_small", "dinov2-small", "dino_small", "dino", "dinov2"}:
        return "dinov2"
    if not key:
        return "dinov2"
    return key


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
        tactile_feat_dim: int = 256,
        tactile_temporal_pool: str = "conv1d",
        state_feat_dim: int = 256,
        state_pool: str = "flatten",
        fusion_hidden_dim: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.use_tactile = bool(use_tactile)
        self.cond_dim = int(cond_dim)

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
            self.tactile_encoder = TactileCNNEncoder(
                in_channels=tactile_channels,
                out_dim=tactile_feat_dim,
                temporal_pool=tactile_temporal_pool,
                dropout=dropout,
            )
            tactile_out = tactile_feat_dim

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
    ) -> torch.Tensor:
        parts = [self.encode_image(image=image, image_backbone_feat=image_backbone_feat), self.state_encoder(state)]
        if self.use_tactile:
            if tactile is None:
                raise ValueError("use_tactile=True but tactile is None")
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
