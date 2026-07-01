from __future__ import annotations

import torch
import torch.nn as nn


class TactileCNNEncoder(nn.Module):
    """Encode tactile deformation flow (B, T, H, W, C) -> (B, out_dim)."""

    def __init__(
        self,
        in_channels: int = 12,
        hidden_dim: int = 64,
        out_dim: int = 256,
        temporal_pool: str = "conv1d",
        dropout: float = 0.1,
    ):
        super().__init__()
        self.temporal_pool = str(temporal_pool)
        self.frame_encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        if self.temporal_pool == "mean":
            self.temporal = None
            self.proj = nn.Sequential(
                nn.Linear(hidden_dim, out_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
            )
        elif self.temporal_pool == "conv1d":
            self.temporal = nn.Sequential(
                nn.Conv1d(hidden_dim, out_dim, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool1d(1),
            )
            self.proj = None
        else:
            raise ValueError(f"unsupported temporal_pool={temporal_pool!r}")

    def forward(self, tactile: torch.Tensor) -> torch.Tensor:
        # (B, T, H, W, C) -> (B, T, C, H, W)
        if tactile.ndim != 5:
            raise ValueError(f"expected tactile (B,T,H,W,C), got {tactile.shape}")
        x = tactile.permute(0, 1, 4, 2, 3)
        b, t, c, h, w = x.shape
        frame_feat = self.frame_encoder(x.reshape(b * t, c, h, w)).reshape(b, t, -1)
        if self.temporal_pool == "mean":
            return self.proj(frame_feat.mean(dim=1))
        x = frame_feat.transpose(1, 2)
        out = self.temporal(x).squeeze(-1)
        return out
