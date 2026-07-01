from __future__ import annotations

try:
    import timm
except ImportError:
    timm = None

import torch
import torch.nn as nn


class DinoV2SmallEncoder(nn.Module):
    """Single-view DINOv2 backbone + projection head."""

    def __init__(
        self,
        out_dim: int = 256,
        pretrained: bool = True,
        freeze: bool = True,
        model_name: str = "vit_small_patch14_dinov2.lvd142m",
    ):
        super().__init__()
        if timm is None:
            raise ImportError("DINOv2 encoder requires timm. Install with: pip install timm")

        self.freeze = bool(freeze)
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            img_size=224,
        )
        backbone_dim = getattr(self.backbone, "num_features", None)
        if backbone_dim is None:
            raise RuntimeError("Cannot infer DINOv2 output dim from timm model.")

        self.head = nn.Sequential(
            nn.LayerNorm(backbone_dim),
            nn.Linear(backbone_dim, out_dim),
            nn.SiLU(),
        )

        if self.freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.backbone.eval()

    @staticmethod
    def _imagenet_normalize(x: torch.Tensor) -> torch.Tensor:
        if x.dtype == torch.uint8:
            x = x.float().div(255.0)
        else:
            x = (x + 1.0) * 0.5
        mean = x.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = x.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        return (x - mean) / std

    @staticmethod
    def _pool_backbone_output(feat: torch.Tensor) -> torch.Tensor:
        if isinstance(feat, (tuple, list)):
            feat = feat[-1]
        if feat.ndim == 3:
            feat = feat[:, 0]
        elif feat.ndim == 4:
            feat = feat.mean(dim=(2, 3))
        return feat

    def extract_backbone_feat(self, x: torch.Tensor) -> torch.Tensor:
        x = self._imagenet_normalize(x)
        if self.freeze:
            with torch.no_grad():
                feat = self.backbone(x)
        else:
            feat = self.backbone(x)
        return self._pool_backbone_output(feat)

    def forward_from_backbone_feat(self, feat: torch.Tensor) -> torch.Tensor:
        return self.head(feat)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze:
            self.backbone.eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_from_backbone_feat(self.extract_backbone_feat(x))


class DinoV2Encoder(nn.Module):
    """Multi-view image encoder: (B, T, V, 3, H, W) -> (B, out_dim)."""

    def __init__(
        self,
        out_dim: int = 256,
        n_views: int = 3,
        view_pool: str = "mean",
        pretrained: bool = True,
        freeze: bool = True,
        model_name: str = "vit_small_patch14_dinov2.lvd142m",
    ):
        super().__init__()
        self.n_views = int(n_views)
        self.view_pool = str(view_pool)
        self.encoder = DinoV2SmallEncoder(
            out_dim=out_dim,
            pretrained=pretrained,
            freeze=freeze,
            model_name=model_name,
        )
        if self.view_pool == "concat":
            self.view_proj = nn.Linear(out_dim * self.n_views, out_dim)
        else:
            self.view_proj = None

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        # (B, T, V, 3, H, W) — use last time step for single-frame policy
        if image.ndim != 6:
            raise ValueError(f"expected image (B,T,V,3,H,W), got {image.shape}")
        x = image[:, -1]
        b, v, c, h, w = x.shape
        feats = self.encoder(x.reshape(b * v, c, h, w)).reshape(b, v, -1)
        if self.view_pool == "concat":
            return self.view_proj(feats.flatten(1))
        if self.view_pool == "mean":
            return feats.mean(dim=1)
        raise ValueError(f"unsupported view_pool={self.view_pool!r}")

    def encode_from_backbone_feat(self, feat: torch.Tensor) -> torch.Tensor:
        """feat: (B, T, V, D_backbone) or (B, V, D_backbone)."""
        if feat.ndim == 3:
            feat = feat.unsqueeze(1)
        if feat.ndim != 4:
            raise ValueError(f"expected backbone feat (B,T,V,D), got {feat.shape}")
        x = feat[:, -1]
        b, v, d = x.shape
        projected = self.encoder.forward_from_backbone_feat(x.reshape(b * v, d)).reshape(b, v, -1)
        if self.view_pool == "concat":
            return self.view_proj(projected.flatten(1))
        if self.view_pool == "mean":
            return projected.mean(dim=1)
        raise ValueError(f"unsupported view_pool={self.view_pool!r}")
