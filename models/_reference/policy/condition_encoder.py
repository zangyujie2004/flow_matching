from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


try:
    import timm
except ImportError:
    timm = None

class DinoV2SmallEncoder(nn.Module):
    def __init__(
        self,
        out_dim: int = 256,
        pretrained: bool = True,
        freeze: bool = True,
        model_name: str = "vit_small_patch14_dinov2.lvd142m",
    ):
        super().__init__()
        if timm is None:
            raise ImportError(
                "DINOv2 encoder requires timm. Install with: pip install timm"
            )

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
        # uint8 [0,255] or float32 in [-1, 1]
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
        # x: [N,3,H,W]
        feat = self.extract_backbone_feat(x)
        return self.forward_from_backbone_feat(feat)


class TactileCrossAttention(nn.Module):
    """Current tactile queries keys from predicted future; values are predicted change (pred - curr).

    Includes per-timestep position embeddings, a residual from current tokens, and an FFN block.
    """

    def __init__(
        self,
        dim: int,
        n_heads: int = 4,
        max_seq_len: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()
        if dim % n_heads != 0:
            raise ValueError(f"embed_dim {dim} must be divisible by n_heads {n_heads}")
        self.dim = dim
        self.max_seq_len = int(max_seq_len)

        self.pos_embed = nn.Parameter(torch.randn(1, self.max_seq_len, dim) * 0.02)

        self.norm_q = nn.LayerNorm(dim)
        self.norm_k = nn.LayerNorm(dim)
        self.norm_v = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm_ff = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, curr: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
        # curr, pred: [B, T, D]
        t = curr.shape[1]
        if pred.shape[1] != t:
            raise ValueError(
                f"T mismatch: curr T={t}, pred T={pred.shape[1]}"
            )
        if t > self.max_seq_len:
            raise ValueError(
                f"Sequence length T={t} exceeds max_seq_len={self.max_seq_len}"
            )

        pos = self.pos_embed[:, :t]
        q = self.norm_q(curr + pos)
        k = self.norm_k(pred + pos)
        v = self.norm_v(pred + pos)

        ctx, _ = self.attn(query=q, key=k, value=v)
        x = curr + ctx
        x = x + self.ffn(self.norm_ff(x))
        return x


class ConditionEncoder(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        force_dim: int,
        state_dim: int,
        cond_dim: int,
        image_encoder_name: str = "dinov2_small",
        freeze_image_encoder: bool = True,
        image_pretrained: bool = True,
        dino_model_name: str = "vit_small_patch14_dinov2.lvd142m",
        image_feat_dim: int = 256,
        latent_feat_dim: int = 256,
        lowdim_feat_dim: int = 256,
        hidden_dim: int = 512,
        dropout: float = 0.1,
        cond_steps: int = 8,
        tactile_cross_heads: int = 2,
    ):
        super().__init__()

        image_encoder_name = image_encoder_name.lower()
        if image_encoder_name in {"dinov2_small", "dinov2-small", "dino_small", "dino"}:
            self.image_encoder = DinoV2SmallEncoder(
                out_dim=image_feat_dim,
                pretrained=image_pretrained,
                freeze=freeze_image_encoder,
                model_name=dino_model_name,
            )
        else:
            raise ValueError(f"Unsupported image_encoder_name: {image_encoder_name}")

        self.cond_dim = int(cond_dim)
        self.cond_steps = int(cond_steps)
        self.image_feat_dim = int(image_feat_dim)
        self.latent_feat_dim = int(latent_feat_dim)
        self.lowdim_feat_dim = int(lowdim_feat_dim)

        lowdim_flat_dim = self.cond_steps * (int(force_dim) + int(state_dim))

        self.curr_token_mlp = nn.Sequential(
            nn.Linear(int(latent_dim), hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_feat_dim),
        )
        self.fut_token_mlp = nn.Sequential(
            nn.Linear(int(latent_dim), hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_feat_dim),
        )

        self.tactile_cross = TactileCrossAttention(
            dim=latent_feat_dim,
            n_heads=int(tactile_cross_heads),
            max_seq_len=int(cond_steps),
            dropout=float(dropout),
        )

        self.img_to_cond = nn.LazyLinear(int(cond_dim))
        if int(latent_feat_dim) != int(cond_dim):
            self.tactile_to_cond = nn.Linear(int(latent_feat_dim), int(cond_dim))
        else:
            self.tactile_to_cond = nn.Identity()

        self.gate_mlp = nn.Sequential(
            nn.Linear(latent_feat_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, cond_dim),
        )

        self.lowdim_proj = nn.Sequential(
            nn.Linear(lowdim_flat_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, lowdim_feat_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

        fused_in = int(cond_dim) + int(lowdim_feat_dim)
        self.cond_lowdim_fuse = nn.Sequential(
            nn.Linear(fused_in, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, cond_dim),
        )

        self.last_gate: torch.Tensor | None = None

    @staticmethod
    def _apply_token_mlp(mlp: nn.Module, x: torch.Tensor) -> torch.Tensor:
        bsz, steps, dim = x.shape
        y = mlp(x.reshape(bsz * steps, dim))
        return y.reshape(bsz, steps, -1)

    def encode_image(
        self,
        *,
        image: torch.Tensor | None,
        image_backbone_feat: torch.Tensor | None,
    ) -> torch.Tensor:
        if image_backbone_feat is not None:
            b, t, v = image_backbone_feat.shape[:3]
            return self.image_encoder.forward_from_backbone_feat(
                image_backbone_feat.reshape(b * t * v, image_backbone_feat.shape[-1])
            ).reshape(b, t, v, -1).flatten(2, 3)

        b, t, v = image.shape[:3]
        return self.image_encoder(
            image.reshape(b * t * v, *image.shape[3:])
        ).reshape(b, t, v, -1).flatten(2, 3)

    def forward(
        self,
        image: torch.Tensor | None,
        image_backbone_feat: torch.Tensor | None,
        tactile_latent_curr: torch.Tensor,
        tactile_latent_future: torch.Tensor,
        force: torch.Tensor,
        state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # image: [B, T, V, 3, H, W]
        # image_backbone_feat: [B, T, V, D_backbone]
        # tactile_latent_*: [B, T, D]
        # force/state: [B, T, Df/Ds]
        steps = tactile_latent_curr.shape[1]
        if tactile_latent_future.shape[1] != steps:
            raise ValueError(
                f"tactile_latent_curr T={steps} vs tactile_latent_future T={tactile_latent_future.shape[1]}"
            )
        if force.shape[1] != steps or state.shape[1] != steps:
            raise ValueError(f"force/state must have time length {steps}, got {force.shape[1]}/{state.shape[1]}")
        if steps != self.cond_steps:
            raise ValueError(f"ConditionEncoder expected cond_steps={self.cond_steps}, got T={steps}")

        img_feat = self.encode_image(
            image=image,
            image_backbone_feat=image_backbone_feat,
        )

        if img_feat.ndim == 3:
            img_feat = img_feat[:, -1]

        curr_tokens = self._apply_token_mlp(self.curr_token_mlp, tactile_latent_curr)
        fut_tokens = self._apply_token_mlp(self.fut_token_mlp, tactile_latent_future)
        tactile_fused = self.tactile_cross(curr_tokens, fut_tokens)
        tactile_summary = tactile_fused.mean(dim=1)

        lowdim_seq = torch.cat([force, state], dim=-1).flatten(1, 2)
        lowdim_feat = self.lowdim_proj(lowdim_seq)

        gate = torch.sigmoid(self.gate_mlp(tactile_summary))
        self.last_gate = gate.detach()
        img_c = self.img_to_cond(img_feat)
        tac_c = self.tactile_to_cond(tactile_summary)
        mixed = img_c * (1.0 - gate) + tac_c * gate

        global_cond = self.cond_lowdim_fuse(torch.cat([mixed, lowdim_feat], dim=-1))

        return global_cond, None
