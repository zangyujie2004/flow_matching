from __future__ import annotations

from pathlib import Path

try:
    import timm
except ImportError:
    timm = None

import torch
import torch.nn as nn
import torch.nn.functional as F

_PRETRAINED_DIR = Path(__file__).resolve().parents[3] / "pretrained_weights"
DINOV2_SMALL_MODEL = "vit_small_patch14_dinov2.lvd142m"
DINOV2_BASE_MODEL = "vit_base_patch14_dinov2.lvd142m"
_LOCAL_WEIGHTS = {
    DINOV2_SMALL_MODEL: _PRETRAINED_DIR / "dinov2_small.safetensors",
    DINOV2_BASE_MODEL: _PRETRAINED_DIR / "dinov2_base.safetensors",
}
DINOV2_NUM_TOKENS = 257
DINOV2_PATCH_GRID = 16


def resolve_dino_model_name(image_encoder_name: str | None, dino_model_name: str | None = None) -> str:
    if dino_model_name:
        return str(dino_model_name)
    key = str(image_encoder_name or "dinov2").strip().lower()
    if key in {"dinov2_base", "dinov2-base", "dino_base", "dinobase"}:
        return DINOV2_BASE_MODEL
    return DINOV2_SMALL_MODEL


def normalize_view_pool(view_pool: str | None) -> str:
    key = str(view_pool or "global_concat").strip().lower()
    if key in {"global_concat", "local_pool", "local_attn"}:
        return key
    raise ValueError(f"unsupported view_pool={view_pool!r}.")

def _create_timm_backbone(model_name: str, *, pretrained: bool):
    if timm is None:
        raise ImportError("DINOv2 encoder requires timm. Install with: pip install timm")

    local_path = _LOCAL_WEIGHTS.get(model_name)
    create_kwargs = {
        "model_name": model_name,
        "pretrained": bool(pretrained),
        "num_classes": 0,
        "img_size": 224,
    }
    if pretrained and local_path is not None and local_path.is_file():
        create_kwargs["pretrained_cfg_overlay"] = {"file": str(local_path)}
    elif pretrained and local_path is not None and not local_path.is_file():
        pass
    return timm.create_model(**create_kwargs)


class DinoV2SmallEncoder(nn.Module):
    def __init__(
        self,
        out_dim: int = 256,
        pretrained: bool = True,
        freeze: bool = True,
        model_name: str = DINOV2_SMALL_MODEL,
    ):
        super().__init__()
        if timm is None:
            raise ImportError("DINOv2 encoder requires timm. Install with: pip install timm")

        self.freeze = bool(freeze)
        self.model_name = str(model_name)
        self.backbone = _create_timm_backbone(self.model_name, pretrained=pretrained)
        backbone_dim = getattr(self.backbone, "num_features", None)
        if backbone_dim is None:
            raise RuntimeError("Cannot infer DINOv2 output dim from timm model.")
        self.backbone_dim = int(backbone_dim)
        self.num_tokens = int(DINOV2_NUM_TOKENS)

        self.head = nn.Sequential(
            nn.LayerNorm(self.backbone_dim),
            nn.Linear(self.backbone_dim, out_dim),
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

    def extract_backbone_feat(self, x: torch.Tensor) -> torch.Tensor:
        x = self._imagenet_normalize(x)
        if self.freeze:
            with torch.no_grad():
                tokens = self.backbone.forward_features(x)
        else:
            tokens = self.backbone.forward_features(x)
        if tokens.ndim != 3:
            raise RuntimeError(f"expected forward_features (B,N,D), got {tuple(tokens.shape)}")
        if int(tokens.shape[1]) != self.num_tokens:
            raise RuntimeError(
                f"expected {self.num_tokens} tokens (CLS+patches), got N={tokens.shape[1]}"
            )
        return tokens

    def forward_from_backbone_feat(self, feat: torch.Tensor) -> torch.Tensor:
        return self.head(feat)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze:
            self.backbone.eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.extract_backbone_feat(x)
        return self.forward_from_backbone_feat(tokens[:, 0])


DinoV2BackboneEncoder = DinoV2SmallEncoder


class _LightCrossAttnPool(nn.Module):
    def __init__(self, dim: int, *, num_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        if dim % int(num_heads) != 0:
            raise ValueError(f"attn dim {dim} must be divisible by num_heads={num_heads}")
        self.dim = int(dim)
        self.query = nn.Parameter(torch.zeros(1, 1, self.dim))
        nn.init.normal_(self.query, std=0.02)
        self.norm_kv = nn.LayerNorm(self.dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=self.dim,
            num_heads=int(num_heads),
            dropout=float(dropout),
            batch_first=True,
        )
        self.norm_q = nn.LayerNorm(self.dim)
        self.ffn = nn.Sequential(
            nn.Linear(self.dim, self.dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(self.dim * 2, self.dim),
            nn.Dropout(dropout),
        )
        self.norm_out = nn.LayerNorm(self.dim)

    def forward(self, kv: torch.Tensor) -> torch.Tensor:
        if kv.ndim != 3 or kv.shape[-1] != self.dim:
            raise ValueError(f"expected kv (B,N,{self.dim}), got {kv.shape}")
        b = kv.shape[0]
        q = self.query.expand(b, -1, -1)
        kv_n = self.norm_kv(kv)
        attn_out, _ = self.attn(q, kv_n, kv_n, need_weights=False)
        x = self.norm_q(q + attn_out)
        x = self.norm_out(x + self.ffn(x))
        return x.squeeze(1)


class DinoV2Encoder(nn.Module):
    def __init__(
        self,
        out_dim: int = 256,
        n_views: int = 3,
        view_pool: str = "global_concat",
        local_pool_size: int = 2,
        local_attn_heads: int = 4,
        local_attn_dropout: float = 0.0,
        pretrained: bool = True,
        freeze: bool = True,
        model_name: str = DINOV2_SMALL_MODEL,
    ):
        super().__init__()
        self.n_views = int(n_views)
        self.out_dim = int(out_dim)
        self.view_pool = normalize_view_pool(view_pool)
        self.local_pool_size = int(local_pool_size)
        self.local_attn_heads = int(local_attn_heads)
        if self.local_pool_size < 1:
            raise ValueError(f"local_pool_size must be >= 1, got {self.local_pool_size}")

        self.encoder = DinoV2SmallEncoder(
            out_dim=out_dim,
            pretrained=pretrained,
            freeze=freeze,
            model_name=model_name,
        )
        self.backbone_dim = int(self.encoder.backbone_dim)
        self.tokens_per_view = 1 + self.local_pool_size * self.local_pool_size
        self.n_vision_tokens = self.n_views * self.tokens_per_view

        # global_concat
        fuse_in = self.out_dim * self.n_views
        self.view_proj = nn.Sequential(
            nn.LayerNorm(fuse_in),
            nn.Linear(fuse_in, self.out_dim),
            nn.SiLU(),
            nn.Linear(self.out_dim, self.out_dim),
        )

        # local_pool / local_atten
        hidden = max(self.out_dim * 2, self.backbone_dim // 2)
        self.token_mlp = nn.Sequential(
            nn.LayerNorm(self.backbone_dim),
            nn.Linear(self.backbone_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, self.out_dim),
        )
        # distinguish views / CLS-vs-local for attention
        self.view_embed = nn.Embedding(self.n_views, self.out_dim)
        self.slot_embed = nn.Embedding(self.tokens_per_view, self.out_dim)
        # local_pool fuse
        self.token_fuse = nn.Sequential(
            nn.LayerNorm(self.n_vision_tokens * self.out_dim),
            nn.Linear(self.n_vision_tokens * self.out_dim, self.out_dim),
            nn.SiLU(),
            nn.Linear(self.out_dim, self.out_dim),
        )

        # local_atten
        self.cross_attn_pool = _LightCrossAttnPool(
            self.out_dim,
            num_heads=self.local_attn_heads,
            dropout=float(local_attn_dropout),
        )

    @staticmethod
    def _to_token_cache(feat: torch.Tensor) -> torch.Tensor:
        if feat.ndim == 5:
            return feat
        if feat.ndim == 4 and feat.shape[-2] == DINOV2_NUM_TOKENS:
            return feat.unsqueeze(1)
        raise ValueError(
            f"local_* requires token cache (B,T,V,257,D) or (B,V,257,D), got {feat.shape}"
        )

    @staticmethod
    def _to_cls_tokens(feat: torch.Tensor) -> torch.Tensor:
        if feat.ndim == 3:
            return feat.unsqueeze(1)
        if feat.ndim == 4:
            if feat.shape[-2] == DINOV2_NUM_TOKENS:
                return feat[:, None, :, 0, :]
            return feat
        if feat.ndim == 5:
            return feat[:, :, :, 0, :]
        raise ValueError(f"expected backbone feat (B,T,V,D) or (B,T,V,N,D), got {feat.shape}")

    def _fuse_view_feats(self, feats: torch.Tensor) -> torch.Tensor:
        if feats.ndim == 3:
            b, v, d = feats.shape
            if v != self.n_views:
                raise ValueError(f"expected n_views={self.n_views}, got V={v}")
            return self.view_proj(feats.reshape(b, v * d))
        if feats.ndim == 4:
            b, t, v, d = feats.shape
            if v != self.n_views:
                raise ValueError(f"expected n_views={self.n_views}, got V={v}")
            return self.view_proj(feats.reshape(b, t, v * d))
        raise ValueError(f"expected view feats (B,V,D) or (B,T,V,D), got {feats.shape}")

    def _pool_local_patches(self, patches: torch.Tensor) -> torch.Tensor:
        if patches.ndim != 5:
            raise ValueError(f"expected patches (B,T,V,256,D), got {patches.shape}")
        b, t, v, n_patch, d = patches.shape
        grid = DINOV2_PATCH_GRID
        if n_patch != grid * grid:
            raise ValueError(f"expected {grid * grid} patches, got {n_patch}")
        k = self.local_pool_size
        x = patches.reshape(b * t * v, grid, grid, d).permute(0, 3, 1, 2).contiguous()
        x = F.adaptive_avg_pool2d(x, output_size=(k, k))
        return x.reshape(b, t, v, d, k * k).permute(0, 1, 2, 4, 3).contiguous()

    def _build_short_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """(B,T,V,257,D) -> (B,T,N,out_dim), N=V*(1+k*k)."""
        if tokens.ndim != 5:
            raise ValueError(f"expected tokens (B,T,V,257,D), got {tokens.shape}")
        b, t, v, n, d = tokens.shape
        if v != self.n_views:
            raise ValueError(f"expected n_views={self.n_views}, got V={v}")
        if n != DINOV2_NUM_TOKENS:
            raise ValueError(f"expected N={DINOV2_NUM_TOKENS}, got {n}")
        if d != self.backbone_dim:
            raise ValueError(f"expected backbone_dim={self.backbone_dim}, got D={d}")

        cls = tokens[:, :, :, :1, :]
        patches = tokens[:, :, :, 1:, :]
        local = self._pool_local_patches(patches)
        per_view = torch.cat([cls, local], dim=3)  # (B,T,V,1+k*k,D)
        flat = per_view.reshape(b, t, self.n_vision_tokens, d)
        tok = self.token_mlp(flat)  # (B,T,N,out_dim)

        # additive view / slot embeddings (broadcast over B,T)
        view_ids = torch.arange(self.n_views, device=tok.device).repeat_interleave(
            self.tokens_per_view
        )
        slot_ids = torch.arange(self.tokens_per_view, device=tok.device).repeat(self.n_views)
        tok = tok + self.view_embed(view_ids)[None, None, :, :]
        tok = tok + self.slot_embed(slot_ids)[None, None, :, :]
        return tok

    def _fuse_vision_tokens_mlp(self, vision_tokens: torch.Tensor) -> torch.Tensor:
        b, t, n, d = vision_tokens.shape
        if n != self.n_vision_tokens or d != self.out_dim:
            raise ValueError(
                f"expected vision tokens (B,T,{self.n_vision_tokens},{self.out_dim}), "
                f"got {vision_tokens.shape}"
            )
        return self.token_fuse(vision_tokens.reshape(b, t, n * d))

    def _fuse_vision_tokens_attn(self, vision_tokens: torch.Tensor) -> torch.Tensor:
        b, t, n, d = vision_tokens.shape
        if n != self.n_vision_tokens or d != self.out_dim:
            raise ValueError(
                f"expected vision tokens (B,T,{self.n_vision_tokens},{self.out_dim}), "
                f"got {vision_tokens.shape}"
            )
        flat = vision_tokens.reshape(b * t, n, d)
        out = self.cross_attn_pool(flat)
        return out.reshape(b, t, d)

    def _encode_from_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """tokens (B,T,V,257,D) -> (B,T,out_dim)."""
        if self.view_pool == "global_concat":
            b, t, v, _, d = tokens.shape
            cls = tokens[:, :, :, 0, :]
            projected = self.encoder.forward_from_backbone_feat(cls.reshape(b * t * v, d)).reshape(
                b, t, v, -1
            )
            return self._fuse_view_feats(projected)

        vision_tokens = self._build_short_tokens(tokens)
        if self.view_pool == "local_pool":
            return self._fuse_vision_tokens_mlp(vision_tokens)
        return self._fuse_vision_tokens_attn(vision_tokens)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim != 6:
            raise ValueError(f"expected image (B,T,V,3,H,W), got {image.shape}")
        x = image[:, -1]
        b, v, c, h, w = x.shape
        if v != self.n_views:
            raise ValueError(f"expected n_views={self.n_views}, got V={v}")
        tokens = self.encoder.extract_backbone_feat(x.reshape(b * v, c, h, w))
        tokens = tokens.reshape(b, 1, v, tokens.shape[1], tokens.shape[2])
        return self._encode_from_tokens(tokens)[:, 0]

    def encode_vision_tokens_from_backbone_feat(self, feat: torch.Tensor) -> torch.Tensor:
        if self.view_pool not in {"local_pool", "local_attn"}:
            raise RuntimeError("encode_vision_tokens_* requires local_pool or local_attn")
        return self._build_short_tokens(self._to_token_cache(feat))

    def encode_all_from_backbone_feat(self, feat: torch.Tensor) -> torch.Tensor:
        if self.view_pool == "global_concat":
            cls = self._to_cls_tokens(feat)
            if cls.ndim != 4:
                raise ValueError(f"expected CLS feat (B,T,V,D), got {cls.shape}")
            b, t, v, d = cls.shape
            projected = self.encoder.forward_from_backbone_feat(cls.reshape(b * t * v, d)).reshape(
                b, t, v, -1
            )
            return self._fuse_view_feats(projected)
        tokens = self._to_token_cache(feat)
        return self._encode_from_tokens(tokens)

    def encode_from_backbone_feat(self, feat: torch.Tensor) -> torch.Tensor:
        return self.encode_all_from_backbone_feat(feat)[:, -1]
