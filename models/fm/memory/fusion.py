"""Fusion memory: CLS visual + Conv state → single token."""

from __future__ import annotations

import torch
import torch.nn as nn

from .types import MemoryOutput


def _as_batch_valid(
    valid: torch.Tensor | None,
    *,
    batch: int,
    seq: int,
    device: torch.device,
) -> torch.Tensor | None:
    if valid is None:
        return None
    if valid.ndim == 1:
        valid = valid.unsqueeze(0).expand(batch, -1)
    if valid.shape != (batch, seq):
        raise ValueError(f"valid shape {tuple(valid.shape)} != ({batch}, {seq})")
    return valid.to(device=device, dtype=torch.bool)


class StateConvMemoryEncoder(nn.Module):
    def __init__(
        self,
        *,
        state_dim: int,
        out_dim: int,
        channels: int = 128,
        layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        layers = max(1, int(layers))
        modules: list[nn.Module] = []
        in_ch = int(state_dim)
        hidden_ch = int(channels)
        for _ in range(layers):
            modules.extend(
                [
                    nn.Conv1d(in_ch, hidden_ch, kernel_size=5, padding=2),
                    nn.GroupNorm(1, hidden_ch),
                    nn.SiLU(),
                    nn.Dropout(dropout),
                ]
            )
            in_ch = hidden_ch
        self.net = nn.Sequential(*modules)
        self.proj = nn.Linear(hidden_ch, int(out_dim))

    def forward(self, state: torch.Tensor, valid: torch.Tensor | None = None) -> torch.Tensor:
        if state.ndim != 3:
            raise ValueError(f"expected memory state (B,T,D), got {state.shape}")
        if valid is not None:
            state = state * valid.unsqueeze(-1).to(device=state.device, dtype=state.dtype)
        x = state.transpose(1, 2)
        x = self.net(x)  # (B,C,T)
        if valid is None:
            pooled = x.mean(dim=-1)
        else:
            mask = valid.to(device=x.device, dtype=x.dtype).unsqueeze(1)  # (B,1,T)
            denom = mask.sum(dim=-1).clamp_min(1.0)
            pooled = (x * mask).sum(dim=-1) / denom
        return self.proj(pooled)


class VisualTemporalMemoryEncoder(nn.Module):
    def __init__(
        self,
        *,
        visual_dim: int,
        out_dim: int,
        max_time_offset: int,
        layers: int = 2,
        heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.visual_dim = int(visual_dim)
        self.time_embed = nn.Embedding(int(max_time_offset) + 1, self.visual_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.visual_dim))
        if int(layers) > 0:
            layer = nn.TransformerEncoderLayer(
                d_model=self.visual_dim,
                nhead=int(heads),
                dim_feedforward=self.visual_dim * 2,
                dropout=float(dropout),
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=int(layers))
        else:
            self.encoder = nn.Identity()
        self.norm = nn.LayerNorm(self.visual_dim)
        self.proj = nn.Linear(self.visual_dim, int(out_dim))
        nn.init.normal_(self.cls_token, std=0.02)

    def forward(
        self,
        visual_tokens: torch.Tensor,
        offsets: torch.Tensor,
        valid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if visual_tokens.ndim != 3:
            raise ValueError(f"expected visual tokens (B,T,D), got {visual_tokens.shape}")
        if offsets.ndim == 1:
            offsets = offsets.unsqueeze(0).expand(visual_tokens.shape[0], -1)
        if offsets.shape[:2] != visual_tokens.shape[:2]:
            raise ValueError(
                f"offset shape {tuple(offsets.shape)} does not match visual tokens {tuple(visual_tokens.shape)}"
            )
        distances = offsets.to(device=visual_tokens.device).abs().long()
        distances = distances.clamp_(0, self.time_embed.num_embeddings - 1)
        x = visual_tokens + self.time_embed(distances).to(dtype=visual_tokens.dtype)
        cls = self.cls_token.to(dtype=x.dtype).expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1)

        key_padding_mask = None
        if valid is not None:
            # TransformerEncoder: True = ignore. CLS always valid.
            pad = ~valid.to(device=x.device, dtype=torch.bool)
            key_padding_mask = torch.cat(
                [torch.zeros(x.shape[0], 1, dtype=torch.bool, device=x.device), pad],
                dim=1,
            )

        if isinstance(self.encoder, nn.Identity):
            x = self.encoder(x)
        else:
            x = self.encoder(x, src_key_padding_mask=key_padding_mask)
        return self.proj(self.norm(x[:, 0]))


class MemoryEncoder(nn.Module):
    """Fusion path: visual CLS + state Conv → single memory vector."""

    def __init__(
        self,
        *,
        state_dim: int,
        visual_dim: int,
        memory_dim: int,
        history_frames: int = 64,
        recent_frame: int = 2,
        visual_layers: int = 2,
        visual_heads: int = 4,
        state_channels: int = 128,
        state_layers: int = 2,
        state_mem_dim: int = 64,
        dropout: float = 0.1,
        **_ignored,
    ):
        super().__init__()
        visual_branch_dim = int(memory_dim)
        state_branch_dim = int(state_mem_dim)
        max_time_offset = int(history_frames) + int(recent_frame)
        self.visual_encoder = VisualTemporalMemoryEncoder(
            visual_dim=int(visual_dim),
            out_dim=visual_branch_dim,
            max_time_offset=max_time_offset,
            layers=int(visual_layers),
            heads=int(visual_heads),
            dropout=float(dropout),
        )
        self.state_encoder = StateConvMemoryEncoder(
            state_dim=int(state_dim),
            out_dim=state_branch_dim,
            channels=int(state_channels),
            layers=int(state_layers),
            dropout=float(dropout),
        )
        self.fusion = nn.Sequential(
            nn.LayerNorm(visual_branch_dim + state_branch_dim),
            nn.Linear(visual_branch_dim + state_branch_dim, int(memory_dim)),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(memory_dim), int(memory_dim)),
        )

    def forward(
        self,
        *,
        visual_tokens: torch.Tensor,
        visual_offsets: torch.Tensor,
        state: torch.Tensor,
        visual_valid: torch.Tensor | None = None,
        state_valid: torch.Tensor | None = None,
    ) -> MemoryOutput:
        bsz, tv, _ = visual_tokens.shape
        ts = state.shape[1]
        device = visual_tokens.device
        vis_valid = _as_batch_valid(visual_valid, batch=bsz, seq=tv, device=device)
        st_valid = _as_batch_valid(state_valid, batch=bsz, seq=ts, device=device)
        visual_mem = self.visual_encoder(visual_tokens, visual_offsets, valid=vis_valid)
        state_mem = self.state_encoder(state, valid=st_valid)
        token = self.fusion(torch.cat([visual_mem, state_mem], dim=-1))
        tokens = token.unsqueeze(1)
        return MemoryOutput(tokens=tokens, memory_global=token)
