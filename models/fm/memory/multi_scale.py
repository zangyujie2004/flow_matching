"""Multi-scale (Fast Query) memory: range visual queries + state query."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .types import MemoryOutput


class DepthwiseTemporalBlock(nn.Module):
    """Residual depthwise temporal conv; LayerNorm is per-timestep (stream-friendly)."""

    def __init__(self, dim: int, *, kernel_size: int = 5, dropout: float = 0.1):
        super().__init__()
        padding = int(kernel_size) // 2
        self.norm = nn.LayerNorm(int(dim))
        self.dw = nn.Conv1d(
            int(dim),
            int(dim),
            kernel_size=int(kernel_size),
            padding=padding,
            groups=int(dim),
        )
        self.pw = nn.Linear(int(dim), int(dim))
        self.act = nn.SiLU()
        self.drop = nn.Dropout(float(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        h = self.norm(x)
        h = self.dw(h.transpose(1, 2)).transpose(1, 2)
        h = self.drop(self.act(self.pw(h)))
        return x + h


def build_range_attn_mask(num_queries: int, seq_len: int, *, device: torch.device) -> torch.Tensor:
    """Bool attn mask (Q, T): True = blocked. Oldest→newest; query i sees newest ceil((i+1)*T/Q)."""
    q = int(num_queries)
    t = int(seq_len)
    mask = torch.ones(q, t, dtype=torch.bool, device=device)
    for i in range(q):
        visible = int(math.ceil((i + 1) * t / q))
        if visible > 0:
            mask[i, t - visible :] = False
    return mask


class FastQueryMemoryEncoder(nn.Module):
    """Range visual queries + one state query → (B, num_queries+1, D)."""

    def __init__(
        self,
        *,
        state_dim: int,
        visual_dim: int,
        memory_dim: int,
        num_queries: int = 3,
        visual_heads: int = 4,
        state_hidden_dim: int = 64,
        state_layers: int = 2,
        state_heads: int = 4,
        dropout: float = 0.1,
        **_ignored,
    ):
        super().__init__()
        self.memory_dim = int(memory_dim)
        self.num_queries = max(1, int(num_queries))
        d = self.memory_dim

        self.visual_in = nn.Identity() if int(visual_dim) == d else nn.Linear(int(visual_dim), d)
        self.time_mlp = nn.Sequential(
            nn.Linear(2, d),
            nn.SiLU(),
            nn.Linear(d, d),
        )
        self.visual_queries = nn.Parameter(torch.zeros(1, self.num_queries, d))
        self.visual_attn = nn.MultiheadAttention(
            embed_dim=d,
            num_heads=int(visual_heads),
            dropout=float(dropout),
            batch_first=True,
        )
        self.visual_norm = nn.LayerNorm(d)

        state_h = int(state_hidden_dim)
        self.state_in = nn.Linear(int(state_dim), state_h)
        self.state_blocks = nn.ModuleList(
            [
                DepthwiseTemporalBlock(state_h, kernel_size=5, dropout=float(dropout))
                for _ in range(max(1, int(state_layers)))
            ]
        )
        self.state_to_mem = nn.Linear(state_h, d) if state_h != d else nn.Identity()
        self.state_query = nn.Parameter(torch.zeros(1, 1, d))
        self.state_attn = nn.MultiheadAttention(
            embed_dim=d,
            num_heads=int(state_heads),
            dropout=float(dropout),
            batch_first=True,
        )
        self.state_norm = nn.LayerNorm(d)

        nn.init.normal_(self.visual_queries, std=0.02)
        nn.init.normal_(self.state_query, std=0.02)

    def _time_features(self, offsets: torch.Tensor, *, dtype: torch.dtype) -> torch.Tensor:
        if offsets.ndim == 1:
            offsets = offsets.unsqueeze(0)
        off = offsets.to(dtype=dtype)
        return torch.stack([off, torch.log1p(off.abs())], dim=-1)

    def _key_padding_mask(self, valid: torch.Tensor | None, batch: int, seq: int, device) -> torch.Tensor | None:
        if valid is None:
            return None
        if valid.ndim == 1:
            valid = valid.unsqueeze(0).expand(batch, -1)
        if valid.shape != (batch, seq):
            raise ValueError(f"valid shape {tuple(valid.shape)} != ({batch}, {seq})")
        # MultiheadAttention: True = ignore
        return ~valid.to(device=device, dtype=torch.bool)

    def forward(
        self,
        *,
        visual_tokens: torch.Tensor,
        visual_offsets: torch.Tensor,
        state: torch.Tensor,
        visual_valid: torch.Tensor | None = None,
        state_valid: torch.Tensor | None = None,
    ) -> MemoryOutput:
        if visual_tokens.ndim != 3:
            raise ValueError(f"expected visual tokens (B,T,D), got {visual_tokens.shape}")
        if state.ndim != 3:
            raise ValueError(f"expected memory state (B,T,D), got {state.shape}")

        bsz, tv, _ = visual_tokens.shape
        device = visual_tokens.device
        dtype = visual_tokens.dtype

        if visual_offsets.ndim == 1:
            visual_offsets = visual_offsets.unsqueeze(0).expand(bsz, -1)
        if visual_offsets.shape[:2] != (bsz, tv):
            raise ValueError(
                f"visual_offsets shape {tuple(visual_offsets.shape)} != ({bsz}, {tv})"
            )

        vis = self.visual_in(visual_tokens)
        vis = vis + self.time_mlp(self._time_features(visual_offsets, dtype=dtype))
        q_vis = self.visual_queries.to(dtype=dtype).expand(bsz, -1, -1)
        range_mask = build_range_attn_mask(self.num_queries, tv, device=device)
        vis_pad = self._key_padding_mask(visual_valid, bsz, tv, device)
        routed_vis, _ = self.visual_attn(
            q_vis,
            vis,
            vis,
            key_padding_mask=vis_pad,
            attn_mask=range_mask,
            need_weights=False,
        )
        routed_vis = self.visual_norm(routed_vis)

        ts = state.shape[1]
        st = self.state_in(state)
        for block in self.state_blocks:
            st = block(st)
        st = self.state_to_mem(st)
        q_st = self.state_query.to(dtype=dtype).expand(bsz, -1, -1)
        st_pad = self._key_padding_mask(state_valid, bsz, ts, device)
        state_tok, _ = self.state_attn(
            q_st,
            st,
            st,
            key_padding_mask=st_pad,
            need_weights=False,
        )
        state_tok = self.state_norm(state_tok)

        tokens = torch.cat([routed_vis, state_tok], dim=1)
        memory_global = tokens.mean(dim=1)
        return MemoryOutput(tokens=tokens, memory_global=memory_global)
