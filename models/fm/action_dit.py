from __future__ import annotations

import math
from typing import Sequence, Union

import torch
import torch.nn as nn


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = int(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / max(half_dim - 1, 1)
        emb = torch.exp(torch.arange(half_dim, device=device, dtype=x.dtype) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        if emb.shape[-1] < self.dim:
            emb = torch.nn.functional.pad(emb, (0, self.dim - emb.shape[-1]))
        return emb


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1.0 + scale[:, None, :]) + shift[:, None, :]


class Mlp(nn.Module):
    def __init__(self, hidden_dim: int, mlp_ratio: float, dropout: float):
        super().__init__()
        inner_dim = int(hidden_dim * mlp_ratio)
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, inner_dim),
            nn.GELU(approximate="tanh"),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DiTBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        use_cross_attn: bool = False,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.cross_attn = None
        self.cross_norm = None
        self.cross_gate = None
        if use_cross_attn:
            self.cross_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
            self.cross_attn = nn.MultiheadAttention(
                hidden_dim,
                num_heads,
                dropout=dropout,
                batch_first=True,
            )
            self.cross_gate = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )

        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.mlp = Mlp(hidden_dim, mlp_ratio, dropout)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 6 * hidden_dim),
        )

        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)
        if self.cross_gate is not None:
            nn.init.zeros_(self.cross_gate[-1].weight)
            nn.init.zeros_(self.cross_gate[-1].bias)

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(cond).chunk(6, dim=-1)
        )
        attn_input = modulate(self.norm1(x), shift_msa, scale_msa)
        attn_out, _ = self.attn(attn_input, attn_input, attn_input, need_weights=False)
        x = x + gate_msa[:, None, :] * attn_out

        if context is not None and self.cross_attn is not None:
            cross_input = self.cross_norm(x)
            cross_out, _ = self.cross_attn(cross_input, context, context, need_weights=False)
            cross_gate = self.cross_gate(cond)
            x = x + cross_gate[:, None, :] * cross_out

        mlp_input = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp[:, None, :] * self.mlp(mlp_input)
        return x


class ActionDiT(nn.Module):
    """DiT velocity network for action flow matching.

    The public forward signature matches ConditionalUnet1D so FlowMatchingPolicy
    can swap velocity backbones without changing loss or sampling code.
    """

    def __init__(
        self,
        *,
        input_dim: int,
        action_horizon: int,
        global_cond_dim: int,
        diffusion_step_embed_dim: int = 256,
        hidden_dim: int = 512,
        depth: int = 8,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        condition_token_dim: int | None = None,
        cross_attn_layers: Sequence[int] | None = None,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.action_horizon = int(action_horizon)
        self.hidden_dim = int(hidden_dim)
        self.condition_token_dim = None if condition_token_dim is None else int(condition_token_dim)
        self.cross_attn_layers = (
            None if cross_attn_layers is None else tuple(int(x) for x in cross_attn_layers)
        )

        self.input_proj = nn.Linear(self.input_dim, self.hidden_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.action_horizon, self.hidden_dim))

        self.time_embed = nn.Sequential(
            SinusoidalPosEmb(diffusion_step_embed_dim),
            nn.Linear(diffusion_step_embed_dim, self.hidden_dim * 4),
            nn.SiLU(),
            nn.Linear(self.hidden_dim * 4, self.hidden_dim),
        )
        self.cond_proj = nn.Sequential(
            nn.Linear(global_cond_dim, self.hidden_dim * 4),
            nn.SiLU(),
            nn.Linear(self.hidden_dim * 4, self.hidden_dim),
        )
        self.context_proj = (
            nn.Linear(self.condition_token_dim, self.hidden_dim)
            if self.condition_token_dim is not None
            else None
        )

        cross_layer_set = set(self.cross_attn_layers or ())
        blocks = []
        for block_idx in range(int(depth)):
            use_cross = self.context_proj is not None and (block_idx + 1) in cross_layer_set
            blocks.append(
                DiTBlock(
                    hidden_dim=self.hidden_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    use_cross_attn=use_cross,
                )
            )
        self.blocks = nn.ModuleList(blocks)
        self.final_norm = nn.LayerNorm(self.hidden_dim, elementwise_affine=False, eps=1e-6)
        self.final_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.hidden_dim, 2 * self.hidden_dim),
        )
        self.output_proj = nn.Linear(self.hidden_dim, self.input_dim)

        nn.init.normal_(self.pos_embed, std=0.02)
        nn.init.zeros_(self.final_modulation[-1].weight)
        nn.init.zeros_(self.final_modulation[-1].bias)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        local_cond=None,
        global_cond=None,
        condition_tokens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if local_cond is not None:
            raise ValueError("ActionDiT does not support local_cond; pass global_cond only.")
        if global_cond is None:
            raise ValueError("ActionDiT requires global_cond.")
        if sample.shape[-1] != self.input_dim:
            raise ValueError(f"sample dim {sample.shape[-1]} != input_dim {self.input_dim}")
        if sample.shape[1] > self.action_horizon:
            raise ValueError(
                f"sample horizon {sample.shape[1]} exceeds configured action_horizon {self.action_horizon}"
            )

        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=sample.dtype, device=sample.device)
        elif timesteps.ndim == 0:
            timesteps = timesteps[None].to(device=sample.device, dtype=sample.dtype)
        else:
            timesteps = timesteps.to(device=sample.device, dtype=sample.dtype)
        timesteps = timesteps.expand(sample.shape[0])

        cond = self.time_embed(timesteps) + self.cond_proj(global_cond)
        context = None
        if condition_tokens is not None:
            if self.context_proj is None:
                raise ValueError("condition_tokens were provided but this ActionDiT has no context_proj.")
            if condition_tokens.ndim != 3:
                raise ValueError(f"expected condition_tokens (B,N,D), got {condition_tokens.shape}")
            context = self.context_proj(condition_tokens.to(dtype=sample.dtype))

        x = self.input_proj(sample) + self.pos_embed[:, : sample.shape[1], :]
        for block in self.blocks:
            x = block(x, cond, context=context)

        shift, scale = self.final_modulation(cond).chunk(2, dim=-1)
        x = modulate(self.final_norm(x), shift, scale)
        return self.output_proj(x)
