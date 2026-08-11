"""DiT velocity model for flow matching in the LIP Stage-1 latent space."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Union

import torch
from torch import nn

from .action_dit import Mlp, SinusoidalPosEmb, modulate


def _key_padding_mask(valid: torch.Tensor | None) -> torch.Tensor | None:
    """Convert the public True=valid convention to PyTorch True=padding."""
    if valid is None:
        return None
    if valid.dtype is not torch.bool or valid.ndim != 2:
        raise ValueError("attention masks must be bool tensors shaped [B,N]")
    return ~valid


class LatentDiTBlock(nn.Module):
    """AdaLN-Zero block with parallel visual/embodiment cross-attention."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        *,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        use_cross_attn: bool = False,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.self_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.cross_norm: nn.LayerNorm | None = None
        self.visual_cross_attn: nn.MultiheadAttention | None = None
        self.embodiment_cross_attn: nn.MultiheadAttention | None = None
        if use_cross_attn:
            self.cross_norm = nn.LayerNorm(
                hidden_dim, elementwise_affine=False, eps=1e-6
            )
            self.visual_cross_attn = nn.MultiheadAttention(
                hidden_dim, num_heads, dropout=dropout, batch_first=True
            )
            self.embodiment_cross_attn = nn.MultiheadAttention(
                hidden_dim, num_heads, dropout=dropout, batch_first=True
            )
            # Stable identity initialization without an explicit branch gate.
            for attention in (
                self.visual_cross_attn,
                self.embodiment_cross_attn,
            ):
                nn.init.zeros_(attention.out_proj.weight)
                nn.init.zeros_(attention.out_proj.bias)

        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.mlp = Mlp(hidden_dim, mlp_ratio, dropout)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_dim, 6 * hidden_dim)
        )
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(
        self,
        x: torch.Tensor,
        time_condition: torch.Tensor,
        *,
        visual_context: torch.Tensor | None = None,
        embodiment_context: torch.Tensor | None = None,
        latent_valid: torch.Tensor | None = None,
        visual_valid: torch.Tensor | None = None,
        embodiment_valid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(time_condition).chunk(6, dim=-1)
        )
        self_input = modulate(self.norm1(x), shift_msa, scale_msa)
        self_out, _ = self.self_attn(
            self_input,
            self_input,
            self_input,
            key_padding_mask=_key_padding_mask(latent_valid),
            need_weights=False,
        )
        x = x + gate_msa[:, None, :] * self_out

        # Both modalities query from the same normalized latent state.
        if self.cross_norm is not None:
            query = self.cross_norm(x)
            if visual_context is not None:
                if self.visual_cross_attn is None:
                    raise RuntimeError("visual cross-attention is not configured")
                visual_out, _ = self.visual_cross_attn(
                    query,
                    visual_context,
                    visual_context,
                    key_padding_mask=_key_padding_mask(visual_valid),
                    need_weights=False,
                )
                x = x + visual_out
            if embodiment_context is not None:
                if self.embodiment_cross_attn is None:
                    raise RuntimeError("embodiment cross-attention is not configured")
                embodiment_out, _ = self.embodiment_cross_attn(
                    query,
                    embodiment_context,
                    embodiment_context,
                    key_padding_mask=_key_padding_mask(embodiment_valid),
                    need_weights=False,
                )
                x = x + embodiment_out

        mlp_input = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp[:, None, :] * self.mlp(mlp_input)
        if latent_valid is not None:
            x = x * latent_valid[:, :, None].to(dtype=x.dtype)
        return x


class LatentDiT(nn.Module):
    """Velocity network for a fixed-length Stage-1 interaction latent.

    Flow time is the only AdaLN condition. Learned absolute positions encode
    latent order. Visual and embodiment tokens use independent cross-attention.
    """

    def __init__(
        self,
        *,
        latent_dim: int = 256,
        latent_horizon: int = 16,
        visual_dim: int = 256,
        embodiment_dim: int = 256,
        diffusion_step_embed_dim: int = 256,
        hidden_dim: int = 512,
        depth: int = 14,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        cross_attn_layers: Sequence[int] = (2, 5, 8, 11),
    ) -> None:
        super().__init__()
        if latent_dim <= 0 or latent_horizon <= 0:
            raise ValueError("latent_dim and latent_horizon must be positive")
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        layers = tuple(int(layer) for layer in cross_attn_layers)
        if len(set(layers)) != len(layers):
            raise ValueError("cross_attn_layers must not contain duplicates")
        if any(layer < 1 or layer > depth for layer in layers):
            raise ValueError(
                f"cross_attn_layers must use one-based indices in [1,{depth}]"
            )

        self.latent_dim = int(latent_dim)
        self.latent_horizon = int(latent_horizon)
        self.hidden_dim = int(hidden_dim)
        self.cross_attn_layers = layers
        self.input_proj = nn.Linear(self.latent_dim, self.hidden_dim)
        self.position_embedding = nn.Parameter(
            torch.empty(1, self.latent_horizon, self.hidden_dim)
        )
        nn.init.normal_(self.position_embedding, std=0.02)

        # Deliberately identical to the original ActionDiT timestep path.
        self.time_embed = nn.Sequential(
            SinusoidalPosEmb(diffusion_step_embed_dim),
            nn.Linear(diffusion_step_embed_dim, self.hidden_dim * 4),
            nn.SiLU(),
            nn.Linear(self.hidden_dim * 4, self.hidden_dim),
        )
        self.visual_proj = nn.Sequential(
            nn.LayerNorm(visual_dim), nn.Linear(visual_dim, self.hidden_dim)
        )
        self.embodiment_proj = nn.Sequential(
            nn.LayerNorm(embodiment_dim), nn.Linear(embodiment_dim, self.hidden_dim)
        )
        cross_layer_set = set(layers)
        self.blocks = nn.ModuleList(
            [
                LatentDiTBlock(
                    self.hidden_dim,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    use_cross_attn=(block_idx + 1) in cross_layer_set,
                )
                for block_idx in range(int(depth))
            ]
        )
        self.final_norm = nn.LayerNorm(
            self.hidden_dim, elementwise_affine=False, eps=1e-6
        )
        self.final_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(self.hidden_dim, 2 * self.hidden_dim)
        )
        self.output_proj = nn.Linear(self.hidden_dim, self.latent_dim)
        nn.init.zeros_(self.final_modulation[-1].weight)
        nn.init.zeros_(self.final_modulation[-1].bias)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    @staticmethod
    def _timesteps(
        timestep: Union[torch.Tensor, float, int], sample: torch.Tensor
    ) -> torch.Tensor:
        if not torch.is_tensor(timestep):
            value = torch.tensor([timestep], device=sample.device, dtype=sample.dtype)
        elif timestep.ndim == 0:
            value = timestep[None].to(device=sample.device, dtype=sample.dtype)
        else:
            value = timestep.reshape(-1).to(device=sample.device, dtype=sample.dtype)
        if value.numel() not in {1, sample.shape[0]}:
            raise ValueError("timestep must be scalar, [1], [B], or [B,1]")
        return value.expand(sample.shape[0])

    @staticmethod
    def _validate_context(
        name: str,
        tokens: torch.Tensor | None,
        valid: torch.Tensor | None,
        batch_size: int,
    ) -> None:
        if tokens is None:
            if valid is not None:
                raise ValueError(f"{name}_valid was provided without {name}_tokens")
            return
        if tokens.ndim != 3 or tokens.shape[0] != batch_size:
            raise ValueError(f"{name}_tokens must be [B,N,C]")
        if valid is not None and valid.shape != tokens.shape[:2]:
            raise ValueError(f"{name}_valid must match {name}_tokens [B,N]")

    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        *,
        visual_tokens: torch.Tensor | None,
        embodiment_tokens: torch.Tensor | None,
        latent_valid: torch.Tensor | None = None,
        visual_valid: torch.Tensor | None = None,
        embodiment_valid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if sample.ndim != 3 or sample.shape[-1] != self.latent_dim:
            raise ValueError(
                f"sample must be [B,L,{self.latent_dim}], got {tuple(sample.shape)}"
            )
        if sample.shape[1] != self.latent_horizon:
            raise ValueError(
                f"fixed latent_horizon is {self.latent_horizon}, got {sample.shape[1]}"
            )
        batch_size = sample.shape[0]
        if latent_valid is not None and latent_valid.shape != sample.shape[:2]:
            raise ValueError("latent_valid must match sample [B,L]")
        self._validate_context("visual", visual_tokens, visual_valid, batch_size)
        self._validate_context(
            "embodiment", embodiment_tokens, embodiment_valid, batch_size
        )

        time_condition = self.time_embed(self._timesteps(timestep, sample))
        visual_context = (
            None
            if visual_tokens is None
            else self.visual_proj(visual_tokens.to(dtype=sample.dtype))
        )
        embodiment_context = (
            None
            if embodiment_tokens is None
            else self.embodiment_proj(embodiment_tokens.to(dtype=sample.dtype))
        )
        x = self.input_proj(sample) + self.position_embedding.to(dtype=sample.dtype)
        if latent_valid is not None:
            x = x * latent_valid[:, :, None].to(dtype=x.dtype)
        for block in self.blocks:
            x = block(
                x,
                time_condition,
                visual_context=visual_context,
                embodiment_context=embodiment_context,
                latent_valid=latent_valid,
                visual_valid=visual_valid,
                embodiment_valid=embodiment_valid,
            )
        shift, scale = self.final_modulation(time_condition).chunk(2, dim=-1)
        velocity = self.output_proj(modulate(self.final_norm(x), shift, scale))
        if latent_valid is not None:
            velocity = velocity * latent_valid[:, :, None].to(dtype=velocity.dtype)
        return velocity
