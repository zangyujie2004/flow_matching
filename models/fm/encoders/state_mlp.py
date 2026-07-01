from __future__ import annotations

import torch
import torch.nn as nn


class StateMLP(nn.Module):
    """Encode proprio history (B, T, D) -> (B, out_dim)."""

    def __init__(
        self,
        state_dim: int,
        cond_steps: int = 8,
        out_dim: int = 256,
        hidden_dim: int = 512,
        pool: str = "flatten",
        dropout: float = 0.1,
    ):
        super().__init__()
        self.pool = str(pool)
        if self.pool == "flatten":
            in_dim = cond_steps * state_dim
        elif self.pool == "last":
            in_dim = state_dim
        else:
            raise ValueError(f"unsupported pool={pool!r}")

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        if state.ndim != 3:
            raise ValueError(f"expected state (B,T,D), got {state.shape}")
        if self.pool == "last":
            x = state[:, -1]
        else:
            x = state.flatten(1)
        return self.mlp(x)
