from __future__ import annotations

from typing import NamedTuple

import torch


class MemoryOutput(NamedTuple):
    """Unified memory encoder output for fusion and multi-scale.

    tokens: (B, N, D) — N=1 for fusion, N=num_queries+1 for multi-scale
    memory_global: (B, D) — pooled token used by concat_global_cond
    """

    tokens: torch.Tensor
    memory_global: torch.Tensor
