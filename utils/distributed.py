"""Minimal DDP helpers for torchrun-launched multi-GPU training.

Single-GPU / CPU runs work unchanged: when no ``torchrun`` environment
variables are present, ``init_distributed`` reports world_size=1 and every
``is_main_process`` check is True.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.distributed as dist


@dataclass
class DistInfo:
    enabled: bool
    rank: int
    local_rank: int
    world_size: int

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def init_distributed() -> DistInfo:
    """Initialise the process group when launched via torchrun.

    Detection relies on the env vars torchrun sets (``RANK``/``WORLD_SIZE``/
    ``LOCAL_RANK``). Returns a :class:`DistInfo` describing this process.
    """
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1 or not torch.cuda.is_available():
        return DistInfo(enabled=False, rank=0, local_rank=0, world_size=1)

    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    return DistInfo(enabled=True, rank=rank, local_rank=local_rank, world_size=world_size)


def is_main_process(info: DistInfo | None) -> bool:
    return info is None or info.is_main


def barrier(info: DistInfo | None) -> None:
    if info is not None and info.enabled and dist.is_initialized():
        dist.barrier()


def cleanup(info: DistInfo | None) -> None:
    if info is not None and info.enabled and dist.is_initialized():
        dist.destroy_process_group()


def reduce_mean(value: float, info: DistInfo | None, device: torch.device) -> float:
    """Average a scalar across ranks (no-op single-GPU)."""
    if info is None or not info.enabled or not dist.is_initialized():
        return value
    tensor = torch.tensor([value], device=device, dtype=torch.float64)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return float(tensor.item() / info.world_size)
