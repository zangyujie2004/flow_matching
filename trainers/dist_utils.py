from __future__ import annotations

import os
from datetime import timedelta
from typing import Any

import torch
import torch.distributed as dist


def is_dist_available_and_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    if not is_dist_available_and_initialized():
        return 0
    return int(dist.get_rank())


def get_local_rank() -> int:
    if "LOCAL_RANK" in os.environ:
        return int(os.environ["LOCAL_RANK"])
    return get_rank()


def get_world_size() -> int:
    if not is_dist_available_and_initialized():
        return 1
    return int(dist.get_world_size())


def is_main_process() -> bool:
    return get_rank() == 0


def barrier() -> None:
    if is_dist_available_and_initialized():
        dist.barrier()


def init_distributed(
    backend: str = "nccl",
    *,
    timeout_s: float = 1800.0,
) -> tuple[int, int, int, torch.device]:
    """Init process group from torchrun env. Returns rank, local_rank, world_size, device."""
    if not dist.is_available():
        raise RuntimeError("torch.distributed is not available")
    if float(timeout_s) <= 0:
        raise ValueError(f"DDP timeout_s must be positive, got {timeout_s}")

    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")

    if world_size > 1 and not dist.is_initialized():
        if device.type == "cuda":
            try:
                dist.init_process_group(
                    backend=backend,
                    init_method="env://",
                    device_id=device,
                    timeout=timedelta(seconds=float(timeout_s)),
                )
            except TypeError:
                dist.init_process_group(
                    backend=backend,
                    init_method="env://",
                    timeout=timedelta(seconds=float(timeout_s)),
                )
        else:
            dist.init_process_group(
                backend=backend,
                init_method="env://",
                timeout=timedelta(seconds=float(timeout_s)),
            )
    return rank, local_rank, world_size, device


def cleanup_distributed() -> None:
    if is_dist_available_and_initialized():
        dist.barrier()
        dist.destroy_process_group()


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if hasattr(model, "module") else model


def reduce_mean(value: float) -> float:
    """All-reduce a scalar mean across ranks (no-op when not distributed)."""
    if not is_dist_available_and_initialized() or get_world_size() == 1:
        return float(value)
    t = torch.tensor(
        [float(value)],
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    t /= float(get_world_size())
    return float(t.item())


def broadcast_object(value: Any, *, src: int = 0) -> Any:
    """Broadcast a small pickleable object, returning it on every rank."""
    if not is_dist_available_and_initialized() or get_world_size() == 1:
        return value
    payload = [value if get_rank() == src else None]
    kwargs: dict[str, Any] = {}
    if dist.get_backend() == "nccl":
        kwargs["device"] = torch.device("cuda", get_local_rank())
    try:
        dist.broadcast_object_list(payload, src=src, **kwargs)
    except TypeError:
        # Older PyTorch versions do not expose the device keyword.
        dist.broadcast_object_list(payload, src=src)
    return payload[0]


def print_rank0(*args: Any, **kwargs: Any) -> None:
    if is_main_process():
        print(*args, **kwargs)
