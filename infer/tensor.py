from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import torch

from tools.normalizer import DatasetNormalizer
from utils.tensor import move_to_device


def as_float32_array(value: Any, *, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN or inf")
    return arr

def image_to_batched_uint8(image: Any) -> torch.Tensor:
    if torch.is_tensor(image):
        tensor = image.detach().cpu()
    else:
        tensor = torch.from_numpy(np.asarray(image))

    if tensor.dtype != torch.uint8:
        if tensor.dtype.is_floating_point:
            tensor = tensor.round().clamp_(0.0, 255.0).to(torch.uint8)
        else:
            tensor = tensor.to(torch.uint8)

    if tensor.ndim == 6:
        out = tensor
    elif tensor.ndim == 5:
        # (T, V, C, H, W) -> (1, T, V, C, H, W)
        out = tensor.unsqueeze(0)
    elif tensor.ndim == 4:
        # (V, C, H, W) -> (1, 1, V, C, H, W)
        out = tensor.unsqueeze(0).unsqueeze(0)
    else:
        raise ValueError(f"obs.image must be 4D/5D/6D, got shape {tuple(tensor.shape)}")

    if out.shape[3] != 3:
        raise ValueError(f"obs.image channel dim must be 3, got shape {tuple(out.shape)}")
    return out.contiguous()

def state_to_batched_float(state: Any) -> torch.Tensor:
    if torch.is_tensor(state):
        tensor = state.detach().cpu().float()
    else:
        tensor = torch.from_numpy(as_float32_array(state, name="obs.state"))

    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)
    elif tensor.ndim != 3:
        raise ValueError(f"obs.state must be (T,D) or (B,T,D), got shape {tuple(tensor.shape)}")
    return tensor.contiguous()

def default_tactile_norm(
    normalizer: DatasetNormalizer,
    window_size: int,
    *,
    spatial: tuple[int, int] = (35, 20),
    channels: int = 12,
) -> np.ndarray:
    raw = np.zeros((window_size, spatial[0], spatial[1], channels), dtype=np.float32)
    return normalizer.normalize_tactile_np(raw).astype(np.float32, copy=False)


def numpy_obs_to_torch(
    obs: Mapping[str, Any],
    device: torch.device,
    *,
    use_tactile: bool = False,
    normalizer: DatasetNormalizer | None = None,
    window_size: int = 8,
) -> dict[str, torch.Tensor]:
    if "image" not in obs:
        raise KeyError("obs must contain 'image' for deployment inference")
    if "state" not in obs:
        raise KeyError("obs must contain 'state'")

    obs_torch = {
        "image": image_to_batched_uint8(obs["image"]),
        "state": state_to_batched_float(obs["state"]),
    }
    if use_tactile:
        if "tactile" in obs:
            tactile = as_float32_array(obs["tactile"], name="obs.tactile")
            if tactile.ndim == 4:
                tactile = tactile[None]
            obs_torch["tactile"] = torch.from_numpy(tactile)
        else:
            if normalizer is None:
                raise ValueError("use_tactile model requires normalizer to synthesize zero tactile")
            tactile = default_tactile_norm(normalizer, window_size)
            obs_torch["tactile"] = torch.from_numpy(tactile[None])
    return move_to_device(obs_torch, device)
