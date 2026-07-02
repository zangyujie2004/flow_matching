from __future__ import annotations

from typing import Any

import numpy as np
import torch

from infer.tensor import as_float32_array


def obs_from_zarr_window(
    dataset: Any,
    window_idx: int,
    *,
    vision_only: bool = True,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """Build deployment-shaped obs from zarr (P0 validation only; not used on robot)."""
    if dataset.normalizer is None:
        raise RuntimeError("dataset.normalizer is required")

    s0, s1 = dataset.state_range(window_idx)
    i0, i1 = dataset.image_range(window_idx)
    a0, a1 = dataset.action_range(window_idx)

    state_raw = as_float32_array(dataset.get_state(s0, s1), name="state_raw")
    state_norm = as_float32_array(
        dataset.normalizer.normalize_state_np(state_raw),
        name="state_norm",
    )
    camera = dataset.get_camera(i0, i1)
    image = dataset._process_image(camera)
    if torch.is_tensor(image):
        image = image.numpy()

    gt_abs = as_float32_array(dataset.get_action(a0, a1), name="gt_action")
    obs: dict[str, np.ndarray] = {
        "state": state_norm,
        "image": np.asarray(image),
    }
    if dataset.use_tactile and not vision_only:
        tactile = dataset.normalizer.normalize_tactile_np(dataset.get_tactile(s0, s1))
        obs["tactile"] = as_float32_array(tactile, name="tactile_norm")
    return obs, state_raw, gt_abs
