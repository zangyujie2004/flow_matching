"""Tactile deformation extraction: 4 sensors x (dx, dy, dz) -> (H, W, 12)."""

from __future__ import annotations

import numpy as np

# Matches preprocess export order.
TACTILE_BUNDLE_ORDER = (
    "left_wrist_0",
    "left_wrist_1",
    "right_wrist_0",
    "right_wrist_1",
)
CHANNELS_PER_BUNDLE = 6
# Per taxel: [x, y, z, dx, dy, dz]
DEFORMATION_SLICE = slice(3, 6)
TACTILE_FEATURE_DIM = len(TACTILE_BUNDLE_ORDER) * 3  # 12


def extract_tactile_deformation(tactile: np.ndarray) -> np.ndarray:
    """
    Extract spatial deformation map from merged pointcloud.

    Input:  (..., H, W, 24)  — 4 bundles x 6 channels
    Output: (..., H, W, 12)  — 4 sensors x (dx, dy, dz) on channel dim
    """
    arr = np.asarray(tactile, dtype=np.float32)
    expected_c = CHANNELS_PER_BUNDLE * len(TACTILE_BUNDLE_ORDER)
    if arr.shape[-1] != expected_c:
        raise ValueError(f"expected tactile last dim {expected_c}, got {arr.shape[-1]}")
    if arr.ndim < 4:
        raise ValueError(f"tactile must be at least 4-D (T,H,W,C), got {arr.shape}")

    feats = []
    for b in range(len(TACTILE_BUNDLE_ORDER)):
        bundle = arr[..., b * CHANNELS_PER_BUNDLE : (b + 1) * CHANNELS_PER_BUNDLE]
        feats.append(bundle[..., DEFORMATION_SLICE])  # (..., H, W, 3)
    return np.concatenate(feats, axis=-1)
