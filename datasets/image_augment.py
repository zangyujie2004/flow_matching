from __future__ import annotations

import numpy as np

BRIGHTNESS_RANGE = (0.6, 1.4)
CONTRAST_RANGE = (0.7, 1.3)
SATURATION_RANGE = (0.7, 1.3)

_RGB_WEIGHTS = np.asarray([0.299, 0.587, 0.114], dtype=np.float32)


def apply_photometric_augment(
    img: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Apply brightness / contrast / saturation on uint8 RGB camera data.

    Supports (T, H, W, C) with C a multiple of 3 (multi-view channel concat).
    One random draw is shared across all time steps and views.
    """
    arr = np.asarray(img)
    if arr.ndim < 3 or int(arr.shape[-1]) % 3 != 0:
        raise ValueError(f"expected image shape (..., H, W, 3k), got {arr.shape}")
    if arr.dtype != np.uint8:
        raise ValueError(f"expected uint8 image, got dtype={arr.dtype}")

    orig_shape = arr.shape
    pixels = arr.reshape(-1, 3).astype(np.float32, copy=False)

    brightness = float(rng.uniform(*BRIGHTNESS_RANGE))
    contrast = float(rng.uniform(*CONTRAST_RANGE))
    saturation = float(rng.uniform(*SATURATION_RANGE))

    out = pixels * brightness
    mean = float(out.mean())
    out = (out - mean) * contrast + mean

    gray = out @ _RGB_WEIGHTS
    out = gray[:, None] + saturation * (out - gray[:, None])

    out = np.clip(out, 0.0, 255.0).astype(np.uint8)
    return out.reshape(orig_shape)
