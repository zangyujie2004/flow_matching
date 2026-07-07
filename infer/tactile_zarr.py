from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Sequence

import numpy as np

from infer.preprocess import build_obs_from_numpy_frames, parse_preprocess_config
from infer.types import DEFAULT_TACTILE_STREAMS, PreprocessConfig, tactile_flow_stream_name
from tools.tactile_feat import CHANNELS_PER_BUNDLE, TACTILE_BUNDLE_ORDER


def merged_tactile_to_pointclouds(
    merged: np.ndarray,
    *,
    pointcloud_shape: tuple[int, int, int] = (35, 20, 6),
) -> dict[str, np.ndarray]:
    """Split zarr merged tactile (H,W,24) into per-sensor flat pointclouds (700,6)."""
    arr = np.asarray(merged, dtype=np.float32)
    if arr.shape[-1] != CHANNELS_PER_BUNDLE * len(TACTILE_BUNDLE_ORDER):
        raise ValueError(f"expected merged tactile last dim 24, got {arr.shape[-1]}")
    height, width, channels = pointcloud_shape
    expected_points = height * width
    out: dict[str, np.ndarray] = {}
    for idx, bundle in enumerate(TACTILE_BUNDLE_ORDER):
        bundle_arr = arr[..., idx * CHANNELS_PER_BUNDLE : (idx + 1) * CHANNELS_PER_BUNDLE]
        if bundle_arr.shape[-3:-1] != (height, width):
            raise ValueError(
                f"bundle {bundle} spatial shape {bundle_arr.shape[-3:-1]} != {(height, width)}"
            )
        flat = bundle_arr.reshape(expected_points, channels)
        out[tactile_flow_stream_name(bundle)] = np.ascontiguousarray(flat, dtype=np.float32)
    return out


def build_numpy_frames_from_zarr_tactile(
    dataset: Any,
    window_idx: int,
    *,
    pointcloud_shape: tuple[int, int, int] = (35, 20, 6),
) -> list[SimpleNamespace]:
    """Simulate Prometheus numpy frames using only zarr tactile windows."""
    s0, s1 = dataset.state_range(window_idx)
    tactile_raw = dataset.get_tactile_raw(s0, s1)
    frames: list[SimpleNamespace] = []
    for t in range(s1 - s0):
        stamp_ns = int((t + 1) * 33_000_000)
        pointclouds = merged_tactile_to_pointclouds(tactile_raw[t], pointcloud_shape=pointcloud_shape)
        sample_names = tuple(pointclouds)
        frames.append(
            SimpleNamespace(
                stamp_ns=stamp_ns,
                skew_ms={name: 0.0 for name in sample_names},
                samples={
                    name: SimpleNamespace(stamp_ns=stamp_ns, recv_ns=stamp_ns, data=data)
                    for name, data in pointclouds.items()
                },
            )
        )
    return frames


def compare_deploy_tactile_with_zarr(
    dataset: Any,
    window_idx: int,
    *,
    cfg: PreprocessConfig | None = None,
) -> dict[str, Any]:
    """Compare deploy preprocess tactile obs against ZarrDataset window tactile."""
    if dataset.normalizer is None:
        raise RuntimeError("dataset.normalizer is required")
    if not dataset.use_tactile:
        raise RuntimeError("dataset.use_tactile must be true")

    preprocess_cfg = cfg or PreprocessConfig(use_tactile=True)
    frames = build_numpy_frames_from_zarr_tactile(
        dataset,
        window_idx,
        pointcloud_shape=preprocess_cfg.tactile_pointcloud_shape,
    )
    obs, _state_raw = build_obs_from_numpy_frames(
        frames,
        preprocess_cfg,
        dataset.normalizer,
        window_size=len(frames),
    )

    s0, s1 = dataset.state_range(window_idx)
    expected = dataset.normalizer.normalize_tactile_np(dataset.get_tactile(s0, s1))
    deploy = np.asarray(obs["tactile"], dtype=np.float32)
    expected = np.asarray(expected, dtype=np.float32)
    if deploy.shape != expected.shape:
        raise ValueError(f"deploy tactile shape {deploy.shape} != zarr {expected.shape}")
    l1 = float(np.mean(np.abs(deploy - expected)))
    return {
        "window_idx": int(window_idx),
        "shape": tuple(deploy.shape),
        "l1_mean": l1,
        "max_abs": float(np.max(np.abs(deploy - expected))),
    }
