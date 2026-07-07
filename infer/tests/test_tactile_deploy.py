from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import zarr

from infer.preprocess import build_obs_from_numpy_frames, reshape_pointcloud, sample_pointcloud_array
from infer.tactile_zarr import build_numpy_frames_from_zarr_tactile, merged_tactile_to_pointclouds
from infer.types import PreprocessConfig

CHAHUA_ZARR_ROOT = Path("/mnt/workspace/zyj/data/processed/chahua/chahua_0706_1611")


def test_pointcloud_flat_array_roundtrip_700x6() -> None:
    rng = np.random.default_rng(0)
    values = rng.random((700, 6), dtype=np.float32)
    sample = SimpleNamespace(data=values)
    flat = sample_pointcloud_array(sample)
    shaped = reshape_pointcloud(flat, (35, 20, 6))
    assert flat.shape == (700, 6)
    assert shaped.shape == (35, 20, 6)
    assert np.all(np.isfinite(shaped))


def test_build_obs_missing_tactile_stream_fails() -> None:
    class _Norm:
        def normalize_state_np(self, state: np.ndarray) -> np.ndarray:
            return state

        def normalize_tactile_np(self, tactile: np.ndarray) -> np.ndarray:
            return tactile

    frame = SimpleNamespace(
        stamp_ns=1,
        skew_ms={},
        samples={
            "robot_state": SimpleNamespace(
                data={
                    "name": [f"left_joint_{i}" for i in range(1, 7)] + ["left_gripper"]
                    + [f"right_joint_{i}" for i in range(1, 7)] + ["right_gripper"],
                    "position": np.zeros(14, dtype=np.float32),
                },
            ),
            "left_wrist_0_color": SimpleNamespace(data=np.zeros((8, 8, 3), dtype=np.uint8)),
        },
    )
    cfg = PreprocessConfig(
        action_type="joint",
        camera_views=("left_wrist_0_color",),
        image_size=8,
        use_tactile=True,
    )
    with pytest.raises(KeyError, match="tactile stream"):
        build_obs_from_numpy_frames([frame], cfg, _Norm(), window_size=1)


@pytest.mark.skipif(not CHAHUA_ZARR_ROOT.is_dir(), reason="chahua zarr not available")
def test_deploy_tactile_deformation_matches_zarr_slice() -> None:
    from tools.tactile_feat import extract_tactile_deformation

    root = zarr.open_group(str(CHAHUA_ZARR_ROOT / "replay_buffer.zarr"), mode="r")
    tactile_raw = np.asarray(root["data"]["tactile"][100:108], dtype=np.float32)

    class _IdentityNorm:
        def normalize_state_np(self, state: np.ndarray) -> np.ndarray:
            return state

        def normalize_tactile_np(self, tactile: np.ndarray) -> np.ndarray:
            return tactile

    frames = []
    for t, merged in enumerate(tactile_raw):
        stamp_ns = int((t + 1) * 33_000_000)
        pointclouds = merged_tactile_to_pointclouds(merged)
        frames.append(
            SimpleNamespace(
                stamp_ns=stamp_ns,
                skew_ms={name: 0.0 for name in pointclouds},
                samples={
                    name: SimpleNamespace(stamp_ns=stamp_ns, recv_ns=stamp_ns, data=data)
                    for name, data in pointclouds.items()
                },
            )
        )

    cfg = PreprocessConfig(use_tactile=True)
    from infer.preprocess import build_tactile_window

    deploy = build_tactile_window(frames, cfg, _IdentityNorm())
    expected = extract_tactile_deformation(tactile_raw)
    assert deploy.shape == expected.shape == (8, 35, 20, 12)
    np.testing.assert_allclose(deploy, expected, rtol=0.0, atol=1e-6)
