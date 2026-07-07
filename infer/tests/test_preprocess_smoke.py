from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
import pytest

from infer.preprocess import (
    build_obs_from_frames,
    build_obs_from_numpy_frames,
    parse_preprocess_config,
    resize_rgb_like_training,
)
import infer.runtime as runtime_mod
from infer.runtime import FMInferenceRuntime
from infer.types import DEFAULT_JOINT_NAMES, DEFAULT_TACTILE_STREAMS, PreprocessConfig


class _IdentityNormalizer:
    def normalize_state_np(self, state: np.ndarray) -> np.ndarray:
        return np.asarray(state, dtype=np.float32)

    def normalize_tactile_np(self, tactile: np.ndarray) -> np.ndarray:
        return np.asarray(tactile, dtype=np.float32)


def _robot_state_msg() -> SimpleNamespace:
    names = list(DEFAULT_JOINT_NAMES)
    positions = np.linspace(0.0, 0.13, num=len(names), dtype=np.float32).tolist()
    return SimpleNamespace(name=names, position=positions)


def _rgb_msg(image: np.ndarray) -> SimpleNamespace:
    h, w = image.shape[:2]
    return SimpleNamespace(
        encoding="rgb8",
        height=h,
        width=w,
        step=w * 3,
        data=image.tobytes(),
    )


def _frame(left_rgb: np.ndarray, right_rgb: np.ndarray) -> SimpleNamespace:
    return SimpleNamespace(
        samples={
            "robot_state": SimpleNamespace(msg=_robot_state_msg()),
            "left_wrist_0_color": SimpleNamespace(msg=_rgb_msg(left_rgb)),
            "right_wrist_0_color": SimpleNamespace(msg=_rgb_msg(right_rgb)),
        }
    )


def test_parse_preprocess_prefers_data_camera_views() -> None:
    cfg = {
        "data": {"action_type": "joint", "camera_views": ["left_wrist_0", "right_wrist_0"]},
        "deploy": {"preprocess": {"camera_views": ["base_0_color"]}},
    }
    out = parse_preprocess_config(cfg)
    assert out.action_type == "joint"
    assert out.camera_views == ("left_wrist_0_color", "right_wrist_0_color")


def test_parse_preprocess_fallback_to_preprocess_views() -> None:
    cfg = {
        "data": {"action_type": "joint"},
        "deploy": {"preprocess": {"camera_views": ["base_0_color", "right_wrist_0_color"]}},
    }
    out = parse_preprocess_config(cfg)
    assert out.camera_views == ("base_0_color", "right_wrist_0_color")


def test_resize_rgb_like_training_matches_bilinear_reference() -> None:
    rng = np.random.default_rng(123)
    image = rng.integers(0, 256, size=(19, 27, 3), dtype=np.uint8)

    actual = resize_rgb_like_training(image, image_size=224)

    tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float()
    ref = F.interpolate(tensor, size=(224, 224), mode="bilinear", align_corners=False)
    expected = ref.round().clamp_(0.0, 255.0).to(torch.uint8).squeeze(0).permute(1, 2, 0).numpy()
    np.testing.assert_array_equal(actual, expected)


def test_build_obs_from_frames_joint_two_views_smoke() -> None:
    left = np.full((12, 18, 3), 10, dtype=np.uint8)
    right = np.full((12, 18, 3), 200, dtype=np.uint8)
    frames = [_frame(left, right), _frame(left, right)]
    cfg = PreprocessConfig(
        action_type="joint",
        camera_views=("left_wrist_0_color", "right_wrist_0_color"),
        image_size=16,
    )

    obs, state_raw = build_obs_from_frames(frames, cfg, _IdentityNormalizer(), window_size=2)
    assert obs["image"].shape == (1, 1, 2, 3, 16, 16)
    assert obs["image"].dtype == np.uint8
    assert state_raw.shape == (2, 14)
    assert obs["state"].shape == (2, 14)
    assert int(obs["image"][0, 0, 0].mean()) == 10
    assert int(obs["image"][0, 0, 1].mean()) == 200


def test_runtime_rejects_mismatched_camera_view_count(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = FMInferenceRuntime.__new__(FMInferenceRuntime)
    runtime.n_image_views = 2
    runtime.cfg = {}

    monkeypatch.setattr(
        runtime_mod,
        "parse_preprocess_config",
        lambda cfg, robot=None: PreprocessConfig(
            action_type="joint",
            camera_views=("base_0_color", "left_wrist_0_color", "right_wrist_0_color"),
        ),
    )

    with pytest.raises(ValueError, match="camera_views count does not match"):
        runtime.infer_from_window(frames=[object()])


def _numpy_tactile_frame(*, fill: float, stamp_ns: int) -> SimpleNamespace:
    rng = np.random.default_rng(int(fill * 1000))
    tactile = (rng.random((700, 6), dtype=np.float32) + fill).astype(np.float32)
    sample_names = (
        "robot_state",
        "left_eef",
        "right_eef",
        "left_wrist_0_color",
        "right_wrist_0_color",
        *DEFAULT_TACTILE_STREAMS,
    )
    return SimpleNamespace(
        stamp_ns=stamp_ns,
        skew_ms={name: 0.0 for name in sample_names},
        samples={
            "robot_state": SimpleNamespace(
                stamp_ns=stamp_ns,
                recv_ns=stamp_ns,
                data={"name": list(DEFAULT_JOINT_NAMES), "position": np.zeros(14, dtype=np.float32)},
            ),
            "left_eef": SimpleNamespace(
                stamp_ns=stamp_ns,
                recv_ns=stamp_ns,
                data=np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            ),
            "right_eef": SimpleNamespace(
                stamp_ns=stamp_ns,
                recv_ns=stamp_ns,
                data=np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            ),
          "left_wrist_0_color": SimpleNamespace(
              stamp_ns=stamp_ns,
              recv_ns=stamp_ns,
              data=np.full((12, 18, 3), 10, dtype=np.uint8),
          ),
          "right_wrist_0_color": SimpleNamespace(
              stamp_ns=stamp_ns,
              recv_ns=stamp_ns,
              data=np.full((12, 18, 3), 200, dtype=np.uint8),
          ),
          **{
              stream: SimpleNamespace(stamp_ns=stamp_ns, recv_ns=stamp_ns, data=tactile.copy())
              for stream in DEFAULT_TACTILE_STREAMS
          },
      },
  )


def test_build_obs_from_numpy_frames_tactile_window() -> None:
    frames = [_numpy_tactile_frame(fill=0.1 * idx, stamp_ns=(idx + 1) * 33_000_000) for idx in range(8)]
    cfg = PreprocessConfig(
        action_type="eef",
        camera_views=("left_wrist_0_color", "right_wrist_0_color"),
        image_size=16,
        use_tactile=True,
    )

    obs, state_raw = build_obs_from_numpy_frames(frames, cfg, _IdentityNormalizer(), window_size=8)

    assert state_raw.shape == (8, 20)
    assert obs["state"].shape == (8, 20)
    assert obs["tactile"].shape == (8, 35, 20, 12)
    assert np.all(np.isfinite(obs["tactile"]))
    assert "timestamp" in obs
