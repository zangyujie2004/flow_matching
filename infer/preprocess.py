from __future__ import annotations

from typing import Any, Mapping, Sequence

import cv2
import numpy as np
from infer.tensor import as_float32_array
from infer.types import DEFAULT_JOINT_NAMES, PreprocessConfig
from tools.normalizer import DatasetNormalizer


def parse_preprocess_config(
    cfg: Mapping[str, Any],
    *,
    robot: Mapping[str, Any] | None = None,
) -> PreprocessConfig:
    data_cfg = dict(cfg.get("data") or {})
    deploy_cfg = dict(cfg.get("deploy") or {})
    preprocess_cfg = dict(deploy_cfg.get("preprocess") or {})
    robot_cfg = dict(robot or {})
    action_type = str(preprocess_cfg.get("action_type", data_cfg.get("action_type", "eef")))

    camera_views = preprocess_cfg.get("camera_views")
    if camera_views is None:
        camera_views = PreprocessConfig().camera_views
    gripper_names = preprocess_cfg.get("gripper_names")
    if gripper_names is None:
        gripper_names = robot_cfg.get("gripper_names", PreprocessConfig().gripper_names)
    joint_names = preprocess_cfg.get("joint_names")
    if joint_names is None:
        joint_names = DEFAULT_JOINT_NAMES

    return PreprocessConfig(
        action_type=action_type,
        camera_views=tuple(str(name) for name in camera_views),
        gripper_names={str(k): str(v) for k, v in dict(gripper_names).items()},
        joint_names=tuple(str(name) for name in joint_names),
        image_size=int(preprocess_cfg.get("image_size", data_cfg.get("image_size", 224))),
        gripper_width_m=float(
            preprocess_cfg.get(
                "gripper_width_m",
                robot_cfg.get("gripper_width_m", PreprocessConfig.gripper_width_m),
            )
        ),
    )


def quat_wxyz_to_rot6d(quat_wxyz: Any) -> np.ndarray:
    q = np.asarray(quat_wxyz, dtype=np.float64)
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    norm = np.where(norm < 1e-12, np.nan, norm)
    q = q / norm
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]

    r00 = 1 - 2 * (y * y + z * z)
    r10 = 2 * (x * y + z * w)
    r20 = 2 * (x * z - y * w)
    r01 = 2 * (x * y - z * w)
    r11 = 1 - 2 * (x * x + z * z)
    r21 = 2 * (y * z + x * w)
    return np.stack([r00, r10, r20, r01, r11, r21], axis=-1).astype(np.float32)

def pose_stamped_to_eef7(msg: Any) -> np.ndarray:
    pose = msg.pose
    xyz = np.array(
        [float(pose.position.x), float(pose.position.y), float(pose.position.z)],
        dtype=np.float32,
    )
    quat = np.array(
        [
            float(pose.orientation.w),
            float(pose.orientation.x),
            float(pose.orientation.y),
            float(pose.orientation.z),
        ],
        dtype=np.float32,
    )
    out = np.concatenate([xyz, quat], axis=0)
    if not np.all(np.isfinite(out)):
        raise ValueError("PoseStamped contains NaN or inf")
    return out

def gripper_opening_m(msg: Any, joint_name: str, gripper_width_m: float) -> float:
    names = list(getattr(msg, "name", []))
    positions = list(getattr(msg, "position", []))
    if joint_name not in names:
        raise ValueError(f"robot_state does not contain gripper joint {joint_name!r}")
    return float(np.clip(positions[names.index(joint_name)], 0.0, gripper_width_m))

def joint_qpos_from_robot_state(
    msg: Any,
    joint_names: Sequence[str],
    *,
    gripper_width_m: float,
) -> np.ndarray:
    names = list(getattr(msg, "name", []))
    positions = list(getattr(msg, "position", []))
    values: list[float] = []
    for joint_name in joint_names:
        if joint_name not in names:
            raise ValueError(f"robot_state does not contain joint {joint_name!r}")
        value = float(positions[names.index(joint_name)])
        if joint_name.endswith("_gripper"):
            value = float(np.clip(value, 0.0, gripper_width_m))
        values.append(value)
    out = np.asarray(values, dtype=np.float32)
    if not np.all(np.isfinite(out)):
        raise ValueError("robot_state joint positions contain NaN or inf")
    return out


def build_arm_eef10(pose_msg: Any, gripper_m: float) -> np.ndarray:
    eef7 = pose_stamped_to_eef7(pose_msg)
    xyz = eef7[:3]
    rot6d = quat_wxyz_to_rot6d(eef7[3:7])
    gripper = np.asarray([gripper_m], dtype=np.float32)
    return np.concatenate([xyz, rot6d, gripper], axis=0).astype(np.float32, copy=False)

def build_state_frame(frame: Any, cfg: PreprocessConfig) -> np.ndarray:
    samples = getattr(frame, "samples", frame)
    if not isinstance(samples, Mapping):
        raise TypeError("frame must expose a mapping-like samples attribute")

    if cfg.action_type == "joint":
        robot_state = samples["robot_state"].msg
        return joint_qpos_from_robot_state(
            robot_state,
            cfg.joint_names,
            gripper_width_m=cfg.gripper_width_m,
        )

    robot_state = samples["robot_state"].msg
    left = build_arm_eef10(
        samples["left_eef"].msg,
        gripper_opening_m(robot_state, cfg.gripper_names["left"], cfg.gripper_width_m),
    )
    right = build_arm_eef10(
        samples["right_eef"].msg,
        gripper_opening_m(robot_state, cfg.gripper_names["right"], cfg.gripper_width_m),
    )
    return np.concatenate([left, right], axis=0).astype(np.float32, copy=False)

def ros_image_to_rgb(msg: Any) -> np.ndarray:
    image = _image_to_array(msg)
    encoding = str(msg.encoding).lower()
    if encoding == "rgb8":
        return image
    if encoding == "rgba8":
        return image[:, :, :3]
    if encoding == "bgr8":
        return image[:, :, ::-1].copy()
    if encoding == "bgra8":
        return image[:, :, 2::-1].copy()
    raise ValueError(f"unsupported RGB image encoding {msg.encoding!r}")

def build_obs_from_frames(
    frames: Sequence[Any],
    cfg: PreprocessConfig,
    normalizer: DatasetNormalizer,
    *,
    window_size: int | None = None,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    if not frames:
        raise ValueError("frames must be non-empty")
    expected = window_size if window_size is not None else len(frames)
    if len(frames) != expected:
        raise ValueError(f"expected {expected} frames, got {len(frames)}")

    state_raw = np.stack([build_state_frame(frame, cfg) for frame in frames], axis=0)
    state_raw = as_float32_array(state_raw, name="state_raw")
    expected_dim = cfg.state_dim
    if state_raw.shape != (expected, expected_dim):
        raise ValueError(
            f"state_raw shape {state_raw.shape} != ({expected}, {expected_dim}) "
            f"for action_type={cfg.action_type!r}"
        )
    state_norm = as_float32_array(normalizer.normalize_state_np(state_raw), name="state_norm")

    anchor = frames[-1]
    samples = getattr(anchor, "samples", anchor)
    views: list[np.ndarray] = []
    for stream_name in cfg.camera_views:
        if stream_name not in samples:
            raise KeyError(f"anchor frame missing camera stream {stream_name!r}")
        rgb = ros_image_to_rgb(samples[stream_name].msg)
        resized = cv2.resize(
            rgb,
            (cfg.image_size, cfg.image_size),
            interpolation=cv2.INTER_AREA,
        )
        views.append(np.transpose(resized, (2, 0, 1)).astype(np.uint8, copy=False))

    image_views = np.stack(views, axis=0)
    image = image_views[None, None, ...]
    obs = {
        "image": np.asarray(image, dtype=np.uint8),
        "state": state_norm,
    }
    return obs, state_raw

def _image_to_array(msg: Any) -> np.ndarray:
    dtype, channels = _image_layout(str(msg.encoding).lower())
    height, width = int(msg.height), int(msg.width)
    row_values = int(msg.step) // np.dtype(dtype).itemsize
    data = np.frombuffer(bytes(msg.data), dtype=dtype).reshape(height, row_values)
    if channels == 1:
        return data[:, :width].copy()
    return data[:, : width * channels].reshape(height, width, channels).copy()


def _image_layout(encoding: str) -> tuple[Any, int]:
    if encoding in {"rgb8", "bgr8"}:
        return np.uint8, 3
    if encoding in {"rgba8", "bgra8"}:
        return np.uint8, 4
    if encoding in {"mono8", "8uc1"}:
        return np.uint8, 1
    if encoding in {"16uc1", "mono16"}:
        return np.uint16, 1
    if encoding == "32fc1":
        return np.float32, 1
    raise ValueError(f"unsupported image encoding {encoding!r}")
