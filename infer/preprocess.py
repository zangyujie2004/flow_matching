from __future__ import annotations

from typing import Any, Mapping, Sequence
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from infer.tensor import as_float32_array
from infer.types import (
    DEFAULT_JOINT_NAMES,
    DEFAULT_TACTILE_POINTCLOUD_SHAPE,
    DEFAULT_TACTILE_STREAMS,
    PreprocessConfig,
    tactile_flow_stream_name,
)
from tools.normalizer import DatasetNormalizer
from tools.tactile_feat import TACTILE_BUNDLE_ORDER, extract_tactile_deformation


def _bundle_to_color_stream_name(name: Any) -> str:
    stream = str(name)
    return stream if stream.endswith("_color") else f"{stream}_color"


def _resolve_camera_views(
    data_cfg: Mapping[str, Any],
    preprocess_cfg: Mapping[str, Any],
) -> tuple[str, ...]:
    # Prefer data.camera_views so deployment behavior follows the run_dir config.
    data_views = data_cfg.get("camera_views")
    if data_views is not None:
        return tuple(_bundle_to_color_stream_name(name) for name in data_views)

    # Fallback to explicit deploy.preprocess override when data.camera_views is absent.
    camera_views = preprocess_cfg.get("camera_views")
    if camera_views is not None:
        return tuple(str(name) for name in camera_views)

    return tuple(PreprocessConfig().camera_views)


def _resolve_tactile_streams(
    data_cfg: Mapping[str, Any],
    preprocess_cfg: Mapping[str, Any],
) -> tuple[bool, tuple[str, ...]]:
    use_tactile = bool(preprocess_cfg.get("use_tactile", data_cfg.get("use_tactile", False)))
    if not use_tactile:
        return False, ()

    bundles = preprocess_cfg.get("tactile_bundles")
    if bundles is None:
        bundles = TACTILE_BUNDLE_ORDER
    streams = preprocess_cfg.get("tactile_streams")
    if streams is not None:
        return True, tuple(str(name) for name in streams)
    return True, tuple(tactile_flow_stream_name(str(bundle)) for bundle in bundles)


def _resolve_pointcloud_shape(
    data_cfg: Mapping[str, Any],
    preprocess_cfg: Mapping[str, Any],
) -> tuple[int, int, int]:
    shape = preprocess_cfg.get("tactile_pointcloud_shape", data_cfg.get("tactile_pointcloud_shape"))
    if shape is None:
        return DEFAULT_TACTILE_POINTCLOUD_SHAPE
    values = tuple(int(v) for v in shape)
    if len(values) != 3:
        raise ValueError(f"tactile_pointcloud_shape must have 3 dims, got {shape!r}")
    return values


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

    camera_views = _resolve_camera_views(data_cfg, preprocess_cfg)
    gripper_names = preprocess_cfg.get("gripper_names")
    if gripper_names is None:
        gripper_names = robot_cfg.get("gripper_names", PreprocessConfig().gripper_names)
    joint_names = preprocess_cfg.get("joint_names")
    if joint_names is None:
        joint_names = DEFAULT_JOINT_NAMES

    use_tactile, tactile_streams = _resolve_tactile_streams(data_cfg, preprocess_cfg)

    return PreprocessConfig(
        action_type=action_type,
        camera_views=camera_views,
        gripper_names={str(k): str(v) for k, v in dict(gripper_names).items()},
        joint_names=tuple(str(name) for name in joint_names),
        image_size=int(preprocess_cfg.get("image_size", data_cfg.get("image_size", 224))),
        gripper_width_m=float(
            preprocess_cfg.get(
                "gripper_width_m",
                robot_cfg.get("gripper_width_m", PreprocessConfig.gripper_width_m),
            )
        ),
        use_tactile=use_tactile,
        tactile_streams=tactile_streams if tactile_streams else DEFAULT_TACTILE_STREAMS,
        tactile_pointcloud_shape=_resolve_pointcloud_shape(data_cfg, preprocess_cfg),
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


def _frame_samples(frame: Any) -> Mapping[str, Any]:
    samples = getattr(frame, "samples", frame)
    if not isinstance(samples, Mapping):
        raise TypeError("frame must expose a mapping-like samples attribute")
    return samples


def _robot_state_values(sample: Any) -> tuple[list[str], list[float]]:
    if hasattr(sample, "msg"):
        msg = sample.msg
        return list(getattr(msg, "name", [])), list(getattr(msg, "position", []))
    if hasattr(sample, "data"):
        data = sample.data
        if isinstance(data, Mapping):
            names = [str(item) for item in np.asarray(data["name"], dtype=object).tolist()]
            positions = np.asarray(data["position"], dtype=np.float32).tolist()
            return names, [float(v) for v in positions]
    raise TypeError("robot_state sample must expose msg or mapping data")


def _eef_values(sample: Any) -> Any:
    if hasattr(sample, "msg"):
        return sample.msg
    if hasattr(sample, "data"):
        return sample.data
    raise TypeError("eef sample must expose msg or data")


def pointcloud2_to_numpy(msg: Any) -> np.ndarray:
    field_names = tuple(field.name for field in getattr(msg, "fields", ()))
    selected = tuple(name for name in ("x", "y", "z", "dx", "dy", "dz") if name in field_names)
    if not selected:
        raise ValueError("PointCloud2 has no supported tactile fields")
    import sensor_msgs_py.point_cloud2 as pc2

    points = pc2.read_points(msg, field_names=selected, skip_nans=False)
    if isinstance(points, np.ndarray) and points.dtype.fields:
        array = np.column_stack([points[field] for field in selected]).astype(np.float32)
    else:
        array = np.asarray(points if isinstance(points, np.ndarray) else list(points), dtype=np.float32)
    if array.ndim == 1:
        array = array.reshape(0, len(selected)) if array.size == 0 else array.reshape(-1, len(selected))
    return np.ascontiguousarray(array, dtype=np.float32)


def sample_pointcloud_array(sample: Any) -> np.ndarray:
    if hasattr(sample, "data"):
        return as_float32_array(sample.data, name="tactile_pointcloud")
    if hasattr(sample, "msg"):
        return pointcloud2_to_numpy(sample.msg)
    raise TypeError("tactile sample must expose data or msg")


def reshape_pointcloud(flat: np.ndarray, shape: tuple[int, int, int]) -> np.ndarray:
    height, width, channels = shape
    arr = as_float32_array(flat, name="tactile_pointcloud")
    expected_points = height * width
    if arr.ndim == 1 and arr.size == expected_points * channels:
        arr = arr.reshape(expected_points, channels)
    if arr.shape != (expected_points, channels):
        raise ValueError(
            f"tactile pointcloud shape {arr.shape} != ({expected_points}, {channels})"
        )
    return arr.reshape(height, width, channels)


def build_tactile_frame(samples: Mapping[str, Any], cfg: PreprocessConfig) -> np.ndarray:
    bundles: list[np.ndarray] = []
    for stream_name in cfg.tactile_streams:
        if stream_name not in samples:
            raise KeyError(f"frame missing tactile stream {stream_name!r}")
        pointcloud = reshape_pointcloud(
            sample_pointcloud_array(samples[stream_name]),
            cfg.tactile_pointcloud_shape,
        )
        bundles.append(pointcloud)
    merged = np.concatenate(bundles, axis=-1)
    return extract_tactile_deformation(merged[np.newaxis, ...])[0]


def build_tactile_window(
    frames: Sequence[Any],
    cfg: PreprocessConfig,
    normalizer: DatasetNormalizer,
) -> np.ndarray:
    if not cfg.use_tactile:
        raise RuntimeError("build_tactile_window called with use_tactile=False")
    tactile = np.stack(
        [build_tactile_frame(_frame_samples(frame), cfg) for frame in frames],
        axis=0,
    )
    tactile = as_float32_array(tactile, name="tactile_window")
    expected = (len(frames), *cfg.tactile_feature_shape)
    if tactile.shape != expected:
        raise ValueError(f"tactile window shape {tactile.shape} != expected {expected}")
    return as_float32_array(normalizer.normalize_tactile_np(tactile), name="tactile_norm")


def timestamp_dict(frames: Sequence[Any]) -> dict[str, Any]:
    names = tuple(_frame_samples(frames[0]))
    return {
        "frame": np.asarray([int(frame.stamp_ns) for frame in frames], dtype=np.int64),
        "samples": {
            name: np.asarray([int(_frame_samples(frame)[name].stamp_ns) for frame in frames], dtype=np.int64)
            for name in names
        },
        "recv": {
            name: np.asarray([int(_frame_samples(frame)[name].recv_ns) for frame in frames], dtype=np.int64)
            for name in names
        },
        "skew_ms": {
            name: np.asarray([float(frame.skew_ms[name]) for frame in frames], dtype=np.float32)
            for name in names
        },
    }


def build_state_frame(frame: Any, cfg: PreprocessConfig) -> np.ndarray:
    samples = _frame_samples(frame)

    if cfg.action_type == "joint":
        names, positions = _robot_state_values(samples["robot_state"])
        return joint_qpos_from_robot_state(
            SimpleNamespace(name=names, position=positions),
            cfg.joint_names,
            gripper_width_m=cfg.gripper_width_m,
        )

    names, positions = _robot_state_values(samples["robot_state"])
    robot_state = SimpleNamespace(name=names, position=positions)
    left = build_arm_eef10_from_values(
        _eef_values(samples["left_eef"]),
        gripper_opening_m(robot_state, cfg.gripper_names["left"], cfg.gripper_width_m),
    )
    right = build_arm_eef10_from_values(
        _eef_values(samples["right_eef"]),
        gripper_opening_m(robot_state, cfg.gripper_names["right"], cfg.gripper_width_m),
    )
    return np.concatenate([left, right], axis=0).astype(np.float32, copy=False)


def build_arm_eef10_from_values(eef_values: Any, gripper_m: float) -> np.ndarray:
    if hasattr(eef_values, "pose"):
        return build_arm_eef10(eef_values, gripper_m)
    eef7 = np.asarray(eef_values, dtype=np.float32)
    xyz = eef7[:3]
    rot6d = quat_wxyz_to_rot6d(eef7[3:7])
    return np.concatenate([xyz, rot6d, np.asarray([gripper_m], dtype=np.float32)], axis=0).astype(
        np.float32, copy=False
    )

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


def build_dino_images(frame: Any, cfg: PreprocessConfig) -> list[torch.Tensor]:
    """Convert the configured two or three camera images to NCHW uint8 tensors."""

    if len(cfg.camera_views) not in {2, 3}:
        raise ValueError("Async DINO needs two or three camera views")
    samples = _frame_samples(frame)
    images = []
    for name in cfg.camera_views:
        if name not in samples:
            raise KeyError(f"frame missing camera stream {name!r}")
        sample = samples[name]
        if hasattr(sample, "data"):
            rgb = np.asarray(sample.data, dtype=np.uint8)
        else:
            rgb = ros_image_to_rgb(sample.msg)
        resized = resize_rgb_like_training(rgb, cfg.image_size)
        image = torch.from_numpy(np.ascontiguousarray(resized.transpose(2, 0, 1)))
        images.append(image.unsqueeze(0))
    return images

def build_obs_from_frames(
    frames: Sequence[Any],
    cfg: PreprocessConfig,
    normalizer: DatasetNormalizer,
    *,
    window_size: int | None = None,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    obs, state_raw = _build_core_obs(frames, cfg, normalizer, window_size=window_size)
    return obs, state_raw


def build_obs_from_numpy_frames(
    frames: Sequence[Any],
    cfg: PreprocessConfig,
    normalizer: DatasetNormalizer,
    *,
    window_size: int | None = None,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    obs, state_raw = _build_core_obs(frames, cfg, normalizer, window_size=window_size)
    obs["timestamp"] = timestamp_dict(frames)
    return obs, state_raw


def _build_core_obs(
    frames: Sequence[Any],
    cfg: PreprocessConfig,
    normalizer: DatasetNormalizer,
    *,
    window_size: int | None,
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
    samples = _frame_samples(anchor)
    views: list[np.ndarray] = []
    for stream_name in cfg.camera_views:
        if stream_name not in samples:
            raise KeyError(f"anchor frame missing camera stream {stream_name!r}")
        sample = samples[stream_name]
        if hasattr(sample, "data"):
            rgb = np.asarray(sample.data, dtype=np.uint8)
            if rgb.ndim != 3 or rgb.shape[2] != 3:
                raise ValueError(f"camera image must be HxWx3 RGB uint8, got {rgb.shape}")
        else:
            rgb = ros_image_to_rgb(sample.msg)
        resized = resize_rgb_like_training(rgb, cfg.image_size)
        views.append(np.transpose(resized, (2, 0, 1)).astype(np.uint8, copy=False))

    image_views = np.stack(views, axis=0)
    obs: dict[str, np.ndarray] = {
        "image": np.asarray(image_views[None, None, ...], dtype=np.uint8),
        "state": state_norm,
    }
    if cfg.use_tactile:
        obs["tactile"] = build_tactile_window(frames, cfg, normalizer)
    return obs, state_raw


def resize_rgb_like_training(rgb: np.ndarray, image_size: int) -> np.ndarray:
    arr = np.asarray(rgb, dtype=np.uint8)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"expected RGB image shape (H, W, 3), got {arr.shape}")
    if arr.shape[0] == image_size and arr.shape[1] == image_size:
        return arr

    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).float()
    resized = F.interpolate(
        tensor,
        size=(image_size, image_size),
        mode="bilinear",
        align_corners=False,
    )
    out = resized.round().clamp_(0.0, 255.0).to(torch.uint8)
    return out.squeeze(0).permute(1, 2, 0).cpu().numpy()

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
