from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import yaml
import zarr
from torch.utils.data import DataLoader, Dataset

_POLICY_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _POLICY_ROOT not in sys.path:
    sys.path.insert(0, _POLICY_ROOT)

from tools.latent_cache import (
    resolve_latent_cache_zarr_path,
    validate_latent_cache_identity,
)
from tools.normalizer import DatasetNormalizer
from tools.tactile_feat import TACTILE_FEATURE_DIM, extract_tactile_deformation

from .image_augment import apply_photometric_augment

_ACTION_TYPES = ("joint", "eef")
_ROBOT_SLICES = {"joint": slice(0, 14), "eef": slice(14, 34)}
_ROBOT_DIMS = {"joint": 14, "eef": 20}
CAMERA_BUNDLE_ORDER = ("base_0", "left_wrist_0", "right_wrist_0")


def resolve_camera_views(
    camera_views: Sequence[str] | None,
    *,
    n_zarr_views: int,
) -> tuple[str, ...]:
    if n_zarr_views <= 0 or n_zarr_views * 3 > 256:
        raise ValueError(f"invalid zarr camera view count: {n_zarr_views}")
    available = CAMERA_BUNDLE_ORDER[:n_zarr_views]
    if camera_views is None:
        return available
    selected = tuple(str(name) for name in camera_views)
    if not selected:
        raise ValueError("camera_views must be non-empty when provided")
    unknown = [name for name in selected if name not in CAMERA_BUNDLE_ORDER]
    if unknown:
        raise ValueError(
            f"unknown camera_views={unknown}; expected subset of {CAMERA_BUNDLE_ORDER}"
        )
    missing = [name for name in selected if name not in available]
    if missing:
        raise ValueError(
            f"camera_views={list(selected)} not available in zarr "
            f"(available={list(available)})"
        )
    if len(set(selected)) != len(selected):
        raise ValueError(f"camera_views must be unique, got {list(selected)}")
    return selected


def camera_view_indices(views: Sequence[str]) -> tuple[int, ...]:
    return tuple(CAMERA_BUNDLE_ORDER.index(str(name)) for name in views)


def cache_view_indices(
    requested_views: Sequence[str],
    cache_views: Sequence[str],
) -> tuple[int, ...]:
    cache_view_list = tuple(str(name) for name in cache_views)
    missing = [name for name in requested_views if str(name) not in cache_view_list]
    if missing:
        raise ValueError(
            "Latent cache camera_views missing requested views: "
            f"cache={list(cache_view_list)!r}, requested={list(requested_views)!r}"
        )
    return tuple(cache_view_list.index(str(name)) for name in requested_views)


def camera_channel_indices(views: Sequence[str]) -> tuple[int, ...]:
    channels: list[int] = []
    for view_idx in camera_view_indices(views):
        start = view_idx * 3
        channels.extend(range(start, start + 3))
    return tuple(channels)


def parse_cache_camera_views(cache_views: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in str(cache_views).split(",") if part.strip())


def resolve_camera_data_config(data_cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Apply camera_augmentation ↔ use_camera_latent rules from workload."""
    cfg = dict(data_cfg)
    if bool(cfg.get("camera_augmentation", False)):
        if bool(cfg.get("use_camera_latent", False)):
            print("[ZarrDataset] camera_augmentation=true; forcing use_camera_latent=false")
        cfg["use_camera_latent"] = False
    return cfg


class ZarrDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        window_size: int = 8,
        stride: int = 1,
        n_image_steps: int = 1,
        action_horizon: int = 32,
        action_type: str = "eef",
        action_representation: str = "relative",
        use_tactile: bool = True,
        image_size: int = 224,
        image_as_uint8: bool = True,
        use_camera_latent: bool = False,
        latent_cache_root_dir: str | None = None,
        latent_cache_image_encoder_name: str | None = None,
        latent_cache_image_model_name: str | None = None,
        fit_normalizer: bool = True,
        camera_key: str = "camera",
        tactile_key: str = "tactile",
        state_key: str = "state_30hz",
        action_key: str = "action_30hz",
        camera_views: Sequence[str] | None = None,
        camera_augmentation: bool = False,
        memory: Mapping[str, Any] | None = None,
        norm_output_range: Tuple[float, float] = (-1.0, 1.0),
        normalizer_max_windows: int | None = None,
        max_windows: int | None = None,
    ):
        self.root_dir = root_dir
        self.window_size = max(1, int(window_size))
        self.stride = max(1, int(stride))
        self.image_size = int(image_size)
        self.max_windows = None if max_windows is None else max(1, int(max_windows))

        self.n_image_steps = self.window_size if n_image_steps is None else max(1, int(n_image_steps))
        self.action_horizon = self.window_size if action_horizon is None else max(1, int(action_horizon))

        self.action_type = self._resolve_action_type(action_type)
        self.action_representation = self._normalize_action_representation(action_representation)
        self.robot_slice = _ROBOT_SLICES[self.action_type]
        self.action_dim = _ROBOT_DIMS[self.action_type]

        self.use_tactile = bool(use_tactile)
        self.tactile_dim = TACTILE_FEATURE_DIM
        self.image_as_uint8 = bool(image_as_uint8)
        self.use_camera_latent = bool(use_camera_latent)
        self.camera_augmentation = bool(camera_augmentation)
        self.latent_cache_root_dir = latent_cache_root_dir
        self.latent_cache_image_encoder_name = latent_cache_image_encoder_name
        self.latent_cache_image_model_name = latent_cache_image_model_name
        self.fit_normalizer = bool(fit_normalizer)
        self.training = True

        memory_cfg = dict(memory or {})
        self.memory_enabled = bool(memory_cfg.get("enabled", False))
        self.memory_history_frames = max(1, int(memory_cfg.get("history_frames", 64)))
        self.memory_visual_history_length = max(
            1, int(memory_cfg.get("visual_history_length", 64))
        )
        self.memory_sample_stride = max(1, int(memory_cfg.get("sample_stride", 8)))
        self.memory_recent_frame = max(1, int(memory_cfg.get("recent_frame", 2)))
        self.memory_visual_recent_frame = max(
            0, int(memory_cfg.get("visual_recent_frame", 0))
        )
        # Locked: pad_first only (ignore any start_mode in config).
        self.memory_start_mode = "pad_first"
        self.memory_visual_offsets = self._build_memory_visual_offsets()

        if self.camera_augmentation and self.use_camera_latent:
            raise ValueError(
                "camera_augmentation=true is incompatible with use_camera_latent=true. "
                "Set use_camera_latent=false or disable camera_augmentation."
            )

        self.latent_cache_zarr = None
        self.latent_cache_group = None
        self.cached_image_backbone_feat = None
        self.cached_frame_image_backbone_feat = None
        self._frame_latent_view_indices: tuple[int, ...] | None = None
        self.image_backbone_dim: int | None = None
        self.cached_norm_action: np.ndarray | None = None
        self.camera_key = camera_key
        self.tactile_key = tactile_key
        self.state_key = state_key
        self.action_key = action_key
        self.norm_output_range = (float(norm_output_range[0]), float(norm_output_range[1]))
        self.normalizer_max_windows = normalizer_max_windows

        self.zarr_path = self._resolve_zarr_path(root_dir)
        self.zarr_root = zarr.open_group(self.zarr_path, mode="r")
        self.data_group = self.zarr_root["data"]
        self.meta_group = self.zarr_root["meta"]

        self.ram_data: Dict[str, np.ndarray] = {}

        self._validate_required_keys()
        self.episode_ends = np.asarray(self.meta_group["episode_ends"][:], dtype=np.int64)
        self.episode_starts = np.concatenate([np.array([0], dtype=np.int64), self.episode_ends[:-1]])

        self._preload_to_ram()
        n_zarr_views = self._zarr_camera_view_count()
        self.camera_views = resolve_camera_views(camera_views, n_zarr_views=n_zarr_views)
        self._camera_channel_indices = camera_channel_indices(self.camera_views)
        self.n_image_views = len(self.camera_views)
        print(
            f"[ZarrDataset] camera_views={list(self.camera_views)} "
            f"(n_image_views={self.n_image_views}, zarr_views={n_zarr_views})"
        )
        if self.memory_enabled:
            print(
                "[ZarrDataset] memory enabled: "
                f"state_history_frames={self.memory_history_frames}, "
                f"visual_history_length={self.memory_visual_history_length}, "
                f"visual_sample_stride={self.memory_sample_stride}, "
                f"state_recent_frame={self.memory_recent_frame}, "
                f"visual_recent_frame={self.memory_visual_recent_frame}, "
                f"visual_offsets={self.memory_visual_offsets.tolist()}, "
                f"start_mode={self.memory_start_mode}"
            )

        self.windows = self._build_windows()
        if self.max_windows is not None:
            cap = max(1, int(self.max_windows))
            if len(self.windows) > cap:
                print(f"[ZarrDataset] truncating windows {len(self.windows)} -> {cap} (data.max_windows)")
                self.windows = self.windows[:cap]
        if len(self.windows) == 0:
            raise ValueError(
                "No valid strict anchor windows. "
                f"window_size={self.window_size}, n_image_steps={self.n_image_steps}, "
                f"action_horizon={self.action_horizon}, stride={self.stride}"
            )
        self.window_lookup = {
            (int(anchor_t), int(ep_idx)): idx
            for idx, (anchor_t, _ep_end, ep_idx) in enumerate(self.windows)
        }

        if self.use_camera_latent:
            self._maybe_open_latent_cache()
        elif self.latent_cache_root_dir:
            cache_path = self._resolve_latent_cache_path(require_exists=False)
            if cache_path is not None and os.path.isdir(cache_path):
                print(
                    f"[ZarrDataset] latent cache found at {cache_path} "
                    "(ignored because use_camera_latent=false)"
                )

        if self.fit_normalizer:
            print("[ZarrDataset] fitting normalizer on full windows...")
            self.normalizer = DatasetNormalizer.build(
                self,
                output_range=self.norm_output_range,
                max_windows=self.normalizer_max_windows,
            )
            self._precompute_normalized_actions()
        else:
            self.normalizer = None
            self.cached_norm_action = None

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "ZarrDataset":
        """Build dataset from policy/configs/config.yaml data section."""
        cfg = resolve_camera_data_config(config)
        cfg = dict(cfg)
        norm = cfg.pop("norm", None) or {}
        if "output_range" in norm:
            out = norm["output_range"]
            cfg["norm_output_range"] = (float(out[0]), float(out[1]))
        if "max_windows" in norm:
            cfg["normalizer_max_windows"] = norm["max_windows"]
        if "fit_normalizer" in cfg:
            cfg["fit_normalizer"] = bool(cfg["fit_normalizer"])
        memory = cfg.pop("memory", None)
        return cls(**cfg, memory=memory)

    def set_training(self, training: bool) -> None:
        self.training = bool(training)

    def _maybe_augment_camera(self, camera: np.ndarray) -> np.ndarray:
        if not self.camera_augmentation or not self.training:
            return camera
        rng = np.random.default_rng()
        return apply_photometric_augment(camera, rng)

    @staticmethod
    def _resolve_action_type(value: str) -> str:
        key = str(value).strip().lower()
        if key not in _ACTION_TYPES:
            raise ValueError(f"action_type must be one of {_ACTION_TYPES}, got {value!r}")
        return key

    @staticmethod
    def _normalize_action_representation(value: str) -> str:
        rep = str(value).strip().lower()
        if rep in {"relative", "relative_action", "chunk_relative"}:
            rep = "relative"
        if rep not in {"absolute", "relative"}:
            raise ValueError(
                f"Unsupported action_representation={value}. Choose from ['absolute', 'relative']."
            )
        return rep

    @staticmethod
    def _resolve_zarr_path(root_dir: str) -> str:
        if root_dir.endswith(".zarr") and os.path.isdir(root_dir):
            return root_dir
        zarr_path = os.path.join(root_dir, "replay_buffer.zarr")
        if os.path.isdir(zarr_path):
            return zarr_path
        raise FileNotFoundError(
            f"Cannot find replay_buffer.zarr from root_dir={root_dir}. Tried: {zarr_path}"
        )

    def _data_keys_to_load(self) -> List[str]:
        keys = [self.state_key, self.action_key]
        if not self.use_camera_latent:
            keys.insert(0, self.camera_key)
        if self.use_tactile:
            keys.append(self.tactile_key)
        return keys

    def _zarr_camera_view_count(self) -> int:
        if self.camera_key not in self.data_group:
            raise KeyError(f"Missing key in zarr data group: {self.camera_key}")
        return int(self.data_group[self.camera_key].shape[-1] // 3)

    def _validate_required_keys(self) -> None:
        for key in self._data_keys_to_load():
            if key not in self.data_group:
                raise KeyError(f"Missing key in zarr data group: {key}")
        if "episode_ends" not in self.meta_group:
            raise KeyError("Missing key in zarr meta group: episode_ends")
        if self.camera_key not in self.data_group:
            raise KeyError(f"Missing key in zarr data group: {self.camera_key}")

    def _resolve_latent_cache_path(self, *, require_exists: bool = True) -> str | None:
        root = self.latent_cache_root_dir or self.root_dir
        path = resolve_latent_cache_zarr_path(str(root))
        if require_exists and not os.path.isdir(path):
            raise FileNotFoundError(
                f"Latent cache not found: {path}. "
                "Run ./scripts/precompute.sh first, then set data.use_camera_latent=true."
            )
        if not os.path.isdir(path):
            return None
        return path

    def _validate_latent_cache_identity(self, cache_path: str, cache_attrs: Mapping[str, Any]) -> None:
        if not self.use_camera_latent:
            return
        if self.latent_cache_image_encoder_name is None or self.latent_cache_image_model_name is None:
            raise ValueError(
                "use_camera_latent=true requires latent_cache_image_encoder_name and "
                "latent_cache_image_model_name. Pass models.fm vision settings when building the dataset."
            )
        validate_latent_cache_identity(
            cache_attrs,
            {
                "image_encoder_name": self.latent_cache_image_encoder_name,
                "dino_model_name": self.latent_cache_image_model_name,
            },
            cache_path=cache_path,
        )

    def _maybe_open_latent_cache(self) -> None:
        cache_path = self._resolve_latent_cache_path(require_exists=True)
        self.latent_cache_zarr = zarr.open_group(cache_path, mode="r")
        if "data" not in self.latent_cache_zarr:
            raise KeyError(f"Invalid latent cache zarr structure: {cache_path}")

        self.latent_cache_group = self.latent_cache_zarr["data"]
        if "frame_image_backbone_feat" not in self.latent_cache_group:
            raise KeyError(
                "Latent cache missing data/frame_image_backbone_feat (scheme A). "
                "Rebuild with ./scripts/precompute.sh"
            )

        cache_attrs = dict(getattr(self.latent_cache_zarr, "attrs", {}))
        self._validate_latent_cache_identity(cache_path, cache_attrs)

        cache_views = cache_attrs.get("camera_views")
        if cache_views is None or not str(cache_views).strip():
            raise KeyError(
                f"Latent cache missing camera_views attrs at {cache_path}. "
                "Rebuild with ./scripts/precompute.sh (scheme A)."
            )
        cache_view_list = parse_cache_camera_views(str(cache_views))
        latent_view_indices = cache_view_indices(self.camera_views, cache_view_list)
        if cache_view_list != self.camera_views:
            print(
                "[ZarrDataset] latent cache camera_views: "
                f"cache={list(cache_view_list)!r}, using={list(self.camera_views)!r}, "
                f"indices={list(latent_view_indices)}"
            )

        frame_src = self.latent_cache_group["frame_image_backbone_feat"]
        cache_n_views = int(cache_attrs.get("n_image_views", frame_src.shape[1]))
        if cache_n_views < self.n_image_views:
            raise ValueError(
                f"Latent cache has fewer views ({cache_n_views}) than requested "
                f"({self.n_image_views}). Rebuild cache."
            )

        frame_count = self.ram_data[self.state_key].shape[0]
        if int(frame_src.shape[0]) != frame_count:
            raise ValueError(
                "frame_image_backbone_feat frame count mismatch: "
                f"cache={frame_src.shape[0]}, data={frame_count}"
            )

        self.image_backbone_dim = int(cache_attrs.get("image_backbone_dim", frame_src.shape[-1]))
        # Materialize full frame cache into RAM so DataLoader workers avoid zarr I/O.
        frame = np.asarray(frame_src[:], dtype=np.float32)
        view_idx = [int(i) for i in latent_view_indices]
        if view_idx != list(range(int(frame.shape[1]))):
            frame = np.ascontiguousarray(frame[:, view_idx, :])
        self._frame_latent_view_indices = tuple(range(int(frame.shape[1])))
        self.cached_frame_image_backbone_feat = frame
        self.cached_image_backbone_feat = None
        frame_gb = frame.nbytes / (1024**3)
        print(
            f"[ZarrDataset] frame latent cache loaded: {cache_path}, "
            f"shape={tuple(frame.shape)}, using_views={self.n_image_views}, "
            f"backbone_dim={self.image_backbone_dim}, size={frame_gb:.3f} GB"
        )
        self.latent_cache_group = None
        self.latent_cache_zarr = None

    def _gather_frame_latent(self, indices) -> np.ndarray:
        if self.cached_frame_image_backbone_feat is None:
            raise RuntimeError(
                "frame camera latent cache is not loaded; run ./scripts/precompute.sh"
            )
        idx = np.asarray(indices, dtype=np.int64)
        return self.cached_frame_image_backbone_feat[idx]

    def get_camera_latent(self, idx: int) -> np.ndarray:
        i0, i1 = self.image_range(idx)
        feat = self._gather_frame_latent(np.arange(i0, i1, dtype=np.int64))
        if feat.ndim == 2:
            feat = feat[None, ...]
        if feat.shape[0] != self.n_image_steps:
            raise ValueError(
                f"camera latent time mismatch: {feat.shape[0]} != {self.n_image_steps}"
            )
        if feat.shape[1] != self.n_image_views:
            raise ValueError(
                f"camera latent view mismatch: {feat.shape[1]} != {self.n_image_views}"
            )
        return feat

    def get_memory_camera_latent(self, anchor_t: int, ep_idx: int) -> np.ndarray:
        indices = self.memory_visual_indices(anchor_t, ep_idx)
        feat = self._gather_frame_latent(indices)
        if feat.shape[0] != len(self.memory_visual_offsets):
            raise ValueError(
                f"memory camera latent time mismatch: {feat.shape[0]} != {len(self.memory_visual_offsets)}"
            )
        if feat.shape[1] != self.n_image_views:
            raise ValueError(
                f"memory camera latent view mismatch: {feat.shape[1]} != {self.n_image_views}"
            )
        return feat

    def _precompute_normalized_actions(self) -> None:
        n = len(self.windows)
        if n == 0 or self.normalizer is None:
            self.cached_norm_action = None
            return
        print(f"[ZarrDataset] precomputing normalized actions for {n} windows...")
        cached = np.empty((n, self.action_horizon, self.action_dim), dtype=np.float32)
        for idx in range(n):
            s0, s1 = self.state_range(idx)
            a0, a1 = self.action_range(idx)
            state_raw = self.get_state(s0, s1)
            action_raw = self.get_action(a0, a1)
            cached[idx] = self.normalizer.normalize_action_np(action_raw, state_raw)
            if idx > 0 and idx % 10000 == 0:
                print(f"[ZarrDataset]   normalized actions: {idx}/{n}")
        self.cached_norm_action = cached
        print(
            f"[ZarrDataset] normalized action cache: shape={cached.shape}, "
            f"size={cached.nbytes / (1024**3):.3f} GB"
        )

    def _preload_to_ram(self) -> None:
        total_gb = 0.0
        print("[ZarrDataset] preloading zarr keys into RAM...")
        for key in self._data_keys_to_load():
            arr = np.asarray(self.data_group[key][:])
            self.ram_data[key] = arr
            arr_gb = arr.nbytes / (1024**3)
            total_gb += arr_gb
            label = key
            print(
                f"[ZarrDataset] loaded {label}: "
                f"shape={arr.shape}, dtype={arr.dtype}, size={arr_gb:.3f} GB"
            )
        print(f"[ZarrDataset] total RAM preload: {total_gb:.3f} GB")

    def _build_windows(self) -> List[Tuple[int, int, int]]:
        windows: List[Tuple[int, int, int]] = []
        cond_len = max(self.window_size, self.n_image_steps)

        for ep_idx, (ep_start, ep_end) in enumerate(zip(self.episode_starts, self.episode_ends)):
            ep_start = int(ep_start)
            ep_end = int(ep_end)
            first_t = ep_start + cond_len - 1
            last_t = ep_end - self.action_horizon
            if last_t < first_t:
                continue
            for t in range(first_t, last_t + 1, self.stride):
                windows.append((t, ep_end, ep_idx))

        return windows

    def _build_memory_visual_offsets(self) -> np.ndarray:
        start = (
            -self.memory_visual_recent_frame
            - self.memory_sample_stride * (self.memory_visual_history_length - 1)
        )
        stop = -self.memory_visual_recent_frame + 1
        return np.arange(start, stop, self.memory_sample_stride, dtype=np.int64)

    def _clamp_memory_indices(self, indices: np.ndarray, ep_idx: int) -> np.ndarray:
        ep_start, ep_end = self.episode_bounds(ep_idx)
        # pad_first: clamp out-of-episode indices to episode bounds.
        return np.clip(indices, ep_start, ep_end - 1).astype(np.int64, copy=False)

    def _memory_index_valid(self, raw_indices: np.ndarray, ep_idx: int) -> np.ndarray:
        ep_start, ep_end = self.episode_bounds(ep_idx)
        return ((raw_indices >= ep_start) & (raw_indices < ep_end)).astype(np.bool_)

    def memory_visual_indices(self, anchor_t: int, ep_idx: int) -> np.ndarray:
        raw = int(anchor_t) + self.memory_visual_offsets
        return self._clamp_memory_indices(raw, ep_idx)

    def memory_visual_valid(self, anchor_t: int, ep_idx: int) -> np.ndarray:
        # Out-of-episode visual indices are clamped to the first episode frame.
        # Those repeated first-frame tokens intentionally participate in attention.
        return np.ones(self.memory_visual_history_length, dtype=np.bool_)

    def memory_state_indices(self, anchor_t: int, ep_idx: int) -> np.ndarray:
        end = int(anchor_t) - self.memory_recent_frame + 1
        start = end - self.memory_history_frames
        raw = np.arange(start, end, dtype=np.int64)
        return self._clamp_memory_indices(raw, ep_idx)

    def memory_state_valid(self, anchor_t: int, ep_idx: int) -> np.ndarray:
        end = int(anchor_t) - self.memory_recent_frame + 1
        start = end - self.memory_history_frames
        raw = np.arange(start, end, dtype=np.int64)
        return self._memory_index_valid(raw, ep_idx)

    def _obs_window_indices(self, anchor_t: int, ep_idx: int) -> Tuple[int, int]:
        obs_start = int(anchor_t - self.window_size + 1)
        obs_end = int(anchor_t + 1)
        ep_start, ep_end = self.episode_bounds(ep_idx)
        if obs_start < ep_start or obs_end > ep_end:
            raise ValueError(
                "Observation window out of episode bounds: "
                f"window=({obs_start},{obs_end}), ep_bounds=({ep_start},{ep_end})"
            )
        return obs_start, obs_end

    def state_range(self, idx: int) -> Tuple[int, int]:
        anchor_t, _, ep_idx = self.windows[idx]
        return self._obs_window_indices(anchor_t, ep_idx)

    def image_range(self, idx: int) -> Tuple[int, int]:
        anchor_t, _, ep_idx = self.windows[idx]
        image_start = int(anchor_t - self.n_image_steps + 1)
        image_end = int(anchor_t + 1)
        ep_start, _ = self.episode_bounds(ep_idx)
        if image_start < ep_start:
            raise ValueError(
                f"Image history start out of episode bounds: image_start={image_start}, ep_start={ep_start}"
            )
        return image_start, image_end

    def action_range(self, idx: int) -> Tuple[int, int]:
        anchor_t, episode_end, _ = self.windows[idx]
        action_start = int(anchor_t)
        action_end = int(anchor_t + self.action_horizon)
        if action_end > int(episode_end):
            raise ValueError(
                "Action horizon exceeds episode end: "
                f"action=({action_start},{action_end}), episode_end={episode_end}"
            )
        return action_start, action_end

    def _read_array(self, key: str, slc: slice | None = None, dtype=None) -> np.ndarray:
        if key in self.ram_data:
            base = self.ram_data[key]
            out = base if slc is None else base[slc]
        else:
            base = self.data_group[key]
            out = base[:] if slc is None else base[slc]
        return np.asarray(out, dtype=dtype) if dtype is not None else np.asarray(out)

    def _slice_robot(self, arr: np.ndarray) -> np.ndarray:
        return np.asarray(arr[..., self.robot_slice])

    def get_state(self, t0: int, t1: int) -> np.ndarray:
        return self._slice_robot(self._read_array(self.state_key, slice(t0, t1), dtype=np.float32))

    def get_action(self, t0: int, t1: int) -> np.ndarray:
        return self._slice_robot(self._read_array(self.action_key, slice(t0, t1), dtype=np.float32))

    def get_camera(self, t0: int, t1: int) -> np.ndarray:
        camera = self._read_array(self.camera_key, slice(t0, t1))
        return self._select_camera_channels(camera)

    def _select_camera_channels(self, camera: np.ndarray) -> np.ndarray:
        if len(self._camera_channel_indices) == camera.shape[-1]:
            return camera
        return np.asarray(camera[..., self._camera_channel_indices])

    def get_memory_state(self, anchor_t: int, ep_idx: int) -> np.ndarray:
        indices = self.memory_state_indices(anchor_t, ep_idx)
        state = self._slice_robot(self.ram_data[self.state_key][indices]).astype(np.float32, copy=False)
        if state.shape[0] != self.memory_history_frames:
            raise ValueError(
                f"memory state length mismatch: {state.shape[0]} != {self.memory_history_frames}"
            )
        return state

    def get_tactile_raw(self, t0: int, t1: int) -> np.ndarray:
        if not self.use_tactile:
            raise RuntimeError("use_tactile=False")
        return self._read_array(self.tactile_key, slice(t0, t1), dtype=np.float32)

    def get_tactile(self, t0: int, t1: int) -> np.ndarray:
        """Deformation map (T, H, W, 12): 4 sensors x (dx, dy, dz)."""
        return extract_tactile_deformation(self.get_tactile_raw(t0, t1))

    @property
    def num_episodes(self) -> int:
        return int(len(self.episode_ends))

    def episode_bounds(self, ep_idx: int) -> Tuple[int, int]:
        ep_idx = int(ep_idx)
        return int(self.episode_starts[ep_idx]), int(self.episode_ends[ep_idx])

    def get_episode(self, ep_idx: int) -> Dict[str, Any]:
        t0, t1 = self.episode_bounds(ep_idx)
        out: Dict[str, Any] = {
            "camera": self.get_camera(t0, t1),
            "state": self.get_state(t0, t1),
            "action": self.get_action(t0, t1),
            "ep_idx": ep_idx,
            "t_start": t0,
            "t_end": t1,
        }
        if self.use_tactile:
            out["tactile"] = self.get_tactile(t0, t1)
        return out

    def _process_image(self, img: np.ndarray) -> torch.Tensor:
        arr = np.asarray(img)
        single_frame = arr.ndim == 3
        if single_frame:
            arr = arr[None, ...]
        if arr.ndim != 4:
            raise ValueError(f"Unsupported image shape: {arr.shape}")
        if arr.shape[-1] % 3 != 0:
            raise ValueError(f"Image channel count must be a multiple of 3, got {arr.shape}")

        n_views = arr.shape[-1] // 3
        x = (
            torch.from_numpy(arr)
            .reshape(arr.shape[0], arr.shape[1], arr.shape[2], n_views, 3)
            .permute(0, 3, 4, 1, 2)
            .contiguous()
        )
        if x.shape[-1] != self.image_size or x.shape[-2] != self.image_size:
            flat = x.flatten(0, 1)
            xf = F.interpolate(
                flat.float(),
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            )
            if self.image_as_uint8:
                out = xf.round().clamp_(0.0, 255.0).to(torch.uint8)
            else:
                out = xf.div_(255.0).mul_(2.0).sub_(1.0)
            out = out.reshape(arr.shape[0], n_views, 3, self.image_size, self.image_size)
            return out[0] if single_frame else out

        if self.image_as_uint8:
            return x[0] if single_frame else x
        out = x.float().div_(255.0).mul_(2.0).sub_(1.0)
        return out[0] if single_frame else out

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        idx = int(idx)
        anchor_t, _episode_end, ep_idx = self.windows[idx]

        s0, s1 = self.state_range(idx)
        i0, i1 = self.image_range(idx)
        a0, a1 = self.action_range(idx)

        state_raw = self.get_state(s0, s1)
        if state_raw.shape[0] != self.window_size:
            raise ValueError(f"state length mismatch: {state_raw.shape[0]} != {self.window_size}")

        if self.normalizer is None:
            raise RuntimeError("normalizer is not available (fit_normalizer=false)")
        state = self.normalizer.normalize_state_np(state_raw)
        if self.cached_norm_action is not None:
            action = self.cached_norm_action[idx]
        else:
            action_raw = self.get_action(a0, a1)
            if action_raw.shape[0] != self.action_horizon:
                raise ValueError(f"action length mismatch: {action_raw.shape[0]} != {self.action_horizon}")
            action = self.normalizer.normalize_action_np(action_raw, state_raw)

        if action.shape[0] != self.action_horizon:
            raise ValueError(f"action length mismatch: {action.shape[0]} != {self.action_horizon}")

        obs: Dict[str, Any] = {
            "state": torch.from_numpy(state.astype(np.float32)),
        }

        if self.use_camera_latent:
            latent = self.get_camera_latent(idx)
            obs["image_backbone_feat"] = torch.from_numpy(latent.astype(np.float32))
        else:
            camera = self.get_camera(i0, i1)
            if camera.shape[0] != self.n_image_steps:
                raise ValueError(
                    f"image length mismatch: {camera.shape[0]} != {self.n_image_steps}"
                )
            camera = self._maybe_augment_camera(camera)
            image = self._process_image(camera)
            if image.shape[0] != self.n_image_steps or image.shape[1] != self.n_image_views:
                raise ValueError(
                    f"image shape {tuple(image.shape)} != "
                    f"(n_image_steps={self.n_image_steps}, n_image_views={self.n_image_views}, 3, H, W)"
                )
            obs["image"] = image

        if self.memory_enabled:
            if not self.use_camera_latent:
                raise RuntimeError(
                    "data.memory.enabled=true currently requires data.use_camera_latent=true "
                    "and a frame_backbone.zarr cache (./scripts/precompute.sh)."
                )
            memory_state_raw = self.get_memory_state(anchor_t, ep_idx)
            memory_state = self.normalizer.normalize_state_np(memory_state_raw)
            obs["memory_state"] = torch.from_numpy(memory_state.astype(np.float32))
            obs["memory_image_backbone_feat"] = torch.from_numpy(
                self.get_memory_camera_latent(anchor_t, ep_idx).astype(np.float32)
            )
            obs["memory_visual_offsets"] = torch.from_numpy(
                self.memory_visual_offsets.astype(np.int64, copy=False)
            )
            obs["memory_visual_valid"] = torch.from_numpy(
                self.memory_visual_valid(anchor_t, ep_idx)
            )
            obs["memory_state_valid"] = torch.from_numpy(
                self.memory_state_valid(anchor_t, ep_idx)
            )

        if self.use_tactile:
            tactile = self.normalizer.normalize_tactile_np(self.get_tactile(s0, s1))
            obs["tactile"] = torch.from_numpy(tactile.astype(np.float32))

        return {
            "obs": obs,
            "action": torch.from_numpy(action.astype(np.float32)),
            "meta": {
                "idx": idx,
                "anchor_t": int(anchor_t),
                "ep_idx": int(ep_idx),
            },
        }


def build_dataloader(
    dataset: ZarrDataset,
    *,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    drop_last: bool = True,
    pin_memory: bool = True,
    persistent_workers: bool | None = None,
    prefetch_factor: int = 2,
) -> DataLoader:
    """Standard DataLoader for ZarrDataset; default collate handles nested obs dict."""
    kwargs: Dict[str, Any] = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "drop_last": drop_last,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = (
            persistent_workers if persistent_workers is not None else True
        )
        kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(dataset, **kwargs)
