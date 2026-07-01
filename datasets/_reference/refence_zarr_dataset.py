from __future__ import annotations

import os
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import zarr

class ZarrDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        window_size: int = 32,
        stride: int = 1,
        action_dim: int = 10,
        image_size: int = 224,
        n_image_steps: int | None = None,
        action_window_size: int | None = None,
        image_as_uint8: bool = True,
        preload_to_ram: bool = False,
        latent_cache_root_dir: str | None = None,
        image_keys: Sequence[str] = ("left_wrist_img",),
        tactile_left_key: str = "left_gripper1_tactile",
        tactile_right_key: str = "left_gripper2_tactile",
        force_key: str = "left_wrist_force",
        state_key: str = "left_robot_tcp_pose",
        force_4x_key: str | None = None,
        state_4x_key: str | None = None,
        action_key: str = "action",
        action_representation: str = "absolute",
    ):
        self.root_dir = root_dir
        self.split = split
        self.window_size = max(1, int(window_size))
        self.stride = max(1, int(stride))
        self.action_dim = int(action_dim)
        self.image_size = int(image_size)

        self.n_image_steps = self.window_size if n_image_steps is None else max(1, int(n_image_steps))
        self.action_window_size = self.window_size if action_window_size is None else max(1, int(action_window_size))

        self.image_as_uint8 = bool(image_as_uint8)
        self.preload_to_ram = bool(preload_to_ram)
        self.latent_cache_root_dir = latent_cache_root_dir

        self.image_keys = list(image_keys)
        self.tactile_left_key = tactile_left_key
        self.tactile_right_key = tactile_right_key
        self.force_key = force_key
        self.state_key = state_key
        self.force_4x_key = force_4x_key or force_key
        self.state_4x_key = state_4x_key or state_key
        self.action_key = action_key
        self.action_representation = self._normalize_action_representation(action_representation)

        self.zarr_path = self._resolve_zarr_path(root_dir, split)
        self.zarr_root = zarr.open_group(self.zarr_path, mode="r")
        self.data_group = self.zarr_root["data"]
        self.meta_group = self.zarr_root["meta"]

        self.latent_cache_zarr = None
        self.latent_cache_group = None
        self.cached_tactile_latent_curr = None
        self.cached_tactile_latent_future = None
        self.cached_image_backbone_feat = None
        self.cached_latent_dim = None
        self.ram_data: Dict[str, np.ndarray] = {}

        self._validate_required_keys()

        self.episode_ends = np.asarray(self.meta_group["episode_ends"][:], dtype=np.int64)
        self.episode_starts = np.concatenate([np.array([0], dtype=np.int64), self.episode_ends[:-1]])

        self.episode_ends_4x = np.asarray(self.meta_group["episode_ends_4x"][:], dtype=np.int64)
        self.episode_starts_4x = np.concatenate([np.array([0], dtype=np.int64), self.episode_ends_4x[:-1]])
        if len(self.episode_ends_4x) != len(self.episode_ends):
            raise ValueError(
                "episode_ends and episode_ends_4x must have the same episode count, "
                f"got {len(self.episode_ends)} vs {len(self.episode_ends_4x)}"
            )

        self.windows = self._build_windows()
        if len(self.windows) == 0:
            raise ValueError(
                "No valid strict anchor windows. "
                f"window_size={self.window_size}, n_image_steps={self.n_image_steps}, "
                f"action_window_size={self.action_window_size}, stride={self.stride}"
            )
        self.window_lookup = {
            (int(anchor_t), int(ep_idx)): idx
            for idx, (anchor_t, _episode_end, ep_idx) in enumerate(self.windows)
        }

        self._maybe_open_latent_cache()
        self._maybe_preload_arrays()

    @staticmethod
    def _resolve_zarr_path(root_dir: str, split: str) -> str:
        if root_dir.endswith(".zarr") and os.path.isdir(root_dir):
            return root_dir

        candidates = [
            os.path.join(root_dir, split, "replay_buffer.zarr"),
            os.path.join(root_dir, "replay_buffer.zarr"),
        ]
        if split == "val":
            candidates.append(os.path.join(root_dir, "test", "replay_buffer.zarr"))
        if split == "test":
            candidates.append(os.path.join(root_dir, "val", "replay_buffer.zarr"))

        for path in candidates:
            if os.path.isdir(path):
                return path
        raise FileNotFoundError(
            f"Cannot find replay_buffer.zarr from root_dir={root_dir}, split={split}. Tried: {candidates}"
        )

    @staticmethod
    def _normalize_action_representation(value: str) -> str:
        rep = str(value).strip().lower()
        if rep in {"relative", "relative_action"}:
            rep = "chunk_relative"
        if rep not in {"absolute", "chunk_relative"}:
            raise ValueError(
                f"Unsupported action_representation={value}. "
                "Choose from ['absolute', 'chunk_relative']."
            )
        return rep

    def _validate_required_keys(self) -> None:
        required = [
            *self.image_keys,
            self.tactile_left_key,
            self.tactile_right_key,
            self.force_key,
            self.state_key,
            self.force_4x_key,
            self.state_4x_key,
            self.action_key,
        ]
        for key in set(required):
            if key not in self.data_group:
                raise KeyError(f"Missing key in zarr data group: {key}")
        for key in ("episode_ends", "episode_ends_4x"):
            if key not in self.meta_group:
                raise KeyError(f"Missing key in zarr meta group: {key}")

    def _resolve_latent_cache_path(self) -> str | None:
        if self.latent_cache_root_dir:
            path = os.path.join(self.latent_cache_root_dir, self.split, "policy_latent_cache.zarr")
            if os.path.isdir(path):
                return path
            raise FileNotFoundError(f"Latent cache path does not exist: {path}")

        return None

    def _maybe_open_latent_cache(self) -> None:
        cache_path = self._resolve_latent_cache_path()
        if cache_path is None:
            return

        self.latent_cache_zarr = zarr.open_group(cache_path, mode="r")
        if "data" not in self.latent_cache_zarr or "meta" not in self.latent_cache_zarr:
            raise KeyError(f"Invalid latent cache zarr structure: {cache_path}")

        self.latent_cache_group = self.latent_cache_zarr["data"]
        cache_meta = self.latent_cache_zarr["meta"]

        for key in ("tactile_latent_curr", "tactile_latent_future"):
            if key not in self.latent_cache_group:
                raise KeyError(f"Missing key in latent cache data group: {key}")
        for key in ("window_anchor_times", "window_episode_ends", "window_episode_indices"):
            if key not in cache_meta:
                raise KeyError(
                    f"Missing key in latent cache meta group: {key}. "
                    "Please rebuild cache using the strict anchor precompute script."
                )

        cache_anchor = np.asarray(cache_meta["window_anchor_times"][:], dtype=np.int64)
        cache_ep_end = np.asarray(cache_meta["window_episode_ends"][:], dtype=np.int64)
        cache_ep_idx = np.asarray(cache_meta["window_episode_indices"][:], dtype=np.int64)
        if not (len(cache_anchor) == len(cache_ep_end) == len(cache_ep_idx)):
            raise ValueError(
                "Latent cache meta length mismatch: "
                f"anchor={len(cache_anchor)}, ep_end={len(cache_ep_end)}, ep_idx={len(cache_ep_idx)}"
            )

        cache_attrs = getattr(self.latent_cache_zarr, "attrs", {})
        expected_attrs = {
            "window_size": self.window_size,
            "action_window_size": self.action_window_size,
            "stride": self.stride,
        }
        for key, expected in expected_attrs.items():
            actual = cache_attrs.get(key)
            if actual is not None and int(actual) != int(expected):
                raise ValueError(
                    f"Latent cache attr mismatch for {key}: cache={actual}, dataset={expected}"
                )

        window_arr = np.asarray(self.windows, dtype=np.int64)
        if len(cache_anchor) != len(window_arr):
            raise ValueError(
                f"Latent cache window count mismatch: cache={len(cache_anchor)} dataset={len(window_arr)}"
            )
        if not (
            np.array_equal(cache_anchor, window_arr[:, 0])
            and np.array_equal(cache_ep_end, window_arr[:, 1])
            and np.array_equal(cache_ep_idx, window_arr[:, 2])
        ):
            raise ValueError(
                "Latent cache windows do not match dataset strict anchor windows. "
                "Please rebuild cache for the current data config."
            )

        self.cached_tactile_latent_curr = self.latent_cache_group["tactile_latent_curr"]
        self.cached_tactile_latent_future = self.latent_cache_group["tactile_latent_future"]
        self.cached_image_backbone_feat = (
            self.latent_cache_group["image_backbone_feat"]
            if "image_backbone_feat" in self.latent_cache_group
            else None
        )
        self.cached_latent_dim = int(self.cached_tactile_latent_curr.shape[-1])

        if self.cached_tactile_latent_curr.shape[0] != len(self.windows):
            raise ValueError(
                "Latent cache row count mismatch: "
                f"cache_rows={self.cached_tactile_latent_curr.shape[0]}, windows={len(self.windows)}"
            )
        if self.cached_tactile_latent_future.shape[0] != len(self.windows):
            raise ValueError(
                "Latent cache future row count mismatch: "
                f"cache_rows={self.cached_tactile_latent_future.shape[0]}, windows={len(self.windows)}"
            )
        if (
            self.cached_image_backbone_feat is not None
            and self.cached_image_backbone_feat.shape[0] != len(self.windows)
        ):
            raise ValueError(
                "Latent cache image feature row count mismatch: "
                f"cache_rows={self.cached_image_backbone_feat.shape[0]}, windows={len(self.windows)}"
            )

    def _maybe_preload_arrays(self) -> None:
        if not self.preload_to_ram:
            return

        keys = [
            self.force_key,
            self.state_key,
            self.force_4x_key,
            self.state_4x_key,
            self.action_key,
        ]
        if self.cached_image_backbone_feat is None:
            keys.extend(self.image_keys)
        if self.cached_tactile_latent_curr is None:
            keys.extend([self.tactile_left_key, self.tactile_right_key])

        total_gb = 0.0
        print("[ZarrDataset] preloading zarr keys into RAM...")
        for key in dict.fromkeys(keys):
            arr = np.asarray(self.data_group[key][:])
            self.ram_data[key] = arr
            arr_gb = arr.nbytes / (1024 ** 3)
            total_gb += arr_gb
            print(
                f"[ZarrDataset] loaded {key}: "
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
            last_t = ep_end - self.action_window_size
            if last_t < first_t:
                continue
            for t in range(first_t, last_t + 1, self.stride):
                windows.append((t, ep_end, ep_idx))

        return windows

    @staticmethod
    def _reshape_tactile(arr: np.ndarray) -> np.ndarray:
        if arr.ndim == 3 and arr.shape[1] == 700:
            return arr.reshape(arr.shape[0], 35, 20, arr.shape[-1])
        if arr.ndim == 4:
            return arr
        raise ValueError(f"Unsupported tactile shape: {arr.shape}")

    def _read_array(self, key: str, slc=None, dtype=None) -> np.ndarray:
        if key in self.ram_data:
            base = self.ram_data[key]
            out = base if slc is None else base[slc]
        else:
            base = self.data_group[key]
            out = base[:] if slc is None else base[slc]
        if dtype is not None:
            out = np.asarray(out, dtype=dtype)
        else:
            out = np.asarray(out)
        return out

    def _process_image(self, img: np.ndarray) -> torch.Tensor:
        arr = np.asarray(img)
        single_frame = arr.ndim == 3
        if single_frame:
            arr = arr[None, ...]
        if arr.ndim != 4:
            raise ValueError(f"Unsupported image shape: {arr.shape}")

        x = torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous()
        if x.shape[-1] != self.image_size or x.shape[-2] != self.image_size:
            xf = F.interpolate(
                x.float(),
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            )
            if self.image_as_uint8:
                out = xf.round().clamp_(0.0, 255.0).to(torch.uint8)
            else:
                out = xf.div_(255.0).mul_(2.0).sub_(1.0)
            return out[0] if single_frame else out

        if self.image_as_uint8:
            return x[0] if single_frame else x
        out = x.float().div_(255.0).mul_(2.0).sub_(1.0)
        return out[0] if single_frame else out

    def _transform_action(self, action: np.ndarray, state: np.ndarray) -> np.ndarray:
        if self.action_representation == "absolute":
            return action
        if state.shape[-1] < self.action_dim:
            raise ValueError(
                "chunk_relative action representation requires state_dim >= action_dim, "
                f"got state_dim={state.shape[-1]}, action_dim={self.action_dim}"
            )
        base_absolute_action = np.asarray(state[-1, : self.action_dim], dtype=np.float32)
        return absolute_actions_to_relative_actions(action, base_absolute_action=base_absolute_action)

    def _force_ratio(self, ep_idx: int) -> int:
        tactile_ep_len = int(self.episode_ends[ep_idx] - self.episode_starts[ep_idx])
        force_ep_len = int(self.episode_ends_4x[ep_idx] - self.episode_starts_4x[ep_idx])
        if tactile_ep_len <= 0 or force_ep_len <= 0:
            raise ValueError(
                f"Invalid episode lengths for ep_idx={ep_idx}: tactile={tactile_ep_len}, force={force_ep_len}"
            )
        if force_ep_len % tactile_ep_len != 0:
            raise ValueError(
                f"force_4x must be integer multiple of 30Hz length, "
                f"got ep_idx={ep_idx}, 30Hz={tactile_ep_len}, 120Hz={force_ep_len}"
            )
        ratio = force_ep_len // tactile_ep_len
        if ratio != 4:
            raise ValueError(f"Expected force_4x / force ratio = 4, got {ratio} for ep_idx={ep_idx}")
        return ratio

    def _map_tactile_window_to_force_window(self, start: int, end: int, ep_idx: int) -> tuple[int, int]:
        tactile_ep_start = int(self.episode_starts[ep_idx])
        force_ep_start = int(self.episode_starts_4x[ep_idx])
        force_ep_end = int(self.episode_ends_4x[ep_idx])
        ratio = self._force_ratio(ep_idx)

        local_start = int(start - tactile_ep_start)
        local_end = int(end - tactile_ep_start)
        force_start = force_ep_start + local_start * ratio
        force_end = force_ep_start + local_end * ratio

        force_start = max(force_ep_start, min(force_start, force_ep_end))
        force_end = max(force_start, min(force_end, force_ep_end))
        return force_start, force_end

    def _obs_window_indices(self, anchor_t: int, ep_idx: int) -> tuple[int, int]:
        obs_start = int(anchor_t - self.window_size + 1)
        obs_end = int(anchor_t + 1)
        ep_start = int(self.episode_starts[ep_idx])
        ep_end = int(self.episode_ends[ep_idx])
        if obs_start < ep_start or obs_end > ep_end:
            raise ValueError(
                "Observation window out of episode bounds: "
                f"window=({obs_start},{obs_end}), ep_bounds=({ep_start},{ep_end})"
            )
        return obs_start, obs_end

    def get_dynamics_input(self, idx: int) -> Dict[str, np.ndarray]:
        anchor_t, _, ep_idx = self.windows[idx]
        obs_start, obs_end = self._obs_window_indices(anchor_t, ep_idx)

        left_tactile = self._read_array(self.tactile_left_key, slice(obs_start, obs_end), dtype=np.float32)
        right_tactile = self._read_array(self.tactile_right_key, slice(obs_start, obs_end), dtype=np.float32)
        if left_tactile.shape[0] != self.window_size or right_tactile.shape[0] != self.window_size:
            raise ValueError(
                "Tactile history length mismatch: "
                f"left={left_tactile.shape[0]}, right={right_tactile.shape[0]}, expected={self.window_size}"
            )

        left_tactile = self._reshape_tactile(left_tactile)
        right_tactile = self._reshape_tactile(right_tactile)
        tactile = np.concatenate([left_tactile, right_tactile], axis=-1)
        tactile = np.concatenate([tactile, tactile[:, -1:, :, :]], axis=1)

        force_start, force_end = self._map_tactile_window_to_force_window(obs_start, obs_end, ep_idx)
        force_4x = self._read_array(self.force_4x_key, slice(force_start, force_end), dtype=np.float32)
        state_4x = self._read_array(self.state_4x_key, slice(force_start, force_end), dtype=np.float32)
        expected_len = int(self._force_ratio(ep_idx) * self.window_size)
        if force_4x.shape[0] != expected_len or state_4x.shape[0] != expected_len:
            raise ValueError(
                "4x history length mismatch: "
                f"force_4x={force_4x.shape[0]}, state_4x={state_4x.shape[0]}, expected={expected_len}"
            )

        return {
            "tactile": tactile,
            "force_4x": force_4x,
            "state_4x": state_4x,
        }

    def __len__(self) -> int:
        return len(self.windows)

    @property
    def num_episodes(self) -> int:
        return int(len(self.episode_ends))

    def episode_bounds(self, ep_idx: int) -> tuple[int, int]:
        ep_idx = int(ep_idx)
        return int(self.episode_starts[ep_idx]), int(self.episode_ends[ep_idx])

    def get_episode_action_raw(self, ep_idx: int) -> np.ndarray:
        ep_start, ep_end = self.episode_bounds(ep_idx)
        return self._read_array(
            self.action_key,
            slice(ep_start, ep_end),
            dtype=np.float32,
        )[..., : self.action_dim]

    def get_episode_images_raw(self, ep_idx: int) -> np.ndarray:
        ep_start, ep_end = self.episode_bounds(ep_idx)
        return np.stack(
            [self._read_array(key, slice(ep_start, ep_end)) for key in self.image_keys],
            axis=1,
        )

    @staticmethod
    def _subsample_rows(arr: np.ndarray, max_rows: int | None) -> np.ndarray:
        x = np.asarray(arr, dtype=np.float32)
        if max_rows is None or max_rows <= 0:
            return x.reshape(-1, x.shape[-1])
        flat = x.reshape(-1, x.shape[-1])
        if flat.shape[0] <= max_rows:
            return flat
        idx = np.linspace(0, flat.shape[0] - 1, num=max_rows, dtype=np.int64)
        return flat[idx]

    def _obs_from_anchor(self, anchor_t: int, ep_idx: int) -> Dict[str, torch.Tensor]:
        obs_start, obs_end = self._obs_window_indices(anchor_t, ep_idx)

        force = self._read_array(self.force_key, slice(obs_start, obs_end), dtype=np.float32)
        state = self._read_array(self.state_key, slice(obs_start, obs_end), dtype=np.float32)
        if force.shape[0] != self.window_size or state.shape[0] != self.window_size:
            raise ValueError(
                "Condition history length mismatch: "
                f"force={force.shape[0]}, state={state.shape[0]}, expected={self.window_size}"
            )

        force_start, force_end = self._map_tactile_window_to_force_window(obs_start, obs_end, ep_idx)
        force_4x = self._read_array(self.force_4x_key, slice(force_start, force_end), dtype=np.float32)
        state_4x = self._read_array(self.state_4x_key, slice(force_start, force_end), dtype=np.float32)
        expected_force_len = int(self._force_ratio(ep_idx) * self.window_size)
        if force_4x.shape[0] != expected_force_len or state_4x.shape[0] != expected_force_len:
            raise ValueError(
                "4x condition history length mismatch: "
                f"force_4x={force_4x.shape[0]}, state_4x={state_4x.shape[0]}, expected={expected_force_len}"
            )

        obs = {
            "force": torch.from_numpy(force),
            "state": torch.from_numpy(state),
            "force_4x": torch.from_numpy(force_4x),
            "state_4x": torch.from_numpy(state_4x),
        }

        image_start = int(anchor_t - self.n_image_steps + 1)
        image_end = int(anchor_t + 1)
        ep_start = int(self.episode_starts[ep_idx])
        if image_start < ep_start:
            raise ValueError(
                "Image history start out of episode bounds: "
                f"image_start={image_start}, ep_start={ep_start}"
            )

        cache_idx = self.window_lookup.get((int(anchor_t), int(ep_idx)))
        if self.cached_image_backbone_feat is None:
            images = []
            for key in self.image_keys:
                image = self._read_array(key, slice(image_start, image_end))
                if image.shape[0] != self.n_image_steps:
                    raise ValueError(
                        f"Image history length mismatch for {key}: got {image.shape[0]}, "
                        f"expected {self.n_image_steps}"
                    )
                images.append(self._process_image(image))
            obs["image"] = torch.stack(images, dim=1)
        elif cache_idx is not None:
            image_backbone_feat = np.asarray(self.cached_image_backbone_feat[cache_idx], dtype=np.float32)
            if image_backbone_feat.ndim == 2:
                image_backbone_feat = image_backbone_feat[None, :, :]
            obs["image_backbone_feat"] = torch.from_numpy(image_backbone_feat)

        if self.cached_tactile_latent_curr is not None and cache_idx is not None:
            obs["tactile_latent_curr"] = torch.from_numpy(
                np.asarray(self.cached_tactile_latent_curr[cache_idx], dtype=np.float32)
            )
            obs["tactile_latent_future"] = torch.from_numpy(
                np.asarray(self.cached_tactile_latent_future[cache_idx], dtype=np.float32)
            )
        else:
            left_tactile = self._read_array(self.tactile_left_key, slice(obs_start, obs_end), dtype=np.float32)
            right_tactile = self._read_array(self.tactile_right_key, slice(obs_start, obs_end), dtype=np.float32)
            if left_tactile.shape[0] != self.window_size or right_tactile.shape[0] != self.window_size:
                raise ValueError(
                    "Tactile history length mismatch: "
                    f"left={left_tactile.shape[0]}, right={right_tactile.shape[0]}, expected={self.window_size}"
                )
            left_tactile = self._reshape_tactile(left_tactile)
            right_tactile = self._reshape_tactile(right_tactile)
            tactile = np.concatenate([left_tactile, right_tactile], axis=-1)
            tactile = np.concatenate([tactile, tactile[:, -1:, :, :]], axis=1)
            obs["tactile"] = torch.from_numpy(tactile)

        return obs

    def get_obs_only_at_time(self, t: int, ep_idx: int) -> Dict[str, torch.Tensor]:
        t = int(t)
        ep_idx = int(ep_idx)
        ep_start, ep_end = self.episode_bounds(ep_idx)
        cond_len = max(self.window_size, self.n_image_steps)
        first_t = ep_start + cond_len - 1
        if t < first_t or t > ep_end - 1:
            raise IndexError(f"t={t} out of obs-only range [{first_t}, {ep_end - 1}] for ep_idx={ep_idx}")
        return {"obs": self._obs_from_anchor(t, ep_idx)}

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        anchor_t, episode_end, ep_idx = self.windows[idx]
        obs_start, obs_end = self._obs_window_indices(anchor_t, ep_idx)

        force = self._read_array(self.force_key, slice(obs_start, obs_end), dtype=np.float32)
        state = self._read_array(self.state_key, slice(obs_start, obs_end), dtype=np.float32)
        if force.shape[0] != self.window_size or state.shape[0] != self.window_size:
            raise ValueError(
                "Condition history length mismatch: "
                f"force={force.shape[0]}, state={state.shape[0]}, expected={self.window_size}"
            )

        force_start, force_end = self._map_tactile_window_to_force_window(obs_start, obs_end, ep_idx)
        force_4x = self._read_array(self.force_4x_key, slice(force_start, force_end), dtype=np.float32)
        state_4x = self._read_array(self.state_4x_key, slice(force_start, force_end), dtype=np.float32)
        expected_force_len = int(self._force_ratio(ep_idx) * self.window_size)
        if force_4x.shape[0] != expected_force_len or state_4x.shape[0] != expected_force_len:
            raise ValueError(
                "4x condition history length mismatch: "
                f"force_4x={force_4x.shape[0]}, state_4x={state_4x.shape[0]}, expected={expected_force_len}"
            )

        action_start = int(anchor_t)
        action_end = int(anchor_t + self.action_window_size)
        if action_end > int(episode_end):
            raise ValueError(
                "Action horizon exceeds episode end: "
                f"action=({action_start},{action_end}), episode_end={episode_end}"
            )
        action = self._read_array(self.action_key, slice(action_start, action_end), dtype=np.float32)[..., : self.action_dim]
        if action.shape[0] != self.action_window_size:
            raise ValueError(
                f"Action horizon mismatch: got {action.shape[0]}, expected {self.action_window_size}"
            )
        action = self._transform_action(action, state)

        obs = {
            "force": torch.from_numpy(force),
            "state": torch.from_numpy(state),
            "force_4x": torch.from_numpy(force_4x),
            "state_4x": torch.from_numpy(state_4x),
        }

        if self.cached_image_backbone_feat is None:
            image_start = int(anchor_t - self.n_image_steps + 1)
            image_end = int(anchor_t + 1)
            ep_start = int(self.episode_starts[ep_idx])
            if image_start < ep_start:
                raise ValueError(
                    "Image history start out of episode bounds: "
                    f"image_start={image_start}, ep_start={ep_start}"
                )
            images = []
            for key in self.image_keys:
                image = self._read_array(key, slice(image_start, image_end))
                if image.shape[0] != self.n_image_steps:
                    raise ValueError(
                        f"Image history length mismatch for {key}: got {image.shape[0]}, "
                        f"expected {self.n_image_steps}"
                    )
                images.append(self._process_image(image))
            obs["image"] = torch.stack(images, dim=1)

        if self.cached_tactile_latent_curr is not None:
            obs["tactile_latent_curr"] = torch.from_numpy(
                np.asarray(self.cached_tactile_latent_curr[idx], dtype=np.float32)
            )
            obs["tactile_latent_future"] = torch.from_numpy(
                np.asarray(self.cached_tactile_latent_future[idx], dtype=np.float32)
            )
            if self.cached_image_backbone_feat is not None:
                image_backbone_feat = np.asarray(self.cached_image_backbone_feat[idx], dtype=np.float32)
                if image_backbone_feat.ndim == 2:
                    image_backbone_feat = image_backbone_feat[None, :, :]
                obs["image_backbone_feat"] = torch.from_numpy(image_backbone_feat)
        else:
            dynamics_input = self.get_dynamics_input(idx)
            obs["tactile"] = torch.from_numpy(dynamics_input["tactile"])

        return {
            "obs": obs,
            "action": torch.from_numpy(action),
        }

    def get_normalizer(self, max_rows: int | None = None) -> MultiFieldNormalizer:
        normalizer = MultiFieldNormalizer()

        raw_action = self._read_array(self.action_key, dtype=np.float32)[..., : self.action_dim]
        force = self._read_array(self.force_key, dtype=np.float32)
        state = self._read_array(self.state_key, dtype=np.float32)
        force_4x = self._read_array(self.force_4x_key, dtype=np.float32)
        state_4x = self._read_array(self.state_4x_key, dtype=np.float32)

        if self.action_representation == "absolute":
            action = raw_action
        else:
            chunks = []
            for anchor_t, episode_end, ep_idx in self.windows:
                action_start = int(anchor_t)
                action_end = int(anchor_t + self.action_window_size)
                if action_end > int(episode_end):
                    raise ValueError(
                        "Action horizon exceeds episode end while building normalizer: "
                        f"action=({action_start},{action_end}), episode_end={episode_end}"
                    )
                obs_start, obs_end = self._obs_window_indices(anchor_t, ep_idx)
                state_hist = state[obs_start:obs_end]
                action_chunk = raw_action[action_start:action_end]
                if state_hist.shape[0] != self.window_size:
                    raise ValueError(
                        f"State history length mismatch: got {state_hist.shape[0]}, expected {self.window_size}"
                    )
                if action_chunk.shape[0] != self.action_window_size:
                    raise ValueError(
                        f"Action chunk length mismatch: got {action_chunk.shape[0]}, expected {self.action_window_size}"
                    )
                chunks.append(self._transform_action(action_chunk, state_hist))
            action = np.asarray(chunks, dtype=np.float32)

        action = self._subsample_rows(action, max_rows)
        force = self._subsample_rows(force, max_rows)
        state = self._subsample_rows(state, max_rows)
        force_4x = self._subsample_rows(force_4x, max_rows)
        state_4x = self._subsample_rows(state_4x, max_rows)

        normalizer["action"] = FieldNormalizer.from_data_limits(action)
        normalizer["force"] = FieldNormalizer.from_data_limits(force)
        normalizer["state"] = FieldNormalizer.from_data_limits(state)
        normalizer["force_4x"] = FieldNormalizer.from_data_limits(force_4x)
        normalizer["state_4x"] = FieldNormalizer.from_data_limits(state_4x)
        return normalizer
