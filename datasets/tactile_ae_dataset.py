from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
import zarr
from torch.utils.data import Dataset

from tools.normalizer import FieldNormalizer
from tools.tactile_feat import extract_tactile_deformation


def resolve_replay_buffer_path(root_dir: str | Path) -> Path:
    root = Path(root_dir).expanduser()
    if root.name.endswith(".zarr") and root.is_dir():
        return root
    candidate = root / "replay_buffer.zarr"
    if candidate.is_dir():
        return candidate
    raise FileNotFoundError(
        f"cannot find replay_buffer.zarr from root_dir={root_dir}; tried {candidate}"
    )


def episode_bounds_from_zarr(zarr_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    root = zarr.open_group(str(zarr_path), mode="r")
    ends = np.asarray(root["meta"]["episode_ends"][:], dtype=np.int64)
    starts = np.concatenate([np.zeros(1, dtype=np.int64), ends[:-1]])
    return starts, ends


def split_episode_indices(
    num_episodes: int,
    *,
    val_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if num_episodes < 2:
        raise ValueError("at least two episodes are required for train/validation split")
    fraction = float(val_fraction)
    if not 0.0 < fraction < 1.0:
        raise ValueError(f"val_fraction must be in (0,1), got {val_fraction}")
    generator = np.random.default_rng(int(seed))
    shuffled = generator.permutation(num_episodes)
    num_val = min(num_episodes - 1, max(1, int(round(num_episodes * fraction))))
    val = np.sort(shuffled[:num_val])
    train = np.sort(shuffled[num_val:])
    return train, val


def fit_tactile_frame_normalizer(
    zarr_path: str | Path,
    episode_indices: Sequence[int],
    *,
    tactile_key: str = "tactile",
    output_range: tuple[float, float] = (-1.0, 1.0),
    batch_frames: int = 512,
) -> FieldNormalizer:
    root = zarr.open_group(str(zarr_path), mode="r")
    tactile = root["data"][str(tactile_key)]
    starts, ends = episode_bounds_from_zarr(zarr_path)
    data_min: np.ndarray | None = None
    data_max: np.ndarray | None = None
    batch_frames = max(1, int(batch_frames))

    for ep_idx in (int(value) for value in episode_indices):
        for start in range(int(starts[ep_idx]), int(ends[ep_idx]), batch_frames):
            stop = min(start + batch_frames, int(ends[ep_idx]))
            deformation = extract_tactile_deformation(
                np.asarray(tactile[start:stop], dtype=np.float32)
            )
            flat = deformation.reshape(-1, deformation.shape[-1])
            chunk_min = flat.min(axis=0)
            chunk_max = flat.max(axis=0)
            data_min = chunk_min if data_min is None else np.minimum(data_min, chunk_min)
            data_max = chunk_max if data_max is None else np.maximum(data_max, chunk_max)

    if data_min is None or data_max is None:
        raise ValueError("no tactile frames were available while fitting normalizer")
    return FieldNormalizer.from_limits(
        data_min,
        data_max,
        output_min=float(output_range[0]),
        output_max=float(output_range[1]),
    )


class TactileAEFrameDataset(Dataset):
    """Lazy single-frame tactile dataset for Stage 1 autoencoder training."""

    def __init__(
        self,
        root_dir: str | Path,
        *,
        episode_indices: Iterable[int],
        normalizer: FieldNormalizer,
        tactile_key: str = "tactile",
        frame_stride: int = 1,
    ) -> None:
        super().__init__()
        self.zarr_path = resolve_replay_buffer_path(root_dir)
        self.tactile_key = str(tactile_key)
        self.normalizer = normalizer
        self.frame_stride = max(1, int(frame_stride))
        starts, ends = episode_bounds_from_zarr(self.zarr_path)

        frames: list[np.ndarray] = []
        episode_for_frame: list[np.ndarray] = []
        for ep_idx in (int(value) for value in episode_indices):
            if ep_idx < 0 or ep_idx >= len(ends):
                raise IndexError(f"episode index out of range: {ep_idx}")
            indices = np.arange(
                int(starts[ep_idx]),
                int(ends[ep_idx]),
                self.frame_stride,
                dtype=np.int64,
            )
            frames.append(indices)
            episode_for_frame.append(np.full(len(indices), ep_idx, dtype=np.int32))
        if not frames or sum(len(values) for values in frames) == 0:
            raise ValueError("TactileAEFrameDataset contains no frames")

        self.frame_indices = np.concatenate(frames)
        self.episode_indices = np.concatenate(episode_for_frame)
        self._tactile_array = None

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_tactile_array"] = None
        return state

    def _array(self):
        if self._tactile_array is None:
            root = zarr.open_group(str(self.zarr_path), mode="r")
            self._tactile_array = root["data"][self.tactile_key]
        return self._tactile_array

    def __len__(self) -> int:
        return int(len(self.frame_indices))

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | int]:
        index = int(index)
        frame_idx = int(self.frame_indices[index])
        raw = np.asarray(self._array()[frame_idx], dtype=np.float32)
        deformation = extract_tactile_deformation(raw[None])[0]
        normalized = self.normalizer.normalize_np(deformation).astype(
            np.float32,
            copy=False,
        )
        return {
            "tactile": torch.from_numpy(normalized),
            "frame_idx": frame_idx,
            "ep_idx": int(self.episode_indices[index]),
        }
