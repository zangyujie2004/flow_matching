"""Evaluate Stage-2 future tactile reconstruction on processed Zarr episodes.

This evaluator intentionally uses:
  * the Stage-2 checkpoint's state/tactile normalizers;
  * the Stage-2 checkpoint's frozen Stage-1 tactile autoencoder;
  * the processed dataset's precomputed DINO frame features.

It reports two different errors:
  1. ``stage2``: policy-predicted latent -> frozen decoder -> ground truth;
  2. ``ae_oracle``: ground-truth tactile -> frozen encoder/decoder -> ground truth.

The second error is the reconstruction floor of the frozen autoencoder.  Keeping
it separate prevents a weak Stage-1 decoder from being mistaken for a Stage-2
forecasting failure.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
import yaml
import zarr
from tqdm import tqdm

_FLOW_MATCHING_ROOT = Path(__file__).resolve().parents[1]
if str(_FLOW_MATCHING_ROOT) not in sys.path:
    sys.path.insert(0, str(_FLOW_MATCHING_ROOT))

from infer.config import load_run_config, load_runtime_checkpoint
from models.fm import resolve_tactile_condition_encoder_type
from tools.latent_cache import (
    default_latent_cache_root_dir,
    infer_token_mode_from_attrs_and_shape,
    resolve_frame_backbone_base_remove_hand_zarr_path,
    resolve_frame_backbone_zarr_path,
    validate_latent_cache_identity,
)
from tools.tactile_feat import TACTILE_BUNDLE_ORDER, extract_tactile_deformation
from utils.train_utils import cfg_get, set_seed


CAMERA_ORDER = ("base_0", "left_wrist_0", "right_wrist_0")
ROBOT_SLICES = {"joint": slice(0, 14), "eef": slice(14, 34)}
CHANNELS_PER_SENSOR = 3


@dataclass(frozen=True)
class EvalWindow:
    episode: int
    anchor: int


@dataclass(frozen=True)
class DistributedContext:
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1
    backend: str | None = None

    @property
    def enabled(self) -> bool:
        return self.world_size > 1

    @property
    def is_main(self) -> bool:
        return self.rank == 0


@dataclass
class ModeEvaluation:
    stage2_raw: "ErrorAccumulator"
    stage2_normalized: "ErrorAccumulator"
    ae_raw: "ErrorAccumulator"
    ae_normalized: "ErrorAccumulator"
    latent_error: "ScalarErrorAccumulator"
    per_episode: dict[int, dict[str, Any]]
    num_windows: int
    num_plots: int


class ErrorAccumulator:
    """Streaming physical-space error statistics for four tactile sensors."""

    def __init__(self, num_sensors: int = 4) -> None:
        self.num_sensors = int(num_sensors)
        self.sq_sum = 0.0
        self.abs_sum = 0.0
        self.count = 0
        self.sensor_axis_sq = np.zeros((num_sensors, 3), dtype=np.float64)
        self.sensor_axis_abs = np.zeros((num_sensors, 3), dtype=np.float64)
        self.sensor_axis_count = np.zeros((num_sensors, 3), dtype=np.int64)
        self.tangent_sq = np.zeros(num_sensors, dtype=np.float64)
        self.tangent_abs = np.zeros(num_sensors, dtype=np.float64)
        self.tangent_count = np.zeros(num_sensors, dtype=np.int64)

    def update(self, prediction: np.ndarray, target: np.ndarray) -> None:
        pred = np.asarray(prediction, dtype=np.float32)
        gt = np.asarray(target, dtype=np.float32)
        if pred.shape != gt.shape or pred.ndim != 5:
            raise ValueError(
                "expected matching tactile arrays (B,T,H,W,12), got "
                f"prediction={pred.shape}, target={gt.shape}"
            )
        expected_channels = self.num_sensors * CHANNELS_PER_SENSOR
        if pred.shape[-1] != expected_channels:
            raise ValueError(
                f"expected {expected_channels} tactile channels, got {pred.shape[-1]}"
            )

        diff = pred.astype(np.float64) - gt.astype(np.float64)
        self.sq_sum += float(np.square(diff).sum())
        self.abs_sum += float(np.abs(diff).sum())
        self.count += int(diff.size)

        for sensor in range(self.num_sensors):
            channel = slice(sensor * 3, (sensor + 1) * 3)
            sensor_diff = diff[..., channel]
            reduce_axes = tuple(range(sensor_diff.ndim - 1))
            self.sensor_axis_sq[sensor] += np.square(sensor_diff).sum(axis=reduce_axes)
            self.sensor_axis_abs[sensor] += np.abs(sensor_diff).sum(axis=reduce_axes)
            self.sensor_axis_count[sensor] += np.prod(sensor_diff.shape[:-1], dtype=np.int64)

            pred_xy = pred[..., channel][..., :2].astype(np.float64)
            gt_xy = gt[..., channel][..., :2].astype(np.float64)
            pred_tangent = np.linalg.norm(pred_xy, axis=-1)
            gt_tangent = np.linalg.norm(gt_xy, axis=-1)
            tangent_diff = pred_tangent - gt_tangent
            self.tangent_sq[sensor] += float(np.square(tangent_diff).sum())
            self.tangent_abs[sensor] += float(np.abs(tangent_diff).sum())
            self.tangent_count[sensor] += int(tangent_diff.size)

    def merge(self, other: "ErrorAccumulator") -> None:
        self.sq_sum += other.sq_sum
        self.abs_sum += other.abs_sum
        self.count += other.count
        self.sensor_axis_sq += other.sensor_axis_sq
        self.sensor_axis_abs += other.sensor_axis_abs
        self.sensor_axis_count += other.sensor_axis_count
        self.tangent_sq += other.tangent_sq
        self.tangent_abs += other.tangent_abs
        self.tangent_count += other.tangent_count

    def state_dict(self) -> dict[str, Any]:
        return {
            "num_sensors": self.num_sensors,
            "sq_sum": self.sq_sum,
            "abs_sum": self.abs_sum,
            "count": self.count,
            "sensor_axis_sq": self.sensor_axis_sq.tolist(),
            "sensor_axis_abs": self.sensor_axis_abs.tolist(),
            "sensor_axis_count": self.sensor_axis_count.tolist(),
            "tangent_sq": self.tangent_sq.tolist(),
            "tangent_abs": self.tangent_abs.tolist(),
            "tangent_count": self.tangent_count.tolist(),
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> "ErrorAccumulator":
        result = cls(num_sensors=int(state["num_sensors"]))
        result.sq_sum = float(state["sq_sum"])
        result.abs_sum = float(state["abs_sum"])
        result.count = int(state["count"])
        result.sensor_axis_sq = np.asarray(
            state["sensor_axis_sq"], dtype=np.float64
        )
        result.sensor_axis_abs = np.asarray(
            state["sensor_axis_abs"], dtype=np.float64
        )
        result.sensor_axis_count = np.asarray(
            state["sensor_axis_count"], dtype=np.int64
        )
        result.tangent_sq = np.asarray(state["tangent_sq"], dtype=np.float64)
        result.tangent_abs = np.asarray(state["tangent_abs"], dtype=np.float64)
        result.tangent_count = np.asarray(
            state["tangent_count"], dtype=np.int64
        )
        return result

    def reduction_vector(self) -> np.ndarray:
        return np.concatenate(
            [
                np.asarray(
                    [self.sq_sum, self.abs_sum, float(self.count)],
                    dtype=np.float64,
                ),
                self.sensor_axis_sq.reshape(-1),
                self.sensor_axis_abs.reshape(-1),
                self.sensor_axis_count.astype(np.float64).reshape(-1),
                self.tangent_sq.reshape(-1),
                self.tangent_abs.reshape(-1),
                self.tangent_count.astype(np.float64).reshape(-1),
            ]
        )

    def load_reduction_vector(self, vector: np.ndarray) -> None:
        values = np.asarray(vector, dtype=np.float64)
        num_axis = self.num_sensors * 3
        expected = 3 + 3 * num_axis + 3 * self.num_sensors
        if values.shape != (expected,):
            raise ValueError(
                f"invalid ErrorAccumulator reduction vector {values.shape}, "
                f"expected ({expected},)"
            )
        offset = 0
        self.sq_sum = float(values[offset])
        self.abs_sum = float(values[offset + 1])
        self.count = int(round(float(values[offset + 2])))
        offset += 3
        self.sensor_axis_sq = values[offset : offset + num_axis].reshape(
            self.num_sensors, 3
        )
        offset += num_axis
        self.sensor_axis_abs = values[offset : offset + num_axis].reshape(
            self.num_sensors, 3
        )
        offset += num_axis
        self.sensor_axis_count = np.rint(
            values[offset : offset + num_axis]
        ).astype(np.int64).reshape(self.num_sensors, 3)
        offset += num_axis
        self.tangent_sq = values[offset : offset + self.num_sensors]
        offset += self.num_sensors
        self.tangent_abs = values[offset : offset + self.num_sensors]
        offset += self.num_sensors
        self.tangent_count = np.rint(
            values[offset : offset + self.num_sensors]
        ).astype(np.int64)

    def summary(self) -> dict[str, Any]:
        if self.count <= 0:
            return {"mse": None, "mae": None, "per_sensor": {}}
        per_sensor: dict[str, Any] = {}
        for sensor, name in enumerate(TACTILE_BUNDLE_ORDER):
            axis_count = np.maximum(self.sensor_axis_count[sensor], 1)
            tangent_count = max(int(self.tangent_count[sensor]), 1)
            per_sensor[name] = {
                "axis_mse": {
                    axis: float(self.sensor_axis_sq[sensor, idx] / axis_count[idx])
                    for idx, axis in enumerate(("dx", "dy", "dz"))
                },
                "axis_mae": {
                    axis: float(self.sensor_axis_abs[sensor, idx] / axis_count[idx])
                    for idx, axis in enumerate(("dx", "dy", "dz"))
                },
                "tangent_magnitude_mse": float(
                    self.tangent_sq[sensor] / tangent_count
                ),
                "tangent_magnitude_mae": float(
                    self.tangent_abs[sensor] / tangent_count
                ),
            }
        return {
            "mse": float(self.sq_sum / self.count),
            "mae": float(self.abs_sum / self.count),
            "num_elements": int(self.count),
            "per_sensor": per_sensor,
        }


class ScalarErrorAccumulator:
    def __init__(self) -> None:
        self.sq_sum = 0.0
        self.abs_sum = 0.0
        self.count = 0

    def update(self, prediction: np.ndarray, target: np.ndarray) -> None:
        pred = np.asarray(prediction, dtype=np.float64)
        gt = np.asarray(target, dtype=np.float64)
        if pred.shape != gt.shape:
            raise ValueError(f"shape mismatch: {pred.shape} != {gt.shape}")
        diff = pred - gt
        self.sq_sum += float(np.square(diff).sum())
        self.abs_sum += float(np.abs(diff).sum())
        self.count += int(diff.size)

    def reduction_vector(self) -> np.ndarray:
        return np.asarray(
            [self.sq_sum, self.abs_sum, float(self.count)], dtype=np.float64
        )

    def load_reduction_vector(self, vector: np.ndarray) -> None:
        values = np.asarray(vector, dtype=np.float64)
        if values.shape != (3,):
            raise ValueError(
                f"invalid ScalarErrorAccumulator reduction vector {values.shape}"
            )
        self.sq_sum = float(values[0])
        self.abs_sum = float(values[1])
        self.count = int(round(float(values[2])))

    def summary(self) -> dict[str, float | int | None]:
        if self.count <= 0:
            return {"mse": None, "mae": None, "num_elements": 0}
        return {
            "mse": float(self.sq_sum / self.count),
            "mae": float(self.abs_sum / self.count),
            "num_elements": int(self.count),
        }


def _resolve_replay_buffer(data_root: Path) -> Path:
    if data_root.name.endswith(".zarr") and data_root.is_dir():
        return data_root
    candidate = data_root / "replay_buffer.zarr"
    if not candidate.is_dir():
        raise FileNotFoundError(f"replay buffer not found: {candidate}")
    return candidate


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"YAML root must be a mapping: {path}")
    return payload


def resolve_subtask_episode_range(
    data_root: Path,
    subtask_path: Path,
    episode_ends: np.ndarray,
) -> tuple[int, int, dict[str, int]]:
    """Recover preprocessing episode offsets from the copied preprocess config."""
    config_path = data_root / "config.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(
            f"processed config is required for --subtask-path: {config_path}"
        )
    config = _load_yaml(config_path)
    io_cfg = config.get("io")
    if not isinstance(io_cfg, Mapping):
        raise KeyError(f"missing io section in {config_path}")
    subtasks = [str(value) for value in io_cfg.get("subtasks", [])]
    subtask_name = subtask_path.expanduser().resolve().name
    if subtask_name not in subtasks:
        raise ValueError(
            f"subtask {subtask_name!r} is not present in processed config: {subtasks}"
        )

    input_base = Path(str(io_cfg["input_base_path"])).expanduser()
    task_name = str(io_cfg["task"])
    max_episode_length = int(
        io_cfg.get("max_episode_length", io_cfg.get("episode_length", -1))
    )
    counts: dict[str, int] = {}
    for name in subtasks:
        raw_dir = input_base / task_name / name
        if not raw_dir.is_dir():
            raise FileNotFoundError(
                f"cannot resolve subtask offsets because raw directory is missing: {raw_dir}; "
                "pass --episode-start/--episode-end explicitly"
            )
        count = len(sorted(entry.name for entry in raw_dir.iterdir() if entry.exists()))
        if max_episode_length != -1:
            count = min(count, max_episode_length)
        counts[name] = int(count)

    if sum(counts.values()) != len(episode_ends):
        raise ValueError(
            "raw subtask counts do not match processed episode count: "
            f"raw={sum(counts.values())}, processed={len(episode_ends)}; "
            "pass --episode-start/--episode-end explicitly"
        )
    start = sum(counts[name] for name in subtasks[: subtasks.index(subtask_name)])
    end = start + counts[subtask_name]
    return int(start), int(end), counts


def _parse_int_list(text: str) -> list[int]:
    values = [int(part.strip()) for part in str(text).split(",") if part.strip()]
    if not values:
        raise ValueError("integer list must contain at least one value")
    return values


def build_windows(
    episode_ends: np.ndarray,
    *,
    episode_start: int,
    episode_end: int,
    observation_steps: int,
    action_horizon: int,
    tactile_target_offset: int,
    sample_stride: int,
) -> list[EvalWindow]:
    starts = np.concatenate(
        [np.array([0], dtype=np.int64), np.asarray(episode_ends[:-1], dtype=np.int64)]
    )
    windows: list[EvalWindow] = []
    for episode in range(int(episode_start), int(episode_end)):
        ep_start = int(starts[episode])
        ep_end = int(episode_ends[episode])
        first_anchor = ep_start + int(observation_steps) - 1
        last_anchor = ep_end - int(action_horizon) - int(tactile_target_offset)
        for anchor in range(first_anchor, last_anchor + 1, int(sample_stride)):
            windows.append(EvalWindow(episode=episode, anchor=anchor))
    return windows


def limit_windows(
    windows: Sequence[EvalWindow],
    *,
    max_windows: int,
    seed: int,
) -> list[EvalWindow]:
    if int(max_windows) < 0 or len(windows) <= int(max_windows):
        return list(windows)
    if int(max_windows) == 0:
        raise ValueError("--max-windows must be positive or -1 for all windows")
    rng = np.random.default_rng(int(seed))
    indices = np.sort(
        rng.choice(len(windows), size=int(max_windows), replace=False)
    )
    return [windows[int(index)] for index in indices]


def initialize_distributed(*, dry_run: bool = False) -> DistributedContext:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size <= 1:
        return DistributedContext()
    if not dry_run and not torch.cuda.is_available():
        raise RuntimeError("multi-process tactile evaluation requires CUDA/NCCL")
    backend = "gloo" if dry_run else "nccl"
    if backend == "nccl":
        torch.cuda.set_device(local_rank)
    dist.init_process_group(backend=backend, init_method="env://")
    return DistributedContext(
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        backend=backend,
    )


def shard_windows(
    windows: Sequence[EvalWindow], context: DistributedContext
) -> list[EvalWindow]:
    """Deterministic non-overlapping rank partition after global sampling."""
    return list(windows[context.rank :: context.world_size])


def _all_reduce_accumulator(
    accumulator: ErrorAccumulator | ScalarErrorAccumulator,
    *,
    context: DistributedContext,
    device: torch.device,
) -> None:
    if not context.enabled:
        return
    vector = torch.from_numpy(accumulator.reduction_vector()).to(
        device=device, dtype=torch.float64
    )
    dist.all_reduce(vector, op=dist.ReduceOp.SUM)
    accumulator.load_reduction_vector(vector.cpu().numpy())


def _serialize_per_episode(
    per_episode: Mapping[int, Mapping[str, Any]],
) -> dict[int, dict[str, Any]]:
    return {
        int(episode): {
            "num_windows": int(record["num_windows"]),
            "stage2": record["stage2"].state_dict(),
            "oracle": record["oracle"].state_dict(),
        }
        for episode, record in per_episode.items()
    }


def _gather_per_episode(
    per_episode: Mapping[int, Mapping[str, Any]],
    *,
    context: DistributedContext,
) -> dict[int, dict[str, Any]]:
    if not context.enabled:
        return {
            int(episode): {
                "num_windows": int(record["num_windows"]),
                "stage2": record["stage2"],
                "oracle": record["oracle"],
            }
            for episode, record in per_episode.items()
        }

    gathered: list[dict[int, dict[str, Any]] | None] = [
        None for _ in range(context.world_size)
    ]
    dist.all_gather_object(gathered, _serialize_per_episode(per_episode))
    if not context.is_main:
        return {}

    merged: dict[int, dict[str, Any]] = {}
    for rank_payload in gathered:
        if rank_payload is None:
            continue
        for episode, record in rank_payload.items():
            target = merged.setdefault(
                int(episode),
                {
                    "num_windows": 0,
                    "stage2": ErrorAccumulator(),
                    "oracle": ErrorAccumulator(),
                },
            )
            target["num_windows"] += int(record["num_windows"])
            target["stage2"].merge(
                ErrorAccumulator.from_state_dict(record["stage2"])
            )
            target["oracle"].merge(
                ErrorAccumulator.from_state_dict(record["oracle"])
            )
    return merged


class TactileEvalData:
    """Lazy Zarr reader for state, tactile, and precomputed visual features."""

    def __init__(
        self,
        *,
        data_root: Path,
        policy_cfg: Mapping[str, Any],
        latent_cache_root: Path | None,
    ) -> None:
        self.data_root = data_root.expanduser().resolve()
        self.replay_path = _resolve_replay_buffer(self.data_root)
        self.replay = zarr.open_group(str(self.replay_path), mode="r")
        self.data = self.replay["data"]
        self.meta = self.replay["meta"]
        self.episode_ends = np.asarray(self.meta["episode_ends"][:], dtype=np.int64)
        self.episode_starts = np.concatenate(
            [np.array([0], dtype=np.int64), self.episode_ends[:-1]]
        )
        self.num_frames = int(self.episode_ends[-1])

        data_cfg = dict(policy_cfg["data"])
        fm_cfg = dict(policy_cfg["models"]["fm"])
        self.window_size = int(data_cfg["window_size"])
        self.n_image_steps = int(data_cfg.get("n_image_steps", 1))
        self.action_horizon = int(data_cfg["action_horizon"])
        self.tactile_obs_steps = int(data_cfg.get("tactile_obs_steps", 1))
        self.tactile_target_offset = int(data_cfg.get("tactile_target_offset", 1))
        self.tactile_condition_encoder_type = (
            resolve_tactile_condition_encoder_type(
                predict_tactile=bool(fm_cfg.get("predict_tactile", False)),
                tactile_encoder_type=fm_cfg.get("tactile_encoder_type"),
                tactile_condition_encoder_type=fm_cfg.get(
                    "tactile_condition_encoder_type"
                ),
            )
        )
        self.action_type = str(data_cfg.get("action_type", "eef"))
        if self.action_type not in ROBOT_SLICES:
            raise ValueError(f"unsupported action_type={self.action_type!r}")
        self.robot_slice = ROBOT_SLICES[self.action_type]
        self.camera_views = tuple(
            str(value) for value in data_cfg.get("camera_views", CAMERA_ORDER)
        )
        if self.camera_views != CAMERA_ORDER:
            raise ValueError(
                "this evaluator currently requires the trained three-camera order "
                f"{CAMERA_ORDER}, got {self.camera_views}"
            )

        cache_root = (
            latent_cache_root.expanduser().resolve()
            if latent_cache_root is not None
            else Path(
                default_latent_cache_root_dir(
                    str(self.data_root),
                    fm_cfg=fm_cfg,
                )
            )
        )
        self.cache_root = cache_root
        main_path = Path(resolve_frame_backbone_zarr_path(str(cache_root)))
        if not main_path.is_dir():
            raise FileNotFoundError(
                f"DINO frame cache not found: {main_path}; run scripts/precompute.sh first"
            )
        main_root = zarr.open_group(str(main_path), mode="r")
        main_attrs = dict(main_root.attrs)
        validate_latent_cache_identity(main_attrs, fm_cfg, cache_path=str(main_path))
        self.visual = main_root["data"]["frame_image_backbone_feat"]
        if int(self.visual.shape[0]) != self.num_frames:
            raise ValueError(
                f"visual cache frames={self.visual.shape[0]} != replay frames={self.num_frames}"
            )
        self.token_mode = infer_token_mode_from_attrs_and_shape(
            main_attrs, tuple(int(value) for value in self.visual.shape)
        )
        cache_views = tuple(
            part.strip()
            for part in str(main_attrs.get("camera_views", "")).split(",")
            if part.strip()
        )
        if cache_views != self.camera_views:
            raise ValueError(
                f"visual cache views={cache_views} != trained views={self.camera_views}"
            )

        self.remove_visual = None
        self.remove_episode_ends = None
        self.remove_flags: list[str] | None = None
        remove_path = Path(
            resolve_frame_backbone_base_remove_hand_zarr_path(str(cache_root))
        )
        meta_path = self.data_root / "meta.json"
        if remove_path.is_dir() and meta_path.is_file():
            remove_root = zarr.open_group(str(remove_path), mode="r")
            remove_attrs = dict(remove_root.attrs)
            validate_latent_cache_identity(
                remove_attrs, fm_cfg, cache_path=str(remove_path)
            )
            remove_mode = infer_token_mode_from_attrs_and_shape(
                remove_attrs,
                tuple(
                    int(value)
                    for value in remove_root["data"]["frame_image_backbone_feat"].shape
                ),
            )
            if remove_mode != self.token_mode:
                raise ValueError(
                    f"remove-hand token mode={remove_mode} != main={self.token_mode}"
                )
            self.remove_visual = remove_root["data"]["frame_image_backbone_feat"]
            self.remove_episode_ends = np.asarray(
                self.meta["episode_ends_remove_hand"][:], dtype=np.int64
            )
            with meta_path.open(encoding="utf-8") as handle:
                metadata = json.load(handle)
            self.remove_flags = list(
                metadata["dataset"]["base_remove_hand"]["per_episode"]
            )
            if len(self.remove_flags) != len(self.episode_ends):
                raise ValueError("base_remove_hand per_episode length mismatch")

    def validate_remove_hand(self, episodes: Iterable[int]) -> None:
        if (
            self.remove_visual is None
            or self.remove_episode_ends is None
            or self.remove_flags is None
        ):
            raise FileNotFoundError(
                f"remove-hand metadata/cache is incomplete below {self.cache_root}"
            )
        missing = [
            int(ep) for ep in episodes if self.remove_flags[int(ep)] != "present"
        ]
        if missing:
            raise ValueError(
                f"remove-hand image is absent for selected episodes: {missing[:10]}"
            )

    def _remove_compact_index(self, episode: int, global_index: int) -> int:
        assert self.remove_episode_ends is not None
        assert self.remove_flags is not None
        present_before = sum(
            flag == "present" for flag in self.remove_flags[: int(episode)]
        )
        compact_start = (
            0
            if present_before == 0
            else int(self.remove_episode_ends[present_before - 1])
        )
        offset = int(global_index) - int(self.episode_starts[int(episode)])
        return compact_start + offset

    def gather_batch(
        self,
        windows: Sequence[EvalWindow],
        *,
        base_mode: str,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        state_batch = []
        visual_batch = []
        tactile_batch = []
        for window in windows:
            obs_start = window.anchor - self.window_size + 1
            image_start = window.anchor - self.n_image_steps + 1
            state = np.asarray(
                self.data["state_30hz"][obs_start : window.anchor + 1],
                dtype=np.float32,
            )[..., self.robot_slice]
            visual = np.asarray(
                self.visual[image_start : window.anchor + 1], dtype=np.float32
            )
            if base_mode == "remove":
                assert self.remove_visual is not None
                visual = np.array(visual, copy=True)
                compact_indices = [
                    self._remove_compact_index(window.episode, frame)
                    for frame in range(image_start, window.anchor + 1)
                ]
                remove = np.asarray(
                    self.remove_visual[
                        compact_indices[0] : compact_indices[-1] + 1
                    ],
                    dtype=np.float32,
                )
                if self.token_mode == "cls":
                    if remove.ndim == 3 and remove.shape[1] == 1:
                        remove = remove[:, 0]
                    visual[:, 0, :] = remove
                else:
                    if remove.ndim == 4 and remove.shape[1] == 1:
                        remove = remove[:, 0]
                    visual[:, 0, :, :] = remove
            tactile = np.asarray(
                self.data["tactile"][
                    window.anchor - self.tactile_obs_steps + 1 : window.anchor + 1
                ],
                dtype=np.float32,
            )
            state_batch.append(state)
            visual_batch.append(visual)
            tactile_batch.append(extract_tactile_deformation(tactile))
        return (
            np.stack(state_batch),
            np.stack(visual_batch),
            np.stack(tactile_batch),
        )

    def gather_ground_truth(self, windows: Sequence[EvalWindow]) -> np.ndarray:
        target_batch = []
        for window in windows:
            start = window.anchor + self.tactile_target_offset
            stop = start + self.action_horizon
            raw = np.asarray(self.data["tactile"][start:stop], dtype=np.float32)
            deformation = extract_tactile_deformation(raw)
            if deformation.shape[0] != self.action_horizon:
                raise ValueError(
                    f"episode={window.episode} anchor={window.anchor}: tactile target "
                    f"length={deformation.shape[0]} != {self.action_horizon}"
                )
            target_batch.append(deformation)
        return np.stack(target_batch)


def build_tactile_condition_obs(
    policy: torch.nn.Module,
    tactile_history_normalized: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Build the policy tactile input for legacy and history-aware checkpoints."""
    condition_type = str(policy.tactile_condition_encoder_type)
    if condition_type != "precomputed":
        return {"tactile": tactile_history_normalized}
    latest = tactile_history_normalized[:, -1]
    current_raw_latent = policy.tactile_autoencoder.encode_flattened(latest)
    current_latent = policy._normalize_tactile_latent(current_raw_latent)
    return {"tactile_latent": current_latent}


def _autocast_context(device: torch.device, amp: str):
    if amp == "none":
        return nullcontext()
    if device.type != "cuda":
        raise ValueError(f"--amp={amp} requires a CUDA device")
    dtype = torch.bfloat16 if amp == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def encode_decode_ground_truth(
    policy: torch.nn.Module,
    normalized_gt: torch.Tensor,
    *,
    frame_batch_size: int,
    amp: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if policy.tactile_autoencoder is None:
        raise RuntimeError("Stage-2 policy has no tactile autoencoder")
    batch, horizon, height, width, channels = normalized_gt.shape
    flat = normalized_gt.reshape(batch * horizon, height, width, channels)
    latent_parts = []
    reconstruction_parts = []
    for start in range(0, flat.shape[0], int(frame_batch_size)):
        chunk = flat[start : start + int(frame_batch_size)]
        with _autocast_context(chunk.device, amp):
            raw_latent = policy.tactile_autoencoder.encode_flattened(chunk)
            latent = policy._normalize_tactile_latent(raw_latent)
            reconstruction = policy.tactile_autoencoder.decode_flattened(raw_latent)
        latent_parts.append(latent.float())
        reconstruction_parts.append(reconstruction.float())
    latent = torch.cat(latent_parts, dim=0).reshape(batch, horizon, -1)
    reconstruction = torch.cat(reconstruction_parts, dim=0).reshape(
        batch, horizon, height, width, channels
    )
    return latent, reconstruction


def decode_prediction(
    policy: torch.nn.Module,
    normalized_latent: torch.Tensor,
    *,
    frame_batch_size: int,
    amp: str,
) -> torch.Tensor:
    batch, horizon, latent_dim = normalized_latent.shape
    flat = normalized_latent.reshape(batch * horizon, latent_dim)
    parts = []
    for start in range(0, flat.shape[0], int(frame_batch_size)):
        chunk = flat[start : start + int(frame_batch_size)]
        with _autocast_context(chunk.device, amp):
            decoded = policy.decode_tactile_latent(chunk)
        parts.append(decoded.float())
    decoded = torch.cat(parts, dim=0)
    return decoded.reshape(batch, horizon, *decoded.shape[1:])


def _symmetric_limit(*arrays: np.ndarray) -> float:
    finite = np.concatenate(
        [np.asarray(array, dtype=np.float32).reshape(-1) for array in arrays]
    )
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 1.0
    limit = float(np.quantile(np.abs(finite), 0.995))
    return max(limit, 1e-8)


def save_heatmaps(
    path: Path,
    *,
    prediction: np.ndarray,
    target: np.ndarray,
    horizons: Sequence[int],
    title: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    valid_horizons = [int(h) for h in horizons if 0 <= int(h) < target.shape[0]]
    for horizon in valid_horizons:
        fig, axes = plt.subplots(4, 6, figsize=(19, 11), constrained_layout=True)
        for sensor, sensor_name in enumerate(TACTILE_BUNDLE_ORDER):
            channel = slice(sensor * 3, (sensor + 1) * 3)
            gt_sensor = target[horizon, ..., channel]
            pred_sensor = prediction[horizon, ..., channel]
            gt_tangent = np.linalg.norm(gt_sensor[..., :2], axis=-1)
            pred_tangent = np.linalg.norm(pred_sensor[..., :2], axis=-1)
            tangent_error = np.abs(pred_tangent - gt_tangent)
            gt_z = gt_sensor[..., 2]
            pred_z = pred_sensor[..., 2]
            z_error = np.abs(pred_z - gt_z)
            tangent_max = max(
                float(np.quantile(np.concatenate([gt_tangent.ravel(), pred_tangent.ravel()]), 0.995)),
                1e-8,
            )
            z_limit = _symmetric_limit(gt_z, pred_z)
            columns = (
                (gt_tangent, "GT |dxy|", "viridis", 0.0, tangent_max),
                (pred_tangent, "Pred |dxy|", "viridis", 0.0, tangent_max),
                (tangent_error, "Abs err |dxy|", "magma", 0.0, tangent_max),
                (gt_z, "GT dz", "coolwarm", -z_limit, z_limit),
                (pred_z, "Pred dz", "coolwarm", -z_limit, z_limit),
                (z_error, "Abs err dz", "magma", 0.0, z_limit),
            )
            for col, (image, label, cmap, vmin, vmax) in enumerate(columns):
                axis = axes[sensor, col]
                handle = axis.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)
                axis.set_xticks([])
                axis.set_yticks([])
                axis.set_title(label if sensor == 0 else "", fontsize=9)
                if col == 0:
                    axis.set_ylabel(sensor_name, fontsize=9)
                fig.colorbar(handle, ax=axis, fraction=0.046, pad=0.02)
        fig.suptitle(f"{title} | future_index={horizon}", fontsize=12)
        fig.savefig(path.with_name(f"{path.stem}_h{horizon:03d}.png"), dpi=130)
        plt.close(fig)


def save_temporal_curves(
    path: Path,
    *,
    prediction: np.ndarray,
    target: np.ndarray,
    title: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(4, 1, figsize=(13, 12), sharex=True, constrained_layout=True)
    time = np.arange(target.shape[0], dtype=np.float32) / 30.0
    colors = {"dx": "tab:red", "dy": "tab:green", "dz": "tab:blue", "|dxy|": "tab:purple"}
    for sensor, sensor_name in enumerate(TACTILE_BUNDLE_ORDER):
        channel = slice(sensor * 3, (sensor + 1) * 3)
        gt = target[..., channel]
        pred = prediction[..., channel]
        gt_axis = np.mean(np.abs(gt), axis=(1, 2))
        pred_axis = np.mean(np.abs(pred), axis=(1, 2))
        gt_tangent = np.mean(np.linalg.norm(gt[..., :2], axis=-1), axis=(1, 2))
        pred_tangent = np.mean(np.linalg.norm(pred[..., :2], axis=-1), axis=(1, 2))
        axis = axes[sensor]
        for idx, name in enumerate(("dx", "dy", "dz")):
            axis.plot(time, gt_axis[:, idx], color=colors[name], label=f"GT {name}")
            axis.plot(
                time,
                pred_axis[:, idx],
                color=colors[name],
                linestyle="--",
                label=f"Pred {name}",
            )
        axis.plot(time, gt_tangent, color=colors["|dxy|"], label="GT |dxy|")
        axis.plot(
            time,
            pred_tangent,
            color=colors["|dxy|"],
            linestyle="--",
            label="Pred |dxy|",
        )
        axis.set_ylabel(sensor_name)
        axis.grid(alpha=0.25)
        if sensor == 0:
            axis.legend(ncol=4, fontsize=8)
    axes[-1].set_xlabel("future time (s), first target is t+1")
    fig.suptitle(title)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _metric_row(
    episode: int,
    num_windows: int,
    stage2: ErrorAccumulator,
    oracle: ErrorAccumulator,
) -> dict[str, Any]:
    stage2_summary = stage2.summary()
    oracle_summary = oracle.summary()
    return {
        "episode": int(episode),
        "num_windows": int(num_windows),
        "stage2_mse": stage2_summary["mse"],
        "stage2_mae": stage2_summary["mae"],
        "ae_oracle_mse": oracle_summary["mse"],
        "ae_oracle_mae": oracle_summary["mae"],
    }


def evaluate_mode(
    *,
    mode: str,
    windows: Sequence[EvalWindow],
    source: TactileEvalData,
    policy: torch.nn.Module,
    normalizer: Any,
    device: torch.device,
    batch_size: int,
    decode_frame_batch_size: int,
    num_inference_steps: int,
    solver: str,
    amp: str,
    seed: int,
    output_dir: Path,
    plot_samples: int,
    plot_horizons: Sequence[int],
    save_arrays: bool,
    context: DistributedContext,
) -> ModeEvaluation:
    # The same rank seed is reused for original/remove so paired modes receive
    # the same flow noise. Different ranks use different, deterministic streams.
    set_seed(int(seed) + int(context.rank))
    stage2_raw = ErrorAccumulator()
    stage2_normalized = ErrorAccumulator()
    ae_raw = ErrorAccumulator()
    ae_normalized = ErrorAccumulator()
    latent_error = ScalarErrorAccumulator()
    per_episode: dict[int, dict[str, Any]] = {}
    plotted = 0

    pbar = tqdm(
        range(0, len(windows), int(batch_size)),
        desc=f"TactileEval[{mode}][rank={context.rank}]",
        disable=not context.is_main,
    )
    for batch_start in pbar:
        batch_windows = list(windows[batch_start : batch_start + int(batch_size)])
        state_raw, visual, tactile_current = source.gather_batch(
            batch_windows, base_mode=mode
        )
        target_raw = source.gather_ground_truth(batch_windows)
        state_normalized = normalizer.normalize_state_np(state_raw)
        tactile_current_normalized = normalizer.normalize_tactile_np(tactile_current)
        target_normalized = normalizer.normalize_tactile_np(target_raw)

        state_tensor = torch.from_numpy(state_normalized).to(device=device)
        visual_tensor = torch.from_numpy(visual).to(device=device)
        current_tensor = torch.from_numpy(tactile_current_normalized).to(device=device)
        target_tensor = torch.from_numpy(target_normalized).to(device=device)

        with torch.inference_mode():
            with _autocast_context(device, amp):
                tactile_condition = build_tactile_condition_obs(
                    policy, current_tensor
                )
                policy_obs = {
                    "state": state_tensor,
                    "image_backbone_feat": visual_tensor,
                    **tactile_condition,
                }
                result = policy.predict_action(
                    policy_obs,
                    num_inference_steps=int(num_inference_steps),
                    solver=str(solver),
                    decode_tactile=False,
                )
            predicted_latent = result["tactile_latent_pred_normalized"].float()
            predicted_normalized = decode_prediction(
                policy,
                predicted_latent,
                frame_batch_size=decode_frame_batch_size,
                amp=amp,
            )
            target_latent, oracle_normalized = encode_decode_ground_truth(
                policy,
                target_tensor,
                frame_batch_size=decode_frame_batch_size,
                amp=amp,
            )

        predicted_raw = normalizer.tactile.unnormalize(predicted_normalized).float()
        oracle_raw = normalizer.tactile.unnormalize(oracle_normalized).float()
        prediction_np = predicted_raw.cpu().numpy()
        oracle_np = oracle_raw.cpu().numpy()
        prediction_normalized_np = predicted_normalized.cpu().numpy()
        oracle_normalized_np = oracle_normalized.cpu().numpy()
        predicted_latent_np = predicted_latent.cpu().numpy()
        target_latent_np = target_latent.cpu().numpy()

        stage2_raw.update(prediction_np, target_raw)
        stage2_normalized.update(prediction_normalized_np, target_normalized)
        ae_raw.update(oracle_np, target_raw)
        ae_normalized.update(oracle_normalized_np, target_normalized)
        latent_error.update(predicted_latent_np, target_latent_np)

        for item_index, window in enumerate(batch_windows):
            record = per_episode.setdefault(
                int(window.episode),
                {
                    "num_windows": 0,
                    "stage2": ErrorAccumulator(),
                    "oracle": ErrorAccumulator(),
                },
            )
            record["num_windows"] += 1
            record["stage2"].update(
                prediction_np[item_index : item_index + 1],
                target_raw[item_index : item_index + 1],
            )
            record["oracle"].update(
                oracle_np[item_index : item_index + 1],
                target_raw[item_index : item_index + 1],
            )

            if context.is_main and plotted < int(plot_samples):
                sample_name = (
                    f"ep{window.episode:04d}_anchor{window.anchor:07d}_{mode}"
                )
                sample_dir = output_dir / "samples" / sample_name
                sample_dir.mkdir(parents=True, exist_ok=True)
                save_heatmaps(
                    sample_dir / "stage2_heatmap.png",
                    prediction=prediction_np[item_index],
                    target=target_raw[item_index],
                    horizons=plot_horizons,
                    title=sample_name,
                )
                save_temporal_curves(
                    sample_dir / "stage2_temporal_curves.png",
                    prediction=prediction_np[item_index],
                    target=target_raw[item_index],
                    title=sample_name,
                )
                if save_arrays:
                    np.savez_compressed(
                        sample_dir / "reconstruction.npz",
                        tactile_gt=target_raw[item_index],
                        tactile_stage2=prediction_np[item_index],
                        tactile_ae_oracle=oracle_np[item_index],
                        tactile_latent_gt_normalized=target_latent_np[item_index],
                        tactile_latent_stage2_normalized=predicted_latent_np[item_index],
                        episode=np.int64(window.episode),
                        anchor=np.int64(window.anchor),
                        tactile_target_offset=np.int64(source.tactile_target_offset),
                    )
                plotted += 1

        current = stage2_raw.summary()
        pbar.set_postfix(mae=f"{current['mae']:.6g}")

    return ModeEvaluation(
        stage2_raw=stage2_raw,
        stage2_normalized=stage2_normalized,
        ae_raw=ae_raw,
        ae_normalized=ae_normalized,
        latent_error=latent_error,
        per_episode=per_episode,
        num_windows=len(windows),
        num_plots=int(plotted),
    )


def reduce_and_summarize_mode(
    *,
    mode: str,
    evaluation: ModeEvaluation,
    context: DistributedContext,
    device: torch.device,
    output_dir: Path,
) -> dict[str, Any] | None:
    for accumulator in (
        evaluation.stage2_raw,
        evaluation.stage2_normalized,
        evaluation.ae_raw,
        evaluation.ae_normalized,
        evaluation.latent_error,
    ):
        _all_reduce_accumulator(
            accumulator,
            context=context,
            device=device,
        )

    merged_per_episode = _gather_per_episode(
        evaluation.per_episode,
        context=context,
    )
    count_tensor = torch.tensor(
        [evaluation.num_windows, evaluation.num_plots],
        dtype=torch.int64,
        device=device,
    )
    if context.enabled:
        dist.all_reduce(count_tensor, op=dist.ReduceOp.SUM)
    total_windows, total_plots = (
        int(value) for value in count_tensor.cpu().tolist()
    )
    if not context.is_main:
        return None

    rows = [
        _metric_row(
            episode,
            int(record["num_windows"]),
            record["stage2"],
            record["oracle"],
        )
        for episode, record in sorted(merged_per_episode.items())
    ]
    if not rows:
        raise RuntimeError("distributed evaluation produced no per-episode rows")
    csv_path = output_dir / f"per_episode_{mode}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    return {
        "base_mode": mode,
        "num_windows": total_windows,
        "num_episodes": len(merged_per_episode),
        "stage2_physical": evaluation.stage2_raw.summary(),
        "stage2_normalized": evaluation.stage2_normalized.summary(),
        "stage2_latent_normalized": evaluation.latent_error.summary(),
        "ae_oracle_physical": evaluation.ae_raw.summary(),
        "ae_oracle_normalized": evaluation.ae_normalized.summary(),
        "per_episode_csv": str(csv_path),
        "num_visualized_samples": total_plots,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate decoded Stage-2 tactile prediction against Zarr ground truth."
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--data-root", required=True)
    parser.add_argument(
        "--subtask-path",
        default=None,
        help="Raw subtask path; episode offsets are recovered from processed config.yaml.",
    )
    parser.add_argument("--episode-start", type=int, default=None)
    parser.add_argument("--episode-end", type=int, default=None, help="Exclusive.")
    parser.add_argument("--latent-cache-root", default=None)
    parser.add_argument(
        "--base-mode", choices=("original", "remove", "both"), default="original"
    )
    parser.add_argument("--sample-stride", type=int, default=30)
    parser.add_argument(
        "--max-windows",
        type=int,
        default=32,
        help="Random cap after stride sampling; -1 evaluates every selected window.",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--decode-frame-batch-size", type=int, default=128)
    parser.add_argument("--num-inference-steps", type=int, default=None)
    parser.add_argument("--solver", choices=("euler", "heun"), default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--amp", choices=("none", "bf16", "fp16"), default="bf16")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--plot-samples", type=int, default=4)
    parser.add_argument("--plot-horizons", default="0,7,31,63,127")
    parser.add_argument(
        "--save-arrays", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate data/cache/subtask mapping without loading the checkpoint.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.sample_stride <= 0:
        raise ValueError("--sample-stride must be positive")
    if args.batch_size <= 0 or args.decode_frame_batch_size <= 0:
        raise ValueError("batch sizes must be positive")
    if args.plot_samples < 0:
        raise ValueError("--plot-samples must be non-negative")

    context = initialize_distributed(dry_run=bool(args.dry_run))
    try:
        run_evaluation(args, context)
    finally:
        if context.enabled and dist.is_initialized():
            dist.destroy_process_group()


def run_evaluation(
    args: argparse.Namespace,
    context: DistributedContext,
) -> None:

    run_dir = Path(args.run_dir).expanduser().resolve()
    data_root = Path(args.data_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    cfg = load_run_config(run_dir)
    source = TactileEvalData(
        data_root=data_root,
        policy_cfg=cfg,
        latent_cache_root=(
            None
            if args.latent_cache_root is None
            else Path(args.latent_cache_root)
        ),
    )

    if args.subtask_path is not None:
        if args.episode_start is not None or args.episode_end is not None:
            raise ValueError(
                "use either --subtask-path or --episode-start/--episode-end, not both"
            )
        episode_start, episode_end, subtask_counts = resolve_subtask_episode_range(
            data_root,
            Path(args.subtask_path),
            source.episode_ends,
        )
    else:
        subtask_counts = {}
        episode_start = 0 if args.episode_start is None else int(args.episode_start)
        episode_end = (
            len(source.episode_ends)
            if args.episode_end is None
            else int(args.episode_end)
        )
    if not 0 <= episode_start < episode_end <= len(source.episode_ends):
        raise ValueError(
            f"invalid episode range [{episode_start}, {episode_end}) for "
            f"{len(source.episode_ends)} episodes"
        )

    all_windows = build_windows(
        source.episode_ends,
        episode_start=episode_start,
        episode_end=episode_end,
        observation_steps=max(
            source.window_size,
            source.n_image_steps,
            source.tactile_obs_steps,
        ),
        action_horizon=source.action_horizon,
        tactile_target_offset=source.tactile_target_offset,
        sample_stride=int(args.sample_stride),
    )
    windows = limit_windows(
        all_windows,
        max_windows=int(args.max_windows),
        seed=int(args.seed),
    )
    if not windows:
        raise RuntimeError(
            "no valid evaluation windows remain after episode/stride selection"
        )
    modes = ["original", "remove"] if args.base_mode == "both" else [args.base_mode]
    if "remove" in modes:
        source.validate_remove_hand(range(episode_start, episode_end))
    rank_windows = shard_windows(windows, context)

    selection = {
        "run_dir": str(run_dir),
        "data_root": str(data_root),
        "replay_buffer": str(source.replay_path),
        "visual_cache_root": str(source.cache_root),
        "visual_token_mode": source.token_mode,
        "action_type": source.action_type,
        "episode_range": [int(episode_start), int(episode_end)],
        "num_selected_episodes": int(episode_end - episode_start),
        "sample_stride": int(args.sample_stride),
        "num_windows_before_cap": len(all_windows),
        "num_windows": len(windows),
        "distributed_world_size": context.world_size,
        "windows_per_rank": [
            len(windows[rank :: context.world_size])
            for rank in range(context.world_size)
        ],
        "base_modes": modes,
        "window_size": source.window_size,
        "tactile_obs_steps": source.tactile_obs_steps,
        "tactile_condition_encoder_type": source.tactile_condition_encoder_type,
        "action_horizon": source.action_horizon,
        "tactile_target_offset": source.tactile_target_offset,
        "subtask_path": args.subtask_path,
        "subtask_counts": subtask_counts,
    }
    if context.is_main:
        print(json.dumps(selection, indent=2, ensure_ascii=False))
    if args.dry_run:
        if context.is_main:
            print("[dry-run] data, cache, episode mapping, and windows are valid")
        return

    if context.is_main:
        output_dir.mkdir(parents=True, exist_ok=True)
    if context.enabled:
        dist.barrier()
    checkpoint = (
        Path(args.checkpoint).expanduser().resolve()
        if args.checkpoint
        else run_dir / "checkpoints" / "latest.pt"
    )
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
    if context.enabled:
        if args.device is not None and not str(args.device).startswith("cuda"):
            raise ValueError("torchrun evaluation requires a CUDA --device")
        device = torch.device("cuda", context.local_rank)
    else:
        device = torch.device(
            args.device
            or cfg_get(
                cfg,
                "runtime.device",
                "cuda" if torch.cuda.is_available() else "cpu",
            )
        )
    policy, normalizer, checkpoint_state = load_runtime_checkpoint(
        checkpoint, cfg, match_training=True
    )
    if not bool(getattr(policy, "predict_tactile", False)):
        raise RuntimeError("checkpoint has models.fm.predict_tactile=false")
    if policy.tactile_autoencoder is None:
        raise RuntimeError("checkpoint does not contain the frozen tactile autoencoder")
    if normalizer.tactile is None:
        raise RuntimeError("checkpoint does not contain tactile normalizer state")
    policy = policy.to(device).eval()

    num_inference_steps = int(
        args.num_inference_steps
        if args.num_inference_steps is not None
        else cfg_get(cfg, "models.fm.num_inference_steps", policy.num_inference_steps)
    )
    solver = str(args.solver or cfg_get(cfg, "models.fm.solver", policy.solver))
    plot_horizons = _parse_int_list(args.plot_horizons)

    mode_metrics: dict[str, Any] = {}
    for mode in modes:
        mode_dir = output_dir / mode
        if context.is_main:
            mode_dir.mkdir(parents=True, exist_ok=True)
        if context.enabled:
            dist.barrier()
        evaluation = evaluate_mode(
            mode=mode,
            windows=rank_windows,
            source=source,
            policy=policy,
            normalizer=normalizer,
            device=device,
            batch_size=int(args.batch_size),
            decode_frame_batch_size=int(args.decode_frame_batch_size),
            num_inference_steps=num_inference_steps,
            solver=solver,
            amp=str(args.amp),
            seed=int(args.seed),
            output_dir=mode_dir,
            plot_samples=int(args.plot_samples),
            plot_horizons=plot_horizons,
            save_arrays=bool(args.save_arrays),
            context=context,
        )
        metrics = reduce_and_summarize_mode(
            mode=mode,
            evaluation=evaluation,
            context=context,
            device=device,
            output_dir=mode_dir,
        )
        if context.is_main:
            assert metrics is not None
            mode_metrics[mode] = metrics

    if not context.is_main:
        return

    payload = {
        "format": "stage2_tactile_reconstruction_eval/v2",
        "selection": selection,
        "checkpoint": str(checkpoint),
        "checkpoint_epoch": int(checkpoint_state.get("epoch", -1)),
        "checkpoint_global_step": int(checkpoint_state.get("global_step", -1)),
        "device": str(device),
        "distributed": {
            "enabled": context.enabled,
            "world_size": context.world_size,
            "backend": context.backend,
            "partition": "global_window_list[rank::world_size]",
        },
        "amp": str(args.amp),
        "num_inference_steps": num_inference_steps,
        "solver": solver,
        "metrics": mode_metrics,
    }
    metrics_path = output_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(f"[complete] metrics={metrics_path}")
    for mode, metrics in mode_metrics.items():
        stage2 = metrics["stage2_physical"]
        oracle = metrics["ae_oracle_physical"]
        print(
            f"[{mode}] stage2_mae={stage2['mae']:.8g} "
            f"stage2_mse={stage2['mse']:.8g} "
            f"ae_oracle_mae={oracle['mae']:.8g} "
            f"ae_oracle_mse={oracle['mse']:.8g}"
        )


if __name__ == "__main__":
    main()
