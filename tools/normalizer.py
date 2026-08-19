from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from .robot_action import transform_robot_action, transform_robot_action_to_absolute
from .tactile_feat import extract_tactile_deformation

_EEF_ARM_SEGMENTS: Tuple[Tuple[int, int, str], ...] = (
    (0, 3, "limits"),
    (3, 9, "identity"),
    (9, 10, "limits"),
)


@dataclass
class FieldNormalizer:
    scale: torch.Tensor
    offset: torch.Tensor
    _scale_np: np.ndarray | None = field(default=None, repr=False, compare=False)
    _offset_np: np.ndarray | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        self._refresh_numpy_cache()

    def _refresh_numpy_cache(self) -> None:
        self._scale_np = self.scale.detach().cpu().numpy().astype(np.float32, copy=False)
        self._offset_np = self.offset.detach().cpu().numpy().astype(np.float32, copy=False)

    @classmethod
    def identity(cls, dim: int) -> "FieldNormalizer":
        return cls(
            scale=torch.ones(dim, dtype=torch.float32),
            offset=torch.zeros(dim, dtype=torch.float32),
        )

    @classmethod
    def from_data_limits(
        cls,
        data: np.ndarray,
        output_min: float = -1.0,
        output_max: float = 1.0,
        eps: float = 1e-7,
    ) -> "FieldNormalizer":
        x = np.asarray(data, dtype=np.float32).reshape(-1, data.shape[-1])
        return cls.from_limits(
            x.min(axis=0),
            x.max(axis=0),
            output_min=output_min,
            output_max=output_max,
            eps=eps,
        )

    @classmethod
    def from_limits(
        cls,
        x_min: np.ndarray,
        x_max: np.ndarray,
        output_min: float = -1.0,
        output_max: float = 1.0,
        eps: float = 1e-7,
    ) -> "FieldNormalizer":
        x_min = np.asarray(x_min, dtype=np.float32)
        x_max = np.asarray(x_max, dtype=np.float32)
        if x_min.ndim != 1 or x_max.shape != x_min.shape:
            raise ValueError(
                f"normalizer limits must be matching 1-D arrays, got {x_min.shape}, {x_max.shape}"
            )
        x_range = np.maximum(x_max - x_min, eps)
        scale = (output_max - output_min) / x_range
        offset = output_min - scale * x_min
        return cls(
            scale=torch.from_numpy(scale.astype(np.float32)),
            offset=torch.from_numpy(offset.astype(np.float32)),
        )

    def normalize_np(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        return x * self._scale_np + self._offset_np

    def unnormalize_np(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        return (x - self._offset_np) / self._scale_np

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.scale.to(device=x.device, dtype=x.dtype)
        offset = self.offset.to(device=x.device, dtype=x.dtype)
        return x * scale + offset

    def unnormalize(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.scale.to(device=x.device, dtype=x.dtype)
        offset = self.offset.to(device=x.device, dtype=x.dtype)
        return (x - offset) / scale

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return {"scale": self.scale.detach().cpu(), "offset": self.offset.detach().cpu()}

    @classmethod
    def from_state_dict(cls, state: Dict[str, torch.Tensor]) -> "FieldNormalizer":
        return cls(scale=state["scale"], offset=state["offset"])


def _fit_eef_segmented(data: np.ndarray, output_range: Tuple[float, float]) -> FieldNormalizer:
    flat = np.asarray(data, dtype=np.float32).reshape(-1, data.shape[-1])
    return _fit_eef_segmented_from_limits(
        flat.min(axis=0),
        flat.max(axis=0),
        output_range,
    )


def _fit_eef_segmented_from_limits(
    data_min: np.ndarray,
    data_max: np.ndarray,
    output_range: Tuple[float, float],
) -> FieldNormalizer:
    output_min, output_max = output_range
    scales: List[np.ndarray] = []
    offsets: List[np.ndarray] = []
    for arm_offset in (0, 10):
        for start, end, mode in _EEF_ARM_SEGMENTS:
            sl = slice(arm_offset + start, arm_offset + end)
            if mode == "limits":
                field = FieldNormalizer.from_limits(
                    data_min[sl],
                    data_max[sl],
                    output_min=output_min,
                    output_max=output_max,
                )
            else:
                dim = end - start
                field = FieldNormalizer.identity(dim)
            scales.append(field.scale.numpy())
            offsets.append(field.offset.numpy())
    return FieldNormalizer(
        scale=torch.from_numpy(np.concatenate(scales).astype(np.float32)),
        offset=torch.from_numpy(np.concatenate(offsets).astype(np.float32)),
    )


def _fit_robot_field(
    data: np.ndarray,
    *,
    action_type: str,
    output_range: Tuple[float, float],
) -> FieldNormalizer:
    if action_type == "joint":
        return FieldNormalizer.from_data_limits(
            data, output_min=output_range[0], output_max=output_range[1]
        )
    if action_type == "eef":
        return _fit_eef_segmented(data, output_range)
    raise ValueError(f"unsupported action_type={action_type}")


def _fit_robot_field_from_limits(
    data_min: np.ndarray,
    data_max: np.ndarray,
    *,
    action_type: str,
    output_range: Tuple[float, float],
) -> FieldNormalizer:
    if action_type == "joint":
        return FieldNormalizer.from_limits(
            data_min,
            data_max,
            output_min=output_range[0],
            output_max=output_range[1],
        )
    if action_type == "eef":
        return _fit_eef_segmented_from_limits(data_min, data_max, output_range)
    raise ValueError(f"unsupported action_type={action_type}")


def _fit_tactile_normalizer(
    tactile: Any,
    *,
    output_range: Tuple[float, float],
    batch_frames: int = 4096,
) -> FieldNormalizer:
    """Fit deformation limits in frame chunks to avoid a second full tactile copy."""
    if tactile.shape[0] == 0:
        raise ValueError("cannot fit tactile normalizer on an empty array")
    batch_frames = max(1, int(batch_frames))
    data_min: np.ndarray | None = None
    data_max: np.ndarray | None = None
    stream_from_zarr = not isinstance(tactile, np.ndarray)
    total_batches = (int(tactile.shape[0]) + batch_frames - 1) // batch_frames
    if stream_from_zarr:
        print(
            "[DatasetNormalizer] streaming raw tactile for limits: "
            f"frames={int(tactile.shape[0])}, batch_frames={batch_frames}, "
            f"batches={total_batches}"
        )
    for start in range(0, tactile.shape[0], batch_frames):
        chunk = np.asarray(tactile[start : start + batch_frames])
        deformation = (
            np.asarray(chunk, dtype=np.float32)
            if chunk.shape[-1] == 12
            else extract_tactile_deformation(chunk)
        )
        flat = deformation.reshape(-1, deformation.shape[-1])
        chunk_min = flat.min(axis=0)
        chunk_max = flat.max(axis=0)
        data_min = chunk_min if data_min is None else np.minimum(data_min, chunk_min)
        data_max = chunk_max if data_max is None else np.maximum(data_max, chunk_max)
        completed = start // batch_frames + 1
        if stream_from_zarr and (completed % 32 == 0 or completed == total_batches):
            print(
                "[DatasetNormalizer] tactile limits: "
                f"{completed}/{total_batches} batches"
            )
    assert data_min is not None and data_max is not None
    return FieldNormalizer.from_limits(
        data_min,
        data_max,
        output_min=output_range[0],
        output_max=output_range[1],
    )


class DatasetNormalizer:
    def __init__(
        self,
        *,
        state: FieldNormalizer,
        action: FieldNormalizer,
        tactile: FieldNormalizer | None,
        action_type: str,
        action_representation: str,
    ) -> None:
        self.state = state
        self.action = action
        self.tactile = tactile
        self.action_type = action_type
        self.action_representation = action_representation

    @classmethod
    def build(
        cls,
        dataset: Any,
        *,
        output_range: Tuple[float, float] = (-1.0, 1.0),
        max_windows: int | None = None,
        batch_windows: int = 1024,
    ) -> "DatasetNormalizer":
        window_indices = np.arange(len(dataset.windows), dtype=np.int64)
        if max_windows is not None and len(window_indices) > max_windows:
            step = max(1, len(window_indices) // max_windows)
            window_indices = window_indices[::step][:max_windows]
        if len(window_indices) == 0:
            raise ValueError("cannot fit normalizer without windows")

        batch_windows = max(1, int(batch_windows))
        state_min: np.ndarray | None = None
        state_max: np.ndarray | None = None
        action_min: np.ndarray | None = None
        action_max: np.ndarray | None = None

        for start in range(0, len(window_indices), batch_windows):
            batch_indices = window_indices[start : start + batch_windows]
            if hasattr(dataset, "get_state_action_batch"):
                state_raw, action_raw = dataset.get_state_action_batch(batch_indices)
            else:
                state_raw = np.stack(
                    [dataset.get_state(*dataset.state_range(int(idx))) for idx in batch_indices]
                )
                action_raw = np.stack(
                    [dataset.get_action(*dataset.action_range(int(idx))) for idx in batch_indices]
                )
            transformed_action = transform_robot_action(
                action_raw,
                state_raw,
                action_type=dataset.action_type,
                action_representation=dataset.action_representation,
            )

            state_flat = np.asarray(state_raw, dtype=np.float32).reshape(-1, state_raw.shape[-1])
            action_flat = np.asarray(transformed_action, dtype=np.float32).reshape(
                -1, transformed_action.shape[-1]
            )
            chunk_state_min = state_flat.min(axis=0)
            chunk_state_max = state_flat.max(axis=0)
            chunk_action_min = action_flat.min(axis=0)
            chunk_action_max = action_flat.max(axis=0)
            state_min = (
                chunk_state_min
                if state_min is None
                else np.minimum(state_min, chunk_state_min)
            )
            state_max = (
                chunk_state_max
                if state_max is None
                else np.maximum(state_max, chunk_state_max)
            )
            action_min = (
                chunk_action_min
                if action_min is None
                else np.minimum(action_min, chunk_action_min)
            )
            action_max = (
                chunk_action_max
                if action_max is None
                else np.maximum(action_max, chunk_action_max)
            )

        assert state_min is not None and state_max is not None
        assert action_min is not None and action_max is not None

        tactile_norm = None
        tactile_override = getattr(dataset, "tactile_normalizer_override", None)
        if dataset.use_tactile and tactile_override is not None:
            tactile_norm = tactile_override
            print(
                "[DatasetNormalizer] using tactile normalization from "
                "the Stage 1 latent cache"
            )
        elif dataset.use_tactile:
            tactile_source = getattr(dataset, "cached_tactile_deformation", None)
            if tactile_source is None:
                tactile_source = dataset.ram_data.get(dataset.tactile_key)
            if tactile_source is None:
                tactile_source = dataset.data_group[dataset.tactile_key]
                print(
                    "[DatasetNormalizer] fitting tactile limits from lazy "
                    "raw-Zarr chunks"
                )
            tactile_norm = _fit_tactile_normalizer(
                tactile_source,
                output_range=output_range,
            )

        print(
            "[DatasetNormalizer] fit complete: "
            f"windows={len(window_indices)}, batch_windows={batch_windows}, "
            f"action_type={dataset.action_type}, repr={dataset.action_representation}"
        )
        return cls(
            state=_fit_robot_field_from_limits(
                state_min,
                state_max,
                action_type=dataset.action_type,
                output_range=output_range,
            ),
            action=_fit_robot_field_from_limits(
                action_min,
                action_max,
                action_type=dataset.action_type,
                output_range=output_range,
            ),
            tactile=tactile_norm,
            action_type=dataset.action_type,
            action_representation=dataset.action_representation,
        )

    def transform_action_np(self, action: np.ndarray, state_history: np.ndarray) -> np.ndarray:
        return transform_robot_action(
            action,
            state_history,
            action_type=self.action_type,
            action_representation=self.action_representation,
        )

    def normalize_state_np(self, x: np.ndarray) -> np.ndarray:
        return self.state.normalize_np(x)

    def normalize_action_np(self, action: np.ndarray, state_history: np.ndarray) -> np.ndarray:
        transformed = self.transform_action_np(action, state_history)
        return self.action.normalize_np(transformed)

    def unnormalize_action_np(self, action: np.ndarray, state_history: np.ndarray) -> np.ndarray:
        relative = self.action.unnormalize_np(np.asarray(action, dtype=np.float32))
        return transform_robot_action_to_absolute(
            relative,
            state_history,
            action_type=self.action_type,
            action_representation=self.action_representation,
        )

    def normalize_tactile_np(self, x: np.ndarray) -> np.ndarray:
        if self.tactile is None:
            raise RuntimeError("tactile normalizer is not configured")
        return self.tactile.normalize_np(np.asarray(x, dtype=np.float32))

    def state_dict(self) -> Dict[str, Any]:
        return {
            "action_type": self.action_type,
            "action_representation": self.action_representation,
            "state": self.state.state_dict(),
            "action": self.action.state_dict(),
            "tactile": None if self.tactile is None else self.tactile.state_dict(),
        }

    @classmethod
    def load_state_dict(cls, state: Dict[str, Any]) -> "DatasetNormalizer":
        tactile_state = state.get("tactile")
        return cls(
            state=FieldNormalizer.from_state_dict(state["state"]),
            action=FieldNormalizer.from_state_dict(state["action"]),
            tactile=None if tactile_state is None else FieldNormalizer.from_state_dict(tactile_state),
            action_type=str(state["action_type"]),
            action_representation=str(state["action_representation"]),
        )
