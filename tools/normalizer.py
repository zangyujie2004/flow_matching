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
        x_min = x.min(axis=0)
        x_max = x.max(axis=0)
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
    output_min, output_max = output_range
    scales: List[np.ndarray] = []
    offsets: List[np.ndarray] = []
    for arm_offset in (0, 10):
        for start, end, mode in _EEF_ARM_SEGMENTS:
            sl = slice(arm_offset + start, arm_offset + end)
            if mode == "limits":
                field = FieldNormalizer.from_data_limits(
                    data[..., sl], output_min=output_min, output_max=output_max
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


def _fit_tactile_normalizer(
    tactile: np.ndarray,
    *,
    output_range: Tuple[float, float],
) -> FieldNormalizer:
    """Fit on deformation maps (..., H, W, 12)."""
    deformation = extract_tactile_deformation(tactile)
    return FieldNormalizer.from_data_limits(
        deformation, output_min=output_range[0], output_max=output_range[1]
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
    ) -> "DatasetNormalizer":
        window_indices = list(range(len(dataset.windows)))
        if max_windows is not None and len(window_indices) > max_windows:
            step = max(1, len(window_indices) // max_windows)
            window_indices = window_indices[::step][:max_windows]

        state_chunks: List[np.ndarray] = []
        action_chunks: List[np.ndarray] = []
        for idx in window_indices:
            s0, s1 = dataset.state_range(idx)
            a0, a1 = dataset.action_range(idx)
            state_raw = dataset.get_state(s0, s1)
            action_raw = dataset.get_action(a0, a1)
            state_chunks.append(state_raw)
            action_chunks.append(
                transform_robot_action(
                    action_raw,
                    state_raw,
                    action_type=dataset.action_type,
                    action_representation=dataset.action_representation,
                )
            )

        state_data = np.concatenate(state_chunks, axis=0)
        action_data = np.concatenate(action_chunks, axis=0)

        tactile_norm = None
        if dataset.use_tactile:
            tactile_norm = _fit_tactile_normalizer(
                dataset.ram_data[dataset.tactile_key],
                output_range=output_range,
            )

        print(
            "[DatasetNormalizer] fit complete: "
            f"state={state_data.shape}, action={action_data.shape}, "
            f"action_type={dataset.action_type}, repr={dataset.action_representation}"
        )
        return cls(
            state=_fit_robot_field(state_data, action_type=dataset.action_type, output_range=output_range),
            action=_fit_robot_field(action_data, action_type=dataset.action_type, output_range=output_range),
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
