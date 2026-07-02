from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


DEFAULT_JOINT_NAMES: tuple[str, ...] = tuple(
    [f"left_joint_{idx}" for idx in range(1, 7)] + ["left_gripper"]
    + [f"right_joint_{idx}" for idx in range(1, 7)] + ["right_gripper"]
)


@dataclass(frozen=True)
class PreprocessConfig:
    action_type: str = "eef"
    camera_views: tuple[str, ...] = (
        "base_0_color",
        "left_wrist_0_color",
        "right_wrist_0_color",
    )
    gripper_names: dict[str, str] = field(
        default_factory=lambda: {"left": "left_gripper", "right": "right_gripper"}
    )
    joint_names: tuple[str, ...] = DEFAULT_JOINT_NAMES
    image_size: int = 224
    gripper_width_m: float = 0.082

    @property
    def state_dim(self) -> int:
        return 14 if self.action_type == "joint" else 20

    @property
    def state_stream_names(self) -> tuple[str, ...]:
        if self.action_type == "joint":
            return ("robot_state",)
        return ("robot_state", "left_eef", "right_eef")


@dataclass(frozen=True)
class InferenceChunk:
    actions: np.ndarray
    action_space: str
    hz: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        actions = np.asarray(self.actions, dtype=np.float32)
        object.__setattr__(self, "actions", actions)
