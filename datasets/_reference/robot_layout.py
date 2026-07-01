"""Robot vector layout: action_type -> slice / dim. arm fixed both."""

from __future__ import annotations

ACTION_TYPES = ("joint", "eef")

# 62-D merged layout (preprocess); arm always both.
ROBOT_SLICES = {
    "joint": slice(0, 14),
    "eef": slice(14, 34),
}

ROBOT_DIMS = {
    "joint": 14,
    "eef": 20,
}


def resolve_action_type(action_type: str) -> str:
    key = str(action_type).strip().lower()
    if key not in ACTION_TYPES:
        raise ValueError(f"action_type must be one of {ACTION_TYPES}, got {action_type!r}")
    return key


def robot_slice(action_type: str) -> slice:
    return ROBOT_SLICES[resolve_action_type(action_type)]


def robot_dim(action_type: str) -> int:
    return ROBOT_DIMS[resolve_action_type(action_type)]
