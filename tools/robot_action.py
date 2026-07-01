from __future__ import annotations

import numpy as np

from .action import rot6d_to_matrix


def matrix_to_rot6d(rot_mat: np.ndarray) -> np.ndarray:
    rot_mat = np.asarray(rot_mat, dtype=np.float32)
    if rot_mat.ndim == 2:
        rot_mat = rot_mat[None, ...]
    return np.concatenate([rot_mat[:, :, 0], rot_mat[:, :, 1]], axis=-1)


def _transform_arm_eef_absolute_to_relative(actions: np.ndarray, base: np.ndarray) -> np.ndarray:
    """Per-arm layout: xyz(3) + rot6d(6) + gripper(1)."""
    actions = np.asarray(actions, dtype=np.float32).copy()
    base = np.asarray(base, dtype=np.float32)

    base_rot = rot6d_to_matrix(base[3:9][None])[0]
    base_mat = np.eye(4, dtype=np.float64)
    base_mat[:3, :3] = base_rot
    base_mat[:3, 3] = base[:3]
    inv_base = np.linalg.inv(base_mat)

    for t in range(actions.shape[0]):
        rot = rot6d_to_matrix(actions[t, 3:9][None])[0]
        mat = np.eye(4, dtype=np.float64)
        mat[:3, :3] = rot
        mat[:3, 3] = actions[t, :3]
        rel = inv_base @ mat
        actions[t, :3] = rel[:3, 3].astype(np.float32)
        actions[t, 3:9] = matrix_to_rot6d(rel[:3, :3]).astype(np.float32)
        actions[t, 9] = actions[t, 9] - base[9]
    return actions


def transform_eef_absolute_to_relative(actions: np.ndarray, anchor_state: np.ndarray) -> np.ndarray:
    actions = np.asarray(actions, dtype=np.float32).copy()
    anchor_state = np.asarray(anchor_state, dtype=np.float32)
    if actions.shape[-1] != 20 or anchor_state.shape[-1] != 20:
        raise ValueError(
            f"eef relative expects dim 20, got action={actions.shape[-1]}, anchor={anchor_state.shape[-1]}"
        )
    actions[:, :10] = _transform_arm_eef_absolute_to_relative(actions[:, :10], anchor_state[:10])
    actions[:, 10:] = _transform_arm_eef_absolute_to_relative(actions[:, 10:], anchor_state[10:])
    return actions


def _transform_arm_eef_relative_to_absolute(actions: np.ndarray, base: np.ndarray) -> np.ndarray:
    """Inverse of _transform_arm_eef_absolute_to_relative."""
    actions = np.asarray(actions, dtype=np.float32).copy()
    base = np.asarray(base, dtype=np.float32)

    base_rot = rot6d_to_matrix(base[3:9][None])[0]
    base_mat = np.eye(4, dtype=np.float64)
    base_mat[:3, :3] = base_rot
    base_mat[:3, 3] = base[:3]

    for t in range(actions.shape[0]):
        rel_rot = rot6d_to_matrix(actions[t, 3:9][None])[0]
        rel_mat = np.eye(4, dtype=np.float64)
        rel_mat[:3, :3] = rel_rot
        rel_mat[:3, 3] = actions[t, :3]
        abs_mat = base_mat @ rel_mat
        actions[t, :3] = abs_mat[:3, 3].astype(np.float32)
        actions[t, 3:9] = matrix_to_rot6d(abs_mat[:3, :3][None]).astype(np.float32)[0]
        actions[t, 9] = actions[t, 9] + base[9]
    return actions


def transform_eef_relative_to_absolute(actions: np.ndarray, anchor_state: np.ndarray) -> np.ndarray:
    actions = np.asarray(actions, dtype=np.float32).copy()
    anchor_state = np.asarray(anchor_state, dtype=np.float32)
    if actions.shape[-1] != 20 or anchor_state.shape[-1] != 20:
        raise ValueError(
            f"eef absolute expects dim 20, got action={actions.shape[-1]}, anchor={anchor_state.shape[-1]}"
        )
    actions[:, :10] = _transform_arm_eef_relative_to_absolute(actions[:, :10], anchor_state[:10])
    actions[:, 10:] = _transform_arm_eef_relative_to_absolute(actions[:, 10:], anchor_state[10:])
    return actions


def transform_robot_action_to_absolute(
    action: np.ndarray,
    state_history: np.ndarray,
    *,
    action_type: str,
    action_representation: str,
) -> np.ndarray:
    action = np.asarray(action, dtype=np.float32)
    if action_representation == "absolute":
        return action

    anchor = np.asarray(state_history[-1], dtype=np.float32)
    if action_type == "joint":
        return action + anchor[None, :]
    if action_type == "eef":
        return transform_eef_relative_to_absolute(action, anchor)
    raise ValueError(f"unsupported action_type={action_type}")


def transform_robot_action(
    action: np.ndarray,
    state_history: np.ndarray,
    *,
    action_type: str,
    action_representation: str,
) -> np.ndarray:
    action = np.asarray(action, dtype=np.float32)
    if action_representation == "absolute":
        return action

    anchor = np.asarray(state_history[-1], dtype=np.float32)
    if action_type == "joint":
        return action - anchor[None, :]
    if action_type == "eef":
        return transform_eef_absolute_to_relative(action, anchor)
    raise ValueError(f"unsupported action_type={action_type}")
