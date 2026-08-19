from __future__ import annotations

import numpy as np

from .action import rot6d_to_matrix


def matrix_to_rot6d(rot_mat: np.ndarray) -> np.ndarray:
    rot_mat = np.asarray(rot_mat, dtype=np.float32)
    single = rot_mat.ndim == 2
    if single:
        rot_mat = rot_mat[None, ...]
    if rot_mat.shape[-2:] != (3, 3):
        raise ValueError(f"rotation matrix must end with (3, 3), got {rot_mat.shape}")
    return np.concatenate([rot_mat[..., :, 0], rot_mat[..., :, 1]], axis=-1)


def _transform_arm_eef_absolute_to_relative(actions: np.ndarray, base: np.ndarray) -> np.ndarray:
    """Vectorized per-arm transform for (T,10) or (B,T,10) actions."""
    actions = np.asarray(actions, dtype=np.float32).copy()
    base = np.asarray(base, dtype=np.float32)
    squeeze_batch = actions.ndim == 2
    if squeeze_batch:
        actions = actions[None, ...]
    if actions.ndim != 3 or actions.shape[-1] != 10:
        raise ValueError(f"arm EEF actions must be (T,10) or (B,T,10), got {actions.shape}")
    if base.ndim == 1:
        base = base[None, ...]
    if base.ndim != 2 or base.shape != (actions.shape[0], 10):
        raise ValueError(
            f"arm EEF base must be (10,) or (B,10), got {base.shape} for actions {actions.shape}"
        )

    batch_size, horizon = actions.shape[:2]
    base_mat = np.broadcast_to(
        np.eye(4, dtype=np.float64), (batch_size, 4, 4)
    ).copy()
    base_mat[:, :3, :3] = rot6d_to_matrix(base[:, 3:9])
    base_mat[:, :3, 3] = base[:, :3]

    action_mat = np.broadcast_to(
        np.eye(4, dtype=np.float64), (batch_size, horizon, 4, 4)
    ).copy()
    action_rot = rot6d_to_matrix(actions[..., 3:9].reshape(-1, 6))
    action_mat[..., :3, :3] = action_rot.reshape(batch_size, horizon, 3, 3)
    action_mat[..., :3, 3] = actions[..., :3]

    relative = np.linalg.inv(base_mat)[:, None, :, :] @ action_mat
    actions[..., :3] = relative[..., :3, 3].astype(np.float32)
    actions[..., 3:9] = matrix_to_rot6d(relative[..., :3, :3]).astype(np.float32)
    actions[..., 9] -= base[:, None, 9]
    return actions[0] if squeeze_batch else actions


def transform_eef_absolute_to_relative(actions: np.ndarray, anchor_state: np.ndarray) -> np.ndarray:
    actions = np.asarray(actions, dtype=np.float32).copy()
    anchor_state = np.asarray(anchor_state, dtype=np.float32)
    if actions.shape[-1] != 20 or anchor_state.shape[-1] != 20:
        raise ValueError(
            f"eef relative expects dim 20, got action={actions.shape[-1]}, anchor={anchor_state.shape[-1]}"
        )
    actions[..., :10] = _transform_arm_eef_absolute_to_relative(
        actions[..., :10], anchor_state[..., :10]
    )
    actions[..., 10:] = _transform_arm_eef_absolute_to_relative(
        actions[..., 10:], anchor_state[..., 10:]
    )
    return actions


def _transform_arm_eef_relative_to_absolute(actions: np.ndarray, base: np.ndarray) -> np.ndarray:
    """Vectorized inverse for (T,10) or (B,T,10) actions."""
    actions = np.asarray(actions, dtype=np.float32).copy()
    base = np.asarray(base, dtype=np.float32)
    squeeze_batch = actions.ndim == 2
    if squeeze_batch:
        actions = actions[None, ...]
    if actions.ndim != 3 or actions.shape[-1] != 10:
        raise ValueError(f"arm EEF actions must be (T,10) or (B,T,10), got {actions.shape}")
    if base.ndim == 1:
        base = base[None, ...]
    if base.ndim != 2 or base.shape != (actions.shape[0], 10):
        raise ValueError(
            f"arm EEF base must be (10,) or (B,10), got {base.shape} for actions {actions.shape}"
        )

    batch_size, horizon = actions.shape[:2]
    base_mat = np.broadcast_to(
        np.eye(4, dtype=np.float64), (batch_size, 4, 4)
    ).copy()
    base_mat[:, :3, :3] = rot6d_to_matrix(base[:, 3:9])
    base_mat[:, :3, 3] = base[:, :3]

    relative_mat = np.broadcast_to(
        np.eye(4, dtype=np.float64), (batch_size, horizon, 4, 4)
    ).copy()
    relative_rot = rot6d_to_matrix(actions[..., 3:9].reshape(-1, 6))
    relative_mat[..., :3, :3] = relative_rot.reshape(batch_size, horizon, 3, 3)
    relative_mat[..., :3, 3] = actions[..., :3]

    absolute = base_mat[:, None, :, :] @ relative_mat
    actions[..., :3] = absolute[..., :3, 3].astype(np.float32)
    actions[..., 3:9] = matrix_to_rot6d(absolute[..., :3, :3]).astype(np.float32)
    actions[..., 9] += base[:, None, 9]
    return actions[0] if squeeze_batch else actions


def transform_eef_relative_to_absolute(actions: np.ndarray, anchor_state: np.ndarray) -> np.ndarray:
    actions = np.asarray(actions, dtype=np.float32).copy()
    anchor_state = np.asarray(anchor_state, dtype=np.float32)
    if actions.shape[-1] != 20 or anchor_state.shape[-1] != 20:
        raise ValueError(
            f"eef absolute expects dim 20, got action={actions.shape[-1]}, anchor={anchor_state.shape[-1]}"
        )
    actions[..., :10] = _transform_arm_eef_relative_to_absolute(
        actions[..., :10], anchor_state[..., :10]
    )
    actions[..., 10:] = _transform_arm_eef_relative_to_absolute(
        actions[..., 10:], anchor_state[..., 10:]
    )
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

    state_history = np.asarray(state_history, dtype=np.float32)
    anchor = state_history[-1] if state_history.ndim == 2 else state_history[..., -1, :]
    if action_type == "joint":
        return action + anchor[..., None, :]
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

    state_history = np.asarray(state_history, dtype=np.float32)
    anchor = state_history[-1] if state_history.ndim == 2 else state_history[..., -1, :]
    if action_type == "joint":
        return action - anchor[..., None, :]
    if action_type == "eef":
        return transform_eef_absolute_to_relative(action, anchor)
    raise ValueError(f"unsupported action_type={action_type}")
