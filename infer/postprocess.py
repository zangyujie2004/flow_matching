from __future__ import annotations

from typing import Any

import numpy as np

from tools.action import rot6d_to_matrix


def rotation_matrix_to_rpy(rot: np.ndarray) -> np.ndarray:
    """Extract roll, pitch, yaw from a rotation matrix (intrinsic ZYX / extrinsic XYZ)."""
    rot = np.asarray(rot, dtype=np.float64).reshape(3, 3)
    sy = float(np.hypot(rot[0, 0], rot[1, 0]))
    if sy > 1e-6:
        roll = float(np.arctan2(rot[2, 1], rot[2, 2]))
        pitch = float(np.arctan2(-rot[2, 0], sy))
        yaw = float(np.arctan2(rot[1, 0], rot[0, 0]))
    else:
        roll = float(np.arctan2(-rot[1, 2], rot[1, 1]))
        pitch = float(np.arctan2(-rot[2, 0], sy))
        yaw = 0.0
    return np.array([roll, pitch, yaw], dtype=np.float32)


def rpy_to_quaternion_wxyz(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Match prometheus.utils.robot.rpy_to_quaternion_wxyz (for roundtrip tests)."""
    cy = np.cos(float(yaw) * 0.5)
    sy = np.sin(float(yaw) * 0.5)
    cp = np.cos(float(pitch) * 0.5)
    sp = np.sin(float(pitch) * 0.5)
    cr = np.cos(float(roll) * 0.5)
    sr = np.sin(float(roll) * 0.5)
    quat = np.array(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ],
        dtype=np.float32,
    )
    norm = float(np.linalg.norm(quat))
    if norm <= 1e-8:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    return (quat / norm).astype(np.float32)


def quat_wxyz_to_rotation_matrix(quat: np.ndarray) -> np.ndarray:
    w, x, y, z = np.asarray(quat, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm([w, x, y, z]))
    if norm <= 0.0:
        raise ValueError("zero quaternion")
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def rot6d_to_rpy(rot6d: Any) -> np.ndarray:
    arr = np.asarray(rot6d, dtype=np.float32)
    single = arr.ndim == 1
    flat = arr.reshape(-1, 6)
    mats = rot6d_to_matrix(flat)
    rpy = np.stack([rotation_matrix_to_rpy(mats[i]) for i in range(mats.shape[0])], axis=0)
    if single:
        return rpy[0].astype(np.float32, copy=False)
    return rpy.reshape(arr.shape[:-1] + (3,)).astype(np.float32, copy=False)


def eef_rot6d_abs_to_rpy_abs(arm20: Any) -> np.ndarray:
    arm20 = np.asarray(arm20, dtype=np.float32).reshape(20)
    left = arm20[:10]
    right = arm20[10:]
    left7 = np.concatenate([left[:3], rot6d_to_rpy(left[3:9]), left[9:10]], axis=0)
    right7 = np.concatenate([right[:3], rot6d_to_rpy(right[3:9]), right[9:10]], axis=0)
    return np.concatenate([left7, right7], axis=0).astype(np.float32, copy=False)


def apply_action_process(traj: Any, process: str) -> np.ndarray:
    traj = np.asarray(traj, dtype=np.float32)
    process = str(process)
    if process == "abs_eef":
        if traj.ndim == 1:
            return eef_rot6d_abs_to_rpy_abs(traj)
        if traj.ndim == 2 and traj.shape[-1] == 20:
            return np.stack([eef_rot6d_abs_to_rpy_abs(row) for row in traj], axis=0)
        raise ValueError(f"abs_eef expects (20,) or (T,20), got shape {traj.shape}")
    if process == "abs_qpos":
        if traj.ndim == 1:
            if traj.shape[0] != 14:
                raise ValueError(f"abs_qpos expects (14,), got shape {traj.shape}")
            return traj.astype(np.float32, copy=False)
        if traj.ndim == 2 and traj.shape[-1] == 14:
            return traj.astype(np.float32, copy=False)
        raise ValueError(f"abs_qpos expects (14,) or (T,14), got shape {traj.shape}")
    raise ValueError(f"unsupported deploy.action_process={process!r}")
