import numpy as np


def rot6d_to_matrix(rot6d):
    x = rot6d[:, :3]
    y = rot6d[:, 3:6]
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    y_proj = np.sum(x * y, axis=1, keepdims=True) * x
    y = y - y_proj
    y = y / np.linalg.norm(y, axis=1, keepdims=True)
    z = np.cross(x, y)
    return np.stack([x, y, z], axis=-1)


def normalize_vector(v: np.ndarray) -> np.ndarray:
    v_mag = np.linalg.norm(v, axis=1, keepdims=True)
    v_mag = np.maximum(v_mag, 1e-8)
    return v / v_mag


def ortho6d_to_rotation_matrix(ortho6d: np.ndarray) -> np.ndarray:
    x_raw = ortho6d[:, 0:3]
    y_raw = ortho6d[:, 3:6]
    x = normalize_vector(x_raw)
    z = np.cross(x, y_raw)
    z = normalize_vector(z)
    y = np.cross(z, x)

    x = x[:, :, np.newaxis]
    y = y[:, :, np.newaxis]
    z = z[:, :, np.newaxis]
    return np.concatenate((x, y, z), axis=2)


def pose_3d_9d_to_homo_matrix_batch(pose: np.ndarray) -> np.ndarray:
    assert pose.shape[1] in [3, 9], "pose should be (N, 3) or (N, 9)"
    mat = np.eye(4)[None, :, :].repeat(pose.shape[0], axis=0)
    mat[:, :3, 3] = pose[:, :3]
    if pose.shape[1] == 9:
        mat[:, :3, :3] = ortho6d_to_rotation_matrix(pose[:, 3:9])
    return mat


def homo_matrix_to_pose_9d_batch(mat: np.ndarray) -> np.ndarray:
    assert mat.shape[1:] == (4, 4), "mat should be (N, 4, 4)"
    pose = np.zeros((mat.shape[0], 9))
    pose[:, :3] = mat[:, :3, 3]
    pose[:, 3:9] = mat[:, :3, :2].swapaxes(1, 2).reshape(mat.shape[0], -1)
    return pose


def absolute_actions_to_relative_actions(actions: np.ndarray, base_absolute_action=None):
    actions = actions.copy()
    _, d = actions.shape

    if d in (3, 4):
        tcp_dim_list = [np.arange(3)]
    elif d in (6, 8):
        tcp_dim_list = [np.arange(3), np.arange(3, 6)]
    elif d in (9, 10):
        tcp_dim_list = [np.arange(9)]
    elif d in (18, 20):
        tcp_dim_list = [np.arange(9), np.arange(9, 18)]
    else:
        raise NotImplementedError

    if base_absolute_action is None:
        base_absolute_action = actions[0].copy()
    for tcp_dim in tcp_dim_list:
        base_tcp_pose_mat = pose_3d_9d_to_homo_matrix_batch(base_absolute_action[None, tcp_dim])
        actions[:, tcp_dim] = homo_matrix_to_pose_9d_batch(
            np.linalg.inv(base_tcp_pose_mat) @ pose_3d_9d_to_homo_matrix_batch(actions[:, tcp_dim])
        )[:, : len(tcp_dim)]
    return actions


def relative_actions_to_absolute_actions(actions: np.ndarray, base_absolute_action: np.ndarray):
    actions = actions.copy()
    _, d = actions.shape

    if d in (3, 4):
        tcp_dim_list = [np.arange(3)]
    elif d in (6, 8):
        tcp_dim_list = [np.arange(3), np.arange(3, 6)]
    elif d in (9, 10):
        tcp_dim_list = [np.arange(9)]
    elif d in (18, 20):
        tcp_dim_list = [np.arange(9), np.arange(9, 18)]
    else:
        raise NotImplementedError

    for tcp_dim in tcp_dim_list:
        base_tcp_pose_mat = pose_3d_9d_to_homo_matrix_batch(base_absolute_action[None, tcp_dim])
        actions[:, tcp_dim] = homo_matrix_to_pose_9d_batch(
            base_tcp_pose_mat @ pose_3d_9d_to_homo_matrix_batch(actions[:, tcp_dim])
        )[:, : len(tcp_dim)]
    return actions
