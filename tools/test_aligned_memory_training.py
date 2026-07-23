"""CPU checks for aligned Dataset indices and from_config -> compute_loss."""

import importlib.util
import sys
import types

import numpy as np
import torch

if importlib.util.find_spec("zarr") is None:
    sys.modules["zarr"] = types.ModuleType("zarr")

from datasets.zarr_dataset import ZarrDataset
from models.fm.encoders.dino_v2 import DINOV2_NUM_TOKENS
from models.fm.flow_policy import FlowMatchingPolicy


def check_dataset_alignment() -> None:
    dataset = object.__new__(ZarrDataset)
    dataset.memory_history_frames = 128
    dataset.memory_visual_history_length = 128
    dataset.memory_sample_stride = 8
    dataset.memory_recent_frame = 0
    dataset.memory_visual_recent_frame = 0
    dataset.memory_visual_offsets = dataset._build_memory_visual_offsets()
    dataset.memory_state_offsets = dataset.memory_visual_offsets
    dataset.episode_starts = np.array([100], dtype=np.int64)
    dataset.episode_ends = np.array([1500], dtype=np.int64)
    dataset.n_image_views = 3
    dataset.robot_slice = slice(0, 14)
    dataset.state_key = "state"

    frame_ids = np.arange(1500, dtype=np.float32)
    dataset.cached_frame_image_backbone_feat = np.broadcast_to(
        frame_ids[:, None, None],
        (1500, 3, 384),
    ).copy()
    dataset.ram_data = {
        "state": np.broadcast_to(frame_ids[:, None], (1500, 14)).copy()
    }

    offsets = dataset.memory_visual_offsets
    assert dataset.memory_state_offsets is offsets
    assert offsets.shape == (128,)
    assert offsets[0] == -1016 and offsets[-1] == 0
    assert np.all(np.diff(offsets) == 8)

    anchor = 124
    visual_indices = dataset.memory_visual_indices(anchor, 0)
    state_indices = dataset.memory_state_indices(anchor, 0)
    assert np.array_equal(visual_indices, state_indices)
    assert visual_indices[0] == 100 and visual_indices[-1] == anchor

    visual = dataset.get_memory_camera_latent(anchor, 0)
    state = dataset.get_memory_state(anchor, 0)
    assert visual.shape == (128, 3, 384)
    assert state.shape == (128, 14)
    assert np.array_equal(visual[:, 0, 0], state[:, 0])
    assert dataset.memory_visual_valid(anchor, 0).all()
    assert dataset.memory_state_valid(anchor, 0).all()


def check_from_config_compute_loss() -> None:
    b, t, v, c, state_dim = 1, 128, 3, 384, 14
    cfg = {
        "data": {
            "memory": {
                "enabled": True,
                "history_frames": t,
                "recent_frame": 0,
                "visual_history_length": t,
                "sample_stride": 8,
                "visual_recent_frame": 0,
            }
        },
        "models": {
            "fm": {
                "cond_dim": 256,
                "n_image_views": v,
                "image_pretrained": False,
                "freeze_image_encoder": True,
                "image_feat_dim": 256,
                "view_pool": "global_concat",
                "use_tactile": False,
                "action_horizon": 64,
                "n_action_steps": 32,
                "velocity_model": "unet",
                "down_dims": [32, 64],
                "n_groups": 8,
                "num_inference_steps": 16,
                "solver": "euler",
            },
            "memory": {
                "method": "fusion",
                "injection": "concat_global_cond",
                "dim": 256,
                "visual_layers": 2,
                "visual_heads": 4,
                "state_channels": 32,
                "state_layers": 2,
                "state_mem_dim": 64,
                "dropout": 0.0,
            },
        },
    }
    policy = FlowMatchingPolicy.from_config(
        cfg,
        action_dim=state_dim,
        state_dim=state_dim,
        cond_steps=8,
    ).eval()
    assert policy.memory_history_frames == policy.memory_visual_history_length == t
    offsets = torch.arange(-1016, 1, 8)
    assert offsets.shape == (t,)

    obs = {
        "image_backbone_feat": torch.randn(b, 1, v, DINOV2_NUM_TOKENS, c),
        "state": torch.randn(b, 8, state_dim),
        "memory_image_backbone_feat": torch.randn(b, t, v, c),
        "memory_state": torch.randn(b, t, state_dim),
        "memory_visual_offsets": offsets,
        "memory_visual_valid": torch.ones(b, t, dtype=torch.bool),
        "memory_state_valid": torch.ones(b, t, dtype=torch.bool),
    }
    state_shapes = []
    hook = policy.memory_encoder.state_encoder.register_forward_pre_hook(
        lambda _module, args: state_shapes.append(tuple(args[0].shape))
    )
    result = policy.compute_loss(
        {
            "obs": obs,
            "action": torch.randn(b, 64, state_dim),
        }
    )
    hook.remove()
    assert state_shapes == [(b, t, state_dim)]
    assert result["loss"].ndim == 0 and torch.isfinite(result["loss"])
    assert policy.global_cond_dim == 512


def main() -> None:
    check_dataset_alignment()
    check_from_config_compute_loss()
    print("PASS: Dataset alignment and from_config -> compute_loss")


if __name__ == "__main__":
    main()
