"""Forward smoke tests for FlowMatchingPolicy."""

from __future__ import annotations

import os
import sys

import numpy as np
import torch
import yaml
import zarr

_POLICY_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _POLICY_ROOT not in sys.path:
    sys.path.insert(0, _POLICY_ROOT)

from models.fm import build_flow_policy
from datasets.zarr_dataset import ZarrDataset
from tools import precompute_policy_latents as precompute
from tools.latent_cache import (
    CACHE_MODE_CLS_LOCAL_NPY,
    LOCAL_FEATURE_NUM_TOKENS,
    local_feature_file_path,
    normalize_cache_mode,
    stored_token_mode,
)
from utils.train_utils import sync_fm_action_horizon_from_data


def _mock_batch(
    *,
    batch_size: int = 2,
    window_size: int = 8,
    action_horizon: int = 32,
    action_dim: int = 20,
    use_tactile: bool = True,
) -> dict:
    obs = {
        "image": torch.randint(0, 255, (batch_size, 1, 3, 3, 224, 224), dtype=torch.uint8),
        "state": torch.randn(batch_size, window_size, action_dim),
    }
    if use_tactile:
        obs["tactile"] = torch.randn(batch_size, window_size, 35, 20, 12)
    return {
        "obs": obs,
        "action": torch.randn(batch_size, action_horizon, action_dim),
    }


def test_mock_forward_backward() -> None:
    cfg = yaml.safe_load(open(os.path.join(_POLICY_ROOT, "configs", "train", "config.yaml")))
    fm = sync_fm_action_horizon_from_data(cfg["models"]["fm"], cfg["data"])
    fm["image_pretrained"] = False
    window = int(cfg["data"]["window_size"])
    horizon = int(fm["action_horizon"])
    n_views = int(fm.get("n_image_views", 3))

    policy = build_flow_policy(
        {"models": {"fm": fm}},
        action_dim=20,
        state_dim=20,
        cond_steps=window,
    )
    batch = _mock_batch(window_size=window, action_horizon=horizon)
    batch["obs"]["image"] = torch.randint(
        0, 255, (2, 1, n_views, 3, 224, 224), dtype=torch.uint8
    )
    out = policy.compute_loss(batch)
    assert torch.isfinite(out["loss"])
    out["loss"].backward()
    print("mock forward/backward OK, loss=", float(out["loss"]))


def test_predict_action_shape() -> None:
    cfg = yaml.safe_load(open(os.path.join(_POLICY_ROOT, "configs", "train", "config.yaml")))
    fm = sync_fm_action_horizon_from_data(cfg["models"]["fm"], cfg["data"])
    fm["image_pretrained"] = False
    window = int(cfg["data"]["window_size"])
    horizon = int(fm["action_horizon"])
    n_views = int(fm.get("n_image_views", 3))

    policy = build_flow_policy(
        {"models": {"fm": fm}},
        action_dim=20,
        state_dim=20,
        cond_steps=window,
    )
    policy.eval()
    batch = _mock_batch(batch_size=1, window_size=window, action_horizon=horizon)
    batch["obs"]["image"] = torch.randint(
        0, 255, (1, 1, n_views, 3, 224, 224), dtype=torch.uint8
    )
    pred = policy.predict_action(batch["obs"], num_inference_steps=4)
    assert pred["action_normalized"].shape == (1, fm["n_action_steps"], 20)
    assert pred["action_pred_normalized"].shape == (1, fm["action_horizon"], 20)
    print("predict_action shapes OK")


def test_backbone_feat_forward_backward() -> None:
    cfg = yaml.safe_load(open(os.path.join(_POLICY_ROOT, "configs", "train", "config.yaml")))
    fm = sync_fm_action_horizon_from_data(cfg["models"]["fm"], cfg["data"])
    fm["image_pretrained"] = False
    window = int(cfg["data"]["window_size"])
    horizon = int(fm["action_horizon"])
    n_views = int(fm.get("n_image_views", 3))
    n_image_steps = int(cfg["data"].get("n_image_steps", 1))

    policy = build_flow_policy(
        {"models": {"fm": fm}},
        action_dim=20,
        state_dim=20,
        cond_steps=window,
    )
    batch = _mock_batch(batch_size=2, window_size=window, action_horizon=horizon)
    del batch["obs"]["image"]
    backbone_dim = int(policy.condition_encoder.image_encoder.backbone_dim)
    batch["obs"]["image_backbone_feat"] = torch.randn(
        2, n_image_steps, n_views, backbone_dim
    )
    out = policy.compute_loss(batch)
    assert torch.isfinite(out["loss"])
    out["loss"].backward()
    print("backbone_feat forward/backward OK, loss=", float(out["loss"]))


def test_cls_local_npy_mode_keeps_cls_in_zarr() -> None:
    assert normalize_cache_mode("cls_local_npy") == CACHE_MODE_CLS_LOCAL_NPY
    assert normalize_cache_mode("local-npy") == CACHE_MODE_CLS_LOCAL_NPY
    assert stored_token_mode(CACHE_MODE_CLS_LOCAL_NPY) == "cls"


def test_local_feature_rows_load_selected_views(tmp_path) -> None:
    local_root = str(tmp_path / "frame_backbone_local")
    feature_index = np.asarray([5, 8], dtype=np.int64)
    expected_shape = (3, LOCAL_FEATURE_NUM_TOKENS, 4)

    for frame_idx, feature_id in enumerate(feature_index):
        path = local_feature_file_path(
            local_root,
            int(feature_id),
            frames_per_directory=2,
        )
        os.makedirs(os.path.dirname(path), exist_ok=True)
        feature = np.full(expected_shape, frame_idx, dtype=np.float16)
        feature[0] += 10
        feature[2] += 30
        np.save(path, feature, allow_pickle=False)

    loaded = ZarrDataset._load_local_feature_rows(
        np.asarray([1, 0], dtype=np.int64),
        feature_index=feature_index,
        local_root=local_root,
        frames_per_directory=2,
        expected_shape=expected_shape,
        view_indices=(2, 0),
    )

    assert loaded.shape == (2, 2, LOCAL_FEATURE_NUM_TOKENS, 4)
    assert loaded.dtype == np.float32
    assert np.all(loaded[0, 0] == 31)
    assert np.all(loaded[0, 1] == 11)
    assert np.all(loaded[1, 0] == 30)
    assert np.all(loaded[1, 1] == 10)

    dataset = object.__new__(ZarrDataset)
    dataset.cached_frame_image_backbone_feat = np.zeros((2, 2, 4), dtype=np.float32)
    dataset.latent_token_mode = CACHE_MODE_CLS_LOCAL_NPY
    dataset.cached_local_feature_index = feature_index
    dataset.local_feature_root = local_root
    dataset.local_feature_frames_per_directory = 2
    dataset.local_feature_shape = expected_shape
    dataset._local_feature_view_indices = (2, 0)
    patches_only = dataset._gather_frame_latent(
        np.asarray([1], dtype=np.int64),
        include_cls=False,
    )
    combined = dataset._gather_frame_latent(
        np.asarray([1], dtype=np.int64),
        include_cls=True,
    )
    assert patches_only.shape == (1, 2, 256, 4)
    assert combined.shape == (1, 2, 257, 4)


def test_precompute_cls_local_npy_layout(tmp_path, monkeypatch) -> None:
    class FakeDataset:
        zarr_path = str(tmp_path / "replay_buffer.zarr")
        camera_key = "camera"
        state_key = "state"
        camera_views = ("base_0", "left_wrist_0", "right_wrist_0")
        image_size = 224
        ram_data = {
            "camera": np.zeros((2, 1), dtype=np.uint8),
            "state": np.zeros((2, 1), dtype=np.float32),
        }

        def get_camera(self, t0: int, t1: int) -> np.ndarray:
            return np.zeros((t1 - t0, 224, 224, 9), dtype=np.uint8)

        def _process_image(self, camera: np.ndarray) -> torch.Tensor:
            return torch.zeros(camera.shape[0], 3, 3, 224, 224, dtype=torch.uint8)

    class FakeEncoder:
        def extract_backbone_feat(self, flat: torch.Tensor) -> torch.Tensor:
            batch = flat.shape[0]
            image_ids = torch.arange(batch, dtype=torch.float32)[:, None, None]
            token_ids = torch.arange(257, dtype=torch.float32)[None, :, None]
            return (image_ids + token_ids).expand(batch, 257, 4).contiguous()

    monkeypatch.setattr(precompute, "_build_encoder", lambda _cfg, _device: FakeEncoder())
    cache_root = tmp_path / "cache"
    output_path = cache_root / "frame_backbone.zarr"
    cfg = {
        "runtime": {"device": "cpu"},
        "data": {
            "root_dir": str(tmp_path),
            "latent_cache_root_dir": str(cache_root),
        },
        "models": {
            "fm": {
                "image_encoder_name": "dinov2_small",
                "dino_model_name": "vit_small_patch14_dinov2.lvd142m",
                "freeze_image_encoder": True,
            }
        },
        "precompute": {
            "batch_size": 2,
            "device": "cpu",
            "output_path": str(output_path),
            "token_mode": "cls_local_npy",
            "local_feature": {"frames_per_directory": 1},
        },
    }

    precompute.precompute_image_latents(cfg, dataset=FakeDataset())

    root = zarr.open_group(str(output_path), mode="r")
    cls = np.asarray(root["data"]["frame_image_backbone_feat"][:])
    index = np.asarray(root["data"]["local_feature_index"][:])
    assert root.attrs["token_mode"] == "cls"
    assert root.attrs["cache_mode"] == "cls_local_npy"
    assert root.attrs["local_feature_complete"] is True
    assert cls.shape == (2, 3, 4)
    assert np.array_equal(index, [0, 1])

    local_root = cache_root / "frame_backbone_local"
    first = np.load(
        local_feature_file_path(str(local_root), 0, frames_per_directory=1),
        allow_pickle=False,
    )
    second = np.load(
        local_feature_file_path(str(local_root), 1, frames_per_directory=1),
        allow_pickle=False,
    )
    assert first.shape == (3, 256, 4)
    assert first.dtype == np.float16
    assert second.shape == (3, 256, 4)
    assert np.all(first[0, 0] == 1)
    assert np.all(second[0, 0] == 4)


def test_precompute_frame_partitions_are_contiguous_and_complete() -> None:
    partitions = [precompute._frame_partition(11, rank, 3) for rank in range(3)]
    assert partitions == [(0, 4), (4, 8), (8, 11)]
    assert [index for start, end in partitions for index in range(start, end)] == list(
        range(11)
    )


def test_precompute_merges_two_rank_shards(tmp_path, monkeypatch) -> None:
    output_path = str(tmp_path / "frame_backbone.zarr")
    shard_dir = precompute._shard_dir(output_path)
    os.makedirs(shard_dir)
    rank_0 = np.full((3, 3, 4), 10.0, dtype=np.float32)
    rank_1 = np.full((2, 3, 4), 20.0, dtype=np.float32)
    np.save(precompute._shard_path(output_path, 0), rank_0)
    np.save(precompute._shard_path(output_path, 1), rank_1)
    monkeypatch.setattr(precompute, "_dist_info", lambda: (0, 2))

    class FakeDataset:
        zarr_path = str(tmp_path / "replay_buffer.zarr")
        image_size = 224

    shape = precompute._merge_rank_shards(
        output_path=output_path,
        total_frames=5,
        batch_size=2,
        dataset=FakeDataset(),
        fm_cfg={
            "image_encoder_name": "dinov2_small",
            "dino_model_name": "vit_small_patch14_dinov2.lvd142m",
        },
        cache_mode="cls",
        token_mode="cls",
        local_enabled=False,
        local_root=str(tmp_path / "unused_local"),
        frames_per_directory=10,
        camera_views=("base_0", "left_wrist_0", "right_wrist_0"),
        selection="all_frames",
    )

    root = zarr.open_group(output_path, mode="r")
    merged = np.asarray(root["data"]["frame_image_backbone_feat"][:])
    assert shape == (5, 3, 4)
    assert np.array_equal(merged[:3], rank_0)
    assert np.array_equal(merged[3:], rank_1)
    assert root.attrs["distributed_world_size"] == 2
    assert not os.path.exists(shard_dir)


def test_precompute_temporary_cleanup_retries_cpfs_enotempty(
    tmp_path, monkeypatch
) -> None:
    temporary = tmp_path / "shards"
    temporary.mkdir()
    (temporary / "rank.npy").write_bytes(b"shard")
    real_rmtree = precompute.shutil.rmtree
    calls = 0

    def flaky_rmtree(path: str) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError(39, "Directory not empty", path)
        real_rmtree(path)

    monkeypatch.setattr(precompute.shutil, "rmtree", flaky_rmtree)
    monkeypatch.setattr(precompute.time, "sleep", lambda _seconds: None)
    assert precompute._remove_tree_best_effort(str(temporary))
    assert calls == 2
    assert not temporary.exists()


if __name__ == "__main__":
    test_mock_forward_backward()
    test_predict_action_shape()
    test_backbone_feat_forward_backward()
    print("[test_forward] all passed")
