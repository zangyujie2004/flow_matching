"""Smoke tests for ZarrDataset + PyTorch DataLoader."""

from __future__ import annotations

import os
import traceback

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from _bootstrap import ensure_policy_root

_POLICY_ROOT = ensure_policy_root()
from datasets import ZarrDataset, build_dataloader


def _load_dataset(root_dir: str, max_windows: int = 100) -> ZarrDataset:
    config_path = os.path.join(_POLICY_ROOT, "configs", "config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)["data"]
    data_cfg["root_dir"] = root_dir
    data_cfg["norm"] = {"max_windows": max_windows}
    return ZarrDataset.from_config(data_cfg)


def _check_sample(sample: dict, ds: ZarrDataset) -> None:
    assert "obs" in sample and "action" in sample and "meta" in sample
    obs = sample["obs"]
    action = sample["action"]

    assert obs["state"].shape == (ds.window_size, ds.action_dim)
    assert obs["state"].dtype == torch.float32
    assert action.shape == (ds.action_horizon, ds.action_dim)
    assert action.dtype == torch.float32

    img = obs["image"]
    n_views = 3
    assert img.shape == (ds.n_image_steps, n_views, 3, ds.image_size, ds.image_size)
    assert img.dtype == torch.uint8 if ds.image_as_uint8 else torch.float32

    if ds.use_tactile:
        assert "tactile" in obs
        assert obs["tactile"].shape == (ds.window_size, 35, 20, ds.tactile_dim)
        assert obs["tactile"].dtype == torch.float32

    for key in ("idx", "anchor_t", "ep_idx"):
        assert key in sample["meta"]


def _check_batch(batch: dict, ds: ZarrDataset, batch_size: int) -> None:
    obs = batch["obs"]
    action = batch["action"]
    meta = batch["meta"]

    assert obs["state"].shape == (batch_size, ds.window_size, ds.action_dim)
    assert action.shape == (batch_size, ds.action_horizon, ds.action_dim)

    n_views = 3
    assert obs["image"].shape == (
        batch_size,
        ds.n_image_steps,
        n_views,
        3,
        ds.image_size,
        ds.image_size,
    )

    if ds.use_tactile:
        assert obs["tactile"].shape == (batch_size, ds.window_size, 35, 20, ds.tactile_dim)

    assert meta["idx"].shape == (batch_size,)
    assert meta["anchor_t"].shape == (batch_size,)
    assert meta["ep_idx"].shape == (batch_size,)

    assert torch.isfinite(obs["state"]).all()
    assert torch.isfinite(action).all()
    if ds.use_tactile:
        assert torch.isfinite(obs["tactile"]).all()


def run_case(ds: ZarrDataset, *, batch_size: int, num_workers: int) -> None:
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=False,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
    )
    batch = next(iter(loader))
    _check_batch(batch, ds, batch_size)
    print(
        f"  OK batch_size={batch_size} num_workers={num_workers} "
        f"action={tuple(batch['action'].shape)} "
        f"state={tuple(batch['obs']['state'].shape)}"
    )


def main() -> int:
    root_dir = os.environ.get(
        "ZARR_DATA_ROOT",
        "/mnt/workspace/zyj/data/processed/peel/peel_0630_1656",
    )
    print(f"[test_dataloader] root_dir={root_dir}")
    ds = _load_dataset(root_dir, max_windows=100)
    print(f"[test_dataloader] windows={len(ds)} action_dim={ds.action_dim}")

    sample = ds[0]
    _check_sample(sample, ds)
    print("[test_dataloader] single sample OK")

    cases = [(2, 0), (4, 0), (2, 2), (4, 4)]
    for batch_size, num_workers in cases:
        try:
            run_case(ds, batch_size=batch_size, num_workers=num_workers)
        except Exception:
            print(f"[test_dataloader] FAILED batch_size={batch_size} num_workers={num_workers}")
            traceback.print_exc()
            return 1

    state = ds.normalizer.state_dict()
    restored = type(ds.normalizer).load_state_dict(state)
    x = sample["obs"]["state"].numpy()
    y1 = ds.normalizer.normalize_state_np(x)
    y2 = restored.normalize_state_np(x)
    assert np.allclose(y1, y2, atol=1e-6)

    # build_dataloader helper + multi-batch iteration
    loader = build_dataloader(ds, batch_size=2, shuffle=True, num_workers=0, drop_last=True)
    for i, batch in enumerate(loader):
        _check_batch(batch, ds, 2)
        if i >= 2:
            break
    print("[test_dataloader] build_dataloader multi-batch OK")

    # use_tactile=False
    cfg = yaml.safe_load(open(os.path.join(_POLICY_ROOT, "configs", "config.yaml")))["data"]
    cfg["root_dir"] = root_dir
    cfg["use_tactile"] = False
    cfg["norm"] = {"max_windows": 50}
    ds_no_tactile = ZarrDataset.from_config(cfg)
    batch = next(iter(build_dataloader(ds_no_tactile, batch_size=2, num_workers=0)))
    assert "tactile" not in batch["obs"]
    print("[test_dataloader] use_tactile=False OK")

    print("[test_dataloader] batch contract for trainer:")
    b = next(iter(build_dataloader(ds, batch_size=2, num_workers=0)))
    print(f"  obs.state:   {tuple(b['obs']['state'].shape)}")
    print(f"  obs.image:   {tuple(b['obs']['image'].shape)}")
    print(f"  obs.tactile: {tuple(b['obs']['tactile'].shape)}")
    print(f"  action:      {tuple(b['action'].shape)}")

    print("[test_dataloader] all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
