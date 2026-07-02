from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import zarr
from tqdm import tqdm

_POLICY_ROOT = Path(__file__).resolve().parents[1]
if str(_POLICY_ROOT) not in sys.path:
    sys.path.insert(0, str(_POLICY_ROOT))

os.environ.setdefault("HF_HUB_OFFLINE", "1")

from datasets import ZarrDataset  # noqa: E402
from models.fm.encoders.dino_v2 import DinoV2SmallEncoder  # noqa: E402
from utils.train_utils import cfg_get, load_config  # noqa: E402


def build_dataset(cfg: dict) -> ZarrDataset:
    data_cfg = dict(cfg["data"])
    data_cfg["use_camera_latent"] = False
    data_cfg["latent_cache_root_dir"] = None
    data_cfg["fit_normalizer"] = False
    return ZarrDataset.from_config(data_cfg)


def resolve_output_path(dataset: ZarrDataset, cfg: dict, output_path: str | None) -> str:
    if output_path:
        return str(output_path)
    root = cfg_get(cfg, "data.latent_cache_root_dir", None) or dataset.root_dir
    return os.path.join(str(root), "policy_latent_cache.zarr")


def build_image_batch(dataset: ZarrDataset, indices: list[int]) -> torch.Tensor:
    batch = []
    for idx in indices:
        i0, i1 = dataset.image_range(idx)
        camera = dataset.get_camera(i0, i1)
        if camera.shape[0] != dataset.n_image_steps:
            raise ValueError(
                f"image length mismatch for idx={idx}: {camera.shape[0]} != {dataset.n_image_steps}"
            )
        processed = dataset._process_image(camera)
        batch.append(processed.numpy())
    return torch.from_numpy(np.stack(batch, axis=0))


def write_window_metadata(meta_group, dataset: ZarrDataset, total_windows: int) -> None:
    windows = np.asarray(dataset.windows[:total_windows], dtype=np.int64)
    meta_group.create_array("window_anchor_times", data=windows[:, 0])
    meta_group.create_array("window_episode_ends", data=windows[:, 1])
    meta_group.create_array("window_episode_indices", data=windows[:, 2])


def precompute_image_latents(cfg: dict) -> str:
    pre_cfg = dict(cfg.get("precompute", {}))
    dataset = build_dataset(cfg)
    output_path = resolve_output_path(dataset, cfg, pre_cfg.get("output_path"))
    overwrite = bool(pre_cfg.get("overwrite", False))
    batch_size = max(1, int(pre_cfg.get("batch_size", 256)))
    device = torch.device(str(pre_cfg.get("device", cfg_get(cfg, "runtime.device", "cuda"))))
    max_windows = pre_cfg.get("max_windows")
    total_windows = len(dataset)
    if max_windows is not None:
        total_windows = max(1, min(int(max_windows), len(dataset)))

    if os.path.isdir(output_path):
        if not overwrite:
            raise FileExistsError(
                f"Output zarr already exists: {output_path}. "
                "Set precompute.overwrite=true in config to rebuild."
            )
        shutil.rmtree(output_path)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    fm_cfg = cfg["models"]["fm"]
    if not bool(fm_cfg.get("freeze_image_encoder", True)):
        raise ValueError("Precompute requires models.fm.freeze_image_encoder=true.")

    image_encoder = DinoV2SmallEncoder(
        out_dim=int(fm_cfg.get("image_feat_dim", 256)),
        pretrained=bool(fm_cfg.get("image_pretrained", True)),
        freeze=True,
        model_name=str(fm_cfg.get("dino_model_name", "vit_small_patch14_dinov2.lvd142m")),
    ).to(device)
    image_encoder.eval()

    out_root = zarr.open_group(output_path, mode="w")
    out_root.attrs["cache_version"] = 1
    out_root.attrs["source_zarr_path"] = dataset.zarr_path
    out_root.attrs["window_size"] = int(dataset.window_size)
    out_root.attrs["action_horizon"] = int(dataset.action_horizon)
    out_root.attrs["n_image_steps"] = int(dataset.n_image_steps)
    out_root.attrs["stride"] = int(dataset.stride)
    out_root.attrs["image_selection"] = "anchor"
    out_root.attrs["dino_model_name"] = str(fm_cfg.get("dino_model_name", ""))
    out_root.attrs["camera_views"] = ",".join(dataset.camera_views)

    data_group = out_root.create_group("data")
    meta_group = out_root.create_group("meta")
    write_window_metadata(meta_group, dataset, total_windows)

    img_arr = None
    chunk_bsz = max(1, min(batch_size, 1024))

    for start_idx in tqdm(
        range(0, total_windows, batch_size),
        desc="precompute:image_backbone_feat",
        unit="window",
    ):
        batch_indices = list(range(start_idx, min(start_idx + batch_size, total_windows)))
        image_batch = build_image_batch(dataset, batch_indices).to(device, non_blocking=True)

        with torch.inference_mode():
            bsz, num_steps, num_views = image_batch.shape[:3]
            flat = image_batch.reshape(bsz * num_steps * num_views, *image_batch.shape[3:])
            image_feat = image_encoder.extract_backbone_feat(flat).reshape(
                bsz, num_steps, num_views, -1
            )

        img = image_feat.detach().cpu().numpy().astype(np.float32, copy=False)

        if img_arr is None:
            img_arr = data_group.create_array(
                "image_backbone_feat",
                shape=(total_windows,) + img.shape[1:],
                chunks=(chunk_bsz,) + img.shape[1:],
                dtype="f4",
            )
            out_root.attrs["image_backbone_dim"] = int(img.shape[-1])
            out_root.attrs["n_image_views"] = int(img.shape[2])

        img_arr[start_idx : start_idx + len(batch_indices)] = img

    print(f"[precompute] saved image backbone cache: {output_path}")
    print(
        f"[precompute] shape=({total_windows}, {img.shape[1]}, {img.shape[2]}, {img.shape[3]}), "
        f"backbone_dim={out_root.attrs['image_backbone_dim']}"
    )
    return output_path


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Precompute frozen DINOv2 backbone features.")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = _POLICY_ROOT / config_path

    cfg = load_config(str(config_path))
    precompute_image_latents(cfg)


if __name__ == "__main__":
    main()
