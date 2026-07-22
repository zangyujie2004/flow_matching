from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
import zarr
from tqdm import tqdm

_POLICY_ROOT = Path(__file__).resolve().parents[1]
if str(_POLICY_ROOT) not in sys.path:
    sys.path.insert(0, str(_POLICY_ROOT))

os.environ.setdefault("HF_HUB_OFFLINE", "1")

from datasets import ZarrDataset  # noqa: E402
from models.fm.encoders.dino_v2 import DinoV2SmallEncoder, resolve_dino_model_name  # noqa: E402
from tools.latent_cache import (  # noqa: E402
    DINOV2_NUM_TOKENS,
    FRAME_CACHE_VERSION,
    apply_resolved_latent_cache_root_dir,
    frame_cache_matches,
    resolve_frame_backbone_zarr_path,
    write_latent_cache_identity_attrs,
)
from utils.train_utils import cfg_get, load_config  # noqa: E402


def build_dataset(cfg: dict) -> ZarrDataset:
    """RGB-only dataset for frame encoding (ignores train window / memory / latent)."""
    data_cfg = dict(cfg["data"])
    data_cfg["use_camera_latent"] = False
    data_cfg["latent_cache_root_dir"] = None
    data_cfg["fit_normalizer"] = False
    data_cfg["camera_augmentation"] = False
    # Always encode all zarr views; train slices later.
    data_cfg.pop("camera_views", None)
    # Frame SSOT: never truncate for partial smoke encodes.
    data_cfg.pop("max_windows", None)
    data_cfg.pop("memory", None)
    return ZarrDataset.from_config(data_cfg)


def resolve_output_path_from_cfg(cfg: dict, output_path: str | None = None) -> str:
    if output_path:
        return str(output_path)
    cfg = apply_resolved_latent_cache_root_dir(dict(cfg))
    root = cfg_get(cfg, "data.latent_cache_root_dir", None) or cfg_get(cfg, "data.root_dir")
    if root is None:
        raise KeyError("data.root_dir is required to resolve precompute output path")
    return resolve_frame_backbone_zarr_path(str(root))


def build_frame_image_batch(dataset: ZarrDataset, frame_indices: list[int]) -> torch.Tensor:
    batch = []
    for frame_idx in frame_indices:
        camera = dataset.get_camera(int(frame_idx), int(frame_idx) + 1)
        processed = dataset._process_image(camera)
        if processed.ndim != 5:
            raise ValueError(f"expected processed frame image (1,V,3,H,W), got {processed.shape}")
        batch.append(processed[0].numpy())
    return torch.from_numpy(np.stack(batch, axis=0))


def precompute_image_latents(cfg: dict, *, force: bool = False) -> str:
    """Write frame-only DINO backbone cache (scheme A). Independent of train windows."""
    cfg = apply_resolved_latent_cache_root_dir(dict(cfg))
    pre_cfg = dict(cfg.get("precompute", {}))
    output_path = resolve_output_path_from_cfg(cfg, pre_cfg.get("output_path"))
    # yaml overwrite=true kept as force alias for backward compat
    force = bool(force) or bool(pre_cfg.get("overwrite", False))

    batch_size = max(1, int(pre_cfg.get("batch_size", 256)))
    device = torch.device(str(pre_cfg.get("device", cfg_get(cfg, "runtime.device", "cuda"))))
    fm_cfg = dict(cfg["models"]["fm"])
    if not bool(fm_cfg.get("freeze_image_encoder", True)):
        raise ValueError("Precompute requires models.fm.freeze_image_encoder=true.")

    model_name = resolve_dino_model_name(
        fm_cfg.get("image_encoder_name"),
        fm_cfg.get("dino_model_name"),
    )
    fm_cfg["dino_model_name"] = model_name

    dataset = build_dataset(cfg)
    total_frames = int(dataset.ram_data[dataset.camera_key].shape[0])
    state_frames = int(dataset.ram_data[dataset.state_key].shape[0])
    if total_frames != state_frames:
        raise ValueError(
            f"camera/state frame count mismatch before encode: camera={total_frames}, state={state_frames}"
        )

    if (not force) and frame_cache_matches(
        output_path,
        fm_cfg=fm_cfg,
        source_zarr_path=dataset.zarr_path,
        image_size=int(dataset.image_size),
        camera_views=dataset.camera_views,
        total_frames=total_frames,
        color_order="rgb",
    ):
        print(f"[precompute] frame cache identity match, skipping: {output_path}")
        return output_path

    if os.path.isdir(output_path):
        print(f"[precompute] removing existing cache (force={force}): {output_path}")
        shutil.rmtree(output_path)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    image_encoder = DinoV2SmallEncoder(
        out_dim=int(fm_cfg.get("image_feat_dim", 256)),
        pretrained=bool(fm_cfg.get("image_pretrained", True)),
        freeze=True,
        model_name=model_name,
    ).to(device)
    image_encoder.eval()

    out_root = zarr.open_group(output_path, mode="w")
    out_root.attrs["cache_version"] = int(FRAME_CACHE_VERSION)
    out_root.attrs["source_zarr_path"] = dataset.zarr_path
    out_root.attrs["image_size"] = int(dataset.image_size)
    out_root.attrs["color_order"] = "rgb"
    out_root.attrs["frame_image_selection"] = "all_frames"
    out_root.attrs["image_num_tokens"] = int(DINOV2_NUM_TOKENS)
    write_latent_cache_identity_attrs(out_root, fm_cfg)
    out_root.attrs["camera_views"] = ",".join(dataset.camera_views)

    data_group = out_root.create_group("data")
    # empty meta group kept for zarr layout stability
    out_root.create_group("meta")

    # Token cache is large (T,V,257,D); keep chunks modest for OSS/CPFS writes.
    chunk_bsz = max(1, min(batch_size, 64))
    frame_arr = None
    print(
        f"[precompute] encoding all frames: T={total_frames}, views={list(dataset.camera_views)}, "
        f"model={model_name}, tokens={DINOV2_NUM_TOKENS}, "
        f"batch_size={batch_size}, device={device}, out={output_path}"
    )

    for start_idx in tqdm(
        range(0, total_frames, batch_size),
        desc="precompute:frame_image_backbone_feat",
        unit="batch",
    ):
        frame_indices = list(range(start_idx, min(start_idx + batch_size, total_frames)))
        image_batch = build_frame_image_batch(dataset, frame_indices).to(device, non_blocking=True)

        with torch.inference_mode():
            bsz, num_views = image_batch.shape[:2]
            flat = image_batch.reshape(bsz * num_views, *image_batch.shape[2:])
            tokens = image_encoder.extract_backbone_feat(flat)  # (B*V, 257, D)
            image_feat = tokens.reshape(bsz, num_views, tokens.shape[1], tokens.shape[2])

        img = image_feat.detach().cpu().numpy().astype(np.float32, copy=False)
        if frame_arr is None:
            frame_arr = data_group.create_array(
                "frame_image_backbone_feat",
                shape=(total_frames,) + img.shape[1:],
                chunks=(chunk_bsz,) + img.shape[1:],
                dtype="f4",
            )
            out_root.attrs["image_backbone_dim"] = int(img.shape[-1])
            out_root.attrs["n_image_views"] = int(img.shape[1])
            out_root.attrs["image_num_tokens"] = int(img.shape[2])
        frame_arr[start_idx : start_idx + len(frame_indices)] = img

    if frame_arr is None:
        raise RuntimeError("no frames were encoded")

    print(f"[precompute] saved frame backbone cache: {output_path}")
    print(
        f"[precompute] frame_image_backbone_feat shape=({total_frames}, "
        f"{out_root.attrs['n_image_views']}, {out_root.attrs['image_num_tokens']}, "
        f"{out_root.attrs['image_backbone_dim']})"
    )
    return output_path


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Precompute frame-only frozen DINOv2 backbone features (scheme A)."
    )
    parser.add_argument("--config", type=str, default="configs/train/config.yaml")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild even when an identity-matching frame cache already exists.",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = _POLICY_ROOT / config_path

    with open(config_path, encoding="utf-8") as handle:
        peek = yaml.safe_load(handle)
    if isinstance(peek, dict) and peek.get("finetune"):
        from utils.finetune_config import resolve_full_config

        cfg = resolve_full_config(config_path, policy_root=_POLICY_ROOT)
    else:
        cfg = load_config(str(config_path))
    cfg = apply_resolved_latent_cache_root_dir(cfg)
    precompute_image_latents(cfg, force=bool(args.force))


if __name__ == "__main__":
    main()
