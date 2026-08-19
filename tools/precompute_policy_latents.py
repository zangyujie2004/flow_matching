from __future__ import annotations

import os
import shutil
import sys
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
import zarr
from tqdm import tqdm

_POLICY_ROOT = Path(__file__).resolve().parents[1]
if str(_POLICY_ROOT) not in sys.path:
    sys.path.insert(0, str(_POLICY_ROOT))

os.environ.setdefault("HF_HUB_OFFLINE", "1")

from datasets.zarr_dataset import CAMERA_BUNDLE_ORDER  # noqa: E402
from models.fm.encoders.dino_v2 import DinoV2SmallEncoder, resolve_dino_model_name  # noqa: E402
from tools.latent_cache import (  # noqa: E402
    CAMERA_BASE_REMOVE_HAND_KEY,
    FRAME_CACHE_VERSION,
    TOKEN_MODE_ALL,
    TOKEN_MODE_CLS,
    TokenMode,
    apply_resolved_latent_cache_root_dir,
    frame_cache_matches,
    normalize_token_mode,
    remove_hand_frame_cache_matches,
    resolve_frame_backbone_base_remove_hand_zarr_path,
    resolve_frame_backbone_zarr_path,
    token_mode_num_tokens,
    write_latent_cache_identity_attrs,
    write_token_mode_attrs,
)
from utils.train_utils import cfg_get, load_config  # noqa: E402


@dataclass(frozen=True)
class FrameSource:
    """Minimal read-only source needed for DINO frame precomputation."""

    zarr_path: str
    camera: zarr.Array
    remove_hand: zarr.Array | None
    image_size: int
    image_as_uint8: bool
    camera_views: tuple[str, ...]


def _resolve_replay_buffer_path(root_dir: str) -> Path:
    root = Path(root_dir).expanduser().resolve()
    if root.name == "replay_buffer.zarr" and root.is_dir():
        return root
    candidate = root / "replay_buffer.zarr"
    if candidate.is_dir():
        return candidate
    raise FileNotFoundError(
        f"Cannot find replay_buffer.zarr from data.root_dir={root_dir}. "
        f"Tried: {candidate}"
    )


def build_frame_source(cfg: dict) -> FrameSource:
    """Open only the arrays required by frame precompute; never preload the dataset."""

    data_cfg = dict(cfg["data"])
    replay_path = _resolve_replay_buffer_path(str(data_cfg["root_dir"]))
    root = zarr.open_group(str(replay_path), mode="r")
    if "data" not in root or "meta" not in root:
        raise KeyError(f"expected data/meta groups in {replay_path}")
    data_group = root["data"]
    meta_group = root["meta"]
    camera_key = str(data_cfg.get("camera_key", "camera"))
    state_key = str(data_cfg.get("state_key", "state_30hz"))
    if camera_key not in data_group:
        raise KeyError(f"missing data/{camera_key} in {replay_path}")
    if state_key not in data_group:
        raise KeyError(f"missing data/{state_key} in {replay_path}")
    if "episode_ends" not in meta_group:
        raise KeyError(f"missing meta/episode_ends in {replay_path}")

    camera = data_group[camera_key]
    if camera.ndim != 4 or int(camera.shape[-1]) % 3 != 0:
        raise ValueError(
            f"data/{camera_key} must be (T,H,W,3*V), got {tuple(camera.shape)}"
        )
    total_frames = int(camera.shape[0])
    state_frames = int(data_group[state_key].shape[0])
    if total_frames != state_frames:
        raise ValueError(
            f"camera/state frame count mismatch: camera={total_frames}, "
            f"state={state_frames}"
        )
    episode_ends = np.asarray(meta_group["episode_ends"][:], dtype=np.int64)
    if len(episode_ends) == 0 or int(episode_ends[-1]) != total_frames:
        raise ValueError(
            "meta/episode_ends does not match camera length: "
            f"last={episode_ends[-1] if len(episode_ends) else None}, "
            f"camera={total_frames}"
        )

    n_views = int(camera.shape[-1]) // 3
    if n_views < 1 or n_views > len(CAMERA_BUNDLE_ORDER):
        raise ValueError(
            f"unsupported camera view count V={n_views}; "
            f"known order={CAMERA_BUNDLE_ORDER}"
        )
    camera_views = tuple(CAMERA_BUNDLE_ORDER[:n_views])

    remove_hand = None
    if CAMERA_BASE_REMOVE_HAND_KEY in data_group:
        candidate = data_group[CAMERA_BASE_REMOVE_HAND_KEY]
        if candidate.ndim != 4 or tuple(candidate.shape[1:]) != (
            int(camera.shape[1]),
            int(camera.shape[2]),
            3,
        ):
            raise ValueError(
                f"data/{CAMERA_BASE_REMOVE_HAND_KEY} must be (T_rh,H,W,3), "
                f"got {tuple(candidate.shape)}"
            )
        remove_hand = candidate

    source = FrameSource(
        zarr_path=str(replay_path),
        camera=camera,
        remove_hand=remove_hand,
        image_size=int(data_cfg.get("image_size", 224)),
        image_as_uint8=bool(data_cfg.get("image_as_uint8", True)),
        camera_views=camera_views,
    )
    print(
        "[precompute] streaming source opened: "
        f"camera={tuple(camera.shape)}, views={list(camera_views)}, "
        f"remove_hand={None if remove_hand is None else tuple(remove_hand.shape)}, "
        f"zarr={replay_path}"
    )
    return source


def resolve_output_path_from_cfg(cfg: dict, output_path: str | None = None) -> str:
    if output_path:
        return str(output_path)
    cfg = apply_resolved_latent_cache_root_dir(dict(cfg))
    root = cfg_get(cfg, "data.latent_cache_root_dir", None) or cfg_get(cfg, "data.root_dir")
    if root is None:
        raise KeyError("data.root_dir is required to resolve precompute output path")
    return resolve_frame_backbone_zarr_path(str(root))


def resolve_remove_hand_output_path(cfg: dict) -> str:
    cfg = apply_resolved_latent_cache_root_dir(dict(cfg))
    root = cfg_get(cfg, "data.latent_cache_root_dir", None) or cfg_get(cfg, "data.root_dir")
    if root is None:
        raise KeyError("data.root_dir is required to resolve remove-hand cache path")
    return resolve_frame_backbone_base_remove_hand_zarr_path(str(root))


def resolve_token_mode_from_cfg(cfg: dict) -> TokenMode:
    pre_cfg = dict(cfg.get("precompute") or {})
    return normalize_token_mode(pre_cfg.get("token_mode"), default=TOKEN_MODE_CLS)


def _process_camera_batch(
    raw: np.ndarray,
    *,
    image_size: int,
    image_as_uint8: bool,
) -> torch.Tensor:
    """Vectorized equivalent of ZarrDataset._process_image for a frame batch."""

    arr = np.asarray(raw)
    if arr.ndim != 4 or int(arr.shape[-1]) % 3 != 0:
        raise ValueError(f"expected camera batch (B,H,W,3*V), got {arr.shape}")
    bsz, height, width, channels = arr.shape
    n_views = int(channels) // 3
    contiguous = np.ascontiguousarray(arr)
    images = (
        torch.from_numpy(contiguous)
        .reshape(bsz, height, width, n_views, 3)
        .permute(0, 3, 4, 1, 2)
        .contiguous()
    )
    if height != image_size or width != image_size:
        flat = images.flatten(0, 1).float()
        resized = F.interpolate(
            flat,
            size=(image_size, image_size),
            mode="bilinear",
            align_corners=False,
        )
        if image_as_uint8:
            resized = resized.round().clamp_(0.0, 255.0).to(torch.uint8)
        else:
            resized = resized.div_(255.0).mul_(2.0).sub_(1.0)
        return resized.reshape(bsz, n_views, 3, image_size, image_size)
    if image_as_uint8:
        return images
    return images.float().div_(255.0).mul_(2.0).sub_(1.0)


def _read_image_batch(
    source_array: zarr.Array,
    start: int,
    stop: int,
    *,
    image_size: int,
    image_as_uint8: bool,
) -> torch.Tensor:
    raw = np.asarray(source_array[start:stop])
    return _process_camera_batch(
        raw,
        image_size=image_size,
        image_as_uint8=image_as_uint8,
    )


def _iter_prefetched_batches(
    source_array: zarr.Array,
    *,
    total_frames: int,
    batch_size: int,
    image_size: int,
    image_as_uint8: bool,
    num_workers: int,
    prefetch_batches: int,
) -> Iterator[tuple[int, torch.Tensor]]:
    """Yield ordered batches while a bounded thread pool reads future Zarr slices."""

    ranges = iter(
        (start, min(start + batch_size, total_frames))
        for start in range(0, total_frames, batch_size)
    )

    def load(start: int, stop: int) -> torch.Tensor:
        return _read_image_batch(
            source_array,
            start,
            stop,
            image_size=image_size,
            image_as_uint8=image_as_uint8,
        )

    workers = max(0, int(num_workers))
    prefetch = max(1, int(prefetch_batches))
    if workers <= 1:
        for start, stop in ranges:
            yield start, load(start, stop)
        return

    pending: deque[tuple[int, Future[torch.Tensor]]] = deque()
    with ThreadPoolExecutor(
        max_workers=workers,
        thread_name_prefix="zarr-frame-reader",
    ) as executor:
        for _ in range(prefetch):
            try:
                start, stop = next(ranges)
            except StopIteration:
                break
            pending.append((start, executor.submit(load, start, stop)))

        while pending:
            start, future = pending.popleft()
            yield start, future.result()
            try:
                next_start, next_stop = next(ranges)
            except StopIteration:
                continue
            pending.append(
                (next_start, executor.submit(load, next_start, next_stop))
            )


class _BackboneFeatureExtractor(nn.Module):
    """DataParallel-compatible wrapper around extract_backbone_feat."""

    def __init__(self, encoder: DinoV2SmallEncoder, token_mode: TokenMode):
        super().__init__()
        self.encoder = encoder
        self.token_mode = token_mode

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        tokens = self.encoder.extract_backbone_feat(images)
        if self.token_mode == TOKEN_MODE_CLS:
            return tokens[:, 0, :]
        return tokens


def _tokens_to_stored_feat(
    tokens: torch.Tensor,
    *,
    batch_size: int,
    num_views: int,
    token_mode: TokenMode,
) -> np.ndarray:
    if token_mode == TOKEN_MODE_CLS:
        if tokens.ndim == 3:
            tokens = tokens[:, 0, :]
        if tokens.ndim != 2:
            raise ValueError(f"expected CLS features (B*V,D), got {tokens.shape}")
        feat = tokens.reshape(batch_size, num_views, tokens.shape[-1])
    else:
        if tokens.ndim != 3:
            raise ValueError(f"expected tokens (B*V,N,D), got {tokens.shape}")
        feat = tokens.reshape(
            batch_size,
            num_views,
            tokens.shape[1],
            tokens.shape[2],
        )
    return feat.detach().cpu().numpy().astype(np.float32, copy=False)


def _build_encoder(fm_cfg: dict, device: torch.device) -> DinoV2SmallEncoder:
    model_name = resolve_dino_model_name(
        fm_cfg.get("image_encoder_name"),
        fm_cfg.get("dino_model_name"),
    )
    fm_cfg["dino_model_name"] = model_name
    encoder = DinoV2SmallEncoder(
        out_dim=int(fm_cfg.get("image_feat_dim", 256)),
        pretrained=bool(fm_cfg.get("image_pretrained", True)),
        freeze=True,
        model_name=model_name,
    ).to(device)
    encoder.eval()
    return encoder


def _build_feature_extractor(
    fm_cfg: dict,
    *,
    device: torch.device,
    token_mode: TokenMode,
    multi_gpu: bool,
) -> tuple[nn.Module, torch.device, bool, int]:
    use_data_parallel = bool(
        multi_gpu and device.type == "cuda" and torch.cuda.device_count() > 1
    )
    if use_data_parallel and device.index not in {None, 0}:
        raise ValueError(
            "precompute.multi_gpu=true requires precompute.device=cuda or cuda:0; "
            f"got {device}"
        )
    primary_device = torch.device("cuda:0") if use_data_parallel else device
    encoder = _build_encoder(fm_cfg, primary_device)
    extractor: nn.Module = _BackboneFeatureExtractor(encoder, token_mode).eval()
    gpu_count = 1 if device.type == "cuda" else 0
    if use_data_parallel:
        device_ids = list(range(torch.cuda.device_count()))
        extractor = nn.DataParallel(
            extractor,
            device_ids=device_ids,
            output_device=device_ids[0],
        ).eval()
        gpu_count = len(device_ids)
    return extractor, primary_device, use_data_parallel, gpu_count


def _encode_image_batch(
    extractor: nn.Module,
    image_batch: torch.Tensor,
    *,
    primary_device: torch.device,
    use_data_parallel: bool,
    token_mode: TokenMode,
) -> np.ndarray:
    bsz, num_views = image_batch.shape[:2]
    flat = image_batch.reshape(bsz * num_views, *image_batch.shape[2:])
    if not use_data_parallel:
        flat = flat.to(primary_device, non_blocking=False)
    with torch.inference_mode():
        tokens = extractor(flat)
    return _tokens_to_stored_feat(
        tokens,
        batch_size=bsz,
        num_views=num_views,
        token_mode=token_mode,
    )


def _precompute_runtime(cfg: dict) -> tuple[int, int, int, bool, torch.device]:
    pre_cfg = dict(cfg.get("precompute") or {})
    batch_size = max(1, int(pre_cfg.get("batch_size", 256)))
    num_workers = max(0, int(pre_cfg.get("num_workers", 4)))
    prefetch_batches = max(1, int(pre_cfg.get("prefetch_batches", max(1, num_workers))))
    multi_gpu = bool(pre_cfg.get("multi_gpu", True))
    device = torch.device(
        str(pre_cfg.get("device", cfg_get(cfg, "runtime.device", "cuda")))
    )
    return batch_size, num_workers, prefetch_batches, multi_gpu, device


def precompute_image_latents(
    cfg: dict,
    *,
    force: bool = False,
    source: FrameSource | None = None,
) -> str:
    """Write frame-only DINO cache from streaming Zarr camera batches."""

    cfg = apply_resolved_latent_cache_root_dir(dict(cfg))
    pre_cfg = dict(cfg.get("precompute") or {})
    output_path = resolve_output_path_from_cfg(cfg, pre_cfg.get("output_path"))
    force = bool(force) or bool(pre_cfg.get("overwrite", False))
    token_mode = resolve_token_mode_from_cfg(cfg)
    batch_size, num_workers, prefetch_batches, multi_gpu, device = _precompute_runtime(cfg)
    fm_cfg = dict(cfg["models"]["fm"])
    if not bool(fm_cfg.get("freeze_image_encoder", True)):
        raise ValueError("Precompute requires models.fm.freeze_image_encoder=true.")

    source = build_frame_source(cfg) if source is None else source
    total_frames = int(source.camera.shape[0])
    if (not force) and frame_cache_matches(
        output_path,
        fm_cfg=fm_cfg,
        source_zarr_path=source.zarr_path,
        image_size=source.image_size,
        camera_views=source.camera_views,
        total_frames=total_frames,
        token_mode=token_mode,
        color_order="rgb",
    ):
        print(
            f"[precompute] frame cache identity match (token_mode={token_mode}), "
            f"skipping: {output_path}"
        )
        return output_path

    if os.path.isdir(output_path):
        print(f"[precompute] removing existing cache (force={force}): {output_path}")
        shutil.rmtree(output_path)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    extractor, primary_device, use_data_parallel, gpu_count = _build_feature_extractor(
        fm_cfg,
        device=device,
        token_mode=token_mode,
        multi_gpu=multi_gpu,
    )
    model_name = str(fm_cfg["dino_model_name"])
    out_root = zarr.open_group(output_path, mode="w")
    out_root.attrs["cache_version"] = int(FRAME_CACHE_VERSION)
    out_root.attrs["source_zarr_path"] = source.zarr_path
    out_root.attrs["image_size"] = int(source.image_size)
    out_root.attrs["color_order"] = "rgb"
    out_root.attrs["frame_image_selection"] = "all_frames"
    write_latent_cache_identity_attrs(out_root, fm_cfg)
    write_token_mode_attrs(out_root, token_mode)
    out_root.attrs["camera_views"] = ",".join(source.camera_views)
    data_group = out_root.create_group("data")
    out_root.create_group("meta")

    chunk_bsz = max(1, min(batch_size, 64))
    frame_arr = None
    n_tok = token_mode_num_tokens(token_mode)
    num_batches = (total_frames + batch_size - 1) // batch_size
    print(
        f"[precompute] encoding all frames: T={total_frames}, "
        f"views={list(source.camera_views)}, model={model_name}, "
        f"token_mode={token_mode}, tokens={n_tok}, batch_size={batch_size}, "
        f"read_workers={num_workers}, prefetch_batches={prefetch_batches}, "
        f"gpus={gpu_count}, device={primary_device}, out={output_path}"
    )

    batches = _iter_prefetched_batches(
        source.camera,
        total_frames=total_frames,
        batch_size=batch_size,
        image_size=source.image_size,
        image_as_uint8=source.image_as_uint8,
        num_workers=num_workers,
        prefetch_batches=prefetch_batches,
    )
    for start_idx, image_batch in tqdm(
        batches,
        total=num_batches,
        desc="precompute:frame_image_backbone_feat",
        unit="batch",
    ):
        img = _encode_image_batch(
            extractor,
            image_batch,
            primary_device=primary_device,
            use_data_parallel=use_data_parallel,
            token_mode=token_mode,
        )
        if frame_arr is None:
            frame_arr = data_group.create_array(
                "frame_image_backbone_feat",
                shape=(total_frames,) + img.shape[1:],
                chunks=(chunk_bsz,) + img.shape[1:],
                dtype="f4",
            )
            out_root.attrs["image_backbone_dim"] = int(img.shape[-1])
            out_root.attrs["n_image_views"] = int(img.shape[1])
            if token_mode == TOKEN_MODE_ALL:
                out_root.attrs["image_num_tokens"] = int(img.shape[2])
        frame_arr[start_idx : start_idx + len(img)] = img

    if frame_arr is None:
        raise RuntimeError("no frames were encoded")
    print(f"[precompute] saved frame backbone cache: {output_path}")
    print(
        f"[precompute] frame_image_backbone_feat shape={tuple(frame_arr.shape)}, "
        f"token_mode={token_mode}"
    )
    return output_path


def precompute_base_remove_hand_latents(
    cfg: dict,
    *,
    force: bool = False,
    source: FrameSource | None = None,
) -> str | None:
    """Encode compact remove-hand frames without materializing the full array."""

    cfg = apply_resolved_latent_cache_root_dir(dict(cfg))
    pre_cfg = dict(cfg.get("precompute") or {})
    force = bool(force) or bool(pre_cfg.get("overwrite", False))
    token_mode = resolve_token_mode_from_cfg(cfg)
    output_path = resolve_remove_hand_output_path(cfg)
    batch_size, num_workers, prefetch_batches, multi_gpu, device = _precompute_runtime(cfg)
    fm_cfg = dict(cfg["models"]["fm"])
    if not bool(fm_cfg.get("freeze_image_encoder", True)):
        raise ValueError("Precompute requires models.fm.freeze_image_encoder=true.")

    source = build_frame_source(cfg) if source is None else source
    if source.remove_hand is None:
        print(
            f"[precompute] no data/{CAMERA_BASE_REMOVE_HAND_KEY} in "
            f"{source.zarr_path}; skip remove-hand cache"
        )
        return None
    total_frames = int(source.remove_hand.shape[0])
    if total_frames == 0:
        print("[precompute] remove-hand array empty; skip")
        return None

    if (not force) and remove_hand_frame_cache_matches(
        output_path,
        fm_cfg=fm_cfg,
        source_zarr_path=source.zarr_path,
        image_size=source.image_size,
        total_frames=total_frames,
        token_mode=token_mode,
        color_order="rgb",
    ):
        print(
            f"[precompute] remove-hand cache identity match "
            f"(token_mode={token_mode}), skipping: {output_path}"
        )
        return output_path

    if os.path.isdir(output_path):
        print(
            f"[precompute] removing existing remove-hand cache "
            f"(force={force}): {output_path}"
        )
        shutil.rmtree(output_path)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    extractor, primary_device, use_data_parallel, gpu_count = _build_feature_extractor(
        fm_cfg,
        device=device,
        token_mode=token_mode,
        multi_gpu=multi_gpu,
    )
    model_name = str(fm_cfg["dino_model_name"])
    out_root = zarr.open_group(output_path, mode="w")
    out_root.attrs["cache_version"] = int(FRAME_CACHE_VERSION)
    out_root.attrs["source_zarr_path"] = source.zarr_path
    out_root.attrs["image_size"] = int(source.image_size)
    out_root.attrs["color_order"] = "rgb"
    out_root.attrs["frame_image_selection"] = "camera_base_remove_hand"
    out_root.attrs["compact"] = True
    out_root.attrs["ties_to"] = CAMERA_BASE_REMOVE_HAND_KEY
    out_root.attrs["base_remove_hand"] = "present"
    out_root.attrs["camera_views"] = "base_0"
    write_latent_cache_identity_attrs(out_root, fm_cfg)
    write_token_mode_attrs(out_root, token_mode)
    data_group = out_root.create_group("data")
    out_root.create_group("meta")

    chunk_bsz = max(1, min(batch_size, 64))
    frame_arr = None
    n_tok = token_mode_num_tokens(token_mode)
    num_batches = (total_frames + batch_size - 1) // batch_size
    print(
        f"[precompute] encoding remove-hand frames: T_rh={total_frames}, "
        f"views=['base_0'], model={model_name}, token_mode={token_mode}, "
        f"tokens={n_tok}, batch_size={batch_size}, read_workers={num_workers}, "
        f"prefetch_batches={prefetch_batches}, gpus={gpu_count}, "
        f"device={primary_device}, out={output_path}"
    )

    batches = _iter_prefetched_batches(
        source.remove_hand,
        total_frames=total_frames,
        batch_size=batch_size,
        image_size=source.image_size,
        image_as_uint8=source.image_as_uint8,
        num_workers=num_workers,
        prefetch_batches=prefetch_batches,
    )
    for start_idx, image_batch in tqdm(
        batches,
        total=num_batches,
        desc="precompute:remove_hand_backbone_feat",
        unit="batch",
    ):
        img = _encode_image_batch(
            extractor,
            image_batch,
            primary_device=primary_device,
            use_data_parallel=use_data_parallel,
            token_mode=token_mode,
        )
        if frame_arr is None:
            frame_arr = data_group.create_array(
                "frame_image_backbone_feat",
                shape=(total_frames,) + img.shape[1:],
                chunks=(chunk_bsz,) + img.shape[1:],
                dtype="f4",
            )
            out_root.attrs["image_backbone_dim"] = int(img.shape[-1])
            out_root.attrs["n_image_views"] = int(img.shape[1])
            if token_mode == TOKEN_MODE_ALL:
                out_root.attrs["image_num_tokens"] = int(img.shape[2])
        frame_arr[start_idx : start_idx + len(img)] = img

    if frame_arr is None:
        raise RuntimeError("no remove-hand frames were encoded")
    print(f"[precompute] saved remove-hand backbone cache: {output_path}")
    print(
        f"[precompute] remove-hand frame_image_backbone_feat "
        f"shape={tuple(frame_arr.shape)}, token_mode={token_mode}"
    )
    return output_path


def precompute_all(cfg: dict, *, force: bool = False) -> dict[str, str | None]:
    source = build_frame_source(cfg)
    main_path = precompute_image_latents(cfg, force=force, source=source)
    rh_path = precompute_base_remove_hand_latents(cfg, force=force, source=source)
    return {"frame_backbone": main_path, "frame_backbone_base_remove_hand": rh_path}


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Precompute frozen DINOv2 features from streaming Zarr frames."
    )
    parser.add_argument("--config", type=str, default="configs/train/config.yaml")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild even when an identity-matching frame cache already exists.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Override precompute.num_workers (parallel Zarr readers).",
    )
    parser.add_argument(
        "--prefetch-batches",
        type=int,
        default=None,
        help="Override precompute.prefetch_batches (bounded queued batches).",
    )
    gpu_group = parser.add_mutually_exclusive_group()
    gpu_group.add_argument(
        "--multi-gpu",
        dest="multi_gpu",
        action="store_true",
        default=None,
        help="Use every visible CUDA device through DataParallel.",
    )
    gpu_group.add_argument(
        "--single-gpu",
        dest="multi_gpu",
        action="store_false",
        help="Use only the first visible CUDA device.",
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
    pre_cfg = dict(cfg.get("precompute") or {})
    if args.workers is not None:
        if args.workers < 0:
            raise ValueError("--workers must be >= 0")
        pre_cfg["num_workers"] = int(args.workers)
    if args.prefetch_batches is not None:
        if args.prefetch_batches < 1:
            raise ValueError("--prefetch-batches must be >= 1")
        pre_cfg["prefetch_batches"] = int(args.prefetch_batches)
    if args.multi_gpu is not None:
        pre_cfg["multi_gpu"] = bool(args.multi_gpu)
    cfg["precompute"] = pre_cfg

    paths = precompute_all(cfg, force=bool(args.force))
    print(f"[precompute] done: {paths}")


if __name__ == "__main__":
    main()
