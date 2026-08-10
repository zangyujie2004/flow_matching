from __future__ import annotations

import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import yaml
import zarr
from tqdm import tqdm

_POLICY_ROOT = Path(__file__).resolve().parents[1]
if str(_POLICY_ROOT) not in sys.path:
    sys.path.insert(0, str(_POLICY_ROOT))

os.environ.setdefault("HF_HUB_OFFLINE", "1")

from datasets import ZarrDataset  # noqa: E402
from datasets.zarr_dataset import (  # noqa: E402
    camera_channel_indices,
    resolve_camera_views,
)
from models.fm.encoders.dino_v2 import DinoV2SmallEncoder, resolve_dino_model_name  # noqa: E402
from tools.latent_cache import (  # noqa: E402
    CACHE_MODE_CLS_LOCAL_NPY,
    CAMERA_BASE_REMOVE_HAND_KEY,
    FRAME_CACHE_VERSION,
    LOCAL_FEATURE_INDEX_KEY,
    LOCAL_FEATURE_NUM_TOKENS,
    LOCAL_FEATURE_STORAGE,
    TOKEN_MODE_ALL,
    TOKEN_MODE_CLS,
    CacheMode,
    apply_resolved_latent_cache_root_dir,
    frame_cache_matches,
    local_feature_file_path,
    normalize_cache_mode,
    remove_hand_frame_cache_matches,
    resolve_frame_backbone_base_remove_hand_zarr_path,
    resolve_frame_backbone_zarr_path,
    resolve_frame_local_feature_base_remove_hand_dir,
    resolve_frame_local_feature_dir,
    stored_token_mode,
    token_mode_num_tokens,
    write_latent_cache_identity_attrs,
    write_token_mode_attrs,
)
from utils.train_utils import cfg_get, load_config  # noqa: E402


class LazyFrameDataset:
    """Minimal, lazy Zarr reader used by distributed feature precompute.

    Unlike the policy dataset, this reader does not preload the full camera
    array into every rank's RAM. Each worker reads only its contiguous shard.
    """

    def __init__(self, data_cfg: dict[str, Any]) -> None:
        self.root_dir = str(data_cfg["root_dir"])
        self.camera_key = str(data_cfg.get("camera_key", "camera"))
        self.state_key = str(data_cfg.get("state_key", "state_30hz"))
        self.image_size = int(data_cfg.get("image_size", 224))
        self.image_as_uint8 = bool(data_cfg.get("image_as_uint8", True))
        self.zarr_path = ZarrDataset._resolve_zarr_path(self.root_dir)
        self.zarr_root = zarr.open_group(self.zarr_path, mode="r")
        self.data_group = self.zarr_root["data"]
        if self.camera_key not in self.data_group:
            raise KeyError(f"Missing data/{self.camera_key} in {self.zarr_path}")
        if self.state_key not in self.data_group:
            raise KeyError(f"Missing data/{self.state_key} in {self.zarr_path}")
        channels = int(self.data_group[self.camera_key].shape[-1])
        if channels % 3 != 0:
            raise ValueError(f"camera channels must be divisible by 3, got {channels}")
        self.camera_views = resolve_camera_views(None, n_zarr_views=channels // 3)
        self._camera_channel_indices = camera_channel_indices(self.camera_views)

    def get_camera(self, t0: int, t1: int) -> np.ndarray:
        camera = np.asarray(self.data_group[self.camera_key][int(t0) : int(t1)])
        if len(self._camera_channel_indices) != camera.shape[-1]:
            camera = np.asarray(camera[..., self._camera_channel_indices])
        return camera

    def _process_image(self, img: np.ndarray) -> torch.Tensor:
        arr = np.asarray(img)
        single_frame = arr.ndim == 3
        if single_frame:
            arr = arr[None, ...]
        if arr.ndim != 4 or arr.shape[-1] % 3 != 0:
            raise ValueError(f"Unsupported image shape: {arr.shape}")
        n_views = arr.shape[-1] // 3
        images = (
            torch.from_numpy(arr)
            .reshape(arr.shape[0], arr.shape[1], arr.shape[2], n_views, 3)
            .permute(0, 3, 4, 1, 2)
            .contiguous()
        )
        if images.shape[-2:] != (self.image_size, self.image_size):
            flat = F.interpolate(
                images.flatten(0, 1).float(),
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            )
            if self.image_as_uint8:
                flat = flat.round().clamp_(0.0, 255.0).to(torch.uint8)
            else:
                flat = flat.div_(255.0).mul_(2.0).sub_(1.0)
            images = flat.reshape(
                arr.shape[0], n_views, 3, self.image_size, self.image_size
            )
        elif not self.image_as_uint8:
            images = images.float().div_(255.0).mul_(2.0).sub_(1.0)
        return images[0] if single_frame else images


def build_dataset(cfg: dict) -> LazyFrameDataset:
    """Build a lazy all-view frame source for single- or multi-GPU encoding."""
    return LazyFrameDataset(dict(cfg["data"]))


def _dist_info() -> tuple[int, int]:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return 0, 1


def _barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def _broadcast_bool(value: bool) -> bool:
    rank, world_size = _dist_info()
    if world_size == 1:
        return bool(value)
    payload = [bool(value) if rank == 0 else False]
    dist.broadcast_object_list(payload, src=0)
    return bool(payload[0])


def _resolve_device(pre_cfg: dict[str, Any]) -> torch.device:
    rank, world_size = _dist_info()
    configured = str(pre_cfg.get("device", "cuda"))
    if world_size > 1 and configured.startswith("cuda"):
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        torch.cuda.set_device(local_rank)
        return torch.device("cuda", local_rank)
    return torch.device(configured)


def _frame_count(dataset: Any, key: str) -> int:
    if hasattr(dataset, "data_group") and key in dataset.data_group:
        return int(dataset.data_group[key].shape[0])
    return int(dataset.ram_data[key].shape[0])


def _frame_partition(total_frames: int, rank: int, world_size: int) -> tuple[int, int]:
    if total_frames < world_size:
        raise ValueError(
            f"Cannot split {total_frames} frames across {world_size} ranks"
        )
    base, remainder = divmod(int(total_frames), int(world_size))
    start = rank * base + min(rank, remainder)
    end = start + base + (1 if rank < remainder else 0)
    return start, end


def _shard_dir(output_path: str) -> str:
    return f"{output_path}.precompute_shards"


def _shard_path(output_path: str, rank: int) -> str:
    return os.path.join(_shard_dir(output_path), f"rank_{rank:05d}.npy")


def _close_memmap(array: Any) -> None:
    """Close an np.memmap explicitly before deleting it on a shared filesystem."""
    mmap = getattr(array, "_mmap", None)
    if mmap is not None and not mmap.closed:
        mmap.close()


def _remove_tree_best_effort(path: str, *, attempts: int = 5) -> bool:
    """Remove temporary output without failing an otherwise complete cache.

    CPFS/NFS may briefly retain an unlinked open memmap as a hidden file. Retry
    metadata cleanup, but never abort distributed workers after the final Zarr
    has already been atomically committed.
    """
    for attempt in range(max(1, int(attempts))):
        try:
            shutil.rmtree(path)
            return True
        except FileNotFoundError:
            return True
        except OSError as exc:
            if attempt + 1 >= attempts:
                print(f"[precompute] warning: could not remove temporary path {path}: {exc}")
                return False
            time.sleep(0.25 * (attempt + 1))
    return False


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


def resolve_token_mode_from_cfg(cfg: dict) -> CacheMode:
    pre_cfg = dict(cfg.get("precompute") or {})
    # Default cls for faster iteration when unset; existing all-token runs should set token_mode: all.
    return normalize_cache_mode(pre_cfg.get("token_mode"), default=TOKEN_MODE_CLS)


def _local_feature_options(pre_cfg: dict) -> tuple[str, int]:
    local_cfg = dict(pre_cfg.get("local_feature") or {})
    dtype = str(local_cfg.get("dtype", "float16")).strip().lower()
    if dtype not in {"float16", "fp16", "f2"}:
        raise ValueError(
            "precompute.local_feature.dtype currently supports only float16"
        )
    frames_per_directory = max(
        1, int(local_cfg.get("frames_per_directory", 1000))
    )
    return "float16", frames_per_directory


def _write_frame_local_feature(
    local_root: str,
    feature_id: int,
    feature: np.ndarray,
    *,
    frames_per_directory: int,
) -> None:
    """Atomically write one [V,256,D] FP16 array."""
    path = local_feature_file_path(
        local_root,
        feature_id,
        frames_per_directory=frames_per_directory,
    )
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "wb") as handle:
        np.save(
            handle,
            np.ascontiguousarray(feature, dtype=np.float16),
            allow_pickle=False,
        )
    os.replace(tmp_path, path)


def _split_cls_and_local_tokens(
    tokens: torch.Tensor,
    *,
    batch_size: int,
    num_views: int,
) -> tuple[np.ndarray, np.ndarray]:
    if tokens.ndim != 3 or int(tokens.shape[1]) != LOCAL_FEATURE_NUM_TOKENS + 1:
        raise ValueError(f"expected tokens (B*V,257,D), got {tuple(tokens.shape)}")
    shaped = tokens.reshape(batch_size, num_views, tokens.shape[1], tokens.shape[2])
    cls = shaped[:, :, 0, :].detach().cpu().numpy().astype(np.float32, copy=False)
    local = shaped[:, :, 1:, :].detach().cpu().numpy().astype(np.float16, copy=False)
    return cls, local


def _contiguous_bounds(frame_indices: list[int]) -> tuple[int, int]:
    if not frame_indices:
        raise ValueError("frame_indices must be non-empty")
    start = int(frame_indices[0])
    end = int(frame_indices[-1]) + 1
    if frame_indices != list(range(start, end)):
        raise ValueError("precompute frame batches must be contiguous")
    return start, end


def build_frame_image_batch(dataset: Any, frame_indices: list[int]) -> torch.Tensor:
    start, end = _contiguous_bounds(frame_indices)
    processed = dataset._process_image(dataset.get_camera(start, end))
    if processed.ndim != 5:
        raise ValueError(f"expected processed frame image (B,V,3,H,W), got {processed.shape}")
    return processed


def build_remove_hand_image_batch(
    dataset: Any,
    rh_frames: Any,
    frame_indices: list[int],
) -> torch.Tensor:
    """Encode compact base_0 remove-hand RGB (T,H,W,3) as single-view batches."""
    start, end = _contiguous_bounds(frame_indices)
    camera = np.asarray(rh_frames[start:end])
    if camera.ndim != 4 or camera.shape[-1] != 3:
        raise ValueError(f"expected remove-hand (B,H,W,3), got {camera.shape}")
    processed = dataset._process_image(camera)
    if processed.ndim != 5 or processed.shape[1] != 1:
        raise ValueError(f"expected processed (B,1,3,H,W), got {processed.shape}")
    return processed


def _tokens_to_stored_feat(
    tokens: torch.Tensor,
    *,
    batch_size: int,
    num_views: int,
    token_mode: str,
) -> np.ndarray:
    """tokens: (B*V, 257, D) → stored array for token_mode."""
    if tokens.ndim != 3:
        raise ValueError(f"expected tokens (B*V,N,D), got {tokens.shape}")
    if token_mode == TOKEN_MODE_CLS:
        cls = tokens[:, 0, :]  # (B*V, D)
        feat = cls.reshape(batch_size, num_views, cls.shape[-1])
    else:
        feat = tokens.reshape(batch_size, num_views, tokens.shape[1], tokens.shape[2])
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


def _prepare_distributed_output(
    output_path: str,
    *,
    local_root: str,
    local_enabled: bool,
) -> None:
    rank, _ = _dist_info()
    if rank == 0:
        for path in (output_path, f"{output_path}.building", _shard_dir(output_path)):
            if os.path.isdir(path):
                print(f"[precompute] removing stale output: {path}")
                shutil.rmtree(path)
        if local_enabled and os.path.isdir(local_root):
            print(f"[precompute] removing stale local features: {local_root}")
            shutil.rmtree(local_root)
        os.makedirs(_shard_dir(output_path), exist_ok=True)
        if local_enabled:
            os.makedirs(local_root, exist_ok=True)
    _barrier()


def _encode_rank_shard(
    *,
    output_path: str,
    total_frames: int,
    batch_size: int,
    device: torch.device,
    image_encoder: Any,
    image_batch_builder: Callable[[list[int]], torch.Tensor],
    token_mode: str,
    local_enabled: bool,
    local_root: str,
    frames_per_directory: int,
    description: str,
) -> None:
    rank, world_size = _dist_info()
    shard_start, shard_end = _frame_partition(total_frames, rank, world_size)
    shard_size = shard_end - shard_start
    shard_array: np.memmap | None = None

    iterator = tqdm(
        range(shard_start, shard_end, batch_size),
        desc=f"{description}:rank{rank}",
        unit="batch",
        disable=rank != 0,
    )
    for start_idx in iterator:
        frame_indices = list(range(start_idx, min(start_idx + batch_size, shard_end)))
        image_batch = image_batch_builder(frame_indices).to(device, non_blocking=True)
        with torch.inference_mode():
            bsz, num_views = image_batch.shape[:2]
            flat = image_batch.reshape(bsz * num_views, *image_batch.shape[2:])
            tokens = image_encoder.extract_backbone_feat(flat)
            if local_enabled:
                img, local_features = _split_cls_and_local_tokens(
                    tokens,
                    batch_size=bsz,
                    num_views=num_views,
                )
            else:
                img = _tokens_to_stored_feat(
                    tokens,
                    batch_size=bsz,
                    num_views=num_views,
                    token_mode=token_mode,
                )

        if shard_array is None:
            shard_array = np.lib.format.open_memmap(
                _shard_path(output_path, rank),
                mode="w+",
                dtype=np.float32,
                shape=(shard_size,) + img.shape[1:],
            )
        local_start = start_idx - shard_start
        shard_array[local_start : local_start + len(frame_indices)] = img
        if local_enabled:
            for batch_idx, feature_id in enumerate(frame_indices):
                _write_frame_local_feature(
                    local_root,
                    feature_id,
                    local_features[batch_idx],
                    frames_per_directory=frames_per_directory,
                )
        del image_batch, tokens

    if shard_array is None:
        raise RuntimeError(f"rank {rank} encoded no frames")
    shard_array.flush()
    _close_memmap(shard_array)
    del shard_array
    print(
        f"[precompute] rank {rank}/{world_size} complete: "
        f"frames=[{shard_start},{shard_end})"
    )


def _merge_rank_shards(
    *,
    output_path: str,
    total_frames: int,
    batch_size: int,
    dataset: Any,
    fm_cfg: dict[str, Any],
    cache_mode: str,
    token_mode: str,
    local_enabled: bool,
    local_root: str,
    frames_per_directory: int,
    camera_views: tuple[str, ...],
    selection: str,
    extra_attrs: dict[str, Any] | None = None,
) -> tuple[int, ...] | None:
    rank, world_size = _dist_info()
    if rank != 0:
        return None

    first = np.load(_shard_path(output_path, 0), mmap_mode="r", allow_pickle=False)
    feature_shape = tuple(int(value) for value in first.shape[1:])
    if not feature_shape:
        raise RuntimeError("encoded shard has no feature dimensions")
    _close_memmap(first)
    del first
    building_path = f"{output_path}.building"
    out_root = zarr.open_group(building_path, mode="w")
    out_root.attrs["cache_version"] = int(FRAME_CACHE_VERSION)
    out_root.attrs["source_zarr_path"] = dataset.zarr_path
    out_root.attrs["image_size"] = int(dataset.image_size)
    out_root.attrs["color_order"] = "rgb"
    out_root.attrs["frame_image_selection"] = selection
    out_root.attrs["camera_views"] = ",".join(camera_views)
    out_root.attrs["distributed_world_size"] = int(world_size)
    write_latent_cache_identity_attrs(out_root, fm_cfg)
    write_token_mode_attrs(out_root, token_mode)
    out_root.attrs["cache_mode"] = cache_mode
    for key, value in (extra_attrs or {}).items():
        out_root.attrs[key] = value

    data_group = out_root.create_group("data")
    out_root.create_group("meta")
    chunk_bsz = max(1, min(batch_size, 64))
    frame_arr = data_group.create_array(
        "frame_image_backbone_feat",
        shape=(total_frames,) + feature_shape,
        chunks=(chunk_bsz,) + feature_shape,
        dtype="f4",
    )
    out_root.attrs["image_backbone_dim"] = int(feature_shape[-1])
    out_root.attrs["n_image_views"] = int(feature_shape[0])
    if token_mode == TOKEN_MODE_ALL:
        out_root.attrs["image_num_tokens"] = int(feature_shape[1])

    for shard_rank in range(world_size):
        start, end = _frame_partition(total_frames, shard_rank, world_size)
        shard = np.load(
            _shard_path(output_path, shard_rank),
            mmap_mode="r",
            allow_pickle=False,
        )
        expected_shape = (end - start,) + feature_shape
        if tuple(int(value) for value in shard.shape) != expected_shape:
            _close_memmap(shard)
            raise ValueError(
                f"rank {shard_rank} shard shape {shard.shape} != {expected_shape}"
            )
        frame_arr[start:end] = shard
        _close_memmap(shard)
        del shard

    if local_enabled:
        out_root.attrs["local_feature_storage"] = LOCAL_FEATURE_STORAGE
        out_root.attrs["local_feature_dir"] = os.path.basename(local_root)
        out_root.attrs["local_feature_dtype"] = "float16"
        out_root.attrs["local_feature_num_tokens"] = int(LOCAL_FEATURE_NUM_TOKENS)
        out_root.attrs["local_feature_layout"] = "V,N,D"
        out_root.attrs["local_feature_frames_per_directory"] = int(
            frames_per_directory
        )
        out_root.attrs["local_feature_shape"] = [
            int(feature_shape[0]),
            int(LOCAL_FEATURE_NUM_TOKENS),
            int(feature_shape[-1]),
        ]
        local_index = data_group.create_array(
            LOCAL_FEATURE_INDEX_KEY,
            shape=(total_frames,),
            chunks=(min(total_frames, max(chunk_bsz, 4096)),),
            dtype="i8",
        )
        local_index[:] = np.arange(total_frames, dtype=np.int64)
        out_root.attrs["local_feature_complete"] = True

    del frame_arr, out_root
    os.replace(building_path, output_path)
    _remove_tree_best_effort(_shard_dir(output_path))
    return (total_frames,) + feature_shape


def _run_distributed_precompute(
    *,
    output_path: str,
    total_frames: int,
    batch_size: int,
    device: torch.device,
    dataset: Any,
    fm_cfg: dict[str, Any],
    cache_mode: str,
    token_mode: str,
    local_enabled: bool,
    local_root: str,
    frames_per_directory: int,
    camera_views: tuple[str, ...],
    selection: str,
    image_batch_builder: Callable[[list[int]], torch.Tensor],
    description: str,
    extra_attrs: dict[str, Any] | None = None,
) -> tuple[int, ...] | None:
    rank, world_size = _dist_info()
    image_encoder = _build_encoder(fm_cfg, device)
    if rank == 0:
        n_tokens = (
            LOCAL_FEATURE_NUM_TOKENS + 1
            if local_enabled
            else token_mode_num_tokens(token_mode)
        )
        print(
            f"[precompute] {description}: frames={total_frames}, "
            f"views={list(camera_views)}, ranks={world_size}, "
            f"batch_size_per_rank={batch_size}, tokens={n_tokens}, device={device}"
        )
    _encode_rank_shard(
        output_path=output_path,
        total_frames=total_frames,
        batch_size=batch_size,
        device=device,
        image_encoder=image_encoder,
        image_batch_builder=image_batch_builder,
        token_mode=token_mode,
        local_enabled=local_enabled,
        local_root=local_root,
        frames_per_directory=frames_per_directory,
        description=description,
    )
    _barrier()
    shape = _merge_rank_shards(
        output_path=output_path,
        total_frames=total_frames,
        batch_size=batch_size,
        dataset=dataset,
        fm_cfg=fm_cfg,
        cache_mode=cache_mode,
        token_mode=token_mode,
        local_enabled=local_enabled,
        local_root=local_root,
        frames_per_directory=frames_per_directory,
        camera_views=camera_views,
        selection=selection,
        extra_attrs=extra_attrs,
    )
    _barrier()
    return shape


def precompute_image_latents(
    cfg: dict,
    *,
    force: bool = False,
    dataset: Any | None = None,
) -> str:
    """Encode all camera frames, sharded across initialized distributed ranks."""
    cfg = apply_resolved_latent_cache_root_dir(dict(cfg))
    pre_cfg = dict(cfg.get("precompute", {}))
    output_path = resolve_output_path_from_cfg(cfg, pre_cfg.get("output_path"))
    force = bool(force) or bool(pre_cfg.get("overwrite", False))
    cache_mode = resolve_token_mode_from_cfg(cfg)
    token_mode = stored_token_mode(cache_mode)
    local_enabled = cache_mode == CACHE_MODE_CLS_LOCAL_NPY
    _local_dtype, frames_per_directory = _local_feature_options(pre_cfg)
    local_root = resolve_frame_local_feature_dir(os.path.dirname(output_path))
    batch_size = max(1, int(pre_cfg.get("batch_size", 256)))
    fm_cfg = dict(cfg["models"]["fm"])
    if not bool(fm_cfg.get("freeze_image_encoder", True)):
        raise ValueError("Precompute requires models.fm.freeze_image_encoder=true.")
    if dataset is None:
        dataset = build_dataset(cfg)
    total_frames = _frame_count(dataset, dataset.camera_key)
    state_frames = _frame_count(dataset, dataset.state_key)
    if total_frames != state_frames:
        raise ValueError(
            f"camera/state frame mismatch: {total_frames} != {state_frames}"
        )

    rank, _ = _dist_info()
    valid = False
    if rank == 0 and not force:
        valid = frame_cache_matches(
            output_path,
            fm_cfg=fm_cfg,
            source_zarr_path=dataset.zarr_path,
            image_size=int(dataset.image_size),
            camera_views=dataset.camera_views,
            total_frames=total_frames,
            token_mode=cache_mode,
            color_order="rgb",
        )
    if _broadcast_bool(valid):
        if rank == 0:
            print(f"[precompute] valid frame cache, skipping: {output_path}")
            _remove_tree_best_effort(_shard_dir(output_path))
        return output_path

    _prepare_distributed_output(
        output_path,
        local_root=local_root,
        local_enabled=local_enabled,
    )
    shape = _run_distributed_precompute(
        output_path=output_path,
        total_frames=total_frames,
        batch_size=batch_size,
        device=_resolve_device(pre_cfg),
        dataset=dataset,
        fm_cfg=fm_cfg,
        cache_mode=cache_mode,
        token_mode=token_mode,
        local_enabled=local_enabled,
        local_root=local_root,
        frames_per_directory=frames_per_directory,
        camera_views=tuple(dataset.camera_views),
        selection="all_frames",
        image_batch_builder=lambda indices: build_frame_image_batch(dataset, indices),
        description="all_frames",
    )
    if rank == 0:
        print(f"[precompute] saved {output_path}, shape={shape}")
    return output_path


def precompute_base_remove_hand_latents(
    cfg: dict,
    *,
    force: bool = False,
    dataset: Any | None = None,
) -> str | None:
    """Encode compact remove-hand base frames across distributed ranks."""
    cfg = apply_resolved_latent_cache_root_dir(dict(cfg))
    pre_cfg = dict(cfg.get("precompute", {}))
    output_path = resolve_remove_hand_output_path(cfg)
    force = bool(force) or bool(pre_cfg.get("overwrite", False))
    cache_mode = resolve_token_mode_from_cfg(cfg)
    token_mode = stored_token_mode(cache_mode)
    local_enabled = cache_mode == CACHE_MODE_CLS_LOCAL_NPY
    _local_dtype, frames_per_directory = _local_feature_options(pre_cfg)
    local_root = resolve_frame_local_feature_base_remove_hand_dir(
        os.path.dirname(output_path)
    )
    batch_size = max(1, int(pre_cfg.get("batch_size", 256)))
    fm_cfg = dict(cfg["models"]["fm"])
    if dataset is None:
        dataset = build_dataset(cfg)
    if CAMERA_BASE_REMOVE_HAND_KEY not in dataset.data_group:
        if _dist_info()[0] == 0:
            print(f"[precompute] no data/{CAMERA_BASE_REMOVE_HAND_KEY}; skipping")
        return None
    rh_frames = dataset.data_group[CAMERA_BASE_REMOVE_HAND_KEY]
    if len(rh_frames.shape) != 4 or int(rh_frames.shape[-1]) != 3:
        raise ValueError(
            f"{CAMERA_BASE_REMOVE_HAND_KEY} expected (T,H,W,3), got {rh_frames.shape}"
        )
    total_frames = int(rh_frames.shape[0])
    if total_frames == 0:
        return None

    rank, _ = _dist_info()
    valid = False
    if rank == 0 and not force:
        valid = remove_hand_frame_cache_matches(
            output_path,
            fm_cfg=fm_cfg,
            source_zarr_path=dataset.zarr_path,
            image_size=int(dataset.image_size),
            total_frames=total_frames,
            token_mode=cache_mode,
            color_order="rgb",
        )
    if _broadcast_bool(valid):
        if rank == 0:
            print(f"[precompute] valid remove-hand cache, skipping: {output_path}")
            _remove_tree_best_effort(_shard_dir(output_path))
        return output_path

    _prepare_distributed_output(
        output_path,
        local_root=local_root,
        local_enabled=local_enabled,
    )
    shape = _run_distributed_precompute(
        output_path=output_path,
        total_frames=total_frames,
        batch_size=batch_size,
        device=_resolve_device(pre_cfg),
        dataset=dataset,
        fm_cfg=fm_cfg,
        cache_mode=cache_mode,
        token_mode=token_mode,
        local_enabled=local_enabled,
        local_root=local_root,
        frames_per_directory=frames_per_directory,
        camera_views=("base_0",),
        selection=CAMERA_BASE_REMOVE_HAND_KEY,
        image_batch_builder=lambda indices: build_remove_hand_image_batch(
            dataset, rh_frames, indices
        ),
        description="remove_hand",
        extra_attrs={
            "compact": True,
            "ties_to": CAMERA_BASE_REMOVE_HAND_KEY,
            "base_remove_hand": "present",
        },
    )
    if rank == 0:
        print(f"[precompute] saved {output_path}, shape={shape}")
    return output_path


def precompute_all(cfg: dict, *, force: bool = False) -> dict[str, str | None]:
    """Run train and optional external-eval caches with the same layout."""
    dataset = build_dataset(cfg)
    main_path = precompute_image_latents(cfg, force=force, dataset=dataset)
    rh_path = precompute_base_remove_hand_latents(cfg, force=force, dataset=dataset)
    paths: dict[str, str | None] = {
        "frame_backbone": main_path,
        "frame_backbone_base_remove_hand": rh_path,
    }

    pre_cfg = dict(cfg.get("precompute") or {})
    eval_overlay = dict(cfg.get("eval_data") or {})
    if bool(pre_cfg.get("include_eval_data", False)) and eval_overlay:
        eval_overlay.pop("episode_groups", None)
        eval_cfg = dict(cfg)
        eval_data = dict(cfg["data"])
        eval_data.update(eval_overlay)
        eval_cfg["data"] = eval_data
        eval_precompute = dict(pre_cfg)
        eval_precompute["output_path"] = pre_cfg.get("eval_output_path")
        eval_cfg["precompute"] = eval_precompute
        eval_cfg = apply_resolved_latent_cache_root_dir(eval_cfg)
        eval_dataset = build_dataset(eval_cfg)
        paths["eval_frame_backbone"] = precompute_image_latents(
            eval_cfg, force=force, dataset=eval_dataset
        )
        paths["eval_frame_backbone_base_remove_hand"] = (
            precompute_base_remove_hand_latents(
                eval_cfg, force=force, dataset=eval_dataset
            )
        )
    return paths


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
    initialized_here = False
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        configured_device = str((cfg.get("precompute") or {}).get("device", "cuda"))
        use_cuda = configured_device.startswith("cuda")
        if use_cuda:
            if not torch.cuda.is_available():
                raise RuntimeError("Distributed CUDA precompute requested but CUDA is unavailable")
            torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl" if use_cuda else "gloo")
        initialized_here = True

    try:
        paths = precompute_all(cfg, force=bool(args.force))
        if _dist_info()[0] == 0:
            print(f"[precompute] done: {paths}")
    finally:
        if initialized_here and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
