from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys

_FLOW_MATCHING_ROOT = Path(__file__).resolve().parents[1]
if str(_FLOW_MATCHING_ROOT) in sys.path:
    sys.path.remove(str(_FLOW_MATCHING_ROOT))
sys.path.insert(0, str(_FLOW_MATCHING_ROOT))

import numpy as np
import torch
import zarr
from tqdm import tqdm

from datasets.tactile_ae_dataset import resolve_replay_buffer_path
from models.fm.encoders import load_tactile_autoencoder_checkpoint
from tools.normalizer import FieldNormalizer
from tools.tactile_feat import extract_tactile_deformation


def file_sha256(path: str | Path, *, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(block_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def create_cache(
    *,
    data_root: str | Path,
    checkpoint_path: str | Path,
    output_path: str | Path,
    tactile_key: str,
    device: str,
    batch_size: int,
    overwrite: bool,
) -> Path:
    source_path = resolve_replay_buffer_path(data_root)
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    output_path = Path(output_path).expanduser().resolve()
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"tactile latent cache already exists: {output_path}; use --overwrite"
        )

    codec, checkpoint = load_tactile_autoencoder_checkpoint(checkpoint_path)
    normalizer_state = checkpoint.get("tactile_normalizer_state")
    if not isinstance(normalizer_state, dict):
        raise KeyError(
            f"AE checkpoint has no tactile_normalizer_state: {checkpoint_path}"
        )
    tactile_normalizer = FieldNormalizer.from_state_dict(normalizer_state)

    torch_device = torch.device(device)
    codec = codec.to(torch_device).eval()
    source = zarr.open_group(str(source_path), mode="r")
    tactile = source["data"][str(tactile_key)]
    num_frames = int(tactile.shape[0])
    latent_dim = int(codec.latent_dim)
    batch_size = max(1, int(batch_size))

    cache = zarr.open_group(str(output_path), mode="w")
    latent_array = cache.create_array(
        "latent",
        shape=(num_frames, latent_dim),
        chunks=(min(batch_size, num_frames), latent_dim),
        dtype="f4",
        overwrite=True,
    )
    episode_ends = np.asarray(source["meta"]["episode_ends"][:], dtype=np.int64)
    cache.create_array(
        "episode_ends",
        data=episode_ends,
        chunks=(min(1024, len(episode_ends)),),
        overwrite=True,
    )

    latent_sum = np.zeros(latent_dim, dtype=np.float64)
    latent_sq_sum = np.zeros(latent_dim, dtype=np.float64)
    count = 0
    with torch.inference_mode():
        for start in tqdm(
            range(0, num_frames, batch_size),
            desc="Encode tactile latents",
        ):
            stop = min(start + batch_size, num_frames)
            raw = np.asarray(tactile[start:stop], dtype=np.float32)
            deformation = extract_tactile_deformation(raw)
            normalized = tactile_normalizer.normalize_np(deformation)
            tensor = torch.from_numpy(normalized.astype(np.float32, copy=False)).to(
                torch_device
            )
            latent = codec.encode_flattened(tensor).float().cpu().numpy()
            latent_array[start:stop] = latent
            latent64 = latent.astype(np.float64, copy=False)
            latent_sum += latent64.sum(axis=0)
            latent_sq_sum += np.square(latent64).sum(axis=0)
            count += len(latent)

    mean = latent_sum / max(count, 1)
    variance = np.maximum(latent_sq_sum / max(count, 1) - np.square(mean), 1e-12)
    std = np.maximum(np.sqrt(variance), 1e-6)

    for start in tqdm(
        range(0, num_frames, batch_size),
        desc="Normalize tactile latents",
    ):
        stop = min(start + batch_size, num_frames)
        values = np.asarray(latent_array[start:stop], dtype=np.float32)
        values = (values - mean.astype(np.float32)) / std.astype(np.float32)
        latent_array[start:stop] = values.astype(np.float32, copy=False)

    cache.attrs.update(
        {
            "format": "flow_matching_tactile_latent_v1",
            "normalized": True,
            "source_zarr": str(source_path),
            "tactile_key": str(tactile_key),
            "ae_checkpoint": str(checkpoint_path),
            "ae_checkpoint_sha256": file_sha256(checkpoint_path),
            "num_frames": num_frames,
            "latent_dim": latent_dim,
            "num_sensors": int(codec.num_sensors),
            "token_dim": int(codec.token_dim),
            "token_layout": "B,16,4_then_flatten",
            "latent_mean": mean.astype(np.float32).tolist(),
            "latent_std": std.astype(np.float32).tolist(),
            "tactile_normalizer_scale": tactile_normalizer.scale.tolist(),
            "tactile_normalizer_offset": tactile_normalizer.offset.tolist(),
        }
    )
    print(
        f"tactile latent cache complete: path={output_path} "
        f"shape=({num_frames},{latent_dim})"
    )
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Encode every tactile frame with a Stage 1 AE checkpoint"
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tactile-key", default="tactile")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    create_cache(
        data_root=args.data_root,
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        tactile_key=args.tactile_key,
        device=args.device,
        batch_size=args.batch_size,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
