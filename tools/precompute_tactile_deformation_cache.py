#!/usr/bin/env python3
"""Create a shared float32 mmap cache for deformation-only tactile frames."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import zarr
from tqdm import tqdm

_POLICY_ROOT = Path(__file__).resolve().parents[1]
if str(_POLICY_ROOT) not in sys.path:
    sys.path.insert(0, str(_POLICY_ROOT))

from tools.tactile_feat import TACTILE_FEATURE_DIM, extract_tactile_deformation


def resolve_replay_buffer_path(root: Path) -> Path:
    root = root.expanduser().resolve()
    if root.name == "replay_buffer.zarr" and root.is_dir():
        return root
    candidate = root / "replay_buffer.zarr"
    if candidate.is_dir():
        return candidate
    raise FileNotFoundError(f"replay_buffer.zarr not found below {root}")


def default_output_path(replay_path: Path) -> Path:
    return replay_path.parent / "tactile_deformation_f32.npy"


def build_cache(
    replay_path: Path,
    output_path: Path,
    *,
    batch_frames: int,
    overwrite: bool,
) -> None:
    root = zarr.open_group(str(replay_path), mode="r")
    if "data" not in root or "tactile" not in root["data"]:
        raise KeyError(f"missing data/tactile in {replay_path}")
    if "meta" not in root or "episode_ends" not in root["meta"]:
        raise KeyError(f"missing meta/episode_ends in {replay_path}")

    tactile = root["data"]["tactile"]
    episode_ends = np.asarray(root["meta"]["episode_ends"][:], dtype=np.int64)
    if tactile.ndim != 4 or tactile.shape[-1] != 24:
        raise ValueError(f"expected raw tactile (T,H,W,24), got {tactile.shape}")
    if len(episode_ends) == 0 or int(episode_ends[-1]) != int(tactile.shape[0]):
        raise ValueError(
            f"episode_ends[-1] does not match tactile length: "
            f"{episode_ends[-1] if len(episode_ends) else None} != {tactile.shape[0]}"
        )

    output_path = output_path.expanduser().resolve()
    metadata_path = Path(f"{output_path}.json")
    if (output_path.exists() or metadata_path.exists()) and not overwrite:
        raise FileExistsError(
            f"cache already exists: {output_path}; pass --overwrite to replace it"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = output_path.with_name(f"{output_path.name}.partial")
    partial_metadata_path = Path(f"{partial_path}.json")
    if partial_path.exists():
        partial_path.unlink()
    if partial_metadata_path.exists():
        partial_metadata_path.unlink()

    output_shape = (*tactile.shape[:-1], TACTILE_FEATURE_DIM)
    output = np.lib.format.open_memmap(
        partial_path,
        mode="w+",
        dtype=np.float32,
        shape=output_shape,
    )
    batch_frames = max(1, int(batch_frames))
    for start in tqdm(
        range(0, tactile.shape[0], batch_frames),
        desc="tactile deformation cache",
        unit="batch",
    ):
        stop = min(start + batch_frames, tactile.shape[0])
        raw = np.asarray(tactile[start:stop], dtype=np.float32)
        output[start:stop] = extract_tactile_deformation(raw)
    output.flush()
    del output

    metadata = {
        "format": "flow_matching_tactile_deformation_f32_v1",
        "source_replay_buffer": str(replay_path),
        "source_tactile_shape": [int(value) for value in tactile.shape],
        "output_shape": [int(value) for value in output_shape],
        "dtype": "float32",
        "episode_ends": episode_ends.tolist(),
    }
    with partial_metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
        handle.write("\n")

    os.replace(partial_path, output_path)
    os.replace(partial_metadata_path, metadata_path)
    print(f"[complete] cache={output_path}")
    print(f"[complete] metadata={metadata_path}")
    print(f"[complete] shape={output_shape}, dtype=float32")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        required=True,
        help="Dataset directory or replay_buffer.zarr path.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output .npy path (default: DATA_ROOT/tactile_deformation_f32.npy).",
    )
    parser.add_argument("--batch-frames", type=int, default=256)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    replay_path = resolve_replay_buffer_path(Path(args.data_root))
    output_path = (
        Path(args.output)
        if args.output
        else default_output_path(replay_path)
    )
    build_cache(
        replay_path,
        output_path,
        batch_frames=args.batch_frames,
        overwrite=bool(args.overwrite),
    )


if __name__ == "__main__":
    main()
