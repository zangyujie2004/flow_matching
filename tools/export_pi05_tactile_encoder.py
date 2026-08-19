from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file

from models.fm.encoders.tactile_cnn import TactileCNNEncoder


ENCODER_MARKER = "condition_encoder.tactile_encoder."


def file_sha256(path: str | Path, *, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def _find_model_state(checkpoint: dict[str, Any]) -> dict[str, torch.Tensor]:
    for key in ("model", "model_state_dict", "policy", "policy_state_dict", "state_dict"):
        value = checkpoint.get(key)
        if isinstance(value, dict) and value and all(isinstance(k, str) for k in value):
            return value
    if checkpoint and all(isinstance(v, torch.Tensor) for v in checkpoint.values()):
        return checkpoint
    raise KeyError("checkpoint does not contain a recognizable model state dict")


def extract_tactile_encoder_state(checkpoint: dict[str, Any]) -> dict[str, torch.Tensor]:
    state = _find_model_state(checkpoint)
    exported: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        marker_pos = key.find(ENCODER_MARKER)
        if marker_pos < 0:
            continue
        exported[key[marker_pos + len(ENCODER_MARKER) :]] = value.detach().cpu().contiguous()
    if not exported:
        raise KeyError(f"checkpoint contains no keys matching '*{ENCODER_MARKER}*'")
    return exported


def _normalizer_candidates(checkpoint: dict[str, Any]):
    for key in (
        "normalizer",
        "normalizer_state",
        "normalizer_state_dict",
        "dataset_normalizer",
        "dataset_normalizer_state",
    ):
        value = checkpoint.get(key)
        if isinstance(value, dict):
            yield value


def extract_tactile_normalizer_state(checkpoint: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    for candidate in _normalizer_candidates(checkpoint):
        tactile = candidate.get("tactile")
        if isinstance(tactile, dict) and "scale" in tactile and "offset" in tactile:
            scale = torch.as_tensor(tactile["scale"], dtype=torch.float32).flatten()
            offset = torch.as_tensor(tactile["offset"], dtype=torch.float32).flatten()
            if scale.numel() == 12 and offset.numel() == 12:
                return scale, offset
    raise KeyError("checkpoint does not contain a 12-D tactile normalizer scale/offset")


def validate_encoder_state(state: dict[str, torch.Tensor]) -> TactileCNNEncoder:
    encoder = TactileCNNEncoder(
        in_channels=12,
        hidden_dim=64,
        out_dim=256,
        temporal_pool="conv1d",
        dropout=0.1,
    )
    missing, unexpected = encoder.load_state_dict(state, strict=True)
    if missing or unexpected:
        raise RuntimeError(f"encoder state mismatch: missing={missing}, unexpected={unexpected}")
    return encoder.eval()


def export_pi05_tactile_encoder_bundle(checkpoint_path: str | Path, output_path: str | Path) -> Path:
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    output_path = Path(output_path).expanduser().resolve()
    if output_path.suffix != ".safetensors":
        raise ValueError("output_path must end with .safetensors")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    if output_path.exists() or output_path.with_suffix(".json").exists():
        raise FileExistsError(f"refusing to overwrite existing bundle: {output_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    encoder_state = extract_tactile_encoder_state(checkpoint)
    encoder = validate_encoder_state(encoder_state)
    scale, offset = extract_tactile_normalizer_state(checkpoint)

    payload = dict(encoder_state)
    payload["tactile_normalizer_scale"] = scale.contiguous()
    payload["tactile_normalizer_offset"] = offset.contiguous()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(payload, output_path)

    with torch.inference_mode():
        probe = torch.linspace(-1, 1, 2 * 8 * 35 * 20 * 12, dtype=torch.float32).reshape(
            2, 8, 35, 20, 12
        )
        probe_digest = hashlib.sha256(encoder(probe).numpy().tobytes()).hexdigest()
    metadata = {
        "schema": "pi05_tactile_encoder_bundle/v1",
        "source_checkpoint": str(checkpoint_path),
        "source_checkpoint_sha256": file_sha256(checkpoint_path),
        "architecture": "TactileCNNEncoder",
        "input_shape": [8, 35, 20, 12],
        "output_dim": 256,
        "temporal_pool": "conv1d",
        "normalization": "x * scale + offset",
        "probe_output_sha256": probe_digest,
        "state_keys": sorted(encoder_state),
    }
    with output_path.with_suffix(".json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Export the trained FM tactile condition encoder for PI0.5")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = export_pi05_tactile_encoder_bundle(args.checkpoint, args.output)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
