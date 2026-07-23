#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CONFIG="configs/train/config.yaml"
GPUS="${CUDA_VISIBLE_DEVICES:-0}"
PYTHON_BIN="${PYTHON:-python}"
FORCE_ARGS=()

usage() {
  cat <<'EOF'
Usage:
  ./scripts/precompute.sh [options]

Options:
  --config PATH   Config yaml (default: configs/train/config.yaml)
  --gpus IDS      CUDA_VISIBLE_DEVICES (default: 0)
  --force         Rebuild even if identity-matching frame cache exists
  -h, --help      Show this help

Writes (scheme A, frame-only):
  {data.latent_cache_root_dir}/frame_backbone.zarr
  {data.latent_cache_root_dir}/frame_backbone_base_remove_hand.zarr  # if dual zarr has remove-hand

precompute.token_mode: cls | all
  cls → (T,V,D) / (T_rh,1,D)
  all → (T,V,257,D) / (T_rh,1,257,D)

Skip rule: existing cache with matching identity + token_mode + full T → skip.
Use --force (or precompute.overwrite=true) to recompute.

Independent of data.window_size / stride / n_image_steps / action_horizon / memory.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG="$2"
      shift 2
      ;;
    --config=*)
      CONFIG="${1#*=}"
      shift
      ;;
    --gpus)
      GPUS="$2"
      shift 2
      ;;
    --gpus=*)
      GPUS="${1#*=}"
      shift
      ;;
    --force)
      FORCE_ARGS+=(--force)
      shift
      ;;
    -h|--help|help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

export CUDA_VISIBLE_DEVICES="$GPUS"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
exec "$PYTHON_BIN" tools/precompute_policy_latents.py --config "$CONFIG" "${FORCE_ARGS[@]}"
