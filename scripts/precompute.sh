#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CONFIG="configs/train/config.yaml"
GPUS="${CUDA_VISIBLE_DEVICES:-0}"
PYTHON_BIN="${PYTHON:-python}"
FORCE_ARGS=()
RUNTIME_ARGS=()

usage() {
  cat <<'EOF'
Usage:
  ./scripts/precompute.sh [options]

Options:
  --config PATH          Config yaml (default: configs/train/config.yaml)
  --gpus IDS             CUDA_VISIBLE_DEVICES (default: 0)
  --workers N            Parallel Zarr readers (overrides precompute.num_workers)
  --prefetch-batches N   Bounded queued batches (overrides config)
  --single-gpu           Use only the first visible GPU
  --multi-gpu            Use all visible GPUs (default from config)
  --force                Rebuild even if identity-matching frame cache exists
  -h, --help             Show this help

Writes (scheme A, frame-only):
  {data.latent_cache_root_dir}/frame_backbone.zarr

Skip rule: existing cache with matching identity + full T frames → skip.
Use --force (or precompute.overwrite=true) to recompute.

Independent of data.window_size / stride / n_image_steps / action_horizon / memory.

The optimized path streams camera-only Zarr batches. It does not preload tactile,
state, action, or the full camera array. When multiple IDs are supplied to
--gpus and multi_gpu is enabled, DINO encoding is split across all visible GPUs.
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
    --workers)
      RUNTIME_ARGS+=(--workers "$2")
      shift 2
      ;;
    --workers=*)
      RUNTIME_ARGS+=(--workers "${1#*=}")
      shift
      ;;
    --prefetch-batches)
      RUNTIME_ARGS+=(--prefetch-batches "$2")
      shift 2
      ;;
    --prefetch-batches=*)
      RUNTIME_ARGS+=(--prefetch-batches "${1#*=}")
      shift
      ;;
    --single-gpu)
      RUNTIME_ARGS+=(--single-gpu)
      shift
      ;;
    --multi-gpu)
      RUNTIME_ARGS+=(--multi-gpu)
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
echo "[precompute] config=$CONFIG gpus=$GPUS ${RUNTIME_ARGS[*]}"
exec "$PYTHON_BIN" tools/precompute_policy_latents.py \
  --config "$CONFIG" \
  "${FORCE_ARGS[@]}" \
  "${RUNTIME_ARGS[@]}"
