#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON:-/mnt/data/lcy/miniconda3/envs/lyc_rl/bin/python}"
DATA_ROOT="${DATA_ROOT:-/mnt/workspace/lyc/data/peel_cucumber/peel_cucumber_0819_1102}"
CHECKPOINT="${TACTILE_AE_CHECKPOINT:-$ROOT/outputs/huanggua_office_0729_tactile_ae/tactile_residual_4x16/checkpoints/best.pt}"
OUTPUT="${TACTILE_LATENT_OUTPUT:-}"
GPU="${TACTILE_CACHE_GPU:-${CUDA_VISIBLE_DEVICES:-0}}"
BATCH_SIZE="${TACTILE_LATENT_BATCH_SIZE:-1024}"
OVERWRITE=0

usage() {
  cat <<EOF
Usage:
  $0 [options]
  $0 AE_CHECKPOINT OUTPUT_ZARR [GPU]   # legacy interface

Generate the normalized 64-D tactile latent cache used as a tactile-prediction
target by Flow Matching and OpenPI.

Options:
  --data-root PATH    Dataset directory containing replay_buffer.zarr
  --checkpoint PATH   Stage-1 tactile AE checkpoint
  --output PATH       Output Zarr directory
  --gpu ID            Physical GPU id exposed to the encoder (default: 0)
  --batch-size N      Encoding batch size (default: 1024)
  --overwrite         Replace an existing output cache
  -h, --help          Show this help

Defaults:
  data root:  $DATA_ROOT
  checkpoint: $CHECKPOINT
  output:     ${OUTPUT:-$DATA_ROOT/tactile_latent_4x16.zarr}

Environment overrides:
  PYTHON, DATA_ROOT, TACTILE_AE_CHECKPOINT, TACTILE_LATENT_OUTPUT,
  TACTILE_CACHE_GPU, TACTILE_LATENT_BATCH_SIZE
EOF
}

# Preserve the original positional interface so existing commands keep working.
if [[ $# -gt 0 && "$1" != -* ]]; then
  if [[ $# -lt 2 || $# -gt 3 ]]; then
    usage >&2
    exit 2
  fi
  CHECKPOINT="$1"
  OUTPUT="$2"
  GPU="${3:-$GPU}"
else
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --data-root)
        DATA_ROOT="$2"
        shift 2
        ;;
      --data-root=*)
        DATA_ROOT="${1#*=}"
        shift
        ;;
      --checkpoint)
        CHECKPOINT="$2"
        shift 2
        ;;
      --checkpoint=*)
        CHECKPOINT="${1#*=}"
        shift
        ;;
      --output)
        OUTPUT="$2"
        shift 2
        ;;
      --output=*)
        OUTPUT="${1#*=}"
        shift
        ;;
      --gpu)
        GPU="$2"
        shift 2
        ;;
      --gpu=*)
        GPU="${1#*=}"
        shift
        ;;
      --batch-size)
        BATCH_SIZE="$2"
        shift 2
        ;;
      --batch-size=*)
        BATCH_SIZE="${1#*=}"
        shift
        ;;
      --overwrite)
        OVERWRITE=1
        shift
        ;;
      -h|--help)
        usage
        exit 0
        ;;
      *)
        echo "Unknown option: $1" >&2
        usage >&2
        exit 2
        ;;
    esac
  done
fi

if [[ -z "$OUTPUT" ]]; then
  OUTPUT_ROOT="$DATA_ROOT"
  if [[ "$OUTPUT_ROOT" == */replay_buffer.zarr ]]; then
    OUTPUT_ROOT="$(dirname "$OUTPUT_ROOT")"
  fi
  OUTPUT="$OUTPUT_ROOT/tactile_latent_4x16.zarr"
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python executable not found: $PYTHON_BIN" >&2
  exit 1
fi
if [[ ! -d "$DATA_ROOT/replay_buffer.zarr" && "$DATA_ROOT" != */replay_buffer.zarr ]]; then
  echo "replay_buffer.zarr not found below data root: $DATA_ROOT" >&2
  exit 1
fi
if [[ ! -f "$CHECKPOINT" ]]; then
  echo "Tactile AE checkpoint not found: $CHECKPOINT" >&2
  exit 1
fi
if [[ ! "$BATCH_SIZE" =~ ^[1-9][0-9]*$ ]]; then
  echo "Invalid --batch-size '$BATCH_SIZE' (expected a positive integer)" >&2
  exit 2
fi

export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONUNBUFFERED=1

ARGS=(
  --data-root "$DATA_ROOT"
  --checkpoint "$CHECKPOINT"
  --output "$OUTPUT"
  --batch-size "$BATCH_SIZE"
)
if [[ "$OVERWRITE" -eq 1 ]]; then
  ARGS+=(--overwrite)
fi

echo "[tactile-latent] data_root=$DATA_ROOT"
echo "[tactile-latent] checkpoint=$CHECKPOINT"
echo "[tactile-latent] output=$OUTPUT gpu=$GPU batch_size=$BATCH_SIZE overwrite=$OVERWRITE"

exec "$PYTHON_BIN" tools/precompute_tactile_latents.py "${ARGS[@]}"
