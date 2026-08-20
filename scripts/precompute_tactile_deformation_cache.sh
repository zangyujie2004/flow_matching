#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON:-/mnt/data/lcy/miniconda3/envs/lyc_rl/bin/python}"
DATA_ROOT="${DATA_ROOT:-/mnt/workspace/lyc/data/peel_cucumber/peel_cucumber_0819_1102}"
OUTPUT="${TACTILE_DEFORMATION_OUTPUT:-}"
BATCH_FRAMES="${TACTILE_DEFORMATION_BATCH_FRAMES:-256}"
OVERWRITE=0

usage() {
  cat <<EOF
Usage: $0 [options]

Generate the float32 deformation cache used for the 8-frame tactile input by
OpenPI and Flow Matching. The Python generator also writes OUTPUT.json.

Options:
  --data-root PATH    Dataset directory containing replay_buffer.zarr
  --output PATH       Output .npy path
  --batch-frames N    Number of source frames per conversion batch (default: 256)
  --overwrite         Replace the existing .npy and .npy.json outputs
  -h, --help          Show this help

Defaults:
  data root: $DATA_ROOT
  output:    ${OUTPUT:-$DATA_ROOT/tactile_deformation_f32.npy}

Environment overrides:
  PYTHON, DATA_ROOT, TACTILE_DEFORMATION_OUTPUT,
  TACTILE_DEFORMATION_BATCH_FRAMES
EOF
}

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
    --output)
      OUTPUT="$2"
      shift 2
      ;;
    --output=*)
      OUTPUT="${1#*=}"
      shift
      ;;
    --batch-frames)
      BATCH_FRAMES="$2"
      shift 2
      ;;
    --batch-frames=*)
      BATCH_FRAMES="${1#*=}"
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

if [[ -z "$OUTPUT" ]]; then
  OUTPUT_ROOT="$DATA_ROOT"
  if [[ "$OUTPUT_ROOT" == */replay_buffer.zarr ]]; then
    OUTPUT_ROOT="$(dirname "$OUTPUT_ROOT")"
  fi
  OUTPUT="$OUTPUT_ROOT/tactile_deformation_f32.npy"
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python executable not found: $PYTHON_BIN" >&2
  exit 1
fi
if [[ ! -d "$DATA_ROOT/replay_buffer.zarr" && "$DATA_ROOT" != */replay_buffer.zarr ]]; then
  echo "replay_buffer.zarr not found below data root: $DATA_ROOT" >&2
  exit 1
fi
if [[ "$OUTPUT" != *.npy ]]; then
  echo "--output must end in .npy: $OUTPUT" >&2
  exit 2
fi
if [[ ! "$BATCH_FRAMES" =~ ^[1-9][0-9]*$ ]]; then
  echo "Invalid --batch-frames '$BATCH_FRAMES' (expected a positive integer)" >&2
  exit 2
fi

ARGS=(
  --data-root "$DATA_ROOT"
  --output "$OUTPUT"
  --batch-frames "$BATCH_FRAMES"
)
if [[ "$OVERWRITE" -eq 1 ]]; then
  ARGS+=(--overwrite)
fi

echo "[tactile-deformation] data_root=$DATA_ROOT"
echo "[tactile-deformation] output=$OUTPUT metadata=$OUTPUT.json"
echo "[tactile-deformation] batch_frames=$BATCH_FRAMES overwrite=$OVERWRITE"

exec "$PYTHON_BIN" tools/precompute_tactile_deformation_cache.py "${ARGS[@]}"
