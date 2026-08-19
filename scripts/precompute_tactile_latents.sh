#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 AE_CHECKPOINT OUTPUT_ZARR [GPU]" >&2
  exit 1
fi

CHECKPOINT="$1"
OUTPUT="$2"
GPU="${3:-${CUDA_VISIBLE_DEVICES:-0}}"
PYTHON_BIN="${PYTHON:-/mnt/data/lcy/miniconda3/envs/lyc_rl/bin/python}"
DATA_ROOT="${DATA_ROOT:-/mnt/workspace/lyc/data/huanggua_office/huanggua_office_0729_2248}"

export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONUNBUFFERED=1

exec "$PYTHON_BIN" tools/precompute_tactile_latents.py \
  --data-root "$DATA_ROOT" \
  --checkpoint "$CHECKPOINT" \
  --output "$OUTPUT"
