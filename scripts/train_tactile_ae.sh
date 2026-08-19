#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CONFIG="configs/train/tactile_ae.yaml"
GPUS="${CUDA_VISIBLE_DEVICES:-0}"
NPROC=""
RESUME=""
AMP_MODE=""
BATCH_SIZE=""
PYTHON_BIN="${PYTHON:-/mnt/data/lcy/miniconda3/envs/lyc_rl/bin/python}"

usage() {
  cat <<'EOF'
Usage:
  ./scripts/train_tactile_ae.sh [options]

Single-GPU or multi-GPU DDP Stage 1 tactile-AE training.

Options:
  --config PATH       Config yaml (default: configs/train/tactile_ae.yaml)
  --gpus IDS          Physical GPU ids, e.g. 0 or 0,1,2,3,4,5,6,7
  --nproc N           Processes (default: number of ids in --gpus)
  --resume PATH       Resume model, optimizer, normalizer and epoch
  --batch-size N      Per-GPU batch size override
  --amp MODE          off | fp16 | bf16
  -h, --help          Show this help

For an 8-GPU continuation of the original single-GPU batch_size=256 run,
use --batch-size 32 so that the global batch remains 256.
EOF
}

# Preserve the old positional config form: train_tactile_ae.sh path/to/config.yaml
if [[ $# -gt 0 && "$1" != --* ]]; then
  CONFIG="$1"
  shift
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) CONFIG="$2"; shift 2 ;;
    --config=*) CONFIG="${1#*=}"; shift ;;
    --gpus) GPUS="$2"; shift 2 ;;
    --gpus=*) GPUS="${1#*=}"; shift ;;
    --nproc) NPROC="$2"; shift 2 ;;
    --nproc=*) NPROC="${1#*=}"; shift ;;
    --resume) RESUME="$2"; shift 2 ;;
    --resume=*) RESUME="${1#*=}"; shift ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --batch-size=*) BATCH_SIZE="${1#*=}"; shift ;;
    --amp) AMP_MODE="$2"; shift 2 ;;
    --amp=*) AMP_MODE="${1#*=}"; shift ;;
    -h|--help|help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 1 ;;
  esac
done

IFS=',' read -r -a GPU_ARR <<< "$GPUS"
GPU_IDS=()
declare -A SEEN_GPUS=()
for gpu in "${GPU_ARR[@]}"; do
  gpu="$(echo "$gpu" | xargs)"
  if [[ ! "$gpu" =~ ^[0-9]+$ ]]; then
    echo "Invalid GPU id '$gpu' in --gpus='$GPUS'" >&2
    exit 1
  fi
  if [[ -n "${SEEN_GPUS[$gpu]:-}" ]]; then
    echo "Duplicate GPU id '$gpu' in --gpus='$GPUS'" >&2
    exit 1
  fi
  SEEN_GPUS[$gpu]=1
  GPU_IDS+=("$gpu")
done
if [[ ${#GPU_IDS[@]} -eq 0 ]]; then
  echo "No GPUs specified" >&2
  exit 1
fi
if [[ -z "$NPROC" ]]; then
  NPROC="${#GPU_IDS[@]}"
fi
if [[ ! "$NPROC" =~ ^[1-9][0-9]*$ ]] || (( NPROC > ${#GPU_IDS[@]} )); then
  echo "Invalid --nproc='$NPROC' for ${#GPU_IDS[@]} selected GPUs" >&2
  exit 1
fi
if [[ -n "$BATCH_SIZE" && ! "$BATCH_SIZE" =~ ^[1-9][0-9]*$ ]]; then
  echo "Invalid --batch-size='$BATCH_SIZE'" >&2
  exit 1
fi
if [[ -n "$AMP_MODE" && "$AMP_MODE" != "off" && "$AMP_MODE" != "fp16" && "$AMP_MODE" != "bf16" ]]; then
  echo "Invalid --amp='$AMP_MODE' (expected off, fp16, or bf16)" >&2
  exit 1
fi
if [[ ! -f "$CONFIG" ]]; then
  echo "Config file not found: $CONFIG" >&2
  exit 1
fi
if [[ -n "$RESUME" && ! -f "$RESUME" ]]; then
  echo "Resume checkpoint not found: $RESUME" >&2
  exit 1
fi

GPU_CSV="$(IFS=,; echo "${GPU_IDS[*]}")"
export CUDA_VISIBLE_DEVICES="$GPU_CSV"
export PYTHONUNBUFFERED=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"

TRAIN_ARGS=(--config "$CONFIG")
if [[ -n "$RESUME" ]]; then TRAIN_ARGS+=(--resume "$RESUME"); fi
if [[ -n "$BATCH_SIZE" ]]; then TRAIN_ARGS+=(--batch-size "$BATCH_SIZE"); fi
if [[ -n "$AMP_MODE" ]]; then TRAIN_ARGS+=(--amp "$AMP_MODE"); fi

if (( NPROC == 1 )); then
  echo "[tactile-ae] single-GPU config=$CONFIG gpu=$GPU_CSV batch=${BATCH_SIZE:-config} resume=${RESUME:-none}"
  exec "$PYTHON_BIN" train_tactile_ae.py "${TRAIN_ARGS[@]}"
fi

echo "[tactile-ae] DDP config=$CONFIG gpus=$GPU_CSV nproc=$NPROC batch=${BATCH_SIZE:-config}/gpu resume=${RESUME:-none}"
exec "$PYTHON_BIN" -m torch.distributed.run \
  --standalone \
  --max_restarts=0 \
  --nproc_per_node="$NPROC" \
  train_tactile_ae.py \
  "${TRAIN_ARGS[@]}"
