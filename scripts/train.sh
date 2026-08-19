#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CONFIG="configs/train/config.yaml"
GPUS="${CUDA_VISIBLE_DEVICES:-0}"
NPROC=""
AMP_MODE=""
PYTHON_BIN="${PYTHON:-python}"

usage() {
  cat <<'EOF'
Usage:
  ./scripts/train.sh [options]

Single-GPU or multi-GPU (DDP) training.
  1 GPU  → python train.py
  N>1    → torchrun train.py

Options:
  --config PATH   Config yaml (default: configs/train/config.yaml)
  --gpus IDS      CUDA_VISIBLE_DEVICES (default: 0). e.g. 0 | 4,5,6
  --nproc N       nproc_per_node (default: number of ids in --gpus)
  --amp MODE      Mixed precision override: off | fp16 | bf16
  -h, --help      Show this help

Notes:
  train.batch_size in yaml is PER-GPU.
  Global batch = batch_size * nproc under DDP.
  BF16 is recommended on H20/Ampere-or-newer GPUs; FP16 uses GradScaler.
  open_loop_test_every / checkpoint.save_every are in EPOCHS (not steps).
  TB: Step/* uses global_step; Epoch/* and OpenLoop/* use epoch.

Examples:
  ./scripts/train.sh
  ./scripts/train.sh --gpus 0
  ./scripts/train.sh --gpus 4
  ./scripts/train.sh --gpus 4,5,6
  ./scripts/train.sh --gpus 0,1,2,3 --amp bf16
  ./scripts/train.sh --gpus 4,5,6 --config configs/train/config.yaml
  PYTHON=/path/to/env/bin/python ./scripts/train.sh --gpus 0,1,2,3,4,5,6,7
  ./scripts/train.sh --config configs/train/smoke_mem.yaml
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
    --nproc)
      NPROC="$2"
      shift 2
      ;;
    --nproc=*)
      NPROC="${1#*=}"
      shift
      ;;
    --amp)
      AMP_MODE="$2"
      shift 2
      ;;
    --amp=*)
      AMP_MODE="${1#*=}"
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

# Validate and canonicalize comma-separated physical GPU ids.
IFS=',' read -r -a GPU_ARR <<< "$GPUS"
GPU_IDS=()
declare -A SEEN_GPUS=()
for g in "${GPU_ARR[@]}"; do
  g="$(echo "$g" | xargs)"
  if [[ ! "$g" =~ ^[0-9]+$ ]]; then
    echo "Invalid GPU id '$g' in --gpus='$GPUS' (expected non-negative integers)" >&2
    exit 1
  fi
  if [[ -n "${SEEN_GPUS[$g]:-}" ]]; then
    echo "Duplicate GPU id '$g' in --gpus='$GPUS'" >&2
    exit 1
  fi
  SEEN_GPUS[$g]=1
  GPU_IDS+=("$g")
done
if [[ ${#GPU_IDS[@]} -eq 0 ]]; then
  echo "No GPUs specified in --gpus='$GPUS'" >&2
  exit 1
fi
if [[ -z "$NPROC" ]]; then
  NPROC="${#GPU_IDS[@]}"
fi
if [[ ! "$NPROC" =~ ^[1-9][0-9]*$ ]]; then
  echo "Invalid --nproc='$NPROC' (expected a positive integer)" >&2
  exit 1
fi
if (( NPROC > ${#GPU_IDS[@]} )); then
  echo "--nproc=$NPROC exceeds the ${#GPU_IDS[@]} GPU ids selected by --gpus" >&2
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

GPU_CSV="$(IFS=,; echo "${GPU_IDS[*]}")"
export CUDA_VISIBLE_DEVICES="$GPU_CSV"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
if [[ -z "${HF_HOME:-}" && -e "$ROOT/.hf_cache" ]]; then
  export HF_HOME="$ROOT/.hf_cache"
fi
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"

TRAIN_ARGS=(--config "$CONFIG")
if [[ -n "$AMP_MODE" ]]; then
  TRAIN_ARGS+=(--amp "$AMP_MODE")
fi

if [[ "$NPROC" -le 1 ]]; then
  echo "[train] single-GPU config=$CONFIG gpus=$GPU_CSV amp=${AMP_MODE:-config}"
  exec "$PYTHON_BIN" train.py "${TRAIN_ARGS[@]}"
fi

echo "[train] DDP config=$CONFIG gpus=$GPU_CSV nproc=$NPROC amp=${AMP_MODE:-config} (batch_size=per-GPU)"
MASTER_PORT="${MASTER_PORT:-29500}"
exec "$PYTHON_BIN" -m torch.distributed.run \
  --standalone \
  --nnodes=1 \
  --max_restarts=0 \
  --nproc_per_node="$NPROC" \
  --master_port="$MASTER_PORT" \
  train.py \
  "${TRAIN_ARGS[@]}"
