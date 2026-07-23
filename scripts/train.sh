#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CONFIG="configs/train/config.yaml"
GPUS="${CUDA_VISIBLE_DEVICES:-0}"
PYTHON_BIN="${PYTHON:-python}"

usage() {
  cat <<'EOF'
Usage:
  ./scripts/train.sh [options]

Options:
  --config PATH   Config yaml (default: configs/train/config.yaml)
  --gpus IDS      CUDA_VISIBLE_DEVICES (default: 0)
  -h, --help      Show this help

Examples:
  ./scripts/train.sh
  ./scripts/train.sh --gpus 0
  PYTHON=/path/to/env/bin/python ./scripts/train.sh --gpus 0,1,2,3,4,5,6,7
  ./scripts/train.sh --config configs/train/smoke_mem.yaml

Edit training hyperparameters in the config yaml (data, train, models, output, checkpoint).
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

# Always launch through PYTHON_BIN so DDP workers use the same environment.
IFS=',' read -ra GPU_ARR <<< "$GPUS"
NGPU="${#GPU_ARR[@]}"

if [[ "$NGPU" -gt 1 ]]; then
  MASTER_PORT="${MASTER_PORT:-29500}"
  exec "$PYTHON_BIN" -m torch.distributed.run \
    --standalone \
    --nnodes=1 \
    --nproc_per_node="$NGPU" \
    --master_port="$MASTER_PORT" \
    train.py --config "$CONFIG"
else
  exec "$PYTHON_BIN" train.py --config "$CONFIG"
fi
