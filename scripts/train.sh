#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CONFIG="configs/config.yaml"
GPUS="${CUDA_VISIBLE_DEVICES:-1}"
PYTHON_BIN="${PYTHON:-python}"

usage() {
  cat <<'EOF'
Usage:
  ./scripts/train.sh [options]

Options:
  --config PATH   Config yaml (default: configs/config.yaml)
  --gpus IDS      CUDA_VISIBLE_DEVICES (default: 0)
  --smoke         Use configs/smoke.yaml for a quick debug run
  -h, --help      Show this help

Examples:
  ./scripts/train.sh
  ./scripts/train.sh --gpus 0
  ./scripts/train.sh --config configs/config.yaml
  ./scripts/train.sh --smoke

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
    --smoke)
      CONFIG="configs/smoke.yaml"
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
exec "$PYTHON_BIN" train.py --config "$CONFIG"
