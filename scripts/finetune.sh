#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CONFIG="configs/finetune/config.yaml"
GPUS="${CUDA_VISIBLE_DEVICES:-0}"
PYTHON_BIN="${PYTHON:-python}"
SKIP_PRECOMPUTE=0

usage() {
  cat <<'EOF'
Usage:
  ./scripts/finetune.sh [options]

Options:
  --config PATH         Finetune config yaml (default: configs/finetune/config.yaml)
  --gpus IDS            CUDA_VISIBLE_DEVICES (default: 0)
  --skip-precompute     Skip latent precompute step
  -h, --help            Show this help

Flow:
  1. precompute policy latents (skipped if cache exists unless precompute.overwrite=true)
  2. finetune.py with merged base + finetune config

Edit dataset / checkpoint / train overrides in configs/finetune/config.yaml.
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
    --skip-precompute)
      SKIP_PRECOMPUTE=1
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

if [[ "$SKIP_PRECOMPUTE" -eq 0 ]]; then
  "$PYTHON_BIN" tools/precompute_policy_latents.py --config "$CONFIG"
fi

exec "$PYTHON_BIN" finetune.py --config "$CONFIG"
