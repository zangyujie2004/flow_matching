#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CONFIG="configs/config_peel.yaml"
GPUS="${CUDA_VISIBLE_DEVICES:-3}"
PYTHON_BIN="${PYTHON:-python}"

usage() {
  cat <<'EOF'
Usage:
  ./scripts/precompute.sh [options]

Options:
  --config PATH   Config yaml (default: configs/config.yaml)
  --gpus IDS      CUDA_VISIBLE_DEVICES (default: 0)
  -h, --help      Show this help

Writes:
  {data.latent_cache_root_dir}/policy_latent_cache.zarr

Then set in config:
  data.use_camera_latent: true

Edit precompute.* settings in the config yaml (batch_size, overwrite, max_windows).
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
exec "$PYTHON_BIN" tools/precompute_policy_latents.py --config "$CONFIG"
