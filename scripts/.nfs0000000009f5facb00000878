#!/usr/bin/env bash
set -euo pipefail

# Route shared flags to both steps; --force is precompute-only.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CONFIG=""
GPUS=""
FORCE=0

usage() {
  cat <<'EOF'
Usage:
  ./scripts/run_all.sh [options]

Runs precompute then train.

Options:
  --config PATH   Passed to precompute and train
  --gpus IDS      CUDA_VISIBLE_DEVICES for both
  --force         Rebuild frame cache even if identity matches (precompute only)
  -h, --help      Show this help
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
    --force)
      FORCE=1
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

SHARED_ARGS=()
PRE_ARGS=()
if [[ -n "$CONFIG" ]]; then
  SHARED_ARGS+=(--config "$CONFIG")
fi
if [[ -n "$GPUS" ]]; then
  SHARED_ARGS+=(--gpus "$GPUS")
fi
if [[ "$FORCE" -eq 1 ]]; then
  PRE_ARGS+=(--force)
fi

./scripts/precompute.sh "${SHARED_ARGS[@]}" "${PRE_ARGS[@]}"
./scripts/train.sh "${SHARED_ARGS[@]}"
