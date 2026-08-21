#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PEEL_DATA_ROOT="/mnt/workspace/lyc/data/peel_cucumber_val/peel_cucumber_val_0821_1058"
EGG_DATA_ROOT="/mnt/workspace/lyc/data/egg_val/egg_val_0821_1124"
PEEL_CACHE_CONFIG="configs/eval/peel_cucumber_val.yaml"
EGG_CACHE_CONFIG="configs/eval/egg_val.yaml"
PEEL_RUN="outputs/peel_cucumber_fm3/peel_cucumber"
EGG_RUN="outputs/egg_fm3/egg"
OUTPUT_ROOT="outputs/tactile_reconstruction_eval"
RUN_ID="${EVAL_RUN_ID:-full_eval_$(date +%Y%m%d_%H%M%S)}"
TARGET="both"
GPUS="${CUDA_VISIBLE_DEVICES:-0}"
PYTHON_BIN="${PYTHON:-python}"
SAMPLE_STRIDE=1
MAX_WINDOWS=-1
FULL_EPISODE_SNAPSHOT_COUNT=8
BASE_MODE="both"
CONTACT_DZ_THRESHOLD=0.005
DDP_TIMEOUT_S=10800
SKIP_PRECOMPUTE=false
DRY_RUN=false

usage() {
  cat <<'EOF'
Usage:
  ./scripts/eval_tactile_reconstruction_validation.sh [options]

Options:
  --target NAME                   peel | egg | both (default: both)
  --gpus IDS                     CUDA_VISIBLE_DEVICES (default: current or 0)
  --output-root PATH             Root for separate validation outputs
  --run-id NAME                  Output subdirectory (default: full_eval_DATE_TIME)
  --sample-stride N              Window anchor stride (default: 1, full dense eval)
  --max-windows N                Random global cap; -1 means all (default: -1)
  --snapshot-count N             Tactile-field snapshots per episode (default: 8)
  --base-mode MODE               original | remove | both (default: both)
  --contact-dz-threshold VALUE   GT abs(dz) contact threshold (default: 0.005)
  --ddp-timeout-s N              Collective timeout (default: 10800, 3 hours)
  --skip-precompute              Require an existing compatible DINO cache
  --dry-run                      Validate data/cache/windows without model inference
  -h, --help                     Show this help

Each checkpoint is evaluated only on its matching validation dataset:
  peel_cucumber_fm3 -> peel_cucumber_val_0821_1058
  egg_fm3           -> egg_val_0821_1124

The two metric sets are intentionally kept separate because scores from different
validation distributions are not a direct model ranking.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --target) TARGET="$2"; shift 2 ;;
    --target=*) TARGET="${1#*=}"; shift ;;
    --gpus) GPUS="$2"; shift 2 ;;
    --gpus=*) GPUS="${1#*=}"; shift ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --output-root=*) OUTPUT_ROOT="${1#*=}"; shift ;;
    --run-id) RUN_ID="$2"; shift 2 ;;
    --run-id=*) RUN_ID="${1#*=}"; shift ;;
    --sample-stride) SAMPLE_STRIDE="$2"; shift 2 ;;
    --sample-stride=*) SAMPLE_STRIDE="${1#*=}"; shift ;;
    --max-windows) MAX_WINDOWS="$2"; shift 2 ;;
    --max-windows=*) MAX_WINDOWS="${1#*=}"; shift ;;
    --snapshot-count) FULL_EPISODE_SNAPSHOT_COUNT="$2"; shift 2 ;;
    --snapshot-count=*) FULL_EPISODE_SNAPSHOT_COUNT="${1#*=}"; shift ;;
    --base-mode) BASE_MODE="$2"; shift 2 ;;
    --base-mode=*) BASE_MODE="${1#*=}"; shift ;;
    --contact-dz-threshold) CONTACT_DZ_THRESHOLD="$2"; shift 2 ;;
    --contact-dz-threshold=*) CONTACT_DZ_THRESHOLD="${1#*=}"; shift ;;
    --ddp-timeout-s) DDP_TIMEOUT_S="$2"; shift 2 ;;
    --ddp-timeout-s=*) DDP_TIMEOUT_S="${1#*=}"; shift ;;
    --skip-precompute) SKIP_PRECOMPUTE=true; shift ;;
    --dry-run) DRY_RUN=true; shift ;;
    -h|--help|help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 1 ;;
  esac
done

case "$TARGET" in
  peel|egg|both) ;;
  *) echo "--target must be peel, egg, or both; got: $TARGET" >&2; exit 1 ;;
esac

export CUDA_VISIBLE_DEVICES="$GPUS"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
IFS=',' read -r -a GPU_IDS <<< "$GPUS"
NPROC="${#GPU_IDS[@]}"
if [[ "$DRY_RUN" == true ]]; then
  NPROC=1
fi

run_one() {
  local target="$1"
  local label="$2"
  local run_dir="$3"
  local data_root="$4"
  local cache_config="$5"
  local dataset_name="$6"
  local checkpoint="$run_dir/checkpoints/epoch_0200.pt"
  local output_dir="$OUTPUT_ROOT/$dataset_name/$label/$RUN_ID"

  if [[ ! -f "$data_root/meta.json" || ! -d "$data_root/replay_buffer.zarr" ]]; then
    echo "[$target] validation data is not finalized: $data_root" >&2
    echo "Wait for preprocessing to finish, then rerun with --target $target." >&2
    exit 1
  fi
  if [[ ! -f "$checkpoint" ]]; then
    echo "[$target] checkpoint not found: $checkpoint" >&2
    exit 1
  fi
  if [[ "$SKIP_PRECOMPUTE" == false ]]; then
    bash scripts/precompute.sh --config "$cache_config" --gpus "$GPUS" --multi-gpu
  fi

  local eval_args=(
    --run-dir "$run_dir"
    --checkpoint "$checkpoint"
    --data-root "$data_root"
    --output-dir "$output_dir"
    --base-mode "$BASE_MODE"
    --sample-stride "$SAMPLE_STRIDE"
    --max-windows "$MAX_WINDOWS"
    --plot-samples 0
    --visualize-full-episodes
    --full-episode-snapshot-count "$FULL_EPISODE_SNAPSHOT_COUNT"
    --contact-dz-threshold "$CONTACT_DZ_THRESHOLD"
    --ddp-timeout-s "$DDP_TIMEOUT_S"
    --seed 42
  )
  if [[ "$DRY_RUN" == true ]]; then
    eval_args+=(--dry-run)
  fi

  echo "[eval] target=$target model=$label data=$data_root nproc=$NPROC"
  "$PYTHON_BIN" -m torch.distributed.run \
    --standalone \
    --nproc_per_node "$NPROC" \
    tools/eval_tactile_reconstruction.py \
    "${eval_args[@]}"
}

if [[ "$TARGET" == peel || "$TARGET" == both ]]; then
  run_one \
    peel \
    peel_cucumber_fm3 \
    "$PEEL_RUN" \
    "$PEEL_DATA_ROOT" \
    "$PEEL_CACHE_CONFIG" \
    peel_cucumber_val_0821_1058
fi

if [[ "$TARGET" == egg || "$TARGET" == both ]]; then
  run_one \
    egg \
    egg_fm3 \
    "$EGG_RUN" \
    "$EGG_DATA_ROOT" \
    "$EGG_CACHE_CONFIG" \
    egg_val_0821_1124
fi

echo "[complete] separate validation outputs: $OUTPUT_ROOT"
