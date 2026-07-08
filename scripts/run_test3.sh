#!/usr/bin/env bash
# Batch run: test3 experiment matrix (9 configs).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON:-python}"
LOG_DIR="${LOG_DIR:-outputs/test3/logs}"
MODE="${MODE:-train}"  # train | precompute | all

CONFIGS=(
  "configs/test3/zhibei_part_eef_v3_rel.yaml"
  "configs/test3/zhibei_part_eef_v2_rel.yaml"
  "configs/test3/zhibei_all_eef_v2_rel_tac.yaml"
  "configs/test3/chahua_all_joint_v2_abs.yaml"
  "configs/test3/chahua_all_eef_v2_rel.yaml"
  "configs/test3/chahua_all_eef_v3_rel.yaml"
  "configs/test3/chahua_all_eef_v2_rel_tac.yaml"
  "configs/test3/chahua_all_eef_v2_rel_aug.yaml"
  "configs/test3/chahua_vq_eef_v2_rel.yaml"
)
GPUS=(0 1 2 3 4 5 6 7)
MAX_PARALLEL=${#GPUS[@]}

usage() {
  cat <<'EOF'
Usage:
  ./scripts/run_test3.sh [options]

Launch test3 training jobs (GPU pool, default GPUs 0-7 — max 8 concurrent).

Options:
  --mode MODE       train | precompute | all (default: train)
                    all = conditional precompute (skip if cache exists) + train
  --log-dir PATH    Log directory (default: outputs/test3/logs)
  -h, --help        Show this help

Environment:
  PYTHON            Python executable (default: python)
  GPUS_OVERRIDE     Space-separated GPU ids (default: "0 1 2 3 4 5 6 7")

Example:
  nohup ./scripts/run_test3.sh --mode all > outputs/test3/run_test3.master.log 2>&1 &
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)
      MODE="$2"
      shift 2
      ;;
    --mode=*)
      MODE="${1#*=}"
      shift
      ;;
    --log-dir)
      LOG_DIR="$2"
      shift 2
      ;;
    --log-dir=*)
      LOG_DIR="${1#*=}"
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

if [[ -n "${GPUS_OVERRIDE:-}" ]]; then
  # shellcheck disable=SC2206
  GPUS=($GPUS_OVERRIDE)
  MAX_PARALLEL=${#GPUS[@]}
fi

if [[ "$MAX_PARALLEL" -lt 1 ]]; then
  echo "Need at least 1 GPU, got ${MAX_PARALLEL}" >&2
  exit 1
fi

mkdir -p "$LOG_DIR"

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-8}"

task_name_from_config() {
  basename "$1" .yaml
}

# Print: skip_no_latent | skip_exists <cache> | run <cache>
precompute_action() {
  local config="$1"
  "$PYTHON_BIN" - "$config" <<'PY'
import sys
from pathlib import Path

import yaml

config_path = Path(sys.argv[1])
cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
data = cfg.get("data", {})

if not bool(data.get("use_camera_latent", False)):
    print("skip_no_latent")
    raise SystemExit(0)

root = data.get("latent_cache_root_dir") or data.get("root_dir")
if root == "${data.root_dir}":
    root = data.get("root_dir")
if not root:
    print("skip_no_latent", file=sys.stderr)
    raise SystemExit(1)

cache = Path(root) / "policy_latent_cache.zarr"
if cache.is_dir():
    print(f"skip_exists {cache}")
else:
    print(f"run {cache}")
PY
}

run_precompute() {
  local config="$1"
  local gpu="$2"
  local task_name log_file action cache_path

  task_name="$(task_name_from_config "$config")"
  log_file="${LOG_DIR}/${task_name}.precompute.log"

  read -r action cache_path <<< "$(precompute_action "$config")"
  case "$action" in
    skip_no_latent)
      echo "[precompute] skip ${task_name} (use_camera_latent=false)"
      return 0
      ;;
    skip_exists)
      echo "[precompute] skip ${task_name} (cache exists: ${cache_path})"
      return 0
      ;;
    run)
      echo "[precompute] start ${task_name} gpu=${gpu} cache=${cache_path} $(date -Is)" | tee -a "$log_file"
      CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" tools/precompute_policy_latents.py --config "$config" >>"$log_file" 2>&1
      echo "[precompute] done  ${task_name} $(date -Is)" | tee -a "$log_file"
      ;;
    *)
      echo "[precompute] unknown action for ${task_name}: ${action}" >&2
      return 1
      ;;
  esac
}

# Run jobs with at most ${#GPUS[@]} concurrent workers (round-robin GPU assignment).
launch_gpu_pool() {
  local runner="$1"
  shift
  local items=("$@")
  local fail=0
  local slot=0
  local -a pids=()

  for config in "${items[@]}"; do
    while ((${#pids[@]} >= MAX_PARALLEL)); do
      local -a alive=()
      for pid in "${pids[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
          alive+=("$pid")
        else
          wait "$pid" || fail=1
        fi
      done
      pids=("${alive[@]}")
      ((${#pids[@]} >= MAX_PARALLEL)) && sleep 1
    done

    local gpu="${GPUS[$((slot % MAX_PARALLEL))]}"
    slot=$((slot + 1))
    local name
    name="$(task_name_from_config "$config")"

    if [[ ! -f "$config" ]]; then
      echo "Missing config: $config" >&2
      return 1
    fi

    echo "[task] launch ${name} on gpu=${gpu} $(date -Is)"
    "$runner" "$config" "$gpu" &
    pids+=("$!")
  done

  for pid in "${pids[@]}"; do
    wait "$pid" || fail=1
  done
  return "$fail"
}

launch_precompute() {
  local -A scheduled_cache=()
  local jobs=()
  local config cache_path action _rest

  for config in "${CONFIGS[@]}"; do
    if [[ ! -f "$config" ]]; then
      echo "Missing config: $config" >&2
      return 1
    fi

    read -r action cache_path _rest <<< "$(precompute_action "$config")"
    local name
    name="$(task_name_from_config "$config")"

    case "$action" in
      skip_no_latent)
        echo "[precompute] skip ${name} (use_camera_latent=false)"
        ;;
      skip_exists)
        echo "[precompute] skip ${name} (cache exists: ${cache_path})"
        ;;
      run)
        if [[ -n "${scheduled_cache[$cache_path]:-}" ]]; then
          echo "[precompute] skip ${name} (already scheduled for ${cache_path})"
        else
          scheduled_cache[$cache_path]=1
          jobs+=("$config")
          echo "[precompute] schedule ${name} -> ${cache_path}"
        fi
        ;;
      *)
        echo "[precompute] unknown action for ${name}: ${action}" >&2
        return 1
        ;;
    esac
  done

  if [[ ${#jobs[@]} -eq 0 ]]; then
    echo "[precompute] nothing to do"
    return 0
  fi

  launch_gpu_pool run_precompute "${jobs[@]}"
}

run_train() {
  local config="$1"
  local gpu="$2"
  local task_name log_file

  task_name="$(task_name_from_config "$config")"
  log_file="${LOG_DIR}/${task_name}.train.log"

  echo "[train] start ${task_name} gpu=${gpu} $(date -Is)" | tee -a "$log_file"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" train.py --config "$config" >>"$log_file" 2>&1
  echo "[train] done  ${task_name} $(date -Is)" | tee -a "$log_file"
}

launch_parallel() {
  local mode="$1"
  case "$mode" in
    precompute)
      launch_precompute
      ;;
    train)
      launch_gpu_pool run_train "${CONFIGS[@]}"
      ;;
    *)
      echo "Unknown mode: $mode" >&2
      return 1
      ;;
  esac
}

echo "=== run_test3 start mode=${MODE} $(date -Is) ==="
echo "log_dir=${LOG_DIR}"
echo "gpus=${GPUS[*]} max_parallel=${MAX_PARALLEL}"

fail=0
case "$MODE" in
  precompute)
    launch_parallel precompute || fail=1
    ;;
  train)
    launch_parallel train || fail=1
    ;;
  all)
    launch_parallel precompute || fail=1
    launch_parallel train || fail=1
    ;;
  *)
    echo "Unknown mode: $MODE" >&2
    usage
    exit 1
    ;;
esac

if [[ "$fail" -ne 0 ]]; then
  echo "=== run_test3 failed $(date -Is) ===" >&2
  exit 1
fi

echo "=== run_test3 success $(date -Is) ==="
