#!/usr/bin/env bash
# Offline batch run: 8 task configs on 8 GPUs (train only).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON:-python}"
LOG_DIR="${LOG_DIR:-outputs/tasks_0707/logs}"

# One config per GPU (0-7).
CONFIGS=(
  "configs/tasks_0707/chahua_all_eef_v2_rel.yaml"
  "configs/tasks_0707/chahua_all_eef_v3_rel.yaml"
  "configs/tasks_0707/chahua_all_eef_v2_rel_tac.yaml"
  "configs/tasks_0707/chahua_all_joint_v2_abs.yaml"
  "configs/tasks_0707/chahua_vq_eef_v2_rel.yaml"
  "configs/tasks_0707/chahua_vq_eef_v3_rel.yaml"
  "configs/tasks_0707/peel_eef_v3_rel.yaml"
  "configs/tasks_0707/zhibei_eef_v2_rel.yaml"
)
GPUS=(0 1 2 3 4 5 6 7)

usage() {
  cat <<'EOF'
Usage:
  ./scripts/run_0707.sh [options]

Launch 8 offline training jobs in parallel (one GPU each).

Options:
  --log-dir PATH      Log directory (default: outputs/tasks_0707/logs)
  -h, --help          Show this help

Environment:
  PYTHON              Python executable (default: python)

Example:
  nohup ./scripts/run_0707.sh > outputs/tasks_0707/run_0707.master.log 2>&1 &
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
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

if [[ ${#CONFIGS[@]} -ne ${#GPUS[@]} ]]; then
  echo "CONFIGS and GPUS length mismatch" >&2
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

echo "=== run_0707 start $(date -Is) ==="
echo "log_dir=${LOG_DIR}"
echo "offline: HF_HUB_OFFLINE=${HF_HUB_OFFLINE}"

pids=()
names=()
for i in "${!CONFIGS[@]}"; do
  config="${CONFIGS[$i]}"
  gpu="${GPUS[$i]}"
  name="$(task_name_from_config "$config")"

  if [[ ! -f "$config" ]]; then
    echo "Missing config: $config" >&2
    exit 1
  fi

  echo "[task] launch ${name} on gpu=${gpu} $(date -Is)"
  run_train "$config" "$gpu" &
  pids+=("$!")
  names+=("$name")
done

fail=0
for i in "${!pids[@]}"; do
  if ! wait "${pids[$i]}"; then
    echo "[fail] ${names[$i]} (pid=${pids[$i]})" >&2
    fail=1
  fi
done

if [[ "$fail" -ne 0 ]]; then
  echo "=== run_0707 failed $(date -Is) ===" >&2
  exit 1
fi

echo "=== run_0707 success $(date -Is) ==="
