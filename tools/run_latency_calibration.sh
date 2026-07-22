#!/usr/bin/env bash

set -u

python_bin="${PYTHON:-python}"
only="all"
skip_stress=0
common_args=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-stress)
            skip_stress=1
            shift
            ;;
        --architecture-only)
            common_args+=("$1")
            shift
            ;;
        --only)
            only="$2"
            shift 2
            ;;
        --run-dir|--checkpoint|--device|--batch-size|--num-views|--warmup-iterations|--iterations|--seed|--output-dir|--tag)
            common_args+=("$1" "$2")
            shift 2
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 2
            ;;
    esac
done

case "$only" in
    all|dino|memory|policy|full|concurrent) ;;
    *)
        echo "--only must be one of: dino memory policy full concurrent" >&2
        exit 2
        ;;
esac

has_run_dir=0
architecture_only=0
for value in "${common_args[@]}"; do
    if [[ "$value" == "--run-dir" ]]; then
        has_run_dir=1
    fi
    if [[ "$value" == "--architecture-only" ]]; then
        architecture_only=1
    fi
done
if [[ $has_run_dir -eq 0 && $architecture_only -eq 0 ]]; then
    echo "--run-dir is required unless --architecture-only is used" >&2
    exit 2
fi

failed=0

run_test() {
    local label="$1"
    local module="$2"
    shift 2
    echo "[latency] START ${label}"
    if "$python_bin" -m "$module" "${common_args[@]}" "$@"; then
        echo "[latency] PASS ${label}"
    else
        echo "[latency] FAIL ${label}" >&2
        failed=1
    fi
}

if [[ "$only" == "all" || "$only" == "dino" ]]; then
    run_test dino tools.bench_dino_latency
fi
if [[ "$only" == "all" || "$only" == "memory" ]]; then
    run_test memory tools.bench_memory_latency
fi
if [[ "$only" == "all" || "$only" == "policy" ]]; then
    run_test policy tools.bench_policy_latency
fi
if [[ "$only" == "all" || "$only" == "full" ]]; then
    run_test full_pipeline tools.bench_full_pipeline_latency
fi
if [[ "$only" == "all" || "$only" == "concurrent" ]]; then
    run_test concurrent_realistic tools.bench_concurrent_latency --scenario realistic
    if [[ $skip_stress -eq 0 ]]; then
        run_test concurrent_stress tools.bench_concurrent_latency --scenario stress
    fi
fi

exit "$failed"
