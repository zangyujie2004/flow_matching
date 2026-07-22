from __future__ import annotations

import argparse
import csv
import json
import platform
import socket
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import yaml

from infer.config import load_run_config
from infer.runtime import FMInferenceRuntime


MIB = 1024**2


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-views", type=int, choices=(2, 3))
    parser.add_argument("--warmup-iterations", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=Path("latency_results"))
    parser.add_argument("--tag", default="")


def validate_common_args(args: argparse.Namespace) -> torch.device:
    if args.batch_size < 1 or args.warmup_iterations < 0 or args.iterations < 1:
        raise ValueError("batch-size/iterations must be positive and warmup non-negative")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("latency benchmarks require an available CUDA device")
    torch.cuda.set_device(device)
    return device


def cuda_memory(label: str, device: torch.device) -> dict[str, Any]:
    torch.cuda.synchronize(device)
    return {
        "stage": label,
        "allocated_mib": torch.cuda.memory_allocated(device) / MIB,
        "reserved_mib": torch.cuda.memory_reserved(device) / MIB,
        "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / MIB,
        "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / MIB,
    }


def load_runtime(args: argparse.Namespace) -> tuple[FMInferenceRuntime, float, list[dict]]:
    device = validate_common_args(args)
    cfg = load_run_config(args.run_dir)
    if not isinstance(cfg.get("models"), dict) or not isinstance(
        cfg["models"].get("fm"), dict
    ):
        raise RuntimeError(
            f"{Path(args.run_dir) / 'resolved_config.yaml'} is not a compatible "
            "Flow Matching RUN_DIR: missing models.fm"
        )
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    snapshots = [cuda_memory("before_load", device)]
    start = time.perf_counter()
    runtime = FMInferenceRuntime(
        args.run_dir,
        checkpoint=args.checkpoint,
        device=str(device),
        warmup=False,
    )
    load_seconds = time.perf_counter() - start
    for stage in ("after_model_load", "after_checkpoint_load"):
        snapshots.append({"stage": stage, **runtime.load_cuda_memory[stage]})
    if args.num_views is not None and args.num_views != runtime.n_image_views:
        raise ValueError(
            f"--num-views={args.num_views} but checkpoint/config expects "
            f"{runtime.n_image_views}"
        )
    runtime.policy.eval()
    assert_model_device(runtime.policy, device)
    print(f"runtime_load_timing_ms = {runtime.load_timing_ms}")
    return runtime, load_seconds, snapshots


def assert_model_device(model: torch.nn.Module, device: torch.device) -> None:
    if model.training:
        raise AssertionError("model must be in eval mode")
    wrong = [name for name, value in model.named_parameters() if value.device != device]
    if wrong:
        raise AssertionError(f"model parameters are not on {device}: {wrong[:5]}")


def assert_finite(value: Any) -> None:
    if torch.is_tensor(value):
        if not torch.isfinite(value).all():
            raise AssertionError("benchmark output contains NaN or Inf")
        return
    if isinstance(value, dict):
        for item in value.values():
            assert_finite(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            assert_finite(item)


def distribution(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    mean = float(array.mean())
    return {
        "count": int(array.size),
        "mean": mean,
        "std": float(array.std()),
        "min": float(array.min()),
        "p50": float(np.percentile(array, 50)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
        "max": float(array.max()),
        "throughput_hz": 1000.0 / mean if mean > 0 else 0.0,
    }


def benchmark_callable(
    name: str,
    run_once: Callable[[], Any],
    *,
    device: torch.device,
    warmup_iterations: int,
    iterations: int,
    memory_snapshots: list[dict[str, Any]] | None = None,
    correctness_run: bool = True,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    with torch.inference_mode():
        if correctness_run:
            first = run_once()
            assert_finite(first)
        for _ in range(warmup_iterations):
            run_once()
    torch.cuda.synchronize(device)
    if memory_snapshots is not None:
        memory_snapshots.append(cuda_memory(f"{name}:after_warmup", device))
    torch.cuda.reset_peak_memory_stats(device)
    if memory_snapshots is not None:
        memory_snapshots.append(cuda_memory(f"{name}:before_formal_test", device))

    rows = []
    with torch.inference_mode():
        for iteration in range(iterations):
            begin = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize(device)
            wall_start = time.perf_counter()
            begin.record()
            output = run_once()
            end.record()
            end.synchronize()
            wall_ms = (time.perf_counter() - wall_start) * 1000.0
            if iteration == 0:
                assert_finite(output)
            rows.append(
                {
                    "benchmark": name,
                    "iteration": iteration,
                    "cuda_ms": float(begin.elapsed_time(end)),
                    "wall_ms": float(wall_ms),
                    "allocated_mib": torch.cuda.memory_allocated(device) / MIB,
                    "reserved_mib": torch.cuda.memory_reserved(device) / MIB,
                    "timestamp": datetime.now().astimezone().isoformat(),
                }
            )

    summary = {
        "cuda_ms": distribution([row["cuda_ms"] for row in rows]),
        "wall_ms": distribution([row["wall_ms"] for row in rows]),
        "actual_warmup_iterations": int(warmup_iterations + int(correctness_run)),
        "first_correctness_run_excluded": bool(correctness_run),
    }
    if memory_snapshots is not None:
        memory_snapshots.append(cuda_memory(f"{name}:peak_during_test", device))
    return summary, rows


def tensor_description(tensor: torch.Tensor) -> dict[str, Any]:
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "numel": tensor.numel(),
        "element_size": tensor.element_size(),
        "theoretical_mib": tensor.numel() * tensor.element_size() / MIB,
    }


def _git_value(args: list[str], cwd: Path) -> str:
    result = subprocess.run(
        ["git", *args], cwd=cwd, text=True, capture_output=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def runtime_metadata(
    runtime: FMInferenceRuntime,
    args: argparse.Namespace,
    *,
    benchmark: str,
    load_seconds: float,
    input_shapes: dict[str, Any],
) -> dict[str, Any]:
    repo = Path(__file__).resolve().parents[1]
    device = runtime.device
    props = torch.cuda.get_device_properties(device)
    parameter_count = sum(value.numel() for value in runtime.policy.parameters())
    dtype = str(next(runtime.policy.parameters()).dtype)
    return {
        "benchmark": benchmark,
        "hostname": socket.gethostname(),
        "git_commit": _git_value(["rev-parse", "HEAD"], repo),
        "current_branch": _git_value(["branch", "--show-current"], repo),
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "pytorch_version": torch.__version__,
        "cuda_runtime_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "gpu_name": props.name,
        "gpu_count": torch.cuda.device_count(),
        "gpu_total_memory_mib": props.total_memory / MIB,
        "device_index": device.index if device.index is not None else torch.cuda.current_device(),
        "device": str(device),
        "checkpoint_path": str(runtime.checkpoint_path),
        "run_dir": str(runtime.run_dir),
        "config_path": str(runtime.run_dir / "resolved_config.yaml"),
        "batch_size": int(args.batch_size),
        "number_of_views": int(runtime.n_image_views),
        "dtype": dtype,
        "input_shapes": input_shapes,
        "model_parameter_count": parameter_count,
        "warmup_iterations": int(args.warmup_iterations),
        "formal_iterations": int(args.iterations),
        "model_and_checkpoint_load_seconds": float(load_seconds),
        "runtime_load_timing_ms": dict(runtime.load_timing_ms),
        "timestamp": datetime.now().astimezone().isoformat(),
    }


def create_result_dir(args: argparse.Namespace, benchmark: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = args.tag.strip() or benchmark
    path = Path(args.output_dir) / f"{stamp}_{socket.gethostname()}_{suffix}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    raise TypeError(f"cannot serialize {type(value).__name__}")


def save_results(
    result_dir: Path,
    *,
    metadata: dict[str, Any],
    summary: dict[str, Any],
    rows: list[dict[str, Any]],
    config: dict[str, Any],
    memory_snapshots: list[dict[str, Any]],
) -> None:
    payload = {**summary, "cuda_memory_snapshots": memory_snapshots}
    with open(result_dir / "metadata.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, default=_json_default)
    with open(result_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=_json_default)
    with open(result_dir / "config_snapshot.yaml", "w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)

    fieldnames = list(rows[0]) if rows else ["benchmark", "iteration", "cuda_ms", "wall_ms"]
    with open(result_dir / "raw_samples.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    lines = [f"result_dir={result_dir}"]
    for name, value in summary.items():
        if isinstance(value, dict) and "wall_ms" in value:
            wall = value["wall_ms"]
            cuda = value["cuda_ms"]
            lines.append(
                f"{name}: wall mean/p95/p99={wall['mean']:.3f}/{wall['p95']:.3f}/"
                f"{wall['p99']:.3f} ms; CUDA={cuda['mean']:.3f}/{cuda['p95']:.3f}/"
                f"{cuda['p99']:.3f} ms"
            )
    text = "\n".join(lines) + "\n"
    print(text, end="")
    with open(result_dir / "console_summary.txt", "w", encoding="utf-8") as handle:
        handle.write(text)


def finalize_memory_snapshots(
    snapshots: list[dict[str, Any]], device: torch.device
) -> list[dict[str, Any]]:
    snapshots.append(cuda_memory("after_test", device))
    return snapshots
