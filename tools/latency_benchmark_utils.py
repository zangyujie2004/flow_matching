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
from typing import Any, Callable, Mapping

import numpy as np
import torch
import yaml

from infer.config import load_run_config
from infer.preprocess import build_dino_images
from infer.runtime import FMInferenceRuntime
from infer.tensor import as_float32_array, numpy_obs_to_torch
from infer.types import PreprocessConfig
from models.fm.flow_policy import FlowMatchingPolicy
from tools.async_dino_buffer import AsyncDinoBuffer
from tools.normalizer import DatasetNormalizer, FieldNormalizer


MIB = 1024**2


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument(
        "--architecture-only",
        action="store_true",
        help="benchmark the real architecture with random weights and synthetic inputs",
    )
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


def architecture_only_config(num_views: int) -> dict[str, Any]:
    """Minimal explicit config describing the requested random-weight architecture."""
    camera_views = [f"camera_{index + 1}" for index in range(num_views)]
    return {
        "runtime": {"device": "cuda"},
        "data": {
            "window_size": 16,
            "image_size": 224,
            "action_type": "joint",
            "action_representation": "absolute",
            "action_horizon": 64,
            "use_tactile": False,
            "camera_views": camera_views,
            "memory": {
                "enabled": True,
                "history_frames": 64,
                "recent_frame": 4,
                "visual_history_length": 64,
                "sample_stride": 8,
                "visual_recent_frame": 0,
            },
        },
        "models": {
            "fm": {
                "n_image_views": num_views,
                "image_encoder_name": "dinov2",
                "dino_model_name": "vit_small_patch14_dinov2.lvd142m",
                "image_pretrained": False,
                "freeze_image_encoder": True,
                "image_feat_dim": 256,
                "cond_dim": 256,
                "use_tactile": False,
                "action_horizon": 64,
                "n_action_steps": 32,
                "velocity_model": "unet",
                "num_inference_steps": 32,
                "solver": "euler",
            },
            "memory": {
                "method": "fusion",
                "injection": "concat_global_cond",
                "dim": 256,
                "visual_layers": 2,
                "visual_heads": 4,
                "state_mem_dim": 64,
                "dropout": 0.0,
            },
        },
        "benchmark": {
            "mode": "architecture_only",
            "checkpoint_loaded": False,
            "policy_weights": "random_initialization",
            "memory_weights": "random_initialization",
            "synthetic_inputs": True,
            "latency_only": True,
            "task_quality_not_measured": True,
        },
    }


class ArchitectureOnlyRuntimeAdapter:
    """Only the small Runtime surface used by latency benchmarks."""

    def __init__(self, policy: FlowMatchingPolicy, device: torch.device, config: dict):
        self.device = device
        self.policy = policy
        self.cfg = config
        self.policy_cfg = config
        self.run_dir = None
        self.checkpoint_path = None
        self.window_size = 16
        self.n_image_views = int(policy.condition_encoder.image_encoder.n_views)
        self.num_inference_steps = 32
        self.solver = "euler"
        self.action_horizon = 64
        self.use_tactile = False
        self.velocity_model = "unet"
        self.memory_visual_history_length = 64
        self.dino_sample_interval_frames = 8
        self.memory_visual_recent_frame = 0
        self.memory_visual_offsets = torch.arange(
            -504, 1, 8, device=device, dtype=torch.long
        )
        identity = FieldNormalizer.identity(policy.state_dim)
        self.normalizer = DatasetNormalizer(
            state=identity,
            action=FieldNormalizer.identity(policy.action_dim),
            tactile=None,
            action_type="joint",
            action_representation="absolute",
        )
        self.async_dino_buffer: AsyncDinoBuffer | None = None
        self.async_dino_preprocess: PreprocessConfig | None = None
        self.load_timing_ms: dict[str, float] = {}
        self.load_cuda_memory: dict[str, dict[str, float]] = {}

    @property
    def action_dim(self) -> int:
        return self.policy.action_dim

    def start_async_dino(
        self,
        *,
        preprocess: PreprocessConfig | None = None,
        sample_interval_frames: int | None = None,
        deadline_ms: float | None = None,
    ) -> AsyncDinoBuffer:
        if self.async_dino_buffer is not None:
            self.async_dino_buffer.stop()
        views = tuple(self.policy_cfg["data"]["camera_views"])
        self.async_dino_preprocess = preprocess or PreprocessConfig(
            action_type="joint", camera_views=views, image_size=224, use_tactile=False
        )
        dino = self.policy.condition_encoder.image_encoder.encoder
        interval = (
            self.dino_sample_interval_frames
            if sample_interval_frames is None
            else int(sample_interval_frames)
        )
        if interval != self.dino_sample_interval_frames:
            raise ValueError(
                "sample_interval_frames must match architecture visual sample_stride: "
                f"{interval} != {self.dino_sample_interval_frames}"
            )
        deadline = 33.0 * interval if deadline_ms is None else float(deadline_ms)
        self.async_dino_buffer = AsyncDinoBuffer(
            dino,
            device=str(self.device),
            sample_interval_frames=interval,
            history_length=self.memory_visual_history_length,
            deadline_ms=deadline,
        )
        self.async_dino_buffer.start()
        print(f"memory_visual_history_length = {self.memory_visual_history_length}")
        print(f"dino_sample_interval_frames = {interval}")
        print("visual_token_source = dino_cls")
        print("startup_padding = repeat_first_frame")
        print(
            "memory_visual_input_shape = "
            f"[B,{self.memory_visual_history_length},{self.n_image_views},384]"
        )
        return self.async_dino_buffer

    def stop_async_dino(self) -> None:
        if self.async_dino_buffer is not None:
            self.async_dino_buffer.stop()

    def submit_async_dino_frame(
        self, frame_id: int, frame: Any, capture_time: float | None = None
    ) -> bool:
        if self.async_dino_buffer is None or self.async_dino_preprocess is None:
            raise RuntimeError("call start_async_dino() first")
        images = build_dino_images(frame, self.async_dino_preprocess)
        return self.async_dino_buffer.submit_frame(
            frame_id, *images, capture_time=capture_time
        )

    @torch.inference_mode()
    def predict_rot6d_abs(
        self,
        obs: Mapping[str, Any],
        *,
        state_raw: np.ndarray,
        memory_state_raw: np.ndarray | None = None,
        num_inference_steps: int | None = None,
        solver: str | None = None,
    ) -> np.ndarray:
        state_raw = as_float32_array(state_raw, name="state_raw")
        expected_state = (self.window_size, self.policy.state_dim)
        if state_raw.shape != expected_state:
            raise ValueError(f"state_raw shape {state_raw.shape} != {expected_state}")
        obs_torch = numpy_obs_to_torch(
            obs,
            self.device,
            use_tactile=False,
            normalizer=self.normalizer,
            window_size=self.window_size,
        )
        feature_window = (
            None
            if self.async_dino_buffer is None
            else self.async_dino_buffer.get_feature_window()
        )
        if feature_window is None:
            raise RuntimeError("memory not ready: DINO buffer needs its first processed sample")
        if memory_state_raw is None:
            raise ValueError("memory_state_raw is required")
        memory_state = as_float32_array(memory_state_raw, name="memory_state_raw")
        expected_memory = (self.policy.memory_history_frames, self.policy.state_dim)
        if memory_state.shape != expected_memory:
            raise ValueError(
                f"memory_state_raw shape {memory_state.shape} != {expected_memory}"
            )
        obs_torch.update(
            {
                "memory_image_backbone_feat": feature_window,
                "memory_state": torch.from_numpy(memory_state).unsqueeze(0).to(self.device),
                "memory_visual_offsets": self.memory_visual_offsets,
            }
        )
        result = self.policy.predict_action(
            obs_torch,
            num_inference_steps=(
                self.num_inference_steps
                if num_inference_steps is None
                else int(num_inference_steps)
            ),
            solver=self.solver if solver is None else str(solver),
        )
        output = result["action_pred_normalized"].detach().cpu().numpy()
        return output[0] if output.ndim == 3 else output


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
    if args.run_dir is None:
        raise ValueError("--run-dir is required unless --architecture-only is used")
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


def build_architecture_only_context(
    args: argparse.Namespace,
) -> tuple[ArchitectureOnlyRuntimeAdapter, float, list[dict]]:
    if args.checkpoint is not None:
        raise ValueError("--checkpoint cannot be used with --architecture-only")
    device = validate_common_args(args)
    views = 3 if args.num_views is None else int(args.num_views)
    config = architecture_only_config(views)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    snapshots = [cuda_memory("before_load", device)]

    start = time.perf_counter()
    policy = FlowMatchingPolicy(
        action_dim=14,
        state_dim=14,
        cond_steps=16,
        cond_dim=256,
        use_tactile=False,
        action_horizon=64,
        n_action_steps=32,
        image_encoder_name="dinov2",
        dino_model_name="vit_small_patch14_dinov2.lvd142m",
        freeze_image_encoder=True,
        image_pretrained=False,
        image_feat_dim=256,
        n_image_views=views,
        velocity_model="unet",
        memory_enabled=True,
        memory_method="fusion",
        memory_injection="concat_global_cond",
        memory_dim=256,
        memory_history_frames=64,
        memory_recent_frame=4,
        memory_visual_layers=2,
        memory_visual_heads=4,
        memory_visual_history_length=64,
        memory_visual_sample_stride=8,
        memory_visual_recent_frame=0,
        memory_state_mem_dim=64,
        memory_dropout=0.0,
        num_inference_steps=32,
        solver="euler",
    ).to(device).eval()
    torch.cuda.synchronize(device)
    load_seconds = time.perf_counter() - start
    runtime = ArchitectureOnlyRuntimeAdapter(policy, device, config)
    after_model = cuda_memory("after_model_load", device)
    snapshots.append(after_model)
    snapshots.append({**after_model, "stage": "after_checkpoint_load"})
    runtime.load_cuda_memory = {
        "after_model_load": {key: value for key, value in after_model.items() if key != "stage"},
        "after_checkpoint_load": {
            key: value for key, value in after_model.items() if key != "stage"
        },
    }
    runtime.load_timing_ms = {
        "config_load": 0.0,
        "checkpoint_deserialize": 0.0,
        "model_build_to_device": load_seconds * 1000.0,
        "checkpoint_apply": 0.0,
        "runtime_total": load_seconds * 1000.0,
    }
    assert_model_device(policy, device)
    print_architecture_only_notice()
    return runtime, load_seconds, snapshots


def load_benchmark_context(args: argparse.Namespace):
    if args.architecture_only:
        return build_architecture_only_context(args)
    return load_runtime(args)


def load_benchmark_config(args: argparse.Namespace) -> dict[str, Any]:
    if args.architecture_only:
        views = 3 if args.num_views is None else int(args.num_views)
        return architecture_only_config(views)
    if args.run_dir is None:
        raise ValueError("--run-dir is required unless --architecture-only is used")
    return load_run_config(args.run_dir)


def print_architecture_only_notice() -> None:
    print("benchmark_mode = architecture_only")
    print("checkpoint_loaded = false")
    print("policy_weights = random_initialization")
    print("memory_weights = random_initialization")
    print("synthetic_inputs = true")
    print("latency_only = true")
    print("task_quality_not_measured = true")


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
    runtime: Any,
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
    architecture_only = bool(args.architecture_only)
    metadata = {
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
        "checkpoint_path": (
            None if runtime.checkpoint_path is None else str(runtime.checkpoint_path)
        ),
        "run_dir": None if runtime.run_dir is None else str(runtime.run_dir),
        "config_path": (
            None
            if runtime.run_dir is None
            else str(runtime.run_dir / "resolved_config.yaml")
        ),
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
    metadata.update(
        {
            "benchmark_mode": (
                "architecture_only" if architecture_only else "checkpoint"
            ),
            "checkpoint_loaded": not architecture_only,
            "policy_weights": (
                "random_initialization" if architecture_only else "checkpoint"
            ),
            "memory_weights": (
                "random_initialization" if architecture_only else "checkpoint"
            ),
            "synthetic_inputs": True,
            "latency_only": architecture_only,
            "task_quality_not_measured": architecture_only,
        }
    )
    return metadata


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
