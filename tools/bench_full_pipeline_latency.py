import argparse
import gc
import threading
import time
from datetime import datetime
from types import SimpleNamespace

import numpy as np
import torch

from infer.runtime import random_smoke_obs
from tools.latency_benchmark_utils import (
    MIB,
    add_common_arguments,
    benchmark_callable,
    create_result_dir,
    cuda_memory,
    distribution,
    finalize_memory_snapshots,
    load_runtime,
    runtime_metadata,
    save_results,
)


def make_camera_frames(runtime, count: int, seed: int):
    rng = np.random.default_rng(seed)
    cfg = runtime.async_dino_preprocess
    frames = []
    for _ in range(count):
        samples = {
            name: SimpleNamespace(
                data=rng.integers(
                    0,
                    256,
                    (cfg.image_size, cfg.image_size, 3),
                    dtype=np.uint8,
                )
            )
            for name in cfg.camera_views
        }
        frames.append(SimpleNamespace(samples=samples))
    return frames


def wait_for_processed(buffer, target: int, timeout_s: float = 60.0) -> None:
    deadline = time.perf_counter() + timeout_s
    while buffer.get_stats()["processed_count"] < target:
        if time.perf_counter() >= deadline:
            raise TimeoutError(f"Async DINO did not reach processed_count={target}")
        time.sleep(0.005)


def timed_rows(name, values, gpu_values=None, allocated_mib=0.0, reserved_mib=0.0):
    rows = []
    gpu_values = gpu_values if gpu_values is not None else [0.0] * len(values)
    for index, (wall_ms, cuda_ms) in enumerate(zip(values, gpu_values)):
        rows.append(
            {
                "benchmark": name,
                "iteration": index,
                "cuda_ms": float(cuda_ms),
                "wall_ms": float(wall_ms),
                "allocated_mib": float(allocated_mib),
                "reserved_mib": float(reserved_mib),
                "timestamp": datetime.now().astimezone().isoformat(),
            }
        )
    return rows


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    add_common_arguments(parser)
    parser.add_argument("--camera-period-ms", type=float, default=33.0)
    parser.add_argument("--dino-sample-interval-frames", type=int, default=4)
    args = parser.parse_args()
    if args.batch_size != 1:
        raise ValueError("FMInferenceRuntime deployment entry currently requires batch-size=1")

    runtime, load_seconds, memory_snapshots = load_runtime(args)
    if not runtime.policy.memory_enabled:
        raise RuntimeError("checkpoint policy does not enable Memory")
    device = runtime.device
    buffer = runtime.start_async_dino(
        sample_interval_frames=args.dino_sample_interval_frames,
        deadline_ms=args.camera_period_ms * args.dino_sample_interval_frames,
    )
    frames = make_camera_frames(runtime, 16, args.seed)

    # Initial history fill is setup, not part of formal latency statistics.
    for sample_index in range(16):
        frame_id = sample_index * args.dino_sample_interval_frames
        accepted = runtime.submit_async_dino_frame(
            frame_id,
            frames[sample_index % len(frames)],
            capture_time=time.perf_counter(),
        )
        if not accepted:
            raise AssertionError(f"initial sampled frame {frame_id} was rejected")
        wait_for_processed(buffer, sample_index + 1)
    if buffer.get_feature_window() is None:
        raise AssertionError("Async DINO Buffer must contain 16 real processed samples")

    obs, state_raw = random_smoke_obs(runtime, seed=args.seed)
    rng = np.random.default_rng(args.seed + 1)
    memory_state_raw = rng.normal(
        0.0,
        0.05,
        (runtime.policy.memory_history_frames, runtime.action_dim),
    ).astype(np.float32)
    action = runtime.predict_rot6d_abs(
        obs,
        state_raw=state_raw,
        memory_state_raw=memory_state_raw,
        num_inference_steps=runtime.num_inference_steps,
        solver=runtime.solver,
    )
    if action.shape != (runtime.action_horizon, runtime.action_dim):
        raise AssertionError(f"action shape {action.shape} is invalid")
    for _ in range(args.warmup_iterations):
        runtime.predict_rot6d_abs(
            obs,
            state_raw=state_raw,
            memory_state_raw=memory_state_raw,
            num_inference_steps=runtime.num_inference_steps,
            solver=runtime.solver,
        )
    torch.cuda.synchronize(device)

    with buffer.lock:
        dino_start = len(buffer.stage_wall_times)
        dropped_start = buffer.dropped_count
        processed_start = buffer.processed_count

    stop_camera = threading.Event()
    camera_submit_times = []
    camera_lock = threading.Lock()
    first_formal_frame = 16 * args.dino_sample_interval_frames

    def camera_loop():
        frame_id = first_formal_frame
        next_time = time.perf_counter()
        while not stop_camera.is_set():
            start = time.perf_counter()
            runtime.submit_async_dino_frame(
                frame_id,
                frames[frame_id % len(frames)],
                capture_time=start,
            )
            elapsed = (time.perf_counter() - start) * 1000.0
            with camera_lock:
                camera_submit_times.append(elapsed)
            frame_id += 1
            next_time += args.camera_period_ms / 1000.0
            stop_camera.wait(max(0.0, next_time - time.perf_counter()))

    memory_events = []

    original_build_memory = runtime.policy._build_memory

    def timed_build_memory(memory_obs):
        begin = torch.cuda.Event(enable_timing=True)
        begin.record()
        output = original_build_memory(memory_obs)
        end = torch.cuda.Event(enable_timing=True)
        end.record()
        memory_events.append([begin, end])
        return output

    runtime.policy._build_memory = timed_build_memory
    camera_thread = threading.Thread(target=camera_loop, daemon=True)
    camera_thread.start()

    policy_summary, policy_rows = benchmark_callable(
        "policy_predict_ms",
        lambda: runtime.predict_rot6d_abs(
            obs,
            state_raw=state_raw,
            memory_state_raw=memory_state_raw,
            num_inference_steps=runtime.num_inference_steps,
            solver=runtime.solver,
        ),
        device=device,
        warmup_iterations=0,
        iterations=args.iterations,
        memory_snapshots=memory_snapshots,
        correctness_run=False,
    )
    stop_camera.set()
    camera_thread.join()
    delattr(runtime.policy, "_build_memory")
    runtime.stop_async_dino()
    torch.cuda.synchronize(device)

    memory_gpu_ms = [begin.elapsed_time(end) for begin, end in memory_events]
    with buffer.lock:
        dino_stage = list(buffer.stage_wall_times[dino_start:])
        dino_end_to_end = list(buffer.end_to_end_times[dino_start:])
        dino_gpu = list(buffer.gpu_total_times[dino_start:])
        dropped_delta = buffer.dropped_count - dropped_start
        processed_delta = buffer.processed_count - processed_start
        buffer_length = len(buffer.buffer)
        deadline_values = [value > buffer.deadline_ms for value in dino_stage]
    with camera_lock:
        submit_values = list(camera_submit_times)
    current_allocated = torch.cuda.memory_allocated(device) / MIB
    current_reserved = torch.cuda.memory_reserved(device) / MIB

    summaries = {
        "policy_predict_ms": policy_summary,
        "full_request_wall_ms": policy_summary,
        "memory_build_ms": {
            "cuda_ms": distribution(memory_gpu_ms),
            "wall_ms": distribution(memory_gpu_ms),
        },
    }
    rows = list(policy_rows)
    rows.extend(
        timed_rows(
            "memory_build_ms",
            memory_gpu_ms,
            memory_gpu_ms,
            current_allocated,
            current_reserved,
        )
    )
    if submit_values:
        summaries["camera_submit_ms"] = {
            "cuda_ms": distribution([0.0] * len(submit_values)),
            "wall_ms": distribution(submit_values),
        }
        rows.extend(
            timed_rows(
                "camera_submit_ms",
                submit_values,
                allocated_mib=current_allocated,
                reserved_mib=current_reserved,
            )
        )
    if dino_stage:
        summaries["async_dino_stage_ms"] = {
            "cuda_ms": distribution(dino_gpu),
            "wall_ms": distribution(dino_stage),
        }
        summaries["async_dino_end_to_end_ms"] = {
            "cuda_ms": distribution(dino_gpu),
            "wall_ms": distribution(dino_end_to_end),
        }
        rows.extend(
            timed_rows(
                "async_dino_stage_ms",
                dino_stage,
                dino_gpu,
                current_allocated,
                current_reserved,
            )
        )
        rows.extend(
            timed_rows(
                "async_dino_end_to_end_ms",
                dino_end_to_end,
                dino_gpu,
                current_allocated,
                current_reserved,
            )
        )

    stats = {
        "dino_deadline_miss_count": int(sum(deadline_values)),
        "dino_dropped_count": int(dropped_delta),
        "dino_processed_count": int(processed_delta),
        "buffer_length": int(buffer_length),
        "camera_frames_submitted": len(submit_values),
    }
    summaries["pipeline_counts"] = stats
    shapes = {
        "current_image": list(np.asarray(obs["image"]).shape),
        "current_state": list(np.asarray(obs["state"]).shape),
        "memory_state_raw": list(memory_state_raw.shape),
        "feature_window": list(buffer.get_feature_window().shape),
        "action": list(action.shape),
    }
    metadata = runtime_metadata(
        runtime,
        args,
        benchmark="full_pipeline_latency",
        load_seconds=load_seconds,
        input_shapes=shapes,
    )
    metadata.update(
        {
            "camera_period_ms": args.camera_period_ms,
            "dino_sample_interval_frames": args.dino_sample_interval_frames,
            "pipeline_counts": stats,
            "note": "Camera feeder does not wait for DINO; latest-only drops are measured.",
        }
    )
    print(stats)

    result_dir = create_result_dir(args, "full_pipeline")
    config = runtime.policy_cfg
    finalize_memory_snapshots(memory_snapshots, device)
    del runtime, buffer, frames, action, obs, memory_state_raw
    del camera_loop, camera_thread, timed_build_memory, original_build_memory
    gc.collect()
    torch.cuda.empty_cache()
    memory_snapshots.append(cuda_memory("after_cleanup", device))
    save_results(
        result_dir,
        metadata=metadata,
        summary=summaries,
        rows=rows,
        config=config,
        memory_snapshots=memory_snapshots,
    )


if __name__ == "__main__":
    main()
