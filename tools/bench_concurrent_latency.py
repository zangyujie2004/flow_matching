from __future__ import annotations

import argparse
import csv
import gc
import json
import queue
import time
from datetime import datetime

import numpy as np
import torch
import torch.multiprocessing as mp

from infer.runtime import random_smoke_obs
from infer.tensor import numpy_obs_to_torch
from tools.async_dino_buffer import AsyncDinoBuffer
from tools.latency_benchmark_utils import (
    MIB,
    add_common_arguments,
    assert_finite,
    create_result_dir,
    cuda_memory,
    distribution,
    load_benchmark_config,
    load_benchmark_context,
    runtime_metadata,
    save_results,
)


def prepare_role(role, runtime, args):
    device = runtime.device
    policy = runtime.policy
    b = args.batch_size
    v = runtime.n_image_views
    c = int(policy.condition_encoder.image_encoder.encoder.head[0].normalized_shape[0])
    t = int(runtime.memory_visual_offsets.numel())

    if role == "dino":
        image_size = int(runtime.policy_cfg["data"].get("image_size", 224))
        images = [
            torch.randint(
                0,
                256,
                (b, 3, image_size, image_size),
                dtype=torch.uint8,
                device=device,
            )
            for _ in range(v)
        ]
        buffer = AsyncDinoBuffer(
            policy.condition_encoder.image_encoder.encoder,
            device=str(device),
            store_local_features=False,
        )
        run_once = lambda: buffer._run_dino(images)[:2]
        shapes = {"images": [list(image.shape) for image in images]}
        return run_once, shapes

    memory_obs = {
        "memory_image_backbone_feat": torch.randn(b, t, v, c, device=device),
        "memory_state": torch.randn(
            b, policy.memory_history_frames, policy.state_dim, device=device
        ),
        "memory_visual_offsets": runtime.memory_visual_offsets,
    }
    if role == "memory":
        if not policy.memory_enabled:
            raise RuntimeError("checkpoint policy does not enable Memory")
        run_once = lambda: policy._build_memory(memory_obs)
        shapes = {name: list(value.shape) for name, value in memory_obs.items()}
        return run_once, shapes

    if role != "policy":
        raise ValueError(f"unknown concurrent benchmark role {role!r}")
    obs_numpy, _state_raw = random_smoke_obs(runtime, seed=args.seed)
    if b > 1:
        obs_numpy["image"] = np.repeat(obs_numpy["image"], b, axis=0)
        obs_numpy["state"] = np.repeat(obs_numpy["state"][None], b, axis=0)
        if "tactile" in obs_numpy:
            obs_numpy["tactile"] = np.repeat(obs_numpy["tactile"][None], b, axis=0)
    obs = numpy_obs_to_torch(
        obs_numpy,
        device,
        use_tactile=runtime.use_tactile,
        normalizer=runtime.normalizer,
        window_size=runtime.window_size,
    )
    obs.update(memory_obs)
    run_once = lambda: policy.predict_action(
        obs,
        num_inference_steps=policy.num_inference_steps,
        solver=policy.solver,
    )
    shapes = {name: list(value.shape) for name, value in obs.items()}
    return run_once, shapes


def role_rate(role, args):
    return {
        "dino": args.dino_rate_hz,
        "memory": args.memory_rate_hz,
        "policy": args.policy_rate_hz,
    }[role]


def worker(role, phase, args, barrier, result_queue, launched_at):
    runtime = None
    try:
        startup_seconds = time.perf_counter() - launched_at
        runtime, load_seconds, memory_snapshots = load_benchmark_context(args)
        device = runtime.device
        run_once, shapes = prepare_role(role, runtime, args)
        process_metadata = runtime_metadata(
            runtime,
            args,
            benchmark=f"concurrent_{phase}_{role}",
            load_seconds=load_seconds,
            input_shapes=shapes,
        )
        if role == "dino":
            # run_once retains the real DINO module; release unrelated Policy modules.
            runtime.policy = None
            gc.collect()
            torch.cuda.empty_cache()
        with torch.inference_mode():
            first = run_once()
            assert_finite(first)
            for _ in range(args.warmup_iterations):
                run_once()
        torch.cuda.synchronize(device)
        after_warmup = cuda_memory("after_warmup", device)
        memory_snapshots.append(after_warmup)
        torch.cuda.reset_peak_memory_stats(device)
        memory_snapshots.append(cuda_memory("before_formal_test", device))
        if barrier is not None:
            barrier.wait(timeout=600)

        rows = []
        rate_hz = role_rate(role, args)
        period = 1.0 / rate_hz
        start_time = time.perf_counter()
        next_time = start_time
        iteration = 0
        deadline_misses = 0
        while time.perf_counter() - start_time < args.duration_seconds:
            begin = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize(device)
            wall_start = time.perf_counter()
            begin.record()
            with torch.inference_mode():
                run_once()
            end.record()
            end.synchronize()
            wall_ms = (time.perf_counter() - wall_start) * 1000.0
            cuda_ms = begin.elapsed_time(end)
            deadline_misses += wall_ms > period * 1000.0
            rows.append(
                {
                    "phase": phase,
                    "role": role,
                    "iteration": iteration,
                    "cuda_ms": float(cuda_ms),
                    "wall_ms": float(wall_ms),
                    "allocated_mib": torch.cuda.memory_allocated(device) / MIB,
                    "reserved_mib": torch.cuda.memory_reserved(device) / MIB,
                    "timestamp": datetime.now().astimezone().isoformat(),
                }
            )
            iteration += 1
            next_time += period
            time.sleep(max(0.0, next_time - time.perf_counter()))

        elapsed = time.perf_counter() - start_time
        memory_snapshots.append(cuda_memory("peak_during_test", device))
        memory_snapshots.append(cuda_memory("after_test", device))
        result = {
            "status": "ok",
            "phase": phase,
            "role": role,
            "rows": rows,
            "wall_ms": distribution([row["wall_ms"] for row in rows]),
            "cuda_ms": distribution([row["cuda_ms"] for row in rows]),
            "throughput_hz": len(rows) / elapsed,
            "deadline_misses": int(deadline_misses),
            "startup_seconds": float(startup_seconds),
            "model_load_seconds": float(load_seconds),
            "input_shapes": shapes,
            "memory_after_warmup": after_warmup,
            "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / MIB,
            "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / MIB,
            "memory_snapshots": memory_snapshots,
            "metadata": process_metadata,
        }
        del first, run_once, runtime
        gc.collect()
        torch.cuda.empty_cache()
        cleanup_memory = cuda_memory("after_cleanup", device)
        result["memory_after_cleanup"] = cleanup_memory
        result["memory_snapshots"].append(cleanup_memory)
        result_queue.put(result)
    except Exception as error:
        if barrier is not None:
            barrier.abort()
        is_oom = isinstance(error, torch.cuda.OutOfMemoryError) or "out of memory" in str(error).lower()
        result_queue.put(
            {
                "status": "error",
                "phase": phase,
                "role": role,
                "error": f"{type(error).__name__}: {error}",
                "oom": bool(is_oom),
            }
        )


def run_processes(ctx, roles, phase, args, concurrent):
    result_queue = ctx.Queue()
    barrier = ctx.Barrier(len(roles)) if concurrent else None
    processes = []
    results = []
    if concurrent:
        launched_at = time.perf_counter()
        for role in roles:
            process = ctx.Process(
                target=worker,
                args=(role, phase, args, barrier, result_queue, launched_at),
            )
            process.start()
            processes.append(process)
    else:
        for role in roles:
            launched_at = time.perf_counter()
            process = ctx.Process(
                target=worker,
                args=(role, phase, args, None, result_queue, launched_at),
            )
            process.start()
            try:
                results.append(result_queue.get(timeout=900))
            except queue.Empty:
                results.append(
                    {"status": "error", "phase": phase, "role": role, "error": "timeout"}
                )
            process.join()
            processes.append(process)

    if concurrent:
        for _ in roles:
            try:
                results.append(result_queue.get(timeout=900))
            except queue.Empty:
                results.append(
                    {"status": "error", "phase": phase, "role": "unknown", "error": "timeout"}
                )
    for process in processes:
        process.join(timeout=30)
        if process.is_alive():
            process.terminate()
            process.join()
    return results


def write_process_csv(path, rows):
    fields = [
        "phase",
        "role",
        "iteration",
        "cuda_ms",
        "wall_ms",
        "allocated_mib",
        "reserved_mib",
        "timestamp",
    ]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    add_common_arguments(parser)
    parser.add_argument("--scenario", choices=("realistic", "stress"), default="realistic")
    parser.add_argument("--duration-seconds", type=float, default=30.0)
    parser.add_argument("--dino-rate-hz", type=float, default=7.5)
    parser.add_argument("--policy-rate-hz", type=float, default=10.0)
    parser.add_argument("--memory-rate-hz", type=float, default=10.0)
    args = parser.parse_args()
    if args.duration_seconds <= 0:
        raise ValueError("duration-seconds must be positive")

    mp.set_start_method("spawn", force=True)
    ctx = mp.get_context("spawn")
    roles = ["dino", "policy"] if args.scenario == "realistic" else ["dino", "memory", "policy"]
    isolated = run_processes(ctx, roles, "isolated", args, concurrent=False)
    concurrent = run_processes(ctx, roles, "concurrent", args, concurrent=True)
    results = isolated + concurrent

    errors = [result for result in results if result["status"] != "ok"]
    summary = {"scenario": args.scenario, "errors": errors}
    all_rows = []
    for role in roles:
        isolated_result = next(
            (item for item in isolated if item.get("role") == role and item["status"] == "ok"),
            None,
        )
        concurrent_result = next(
            (item for item in concurrent if item.get("role") == role and item["status"] == "ok"),
            None,
        )
        if isolated_result is None or concurrent_result is None:
            continue
        isolated_p95 = isolated_result["wall_ms"]["p95"]
        concurrent_p95 = concurrent_result["wall_ms"]["p95"]
        summary[role] = {
            "isolated": {key: value for key, value in isolated_result.items() if key != "rows"},
            "concurrent": {key: value for key, value in concurrent_result.items() if key != "rows"},
            "p95_slowdown_percent": (concurrent_p95 / isolated_p95 - 1.0) * 100.0,
        }
        all_rows.extend(isolated_result["rows"])
        all_rows.extend(concurrent_result["rows"])
    summary["total_concurrent_peak_allocated_mib"] = sum(
        item.get("peak_allocated_mib", 0.0)
        for item in concurrent
        if item["status"] == "ok"
    )
    summary["oom"] = any(item.get("oom", False) for item in errors)

    result_dir = create_result_dir(args, f"concurrent_{args.scenario}")
    first_success = next((item for item in results if item["status"] == "ok"), None)
    metadata = dict(first_success["metadata"]) if first_success is not None else {}
    metadata.update({
        "benchmark": "concurrent_latency",
        "scenario": "three_process_stress" if args.scenario == "stress" else "realistic",
        "run_dir": None if args.run_dir is None else str(args.run_dir),
        "checkpoint": (
            None
            if args.architecture_only
            else str(args.checkpoint) if args.checkpoint else "auto latest.pt"
        ),
        "start_method": "spawn",
        "duration_seconds": args.duration_seconds,
        "roles": roles,
        "timestamp": datetime.now().astimezone().isoformat(),
    })
    config = load_benchmark_config(args)
    standard_summary = {}
    for role in roles:
        if role in summary:
            standard_summary[f"{role}_isolated"] = {
                "wall_ms": summary[role]["isolated"]["wall_ms"],
                "cuda_ms": summary[role]["isolated"]["cuda_ms"],
            }
            standard_summary[f"{role}_concurrent"] = {
                "wall_ms": summary[role]["concurrent"]["wall_ms"],
                "cuda_ms": summary[role]["concurrent"]["cuda_ms"],
            }
    save_results(
        result_dir,
        metadata=metadata,
        summary=standard_summary,
        rows=all_rows,
        config=config,
        memory_snapshots=[],
    )
    with open(result_dir / "concurrent_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    for role in ("dino", "memory", "policy"):
        write_process_csv(
            result_dir / f"process_{role}.csv",
            [row for row in all_rows if row["role"] == role],
        )
    if errors:
        raise RuntimeError(f"concurrent benchmark failed: {errors}")


if __name__ == "__main__":
    main()
