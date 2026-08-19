#!/usr/bin/env python3
"""Generate a configurable synthetic CUDA memory and compute load.

This utility is intended for GPU burn-in, monitoring checks, and scheduler
experiments.  It does not make another training process faster; running it next
to training competes for compute and memory.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import signal
import sys
import time
from dataclasses import dataclass

import torch


MIB = 1024 * 1024


@dataclass(frozen=True)
class WorkerConfig:
    device_index: int
    memory_util: float
    matrix_size: int
    dtype_name: str
    duration: float
    duty_cycle: float
    log_interval: float
    memory_chunk_mib: int
    safety_mib: int
    seed: int


def _dtype(name: str) -> torch.dtype:
    choices = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    return choices[name]


def _format_gib(value: int) -> str:
    return f"{value / (1024**3):.2f} GiB"


def _allocate_memory_to_target(
    device: torch.device,
    *,
    target_util: float,
    chunk_mib: int,
    safety_mib: int,
) -> list[torch.Tensor]:
    """Allocate and touch byte tensors until total device use nears target."""
    if target_util <= 0:
        return []

    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    used_bytes = total_bytes - free_bytes
    target_used_bytes = int(total_bytes * target_util)
    safe_free_bytes = max(0, free_bytes - safety_mib * MIB)
    requested_bytes = max(0, target_used_bytes - used_bytes)
    bytes_to_allocate = min(requested_bytes, safe_free_bytes)
    chunk_bytes = max(1, chunk_mib) * MIB

    buffers: list[torch.Tensor] = []
    remaining = bytes_to_allocate
    while remaining > 0:
        allocation_size = min(chunk_bytes, remaining)
        try:
            buffer = torch.empty(allocation_size, dtype=torch.uint8, device=device)
            buffer.zero_()  # Materialize the allocation instead of leaving it lazy.
            buffers.append(buffer)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(
                f"[cuda:{device.index}] stopped memory allocation after "
                f"{_format_gib(sum(item.numel() for item in buffers))}: OOM avoided",
                flush=True,
            )
            break
        remaining -= allocation_size
    return buffers


def _worker(cfg: WorkerConfig) -> None:
    device = torch.device("cuda", cfg.device_index)
    torch.cuda.set_device(device)
    torch.manual_seed(cfg.seed + cfg.device_index)
    torch.cuda.manual_seed(cfg.seed + cfg.device_index)
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = True

    name = torch.cuda.get_device_name(device)
    dtype = _dtype(cfg.dtype_name)
    print(
        f"[cuda:{cfg.device_index}] starting on {name}; "
        f"matrix={cfg.matrix_size}x{cfg.matrix_size}, dtype={cfg.dtype_name}",
        flush=True,
    )

    try:
        # Allocate compute tensors first so the optional filler cannot starve the
        # matrix multiplication workspace.
        a = torch.randn(
            cfg.matrix_size,
            cfg.matrix_size,
            device=device,
            dtype=dtype,
        )
        b = torch.randn_like(a)
        output = torch.empty_like(a)
        torch.mm(a, b, out=output)
        torch.cuda.synchronize(device)

        buffers = _allocate_memory_to_target(
            device,
            target_util=cfg.memory_util,
            chunk_mib=cfg.memory_chunk_mib,
            safety_mib=cfg.safety_mib,
        )
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        print(
            f"[cuda:{cfg.device_index}] ready; device used="
            f"{_format_gib(total_bytes - free_bytes)}/{_format_gib(total_bytes)}, "
            f"filler={_format_gib(sum(item.numel() for item in buffers))}",
            flush=True,
        )

        started = time.monotonic()
        last_log = started
        iterations = 0
        last_iterations = 0
        duty_window_seconds = 1.0

        while cfg.duration == 0 or time.monotonic() - started < cfg.duration:
            window_start = time.monotonic()
            active_until = window_start + duty_window_seconds * cfg.duty_cycle

            while time.monotonic() < active_until:
                torch.mm(a, b, out=output)
                iterations += 1
                # Periodic synchronization prevents an unbounded CUDA queue and
                # makes duty-cycle timing representative of actual GPU work.
                if iterations % 16 == 0:
                    torch.cuda.synchronize(device)

            torch.cuda.synchronize(device)
            window_end = window_start + duty_window_seconds
            remaining_sleep = window_end - time.monotonic()
            if remaining_sleep > 0:
                time.sleep(remaining_sleep)

            now = time.monotonic()
            if now - last_log >= cfg.log_interval:
                elapsed = now - last_log
                completed = iterations - last_iterations
                allocated = torch.cuda.memory_allocated(device)
                reserved = torch.cuda.memory_reserved(device)
                print(
                    f"[cuda:{cfg.device_index}] {completed / elapsed:.2f} matmul/s; "
                    f"allocated={_format_gib(allocated)}, "
                    f"reserved={_format_gib(reserved)}",
                    flush=True,
                )
                last_log = now
                last_iterations = iterations

        torch.cuda.synchronize(device)
        print(
            f"[cuda:{cfg.device_index}] finished after {iterations} matmuls",
            flush=True,
        )
    except KeyboardInterrupt:
        print(f"[cuda:{cfg.device_index}] interrupted", flush=True)
    except torch.cuda.OutOfMemoryError as exc:
        print(
            f"[cuda:{cfg.device_index}] CUDA OOM: reduce --memory-util or "
            f"--matrix-size ({exc})",
            file=sys.stderr,
            flush=True,
        )
        raise


def _parse_devices(value: str) -> list[int]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in this Python environment")
    count = torch.cuda.device_count()
    text = value.strip().lower()
    if text == "all":
        return list(range(count))
    devices = [int(item.strip()) for item in text.split(",") if item.strip()]
    if not devices:
        raise ValueError("--devices must be 'all' or a comma-separated list")
    if len(set(devices)) != len(devices):
        raise ValueError(f"duplicate CUDA device in --devices={value!r}")
    invalid = [index for index in devices if index < 0 or index >= count]
    if invalid:
        raise ValueError(
            f"invalid visible CUDA indices {invalid}; this process sees {count} GPU(s)"
        )
    return devices


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create synthetic matrix-multiply and memory load on CUDA GPUs."
    )
    parser.add_argument(
        "--devices",
        default="0",
        help="Visible CUDA indices, for example 0,1, or 'all' (default: 0).",
    )
    parser.add_argument(
        "--memory-util",
        type=float,
        default=0.80,
        help="Target total device-memory utilization in [0, 0.98]; 0 disables filler.",
    )
    parser.add_argument("--matrix-size", type=int, default=8192)
    parser.add_argument(
        "--dtype",
        choices=("float16", "bfloat16", "float32"),
        default="float16",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=0,
        help="Seconds to run; 0 means until Ctrl-C (default: 0).",
    )
    parser.add_argument(
        "--duty-cycle",
        type=float,
        default=1.0,
        help="Compute-active fraction in [0, 1] for each one-second window.",
    )
    parser.add_argument("--log-interval", type=float, default=5.0)
    parser.add_argument("--memory-chunk-mib", type=int, default=256)
    parser.add_argument(
        "--safety-mib",
        type=int,
        default=512,
        help="Free memory to leave untouched after filler allocation.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if not 0 <= args.memory_util <= 0.98:
        raise ValueError("--memory-util must be between 0 and 0.98")
    if args.matrix_size <= 0:
        raise ValueError("--matrix-size must be positive")
    if args.duration < 0:
        raise ValueError("--duration must be non-negative")
    if not 0 <= args.duty_cycle <= 1:
        raise ValueError("--duty-cycle must be between 0 and 1")
    if args.log_interval <= 0:
        raise ValueError("--log-interval must be positive")
    if args.memory_chunk_mib <= 0 or args.safety_mib < 0:
        raise ValueError("memory chunk must be positive and safety margin non-negative")

    devices = _parse_devices(args.devices)
    configs = [
        WorkerConfig(
            device_index=device_index,
            memory_util=args.memory_util,
            matrix_size=args.matrix_size,
            dtype_name=args.dtype,
            duration=args.duration,
            duty_cycle=args.duty_cycle,
            log_interval=args.log_interval,
            memory_chunk_mib=args.memory_chunk_mib,
            safety_mib=args.safety_mib,
            seed=args.seed,
        )
        for device_index in devices
    ]

    if len(configs) == 1:
        _worker(configs[0])
        return

    context = mp.get_context("spawn")
    workers = [context.Process(target=_worker, args=(cfg,)) for cfg in configs]
    for worker in workers:
        worker.start()
    try:
        exit_codes = []
        for worker in workers:
            worker.join()
            exit_codes.append(worker.exitcode)
    except KeyboardInterrupt:
        print("Stopping GPU workers...", flush=True)
        for worker in workers:
            if worker.is_alive():
                os.kill(worker.pid, signal.SIGINT)
        for worker in workers:
            worker.join(timeout=10)
        for worker in workers:
            if worker.is_alive():
                worker.terminate()
                worker.join()
        return

    failed = [code for code in exit_codes if code not in {0, None}]
    if failed:
        raise SystemExit(f"GPU worker failure exit codes: {failed}")


if __name__ == "__main__":
    main()
