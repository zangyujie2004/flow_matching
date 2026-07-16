"""Forward latency benchmark: DINOv2 + UNet vs DiT (B=1, deploy-like).

Modes:
  synthetic   Build policies from config (no checkpoint required).
  checkpoint  Load trained runs via FMInferenceRuntime.

Example:
  CUDA_VISIBLE_DEVICES=0 python tools/benchmark_forward_matrix.py \\
    --config configs/train/config.yaml --mode synthetic --warmup 20 --repeats 50
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from infer.runtime import FMInferenceRuntime, random_smoke_obs
from infer.tensor import numpy_obs_to_torch
from models.fm import build_flow_policy
from utils.train_utils import cfg_get, load_config

BACKBONE_SWEEP = ("unet", "dit")


def _cuda_sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.inference_mode()
def _time_fn(fn, *, device: torch.device, warmup: int, repeats: int) -> dict[str, float]:
    for _ in range(max(0, warmup)):
        fn()
    _cuda_sync(device)

    times_ms: list[float] = []
    for _ in range(max(1, repeats)):
        _cuda_sync(device)
        t0 = time.perf_counter()
        fn()
        _cuda_sync(device)
        times_ms.append((time.perf_counter() - t0) * 1000.0)

    arr = np.asarray(times_ms, dtype=np.float64)
    return {
        "mean_ms": float(arr.mean()),
        "std_ms": float(arr.std(ddof=0)),
        "p50_ms": float(np.percentile(arr, 50)),
        "p90_ms": float(np.percentile(arr, 90)),
        "min_ms": float(arr.min()),
        "max_ms": float(arr.max()),
        "repeats": float(len(arr)),
    }


def _build_obs_torch(
    *,
    device: torch.device,
    window_size: int,
    action_dim: int,
    n_image_views: int,
    use_backbone_feat: bool,
    image_size: int,
) -> dict[str, torch.Tensor]:
    state = torch.randn(1, window_size, action_dim, device=device, dtype=torch.float32)
    obs: dict[str, torch.Tensor] = {"state": state}
    if use_backbone_feat:
        obs["image_backbone_feat"] = torch.randn(
            1, 1, n_image_views, 384, device=device, dtype=torch.float32
        )
    else:
        obs["image"] = torch.randint(
            0,
            256,
            (1, 1, n_image_views, 3, image_size, image_size),
            device=device,
            dtype=torch.uint8,
        )
    return obs


@torch.inference_mode()
def benchmark_policy(
    policy,
    obs: dict[str, torch.Tensor],
    *,
    device: torch.device,
    warmup: int,
    repeats: int,
    num_inference_steps: int,
    solver: str,
) -> dict[str, Any]:
    def total():
        policy.predict_action(
            obs,
            num_inference_steps=num_inference_steps,
            solver=solver,
        )

    def encode():
        policy._build_condition(obs)

    global_cond = policy._build_condition(obs)

    def sample():
        steps = int(num_inference_steps)
        bsz = global_cond.shape[0]
        dtype = global_cond.dtype
        trajectory = torch.randn(
            bsz,
            policy.action_horizon,
            policy.action_dim,
            device=device,
            dtype=dtype,
        )
        times = torch.linspace(0.0, 1.0, steps + 1, device=device, dtype=dtype)
        for i in range(steps):
            t0 = times[i]
            t1 = times[i + 1]
            dt = t1 - t0
            velocity = policy.model(
                trajectory, t0.expand(bsz), local_cond=None, global_cond=global_cond
            )
            if solver == "heun" and i < steps - 1:
                x_euler = trajectory + dt * velocity
                velocity_next = policy.model(
                    x_euler, t1.expand(bsz), local_cond=None, global_cond=global_cond
                )
                trajectory = trajectory + 0.5 * dt * (velocity + velocity_next)
            else:
                trajectory = trajectory + dt * velocity
        return trajectory

    total_stats = _time_fn(total, device=device, warmup=warmup, repeats=repeats)
    encode_stats = _time_fn(encode, device=device, warmup=warmup, repeats=repeats)
    sample_stats = _time_fn(sample, device=device, warmup=warmup, repeats=repeats)

    return {
        "velocity_model": str(policy.velocity_model),
        "total_ms": total_stats["mean_ms"],
        "total_std_ms": total_stats["std_ms"],
        "total_p50_ms": total_stats["p50_ms"],
        "encode_ms": encode_stats["mean_ms"],
        "encode_std_ms": encode_stats["std_ms"],
        "sample_ms": sample_stats["mean_ms"],
        "sample_std_ms": sample_stats["std_ms"],
        "hz": 1000.0 / max(total_stats["mean_ms"], 1e-6),
    }


@torch.inference_mode()
def benchmark_runtime(
    runtime: FMInferenceRuntime,
    *,
    warmup: int,
    repeats: int,
    num_inference_steps: int,
    solver: str,
    seed: int,
) -> dict[str, Any]:
    obs_np, _state_raw = random_smoke_obs(runtime, seed=seed)
    obs = numpy_obs_to_torch(
        obs_np,
        runtime.device,
        use_tactile=runtime.use_tactile,
        normalizer=runtime.normalizer,
        window_size=runtime.window_size,
    )
    stats = benchmark_policy(
        runtime.policy,
        obs,
        device=runtime.device,
        warmup=warmup,
        repeats=repeats,
        num_inference_steps=num_inference_steps,
        solver=solver,
    )
    fm_cfg = runtime.policy_cfg["models"]["fm"]
    stats.update(
        {
            "mode": "checkpoint",
            "image_encoder_name": str(fm_cfg.get("image_encoder_name")),
            "dino_model_name": str(fm_cfg.get("dino_model_name")),
            "num_inference_steps": int(num_inference_steps),
            "solver": solver,
            "n_image_views": int(runtime.n_image_views),
            "action_dim": int(runtime.action_dim),
            "action_horizon": int(runtime.action_horizon),
            "device": str(runtime.device),
        }
    )
    return stats


def run_synthetic(
    cfg: dict[str, Any],
    *,
    device: str,
    warmup: int,
    repeats: int,
    num_inference_steps: int,
    solver: str,
    use_backbone_feat: bool,
) -> list[dict[str, Any]]:
    data_cfg = cfg["data"]
    fm_cfg_base = dict(cfg["models"]["fm"])
    fm_cfg_base["image_pretrained"] = False
    fm_cfg_base["use_tactile"] = bool(data_cfg.get("use_tactile", False))

    action_type = str(data_cfg.get("action_type", "joint"))
    action_dim = 14 if action_type == "joint" else 20
    window_size = int(data_cfg["window_size"])
    n_image_views = int(fm_cfg_base.get("n_image_views", 2))
    image_size = int(data_cfg.get("image_size", 224))
    torch_device = torch.device(device)

    rows: list[dict[str, Any]] = []
    for velocity_model in BACKBONE_SWEEP:
        fm_cfg = dict(fm_cfg_base)
        fm_cfg["velocity_model"] = velocity_model
        policy = build_flow_policy(
            {"models": {"fm": fm_cfg}},
            action_dim=action_dim,
            state_dim=action_dim,
            cond_steps=window_size,
        ).to(torch_device)
        policy.eval()

        obs = _build_obs_torch(
            device=torch_device,
            window_size=window_size,
            action_dim=action_dim,
            n_image_views=n_image_views,
            use_backbone_feat=use_backbone_feat,
            image_size=image_size,
        )
        stats = benchmark_policy(
            policy,
            obs,
            device=torch_device,
            warmup=warmup,
            repeats=repeats,
            num_inference_steps=num_inference_steps,
            solver=solver,
        )
        row = {
            "mode": "synthetic",
            "label_backbone": velocity_model,
            "image_encoder_name": str(fm_cfg.get("image_encoder_name")),
            "dino_model_name": str(fm_cfg.get("dino_model_name")),
            "obs_type": "image_backbone_feat" if use_backbone_feat else "uint8_image",
            "num_inference_steps": int(num_inference_steps),
            "solver": solver,
            "n_image_views": n_image_views,
            "action_dim": action_dim,
            "action_horizon": int(fm_cfg.get("action_horizon", 32)),
            "device": str(torch_device),
            **stats,
        }
        rows.append(row)
        print(
            f"{velocity_model:4s} total={row['total_ms']:.2f}±{row['total_std_ms']:.2f} ms | "
            f"encode={row['encode_ms']:.2f} | sample={row['sample_ms']:.2f} | ~{row['hz']:.1f} Hz"
        )

        del policy
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return rows


def run_checkpoint_sweep(
    *,
    run_dirs: list[Path],
    device: str | None,
    warmup: int,
    repeats: int,
    num_inference_steps: int,
    solver: str,
    seed: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        ckpt = run_dir / "checkpoints" / "latest.pt"
        print(f"\n=== checkpoint: {run_dir.name} ===")
        if not ckpt.is_file():
            print(f"[skip] missing checkpoint: {ckpt}")
            rows.append({"run_dir": str(run_dir), "error": f"missing {ckpt}"})
            continue

        runtime = FMInferenceRuntime(run_dir, device=device, warmup=True)
        stats = benchmark_runtime(
            runtime,
            warmup=warmup,
            repeats=repeats,
            num_inference_steps=num_inference_steps,
            solver=solver,
            seed=seed,
        )
        row = {"run_dir": str(run_dir), "label_backbone": stats["velocity_model"], **stats}
        rows.append(row)
        print(
            f"{row['label_backbone']:4s} total={row['total_ms']:.2f}±{row['total_std_ms']:.2f} ms | "
            f"encode={row['encode_ms']:.2f} | sample={row['sample_ms']:.2f} | ~{row['hz']:.1f} Hz"
        )
        del runtime
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return rows


def _print_table(title: str, rows: list[dict[str, Any]]) -> None:
    print(f"\n######## {title} ########")
    hdr = f"{'backbone':8s} {'total_ms':>10s} {'encode_ms':>10s} {'sample_ms':>10s} {'Hz':>8s}"
    print(hdr)
    print("-" * len(hdr))
    for row in rows:
        if "error" in row:
            print(f"{row.get('label_backbone', '?'):8s}  ERROR: {row['error']}")
            continue
        print(
            f"{row['label_backbone']:8s} "
            f"{row['total_ms']:10.2f} {row['encode_ms']:10.2f} {row['sample_ms']:10.2f} {row['hz']:8.1f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/train/config.yaml")
    parser.add_argument("--mode", type=str, default="synthetic", choices=["synthetic", "checkpoint"])
    parser.add_argument("--run-dir", type=str, action="append", default=[])
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--num-inference-steps", type=int, default=16)
    parser.add_argument("--solver", type=str, default="euler")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--obs-type",
        type=str,
        default="image_backbone_feat",
        choices=["image_backbone_feat", "uint8_image"],
        help="Synthetic mode only. backbone_feat matches cached-latent train path.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(_ROOT / "outputs" / "exp" / "forward_bench"),
    )
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    protocol = {
        "mode": args.mode,
        "batch_size": 1,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "num_inference_steps": args.num_inference_steps,
        "solver": args.solver,
        "seed": args.seed,
        "timing": "cuda.synchronize + perf_counter",
        "note": "encode = condition; sample = ODE only; total = full predict_action",
    }

    if args.mode == "synthetic":
        config_path = Path(args.config)
        if not config_path.is_absolute():
            config_path = _ROOT / config_path
        cfg = load_config(str(config_path))
        rows = run_synthetic(
            cfg,
            device=device,
            warmup=args.warmup,
            repeats=args.repeats,
            num_inference_steps=args.num_inference_steps,
            solver=args.solver,
            use_backbone_feat=args.obs_type == "image_backbone_feat",
        )
        title = "DINOv2 synthetic: DiT vs UNet"
        stem = "dinov2_synthetic"
    else:
        if not args.run_dir:
            raise SystemExit("checkpoint mode requires at least one --run-dir")
        run_dirs = [Path(p).expanduser().resolve() for p in args.run_dir]
        rows = run_checkpoint_sweep(
            run_dirs=run_dirs,
            device=device,
            warmup=args.warmup,
            repeats=args.repeats,
            num_inference_steps=args.num_inference_steps,
            solver=args.solver,
            seed=args.seed,
        )
        title = "Checkpoint forward bench"
        stem = "checkpoint"

    _print_table(title, rows)

    payload = {"protocol": protocol, "rows": rows}
    json_path = out_dir / f"forward_bench_{stem}.json"
    csv_path = out_dir / f"forward_bench_{stem}.csv"
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    if rows:
        with open(csv_path, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=sorted({k for row in rows for k in row.keys()}))
            writer.writeheader()
            writer.writerows(rows)
    print(f"\nSaved: {json_path}")
    if rows:
        print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()
