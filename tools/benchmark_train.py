"""Profile per-step training time breakdown."""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_POLICY_ROOT = Path(__file__).resolve().parents[1]
if str(_POLICY_ROOT) not in sys.path:
    sys.path.insert(0, str(_POLICY_ROOT))

import numpy as np
import torch

os.environ.setdefault("HF_HUB_OFFLINE", "1")

from trainers.policy_trainer import build_dataset_and_loader, build_policy
from utils.tensor import move_to_device
from utils.train_utils import load_config, set_seed


@dataclass
class Timings:
    samples: list[float] = field(default_factory=list)

    def add(self, seconds: float) -> None:
        self.samples.append(seconds)

    def summary(self) -> dict[str, float]:
        arr = np.asarray(self.samples, dtype=np.float64)
        if arr.size == 0:
            return {"mean_ms": 0.0, "std_ms": 0.0, "total_s": 0.0}
        return {
            "mean_ms": float(arr.mean() * 1000.0),
            "std_ms": float(arr.std() * 1000.0),
            "p50_ms": float(np.percentile(arr, 50) * 1000.0),
            "p95_ms": float(np.percentile(arr, 95) * 1000.0),
            "total_s": float(arr.sum()),
        }


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _cuda_timer(device: torch.device):
    if device.type != "cuda":
        return None, None
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    return start, end


def _elapsed_ms(start, end) -> float:
    if start is None:
        return 0.0
    end.synchronize()
    return float(start.elapsed_time(end))


def profile_forward_components(policy, batch: dict[str, Any], device: torch.device, repeats: int = 20) -> dict[str, float]:
    obs = batch["obs"]
    enc = policy.condition_encoder
    policy.eval()

    image = obs.get("image")
    image_backbone_feat = obs.get("image_backbone_feat")
    state = obs["state"]
    tactile = obs.get("tactile")

    results: dict[str, list[float]] = {
        "dino": [],
        "tactile": [],
        "state": [],
        "fusion": [],
        "unet": [],
        "loss_total": [],
    }

    with torch.no_grad():
        for _ in range(repeats):
            _sync(device)
            t0 = time.perf_counter()
            if image_backbone_feat is not None:
                vis = enc.image_encoder.encode_from_backbone_feat(image_backbone_feat)
            else:
                vis = enc.image_encoder(image)
            _sync(device)
            results["dino"].append(time.perf_counter() - t0)

            if enc.use_tactile and tactile is not None:
                _sync(device)
                t0 = time.perf_counter()
                tac = enc.tactile_encoder(tactile)
                _sync(device)
                results["tactile"].append(time.perf_counter() - t0)
            else:
                tac = None

            _sync(device)
            t0 = time.perf_counter()
            st = enc.state_encoder(state)
            _sync(device)
            results["state"].append(time.perf_counter() - t0)

            parts = [vis, st] if tac is None else [vis, tac, st]
            _sync(device)
            t0 = time.perf_counter()
            global_cond = enc.fusion(torch.cat(parts, dim=-1))
            _sync(device)
            results["fusion"].append(time.perf_counter() - t0)

            actions = policy._pad_or_trim_time(batch["action"], policy.action_horizon)
            bsz = actions.shape[0]
            t = torch.rand(bsz, device=device, dtype=actions.dtype)
            t_broadcast = t.view(bsz, 1, 1)
            x1 = actions
            x0 = torch.randn_like(x1)
            xt = (1.0 - t_broadcast) * x0 + t_broadcast * x1
            _sync(device)
            t0 = time.perf_counter()
            _ = policy.model(xt, t, local_cond=None, global_cond=global_cond)
            _sync(device)
            results["unet"].append(time.perf_counter() - t0)

    policy.train()
    return {key: float(np.mean(vals) * 1000.0) for key, vals in results.items()}


def benchmark_train_step(
    cfg: dict,
    *,
    warmup: int = 5,
    steps: int = 30,
    data_root: str | None = None,
) -> None:
    set_seed(int(cfg.get("seed", 42)))
    device = torch.device(cfg_get_device(cfg))

    if data_root is not None:
        cfg = dict(cfg)
        cfg["data"] = dict(cfg["data"])
        cfg["data"]["root_dir"] = data_root

    dataset, loader = build_dataset_and_loader(cfg)
    policy = build_policy(cfg, device, dataset)
    optimizer = torch.optim.AdamW(
        [p for p in policy.parameters() if p.requires_grad],
        lr=float(cfg["train"]["optimizer"]["lr"]),
    )

    batch_size = int(cfg["train"]["batch_size"])
    steps_per_epoch = len(dataset) // batch_size
    num_workers = int(cfg["train"].get("num_workers", 0))

    print("=== Train Benchmark ===")
    print(f"device: {device}")
    print(f"dataset windows: {len(dataset)}")
    print(f"batch_size: {batch_size}, num_workers: {num_workers}")
    print(f"steps_per_epoch (drop_last): {steps_per_epoch}")
    print(f"trainable params: {sum(p.numel() for p in policy.parameters() if p.requires_grad):,}")

    data_wait = Timings()
    h2d = Timings()
    forward = Timings()
    backward = Timings()
    optim_step = Timings()
    total = Timings()

    it = iter(loader)
    fixed_batch = None

    for step in range(warmup + steps):
        _sync(device)
        t_step0 = time.perf_counter()

        t0 = time.perf_counter()
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        _sync(device)
        dt_data = time.perf_counter() - t0

        t0 = time.perf_counter()
        batch = move_to_device(batch, device)
        _sync(device)
        dt_h2d = time.perf_counter() - t0

        if fixed_batch is None and step == warmup - 1:
            fixed_batch = batch

        optimizer.zero_grad(set_to_none=True)

        t0 = time.perf_counter()
        out = policy(batch)
        loss = out["loss"]
        _sync(device)
        dt_fwd = time.perf_counter() - t0

        t0 = time.perf_counter()
        loss.backward()
        _sync(device)
        dt_bwd = time.perf_counter() - t0

        t0 = time.perf_counter()
        optimizer.step()
        _sync(device)
        dt_opt = time.perf_counter() - t0

        dt_total = time.perf_counter() - t_step0

        if step >= warmup:
            data_wait.add(dt_data)
            h2d.add(dt_h2d)
            forward.add(dt_fwd)
            backward.add(dt_bwd)
            optim_step.add(dt_opt)
            total.add(dt_total)

    sections = {
        "dataloader": data_wait.summary(),
        "h2d": h2d.summary(),
        "forward": forward.summary(),
        "backward": backward.summary(),
        "optimizer": optim_step.summary(),
        "total_step": total.summary(),
    }

    print("\n--- Per-step timing (after warmup) ---")
    total_mean = sections["total_step"]["mean_ms"]
    for name, stats in sections.items():
        pct = 100.0 * stats["mean_ms"] / total_mean if total_mean > 0 else 0.0
        print(
            f"{name:12s}  mean={stats['mean_ms']:7.1f} ms  "
            f"p50={stats.get('p50_ms', stats['mean_ms']):7.1f} ms  "
            f"p95={stats.get('p95_ms', stats['mean_ms']):7.1f} ms  "
            f"({pct:5.1f}%)"
        )

    samples_per_sec = batch_size / (total_mean / 1000.0) if total_mean > 0 else 0.0
    epoch_hours = steps_per_epoch * (total_mean / 1000.0) / 3600.0
    print(f"\nsamples/sec: {samples_per_sec:.2f}")
    print(f"estimated 1 epoch: {epoch_hours:.2f} h  ({steps_per_epoch} steps)")

    if fixed_batch is not None:
        print("\n--- Forward component breakdown (eval, no grad) ---")
        comp = profile_forward_components(policy, fixed_batch, device, repeats=30)
        comp_total = sum(comp.values())
        for name, ms in sorted(comp.items(), key=lambda x: -x[1]):
            pct = 100.0 * ms / comp_total if comp_total > 0 else 0.0
            print(f"{name:12s}  {ms:7.1f} ms  ({pct:5.1f}% of forward-ish compute)")


def cfg_get_device(cfg: dict) -> str:
    from utils.train_utils import cfg_get

    return cfg_get(cfg, "runtime.device", "cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--data-root", default=None, help="override data.root_dir for faster benchmark")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--steps", type=int, default=30)
    args = parser.parse_args()

    cfg = load_config(args.config)
    benchmark_train_step(
        cfg,
        warmup=args.warmup,
        steps=args.steps,
        data_root=args.data_root,
    )


if __name__ == "__main__":
    main()
