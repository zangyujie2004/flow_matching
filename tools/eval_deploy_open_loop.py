"""Offline open-loop validation for deployment inference (online DINO, no zarr at runtime)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

_FLOW_MATCHING_ROOT = Path(__file__).resolve().parents[1]
if str(_FLOW_MATCHING_ROOT) not in sys.path:
    sys.path.insert(0, str(_FLOW_MATCHING_ROOT))

from datasets import ZarrDataset
from infer import FMInferenceRuntime, load_run_config, obs_from_zarr_window
from utils.train_utils import cfg_get, set_seed


def build_eval_dataset(cfg: dict) -> ZarrDataset:
    data_cfg = dict(cfg["data"])
    data_cfg["fit_normalizer"] = False
    return ZarrDataset.from_config(data_cfg)


def evaluate_deploy_open_loop(
    runtime: FMInferenceRuntime,
    dataset: ZarrDataset,
    *,
    seed: int,
    max_batches: int,
    batch_size: int,
    vision_only: bool,
) -> dict[str, float]:
    rng = np.random.default_rng(int(seed))
    eval_batch_size = max(1, int(batch_size))
    total_windows = max(1, int(max_batches)) * eval_batch_size
    total_windows = min(total_windows, len(dataset))
    if total_windows <= 0:
        raise RuntimeError("Dataset has no windows for deploy open-loop evaluation.")

    chosen = rng.choice(len(dataset), size=total_windows, replace=False)
    sq_sum = 0.0
    abs_sum = 0.0
    elem_count = 0
    sample_count = 0

    pbar = tqdm(range(int(max_batches)), desc="DeployOpenLoop", leave=False)
    for batch_idx in pbar:
        start = batch_idx * eval_batch_size
        end = start + eval_batch_size
        if start >= len(chosen):
            break
        batch_indices = [int(chosen[i]) for i in range(start, min(end, len(chosen)))]

        obs_list = []
        state_raw_list = []
        gt_list = []
        for window_idx in batch_indices:
            obs, state_raw, gt_abs = obs_from_zarr_window(dataset, window_idx, vision_only=vision_only)
            obs_list.append(obs)
            state_raw_list.append(state_raw)
            gt_list.append(gt_abs)

        pred_abs = runtime.predict_rot6d_abs_batch(obs_list, state_raw_list)
        gt_abs = np.stack(gt_list, axis=0).astype(np.float32, copy=False)

        horizon = min(pred_abs.shape[1], gt_abs.shape[1])
        action_dim = min(pred_abs.shape[2], gt_abs.shape[2])
        if horizon <= 0 or action_dim <= 0:
            continue

        pred_abs = pred_abs[:, :horizon, :action_dim]
        gt_abs = gt_abs[:, :horizon, :action_dim]
        diff = pred_abs - gt_abs

        sq_sum += float(np.sum(diff * diff))
        abs_sum += float(np.sum(np.abs(diff)))
        elem_count += int(diff.size)
        sample_count += int(diff.shape[0])
        pbar.set_postfix(l1=f"{float(np.mean(np.abs(diff))):.4f}")

    if elem_count == 0:
        raise RuntimeError("No valid deploy open-loop samples were evaluated.")

    return {
        "action_mse": sq_sum / elem_count,
        "action_l1": abs_sum / elem_count,
        "num_samples": float(sample_count),
        "num_windows": float(total_windows),
        "batch_size": float(eval_batch_size),
        "max_batches": float(max_batches),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate deployment inference on zarr windows.")
    parser.add_argument("--run-dir", type=str, required=True, help="e.g. outputs/chahua_eef_vision")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-batches", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument(
        "--vision-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Simulate deploy: no tactile in obs (zero tactile if model has tactile branch).",
    )
    parser.add_argument("--no-warmup", action="store_true")
    args = parser.parse_args()

    set_seed(int(args.seed))
    run_dir = Path(args.run_dir).expanduser().resolve()
    cfg = load_run_config(run_dir)
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else run_dir / "deploy_validation"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[eval_deploy_open_loop] run_dir={run_dir}")
    dataset = build_eval_dataset(cfg)
    print(f"[eval_deploy_open_loop] windows={len(dataset)} root={dataset.root_dir}")

    runtime = FMInferenceRuntime(
        run_dir,
        checkpoint=args.checkpoint,
        device=args.device,
        warmup=not args.no_warmup,
    )
    dataset.normalizer = runtime.normalizer
    print(f"[eval_deploy_open_loop] checkpoint={runtime.checkpoint_path}")
    print(f"[eval_deploy_open_loop] device={runtime.device}")

    metrics = evaluate_deploy_open_loop(
        runtime,
        dataset,
        seed=int(args.seed),
        max_batches=int(args.max_batches),
        batch_size=int(args.batch_size),
        vision_only=bool(args.vision_only),
    )

    sample_idx = int(np.random.default_rng(int(args.seed)).choice(len(dataset)))
    obs, state_raw, _ = obs_from_zarr_window(dataset, sample_idx, vision_only=bool(args.vision_only))
    metrics.update(runtime.benchmark_single(obs, state_raw=state_raw, repeats=3))

    payload = {
        "run_dir": str(run_dir),
        "checkpoint": str(runtime.checkpoint_path),
        "mode": "deploy_online_image",
        "vision_only": bool(args.vision_only),
        "use_tactile_model": bool(runtime.use_tactile),
        "metrics": metrics,
        "train_open_loop_defaults": {
            "open_loop_test_max_batches": cfg_get(cfg, "train.open_loop_test_max_batches", None),
            "batch_size": cfg_get(cfg, "train.batch_size", None),
        },
    }
    metrics_path = out_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")

    print(f"[eval_deploy_open_loop] action_l1={metrics['action_l1']:.6f}")
    print(f"[eval_deploy_open_loop] action_mse={metrics['action_mse']:.6f}")
    print(f"[eval_deploy_open_loop] infer_ms={metrics.get('infer_ms', float('nan')):.2f}")
    print(f"[eval_deploy_open_loop] wrote {metrics_path}")


if __name__ == "__main__":
    main()
