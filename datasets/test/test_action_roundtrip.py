"""Round-trip check: absolute -> relative -> norm -> unnorm -> absolute."""

from __future__ import annotations

import argparse
import os
import random

import numpy as np
import yaml

from _bootstrap import ensure_policy_root

_POLICY_ROOT = ensure_policy_root()
from datasets.zarr_dataset import ZarrDataset
from tools.robot_action import (
    transform_robot_action,
    transform_robot_action_to_absolute,
)


def _max_abs_err(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a - b)))


def check_window(ds: ZarrDataset, idx: int) -> dict[str, float]:
    s0, s1 = ds.state_range(idx)
    a0, a1 = ds.action_range(idx)
    state_raw = ds.get_state(s0, s1)
    action_abs = ds.get_action(a0, a1)

    rel = transform_robot_action(
        action_abs,
        state_raw,
        action_type=ds.action_type,
        action_representation=ds.action_representation,
    )
    rel_back = transform_robot_action_to_absolute(
        rel,
        state_raw,
        action_type=ds.action_type,
        action_representation=ds.action_representation,
    )

    norm = ds.normalizer.normalize_action_np(action_abs, state_raw)
    unnorm_abs = ds.normalizer.unnormalize_action_np(norm, state_raw)

    return {
        "rel_roundtrip": _max_abs_err(action_abs, rel_back),
        "full_roundtrip": _max_abs_err(action_abs, unnorm_abs),
        "norm_range_min": float(norm.min()),
        "norm_range_max": float(norm.max()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root-dir",
        default="/mnt/workspace/zyj/data/processed/peel/peel_0630_1656",
    )
    parser.add_argument("--num-samples", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-windows", type=int, default=500)
    args = parser.parse_args()

    config_path = os.path.join(_POLICY_ROOT, "configs", "config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)["data"]
    data_cfg["root_dir"] = args.root_dir
    data_cfg["norm"] = {"max_windows": args.max_windows}

    print(f"[action_roundtrip] loading {args.root_dir} ...")
    ds = ZarrDataset.from_config(data_cfg)
    rng = random.Random(args.seed)
    indices = rng.sample(range(len(ds)), k=min(args.num_samples, len(ds)))

    rel_errs: list[float] = []
    full_errs: list[float] = []
    for idx in indices:
        m = check_window(ds, idx)
        rel_errs.append(m["rel_roundtrip"])
        full_errs.append(m["full_roundtrip"])
        if m["full_roundtrip"] > 1e-3:
            print(
                f"  idx={idx}: rel_rt={m['rel_roundtrip']:.2e} "
                f"full_rt={m['full_roundtrip']:.2e} norm=[{m['norm_range_min']:.3f},{m['norm_range_max']:.3f}]"
            )

    print(f"[action_roundtrip] action_type={ds.action_type} repr={ds.action_representation}")
    print(f"  relative <-> absolute: max_err={max(rel_errs):.6e}, mean_err={np.mean(rel_errs):.6e}")
    print(f"  full pipeline:         max_err={max(full_errs):.6e}, mean_err={np.mean(full_errs):.6e}")

    tol = 1e-4
    if max(full_errs) > tol:
        print(f"[action_roundtrip] FAILED (tol={tol})")
        return 1
    print(f"[action_roundtrip] PASSED (tol={tol})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
