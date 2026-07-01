"""Compare tools/action.py vs tools/robot_action.py relative/absolute transforms."""

from __future__ import annotations

import os
import random

import numpy as np
import yaml

from _bootstrap import ensure_policy_root

_POLICY_ROOT = ensure_policy_root()
from datasets.zarr_dataset import ZarrDataset
from tools.action import (
    absolute_actions_to_relative_actions,
    relative_actions_to_absolute_actions,
)
from tools.robot_action import (
    transform_eef_absolute_to_relative,
    transform_eef_relative_to_absolute,
)


def _max_err(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a - b)))


def compare_on_sample(action_abs: np.ndarray, anchor: np.ndarray) -> dict[str, float]:
    """action_abs: (T, 20), anchor: (20,)"""
    base = anchor

    rel_tools = absolute_actions_to_relative_actions(action_abs.copy(), base_absolute_action=base)
    rel_robot = transform_eef_absolute_to_relative(action_abs.copy(), base)

    rel_tools2 = rel_tools.copy()
    relative_actions_to_absolute_actions(rel_tools2, base)

    abs_from_robot = transform_eef_relative_to_absolute(rel_robot.copy(), base)

    return {
        "rel_tools_vs_robot": _max_err(rel_tools, rel_robot),
        "abs_tools_inplace_vs_orig": _max_err(rel_tools2, action_abs),
        "abs_robot_vs_orig": _max_err(abs_from_robot, action_abs),
        "abs_tools_vs_robot_path": _max_err(rel_tools2, abs_from_robot),
    }


def main() -> None:
    cfg = yaml.safe_load(open(os.path.join(_POLICY_ROOT, "configs", "config.yaml")))["data"]
    cfg["root_dir"] = "/mnt/workspace/zyj/data/processed/peel/peel_0630_1656"
    cfg["norm"] = {"max_windows": 50}
    print("[compare] loading dataset...")
    ds = ZarrDataset.from_config(cfg)

    rng = random.Random(0)
    indices = rng.sample(range(len(ds)), 30)

    keys = [
        "rel_tools_vs_robot",
        "abs_tools_inplace_vs_orig",
        "abs_robot_vs_orig",
        "abs_tools_vs_robot_path",
    ]
    stats = {k: [] for k in keys}

    for idx in indices:
        s0, s1 = ds.state_range(idx)
        a0, a1 = ds.action_range(idx)
        state = ds.get_state(s0, s1)
        action = ds.get_action(a0, a1)
        anchor = state[-1]
        m = compare_on_sample(action, anchor)
        for k in keys:
            stats[k].append(m[k])
        if m["rel_tools_vs_robot"] > 1e-4:
            print(f"  idx={idx} rel_err={m['rel_tools_vs_robot']:.4f} abs_cross={m['abs_tools_vs_robot_path']:.4f}")

    print("\n=== Real data (eef 20D, 30 windows) ===")
    for k in keys:
        arr = np.array(stats[k])
        print(f"  {k:28s} max={arr.max():.6e} mean={arr.mean():.6e}")

    idx = indices[0]
    s0, s1 = ds.state_range(idx)
    a0, a1 = ds.action_range(idx)
    action = ds.get_action(a0, a1)
    anchor = ds.get_state(s0, s1)[-1]
    rel_t = absolute_actions_to_relative_actions(action.copy(), base_absolute_action=anchor)
    rel_r = transform_eef_absolute_to_relative(action.copy(), anchor)
    print("\n=== Layout probe (anchor frame, t=0) ===")
    print("  dim | abs_action | rel_tools | rel_robot | diff")
    for d in range(20):
        print(
            f"  {d:2d}  | {action[0,d]:+.4f} | {rel_t[0,d]:+.4f} | {rel_r[0,d]:+.4f} | {abs(rel_t[0,d]-rel_r[0,d]):.4f}"
        )


if __name__ == "__main__":
    main()
