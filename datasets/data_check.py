"""Randomly sample windows and visualize camera / tactile / state / action."""

from __future__ import annotations

import argparse
import os
import random
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

_POLICY_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_DATASETS_DIR = os.path.dirname(os.path.abspath(__file__))
if _POLICY_ROOT not in sys.path:
    sys.path.insert(0, _POLICY_ROOT)

from datasets.zarr_dataset import ZarrDataset
from tools.tactile_feat import TACTILE_BUNDLE_ORDER

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

_EEF_STATE_LABELS = (
    "L_xyz",
    "L_rot6d",
    "L_grip",
    "R_xyz",
    "R_rot6d",
    "R_grip",
)
_EEF_SEGMENTS = ((0, 3), (3, 9), (9, 10), (10, 13), (13, 19), (19, 20))


def _split_camera_views(camera: np.ndarray) -> list[np.ndarray]:
    """(H, W, 9) uint8 -> list of 3 RGB views."""
    views = []
    for v in range(camera.shape[-1] // 3):
        views.append(camera[..., v * 3 : (v + 1) * 3])
    return views


def _tactile_sensor_maps(deform: np.ndarray) -> list[tuple[str, np.ndarray]]:
    """(H, W, 12) -> [(name, |d| magnitude map), ...]."""
    maps = []
    for i, name in enumerate(TACTILE_BUNDLE_ORDER):
        dxyz = deform[..., i * 3 : (i + 1) * 3]
        mag = np.linalg.norm(dxyz, axis=-1)
        maps.append((name, mag))
    return maps


def _plot_eef_traj(ax: plt.Axes, data: np.ndarray, title: str) -> None:
    """Plot eef (T, 20) grouped by arm segments."""
    t = np.arange(data.shape[0])
    colors = plt.cm.tab10(np.linspace(0, 1, len(_EEF_SEGMENTS)))
    for (start, end), label, color in zip(_EEF_SEGMENTS, _EEF_STATE_LABELS, colors):
        seg = data[:, start:end]
        if seg.shape[1] == 1:
            ax.plot(t, seg[:, 0], label=label, color=color, linewidth=1.5)
        else:
            ax.plot(t, seg.mean(axis=1), label=f"{label}(mean)", color=color, linewidth=1.5)
    ax.set_title(title)
    ax.set_xlabel("step")
    ax.legend(fontsize=7, ncol=2, loc="upper right")
    ax.grid(True, alpha=0.3)


def visualize_sample(ds: ZarrDataset, idx: int, out_dir: str) -> str:
    idx = int(idx)
    anchor_t, ep_end, ep_idx = ds.windows[idx]
    s0, s1 = ds.state_range(idx)
    i0, i1 = ds.image_range(idx)
    a0, a1 = ds.action_range(idx)

    state = ds.get_state(s0, s1)
    action = ds.get_action(a0, a1)
    camera = ds.get_camera(i0, i1)
    tactile = ds.get_tactile(s0, s1) if ds.use_tactile else None

    anchor_in_obs = ds.window_size - 1
    anchor_in_action = 0

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(
        f"sample idx={idx}  ep={ep_idx}  anchor_t={anchor_t}  "
        f"action_type={ds.action_type}  repr={ds.action_representation}",
        fontsize=11,
    )
    gs = fig.add_gridspec(3, 4, height_ratios=[1.2, 1.0, 1.0], hspace=0.35, wspace=0.3)

    # camera: anchor frame, 3 views
    cam_anchor = camera[-1]
    views = _split_camera_views(cam_anchor)
    for v, img in enumerate(views):
        ax = fig.add_subplot(gs[0, v])
        ax.imshow(np.clip(img, 0, 255).astype(np.uint8))
        ax.set_title(f"camera view {v} (anchor)")
        ax.axis("off")

    # meta text
    ax_meta = fig.add_subplot(gs[0, 3])
    ax_meta.axis("off")
    lines = [
        f"state window: [{s0}, {s1})",
        f"image window: [{i0}, {i1})",
        f"action window: [{a0}, {a1})",
        f"state shape: {state.shape}",
        f"action shape: {action.shape}",
    ]
    if tactile is not None:
        lines.append(f"tactile shape: {tactile.shape}")
    ax_meta.text(0.0, 0.95, "\n".join(lines), va="top", fontsize=9, family="monospace")

    # tactile: 4 sensors at anchor obs frame
    if tactile is not None:
        deform_anchor = tactile[anchor_in_obs]
        sensor_maps = _tactile_sensor_maps(deform_anchor)
        for i, (name, mag) in enumerate(sensor_maps):
            ax = fig.add_subplot(gs[1, i])
            im = ax.imshow(mag, origin="lower", cmap="magma", aspect="auto")
            ax.set_title(name)
            ax.axis("off")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    else:
        ax = fig.add_subplot(gs[1, :])
        ax.text(0.5, 0.5, "use_tactile=False", ha="center", va="center")
        ax.axis("off")

    # state / action trajectories
    ax_state = fig.add_subplot(gs[2, :2])
    _plot_eef_traj(ax_state, state, f"state (eef, {state.shape[0]} steps)")

    ax_action = fig.add_subplot(gs[2, 2:])
    _plot_eef_traj(ax_action, action, f"action (eef, {action.shape[0]} steps)")

    out_path = os.path.join(out_dir, f"sample_{idx:06d}_ep{ep_idx}_t{anchor_t}.png")
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize random ZarrDataset samples")
    parser.add_argument("--config", default=os.path.join(_POLICY_ROOT, "configs", "config.yaml"))
    parser.add_argument("--num-samples", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", default=os.path.join(_DATASETS_DIR, "check_output"))
    parser.add_argument("--max-windows", type=int, default=None, help="override norm.max_windows for fast load")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)["data"]

    if args.max_windows is not None:
        data_cfg.setdefault("norm", {})["max_windows"] = args.max_windows

    print("[data_check] loading dataset...")
    ds = ZarrDataset.from_config(data_cfg)
    print(f"[data_check] {len(ds)} windows, {ds.num_episodes} episodes")

    os.makedirs(args.out_dir, exist_ok=True)
    rng = random.Random(args.seed)
    indices = rng.sample(range(len(ds)), k=min(args.num_samples, len(ds)))

    for i, idx in enumerate(indices):
        path = visualize_sample(ds, idx, args.out_dir)
        print(f"[data_check] [{i + 1}/{len(indices)}] saved {path}")

    # quick tensor sample check via __getitem__
    sample = ds[indices[0]]
    print("[data_check] __getitem__ shapes:")
    print("  obs.state:", tuple(sample["obs"]["state"].shape))
    print("  obs.image:", tuple(sample["obs"]["image"].shape))
    if "tactile" in sample["obs"]:
        print("  obs.tactile:", tuple(sample["obs"]["tactile"].shape))
    print("  action:", tuple(sample["action"].shape))


if __name__ == "__main__":
    main()
