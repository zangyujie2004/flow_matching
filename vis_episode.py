"""Visualize full-episode open-loop predictions: camera video + action curves."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Literal

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import cv2
import matplotlib

matplotlib.use("Agg")
import numpy as np
import torch
import yaml
from torch.utils.data.dataloader import default_collate
from tqdm import tqdm

_POLICY_ROOT = Path(__file__).resolve().parent
if str(_POLICY_ROOT) not in sys.path:
    sys.path.insert(0, str(_POLICY_ROOT))

from datasets import ZarrDataset
from models.fm import FlowMatchingPolicy, build_flow_policy
from tools.normalizer import DatasetNormalizer
from trainers.eval_open_loop import parse_dims, plot_action_curves
from trainers.policy_trainer import load_checkpoint
from utils.tensor import move_to_device
from utils.train_utils import cfg_get, set_seed, sync_fm_action_horizon_from_data

# preprocess/export/layout.py: ("base_0", "left_wrist_0", "right_wrist_0")
MAIN_CAMERA_VIEW_INDEX = 1
StitchMode = Literal["step0", "chunk"]


def load_run_config(run_dir: Path) -> dict:
    config_path = run_dir / "resolved_config.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing resolved config: {config_path}")
    with open(config_path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def build_dataset(cfg: dict) -> ZarrDataset:
    data_cfg = dict(cfg["data"])
    data_cfg["fit_normalizer"] = False
    return ZarrDataset.from_config(data_cfg)


def build_policy_from_cfg(cfg: dict, device: torch.device, dataset: ZarrDataset) -> FlowMatchingPolicy:
    data_cfg = cfg["data"]
    fm_cfg = sync_fm_action_horizon_from_data(cfg["models"]["fm"], data_cfg)
    fm_cfg["use_tactile"] = bool(data_cfg.get("use_tactile", True))
    cfg_for_build = dict(cfg)
    cfg_for_build["models"] = dict(cfg["models"])
    cfg_for_build["models"]["fm"] = fm_cfg
    return build_flow_policy(
        cfg_for_build,
        action_dim=dataset.action_dim,
        state_dim=dataset.action_dim,
        cond_steps=dataset.window_size,
    ).to(device)


def episode_action_bounds(dataset: ZarrDataset, ep_idx: int) -> tuple[int, int] | None:
    ep_start, ep_end = dataset.episode_bounds(ep_idx)
    cond_len = max(dataset.window_size, dataset.n_image_steps)
    first_t = int(ep_start + cond_len - 1)
    last_t = int(ep_end - dataset.action_horizon)
    if last_t < first_t:
        return None
    return first_t, last_t


def episode_window_indices(dataset: ZarrDataset, ep_idx: int) -> list[int]:
    return [
        idx
        for idx, (_anchor_t, _ep_end, window_ep_idx, base_mode) in enumerate(dataset.windows)
        if base_mode == "original"
        if int(window_ep_idx) == int(ep_idx)
    ]


def stitch_anchors(
    dataset: ZarrDataset,
    first_t: int,
    last_t: int,
    mode: StitchMode,
) -> list[int]:
    if mode == "step0":
        return list(range(first_t, last_t + 1, dataset.stride))
    return list(range(first_t, last_t + 1, dataset.action_horizon))


@torch.no_grad()
def predict_absolute_actions(
    policy: FlowMatchingPolicy,
    dataset: ZarrDataset,
    normalizer: DatasetNormalizer,
    window_indices: list[int],
    device: torch.device,
    *,
    batch_size: int,
    num_inference_steps: int,
    solver: str,
) -> dict[int, np.ndarray]:
    policy.eval()
    preds: dict[int, np.ndarray] = {}
    batch_size = max(1, int(batch_size))

    for start in tqdm(range(0, len(window_indices), batch_size), desc="Infer", leave=False):
        batch_indices = window_indices[start : start + batch_size]
        items = [dataset[int(window_idx)] for window_idx in batch_indices]
        batch = move_to_device(default_collate(items), device)
        result = policy.predict_action(
            batch["obs"],
            num_inference_steps=num_inference_steps,
            solver=solver,
        )
        pred_norm = result["action_pred_normalized"].detach().cpu().numpy()
        meta_idx = batch["meta"]["idx"]
        if torch.is_tensor(meta_idx):
            meta_indices = [int(value) for value in meta_idx.tolist()]
        else:
            meta_indices = [int(value) for value in meta_idx]

        for batch_pos, window_idx in enumerate(meta_indices):
            s0, s1 = dataset.state_range(window_idx)
            state_raw = dataset.get_state(s0, s1)
            pred_abs = normalizer.unnormalize_action_np(pred_norm[batch_pos], state_raw)
            preds[window_idx] = pred_abs.astype(np.float32, copy=False)

    return preds


def stitch_trajectory(
    dataset: ZarrDataset,
    ep_idx: int,
    preds_by_window: dict[int, np.ndarray],
    mode: StitchMode,
) -> tuple[np.ndarray, np.ndarray]:
    bounds = episode_action_bounds(dataset, ep_idx)
    if bounds is None:
        raise ValueError(f"Episode {ep_idx} has no valid action windows.")
    first_t, last_t = bounds
    anchors = stitch_anchors(dataset, first_t, last_t, mode)

    pred_steps: list[np.ndarray] = []
    for anchor_t in anchors:
        window_idx = dataset.window_lookup[(int(anchor_t), int(ep_idx))]
        pred_horizon = preds_by_window[window_idx]
        if mode == "step0":
            pred_steps.append(pred_horizon[0])
        else:
            pred_steps.append(pred_horizon)

    pred_traj = np.stack(pred_steps, axis=0) if mode == "step0" else np.concatenate(pred_steps, axis=0)
    gt_traj = dataset.get_action(first_t, first_t + pred_traj.shape[0]).astype(np.float32, copy=False)
    if gt_traj.shape[0] != pred_traj.shape[0]:
        raise RuntimeError(
            f"Trajectory length mismatch for ep={ep_idx}, mode={mode}: "
            f"pred={pred_traj.shape[0]}, gt={gt_traj.shape[0]}"
        )
    return pred_traj, gt_traj


def extract_main_camera_frames(dataset: ZarrDataset, ep_idx: int) -> np.ndarray:
    ep_start, ep_end = dataset.episode_bounds(ep_idx)
    camera = dataset.get_camera(ep_start, ep_end)
    view_offset = MAIN_CAMERA_VIEW_INDEX * 3
    return np.clip(camera[..., view_offset : view_offset + 3], 0, 255).astype(np.uint8)


def write_camera_mp4(frames: np.ndarray, out_path: Path, *, fps: int) -> None:
    if frames.size == 0:
        raise ValueError("Cannot write mp4 from empty camera frames.")
    height, width = int(frames.shape[1]), int(frames.shape[2])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer: {out_path}")
    try:
        for frame in frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()


def visualize_episode(
    dataset: ZarrDataset,
    normalizer: DatasetNormalizer,
    policy: FlowMatchingPolicy,
    device: torch.device,
    ep_idx: int,
    out_dir: Path,
    *,
    batch_size: int,
    num_inference_steps: int,
    solver: str,
    plot_dims: str,
    fps: int,
) -> dict[str, str]:
    bounds = episode_action_bounds(dataset, ep_idx)
    if bounds is None:
        raise ValueError(f"Episode {ep_idx} has no valid windows.")

    window_indices = episode_window_indices(dataset, ep_idx)
    if not window_indices:
        raise ValueError(f"Episode {ep_idx} has no dataset windows.")

    preds_by_window = predict_absolute_actions(
        policy,
        dataset,
        normalizer,
        window_indices,
        device,
        batch_size=batch_size,
        num_inference_steps=num_inference_steps,
        solver=solver,
    )

    outputs: dict[str, str] = {}
    camera_path = out_dir / f"ep{ep_idx:04d}_camera.mp4"
    write_camera_mp4(extract_main_camera_frames(dataset, ep_idx), camera_path, fps=fps)
    outputs["camera"] = str(camera_path)

    action_dim = int(next(iter(preds_by_window.values())).shape[1])
    dims_for_plot = parse_dims(plot_dims, action_dim)

    for mode in ("step0", "chunk"):
        pred_traj, gt_traj = stitch_trajectory(dataset, ep_idx, preds_by_window, mode)
        plot_path = out_dir / f"ep{ep_idx:04d}_action_{mode}.png"
        plot_action_curves(
            pred_traj,
            gt_traj,
            dims_for_plot,
            plot_path,
            title=f"Episode {ep_idx} | stitch={mode} | len={pred_traj.shape[0]}",
        )
        outputs[f"action_{mode}"] = str(plot_path)

    return outputs


def choose_episodes(dataset: ZarrDataset, num_episodes: int, seed: int) -> list[int]:
    valid = [
        ep_idx
        for ep_idx in range(dataset.num_episodes)
        if episode_action_bounds(dataset, ep_idx) is not None
    ]
    if not valid:
        raise RuntimeError("No episodes with valid action windows.")
    rng = np.random.default_rng(seed)
    count = min(int(num_episodes), len(valid))
    chosen = rng.choice(valid, size=count, replace=False)
    return sorted(int(ep_idx) for ep_idx in chosen.tolist())


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize full-episode camera and action trajectories.")
    parser.add_argument(
        "--run-dir",
        type=str,
        required=True,
        help="Training run directory, e.g. outputs/chahua_eef_vision",
    )
    parser.add_argument("--num-episodes", type=int, default=3, help="Number of random episodes to visualize.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None, help="Defaults to {run_dir}/checkpoints/latest.pt")
    parser.add_argument("--num-inference-steps", type=int, default=None)
    parser.add_argument("--solver", type=str, default=None, choices=["euler", "heun"])
    parser.add_argument("--plot-dims", type=str, default="auto")
    parser.add_argument("--fps", type=int, default=30, help="Camera mp4 frame rate.")
    args = parser.parse_args()

    set_seed(int(args.seed))
    run_dir = Path(args.run_dir).expanduser().resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    cfg = load_run_config(run_dir)
    device = torch.device(args.device or cfg_get(cfg, "runtime.device", "cuda" if torch.cuda.is_available() else "cpu"))

    dataset = build_dataset(cfg)
    policy = build_policy_from_cfg(cfg, device, dataset)

    checkpoint_path = Path(args.checkpoint).expanduser().resolve() if args.checkpoint else run_dir / "checkpoints" / "latest.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    load_checkpoint(checkpoint_path, policy, None, dataset)
    if dataset.normalizer is None:
        raise RuntimeError("Normalizer was not restored from checkpoint.")

    num_inference_steps = int(
        args.num_inference_steps
        if args.num_inference_steps is not None
        else cfg_get(cfg, "models.fm.num_inference_steps", 16)
    )
    solver = str(args.solver or cfg_get(cfg, "models.fm.solver", "euler"))

    out_dir = run_dir / "episode_visualization"
    out_dir.mkdir(parents=True, exist_ok=True)

    episode_ids = choose_episodes(dataset, args.num_episodes, args.seed)
    print(f"[vis_episode] run_dir={run_dir}")
    print(f"[vis_episode] checkpoint={checkpoint_path}")
    print(f"[vis_episode] episodes={episode_ids}, out_dir={out_dir}")

    for ep_idx in episode_ids:
        print(f"[vis_episode] processing episode {ep_idx} ...")
        outputs = visualize_episode(
            dataset,
            dataset.normalizer,
            policy,
            device,
            ep_idx,
            out_dir,
            batch_size=int(args.batch_size),
            num_inference_steps=num_inference_steps,
            solver=solver,
            plot_dims=str(args.plot_dims),
            fps=int(args.fps),
        )
        for key, path in outputs.items():
            print(f"  {key}: {path}")


if __name__ == "__main__":
    main()
