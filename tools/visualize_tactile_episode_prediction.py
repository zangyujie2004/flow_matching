"""Stitch Stage-2 tactile chunks into a complete episode visualization.

The policy predicts 128 future tactile frames from each anchor.  To create one
continuous episode timeline, this tool advances by ``chunk_stride`` frames and
keeps only the first ``chunk_stride`` predictions from each chunk.  This is a
receding-horizon visualization, not an average of overlapping predictions.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

_FLOW_MATCHING_ROOT = Path(__file__).resolve().parents[1]
if str(_FLOW_MATCHING_ROOT) not in sys.path:
    sys.path.insert(0, str(_FLOW_MATCHING_ROOT))

from infer.config import load_run_config, load_runtime_checkpoint
from tools.eval_tactile_reconstruction import (
    ErrorAccumulator,
    EvalWindow,
    TactileEvalData,
    _autocast_context,
    build_tactile_condition_obs,
    decode_prediction,
)
from tools.tactile_feat import TACTILE_BUNDLE_ORDER, extract_tactile_deformation
from utils.train_utils import cfg_get, set_seed


@dataclass(frozen=True)
class PredictionSegment:
    anchor: int
    target_start: int
    length: int


def build_episode_segments(
    *,
    episode_start: int,
    episode_end: int,
    observation_steps: int,
    action_horizon: int,
    tactile_target_offset: int,
    chunk_stride: int,
) -> list[PredictionSegment]:
    if chunk_stride <= 0 or chunk_stride > action_horizon:
        raise ValueError(
            f"chunk_stride must be in [1,{action_horizon}], got {chunk_stride}"
        )
    first_anchor = int(episode_start) + int(observation_steps) - 1
    first_target = first_anchor + int(tactile_target_offset)
    segments: list[PredictionSegment] = []
    target_start = first_target
    while target_start < int(episode_end):
        length = min(
            int(chunk_stride),
            int(action_horizon),
            int(episode_end) - target_start,
        )
        segments.append(
            PredictionSegment(
                anchor=target_start - int(tactile_target_offset),
                target_start=target_start,
                length=length,
            )
        )
        target_start += length
    return segments


def predict_episode(
    *,
    source: TactileEvalData,
    policy: torch.nn.Module,
    normalizer: Any,
    segments: Sequence[PredictionSegment],
    episode: int,
    episode_start: int,
    episode_length: int,
    base_mode: str,
    device: torch.device,
    batch_size: int,
    decode_frame_batch_size: int,
    num_inference_steps: int,
    solver: str,
    amp: str,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    set_seed(int(seed))
    prediction = np.full(
        (episode_length, 35, 20, 12), np.nan, dtype=np.float32
    )
    persistence = np.full_like(prediction, np.nan)

    pbar = tqdm(
        range(0, len(segments), int(batch_size)),
        desc=f"EpisodePredict[{base_mode}]",
    )
    for start in pbar:
        batch_segments = list(segments[start : start + int(batch_size)])
        windows = [
            EvalWindow(episode=int(episode), anchor=int(segment.anchor))
            for segment in batch_segments
        ]
        state_raw, visual, tactile_current = source.gather_batch(
            windows,
            base_mode=base_mode,
        )
        state_normalized = normalizer.normalize_state_np(state_raw)
        tactile_normalized = normalizer.normalize_tactile_np(tactile_current)
        state_tensor = torch.from_numpy(state_normalized).to(device=device)
        visual_tensor = torch.from_numpy(visual).to(device=device)
        tactile_tensor = torch.from_numpy(tactile_normalized).to(device=device)

        with torch.inference_mode():
            with _autocast_context(device, amp):
                tactile_condition = build_tactile_condition_obs(
                    policy, tactile_tensor
                )
                policy_obs = {
                    "state": state_tensor,
                    "image_backbone_feat": visual_tensor,
                    **tactile_condition,
                }
                result = policy.predict_action(
                    policy_obs,
                    num_inference_steps=int(num_inference_steps),
                    solver=str(solver),
                    decode_tactile=False,
                )
            keep = max(segment.length for segment in batch_segments)
            predicted_normalized = decode_prediction(
                policy,
                result["tactile_latent_pred_normalized"][:, :keep].float(),
                frame_batch_size=int(decode_frame_batch_size),
                amp=amp,
            )
            predicted_raw = normalizer.tactile.unnormalize(
                predicted_normalized
            ).float()

        predicted_np = predicted_raw.cpu().numpy()
        for item, segment in enumerate(batch_segments):
            local_start = int(segment.target_start) - int(episode_start)
            local_stop = local_start + int(segment.length)
            prediction[local_start:local_stop] = predicted_np[
                item, : segment.length
            ]
            persistence[local_start:local_stop] = tactile_current[item, -1]
    return prediction, persistence


def reconstruct_oracle(
    *,
    ground_truth: np.ndarray,
    available: np.ndarray,
    policy: torch.nn.Module,
    normalizer: Any,
    device: torch.device,
    frame_batch_size: int,
    amp: str,
) -> np.ndarray:
    oracle = np.full_like(ground_truth, np.nan)
    indices = np.flatnonzero(available)
    if indices.size == 0:
        return oracle
    for start in tqdm(
        range(0, len(indices), int(frame_batch_size)),
        desc="AEOracle",
    ):
        batch_indices = indices[start : start + int(frame_batch_size)]
        normalized = normalizer.normalize_tactile_np(ground_truth[batch_indices])
        tensor = torch.from_numpy(normalized).to(device=device)
        with torch.inference_mode(), _autocast_context(device, amp):
            latent = policy.tactile_autoencoder.encode_flattened(tensor)
            reconstruction = policy.tactile_autoencoder.decode_flattened(latent)
        raw = normalizer.tactile.unnormalize(reconstruction.float())
        oracle[batch_indices] = raw.cpu().numpy()
    return oracle


def summarize_errors(
    prediction: np.ndarray,
    target: np.ndarray,
    available: np.ndarray,
) -> tuple[ErrorAccumulator, np.ndarray, np.ndarray]:
    accumulator = ErrorAccumulator()
    accumulator.update(prediction[available][None], target[available][None])
    frame_mae = np.full(len(prediction), np.nan, dtype=np.float64)
    frame_mse = np.full(len(prediction), np.nan, dtype=np.float64)
    diff = (
        prediction[available].astype(np.float64)
        - target[available].astype(np.float64)
    )
    frame_mae[available] = np.mean(np.abs(diff), axis=(1, 2, 3))
    frame_mse[available] = np.mean(np.square(diff), axis=(1, 2, 3))
    return accumulator, frame_mae, frame_mse


def save_frame_metrics(
    path: Path,
    *,
    available: np.ndarray,
    stage2_mae: np.ndarray,
    stage2_mse: np.ndarray,
    oracle_mae: np.ndarray,
    oracle_mse: np.ndarray,
    persistence_mae: np.ndarray,
    persistence_mse: np.ndarray,
    fps: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "episode_frame",
                "time_s",
                "available",
                "stage2_mae",
                "stage2_mse",
                "ae_oracle_mae",
                "ae_oracle_mse",
                "persistence_mae",
                "persistence_mse",
            ]
        )
        for frame in range(len(available)):
            writer.writerow(
                [
                    frame,
                    frame / float(fps),
                    int(available[frame]),
                    stage2_mae[frame],
                    stage2_mse[frame],
                    oracle_mae[frame],
                    oracle_mse[frame],
                    persistence_mae[frame],
                    persistence_mse[frame],
                ]
            )


def _mean_tactile_curves(tactile: np.ndarray) -> np.ndarray:
    curves = np.full((len(tactile), 4, 4), np.nan, dtype=np.float32)
    for sensor in range(4):
        channels = tactile[..., sensor * 3 : (sensor + 1) * 3]
        valid = np.all(np.isfinite(channels), axis=(1, 2, 3))
        curves[valid, sensor, :3] = np.mean(
            np.abs(channels[valid]), axis=(1, 2)
        )
        curves[valid, sensor, 3] = np.mean(
            np.linalg.norm(channels[valid, ..., :2], axis=-1), axis=(1, 2)
        )
    return curves


def save_episode_curves(
    path: Path,
    *,
    prediction: np.ndarray,
    ground_truth: np.ndarray,
    fps: float,
    title: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    gt = _mean_tactile_curves(ground_truth)
    pred = _mean_tactile_curves(prediction)
    time = np.arange(len(ground_truth), dtype=np.float32) / float(fps)
    colors = ("tab:red", "tab:green", "tab:blue", "tab:purple")
    labels = ("dx", "dy", "dz", "|dxy|")
    fig, axes = plt.subplots(
        4, 1, figsize=(15, 12), sharex=True, constrained_layout=True
    )
    for sensor, sensor_name in enumerate(TACTILE_BUNDLE_ORDER):
        axis = axes[sensor]
        for curve, (color, label) in enumerate(zip(colors, labels)):
            axis.plot(time, gt[:, sensor, curve], color=color, label=f"GT {label}")
            axis.plot(
                time,
                pred[:, sensor, curve],
                color=color,
                linestyle="--",
                label=f"Pred {label}",
            )
        axis.set_ylabel(sensor_name)
        axis.grid(alpha=0.25)
        if sensor == 0:
            axis.legend(ncol=4, fontsize=8)
    axes[-1].set_xlabel("episode time (s)")
    fig.suptitle(title)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def save_error_curves(
    path: Path,
    *,
    stage2_mae: np.ndarray,
    stage2_mse: np.ndarray,
    oracle_mae: np.ndarray,
    oracle_mse: np.ndarray,
    persistence_mae: np.ndarray,
    persistence_mse: np.ndarray,
    fps: float,
    title: str,
) -> None:
    time = np.arange(len(stage2_mae), dtype=np.float32) / float(fps)
    fig, axes = plt.subplots(2, 1, figsize=(15, 7), sharex=True, constrained_layout=True)
    for axis, stage2, oracle, persistence, name in (
        (axes[0], stage2_mae, oracle_mae, persistence_mae, "MAE"),
        (axes[1], stage2_mse, oracle_mse, persistence_mse, "MSE"),
    ):
        axis.plot(time, stage2, label="Stage2", color="tab:red")
        axis.plot(time, oracle, label="AE oracle", color="tab:blue")
        axis.plot(time, persistence, label="Persistence", color="tab:gray")
        axis.set_ylabel(name)
        axis.grid(alpha=0.25)
        axis.legend()
    axes[-1].set_xlabel("episode time (s)")
    fig.suptitle(title)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _finite_quantile(values: np.ndarray, quantile: float, minimum: float) -> float:
    flat = np.asarray(values, dtype=np.float32).reshape(-1)
    flat = flat[np.isfinite(flat)]
    if flat.size == 0:
        return float(minimum)
    return max(float(np.quantile(flat, quantile)), float(minimum))


def _colorize(
    image: np.ndarray,
    *,
    cmap_name: str,
    vmin: float,
    vmax: float,
    width: int,
    height: int,
) -> np.ndarray:
    array = np.asarray(image, dtype=np.float32)
    if not np.any(np.isfinite(array)):
        return np.full((height, width, 3), 72, dtype=np.uint8)
    denominator = max(float(vmax) - float(vmin), 1e-12)
    normalized = np.clip((np.nan_to_num(array, nan=vmin) - vmin) / denominator, 0.0, 1.0)
    rgb = (
        matplotlib.colormaps[cmap_name](normalized)[..., :3] * 255.0
    ).astype(np.uint8)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return cv2.resize(bgr, (width, height), interpolation=cv2.INTER_NEAREST)


def _video_scales(
    prediction: np.ndarray,
    ground_truth: np.ndarray,
    available: np.ndarray,
) -> list[dict[str, float]]:
    scales = []
    for sensor in range(4):
        channel = slice(sensor * 3, (sensor + 1) * 3)
        gt = ground_truth[available, ..., channel]
        pred = prediction[available, ..., channel]
        gt_tangent = np.linalg.norm(gt[..., :2], axis=-1)
        pred_tangent = np.linalg.norm(pred[..., :2], axis=-1)
        tangent_max = _finite_quantile(
            np.concatenate([gt_tangent.reshape(-1), pred_tangent.reshape(-1)]),
            0.995,
            1e-8,
        )
        z_limit = _finite_quantile(
            np.concatenate(
                [np.abs(gt[..., 2]).reshape(-1), np.abs(pred[..., 2]).reshape(-1)]
            ),
            0.995,
            1e-8,
        )
        tangent_error_max = _finite_quantile(
            np.abs(pred_tangent - gt_tangent), 0.995, 1e-8
        )
        z_error_max = _finite_quantile(
            np.abs(pred[..., 2] - gt[..., 2]), 0.995, 1e-8
        )
        scales.append(
            {
                "tangent_max": tangent_max,
                "z_limit": z_limit,
                "tangent_error_max": tangent_error_max,
                "z_error_max": z_error_max,
            }
        )
    return scales


def save_episode_video(
    path: Path,
    *,
    prediction: np.ndarray,
    ground_truth: np.ndarray,
    available: np.ndarray,
    frame_mae: np.ndarray,
    fps: float,
    max_frames: int,
    title: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    scales = _video_scales(prediction, ground_truth, available)
    cell_width, cell_height = 120, 210
    label_width, top_height, bottom_height, gap = 145, 48, 34, 7
    width = label_width + 6 * cell_width + 5 * gap
    height = top_height + 4 * cell_height + 3 * gap + bottom_height
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"failed to open video writer: {path}")

    headers = ("GT |dxy|", "Pred |dxy|", "Error", "GT dz", "Pred dz", "Error")
    frame_count = len(ground_truth) if max_frames < 0 else min(len(ground_truth), max_frames)
    try:
        for frame in tqdm(range(frame_count), desc="WriteEpisodeVideo"):
            canvas = np.full((height, width, 3), 245, dtype=np.uint8)
            cv2.putText(
                canvas,
                title,
                (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (20, 20, 20),
                1,
                cv2.LINE_AA,
            )
            for col, header in enumerate(headers):
                x = label_width + col * (cell_width + gap)
                cv2.putText(
                    canvas,
                    header,
                    (x + 4, 43),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.38,
                    (20, 20, 20),
                    1,
                    cv2.LINE_AA,
                )
            for sensor, sensor_name in enumerate(TACTILE_BUNDLE_ORDER):
                y = top_height + sensor * (cell_height + gap)
                cv2.putText(
                    canvas,
                    sensor_name,
                    (5, y + cell_height // 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.42,
                    (20, 20, 20),
                    1,
                    cv2.LINE_AA,
                )
                channel = slice(sensor * 3, (sensor + 1) * 3)
                gt = ground_truth[frame, ..., channel]
                pred = prediction[frame, ..., channel]
                gt_tangent = np.linalg.norm(gt[..., :2], axis=-1)
                pred_tangent = np.linalg.norm(pred[..., :2], axis=-1)
                scale = scales[sensor]
                images = (
                    _colorize(gt_tangent, cmap_name="viridis", vmin=0.0, vmax=scale["tangent_max"], width=cell_width, height=cell_height),
                    _colorize(pred_tangent, cmap_name="viridis", vmin=0.0, vmax=scale["tangent_max"], width=cell_width, height=cell_height),
                    _colorize(np.abs(pred_tangent - gt_tangent), cmap_name="magma", vmin=0.0, vmax=scale["tangent_error_max"], width=cell_width, height=cell_height),
                    _colorize(gt[..., 2], cmap_name="coolwarm", vmin=-scale["z_limit"], vmax=scale["z_limit"], width=cell_width, height=cell_height),
                    _colorize(pred[..., 2], cmap_name="coolwarm", vmin=-scale["z_limit"], vmax=scale["z_limit"], width=cell_width, height=cell_height),
                    _colorize(np.abs(pred[..., 2] - gt[..., 2]), cmap_name="magma", vmin=0.0, vmax=scale["z_error_max"], width=cell_width, height=cell_height),
                )
                for col, image in enumerate(images):
                    x = label_width + col * (cell_width + gap)
                    canvas[y : y + cell_height, x : x + cell_width] = image
            status = (
                f"frame={frame:04d} time={frame / float(fps):.2f}s "
                + (
                    f"MAE={frame_mae[frame]:.6g}"
                    if available[frame]
                    else "prediction unavailable"
                )
            )
            cv2.putText(
                canvas,
                status,
                (10, height - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (20, 20, 20),
                1,
                cv2.LINE_AA,
            )
            writer.write(canvas)
    finally:
        writer.release()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Visualize a stitched complete-episode Stage-2 tactile prediction."
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--episode", type=int, required=True, help="Global processed episode index.")
    parser.add_argument("--latent-cache-root", default=None)
    parser.add_argument("--base-mode", choices=("original", "remove"), default="original")
    parser.add_argument("--chunk-stride", type=int, default=30)
    parser.add_argument("--max-segments", type=int, default=-1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--decode-frame-batch-size", type=int, default=128)
    parser.add_argument("--num-inference-steps", type=int, default=None)
    parser.add_argument("--solver", choices=("euler", "heun"), default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--amp", choices=("none", "bf16", "fp16"), default="bf16")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--max-video-frames", type=int, default=-1)
    parser.add_argument("--save-video", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-arrays", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.batch_size <= 0 or args.decode_frame_batch_size <= 0:
        raise ValueError("batch sizes must be positive")
    if args.fps <= 0:
        raise ValueError("--fps must be positive")
    if args.max_segments == 0:
        raise ValueError("--max-segments must be positive or -1")

    run_dir = Path(args.run_dir).expanduser().resolve()
    data_root = Path(args.data_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    cfg = load_run_config(run_dir)
    source = TactileEvalData(
        data_root=data_root,
        policy_cfg=cfg,
        latent_cache_root=(
            None if args.latent_cache_root is None else Path(args.latent_cache_root)
        ),
    )
    episode = int(args.episode)
    if not 0 <= episode < len(source.episode_ends):
        raise ValueError(
            f"episode={episode} outside [0,{len(source.episode_ends)})"
        )
    if args.base_mode == "remove":
        source.validate_remove_hand([episode])
    episode_start = int(source.episode_starts[episode])
    episode_end = int(source.episode_ends[episode])
    episode_length = episode_end - episode_start
    observation_steps = max(
        source.window_size,
        source.n_image_steps,
        source.tactile_obs_steps,
    )
    all_segments = build_episode_segments(
        episode_start=episode_start,
        episode_end=episode_end,
        observation_steps=observation_steps,
        action_horizon=source.action_horizon,
        tactile_target_offset=source.tactile_target_offset,
        chunk_stride=int(args.chunk_stride),
    )
    segments = (
        all_segments
        if int(args.max_segments) < 0
        else all_segments[: int(args.max_segments)]
    )
    if not segments:
        raise RuntimeError("episode has no valid prediction segments")

    selection = {
        "run_dir": str(run_dir),
        "data_root": str(data_root),
        "episode": episode,
        "base_mode": args.base_mode,
        "episode_global_range": [episode_start, episode_end],
        "episode_length": episode_length,
        "observation_steps": observation_steps,
        "action_horizon": source.action_horizon,
        "tactile_target_offset": source.tactile_target_offset,
        "chunk_stride": int(args.chunk_stride),
        "num_segments_full_episode": len(all_segments),
        "num_segments": len(segments),
        "first_predicted_episode_frame": observation_steps,
    }
    print(json.dumps(selection, indent=2, ensure_ascii=False))
    if args.dry_run:
        print("[dry-run] episode, visual cache, and rolling segments are valid")
        return

    checkpoint = (
        Path(args.checkpoint).expanduser().resolve()
        if args.checkpoint
        else run_dir / "checkpoints" / "latest.pt"
    )
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
    device = torch.device(
        args.device
        or cfg_get(
            cfg,
            "runtime.device",
            "cuda" if torch.cuda.is_available() else "cpu",
        )
    )
    policy, normalizer, checkpoint_state = load_runtime_checkpoint(
        checkpoint, cfg, match_training=True
    )
    if not bool(getattr(policy, "predict_tactile", False)):
        raise RuntimeError("checkpoint has models.fm.predict_tactile=false")
    if policy.tactile_autoencoder is None or normalizer.tactile is None:
        raise RuntimeError("checkpoint lacks tactile autoencoder/normalizer")
    policy = policy.to(device).eval()
    num_inference_steps = int(
        args.num_inference_steps
        if args.num_inference_steps is not None
        else cfg_get(cfg, "models.fm.num_inference_steps", policy.num_inference_steps)
    )
    solver = str(args.solver or cfg_get(cfg, "models.fm.solver", policy.solver))

    prediction, persistence = predict_episode(
        source=source,
        policy=policy,
        normalizer=normalizer,
        segments=segments,
        episode=episode,
        episode_start=episode_start,
        episode_length=episode_length,
        base_mode=str(args.base_mode),
        device=device,
        batch_size=int(args.batch_size),
        decode_frame_batch_size=int(args.decode_frame_batch_size),
        num_inference_steps=num_inference_steps,
        solver=solver,
        amp=str(args.amp),
        seed=int(args.seed),
    )
    raw_gt = np.asarray(
        source.data["tactile"][episode_start:episode_end], dtype=np.float32
    )
    ground_truth = extract_tactile_deformation(raw_gt)
    available = np.all(np.isfinite(prediction), axis=(1, 2, 3))
    oracle = reconstruct_oracle(
        ground_truth=ground_truth,
        available=available,
        policy=policy,
        normalizer=normalizer,
        device=device,
        frame_batch_size=int(args.decode_frame_batch_size),
        amp=str(args.amp),
    )
    zero = np.zeros_like(ground_truth)
    stage2_acc, stage2_mae, stage2_mse = summarize_errors(
        prediction, ground_truth, available
    )
    oracle_acc, oracle_mae, oracle_mse = summarize_errors(
        oracle, ground_truth, available
    )
    persistence_acc, persistence_mae, persistence_mse = summarize_errors(
        persistence, ground_truth, available
    )
    zero_acc, _, _ = summarize_errors(zero, ground_truth, available)

    output_dir.mkdir(parents=True, exist_ok=True)
    title = f"episode={episode} base_mode={args.base_mode} stride={args.chunk_stride}"
    curves_path = output_dir / "episode_temporal_curves.png"
    errors_path = output_dir / "episode_error_curves.png"
    frame_metrics_path = output_dir / "frame_metrics.csv"
    save_episode_curves(
        curves_path,
        prediction=prediction,
        ground_truth=ground_truth,
        fps=float(args.fps),
        title=title,
    )
    save_error_curves(
        errors_path,
        stage2_mae=stage2_mae,
        stage2_mse=stage2_mse,
        oracle_mae=oracle_mae,
        oracle_mse=oracle_mse,
        persistence_mae=persistence_mae,
        persistence_mse=persistence_mse,
        fps=float(args.fps),
        title=title,
    )
    save_frame_metrics(
        frame_metrics_path,
        available=available,
        stage2_mae=stage2_mae,
        stage2_mse=stage2_mse,
        oracle_mae=oracle_mae,
        oracle_mse=oracle_mse,
        persistence_mae=persistence_mae,
        persistence_mse=persistence_mse,
        fps=float(args.fps),
    )

    video_path = output_dir / "episode_tactile_prediction.mp4"
    if args.save_video:
        save_episode_video(
            video_path,
            prediction=prediction,
            ground_truth=ground_truth,
            available=available,
            frame_mae=stage2_mae,
            fps=float(args.fps),
            max_frames=int(args.max_video_frames),
            title=title,
        )
    arrays_path = output_dir / "episode_prediction.npz"
    if args.save_arrays:
        np.savez_compressed(
            arrays_path,
            tactile_gt=ground_truth,
            tactile_stage2=prediction,
            tactile_ae_oracle=oracle,
            tactile_persistence=persistence,
            available=available,
            episode=np.int64(episode),
            episode_start=np.int64(episode_start),
            chunk_stride=np.int64(args.chunk_stride),
        )

    payload = {
        "format": "stage2_tactile_episode_visualization/v1",
        "selection": selection,
        "checkpoint": str(checkpoint),
        "checkpoint_epoch": int(checkpoint_state.get("epoch", -1)),
        "checkpoint_global_step": int(checkpoint_state.get("global_step", -1)),
        "device": str(device),
        "amp": str(args.amp),
        "num_inference_steps": num_inference_steps,
        "solver": solver,
        "coverage": {
            "predicted_frames": int(available.sum()),
            "episode_frames": episode_length,
            "fraction": float(available.mean()),
        },
        "metrics": {
            "stage2_physical": stage2_acc.summary(),
            "ae_oracle_physical": oracle_acc.summary(),
            "persistence_physical": persistence_acc.summary(),
            "zero_physical": zero_acc.summary(),
        },
        "artifacts": {
            "video": str(video_path) if args.save_video else None,
            "temporal_curves": str(curves_path),
            "error_curves": str(errors_path),
            "frame_metrics": str(frame_metrics_path),
            "arrays": str(arrays_path) if args.save_arrays else None,
        },
    }
    metrics_path = output_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(f"[complete] metrics={metrics_path}")
    print(
        f"stage2_mae={payload['metrics']['stage2_physical']['mae']:.8g} "
        f"oracle_mae={payload['metrics']['ae_oracle_physical']['mae']:.8g} "
        f"persistence_mae={payload['metrics']['persistence_physical']['mae']:.8g}"
    )


if __name__ == "__main__":
    main()
