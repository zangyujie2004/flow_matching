"""Visualize several Stage-2 tactile predictions as four-sensor quiver videos.

Adapted from:
  /mnt/workspace/zyh/omnivta/ours/infer_logs/tools/vis_tactile.py

The reference script expects two sensors packed as six channels.  Flow Matching
Stage 2 stores four sensors in twelve channels:
  [left_wrist_0(dx,dy,dz), left_wrist_1(dx,dy,dz),
   right_wrist_0(dx,dy,dz), right_wrist_1(dx,dy,dz)].
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
from matplotlib.lines import Line2D
import numpy as np
from tqdm import tqdm


SENSOR_NAMES = (
    "left_wrist_0",
    "left_wrist_1",
    "right_wrist_0",
    "right_wrist_1",
)
SPATIAL_SHAPE = (35, 20)
CHANNELS_PER_SENSOR = 3
TOTAL_CHANNELS = len(SENSOR_NAMES) * CHANNELS_PER_SENSOR


def _validate_tactile(name: str, value: np.ndarray) -> np.ndarray:
    tactile = np.asarray(value, dtype=np.float32)
    if tactile.ndim == 5 and tactile.shape[0] == 1:
        tactile = tactile[0]
    expected_tail = (*SPATIAL_SHAPE, TOTAL_CHANNELS)
    if tactile.ndim != 4 or tuple(tactile.shape[1:]) != expected_tail:
        raise ValueError(
            f"{name} must have shape (T,35,20,12), got {tactile.shape}"
        )
    return tactile


def load_stage2_prediction(
    path: str | Path,
    *,
    pred_key: str = "tactile_stage2",
    gt_key: str = "tactile_gt",
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """Load one Stage-2 window/full-episode NPZ without pickle objects."""

    npz_path = Path(path).expanduser().resolve()
    with np.load(npz_path, allow_pickle=False) as payload:
        missing = [key for key in (pred_key, gt_key) if key not in payload]
        if missing:
            raise KeyError(
                f"{npz_path} is missing {missing}; available keys={payload.files}"
            )
        prediction = _validate_tactile(pred_key, payload[pred_key])
        ground_truth = _validate_tactile(gt_key, payload[gt_key])
        metadata = {
            key: int(np.asarray(payload[key]).item())
            for key in ("episode", "anchor", "episode_start")
            if key in payload and np.asarray(payload[key]).size == 1
        }
    if prediction.shape != ground_truth.shape:
        raise ValueError(
            f"prediction/GT shape mismatch: {prediction.shape} != {ground_truth.shape}"
        )
    return prediction, ground_truth, metadata


def select_valid_frames(
    prediction: np.ndarray,
    ground_truth: np.ndarray,
    *,
    start_frame: int,
    temporal_stride: int,
    max_frames: int,
) -> np.ndarray:
    if start_frame < 0:
        raise ValueError("start_frame must be non-negative")
    if temporal_stride <= 0:
        raise ValueError("temporal_stride must be positive")
    if max_frames == 0 or max_frames < -1:
        raise ValueError("max_frames must be positive or -1")

    valid = np.all(np.isfinite(prediction), axis=(1, 2, 3))
    valid &= np.all(np.isfinite(ground_truth), axis=(1, 2, 3))
    indices = np.arange(start_frame, len(prediction), temporal_stride, dtype=np.int64)
    indices = indices[valid[indices]]
    if max_frames > 0:
        indices = indices[:max_frames]
    if indices.size == 0:
        raise ValueError("no finite prediction/GT frames remain after frame selection")
    return indices


def _resolve_vector_gain(
    prediction: np.ndarray,
    ground_truth: np.ndarray,
    frame_indices: np.ndarray,
    value: str,
) -> float:
    text = str(value).strip().lower()
    if text != "auto":
        gain = float(text)
        if not np.isfinite(gain) or gain <= 0:
            raise ValueError("--vector-gain must be positive or 'auto'")
        return gain

    selected = np.concatenate(
        [prediction[frame_indices], ground_truth[frame_indices]], axis=0
    )
    sensors = selected.reshape(
        len(selected), *SPATIAL_SHAPE, len(SENSOR_NAMES), CHANNELS_PER_SENSOR
    )
    magnitude = np.linalg.norm(sensors[..., :2], axis=-1)
    robust_max = float(np.quantile(magnitude, 0.99))
    # Target a 0.75 grid-coordinate arrow at the joint GT/pred 99th percentile.
    return 1.0 if robust_max <= 1e-12 else 0.75 / robust_max


def plot_tactile_grids_animation_v2(
    pred: np.ndarray,
    gt: np.ndarray,
    save_path: str | Path,
    *,
    plt_gt: bool = True,
    plt_pred: bool = True,
    fps: float = 8.0,
    frame_indices: Sequence[int] | None = None,
    taxel_stride: int = 2,
    vector_gain: str = "auto",
    dpi: int = 120,
    title: str = "Stage-2 tactile prediction",
    save_preview: bool = True,
) -> dict[str, float | int | str]:
    """Render four tactile sensors; blue is GT and red is Stage-2 prediction."""

    prediction = _validate_tactile("pred", pred)
    ground_truth = _validate_tactile("gt", gt)
    if prediction.shape != ground_truth.shape:
        raise ValueError(
            f"prediction/GT shape mismatch: {prediction.shape} != {ground_truth.shape}"
        )
    if not plt_gt and not plt_pred:
        raise ValueError("at least one of plt_gt/plt_pred must be true")
    if fps <= 0 or taxel_stride <= 0 or dpi <= 0:
        raise ValueError("fps, taxel_stride, and dpi must be positive")

    if frame_indices is None:
        indices = np.arange(len(prediction), dtype=np.int64)
    else:
        indices = np.asarray(frame_indices, dtype=np.int64)
    if indices.ndim != 1 or indices.size == 0:
        raise ValueError("frame_indices must be a non-empty 1-D sequence")
    if int(indices.min()) < 0 or int(indices.max()) >= len(prediction):
        raise IndexError("frame_indices are outside the prediction timeline")

    gain = _resolve_vector_gain(prediction, ground_truth, indices, vector_gain)
    stride = int(taxel_stride)
    x = np.linspace(-8.5, 8.5, SPATIAL_SHAPE[1])[::stride]
    y = np.linspace(30.0, 0.0, SPATIAL_SHAPE[0])[::stride]
    grid_x, grid_y = np.meshgrid(x, y)

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(12, 14),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    axes_flat = axes.reshape(-1)
    gt_quivers = []
    pred_quivers = []

    def uv(array: np.ndarray, frame: int, sensor: int) -> tuple[np.ndarray, np.ndarray]:
        channel = sensor * CHANNELS_PER_SENSOR
        return (
            array[frame, ::stride, ::stride, channel] * gain,
            array[frame, ::stride, ::stride, channel + 1] * gain,
        )

    first_frame = int(indices[0])
    for sensor, (axis, sensor_name) in enumerate(zip(axes_flat, SENSOR_NAMES)):
        axis.scatter(grid_x, grid_y, color="black", s=8, alpha=0.25)
        axis.set_aspect("equal")
        axis.set_xlim(float(x.min()) - 1.0, float(x.max()) + 1.0)
        axis.set_ylim(float(y.min()) - 1.0, float(y.max()) + 1.0)
        axis.set_xlabel("taxel x")
        axis.set_ylabel("taxel y")

        if plt_gt:
            u_gt, v_gt = uv(ground_truth, first_frame, sensor)
            gt_quivers.append(
                axis.quiver(
                    grid_x,
                    grid_y,
                    u_gt,
                    v_gt,
                    color="tab:blue",
                    angles="xy",
                    scale_units="xy",
                    scale=1.0,
                    width=0.003,
                    alpha=0.65,
                )
            )
        else:
            gt_quivers.append(None)
        if plt_pred:
            u_pred, v_pred = uv(prediction, first_frame, sensor)
            pred_quivers.append(
                axis.quiver(
                    grid_x,
                    grid_y,
                    u_pred,
                    v_pred,
                    color="tab:red",
                    angles="xy",
                    scale_units="xy",
                    scale=1.0,
                    width=0.003,
                    alpha=0.65,
                )
            )
        else:
            pred_quivers.append(None)
        axis.set_title(sensor_name)

    legend_handles = []
    if plt_gt:
        legend_handles.append(Line2D([0], [0], color="tab:blue", label="GT dx/dy"))
    if plt_pred:
        legend_handles.append(
            Line2D([0], [0], color="tab:red", label="Stage2 dx/dy")
        )
    axes_flat[0].legend(handles=legend_handles, loc="upper right")
    subtitle = fig.suptitle("")

    def update(animation_index: int):
        frame = int(indices[animation_index])
        artists = [subtitle]
        frame_diff = prediction[frame] - ground_truth[frame]
        frame_mae = float(np.mean(np.abs(frame_diff)))
        subtitle.set_text(
            f"{title} | source frame={frame} | MAE={frame_mae:.6g} | "
            f"display gain={gain:.3g}x"
        )
        for sensor, axis in enumerate(axes_flat):
            channel = sensor * CHANNELS_PER_SENSOR
            gt_dxy = np.linalg.norm(
                ground_truth[frame, ..., channel : channel + 2], axis=-1
            )
            pred_dxy = np.linalg.norm(
                prediction[frame, ..., channel : channel + 2], axis=-1
            )
            axis.set_title(
                f"{SENSOR_NAMES[sensor]} | mean |dxy| "
                f"GT={gt_dxy.mean():.4g}, Pred={pred_dxy.mean():.4g}"
            )
            artists.append(axis.title)
            if gt_quivers[sensor] is not None:
                gt_quivers[sensor].set_UVC(*uv(ground_truth, frame, sensor))
                artists.append(gt_quivers[sensor])
            if pred_quivers[sensor] is not None:
                pred_quivers[sensor].set_UVC(*uv(prediction, frame, sensor))
                artists.append(pred_quivers[sensor])
        return artists

    animation = FuncAnimation(
        fig,
        update,
        frames=len(indices),
        interval=1000.0 / float(fps),
        blit=False,
    )
    output_path = Path(save_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix.lower()
    if suffix == ".mp4":
        writer = FFMpegWriter(fps=float(fps))
    elif suffix == ".gif":
        writer = PillowWriter(fps=float(fps))
    else:
        raise ValueError("save_path suffix must be .mp4 or .gif")

    animation.save(str(output_path), writer=writer, dpi=int(dpi))
    preview_path = output_path.with_suffix(".png")
    if save_preview:
        update(0)
        fig.savefig(preview_path, dpi=int(dpi))
    plt.close(fig)
    return {
        "output": str(output_path),
        "preview": str(preview_path) if save_preview else "",
        "num_frames": int(len(indices)),
        "first_source_frame": int(indices[0]),
        "last_source_frame": int(indices[-1]),
        "vector_gain": float(gain),
    }


def _discover_files(input_path: Path, pattern: str) -> list[Path]:
    if input_path.is_file():
        return [input_path.resolve()]
    if not input_path.is_dir():
        raise FileNotFoundError(f"input path does not exist: {input_path}")
    files = sorted(path.resolve() for path in input_path.rglob(pattern))
    if not files:
        raise FileNotFoundError(f"no files matching {pattern!r} below {input_path}")
    return files


def _select_files(files: Sequence[Path], num_samples: int, indices: str | None) -> list[Path]:
    if indices:
        chosen_indices = [int(part.strip()) for part in indices.split(",") if part.strip()]
        if not chosen_indices:
            raise ValueError("--indices did not contain any integers")
        for index in chosen_indices:
            if index < 0 or index >= len(files):
                raise IndexError(f"sample index {index} outside [0,{len(files)})")
        return [files[index] for index in chosen_indices]
    if num_samples <= 0:
        raise ValueError("--num-samples must be positive")
    count = min(int(num_samples), len(files))
    selected = np.linspace(0, len(files) - 1, count, dtype=np.int64)
    return [files[int(index)] for index in np.unique(selected)]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render four-sensor Stage-2 tactile quiver animations."
    )
    parser.add_argument(
        "--input",
        default="outputs/tactile_eval/0730_65_stride30_4gpu/remove/samples",
        help="A reconstruction.npz file or a directory searched recursively.",
    )
    parser.add_argument("--pattern", default="reconstruction.npz")
    parser.add_argument(
        "--output-dir",
        default="outputs/tactile_eval/0730_65_stride30_4gpu/remove/quiver_vis",
    )
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument(
        "--indices",
        default=None,
        help="Comma-separated indices into the sorted discovered file list.",
    )
    parser.add_argument("--pred-key", default="tactile_stage2")
    parser.add_argument("--gt-key", default="tactile_gt")
    parser.add_argument("--mode", choices=("overlay", "pred", "gt"), default="overlay")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--temporal-stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=-1)
    parser.add_argument("--taxel-stride", type=int, default=2)
    parser.add_argument("--vector-gain", default="auto")
    parser.add_argument("--fps", type=float, default=8.0)
    parser.add_argument("--dpi", type=int, default=120)
    parser.add_argument("--format", choices=("mp4", "gif"), default="mp4")
    parser.add_argument(
        "--save-preview", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    input_path = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    discovered = _discover_files(input_path, str(args.pattern))
    selected = _select_files(discovered, int(args.num_samples), args.indices)
    print(
        json.dumps(
            {
                "input": str(input_path),
                "num_discovered": len(discovered),
                "selected": [str(path) for path in selected],
                "output_dir": str(output_dir),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    if args.dry_run:
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for sample_index, path in enumerate(tqdm(selected, desc="TactileQuiverSamples")):
        prediction, ground_truth, metadata = load_stage2_prediction(
            path,
            pred_key=str(args.pred_key),
            gt_key=str(args.gt_key),
        )
        frame_indices = select_valid_frames(
            prediction,
            ground_truth,
            start_frame=int(args.start_frame),
            temporal_stride=int(args.temporal_stride),
            max_frames=int(args.max_frames),
        )
        sample_name = path.parent.name if path.name == "reconstruction.npz" else path.stem
        output_path = output_dir / f"{sample_index:02d}_{sample_name}.{args.format}"
        result = plot_tactile_grids_animation_v2(
            prediction,
            ground_truth,
            output_path,
            plt_gt=args.mode in {"overlay", "gt"},
            plt_pred=args.mode in {"overlay", "pred"},
            fps=float(args.fps),
            frame_indices=frame_indices,
            taxel_stride=int(args.taxel_stride),
            vector_gain=str(args.vector_gain),
            dpi=int(args.dpi),
            title=sample_name,
            save_preview=bool(args.save_preview),
        )
        records.append(
            {
                "input": str(path),
                "metadata": metadata,
                **result,
            }
        )
        print(f"[saved] {output_path}")

    manifest_path = output_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(records, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(f"[complete] manifest={manifest_path}")


if __name__ == "__main__":
    main()
