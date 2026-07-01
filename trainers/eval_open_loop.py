from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Mapping

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data.dataloader import default_collate
from tqdm import tqdm

from datasets import ZarrDataset
from tools.normalizer import DatasetNormalizer
from utils.tensor import move_to_device


_EEF_DUAL_ARM_PLOT_DIMS = (0, 1, 2, 9, 10, 11, 12, 19)
_JOINT_DUAL_ARM_PLOT_DIMS = tuple(range(14))
_EEF_DIM_LABELS = ("x", "y", "z", "grip")
_JOINT_DIM_LABELS = tuple(f"j{i}" for i in range(7))
_DEFAULT_EEF_PLOT_DIMS_TEXT = "0,1,2,9,10,11,12,19"


def default_plot_dims(action_dim: int) -> list[int]:
    if action_dim == 20:
        return list(_EEF_DUAL_ARM_PLOT_DIMS)
    if action_dim == 14:
        return list(_JOINT_DUAL_ARM_PLOT_DIMS)
    return list(range(action_dim))


def parse_dims(value: str, action_dim: int) -> list[int]:
    text = str(value).strip().lower()
    if text in {"all", "*", "auto"}:
        return default_plot_dims(action_dim)
    dims = [int(part.strip()) for part in text.split(",") if part.strip()]
    bad = [dim for dim in dims if dim < 0 or dim >= action_dim]
    if bad:
        normalized = ",".join(str(dim) for dim in dims)
        if normalized == _DEFAULT_EEF_PLOT_DIMS_TEXT and action_dim != 20:
            return default_plot_dims(action_dim)
        raise ValueError(f"plot dim out of range for action_dim={action_dim}: {bad}")
    return dims


def _ylim_for_dims(
    pred: np.ndarray,
    gt: np.ndarray,
    dims: list[int],
    *,
    margin: float = 0.08,
) -> tuple[float, float]:
    if not dims:
        return -1.0, 1.0
    values = np.concatenate([pred[:, dims].reshape(-1), gt[:, dims].reshape(-1)])
    vmin = float(np.min(values))
    vmax = float(np.max(values))
    if vmax - vmin < 1e-8:
        pad = max(abs(vmin), 1.0) * 0.1 + 1e-3
    else:
        pad = (vmax - vmin) * margin
    return vmin - pad, vmax + pad


def _plot_dual_arm_curves(
    pred: np.ndarray,
    gt: np.ndarray,
    dims: list[int],
    out_path: Path,
    title: str,
) -> None:
    if len(dims) != 8:
        raise ValueError(
            f"dual-arm plot expects 8 dims (left xyz+grip, right xyz+grip), got {len(dims)}"
        )

    left_dims = dims[:4]
    right_dims = dims[4:]
    steps = np.arange(pred.shape[0])

    left_xyz_ylim = _ylim_for_dims(pred, gt, left_dims[:3])
    right_xyz_ylim = _ylim_for_dims(pred, gt, right_dims[:3])
    grip_ylim = _ylim_for_dims(pred, gt, [left_dims[3], right_dims[3]])

    fig, axes = plt.subplots(4, 2, figsize=(9.0, 10.0), squeeze=False)
    column_titles = ("left arm", "right arm")
    for col, (arm_dims, xyz_ylim, arm_title) in enumerate(
        (
            (left_dims, left_xyz_ylim, column_titles[0]),
            (right_dims, right_xyz_ylim, column_titles[1]),
        )
    ):
        for row in range(3):
            dim = arm_dims[row]
            axis = axes[row, col]
            axis.plot(steps, gt[:, dim], label="gt", linewidth=1.8)
            axis.plot(steps, pred[:, dim], label="pred", linewidth=1.4, linestyle="--")
            axis.set_ylabel(_EEF_DIM_LABELS[row])
            axis.set_ylim(*xyz_ylim)
            axis.grid(True, alpha=0.25)
            if row == 0:
                axis.set_title(arm_title)

        grip_dim = arm_dims[3]
        axis = axes[3, col]
        axis.plot(steps, gt[:, grip_dim], label="gt", linewidth=1.8)
        axis.plot(steps, pred[:, grip_dim], label="pred", linewidth=1.4, linestyle="--")
        axis.set_ylabel(_EEF_DIM_LABELS[3])
        axis.set_xlabel("step")
        axis.set_ylim(*grip_ylim)
        axis.grid(True, alpha=0.25)

    for row in range(3):
        axes[row, 0].set_xticklabels([])
        axes[row, 1].set_xticklabels([])

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 0.98, 0.95))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_dual_arm_joint_curves(
    pred: np.ndarray,
    gt: np.ndarray,
    dims: list[int],
    out_path: Path,
    title: str,
) -> None:
    if len(dims) != 14:
        raise ValueError(
            f"dual-arm joint plot expects 14 dims (left j0-j6, right j0-j6), got {len(dims)}"
        )

    left_dims = dims[:7]
    right_dims = dims[7:]
    steps = np.arange(pred.shape[0])

    fig, axes = plt.subplots(7, 2, figsize=(9.0, 14.0), squeeze=False)
    column_titles = ("left arm", "right arm")
    for col, (arm_dims, arm_title) in enumerate(
        (
            (left_dims, column_titles[0]),
            (right_dims, column_titles[1]),
        )
    ):
        for row, dim in enumerate(arm_dims):
            axis = axes[row, col]
            ylim = _ylim_for_dims(pred, gt, [dim])
            axis.plot(steps, gt[:, dim], label="gt", linewidth=1.8)
            axis.plot(steps, pred[:, dim], label="pred", linewidth=1.4, linestyle="--")
            axis.set_ylabel(_JOINT_DIM_LABELS[row])
            axis.set_ylim(*ylim)
            axis.grid(True, alpha=0.25)
            if row == 0:
                axis.set_title(arm_title)
            if row < 6:
                axis.set_xticklabels([])

        axes[-1, col].set_xlabel("step")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 0.98, 0.95))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_action_curves(
    pred: np.ndarray,
    gt: np.ndarray,
    dims: list[int],
    out_path: Path,
    title: str,
) -> None:
    if len(dims) == 8:
        _plot_dual_arm_curves(pred, gt, dims, out_path, title)
        return
    if len(dims) == 14:
        _plot_dual_arm_joint_curves(pred, gt, dims, out_path, title)
        return

    steps = np.arange(pred.shape[0])
    n_dims = len(dims)
    ncols = min(4, max(1, n_dims))
    nrows = int(np.ceil(n_dims / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 2.5 * nrows), squeeze=False)

    for axis, dim in zip(axes.ravel(), dims):
        axis.plot(steps, gt[:, dim], label="gt", linewidth=1.8)
        axis.plot(steps, pred[:, dim], label="pred", linewidth=1.4, linestyle="--")
        axis.set_title(f"action[{dim}]")
        axis.set_xlabel("step")
        axis.grid(True, alpha=0.25)

    for axis in axes.ravel()[n_dims:]:
        axis.axis("off")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 0.98, 0.95))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _absolute_actions_for_batch(
    dataset: ZarrDataset,
    normalizer: DatasetNormalizer,
    pred_norm: torch.Tensor,
    meta: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Return (pred_abs, gt_abs) arrays with shape (B, T, D) in zarr absolute space."""
    idx_values = meta["idx"]
    if torch.is_tensor(idx_values):
        indices = [int(value) for value in idx_values.detach().cpu().tolist()]
    else:
        indices = [int(value) for value in idx_values]

    pred_np = pred_norm.detach().cpu().numpy()
    batch_size = len(indices)
    pred_abs_list: list[np.ndarray] = []
    gt_abs_list: list[np.ndarray] = []

    for batch_idx, window_idx in enumerate(indices):
        s0, s1 = dataset.state_range(window_idx)
        a0, a1 = dataset.action_range(window_idx)
        state_raw = dataset.get_state(s0, s1)
        gt_abs = dataset.get_action(a0, a1).astype(np.float32, copy=False)
        pred_single = normalizer.unnormalize_action_np(pred_np[batch_idx], state_raw)
        pred_abs_list.append(pred_single.astype(np.float32, copy=False))
        gt_abs_list.append(gt_abs)

    pred_abs = np.stack(pred_abs_list, axis=0)
    gt_abs = np.stack(gt_abs_list, axis=0)
    return pred_abs, gt_abs


@torch.no_grad()
def evaluate_open_loop(
    policy: torch.nn.Module,
    dataset: ZarrDataset,
    normalizer: DatasetNormalizer,
    device: torch.device,
    *,
    epoch: int,
    seed: int,
    max_batches: int,
    batch_size: int,
    plot_samples: int,
    plot_dims: str,
    out_dir: Path | None,
    writer=None,
    num_inference_steps: int | None = None,
    solver: str | None = None,
) -> Dict[str, float]:
    policy.eval()
    rng = np.random.default_rng(int(seed) + int(epoch))
    eval_batch_size = max(1, int(batch_size))
    total_windows = max(1, int(max_batches)) * eval_batch_size
    total_windows = min(total_windows, len(dataset))
    if total_windows <= 0:
        raise RuntimeError("Dataset has no windows for open-loop evaluation.")

    chosen = rng.choice(len(dataset), size=total_windows, replace=False)
    plot_dir = None if out_dir is None else out_dir / "plots"
    if plot_dir is not None:
        plot_dir.mkdir(parents=True, exist_ok=True)

    sq_sum = 0.0
    abs_sum = 0.0
    elem_count = 0
    sample_count = 0
    plotted = 0
    dims_for_plot: list[int] | None = None

    pbar = tqdm(range(int(max_batches)), desc="OpenLoop", leave=False)
    for batch_idx in pbar:
        start = batch_idx * eval_batch_size
        end = start + eval_batch_size
        if start >= len(chosen):
            break
        batch_indices = chosen[start:end]
        items = [dataset[int(window_idx)] for window_idx in batch_indices]
        batch = default_collate(items)
        batch = move_to_device(batch, device)

        result = policy.predict_action(
            batch["obs"],
            num_inference_steps=num_inference_steps,
            solver=solver,
        )
        pred_norm = result["action_pred_normalized"]
        pred_abs, gt_abs = _absolute_actions_for_batch(
            dataset,
            normalizer,
            pred_norm,
            batch["meta"],
        )

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

        batch_l1 = float(np.mean(np.abs(diff)))
        pbar.set_postfix(l1=f"{batch_l1:.4f}")

        if plot_samples != 0 and plot_dir is not None:
            if dims_for_plot is None:
                dims_for_plot = parse_dims(plot_dims, action_dim)
            if plot_samples < 0 or plotted < plot_samples:
                out_path = plot_dir / f"epoch_{epoch:04d}_batch{batch_idx:04d}_item00.png"
                plot_action_curves(
                    pred_abs[0],
                    gt_abs[0],
                    dims_for_plot,
                    out_path,
                    title=f"Open-loop absolute action | epoch={epoch}, batch={batch_idx}",
                )
                plotted += 1

    if elem_count == 0:
        raise RuntimeError("No valid open-loop samples were evaluated.")

    metrics = {
        "action_mse": sq_sum / elem_count,
        "action_l1": abs_sum / elem_count,
        "num_samples": float(sample_count),
        "num_plots": float(plotted),
    }

    if writer is not None:
        writer.add_scalar("OpenLoop/action_mse", metrics["action_mse"], epoch)
        writer.add_scalar("OpenLoop/action_l1", metrics["action_l1"], epoch)

    return metrics
