from __future__ import annotations

from collections import defaultdict
from datetime import datetime
import os
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.distributed as dist
import torch.nn.functional as F
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from datasets import (
    TactileAEFrameDataset,
    fit_tactile_frame_normalizer,
    split_episode_indices,
)
from datasets.tactile_ae_dataset import (
    episode_bounds_from_zarr,
    resolve_replay_buffer_path,
)
from models.fm.encoders import build_tactile_autoencoder
from tools.normalizer import FieldNormalizer
from trainers.policy_trainer import (
    build_grad_scaler,
    get_autocast_context,
    resolve_mixed_precision,
)
from trainers.dist_utils import (
    barrier,
    broadcast_object,
    cleanup_distributed,
    get_rank,
    get_world_size,
    init_distributed,
    is_dist_available_and_initialized,
    is_main_process,
    print_rank0,
    unwrap_model,
)
from utils.train_utils import cfg_get, set_seed


def reconstruction_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    loss_type: str,
) -> torch.Tensor:
    key = str(loss_type).strip().lower()
    if key == "l1":
        return F.l1_loss(prediction, target)
    if key in {"mse", "l2"}:
        return F.mse_loss(prediction, target)
    raise ValueError(f"unsupported tactile AE loss type={loss_type!r}")


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    optimizer: torch.optim.Optimizer | None,
    loss_type: str,
    mixed_precision: str,
    scaler,
    grad_clip: float | None,
    show_progress: bool,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals: dict[str, float] = defaultdict(float)
    count = 0
    context = torch.enable_grad if training else torch.no_grad

    with context():
        progress = tqdm(
            loader,
            desc="AE Train" if training else "AE Val",
            leave=False,
            disable=not show_progress,
        )
        for batch in progress:
            tactile = batch["tactile"].to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with get_autocast_context(device, mixed_precision):
                output = model(tactile)
                reconstruction = output["reconstruction"]
                loss = reconstruction_loss(
                    reconstruction,
                    tactile,
                    loss_type=loss_type,
                )
                l1 = F.l1_loss(reconstruction, tactile)
                mse = F.mse_loss(reconstruction, tactile)

            if training:
                if scaler is not None:
                    scaler.scale(loss).backward()
                    if grad_clip is not None and grad_clip > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    if grad_clip is not None and grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    optimizer.step()

            batch_size = int(tactile.shape[0])
            totals["loss"] += float(loss.detach()) * batch_size
            totals["l1"] += float(l1.detach()) * batch_size
            totals["mse"] += float(mse.detach()) * batch_size
            count += batch_size
            if show_progress:
                progress.set_postfix(loss=f"{float(loss.detach()):.5f}")

    metric_names = ("loss", "l1", "mse")
    packed = torch.tensor(
        [*(totals[name] for name in metric_names), float(count)],
        dtype=torch.float64,
        device=device,
    )
    if is_dist_available_and_initialized():
        dist.all_reduce(packed, op=dist.ReduceOp.SUM)
    global_count = max(float(packed[-1].item()), 1.0)
    return {
        name: float(packed[index].item()) / global_count
        for index, name in enumerate(metric_names)
    }


def checkpoint_state(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    normalizer: FieldNormalizer,
    cfg: Mapping[str, Any],
    epoch: int,
    best_val_loss: float,
    scaler=None,
) -> dict[str, Any]:
    state = {
        "stage": "tactile_autoencoder",
        "epoch": int(epoch),
        "best_val_loss": float(best_val_loss),
        "model_config": model.config_dict(),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "tactile_normalizer_state": normalizer.state_dict(),
        "config": dict(cfg),
    }
    if scaler is not None:
        state["scaler_state_dict"] = scaler.state_dict()
    return state


def load_checkpoint_state(path: str | Path) -> dict[str, Any]:
    checkpoint_path = Path(path).expanduser()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"tactile AE checkpoint not found: {checkpoint_path}")
    try:
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        state = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(state, dict):
        raise TypeError(
            f"tactile AE checkpoint must contain a mapping, got {type(state).__name__}"
        )
    if state.get("stage") != "tactile_autoencoder":
        raise ValueError(
            f"checkpoint stage={state.get('stage')!r} is not 'tactile_autoencoder': "
            f"{checkpoint_path}"
        )
    for key in (
        "epoch",
        "model_state_dict",
        "optimizer_state_dict",
        "tactile_normalizer_state",
    ):
        if key not in state:
            raise KeyError(f"checkpoint is missing {key!r}: {checkpoint_path}")
    return state


def _resolve_runtime(cfg: Mapping[str, Any]) -> tuple[int, int, int, torch.device]:
    """Use torchrun ranks when present, while preserving the original single-process path."""
    world_size_env = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size_env > 1:
        ddp_cfg = dict(cfg_get(cfg, "runtime.ddp", {}) or {})
        backend = str(ddp_cfg.get("backend", "nccl"))
        timeout_s = float(ddp_cfg.get("timeout_s", 1800.0))
        return init_distributed(backend=backend, timeout_s=timeout_s)

    device = torch.device(
        str(
            dict(cfg.get("runtime") or {}).get(
                "device",
                "cuda" if torch.cuda.is_available() else "cpu",
            )
        )
    )
    return 0, 0, 1, device


def main(cfg: dict[str, Any]) -> None:
    if not bool(cfg.get("predict_tactile", True)):
        raise ValueError(
            "Stage 1 AE training is disabled because predict_tactile=false"
        )
    rank, local_rank, world_size, device = _resolve_runtime(cfg)
    seed = int(cfg.get("seed", 42))
    set_seed(seed + rank)
    ddp_cfg = dict(cfg_get(cfg, "runtime.ddp", {}) or {})
    backend = str(ddp_cfg.get("backend", "nccl"))
    find_unused = bool(ddp_cfg.get("find_unused_parameters", False))
    print_rank0(
        f"[tactile-ae] world_size={world_size} backend={backend if world_size > 1 else 'none'} "
        f"device={device}"
    )

    data_cfg = dict(cfg["data"])
    train_cfg = dict(cfg["train"])
    resume_path_raw = train_cfg.get("resume_path")
    resume_path = (
        None if not resume_path_raw else str(Path(str(resume_path_raw)).expanduser())
    )
    resume_state = None
    if resume_path is not None and is_main_process():
        resume_state = load_checkpoint_state(resume_path)

    zarr_path = resolve_replay_buffer_path(data_cfg["root_dir"])
    starts, ends = episode_bounds_from_zarr(zarr_path)
    train_episodes, val_episodes = split_episode_indices(
        len(ends),
        val_fraction=float(data_cfg.get("val_fraction", 0.1)),
        seed=int(cfg.get("seed", 42)),
    )
    output_range_raw = data_cfg.get("output_range", (-1.0, 1.0))
    output_range = (float(output_range_raw[0]), float(output_range_raw[1]))
    normalizer_state = None
    if is_main_process():
        if resume_state is not None:
            normalizer_state = resume_state["tactile_normalizer_state"]
            print_rank0(f"[tactile-ae] restoring normalizer from {resume_path}")
        else:
            print_rank0("[tactile-ae] fitting tactile normalizer on rank 0...")
            normalizer_state = fit_tactile_frame_normalizer(
                zarr_path,
                train_episodes,
                tactile_key=str(data_cfg.get("tactile_key", "tactile")),
                output_range=output_range,
                batch_frames=int(data_cfg.get("normalizer_batch_frames", 512)),
            ).state_dict()
    normalizer_state = broadcast_object(normalizer_state, src=0)
    if normalizer_state is None:
        raise RuntimeError("rank 0 did not provide a tactile normalizer state")
    normalizer = FieldNormalizer.from_state_dict(normalizer_state)
    print_rank0("[tactile-ae] tactile normalizer ready on all ranks")

    dataset_kwargs = {
        "root_dir": zarr_path,
        "normalizer": normalizer,
        "tactile_key": str(data_cfg.get("tactile_key", "tactile")),
        "frame_stride": int(data_cfg.get("frame_stride", 1)),
    }
    train_dataset = TactileAEFrameDataset(
        episode_indices=train_episodes,
        **dataset_kwargs,
    )
    val_dataset = TactileAEFrameDataset(
        episode_indices=val_episodes,
        **dataset_kwargs,
    )

    num_workers = int(train_cfg.get("num_workers", 0))
    loader_kwargs = {
        "batch_size": int(train_cfg.get("batch_size", 256)),
        "num_workers": num_workers,
        "pin_memory": bool(train_cfg.get("pin_memory", True)),
        "persistent_workers": bool(
            train_cfg.get("persistent_workers", num_workers > 0)
        )
        and num_workers > 0,
    }
    train_sampler = None
    val_sampler = None
    if world_size > 1:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=bool(train_cfg.get("drop_last", True)),
            seed=seed,
        )
        # Padding by at most world_size-1 frames keeps every DDP rank on the
        # same number of validation forwards and avoids collective hangs.
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            drop_last=False,
            seed=seed,
        )

    train_loader = DataLoader(
        train_dataset,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        drop_last=bool(train_cfg.get("drop_last", True)),
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        sampler=val_sampler,
        drop_last=False,
        **loader_kwargs,
    )

    model = build_tactile_autoencoder(cfg.get("model")).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("lr", 3.0e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 1.0e-4)),
    )
    mixed_precision = resolve_mixed_precision(train_cfg, device)
    scaler = build_grad_scaler(mixed_precision)

    start_epoch = 1
    best_val_loss = float("inf")
    if resume_path is not None:
        if resume_state is None:
            resume_state = load_checkpoint_state(resume_path)
        model.load_state_dict(resume_state["model_state_dict"], strict=True)
        optimizer.load_state_dict(resume_state["optimizer_state_dict"])
        start_epoch = int(resume_state["epoch"]) + 1
        best_val_loss = float(resume_state.get("best_val_loss", float("inf")))
        if scaler is not None and "scaler_state_dict" in resume_state:
            scaler.load_state_dict(resume_state["scaler_state_dict"])
        print_rank0(
            f"[tactile-ae] resumed {resume_path}: completed_epoch={start_epoch - 1}, "
            f"next_epoch={start_epoch}, best_val={best_val_loss:.6f}"
        )

    if world_size > 1:
        model = DDP(
            model,
            device_ids=[local_rank] if device.type == "cuda" else None,
            output_device=local_rank if device.type == "cuda" else None,
            find_unused_parameters=find_unused,
        )

    output_cfg = dict(cfg.get("output") or {})
    root = Path(str(output_cfg.get("root_dir", "outputs/tactile_ae")))
    run_name = str(
        output_cfg.get("run_name") or datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    run_dir = root / run_name
    checkpoint_dir = run_dir / "checkpoints"
    writer = None
    if is_main_process():
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        with (run_dir / "resolved_config.yaml").open("w", encoding="utf-8") as handle:
            yaml.safe_dump(cfg, handle, sort_keys=False)
        writer = SummaryWriter(str(run_dir))
    barrier()

    per_gpu_batch = int(train_cfg.get("batch_size", 256))
    print_rank0(
        f"[tactile-ae] train_episodes={len(train_episodes)} "
        f"val_episodes={len(val_episodes)} train_frames={len(train_dataset)} "
        f"val_frames={len(val_dataset)} params="
        f"{sum(parameter.numel() for parameter in unwrap_model(model).parameters()):,} "
        f"batch={per_gpu_batch}/gpu global_batch={per_gpu_batch * world_size} "
        f"mixed_precision={mixed_precision}"
    )

    epochs = int(train_cfg.get("epochs", 100))
    save_every = max(1, int(dict(cfg.get("checkpoint") or {}).get("save_every", 10)))
    loss_type = str(dict(cfg.get("loss") or {}).get("type", "l1"))
    grad_clip = train_cfg.get("grad_clip")
    if start_epoch > epochs:
        print_rank0(
            f"[tactile-ae] checkpoint already completed epoch {start_epoch - 1}; "
            f"configured epochs={epochs}, nothing to train"
        )

    for epoch in range(start_epoch, epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        train_metrics = run_epoch(
            model,
            train_loader,
            device,
            optimizer=optimizer,
            loss_type=loss_type,
            mixed_precision=mixed_precision,
            scaler=scaler,
            grad_clip=grad_clip,
            show_progress=is_main_process(),
        )
        val_metrics = run_epoch(
            model,
            val_loader,
            device,
            optimizer=None,
            loss_type=loss_type,
            mixed_precision=mixed_precision,
            scaler=None,
            grad_clip=None,
            show_progress=False,
        )
        improved = val_metrics["loss"] < best_val_loss
        if improved:
            best_val_loss = val_metrics["loss"]
        if is_main_process():
            assert writer is not None
            for key, value in train_metrics.items():
                writer.add_scalar(f"Train/{key}", value, epoch)
            for key, value in val_metrics.items():
                writer.add_scalar(f"Val/{key}", value, epoch)

            state = checkpoint_state(
                model=unwrap_model(model),
                optimizer=optimizer,
                normalizer=normalizer,
                cfg=cfg,
                epoch=epoch,
                best_val_loss=best_val_loss,
                scaler=scaler,
            )
            torch.save(state, checkpoint_dir / "latest.pt")
            if improved:
                torch.save(state, checkpoint_dir / "best.pt")
            if epoch % save_every == 0:
                torch.save(state, checkpoint_dir / f"epoch_{epoch:04d}.pt")
            writer.flush()
            print(
                f"[AE Epoch {epoch:03d}] "
                f"train={train_metrics['loss']:.6f} "
                f"val={val_metrics['loss']:.6f} best={best_val_loss:.6f}"
            )
        barrier()

    if is_main_process() and writer is not None:
        writer.close()
        print(f"Tactile AE training finished: {run_dir}")
    cleanup_distributed()
