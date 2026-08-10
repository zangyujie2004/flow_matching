from __future__ import annotations

from collections import defaultdict
from contextlib import nullcontext
from copy import copy, deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import torch
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from datasets import ZarrDataset, build_dataloader
from models.fm import FlowMatchingPolicy, build_flow_policy
from tools.normalizer import DatasetNormalizer
from trainers.eval_open_loop import evaluate_open_loop
from utils.tensor import move_to_device
from utils.train_utils import cfg_get, detach_scalar_dict, log_hparams_to_tensorboard, set_seed, sync_fm_action_horizon_from_data


def get_autocast_context(device: torch.device, use_amp: bool):
    enabled = bool(use_amp and device.type == "cuda")
    if not enabled:
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=torch.float16)


def _inject_latent_identity(data_cfg: dict, fm_cfg: Mapping[str, Any]) -> None:
    if bool(data_cfg.get("use_camera_latent", False)):
        from models.fm.encoders.dino_v2 import resolve_dino_model_name

        data_cfg["latent_cache_image_encoder_name"] = fm_cfg.get("image_encoder_name", "dinov2")
        data_cfg["latent_cache_image_model_name"] = resolve_dino_model_name(
            fm_cfg.get("image_encoder_name"),
            fm_cfg.get("dino_model_name"),
        )


def _expand_episode_groups(value: Any) -> dict[str, tuple[int, ...]]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError("eval_data.episode_groups must be a non-empty mapping")
    groups: dict[str, tuple[int, ...]] = {}
    for name, bounds in value.items():
        if isinstance(bounds, (str, bytes)) or not isinstance(bounds, Sequence) or len(bounds) != 2:
            raise TypeError(f"eval_data.episode_groups.{name} must be [start, stop]")
        start, stop = (int(item) for item in bounds)
        if start < 0 or stop <= start:
            raise ValueError(
                f"eval_data.episode_groups.{name} must satisfy 0 <= start < stop"
            )
        groups[str(name)] = tuple(range(start, stop))
    return groups


def build_dataset_and_loader(cfg: dict) -> tuple[ZarrDataset, DataLoader]:
    data_cfg = dict(cfg["data"])
    fm_cfg = dict(cfg.get("models", {}).get("fm", {}))
    _inject_latent_identity(data_cfg, fm_cfg)
    train_cfg = cfg["train"]
    dataset = ZarrDataset.from_config(data_cfg)
    if bool(data_cfg.get("use_camera_latent", False)):
        token_mode = getattr(dataset, "latent_token_mode", None)
        view_pool = str(fm_cfg.get("view_pool", "global_concat")).strip().lower()
        if token_mode == "cls" and view_pool in {"local_pool", "local_attn"}:
            raise ValueError(
                f"latent cache token_mode=cls is incompatible with models.fm.view_pool={view_pool!r}. "
                "Use precompute.token_mode=all, or set view_pool=global_concat."
            )
    loader = build_dataloader(
        dataset,
        batch_size=int(train_cfg.get("batch_size", 32)),
        shuffle=True,
        num_workers=int(train_cfg.get("num_workers", 0)),
        drop_last=bool(train_cfg.get("drop_last", True)),
        pin_memory=bool(train_cfg.get("pin_memory", True)),
        persistent_workers=train_cfg.get("persistent_workers"),
        prefetch_factor=int(train_cfg.get("prefetch_factor", 2)),
    )
    return dataset, loader


def build_eval_datasets(
    cfg: dict,
    train_dataset: ZarrDataset,
) -> dict[str, ZarrDataset]:
    """Build named external test groups using only the train normalizer."""
    eval_overlay = dict(cfg.get("eval_data") or {})
    if not eval_overlay:
        return {}
    groups = _expand_episode_groups(eval_overlay.pop("episode_groups", None))
    base_cfg = dict(cfg["data"])
    base_cfg.pop("episode_range", None)
    base_cfg.pop("episode_indices", None)
    base_cfg.update(eval_overlay)
    base_cfg["fit_normalizer"] = False
    fm_cfg = dict(cfg.get("models", {}).get("fm", {}))
    _inject_latent_identity(base_cfg, fm_cfg)

    datasets: dict[str, ZarrDataset] = {}
    for name, episode_indices in groups.items():
        if (
            str(base_cfg["root_dir"]) == str(train_dataset.root_dir)
            and str(base_cfg.get("latent_cache_root_dir"))
            == str(train_dataset.latent_cache_root_dir)
        ):
            # Same Zarr/cache: share RAM arrays and cache handles, but keep an
            # independent window table and normalized-action cache per group.
            dataset = copy(train_dataset)
            dataset.requested_episode_indices = episode_indices
            episode_set = set(episode_indices)
            dataset.windows = [
                window
                for window in dataset._build_windows()
                if int(window[2]) in episode_set
            ]
            dataset.window_lookup = {
                (int(anchor_t), int(ep_idx)): idx
                for idx, (anchor_t, _ep_end, ep_idx, base_mode) in enumerate(dataset.windows)
                if base_mode == "original"
            }
        else:
            group_cfg = dict(base_cfg)
            group_cfg["episode_indices"] = episode_indices
            dataset = ZarrDataset.from_config(group_cfg)
        dataset.normalizer = train_dataset.normalizer
        dataset._precompute_normalized_actions()
        datasets[name] = dataset
    return datasets


def build_policy(cfg: dict, device: torch.device, dataset: ZarrDataset) -> FlowMatchingPolicy:
    data_cfg = cfg["data"]
    fm_cfg = sync_fm_action_horizon_from_data(cfg["models"]["fm"], data_cfg)
    use_tactile = bool(data_cfg.get("use_tactile", True))
    fm_cfg["use_tactile"] = use_tactile

    model_n_views = int(fm_cfg.get("n_image_views", dataset.n_image_views))
    if model_n_views != dataset.n_image_views:
        raise ValueError(
            f"models.fm.n_image_views={model_n_views} does not match "
            f"data.camera_views (n_image_views={dataset.n_image_views})"
        )
    fm_cfg["n_image_views"] = model_n_views

    if dataset.use_camera_latent and not bool(fm_cfg.get("freeze_image_encoder", True)):
        raise ValueError("use_camera_latent=True requires models.fm.freeze_image_encoder=true.")

    cfg_for_build = dict(cfg)
    cfg_for_build["models"] = dict(cfg["models"])
    cfg_for_build["models"]["fm"] = fm_cfg
    policy = build_flow_policy(
        cfg_for_build,
        action_dim=dataset.action_dim,
        state_dim=dataset.action_dim,
        cond_steps=dataset.window_size,
    ).to(device)
    return policy


def train_one_epoch(
    policy: FlowMatchingPolicy,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    grad_clip: float | None = None,
    global_step: int = 0,
    writer: SummaryWriter | None = None,
    scaler: Optional[torch.amp.GradScaler] = None,
    use_amp: bool = False,
    max_batches: int | None = None,
) -> tuple[dict[str, float], int]:
    policy.train()
    metric_sum: dict[str, float] = defaultdict(float)
    count = 0

    pbar = tqdm(loader, desc="Train", leave=False)
    for batch_idx, batch in enumerate(pbar):
        if max_batches is not None and batch_idx >= max_batches:
            break
        global_step += 1
        batch = move_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)

        with get_autocast_context(device, use_amp):
            out = policy(batch)
            loss = out["loss"]
            scalar_metrics = detach_scalar_dict(out.get("metrics", {}))

        if scaler is not None:
            scaler.scale(loss).backward()
            if grad_clip is not None and grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(policy.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip is not None and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(policy.parameters(), grad_clip)
            optimizer.step()

        batch_size = int(batch["action"].shape[0])
        metric_sum["loss"] += float(loss.detach().item()) * batch_size
        for key, value in scalar_metrics.items():
            metric_sum[key] += value * batch_size
        count += batch_size

        step_lr = optimizer.param_groups[0]["lr"]
        if writer is not None:
            writer.add_scalar("Step/lr", step_lr, global_step)
            writer.add_scalar("Step/loss", float(loss.detach().item()), global_step)
            for key, value in scalar_metrics.items():
                writer.add_scalar(f"Step/{key}", value, global_step)

        postfix = {"loss": f"{loss.detach().item():.4f}", "lr": f"{step_lr:.2e}"}
        pbar.set_postfix(postfix)

    avg = {key: value / max(count, 1) for key, value in metric_sum.items()}
    return avg, global_step


def get_checkpoint_state(
    policy: FlowMatchingPolicy,
    optimizer: torch.optim.Optimizer,
    dataset: ZarrDataset,
    *,
    epoch: int,
    global_step: int,
    cfg: dict,
) -> dict[str, Any]:
    return {
        "epoch": int(epoch),
        "global_step": int(global_step),
        "policy_state_dict": policy.state_dict(),
        "normalizer_state_dict": deepcopy(dataset.normalizer.state_dict()),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": cfg,
    }


def save_checkpoint(path: str | Path, state: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def load_checkpoint(
    path: str | Path,
    policy: FlowMatchingPolicy,
    optimizer: torch.optim.Optimizer | None,
    dataset: ZarrDataset,
) -> dict[str, Any]:
    state = torch.load(path, map_location="cpu", weights_only=False)
    policy.load_state_dict(state["policy_state_dict"])
    dataset.normalizer = DatasetNormalizer.load_state_dict(state["normalizer_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in state:
        optimizer.load_state_dict(state["optimizer_state_dict"])
    return state


def main(cfg: dict) -> None:
    set_seed(int(cfg.get("seed", 42)))
    device_name = cfg_get(cfg, "runtime.device", "cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)

    dataset, train_loader = build_dataset_and_loader(cfg)
    print(f"Train windows: {len(dataset)}")
    eval_datasets = build_eval_datasets(cfg, dataset)
    for name, eval_dataset in eval_datasets.items():
        print(
            f"Eval {name}: episodes={len(eval_dataset.requested_episode_indices or ())}, "
            f"windows={len(eval_dataset)}, root={eval_dataset.root_dir}"
        )

    # Persist data→fm action horizon into resolved_config for deploy/infer.
    cfg = dict(cfg)
    cfg["models"] = dict(cfg.get("models") or {})
    cfg["models"]["fm"] = sync_fm_action_horizon_from_data(
        cfg["models"].get("fm") or {},
        cfg["data"],
    )

    policy = build_policy(cfg, device, dataset)
    trainable_params = [param for param in policy.parameters() if param.requires_grad]
    print(f"Trainable parameters: {sum(param.numel() for param in trainable_params):,}")

    train_cfg = cfg["train"]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=float(train_cfg["optimizer"]["lr"]),
        weight_decay=float(train_cfg["optimizer"]["weight_decay"]),
    )

    output_cfg = cfg["output"]
    output_root = str(output_cfg.get("root_dir", "outputs"))
    run_name = output_cfg.get("run_name") or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(output_root) / str(run_name)
    ckpt_dir = run_dir / "checkpoints"
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    with open(run_dir / "resolved_config.yaml", "w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg, handle, sort_keys=False)

    writer = SummaryWriter(log_dir=str(run_dir))
    log_hparams_to_tensorboard(writer, cfg, str(run_dir))
    print(f"TensorBoard log dir: {run_dir}")

    epochs = int(train_cfg.get("epochs", 100))
    ckpt_cfg = cfg.get("checkpoint", {})
    save_every = max(1, int(ckpt_cfg.get("save_every", 100)))
    grad_clip = train_cfg.get("grad_clip")
    open_loop_every = int(train_cfg.get("open_loop_test_every", 0))
    open_loop_max_batches = max(1, int(train_cfg.get("open_loop_test_max_batches", 20)))
    max_train_batches = train_cfg.get("max_train_batches")
    if max_train_batches is not None:
        max_train_batches = max(1, int(max_train_batches))
    plot_samples = int(train_cfg.get("plot_samples", 4))
    plot_dims = str(train_cfg.get("plot_dims", "auto"))
    use_amp = bool(train_cfg.get("use_amp", False) and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    fm_cfg = cfg["models"]["fm"]
    num_inference_steps = int(fm_cfg.get("num_inference_steps", 16))
    solver = str(fm_cfg.get("solver", "euler"))

    global_step = 0
    start_epoch = 1
    resume_path = train_cfg.get("resume_path")
    if resume_path:
        resume_state = load_checkpoint(resume_path, policy, optimizer, dataset)
        for eval_dataset in eval_datasets.values():
            eval_dataset.normalizer = dataset.normalizer
            eval_dataset._precompute_normalized_actions()
        global_step = int(resume_state.get("global_step", 0))
        start_epoch = int(resume_state.get("epoch", 0)) + 1
        policy.to(device)
        print(f"Resumed from {resume_path} at epoch={start_epoch}, global_step={global_step}")

    for epoch in range(start_epoch, epochs + 1):
        train_avg, global_step = train_one_epoch(
            policy,
            train_loader,
            optimizer,
            device,
            grad_clip=grad_clip,
            global_step=global_step,
            writer=writer,
            scaler=scaler if use_amp else None,
            use_amp=use_amp,
            max_batches=max_train_batches,
        )
        train_loss = train_avg["loss"]

        open_loop_metrics: dict[str, dict[str, float]] = {}
        if open_loop_every > 0 and (epoch % open_loop_every == 0 or epoch == epochs):
            targets = eval_datasets or {"train": dataset}
            for name, eval_dataset in targets.items():
                metrics = evaluate_open_loop(
                    policy,
                    eval_dataset,
                    dataset.normalizer,
                    device,
                    epoch=epoch,
                    seed=int(cfg.get("seed", 42)),
                    max_batches=open_loop_max_batches,
                    batch_size=int(train_cfg.get("batch_size", 32)),
                    plot_samples=plot_samples,
                    plot_dims=plot_dims,
                    out_dir=run_dir / "open_loop" / name,
                    writer=None,
                    num_inference_steps=num_inference_steps,
                    solver=solver,
                )
                open_loop_metrics[name] = metrics
                writer.add_scalar(
                    f"OpenLoop/{name}/action_mse", metrics["action_mse"], epoch
                )
                writer.add_scalar(
                    f"OpenLoop/{name}/action_l1", metrics["action_l1"], epoch
                )

        curr_lr = optimizer.param_groups[0]["lr"]
        writer.add_scalar("Epoch/lr", curr_lr, epoch)
        writer.add_scalar("Epoch/train_loss", train_loss, epoch)
        for key, value in train_avg.items():
            if key != "loss":
                writer.add_scalar(f"Epoch/train_{key}", value, epoch)

        message = f"[Epoch {epoch:03d}] train_loss={train_loss:.6f}"
        for name, metrics in open_loop_metrics.items():
            message += (
                f", {name}_l1={metrics['action_l1']:.6f}"
                f", {name}_mse={metrics['action_mse']:.6f}"
            )
        print(message)

        state = get_checkpoint_state(
            policy,
            optimizer,
            dataset,
            epoch=epoch,
            global_step=global_step,
            cfg=cfg,
        )
        save_checkpoint(ckpt_dir / "latest.pt", state)
        if epoch % save_every == 0:
            save_checkpoint(ckpt_dir / f"epoch_{epoch:04d}.pt", state)

        writer.flush()

    writer.close()
    print(f"Training finished. Artifacts saved in: {run_dir}")
