from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.tensorboard import SummaryWriter

from models.fm.partial_ft import apply_partial_ft, resolve_partial_ft_preset
from tools.normalizer import DatasetNormalizer
from trainers.eval_open_loop import evaluate_open_loop
from trainers.policy_trainer import (
    build_dataset_and_loader,
    build_policy,
    get_checkpoint_state,
    save_checkpoint,
    train_one_epoch,
)
from utils.finetune_config import resolve_run_dir, write_finetune_manifest
from utils.train_utils import cfg_get, log_hparams_to_tensorboard, set_seed


def load_policy_weights_only(path: str | Path, policy: torch.nn.Module) -> dict[str, Any]:
    state = torch.load(path, map_location="cpu", weights_only=False)
    if "policy_state_dict" not in state:
        raise KeyError(f"Checkpoint missing policy_state_dict: {path}")
    policy.load_state_dict(state["policy_state_dict"])
    return state


def apply_normalizer_mode(
    dataset,
    ckpt_state: dict[str, Any],
    *,
    normalizer_mode: str,
) -> None:
    if normalizer_mode == "keep_base":
        if "normalizer_state_dict" not in ckpt_state:
            raise KeyError("keep_base requires normalizer_state_dict in base checkpoint")
        dataset.normalizer = DatasetNormalizer.load_state_dict(ckpt_state["normalizer_state_dict"])
        dataset._precompute_normalized_actions()
        return
    if normalizer_mode == "refit":
        if dataset.normalizer is None:
            raise RuntimeError("refit mode requires dataset.normalizer from ZarrDataset fit")
        return
    raise ValueError(f"Unsupported normalizer_mode: {normalizer_mode!r}")


def main(
    cfg: dict[str, Any],
    *,
    policy_root: Path | None = None,
    finetune_config_path: Path | None = None,
) -> Path:
    policy_root = policy_root or Path(__file__).resolve().parents[1]
    set_seed(int(cfg.get("seed", 42)))

    finetune_cfg = cfg.get("finetune") or {}
    normalizer_mode = str(finetune_cfg.get("normalizer_mode", "keep_base"))
    if normalizer_mode not in ("refit", "keep_base"):
        raise ValueError("finetune.normalizer_mode must be 'refit' or 'keep_base'")

    device_name = cfg_get(cfg, "runtime.device", "cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)

    run_dir = resolve_run_dir(cfg, policy_root=policy_root)
    ckpt_dir = run_dir / "checkpoints"
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    with open(run_dir / "resolved_config.yaml", "w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg, handle, sort_keys=False)
    write_finetune_manifest(run_dir, cfg, finetune_config_path=finetune_config_path)

    dataset, train_loader = build_dataset_and_loader(cfg)
    print(f"[finetune] Train windows: {len(dataset)}")
    if normalizer_mode == "refit":
        print(f"[finetune] Normalizer refit on dataset: {cfg_get(cfg, 'data.root_dir')}")
    else:
        print("[finetune] Normalizer keep_base: will load from base checkpoint")

    policy = build_policy(cfg, device, dataset)

    base_checkpoint = Path(str(finetune_cfg["base_checkpoint"]))
    ckpt_state = torch.load(base_checkpoint, map_location="cpu", weights_only=False)
    if "policy_state_dict" not in ckpt_state:
        raise KeyError(f"Checkpoint missing policy_state_dict: {base_checkpoint}")
    policy.load_state_dict(ckpt_state["policy_state_dict"])
    apply_normalizer_mode(dataset, ckpt_state, normalizer_mode=normalizer_mode)

    partial_ft_preset = resolve_partial_ft_preset(finetune_cfg)
    if partial_ft_preset is not None:
        trainable_params = apply_partial_ft(policy, partial_ft_preset)
    else:
        trainable_params = [param for param in policy.parameters() if param.requires_grad]
        print(f"[finetune] Full finetune trainable parameters: {sum(param.numel() for param in trainable_params):,}")

    policy.to(device)
    print(f"[finetune] Loaded policy weights from: {base_checkpoint}")
    if normalizer_mode == "keep_base":
        print("[finetune] Loaded base normalizer from checkpoint")
    if finetune_cfg.get("reset_optimizer", True):
        print("[finetune] Optimizer reset: using fresh AdamW")
    if finetune_cfg.get("reset_epoch_counter", True):
        print("[finetune] Epoch counter reset: starting from epoch 1")

    train_cfg = cfg["train"]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=float(train_cfg["optimizer"]["lr"]),
        weight_decay=float(cfg_get(train_cfg, "optimizer.weight_decay", 1.0e-4)),
    )

    writer = SummaryWriter(log_dir=str(run_dir))
    log_hparams_to_tensorboard(writer, cfg, str(run_dir))
    print(f"[finetune] TensorBoard log dir: {run_dir}")
    print(f"[finetune] Output dir: {run_dir}")

    epochs = int(train_cfg.get("epochs", 64))
    ckpt_cfg = cfg.get("checkpoint", {})
    save_every = max(1, int(ckpt_cfg.get("save_every", 50)))
    grad_clip = train_cfg.get("grad_clip")
    open_loop_every = int(train_cfg.get("open_loop_test_every", 0))
    open_loop_max_batches = max(1, int(train_cfg.get("open_loop_test_max_batches", 16)))
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
    if not finetune_cfg.get("reset_epoch_counter", True):
        global_step = int(ckpt_state.get("global_step", 0))
        start_epoch = int(ckpt_state.get("epoch", 0)) + 1

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

        open_loop_metrics = None
        if open_loop_every > 0 and (epoch % open_loop_every == 0 or epoch == epochs):
            open_loop_metrics = evaluate_open_loop(
                policy,
                dataset,
                dataset.normalizer,
                device,
                epoch=epoch,
                seed=int(cfg.get("seed", 42)),
                max_batches=open_loop_max_batches,
                batch_size=int(train_cfg.get("batch_size", 32)),
                plot_samples=plot_samples,
                plot_dims=plot_dims,
                out_dir=run_dir / "open_loop",
                writer=writer,
                num_inference_steps=num_inference_steps,
                solver=solver,
            )

        curr_lr = optimizer.param_groups[0]["lr"]
        writer.add_scalar("Epoch/lr", curr_lr, epoch)
        writer.add_scalar("Epoch/train_loss", train_loss, epoch)
        for key, value in train_avg.items():
            if key != "loss":
                writer.add_scalar(f"Epoch/train_{key}", value, epoch)

        message = f"[Finetune Epoch {epoch:03d}] train_loss={train_loss:.6f}"
        if open_loop_metrics is not None:
            message += (
                f", open_loop_l1={open_loop_metrics['action_l1']:.6f}"
                f", open_loop_mse={open_loop_metrics['action_mse']:.6f}"
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
    print(f"[finetune] Finished. Artifacts saved in: {run_dir}")
    return run_dir
