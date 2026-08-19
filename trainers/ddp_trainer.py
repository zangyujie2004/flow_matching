from __future__ import annotations

from datetime import datetime
from pathlib import Path
from time import perf_counter

import torch
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

from datasets import ZarrDataset, build_dataloader
from models.fm import resolve_tactile_condition_encoder_type
from tools.normalizer import DatasetNormalizer
from trainers.dist_utils import (
    barrier,
    broadcast_object,
    cleanup_distributed,
    get_rank,
    get_world_size,
    init_distributed,
    is_main_process,
    print_rank0,
    reduce_mean,
    unwrap_model,
)
from trainers.eval_open_loop import evaluate_open_loop
from trainers.policy_trainer import (
    build_grad_scaler,
    build_policy,
    get_checkpoint_state,
    load_checkpoint,
    resolve_mixed_precision,
    save_checkpoint,
    train_one_epoch,
)
from utils.train_utils import cfg_get, log_hparams_to_tensorboard, set_seed, sync_fm_action_horizon_from_data


def build_dataset_and_ddp_loader(cfg: dict):
    """Build one dataset per rank, but fit the normalizer only on rank 0."""
    data_cfg = dict(cfg["data"])
    fm_cfg = dict(cfg.get("models", {}).get("fm", {}))
    data_cfg["predict_tactile"] = bool(fm_cfg.get("predict_tactile", False))
    data_cfg["tactile_condition_encoder_type"] = (
        resolve_tactile_condition_encoder_type(
            predict_tactile=data_cfg["predict_tactile"],
            tactile_encoder_type=fm_cfg.get("tactile_encoder_type"),
            tactile_condition_encoder_type=fm_cfg.get(
                "tactile_condition_encoder_type"
            ),
        )
    )
    if bool(data_cfg.get("use_camera_latent", False)):
        from models.fm.encoders.dino_v2 import resolve_dino_model_name

        data_cfg["latent_cache_image_encoder_name"] = fm_cfg.get("image_encoder_name", "dinov2")
        data_cfg["latent_cache_image_model_name"] = resolve_dino_model_name(
            fm_cfg.get("image_encoder_name"),
            fm_cfg.get("dino_model_name"),
        )
    train_cfg = cfg["train"]
    # DDP must not repeat the expensive full-window fit/cache on every rank.
    # The shared normalizer is fitted below; actions are normalized per sample
    # by DataLoader workers.
    data_cfg["fit_normalizer"] = False
    norm_cfg = dict(data_cfg.get("norm") or {})
    norm_cfg["cache_actions"] = False
    data_cfg["norm"] = norm_cfg
    dataset = ZarrDataset.from_config(data_cfg)
    if bool(data_cfg.get("use_camera_latent", False)):
        token_mode = getattr(dataset, "latent_token_mode", None)
        view_pool = str(fm_cfg.get("view_pool", "global_concat")).strip().lower()
        if token_mode == "cls" and view_pool in {"local_pool", "local_attn"}:
            raise ValueError(
                f"latent cache token_mode=cls is incompatible with models.fm.view_pool={view_pool!r}. "
                "Use precompute.token_mode=all, or set view_pool=global_concat."
            )

    normalizer_state = None
    if is_main_process():
        start = perf_counter()
        print_rank0(
            "[ddp] fitting normalizer on rank 0 only "
            f"(max_windows={dataset.normalizer_max_windows}, "
            f"batch_windows={dataset.normalizer_batch_windows})..."
        )
        normalizer = DatasetNormalizer.build(
            dataset,
            output_range=dataset.norm_output_range,
            max_windows=dataset.normalizer_max_windows,
            batch_windows=dataset.normalizer_batch_windows,
        )
        normalizer_state = normalizer.state_dict()
        print_rank0(f"[ddp] rank 0 normalizer fit finished in {perf_counter() - start:.1f}s")

    normalizer_state = broadcast_object(normalizer_state, src=0)
    if normalizer_state is None:
        raise RuntimeError("rank 0 did not broadcast a normalizer state")
    dataset.normalizer = DatasetNormalizer.load_state_dict(normalizer_state)
    dataset.cached_norm_action = None
    print_rank0("[ddp] normalizer broadcast complete; action normalization mode=per-sample")

    sampler = DistributedSampler(
        dataset,
        num_replicas=get_world_size(),
        rank=get_rank(),
        shuffle=True,
        drop_last=bool(train_cfg.get("drop_last", True)),
    )
    loader = build_dataloader(
        dataset,
        batch_size=int(train_cfg.get("batch_size", 32)),
        shuffle=False,
        num_workers=int(train_cfg.get("num_workers", 0)),
        drop_last=bool(train_cfg.get("drop_last", True)),
        pin_memory=bool(train_cfg.get("pin_memory", True)),
        persistent_workers=train_cfg.get("persistent_workers"),
        prefetch_factor=int(train_cfg.get("prefetch_factor", 2)),
        sampler=sampler,
    )
    return dataset, loader, sampler


def main(cfg: dict) -> None:
    ddp_cfg = dict(cfg_get(cfg, "runtime.ddp", {}) or {})
    backend = str(ddp_cfg.get("backend", "nccl"))
    find_unused = bool(ddp_cfg.get("find_unused_parameters", True))
    timeout_s = float(ddp_cfg.get("timeout_s", 1800.0))

    rank, local_rank, world_size, device = init_distributed(
        backend=backend,
        timeout_s=timeout_s,
    )
    seed = int(cfg.get("seed", 42))
    set_seed(seed + rank)
    print_rank0(
        f"[ddp] world_size={world_size} backend={backend} "
        f"timeout_s={timeout_s:g} device={device}"
    )

    dataset, train_loader, sampler = build_dataset_and_ddp_loader(cfg)
    print_rank0(f"Train windows: {len(dataset)} (per-rank batches≈{len(train_loader)})")

    cfg = dict(cfg)
    cfg["models"] = dict(cfg.get("models") or {})
    cfg["models"]["fm"] = sync_fm_action_horizon_from_data(
        cfg["models"].get("fm") or {},
        cfg["data"],
    )

    policy = build_policy(cfg, device, dataset)
    trainable_params = [p for p in policy.parameters() if p.requires_grad]
    print_rank0(f"Trainable parameters: {sum(p.numel() for p in trainable_params):,}")

    train_cfg = cfg["train"]
    per_gpu_bs = int(train_cfg.get("batch_size", 32))
    print_rank0(
        f"[ddp] batch_size per GPU={per_gpu_bs}, global_batch={per_gpu_bs * world_size}"
    )

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

    writer = None
    if is_main_process():
        run_dir.mkdir(parents=True, exist_ok=True)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        with open(run_dir / "resolved_config.yaml", "w", encoding="utf-8") as handle:
            yaml.safe_dump(cfg, handle, sort_keys=False)
        writer = SummaryWriter(log_dir=str(run_dir))
        log_hparams_to_tensorboard(writer, cfg, str(run_dir))
        print(f"TensorBoard log dir: {run_dir}")
    barrier()

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
    mixed_precision = resolve_mixed_precision(train_cfg, device)
    scaler = build_grad_scaler(mixed_precision)

    print_rank0(
        f"[train] mode=ddp epochs={epochs} "
        f"open_loop_every={open_loop_every}(epoch) save_every={save_every}(epoch) "
        f"batch_size={per_gpu_bs}/gpu nproc={world_size} "
        f"global_batch={per_gpu_bs * world_size} "
        f"mixed_precision={mixed_precision} "
        f"| TB: Step/*=global_step(per-rank), Epoch/* & OpenLoop/*=epoch"
    )

    fm_cfg = cfg["models"]["fm"]
    num_inference_steps = int(fm_cfg.get("num_inference_steps", 16))
    solver = str(fm_cfg.get("solver", "euler"))

    global_step = 0
    start_epoch = 1
    resume_path = train_cfg.get("resume_path")
    if resume_path:
        resume_state = load_checkpoint(resume_path, policy, optimizer, dataset)
        global_step = int(resume_state.get("global_step", 0))
        start_epoch = int(resume_state.get("epoch", 0)) + 1
        policy.to(device)
        print_rank0(
            f"Resumed from {resume_path} at epoch={start_epoch}, global_step={global_step}"
        )

    if world_size > 1:
        policy = DDP(
            policy,
            device_ids=[local_rank] if device.type == "cuda" else None,
            output_device=local_rank if device.type == "cuda" else None,
            find_unused_parameters=find_unused,
        )

    for epoch in range(start_epoch, epochs + 1):
        sampler.set_epoch(epoch)
        train_avg, global_step = train_one_epoch(
            policy,
            train_loader,
            optimizer,
            device,
            grad_clip=grad_clip,
            global_step=global_step,
            writer=writer if is_main_process() else None,
            scaler=scaler,
            mixed_precision=mixed_precision,
            max_batches=max_train_batches,
            is_main=is_main_process(),
        )

        train_loss = reduce_mean(float(train_avg["loss"]))
        reduced_avg = {k: reduce_mean(float(v)) for k, v in train_avg.items()}

        open_loop_metrics = None
        if open_loop_every > 0 and (epoch % open_loop_every == 0 or epoch == epochs):
            if is_main_process():
                open_loop_metrics = evaluate_open_loop(
                    unwrap_model(policy),
                    dataset,
                    dataset.normalizer,
                    device,
                    epoch=epoch,
                    seed=seed,
                    max_batches=open_loop_max_batches,
                    batch_size=per_gpu_bs,
                    plot_samples=plot_samples,
                    plot_dims=plot_dims,
                    out_dir=run_dir / "open_loop",
                    writer=writer,
                    num_inference_steps=num_inference_steps,
                    solver=solver,
                )
            barrier()

        if is_main_process():
            curr_lr = optimizer.param_groups[0]["lr"]
            assert writer is not None
            writer.add_scalar("Epoch/lr", curr_lr, epoch)
            writer.add_scalar("Epoch/train_loss", train_loss, epoch)
            for key, value in reduced_avg.items():
                if key != "loss":
                    writer.add_scalar(f"Epoch/train_{key}", value, epoch)

            message = (
                f"[Epoch {epoch:03d}] train_loss={train_loss:.6f} "
                f"(global_bs={per_gpu_bs * world_size})"
            )
            if open_loop_metrics is not None:
                message += (
                    f", open_loop_l1={open_loop_metrics['action_l1']:.6f}"
                    f", open_loop_mse={open_loop_metrics['action_mse']:.6f}"
                )
            print(message)

            state = get_checkpoint_state(
                unwrap_model(policy),
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
        barrier()

    if is_main_process() and writer is not None:
        writer.close()
        print(f"Training finished. Artifacts saved in: {run_dir}")
    cleanup_distributed()
