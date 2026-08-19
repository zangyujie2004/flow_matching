from __future__ import annotations

import argparse
from pathlib import Path

from trainers.tactile_ae_trainer import main as train_main
from utils.train_utils import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 1 tactile autoencoder training")
    parser.add_argument(
        "--config",
        default="configs/train/tactile_ae.yaml",
        help="Path to tactile AE yaml config",
    )
    parser.add_argument(
        "--resume",
        default=None,
        help="Resume from a Stage 1 checkpoint (for example checkpoints/latest.pt)",
    )
    parser.add_argument(
        "--amp",
        choices=("off", "fp16", "bf16"),
        default=None,
        help="Override train.mixed_precision from the config",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override per-process batch size (use 32 on each rank for 8-GPU global batch 256)",
    )
    args = parser.parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = Path(__file__).resolve().parent / config_path
    cfg = load_config(str(config_path))
    cfg["train"] = dict(cfg["train"])
    if args.resume is not None:
        cfg["train"]["resume_path"] = str(Path(args.resume).expanduser())
    if args.amp is not None:
        cfg["train"]["mixed_precision"] = args.amp
    if args.batch_size is not None:
        if args.batch_size < 1:
            parser.error("--batch-size must be positive")
        cfg["train"]["batch_size"] = int(args.batch_size)
    train_main(cfg)


if __name__ == "__main__":
    main()
