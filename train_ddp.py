from __future__ import annotations

import argparse
from pathlib import Path

from trainers.ddp_trainer import main as train_ddp_main
from tools.latent_cache import apply_resolved_latent_cache_root_dir
from utils.train_utils import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="DDP train flow-matching policy (torchrun)")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train/config.yaml",
        help="Path to training config yaml",
    )
    parser.add_argument(
        "--amp",
        choices=("off", "fp16", "bf16"),
        default=None,
        help="Override train.mixed_precision from the config",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        policy_root = Path(__file__).resolve().parent
        config_path = policy_root / config_path

    cfg = load_config(str(config_path))
    if args.amp is not None:
        cfg["train"] = dict(cfg["train"])
        cfg["train"]["mixed_precision"] = args.amp
    cfg = apply_resolved_latent_cache_root_dir(cfg)
    train_ddp_main(cfg)


if __name__ == "__main__":
    main()
