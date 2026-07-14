from __future__ import annotations

import argparse
from pathlib import Path

from trainers.finetune_trainer import main as finetune_main
from utils.finetune_config import resolve_full_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Finetune flow-matching policy from a base checkpoint")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/finetune/config.yaml",
        help="Path to finetune config yaml",
    )
    args = parser.parse_args()

    policy_root = Path(__file__).resolve().parent
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = policy_root / config_path

    cfg = resolve_full_config(config_path, policy_root=policy_root)
    finetune_main(cfg, policy_root=policy_root, finetune_config_path=config_path)


if __name__ == "__main__":
    main()
