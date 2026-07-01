from __future__ import annotations

import argparse
from pathlib import Path

from trainers.policy_trainer import main as train_main
from utils.train_utils import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Train flow-matching policy")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to training config yaml",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        policy_root = Path(__file__).resolve().parent
        config_path = policy_root / config_path

    cfg = load_config(str(config_path))
    train_main(cfg)


if __name__ == "__main__":
    main()
