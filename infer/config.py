from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
import yaml

from models.fm import FlowMatchingPolicy, build_flow_policy
from tools.normalizer import DatasetNormalizer
from utils.train_utils import cfg_get

ACTION_DIMS = {"joint": 14, "eef": 20}
DEFAULT_NUM_INFERENCE_STEPS = 16
DEFAULT_SOLVER = "euler"


@dataclass(frozen=True)
class DeployConfig:
    action_process: str = "abs_eef"
    action_hz: float = 30.0


def load_run_config(run_dir: Path | str) -> dict[str, Any]:
    run_dir = Path(run_dir).expanduser().resolve()
    config_path = run_dir / "resolved_config.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing resolved config: {config_path}")
    with open(config_path, encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if not isinstance(cfg, dict):
        raise TypeError(f"resolved_config.yaml must be a mapping: {config_path}")
    return cfg


def default_action_process(cfg: Mapping[str, Any]) -> str:
    action_type = str(cfg_get(cfg, "data.action_type", "eef"))
    if action_type == "joint":
        return "abs_qpos"
    if action_type == "eef":
        return "abs_eef"
    raise ValueError(f"unsupported data.action_type={action_type!r}")


def parse_deploy_config(cfg: Mapping[str, Any]) -> DeployConfig:
    deploy = dict(cfg.get("deploy") or {})
    return DeployConfig(
        action_process=str(deploy.get("action_process", default_action_process(cfg))),
        action_hz=float(deploy.get("action_hz", 30.0)),
    )


def action_dim_for_config(cfg: Mapping[str, Any]) -> int:
    action_type = str(cfg_get(cfg, "data.action_type", "eef"))
    if action_type not in ACTION_DIMS:
        raise ValueError(f"unsupported data.action_type={action_type!r}")
    return ACTION_DIMS[action_type]


def build_policy_from_cfg(
    cfg: Mapping[str, Any],
    *,
    match_training: bool = True,
) -> FlowMatchingPolicy:
    data_cfg = cfg["data"]
    fm_cfg = dict(cfg["models"]["fm"])
    if match_training:
        fm_cfg["use_tactile"] = bool(data_cfg.get("use_tactile", fm_cfg.get("use_tactile", True)))
    return build_flow_policy(
        {"models": {"fm": fm_cfg}},
        action_dim=action_dim_for_config(cfg),
        state_dim=action_dim_for_config(cfg),
        cond_steps=int(data_cfg["window_size"]),
    )


def policy_config_from_checkpoint_state(state: Mapping[str, Any], fallback: Mapping[str, Any]) -> dict[str, Any]:
    ckpt_cfg = state.get("config")
    if isinstance(ckpt_cfg, dict) and ckpt_cfg:
        return dict(ckpt_cfg)
    return dict(fallback)


def load_runtime_checkpoint(
    path: Path | str,
    policy: FlowMatchingPolicy,
) -> tuple[DatasetNormalizer, dict[str, Any]]:
    state = torch.load(path, map_location="cpu", weights_only=False)
    policy.load_state_dict(state["policy_state_dict"])
    normalizer = DatasetNormalizer.load_state_dict(state["normalizer_state_dict"])
    return normalizer, state
