from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
import yaml

from models.fm import FlowMatchingPolicy, build_flow_policy
from tools.normalizer import DatasetNormalizer
from utils.train_utils import cfg_get, sync_fm_action_horizon_from_data

ACTION_DIMS = {"joint": 14, "eef": 20}
DEFAULT_NUM_INFERENCE_STEPS = 16
DEFAULT_SOLVER = "euler"
DEFAULT_VELOCITY_MODEL = "unet"


@dataclass(frozen=True)
class DeployConfig:
    action_process: str = "abs_eef"
    action_hz: float = 30.0


def infer_velocity_model_from_state_dict(state_dict: Mapping[str, Any]) -> str | None:
    has_dit = any(str(key).startswith("model.blocks.") for key in state_dict)
    has_unet = any(str(key).startswith("model.down_modules.") for key in state_dict)
    if has_dit and has_unet:
        raise ValueError("ambiguous checkpoint: contains both DiT and UNet weight keys")
    if has_dit:
        return "dit"
    if has_unet:
        return "unet"
    return None


def resolve_fm_cfg_for_inference(
    fm_cfg: Mapping[str, Any],
    policy_state_dict: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    resolved = dict(fm_cfg)
    inferred = (
        infer_velocity_model_from_state_dict(policy_state_dict)
        if policy_state_dict is not None
        else None
    )
    configured = str(resolved.get("velocity_model", "")).strip().lower()

    if inferred is not None:
        if not configured:
            resolved["velocity_model"] = inferred
        elif configured != inferred:
            raise ValueError(
                "velocity_model mismatch between config and checkpoint: "
                f"config={configured!r}, checkpoint={inferred!r}. "
                "Use resolved_config / checkpoint from the same training run."
            )
    else:
        resolved.setdefault("velocity_model", DEFAULT_VELOCITY_MODEL)
    return resolved


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
    policy_state_dict: Mapping[str, Any] | None = None,
) -> FlowMatchingPolicy:
    data_cfg = cfg["data"]
    fm_cfg = sync_fm_action_horizon_from_data(cfg["models"]["fm"], data_cfg)
    if match_training:
        fm_cfg["use_tactile"] = bool(data_cfg.get("use_tactile", fm_cfg.get("use_tactile", True)))
    fm_cfg = resolve_fm_cfg_for_inference(fm_cfg, policy_state_dict)
    cfg_for_build = dict(cfg)
    cfg_for_build["models"] = dict(cfg["models"])
    cfg_for_build["models"]["fm"] = fm_cfg
    return build_flow_policy(
        cfg_for_build,
        action_dim=action_dim_for_config(cfg),
        state_dim=action_dim_for_config(cfg),
        cond_steps=int(data_cfg["window_size"]),
    )


def policy_config_from_checkpoint_state(state: Mapping[str, Any], fallback: Mapping[str, Any]) -> dict[str, Any]:
    ckpt_cfg = state.get("config")
    if isinstance(ckpt_cfg, dict) and ckpt_cfg:
        merged = deepcopy(dict(fallback))
        merged.update(ckpt_cfg)
        if isinstance(fallback.get("models"), dict) and isinstance(ckpt_cfg.get("models"), dict):
            merged_models = deepcopy(dict(fallback["models"]))
            for key, value in ckpt_cfg["models"].items():
                if isinstance(value, dict) and isinstance(merged_models.get(key), dict):
                    merged_section = dict(merged_models[key])
                    merged_section.update(value)
                    merged_models[key] = merged_section
                else:
                    merged_models[key] = value
            merged["models"] = merged_models
        if isinstance(fallback.get("data"), dict) and isinstance(ckpt_cfg.get("data"), dict):
            merged_data = deepcopy(dict(fallback["data"]))
            merged_data.update(ckpt_cfg["data"])
            merged["data"] = merged_data
        return merged
    return dict(fallback)


def load_policy_state_dict_checked(
    policy: FlowMatchingPolicy,
    state_dict: Mapping[str, Any],
    checkpoint_path: Path | str,
) -> None:
    """Reject and clearly report missing, unexpected, or size-mismatched weights."""
    model_state = policy.state_dict()
    missing = sorted(key for key in model_state if key not in state_dict)
    unexpected = sorted(key for key in state_dict if key not in model_state)
    mismatched = sorted(
        (
            key,
            tuple(state_dict[key].shape),
            tuple(model_state[key].shape),
        )
        for key in model_state.keys() & state_dict.keys()
        if tuple(state_dict[key].shape) != tuple(model_state[key].shape)
    )
    if missing or unexpected or mismatched:
        lines = [f"Checkpoint is incompatible with current policy: {checkpoint_path}"]
        lines.append(f"Missing keys: {missing}")
        lines.append(f"Unexpected keys: {unexpected}")
        lines.append(f"Size mismatches: {mismatched}")
        # Legacy concat-fusion checkpoints carried a `memory_cond_fusion` MLP that
        # compressed [obs_cond|memory_global] 512->256. The current path feeds the
        # raw 512-wide condition straight into the velocity model, so those params
        # no longer exist. Never load such a checkpoint with strict=False.
        if any("memory_cond_fusion" in key for key in unexpected):
            lines.append(
                "Detected legacy `memory_cond_fusion` parameters: this checkpoint was "
                "trained with the old 512->256 fusion MLP, which has been removed. The "
                "current model concatenates [obs_cond | memory_global] into a 512-wide "
                "global_cond directly. Retrain/fine-tune; do NOT load with strict=False."
            )
        # global_cond width changes (e.g. 256->512 when concat_global_cond is enabled)
        # land on the UNet FiLM first block or the DiT cond_proj input dimension.
        if any(
            ("cond_encoder" in key or "cond_proj" in key)
            for key, _, _ in mismatched
        ):
            lines.append(
                "Velocity-model condition input dim mismatch (UNet FiLM / DiT cond_proj): "
                "the checkpoint's global_cond width differs from this model "
                "(memory_injection / concat_global_cond changes cond_dim*2). "
                "Rebuild the policy with the matching config; do NOT load with strict=False."
            )
        if any("view_fusion_proj" in key for key in missing) or any(
            "view_fusion_proj" in key for key, _, _ in mismatched
        ):
            lines.append(
                "The configured view_fusion_proj requires matching view-count weights; "
                "train or fine-tune it."
            )
        if any(key.endswith("visual_encoder.cls_token") for key in unexpected):
            lines.append(
                "The temporal learned CLS token was removed; the model now pools the "
                "16 transformed per-frame DINO CLS tokens and requires fine-tuning."
            )
        message = "\n".join(lines)
        print(message)
        raise RuntimeError(message)
    policy.load_state_dict(state_dict)


def load_runtime_checkpoint(
    path: Path | str,
    cfg: Mapping[str, Any],
    *,
    match_training: bool = True,
) -> tuple[FlowMatchingPolicy, DatasetNormalizer, dict[str, Any]]:
    state = torch.load(path, map_location="cpu", weights_only=False)
    policy_cfg = policy_config_from_checkpoint_state(state, cfg)
    policy = build_policy_from_cfg(
        policy_cfg,
        match_training=match_training,
        policy_state_dict=state.get("policy_state_dict"),
    )
    load_policy_state_dict_checked(policy, state["policy_state_dict"], path)
    normalizer = DatasetNormalizer.load_state_dict(state["normalizer_state_dict"])
    policy.eval()
    return policy, normalizer, state
