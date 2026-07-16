from __future__ import annotations

import os
import sys

import torch

_POLICY_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _POLICY_ROOT not in sys.path:
    sys.path.insert(0, _POLICY_ROOT)

from infer.config import (
    build_policy_from_cfg,
    infer_velocity_model_from_state_dict,
    load_runtime_checkpoint,
    resolve_fm_cfg_for_inference,
)
from models.fm import build_flow_policy


def _base_cfg(*, velocity_model: str) -> dict:
    return {
        "data": {
            "action_type": "joint",
            "window_size": 8,
            "use_tactile": False,
        },
        "models": {
            "fm": {
                "cond_dim": 256,
                "n_image_views": 2,
                "image_encoder_name": "dinov2",
                "dino_model_name": "vit_small_patch14_dinov2.lvd142m",
                "freeze_image_encoder": True,
                "image_pretrained": False,
                "image_feat_dim": 256,
                "use_tactile": False,
                "action_horizon": 32,
                "n_action_steps": 32,
                "diffusion_step_embed_dim": 256,
                "velocity_model": velocity_model,
                "down_dims": [256, 512, 1024],
                "kernel_size": 5,
                "n_groups": 8,
                "dit_hidden_dim": 512,
                "dit_depth": 14,
                "dit_num_heads": 8,
                "dit_mlp_ratio": 4.0,
                "dit_dropout": 0.1,
                "num_inference_steps": 4,
                "solver": "euler",
            }
        },
    }


def _build_and_state(*, velocity_model: str) -> dict[str, torch.Tensor]:
    policy = build_flow_policy(
        _base_cfg(velocity_model=velocity_model),
        action_dim=14,
        state_dim=14,
        cond_steps=8,
    )
    return policy.state_dict()


def test_infer_velocity_model_from_state_dict() -> None:
    assert infer_velocity_model_from_state_dict(_build_and_state(velocity_model="unet")) == "unet"
    assert infer_velocity_model_from_state_dict(_build_and_state(velocity_model="dit")) == "dit"


def test_resolve_fm_cfg_infers_missing_velocity_model() -> None:
    fm_cfg = dict(_base_cfg(velocity_model="unet")["models"]["fm"])
    fm_cfg.pop("velocity_model")
    resolved = resolve_fm_cfg_for_inference(fm_cfg, _build_and_state(velocity_model="dit"))
    assert resolved["velocity_model"] == "dit"


def test_build_policy_from_cfg_rejects_mismatch() -> None:
    cfg = _base_cfg(velocity_model="unet")
    state_dict = _build_and_state(velocity_model="dit")
    try:
        build_policy_from_cfg(cfg, policy_state_dict=state_dict)
    except ValueError as exc:
        assert "velocity_model mismatch" in str(exc)
    else:
        raise AssertionError("expected velocity_model mismatch error")


def test_load_runtime_checkpoint_roundtrip(tmp_path) -> None:
    for velocity_model in ("unet", "dit"):
        cfg = _base_cfg(velocity_model=velocity_model)
        policy = build_flow_policy(cfg, action_dim=14, state_dim=14, cond_steps=8)
        ckpt_path = tmp_path / f"{velocity_model}.pt"
        torch.save(
            {
                "policy_state_dict": policy.state_dict(),
                "normalizer_state_dict": {
                    "action_type": "joint",
                    "action_representation": "relative",
                    "state": {
                        "scale": torch.ones(14),
                        "offset": torch.zeros(14),
                    },
                    "action": {
                        "scale": torch.ones(14),
                        "offset": torch.zeros(14),
                    },
                    "tactile": None,
                },
                "config": cfg,
            },
            ckpt_path,
        )

        loaded_policy, normalizer, _state = load_runtime_checkpoint(ckpt_path, cfg)
        assert loaded_policy.velocity_model == velocity_model
        assert normalizer.action_type == "joint"

        obs = {
            "image_backbone_feat": torch.randn(1, 1, 2, 384),
            "state": torch.randn(1, 8, 14),
        }
        loaded_policy.eval()
        pred = loaded_policy.predict_action(obs, num_inference_steps=2)
        assert pred["action_pred_normalized"].shape == (1, 32, 14)


if __name__ == "__main__":
    test_infer_velocity_model_from_state_dict()
    test_resolve_fm_cfg_infers_missing_velocity_model()
    test_build_policy_from_cfg_rejects_mismatch()
    print("infer backbone tests OK")
