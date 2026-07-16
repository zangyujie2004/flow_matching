"""Tests for partial fine-tuning presets."""

from __future__ import annotations

import os
import sys

import torch

_POLICY_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _POLICY_ROOT not in sys.path:
    sys.path.insert(0, _POLICY_ROOT)

from models.fm import build_flow_policy
from models.fm.partial_ft import apply_partial_ft


def _build_policy():
    fm = {
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
        "down_dims": [256, 512, 1024],
        "kernel_size": 5,
        "n_groups": 8,
        "num_inference_steps": 16,
        "solver": "euler",
    }
    return build_flow_policy(
        {"models": {"fm": fm}},
        action_dim=20,
        state_dim=20,
        cond_steps=8,
    )


def test_action_head_freezes_condition_encoder() -> None:
    policy = _build_policy()
    trainable = apply_partial_ft(policy, "action_head")

    cond_trainable = [p for p in policy.condition_encoder.parameters() if p.requires_grad]
    model_trainable = [p for p in policy.model.parameters() if p.requires_grad]

    assert len(cond_trainable) == 0
    assert len(model_trainable) > 0
    assert len(trainable) == len(model_trainable)

    batch = {
        "obs": {
            "image_backbone_feat": torch.randn(1, 1, 2, 384),
            "state": torch.randn(1, 8, 20),
        },
        "action": torch.randn(1, 32, 20),
    }
    out = policy.compute_loss(batch)
    out["loss"].backward()

    assert all(p.grad is None for p in policy.condition_encoder.parameters())
    assert any(p.grad is not None for p in policy.model.parameters())


def test_dit_velocity_model_forward() -> None:
    fm = {
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
        "velocity_model": "dit",
        "dit_hidden_dim": 512,
        "dit_depth": 14,
        "dit_num_heads": 8,
        "dit_mlp_ratio": 4.0,
        "dit_dropout": 0.1,
        "num_inference_steps": 16,
        "solver": "euler",
    }
    policy = build_flow_policy(
        {"models": {"fm": fm}},
        action_dim=20,
        state_dim=20,
        cond_steps=8,
    )
    assert policy.velocity_model == "dit"

    batch = {
        "obs": {
            "image_backbone_feat": torch.randn(1, 1, 2, 384),
            "state": torch.randn(1, 8, 20),
        },
        "action": torch.randn(1, 32, 20),
    }
    out = policy.compute_loss(batch)
    assert torch.isfinite(out["loss"])
    out["loss"].backward()

    policy.eval()
    pred = policy.predict_action(batch["obs"], num_inference_steps=4)
    assert pred["action_normalized"].shape == (1, 32, 20)


if __name__ == "__main__":
    test_action_head_freezes_condition_encoder()
    test_dit_velocity_model_forward()
    print("partial_ft action_head OK")
