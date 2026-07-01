"""Forward smoke tests for FlowMatchingPolicy."""

from __future__ import annotations

import os
import sys

import torch
import yaml

_POLICY_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _POLICY_ROOT not in sys.path:
    sys.path.insert(0, _POLICY_ROOT)

from models.fm import build_flow_policy


def _mock_batch(
    *,
    batch_size: int = 2,
    window_size: int = 8,
    action_horizon: int = 32,
    action_dim: int = 20,
    use_tactile: bool = True,
) -> dict:
    obs = {
        "image": torch.randint(0, 255, (batch_size, 1, 3, 3, 224, 224), dtype=torch.uint8),
        "state": torch.randn(batch_size, window_size, action_dim),
    }
    if use_tactile:
        obs["tactile"] = torch.randn(batch_size, window_size, 35, 20, 12)
    return {
        "obs": obs,
        "action": torch.randn(batch_size, action_horizon, action_dim),
    }


def test_mock_forward_backward() -> None:
    cfg = yaml.safe_load(open(os.path.join(_POLICY_ROOT, "configs", "config.yaml")))
    fm = dict(cfg["models"]["fm"])
    fm["image_pretrained"] = False

    policy = build_flow_policy(
        {"models": {"fm": fm}},
        action_dim=20,
        state_dim=20,
        cond_steps=cfg["data"]["window_size"],
    )
    batch = _mock_batch()
    out = policy.compute_loss(batch)
    assert torch.isfinite(out["loss"])
    out["loss"].backward()
    print("mock forward/backward OK, loss=", float(out["loss"]))


def test_predict_action_shape() -> None:
    cfg = yaml.safe_load(open(os.path.join(_POLICY_ROOT, "configs", "config.yaml")))
    fm = dict(cfg["models"]["fm"])
    fm["image_pretrained"] = False

    policy = build_flow_policy(
        {"models": {"fm": fm}},
        action_dim=20,
        state_dim=20,
        cond_steps=cfg["data"]["window_size"],
    )
    policy.eval()
    batch = _mock_batch(batch_size=1)
    pred = policy.predict_action(batch["obs"], num_inference_steps=4)
    assert pred["action_normalized"].shape == (1, fm["n_action_steps"], 20)
    assert pred["action_pred_normalized"].shape == (1, fm["action_horizon"], 20)
    print("predict_action shapes OK")


def test_backbone_feat_forward_backward() -> None:
    cfg = yaml.safe_load(open(os.path.join(_POLICY_ROOT, "configs", "config.yaml")))
    fm = dict(cfg["models"]["fm"])
    fm["image_pretrained"] = False

    policy = build_flow_policy(
        {"models": {"fm": fm}},
        action_dim=20,
        state_dim=20,
        cond_steps=cfg["data"]["window_size"],
    )
    batch = _mock_batch(batch_size=2)
    del batch["obs"]["image"]
    batch["obs"]["image_backbone_feat"] = torch.randn(2, 1, 3, 384)
    out = policy.compute_loss(batch)
    assert torch.isfinite(out["loss"])
    out["loss"].backward()
    print("backbone_feat forward/backward OK, loss=", float(out["loss"]))


if __name__ == "__main__":
    test_mock_forward_backward()
    test_predict_action_shape()
    test_backbone_feat_forward_backward()
    print("[test_forward] all passed")
