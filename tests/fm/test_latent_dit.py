"""Tests for the LIP latent Flow-Matching DiT."""

from __future__ import annotations

import torch

from models.fm import LatentDiT, LatentFlowMatchingPolicy


def _contexts(batch_size: int = 2):
    return (
        torch.randn(batch_size, 12, 16),
        torch.randn(batch_size, 18, 16),
    )


def test_latent_dit_uses_time_only_adaln_and_parallel_contexts() -> None:
    model = LatentDiT(
        latent_dim=8,
        latent_horizon=4,
        visual_dim=16,
        embodiment_dim=16,
        diffusion_step_embed_dim=16,
        hidden_dim=32,
        depth=4,
        num_heads=4,
        cross_attn_layers=(2, 4),
    )
    visual, embodiment = _contexts()
    velocity = model(
        torch.randn(2, 4, 8),
        torch.rand(2),
        visual_tokens=visual,
        embodiment_tokens=embodiment,
    )
    assert velocity.shape == (2, 4, 8)
    assert not hasattr(model, "cond_proj")
    for index in (1, 3):
        block = model.blocks[index]
        assert block.visual_cross_attn is not block.embodiment_cross_attn
        assert not hasattr(block, "visual_gate")
        assert not hasattr(block, "embodiment_gate")
        assert torch.count_nonzero(block.visual_cross_attn.out_proj.weight) == 0
        assert torch.count_nonzero(block.embodiment_cross_attn.out_proj.weight) == 0


def test_latent_flow_loss_and_sampling() -> None:
    policy = LatentFlowMatchingPolicy(
        latent_dim=8,
        latent_horizon=4,
        visual_dim=16,
        embodiment_dim=16,
        diffusion_step_embed_dim=16,
        hidden_dim=32,
        depth=4,
        num_heads=4,
        cross_attn_layers=(2, 4),
        num_inference_steps=2,
    )
    visual, embodiment = _contexts()
    result = policy.compute_loss(
        torch.randn(2, 4, 8),
        visual_tokens=visual,
        embodiment_tokens=embodiment,
    )
    assert torch.isfinite(result["loss"])
    result["loss"].backward()
    assert policy.model.output_proj.weight.grad is not None

    sampled = policy.sample(
        batch_size=2,
        visual_tokens=visual,
        embodiment_tokens=embodiment,
        num_inference_steps=2,
        solver="heun",
    )
    assert sampled.shape == (2, 4, 8)


def test_fixed_horizon_is_enforced() -> None:
    model = LatentDiT(
        latent_dim=8,
        latent_horizon=4,
        visual_dim=16,
        embodiment_dim=16,
        hidden_dim=32,
        depth=2,
        num_heads=4,
        cross_attn_layers=(2,),
    )
    visual, embodiment = _contexts()
    try:
        model(
            torch.randn(2, 5, 8),
            torch.rand(2),
            visual_tokens=visual,
            embodiment_tokens=embodiment,
        )
    except ValueError as exc:
        assert "latent_horizon" in str(exc)
    else:
        raise AssertionError("variable Stage-2 horizon should be rejected")
