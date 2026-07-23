"""Shape test for the Memory global-condition dimension.

Memory disabled -> global_cond [B, cond_dim]           (256)
Memory enabled  -> concat([obs_cond, memory_global])    -> global_cond [B, 2*cond_dim] (512)

The concat order is fixed as [obs_cond, memory_global]; there is no 512->256
compression layer. UNet FiLM and DiT cond_proj must accept the enabled width.
"""

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch

from models.diffusion.conditional_unet1d import ConditionalUnet1D
from models.fm.action_dit import ActionDiT
from models.fm.flow_policy import FlowMatchingPolicy
from models.fm.encoders.dino_v2 import DINOV2_NUM_TOKENS

B, V, D = 2, 3, 384
COND = 256
ACT_DIM, STATE_DIM = 14, 14
WINDOW, HORIZON = 8, 64
TVIS, TSTATE = 128, 64


def _base_kwargs(velocity_model: str) -> dict:
    return dict(
        action_dim=ACT_DIM,
        state_dim=STATE_DIM,
        cond_steps=WINDOW,
        cond_dim=COND,
        use_tactile=False,
        action_horizon=HORIZON,
        n_action_steps=32,
        n_image_views=V,
        image_pretrained=False,
        image_feat_dim=COND,
        velocity_model=velocity_model,
    )


def _obs(memory: bool) -> dict:
    obs = {
        "image_backbone_feat": torch.randn(B, 1, V, DINOV2_NUM_TOKENS, D),
        "state": torch.randn(B, WINDOW, STATE_DIM),
    }
    if memory:
        obs.update(
            {
                "memory_image_backbone_feat": torch.randn(B, TVIS, V, D),
                "memory_state": torch.randn(B, TSTATE, STATE_DIM),
                "memory_visual_offsets": torch.arange(-1016, 1, 8),
                "memory_visual_valid": torch.ones(B, TVIS, dtype=torch.bool),
                "memory_state_valid": torch.ones(B, TSTATE, dtype=torch.bool),
            }
        )
    return obs


@torch.no_grad()
def check_disabled(velocity_model: str) -> None:
    pol = FlowMatchingPolicy(**_base_kwargs(velocity_model), memory_enabled=False).eval()
    assert pol.global_cond_dim == COND, pol.global_cond_dim
    global_cond, tokens = pol._build_condition(_obs(memory=False))
    assert global_cond.shape == (B, COND), global_cond.shape
    assert tokens is None
    print(f"[{velocity_model}] disabled: global_cond {tuple(global_cond.shape)} OK")


@torch.no_grad()
def check_enabled(velocity_model: str) -> None:
    pol = FlowMatchingPolicy(
        **_base_kwargs(velocity_model),
        memory_enabled=True,
        memory_method="fusion",
        memory_injection="concat_global_cond",
        memory_dim=COND,
        memory_history_frames=TSTATE,
        memory_recent_frame=4,
        memory_visual_history_length=TVIS,
        memory_visual_sample_stride=8,
        memory_visual_recent_frame=0,
        memory_state_mem_dim=64,
        memory_dropout=0.0,
    ).eval()
    assert pol.global_cond_dim == 2 * COND, pol.global_cond_dim
    assert not hasattr(pol, "memory_cond_fusion") or pol.memory_cond_fusion is None

    obs = _obs(memory=True)
    obs_cond = pol._build_obs_condition(obs)
    _, memory_global = pol._build_memory(obs)
    global_cond, tokens = pol._build_condition(obs)
    assert global_cond.shape == (B, 2 * COND), global_cond.shape
    assert tokens is None
    # Order is fixed: obs_cond first, memory_global second.
    assert torch.allclose(global_cond[:, :COND], obs_cond)
    assert torch.allclose(global_cond[:, COND:], memory_global)

    # The velocity model consumes the 512-wide condition end-to-end.
    sample = torch.randn(B, HORIZON, ACT_DIM)
    t = torch.full((B,), 0.5)
    velocity = pol._model_forward(sample, t, global_cond=global_cond, condition_tokens=tokens)
    assert velocity.shape == (B, HORIZON, ACT_DIM), velocity.shape

    if velocity_model == "unet":
        # FiLM projection sees dsed + global_cond_dim.
        first_block = pol.model.down_modules[0][0]
        in_features = first_block.cond_encoder[1].in_features
        dsed = pol.model.diffusion_step_encoder[-1].out_features
        assert in_features == dsed + 2 * COND, (in_features, dsed)
    else:
        assert pol.model.cond_proj[0].in_features == 2 * COND, pol.model.cond_proj[0].in_features
    print(f"[{velocity_model}] enabled: concat [obs_cond|memory_global] -> "
          f"global_cond {tuple(global_cond.shape)}, velocity {tuple(velocity.shape)} OK")


def main() -> None:
    for vm in ("unet", "dit"):
        check_disabled(vm)
        check_enabled(vm)
    print("PASS: memory global_cond dim (disabled 256 / enabled 512), order fixed, no compression")


if __name__ == "__main__":
    main()
