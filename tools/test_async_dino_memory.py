"""CUDA smoke test for two/three-view Async DINO history -> fusion Memory."""

import argparse
import time

import torch

from infer.preprocess import build_policy_memory_input
from models.fm.flow_policy import FlowMatchingPolicy
from tools.async_dino_buffer import AsyncDinoBuffer


def wait_until_processed(buffer: AsyncDinoBuffer, target: int) -> None:
    deadline = time.perf_counter() + 60.0
    while buffer.get_stats()["processed_count"] < target:
        if time.perf_counter() >= deadline:
            raise TimeoutError(f"DINO did not finish sample {target} within 60 seconds")
        time.sleep(0.01)


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-views", type=int, choices=(2, 3), default=3)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This smoke test requires CUDA")

    device = torch.device("cuda")
    b, t, v, c = 1, 16, args.num_views, 384
    projected_dim = memory_dim = 256
    state_dim, state_history = 14, 64
    policy = FlowMatchingPolicy(
        action_dim=state_dim,
        state_dim=state_dim,
        cond_steps=16,
        cond_dim=256,
        use_tactile=False,
        action_horizon=64,
        n_action_steps=32,
        n_image_views=v,
        image_pretrained=True,
        image_feat_dim=projected_dim,
        velocity_model="unet",
        memory_enabled=True,
        memory_method="fusion",
        memory_injection="concat_global_cond",
        memory_dim=memory_dim,
        memory_history_frames=state_history,
        memory_recent_frame=4,
        memory_visual_layers=2,
        memory_visual_heads=4,
        memory_state_mem_dim=64,
        memory_dropout=0.0,
    ).to(device).eval()

    dino = policy.condition_encoder.image_encoder.encoder
    buffer = AsyncDinoBuffer(dino, device="cuda", store_local_features=False)
    buffer.start()
    for sample_id in range(t):
        images = [
            torch.randint(0, 256, (b, 3, 224, 224), dtype=torch.uint8)
            for _ in range(v)
        ]
        assert buffer.submit_frame(sample_id * 4, *images)
        wait_until_processed(buffer, sample_id + 1)

    feature_window = buffer.get_feature_window()
    assert feature_window is not None
    assert feature_window.shape == (b, t, v, c)
    assert feature_window.device.type == "cuda"
    assert not feature_window.requires_grad and feature_window.grad_fn is None

    memory_state = torch.randn(b, state_history, state_dim, device=device)
    offsets = torch.arange(-64, -3, 4, device=device)
    assert offsets.shape == (t,)
    result = build_policy_memory_input(
        feature_window,
        policy,
        memory_state,
        offsets,
    )

    assert result["after_permute"].shape == (b, v, t, c)
    assert result["view_batch_input"].shape == (b * v, t, c)
    assert result["projected_features"].shape == (b * v, t, projected_dim)
    assert result["transformer_input"].shape == (b * v, t, projected_dim)
    assert result["view_summary"].shape == (b * v, memory_dim)
    assert result["restored_view_summary"].shape == (b, v, memory_dim)
    assert result["view_concat"].shape == (b, v * memory_dim)
    assert result["visual_global"].shape == (b, memory_dim)
    assert result["state_global"].shape == (b, 64)
    assert result["memory_global"].shape == (b, memory_dim)

    # The standard Policy path must produce exactly the same Memory output.
    memory_obs = {
        "memory_image_backbone_feat": feature_window,
        "memory_state": memory_state,
        "memory_visual_offsets": offsets,
    }
    policy_tokens, policy_global = policy._build_memory(memory_obs)
    assert torch.allclose(result["memory_tokens"], policy_tokens)
    assert torch.allclose(result["memory_global"], policy_global)

    obs_cond = torch.randn(b, policy.cond_dim, device=device)
    final_global_cond = policy.memory_cond_fusion(
        torch.cat([obs_cond, policy_global], dim=-1)
    )
    assert final_global_cond.shape == (b, policy.cond_dim)

    # There is one shared projection head and one shared temporal Transformer.
    shared_head = policy.condition_encoder.image_encoder.encoder.head
    shared_temporal = policy.memory_encoder.visual_encoder
    assert sum(module is shared_head for module in policy.modules()) == 1
    assert sum(module is shared_temporal for module in policy.modules()) == 1
    expected_projected = shared_head(result["view_batch_input"])
    assert torch.allclose(result["projected_features"], expected_projected)
    summaries = result["restored_view_summary"]
    assert not torch.allclose(result["view_batch_input"][0], result["view_batch_input"][1])
    assert not torch.allclose(summaries[:, 0], summaries[:, 1])
    for value in result.values():
        if torch.is_tensor(value):
            assert value.device.type == "cuda"
            assert not value.requires_grad and value.grad_fn is None
    assert final_global_cond.device.type == "cuda"
    assert not final_global_cond.requires_grad and final_global_cond.grad_fn is None

    print("buffer output:", tuple(feature_window.shape))
    print("after permute:", tuple(result["after_permute"].shape))
    print("view-as-batch input:", tuple(result["view_batch_input"].shape))
    print("after shared DINO projection:", tuple(result["projected_features"].shape))
    print("Transformer input:", tuple(result["transformer_input"].shape))
    print("per-view CLS summaries:", tuple(result["view_summary"].shape))
    print("restored view summaries:", tuple(result["restored_view_summary"].shape))
    print("view concatenation:", tuple(result["view_concat"].shape))
    print("visual_global:", tuple(result["visual_global"].shape))
    print("state_global:", tuple(result["state_global"].shape))
    print("memory_global:", tuple(result["memory_global"].shape))
    print("final global_cond:", tuple(final_global_cond.shape))
    print("offsets:", offsets.tolist())
    print("shared projection head: yes")
    print("shared temporal Transformer: yes")
    buffer.stop()


if __name__ == "__main__":
    main()
