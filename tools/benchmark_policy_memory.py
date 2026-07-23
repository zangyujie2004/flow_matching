import argparse
import time

import numpy as np
import torch

from models.fm.flow_policy import FlowMatchingPolicy


def measure(name, function, warmup, repeats):
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()

    wall_times = []
    gpu_times = []
    for _ in range(repeats):
        begin = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        wall_start = time.perf_counter()
        begin.record()
        function()
        end.record()
        torch.cuda.synchronize()
        wall_times.append((time.perf_counter() - wall_start) * 1000)
        gpu_times.append(begin.elapsed_time(end))

    print(
        f"{name}: "
        f"wall mean/p95/max={np.mean(wall_times):.3f}/"
        f"{np.percentile(wall_times, 95):.3f}/{np.max(wall_times):.3f} ms, "
        f"GPU mean/p95/max={np.mean(gpu_times):.3f}/"
        f"{np.percentile(gpu_times, 95):.3f}/{np.max(gpu_times):.3f} ms"
    )


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-views", type=int, choices=(2, 3), default=3)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--inference-steps", type=int, default=32)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    device = torch.device("cuda")
    b, t, v, c = 1, 128, args.num_views, 384
    state_dim, state_history, cond_steps = 14, 64, 16
    policy = FlowMatchingPolicy(
        action_dim=state_dim,
        state_dim=state_dim,
        cond_steps=cond_steps,
        cond_dim=256,
        use_tactile=False,
        action_horizon=64,
        n_action_steps=32,
        n_image_views=v,
        image_pretrained=True,
        image_feat_dim=256,
        velocity_model="unet",
        memory_enabled=True,
        memory_method="fusion",
        memory_injection="concat_global_cond",
        memory_dim=256,
        memory_history_frames=state_history,
        memory_recent_frame=4,
        memory_visual_history_length=t,
        memory_visual_sample_stride=8,
        memory_visual_recent_frame=0,
        memory_visual_layers=2,
        memory_visual_heads=4,
        memory_state_mem_dim=64,
        memory_dropout=0.0,
        num_inference_steps=args.inference_steps,
        solver="euler",
    ).to(device).eval()

    memory_obs = {
        "memory_image_backbone_feat": torch.randn(b, t, v, c, device=device),
        "memory_state": torch.randn(b, state_history, state_dim, device=device),
        "memory_visual_offsets": torch.arange(-1016, 1, 8, device=device),
    }
    common_obs = {
        "state": torch.randn(b, cond_steps, state_dim, device=device),
        **memory_obs,
    }
    cached_obs = {
        "image_backbone_feat": torch.randn(b, 1, v, c, device=device),
        **common_obs,
    }
    raw_obs = {
        "image": torch.randint(
            0, 256, (b, 1, v, 3, 224, 224), dtype=torch.uint8, device=device
        ),
        **common_obs,
    }
    sample = torch.randn(b, policy.action_horizon, state_dim, device=device)
    timestep = torch.full((b,), 0.5, device=device)

    def memory_only():
        return policy._build_memory(memory_obs)

    def one_forward(obs):
        global_cond, condition_tokens = policy._build_condition(obs)
        return policy._model_forward(
            sample,
            timestep,
            global_cond=global_cond,
            condition_tokens=condition_tokens,
        )

    print("Current architecture: fusion Memory + UNet + concat_global_cond")
    print(f"B={b}, T={t}, V={v}, C={c}, random policy weights")
    measure("memory_only", memory_only, args.warmup, args.repeats)
    measure(
        "one_forward_cached_current_DINO",
        lambda: one_forward(cached_obs),
        args.warmup,
        args.repeats,
    )
    measure(
        "one_forward_raw_current_images",
        lambda: one_forward(raw_obs),
        args.warmup,
        args.repeats,
    )
    measure(
        f"predict_action_{args.inference_steps}_steps_cached_current_DINO",
        lambda: policy.predict_action(
            cached_obs,
            num_inference_steps=args.inference_steps,
            solver="euler",
        ),
        max(2, args.warmup // 2),
        max(10, args.repeats // 2),
    )


if __name__ == "__main__":
    main()
