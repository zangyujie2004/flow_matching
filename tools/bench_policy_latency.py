import argparse
import gc

import numpy as np
import torch

from infer.runtime import random_smoke_obs
from infer.tensor import numpy_obs_to_torch
from tools.latency_benchmark_utils import (
    add_common_arguments,
    benchmark_callable,
    create_result_dir,
    cuda_memory,
    finalize_memory_snapshots,
    load_benchmark_context,
    runtime_metadata,
    save_results,
)


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    add_common_arguments(parser)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    runtime, load_seconds, memory_snapshots = load_benchmark_context(args)
    policy = runtime.policy
    device = runtime.device
    if not policy.memory_enabled:
        raise RuntimeError("benchmark policy does not enable Memory")

    obs_numpy, _state_raw = random_smoke_obs(runtime, seed=args.seed)
    if args.batch_size > 1:
        obs_numpy["image"] = np.repeat(obs_numpy["image"], args.batch_size, axis=0)
        obs_numpy["state"] = np.repeat(obs_numpy["state"][None], args.batch_size, axis=0)
        if "tactile" in obs_numpy:
            obs_numpy["tactile"] = np.repeat(
                obs_numpy["tactile"][None], args.batch_size, axis=0
            )
    obs = numpy_obs_to_torch(
        obs_numpy,
        device,
        use_tactile=runtime.use_tactile,
        normalizer=runtime.normalizer,
        window_size=runtime.window_size,
    )

    b = args.batch_size
    t = int(runtime.memory_visual_offsets.numel())
    v = runtime.n_image_views
    c = int(
        policy.condition_encoder.image_encoder.encoder.head[0].normalized_shape[0]
    )
    memory_obs = {
        "memory_image_backbone_feat": torch.randn(b, t, v, c, device=device),
        "memory_state": torch.randn(
            b, policy.memory_history_frames, policy.state_dim, device=device
        ),
        "memory_visual_offsets": runtime.memory_visual_offsets,
    }
    obs.update(memory_obs)
    cached_obs = {name: value for name, value in obs.items() if name != "image"}
    cached_obs["image_backbone_feat"] = torch.randn(
        b, 1, v, c, device=device
    )
    for name, tensor in {**obs, **cached_obs}.items():
        if torch.is_tensor(tensor) and tensor.device != device:
            raise AssertionError(f"{name} is on {tensor.device}, expected {device}")

    obs_cond = policy._build_obs_condition(obs)
    memory_tokens, memory_global = policy._build_memory(obs)
    if policy.memory_injection == "concat_global_cond":
        if policy.memory_cond_fusion is None:
            raise AssertionError("concat_global_cond requires memory_cond_fusion")
        global_cond = policy.memory_cond_fusion(
            torch.cat([obs_cond, memory_global], dim=-1)
        )
        condition_tokens = None
        fusion_function = lambda: policy.memory_cond_fusion(
            torch.cat([obs_cond, memory_global], dim=-1)
        )
    else:
        global_cond = obs_cond
        condition_tokens = memory_tokens
        fusion_function = lambda: (obs_cond, memory_tokens)

    sample = torch.randn(b, policy.action_horizon, policy.action_dim, device=device)
    timestep = torch.full((b,), 0.5, device=device)
    velocity = policy._model_forward(
        sample,
        timestep,
        global_cond=global_cond,
        condition_tokens=condition_tokens,
    )
    cached_velocity = _condition_and_velocity(
        policy, cached_obs, sample, timestep
    )
    prediction = policy.predict_action(
        obs,
        num_inference_steps=policy.num_inference_steps,
        solver=policy.solver,
    )
    if velocity.shape != sample.shape:
        raise AssertionError(f"velocity shape {velocity.shape} != sample {sample.shape}")
    if cached_velocity.shape != sample.shape:
        raise AssertionError(
            f"cached velocity shape {cached_velocity.shape} != sample {sample.shape}"
        )
    expected_action = (b, policy.action_horizon, policy.action_dim)
    if prediction["action_pred_normalized"].shape != expected_action:
        raise AssertionError(
            f"action output {prediction['action_pred_normalized'].shape} != {expected_action}"
        )

    summaries = {}
    rows = []

    def run(name, function, iterations=None):
        summary, samples = benchmark_callable(
            name,
            function,
            device=device,
            warmup_iterations=args.warmup_iterations,
            iterations=args.iterations if iterations is None else iterations,
            memory_snapshots=memory_snapshots,
        )
        summaries[name] = summary
        rows.extend(samples)

    run("condition_encoder_ms", lambda: policy._build_obs_condition(obs))
    run("memory_encoder_ms", lambda: policy._build_memory(obs))
    run("condition_fusion_ms", fusion_function)
    run(
        "single_velocity_forward_ms",
        lambda: policy._model_forward(
            sample,
            timestep,
            global_cond=global_cond,
            condition_tokens=condition_tokens,
        ),
    )
    run(
        "single_network_forward_total_ms",
        lambda: _condition_and_velocity(policy, obs, sample, timestep),
    )
    run(
        "one_forward_cached_current_DINO_ms",
        lambda: _condition_and_velocity(policy, cached_obs, sample, timestep),
    )
    full_iterations = max(10, args.iterations // 4)
    policy._build_condition = lambda _obs: (global_cond, condition_tokens)
    run(
        "solver_total_ms",
        lambda: policy.conditional_sample(
            obs,
            num_inference_steps=policy.num_inference_steps,
            solver=policy.solver,
        ),
        iterations=full_iterations,
    )
    delattr(policy, "_build_condition")
    run(
        "predict_action_total_ms",
        lambda: policy.predict_action(
            obs,
            num_inference_steps=policy.num_inference_steps,
            solver=policy.solver,
        ),
        iterations=full_iterations,
    )

    shapes = {
        "current_image": list(obs["image"].shape),
        "cached_current_image_backbone_feat": list(
            cached_obs["image_backbone_feat"].shape
        ),
        "current_state": list(obs["state"].shape),
        "memory_image_backbone_feat": list(memory_obs["memory_image_backbone_feat"].shape),
        "memory_state": list(memory_obs["memory_state"].shape),
        "memory_visual_offsets": list(memory_obs["memory_visual_offsets"].shape),
        "obs_cond": list(obs_cond.shape),
        "memory_tokens": list(memory_tokens.shape),
        "memory_global": list(memory_global.shape),
        "global_cond": list(global_cond.shape),
        "single_velocity_output": list(velocity.shape),
        "action_pred_normalized": list(prediction["action_pred_normalized"].shape),
        "action_normalized": list(prediction["action_normalized"].shape),
    }
    metadata = runtime_metadata(
        runtime,
        args,
        benchmark="policy_latency",
        load_seconds=load_seconds,
        input_shapes=shapes,
    )
    metadata.update(
        {
            "velocity_model": policy.velocity_model,
            "memory_method": policy.memory_method,
            "memory_injection": policy.memory_injection,
            "number_of_solver_steps": policy.num_inference_steps,
            "solver": policy.solver,
            "number_of_action_steps": policy.n_action_steps,
            "action_horizon": policy.action_horizon,
            "solver_total_excludes_condition_build": True,
            "one_forward_cached_current_DINO_scope": (
                "cached current DINO feature + observation/state condition + Memory "
                "+ condition fusion + one velocity-model forward; excludes DINO "
                "backbone and solver loop"
            ),
        }
    )
    print(f"velocity_model = {policy.velocity_model}")
    print(f"memory_method = {policy.memory_method}")
    print(f"memory_injection = {policy.memory_injection}")
    print(f"number_of_solver_steps = {policy.num_inference_steps}")
    for name, shape in shapes.items():
        print(f"{name}: {tuple(shape)}")

    result_dir = create_result_dir(args, "policy")
    config = runtime.policy_cfg
    finalize_memory_snapshots(memory_snapshots, device)
    del runtime, policy, obs, cached_obs, memory_obs, prediction, velocity
    del cached_velocity
    del obs_cond, memory_tokens, memory_global, global_cond, sample, timestep
    del fusion_function
    gc.collect()
    torch.cuda.empty_cache()
    memory_snapshots.append(cuda_memory("after_cleanup", device))
    save_results(
        result_dir,
        metadata=metadata,
        summary=summaries,
        rows=rows,
        config=config,
        memory_snapshots=memory_snapshots,
    )


def _condition_and_velocity(policy, obs, sample, timestep):
    global_cond, condition_tokens = policy._build_condition(obs)
    return policy._model_forward(
        sample,
        timestep,
        global_cond=global_cond,
        condition_tokens=condition_tokens,
    )


if __name__ == "__main__":
    main()
