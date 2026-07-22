import argparse
import gc

import torch

from infer.preprocess import build_policy_memory_input
from tools.latency_benchmark_utils import (
    add_common_arguments,
    benchmark_callable,
    create_result_dir,
    cuda_memory,
    finalize_memory_snapshots,
    load_benchmark_context,
    runtime_metadata,
    save_results,
    tensor_description,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    add_common_arguments(parser)
    parser.add_argument("--history-length", type=int)
    parser.add_argument("--state-history-length", type=int)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    runtime, load_seconds, memory_snapshots = load_benchmark_context(args)
    policy = runtime.policy
    device = runtime.device
    if not policy.memory_enabled or policy.memory_encoder is None:
        raise RuntimeError("benchmark policy does not enable Memory")
    if policy.memory_method != "fusion" or not hasattr(
        policy.memory_encoder, "encode_visual_views"
    ):
        raise RuntimeError(
            f"detailed per-view benchmark requires current fusion Memory, got "
            f"{policy.memory_method!r}"
        )

    b = args.batch_size
    t = int(runtime.memory_visual_offsets.numel())
    v = runtime.n_image_views
    c = int(
        policy.condition_encoder.image_encoder.encoder.head[0].normalized_shape[0]
    )
    state_t = int(policy.memory_history_frames)
    state_dim = int(policy.state_dim)
    if args.history_length is not None and args.history_length != t:
        raise ValueError(f"--history-length={args.history_length}, checkpoint expects {t}")
    if args.state_history_length is not None and args.state_history_length != state_t:
        raise ValueError(
            f"--state-history-length={args.state_history_length}, checkpoint expects {state_t}"
        )

    feature_window = torch.randn(b, t, v, c, device=device)
    memory_state = torch.randn(b, state_t, state_dim, device=device)
    offsets = runtime.memory_visual_offsets
    image_encoder = policy.condition_encoder.image_encoder
    memory_encoder = policy.memory_encoder

    temporal_outputs = []
    hook = memory_encoder.visual_encoder.encoder.register_forward_hook(
        lambda _module, _inputs, output: temporal_outputs.append(output.detach())
    )
    with torch.inference_mode():
        details = build_policy_memory_input(
            feature_window,
            policy,
            memory_state,
            offsets,
        )
    hook.remove()
    if len(temporal_outputs) != 1:
        raise AssertionError("expected exactly one shared Temporal Transformer call")
    temporal_output = temporal_outputs[0]

    projected = details["projected_features"]
    view_summary = details["view_summary"]
    restored = details["restored_view_summary"]
    view_concat = details["view_concat"]
    visual_global = details["visual_global"]
    state_global = details["state_global"]
    memory_global = details["memory_global"]
    memory_tokens = details["memory_tokens"]
    expected_bv = b * v
    assert details["view_batch_input"].shape == (expected_bv, t, c)
    assert projected.shape[:2] == (expected_bv, t)
    assert temporal_output.shape[:2] == (expected_bv, t)
    assert view_summary.shape == (expected_bv, visual_global.shape[-1])
    assert restored.shape[:2] == (b, v)
    assert view_concat.shape == (b, v * restored.shape[-1])
    assert visual_global.shape == (b, restored.shape[-1])
    assert state_global.shape[:1] == (b,)
    assert memory_global.shape == (b, policy.cond_dim)
    assert memory_tokens.shape == (b, 1, policy.cond_dim)
    fusion_input_dim = int(memory_encoder.view_fusion_proj[0].normalized_shape[0])
    if fusion_input_dim != view_concat.shape[-1]:
        raise AssertionError(
            f"view_fusion_proj expects {fusion_input_dim}, got {view_concat.shape[-1]}"
        )

    summaries = {}
    rows = []

    def run(name, function):
        summary, samples = benchmark_callable(
            name,
            function,
            device=device,
            warmup_iterations=args.warmup_iterations,
            iterations=args.iterations,
            memory_snapshots=memory_snapshots,
        )
        summaries[name] = summary
        rows.extend(samples)

    run(
        "dino_projection_ms",
        lambda: image_encoder.project_view_histories_from_backbone_feat(feature_window),
    )
    run(
        "visual_temporal_encoder_ms",
        lambda: memory_encoder.visual_encoder(projected, offsets),
    )
    run("view_fusion_ms", lambda: memory_encoder.view_fusion_proj(view_concat))
    run("state_encoder_ms", lambda: memory_encoder.state_encoder(memory_state))
    fusion_input = torch.cat([visual_global, state_global], dim=-1)
    run("memory_fusion_ms", lambda: memory_encoder.fusion(fusion_input))
    memory_obs = {
        "memory_image_backbone_feat": feature_window,
        "memory_state": memory_state,
        "memory_visual_offsets": offsets,
    }
    run("memory_total_ms", lambda: policy._build_memory(memory_obs))

    shapes = {
        "input_feature_window": list(feature_window.shape),
        "view_as_batch_input": list(details["view_batch_input"].shape),
        "projected_visual_sequence": list(projected.shape),
        "per_view_temporal_output": list(temporal_output.shape),
        "per_view_summary": list(view_summary.shape),
        "restored_views": list(restored.shape),
        "view_concatenation": list(view_concat.shape),
        "visual_global": list(visual_global.shape),
        "state_global": list(state_global.shape),
        "memory_global": list(memory_global.shape),
        "memory_tokens": list(memory_tokens.shape),
    }
    temporal_pooling = (
        "cls" if hasattr(memory_encoder.visual_encoder, "cls_token") else "valid_aware_mean"
    )
    metadata = runtime_metadata(
        runtime,
        args,
        benchmark="memory_latency",
        load_seconds=load_seconds,
        input_shapes=shapes,
    )
    metadata.update(
        {
            "memory_method": policy.memory_method,
            "temporal_pooling": temporal_pooling,
            "view_processing": "view_as_batch",
            "theoretical_tensor_memory": {
                name: tensor_description(tensor)
                for name, tensor in {
                    "feature_window": feature_window,
                    "projected_visual_sequence": projected,
                    "temporal_output": temporal_output,
                    "view_summary": view_summary,
                    "view_concat": view_concat,
                    "visual_global": visual_global,
                    "state_global": state_global,
                    "memory_global": memory_global,
                    "memory_tokens": memory_tokens,
                }.items()
            },
        }
    )
    print(f"temporal_pooling = {temporal_pooling}")
    print("view_processing = view_as_batch")
    for name, shape in shapes.items():
        print(f"{name}: {tuple(shape)}")

    result_dir = create_result_dir(args, "memory")
    config = runtime.policy_cfg
    finalize_memory_snapshots(memory_snapshots, device)
    del runtime, policy, details, temporal_outputs, temporal_output
    del feature_window, memory_state, projected, view_summary, restored, view_concat
    del visual_global, state_global, memory_global, memory_tokens, memory_obs
    del image_encoder, memory_encoder, offsets, fusion_input
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


if __name__ == "__main__":
    main()
