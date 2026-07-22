import argparse
import gc

import torch

from tools.async_dino_buffer import AsyncDinoBuffer
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
    parser.add_argument("--image-height", type=int)
    parser.add_argument("--image-width", type=int)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    runtime, load_seconds, memory_snapshots = load_benchmark_context(args)
    device = runtime.device
    model = runtime.policy.condition_encoder.image_encoder.encoder.eval()
    views = runtime.n_image_views
    image_size = int(runtime.policy_cfg["data"].get("image_size", 224))
    height = args.image_height or image_size
    width = args.image_width or image_size
    images = [
        torch.randint(
            0,
            256,
            (args.batch_size, 3, height, width),
            dtype=torch.uint8,
            device=device,
        )
        for _ in range(views)
    ]
    dino_buffer = AsyncDinoBuffer(
        model,
        device=str(device),
        store_local_features=True,
    )

    with torch.inference_mode():
        prepared = [model._imagenet_normalize(image) for image in images]
        tokens = [model.backbone.forward_features(image) for image in prepared]
        local = [model.patch_tokens_from_output(value).detach() for value in tokens]
        cls_global = [model.cls_token_from_output(value).detach() for value in tokens]
        avg_global = [value.mean(dim=1, keepdim=True) for value in local]
        expected_global = torch.stack([value.squeeze(1) for value in cls_global], dim=1)
        expected_local = torch.stack(local, dim=1)
        actual_global, actual_local, _timing = dino_buffer._run_dino(images)

    if torch.allclose(actual_global, expected_global, atol=1e-5, rtol=1e-4):
        global_source = "cls_token"
    elif torch.allclose(
        actual_global,
        torch.stack([value.squeeze(1) for value in avg_global], dim=1),
        atol=1e-5,
        rtol=1e-4,
    ):
        global_source = "patch_average"
    else:
        raise AssertionError("cannot identify AsyncDinoBuffer global feature source")
    selected_global = cls_global if global_source == "cls_token" else avg_global
    assert actual_local is not None and torch.allclose(
        actual_local, expected_local, atol=1e-5, rtol=1e-4
    )
    assert actual_global.shape[:2] == (args.batch_size, views)
    assert actual_local.shape[:2] == (args.batch_size, views)

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

    run("preprocess_ms", lambda: [model._imagenet_normalize(image) for image in images])
    for view_index in range(views):
        run(
            f"view_{view_index + 1}_forward_ms",
            lambda index=view_index: model.backbone.forward_features(prepared[index]),
        )
    run(
        "all_views_forward_ms",
        lambda: [model.backbone.forward_features(image) for image in prepared],
    )
    run("patch_extract_ms", lambda: [model.patch_tokens_from_output(value) for value in tokens])
    run(
        "pooling_ms",
        (
            lambda: [model.cls_token_from_output(value) for value in tokens]
            if global_source == "cls_token"
            else lambda: [value.mean(dim=1, keepdim=True) for value in local]
        ),
    )
    run(
        "view_stack_ms",
        lambda: torch.stack([value.squeeze(1) for value in selected_global], dim=1),
    )
    run("single_view_dino_ms", lambda: dino_buffer._run_dino(images[:1])[:2])
    run("sample_total_ms", lambda: dino_buffer._run_dino(images)[:2])

    input_shapes = {
        "single_view_input": list(images[0].shape),
        "single_view_backbone_output": list(tokens[0].shape),
        "single_view_local": list(local[0].shape),
        "single_view_global": list(selected_global[0].shape),
        "multi_view_global": list(actual_global.shape),
        "multi_view_local": list(actual_local.shape),
    }
    metadata = runtime_metadata(
        runtime,
        args,
        benchmark="dino_latency",
        load_seconds=load_seconds,
        input_shapes=input_shapes,
    )
    metadata.update(
        {
            "global_feature_source": global_source,
            "pooling_metric_operation": (
                "CLS extraction; no average pooling" if global_source == "cls_token" else "patch mean"
            ),
            "dino_parameter_count": sum(value.numel() for value in model.parameters()),
            "tensor_descriptions": {
                "multi_view_global": tensor_description(actual_global),
                "multi_view_local": tensor_description(actual_local),
            },
        }
    )
    print(f"global_feature_source = {global_source}")
    for name, shape in input_shapes.items():
        print(f"{name}: {tuple(shape)}")

    result_dir = create_result_dir(args, "dino")
    config = runtime.policy_cfg
    finalize_memory_snapshots(memory_snapshots, device)
    del runtime, dino_buffer, model, prepared, tokens, local, cls_global, avg_global
    del selected_global
    del actual_global, actual_local, expected_global, expected_local, images
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
