"""Measure DINO patch-local and per-image CLS global feature storage on CUDA."""

import gc
import sys
import time
from pathlib import Path

import torch

from infer.preprocess import cuda_memory_snapshot, tensor_mib
from infer.runtime import FMInferenceRuntime
from models.fm.encoders.dino_v2 import DinoV2SmallEncoder
from tools.async_dino_buffer import AsyncDinoBuffer


def wait_until_processed(buffer: AsyncDinoBuffer, target: int) -> None:
    deadline = time.perf_counter() + 60.0
    while buffer.get_stats()["processed_count"] < target:
        if time.perf_counter() >= deadline:
            raise TimeoutError(f"DINO did not finish sample {target} within 60 seconds")
        time.sleep(0.01)


def fill_buffer(
    buffer: AsyncDinoBuffer,
    image_size: int,
    num_views: int,
    sample_count: int = 16,
) -> None:
    start_count = buffer.get_stats()["processed_count"]
    buffer.start()
    for sample_id in range(sample_count):
        images = [
            torch.randint(0, 256, (1, 3, image_size, image_size), dtype=torch.uint8)
            for _ in range(num_views)
        ]
        assert buffer.submit_frame(
            (start_count + sample_id) * 4,
            *images,
            capture_time=time.perf_counter(),
        )
        wait_until_processed(buffer, start_count + sample_id + 1)


def worker_warmed_empty_snapshot(
    buffer: AsyncDinoBuffer,
    image_size: int,
    num_views: int,
    label: str,
) -> dict:
    """Warm the actual worker, then measure with no Buffer-owned feature tensors."""
    fill_buffer(buffer, image_size, num_views, sample_count=2)
    buffer.stop()
    buffer.clear()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    return cuda_memory_snapshot(label, buffer.device)


def describe_tensor(name: str, tensor: torch.Tensor) -> None:
    print(f"{name}:")
    print(f"  shape={tuple(tensor.shape)}")
    print(f"  dtype={tensor.dtype}")
    print(f"  device={tensor.device}")
    print(f"  numel={tensor.numel()}")
    print(f"  element_size={tensor.element_size()}")
    print(f"  memory_mib={tensor_mib(tensor):.6f}")


def print_timing(stats: dict) -> None:
    names = (
        "backbone_forward_ms",
        "patch_extract_ms",
        "global_extract_ms",
        "view_stack_ms",
        "buffer_append_ms",
        "sample_total_ms",
    )
    print("Timing, including 2 worker warmup samples (mean / p95 / max ms):")
    for name in names:
        values = stats[name]
        print(
            f"  {name}: {values['mean']:.4f} / "
            f"{values['p95']:.4f} / {values['max']:.4f}"
        )


def main() -> None:
    if len(sys.argv) > 2:
        raise SystemExit(
            "Usage: python -m tools.test_dino_global_local_memory [RUN_DIR]"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this GPU-memory smoke test")

    num_views = 3
    device = torch.device("cuda")
    if len(sys.argv) == 2:
        runtime = FMInferenceRuntime(Path(sys.argv[1]), device="cuda", warmup=False)
        model = runtime.policy.condition_encoder.image_encoder.encoder
        image_size = int(runtime.policy_cfg["data"].get("image_size", 224))
        run_dir_views = runtime.n_image_views
    else:
        model = DinoV2SmallEncoder(pretrained=True, freeze=True).to(device).eval()
        image_size = 224
        run_dir_views = None
    backbone = model.backbone
    print(f"test_num_views={num_views}, run_dir_num_views={run_dir_views}")

    patch_size = backbone.patch_embed.patch_size
    if isinstance(patch_size, int):
        patch_size = (patch_size, patch_size)
    patch_h, patch_w = (int(patch_size[0]), int(patch_size[1]))
    if image_size % patch_h != 0 or image_size % patch_w != 0:
        raise ValueError(f"image size {image_size} is not divisible by patch size {patch_size}")
    expected_patches = (image_size // patch_h) * (image_size // patch_w)
    prefix_count = int(getattr(backbone, "num_prefix_tokens", 0))
    register_count = int(getattr(backbone, "num_reg_tokens", 0))

    inspection_image = torch.randint(
        0, 256, (1, 3, image_size, image_size), dtype=torch.uint8, device="cuda"
    )
    with torch.inference_mode():
        tokens = model.forward_tokens(inspection_image)
        features = model.extract_local_global_features(inspection_image)
        cls_token = tokens[:, :1] if backbone.cls_token is not None else None
        cls_count = 1 if cls_token is not None else 0
        register_tokens = tokens[:, cls_count : cls_count + register_count]
        current_global = model.extract_backbone_feat(inspection_image).unsqueeze(1)

    print("DINO implementation: timm VisionTransformer.forward_features -> Tensor")
    print(f"patch_size={patch_size}, prefix_tokens={prefix_count}, registers={register_count}")
    print(f"existing backbone global source={backbone.global_pool!r} (CLS token)")
    print("CLS token shape:", None if cls_token is None else tuple(cls_token.shape))
    print("register token shape:", tuple(register_tokens.shape))
    print("existing CLS global shape:", tuple(current_global.shape))
    print("DINO CLS global shape:", tuple(features["global"].shape))
    print("average-pooled patch shape:", tuple(features["avg_global"].shape))
    if backbone.global_pool == "token":
        assert cls_token is not None
        assert torch.allclose(current_global, cls_token, atol=1e-5, rtol=1e-4)
    assert torch.allclose(features["global"], cls_token, atol=1e-5, rtol=1e-4)
    assert features["local"].shape[1] == expected_patches
    assert torch.allclose(
        features["avg_global"],
        features["local"].mean(dim=1, keepdim=True),
        atol=1e-5,
        rtol=1e-4,
    )

    # Warm-up is separate: these outputs are not put into either Buffer.
    with torch.inference_mode():
        for _ in range(10):
            for _view in range(num_views):
                model.extract_local_global_features(inspection_image)
    torch.cuda.synchronize()
    tokens = features = cls_token = register_tokens = current_global = None
    inspection_image = None
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    empty_snapshot = cuda_memory_snapshot("model_loaded_buffer_empty", device)

    global_buffer = AsyncDinoBuffer(model, device="cuda", store_local_features=False)
    global_empty_snapshot = worker_warmed_empty_snapshot(
        global_buffer,
        image_size,
        num_views,
        "worker_warmed_global_buffer_empty",
    )
    fill_buffer(global_buffer, image_size, num_views)
    global_buffer.stop()
    torch.cuda.synchronize()
    global_snapshot = cuda_memory_snapshot("global_buffer_16", device)
    global_window = global_buffer.get_global_feature_window()
    assert global_window is not None
    assert global_window.shape[:3] == (1, 16, num_views)
    describe_tensor("global_16_frame", global_window)
    global_mib = tensor_mib(global_window)
    global_window = None
    global_buffer.clear()
    torch.cuda.empty_cache()

    both_buffer = AsyncDinoBuffer(model, device="cuda", store_local_features=True)
    both_empty_snapshot = worker_warmed_empty_snapshot(
        both_buffer,
        image_size,
        num_views,
        "worker_warmed_local_global_buffer_empty",
    )
    fill_buffer(both_buffer, image_size, num_views)
    both_buffer.stop()
    torch.cuda.synchronize()
    both_snapshot = cuda_memory_snapshot("global_plus_local_buffer_16", device)
    latest = both_buffer.get_latest()
    local_window = both_buffer.get_local_feature_window()
    global_window = both_buffer.get_global_feature_window()

    b, t, v = 1, 16, num_views
    n, c = expected_patches, int(backbone.num_features)
    assert latest["local_feature"].shape == (b, v, n, c)
    assert latest["global_feature"].shape == (b, v, c)
    assert local_window.shape == (b, t, v, n, c)
    assert global_window.shape == (b, t, v, c)
    assert not local_window.requires_grad and local_window.grad_fn is None
    assert not global_window.requires_grad and global_window.grad_fn is None
    assert local_window.device.type == global_window.device.type == "cuda"

    describe_tensor("local_single_view_single_frame", latest["local_feature"][:, 0])
    describe_tensor("global_single_view_single_frame", latest["global_feature"][:, 0])
    describe_tensor("local_multi_view_timestep", latest["local_feature"])
    describe_tensor("global_multi_view_timestep", latest["global_feature"])
    describe_tensor("local_16_frame", local_window)
    describe_tensor("global_16_frame", global_window)
    local_mib = tensor_mib(local_window)
    print(f"local_plus_global_16_frame_memory_mib={local_mib + global_mib:.6f}")

    print("CUDA allocator snapshots:")
    for item in (
        empty_snapshot,
        global_empty_snapshot,
        global_snapshot,
        both_empty_snapshot,
        both_snapshot,
    ):
        print(item)
    global_empty_alloc = float(global_empty_snapshot["allocated_mib"])
    both_empty_alloc = float(both_empty_snapshot["allocated_mib"])
    global_alloc = float(global_snapshot["allocated_mib"])
    both_alloc = float(both_snapshot["allocated_mib"])
    global_delta = global_alloc - global_empty_alloc
    both_delta = both_alloc - both_empty_alloc
    print(f"global_buffer_allocated_delta_mib={global_delta:.6f}")
    print(f"local_plus_global_allocated_delta_mib={both_delta:.6f}")
    print(f"local_buffer_extra_allocated_mib={both_delta - global_delta:.6f}")
    print_timing(both_buffer.get_stats())

    latest = local_window = global_window = None
    both_buffer.clear()
    torch.cuda.empty_cache()
    cleared_snapshot = cuda_memory_snapshot("buffers_cleared", device)
    print(cleared_snapshot)
    print("DINO local = patch-level [B,T,V,N,C]")
    print("DINO global = per-image CLS feature [B,T,V,C]")
    print("This is a shape/memory smoke test, not a realtime drop-pressure test.")


if __name__ == "__main__":
    main()
