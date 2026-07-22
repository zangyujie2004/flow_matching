"""CUDA smoke test for three-view Async DINO Buffer -> Policy Memory."""

import sys
import time
from pathlib import Path

import numpy as np
import torch

from infer.preprocess import cuda_memory_snapshot, tensor_mib
from infer.runtime import FMInferenceRuntime


def wait_until_processed(buffer, target: int, timeout_s: float = 30.0) -> None:
    deadline = time.perf_counter() + timeout_s
    while buffer.get_stats()["processed_count"] < target:
        if time.perf_counter() >= deadline:
            raise TimeoutError(f"DINO did not finish sample {target} within {timeout_s}s")
        time.sleep(0.01)


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python -m tools.test_async_dino_memory RUN_DIR")
    if not torch.cuda.is_available():
        raise RuntimeError("This test requires CUDA because the Buffer must stay on GPU")

    runtime = FMInferenceRuntime(Path(sys.argv[1]), device="cuda", warmup=False)
    if not runtime.policy.memory_enabled:
        raise RuntimeError("RUN_DIR policy must have data.memory.enabled=true")
    if runtime.n_image_views != 3:
        raise RuntimeError(f"RUN_DIR policy must use 3 views, got {runtime.n_image_views}")

    image_size = int(runtime.policy_cfg["data"].get("image_size", 224))
    buffer = runtime.start_async_dino()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    cuda_before = cuda_memory_snapshot("before_dino", runtime.device)

    # Wait after each submission so latest-only behavior does not drop test samples.
    for sample_id in range(16):
        images = [
            torch.randint(0, 256, (1, 3, image_size, image_size), dtype=torch.uint8)
            for _ in range(3)
        ]
        accepted = buffer.submit_frame(
            sample_id * 4, *images, capture_time=time.perf_counter()
        )
        assert accepted
        wait_until_processed(buffer, sample_id + 1)

    latest = buffer.get_latest()["feature"]
    window = buffer.get_feature_window()
    assert window is not None
    assert latest.requires_grad is False and latest.grad_fn is None
    assert window.requires_grad is False and window.grad_fn is None
    assert latest.device.type == "cuda" and window.device.type == "cuda"
    assert latest.shape[:2] == (1, 3)
    assert window.shape[:3] == (1, 16, 3)
    cuda_buffer = cuda_memory_snapshot("buffer_full", runtime.device)

    state_shape = (runtime.policy.memory_history_frames, runtime.policy.state_dim)
    memory_state_raw = np.zeros(state_shape, dtype=np.float32)
    result = runtime.build_async_policy_memory(
        memory_state_raw,
        measure_cuda_memory=True,
    )
    assert result is not None
    memory_obs = runtime._get_async_memory_obs(memory_state_raw)
    policy_tokens, policy_global = runtime.policy._build_memory(memory_obs)
    assert torch.allclose(result["memory_tokens"], policy_tokens)
    assert torch.allclose(result["memory_global"], policy_global)

    b, t, v, c = window.shape
    c_prime = result["projected_features"].shape[-1]
    assert result["flattened"].shape == (b, 48, c)
    assert result["projected_features"].shape == (b, 16, 3, c_prime)
    assert result["view_concat"].shape == (b, 16, 3 * c_prime)
    assert result["transformer_input"].shape == (b, 48, c_prime)

    print("DINO output shape:", tuple(latest[:, 0].shape))
    print("single timestep shape:", tuple(latest.shape))
    print("16-frame buffer shape:", tuple(window.shape))
    print("DINO projection-head input shape:", tuple(result["head_input"].shape))
    print("DINO projection-head output shape:", tuple(result["projected_features"].shape))
    print("Transformer input shape:", tuple(result["transformer_input"].shape))
    print("memory output shape:", tuple(result["memory_tokens"].shape))
    projection = result["projection_output"]
    print("projection output shape:", None if projection is None else tuple(projection.shape))

    theoretical = {
        "single_view_single_frame_mib": tensor_mib(latest[:, 0]),
        "three_view_single_timestep_mib": tensor_mib(latest),
        "three_view_16_frame_buffer_mib": tensor_mib(window),
        "projection_head_output_mib": tensor_mib(result["projected_features"]),
        "transformer_input_mib": tensor_mib(result["transformer_input"]),
        "final_memory_mib": tensor_mib(result["memory_tokens"]),
    }
    print("theoretical tensor memory:", theoretical)
    print("CUDA allocated/reserved/peak memory:")
    for snapshot in (
        cuda_before,
        cuda_buffer,
        result["cuda_after_projection_head"],
        result["cuda_after_transformer"],
    ):
        print(snapshot)
    print("DINO timing:", buffer.get_stats())
    buffer.stop()


if __name__ == "__main__":
    main()
