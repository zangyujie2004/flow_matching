"""CPU checks for 64-token repeat-first Async DINO history windows."""

import time

import torch
from torch import nn

from tools.async_dino_buffer import AsyncDinoBuffer


class IdentityGlobalDino(nn.Module):
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return image


def wait_for_count(buffer: AsyncDinoBuffer, count: int) -> None:
    deadline = time.perf_counter() + 10.0
    while buffer.get_stats()["processed_count"] < count:
        if time.perf_counter() >= deadline:
            raise TimeoutError(f"worker did not process sample {count}")
        time.sleep(0.001)


def timestep(sample_id: int, num_views: int) -> torch.Tensor:
    views = [
        torch.full((1, 384), sample_id * 10 + view, dtype=torch.float32)
        for view in range(num_views)
    ]
    return torch.stack(views, dim=1)  # (B,V,C)


def submit_sample(
    buffer: AsyncDinoBuffer,
    sample_id: int,
    num_views: int,
) -> None:
    images = [timestep(sample_id, num_views)[:, view] for view in range(num_views)]
    assert buffer.submit_frame(sample_id * 8, *images)
    wait_for_count(buffer, sample_id)


def check_num_views(num_views: int) -> None:
    H = 128  # visual history length
    buffer = AsyncDinoBuffer(
        IdentityGlobalDino(),
        device="cpu",
        sample_interval_frames=8,
        history_length=H,
    )
    buffer.start()

    submit_sample(buffer, 1, num_views)
    first = timestep(1, num_views)
    window = buffer.get_feature_window()
    assert window.shape == (1, H, num_views, 384)
    assert torch.equal(window, first.unsqueeze(1).expand(-1, H, -1, -1))
    assert len(buffer.get_buffer()) == 1

    submit_sample(buffer, 2, num_views)
    submit_sample(buffer, 3, num_views)
    window = buffer.get_feature_window()
    assert torch.equal(window[:, : H - 3], first.unsqueeze(1).expand(-1, H - 3, -1, -1))
    assert torch.equal(window[:, -3], timestep(1, num_views))
    assert torch.equal(window[:, -2], timestep(2, num_views))
    assert torch.equal(window[:, -1], timestep(3, num_views))

    for sample_id in range(4, H + 1):
        submit_sample(buffer, sample_id, num_views)
    window = buffer.get_feature_window()
    expected = torch.stack(
        [timestep(sample_id, num_views) for sample_id in range(1, H + 1)],
        dim=1,
    )
    assert torch.equal(window, expected)

    submit_sample(buffer, H + 1, num_views)
    window = buffer.get_feature_window()
    expected = torch.stack(
        [timestep(sample_id, num_views) for sample_id in range(2, H + 2)],
        dim=1,
    )
    assert torch.equal(window, expected)
    assert len(buffer.get_buffer()) == H
    assert window.ndim == 4
    assert buffer.get_local_feature_window() is None

    view_batch = window.permute(0, 2, 1, 3).reshape(
        num_views, H, 384
    )
    assert view_batch.shape == (num_views, H, 384)
    buffer.stop()


def main() -> None:
    offsets = torch.arange(-1016, 1, 8)  # 128 visual instants, stride 8
    assert offsets.shape == (128,)
    for num_views in (2, 3):
        check_num_views(num_views)
    print("PASS: repeat-first [B,128,V,384] history for V=2 and V=3")


if __name__ == "__main__":
    main()
