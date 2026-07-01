"""② WindowIndex: strict anchor windows."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

Window = Tuple[int, int, int]  # (anchor_t, ep_end, ep_idx)


class WindowIndex:
    """Enumerate valid (anchor_t, ep_end, ep_idx) over episodes."""

    def __init__(
        self,
        episode_starts: np.ndarray,
        episode_ends: np.ndarray,
        *,
        window_length: int,
        n_image_steps: int,
        action_horizon: int,
        stride: int = 1,
    ) -> None:
        raise NotImplementedError("See planning/architecture.md §3")

    @property
    def windows(self) -> List[Window]:
        raise NotImplementedError

    def state_range(self, idx: int) -> tuple[int, int]:
        raise NotImplementedError

    def image_range(self, idx: int) -> tuple[int, int]:
        raise NotImplementedError

    def action_range(self, idx: int) -> tuple[int, int]:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError
