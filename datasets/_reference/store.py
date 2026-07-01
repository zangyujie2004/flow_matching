"""① ReplayBufferStore: zarr -> RAM, 30Hz only."""

from __future__ import annotations

from typing import Any

import numpy as np

from .robot_layout import robot_slice


class ReplayBufferStore:
    """Load replay_buffer.zarr into RAM; slice state/action by action_type."""

    def __init__(
        self,
        root_dir: str,
        *,
        action_type: str,
        camera_key: str = "camera",
        tactile_key: str = "tactile",
        state_key: str = "state_30hz",
        action_key: str = "action_30hz",
        preload_to_ram: bool = True,
        use_camera_latent: bool = False,
        latent_cache_root_dir: str | None = None,
    ) -> None:
        raise NotImplementedError("See planning/architecture.md §2")

    @property
    def episode_ends(self) -> np.ndarray:
        raise NotImplementedError

    @property
    def episode_starts(self) -> np.ndarray:
        raise NotImplementedError

    def get_camera(self, t0: int, t1: int) -> np.ndarray:
        raise NotImplementedError

    def get_tactile(self, t0: int, t1: int) -> np.ndarray:
        raise NotImplementedError

    def get_state(self, t0: int, t1: int) -> np.ndarray:
        raise NotImplementedError

    def get_action(self, t0: int, t1: int) -> np.ndarray:
        raise NotImplementedError

    def get_episode(self, ep_idx: int) -> dict[str, Any]:
        raise NotImplementedError

    def get_camera_latent(self, t0: int, t1: int) -> np.ndarray | None:
        raise NotImplementedError
