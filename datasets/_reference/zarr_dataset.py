"""
④ PolicyDataset (ZarrDataset): orchestrates Store + WindowIndex + Normalizer.

Architecture: policy/datasets/planning/architecture.md
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import torch
from torch.utils.data import Dataset

from .normalizer import DatasetNormalizer
from .store import ReplayBufferStore
from .window_index import WindowIndex


@dataclass
class DataConfig:
    """Mirrors policy/configs/config.yaml data section."""

    root_dir: str
    window_size: int = 8
    stride: int = 1
    n_image_steps: int = 1
    action_horizon: int = 32
    action_type: str = "eef"
    action_representation: str = "relative"
    preload_to_ram: bool = True
    use_tactile: bool = True
    image_size: int = 224
    image_as_uint8: bool = True
    use_camera_latent: bool = False
    latent_cache_root_dir: Optional[str] = None
    camera_key: str = "camera"
    tactile_key: str = "tactile"
    state_key: str = "state_30hz"
    action_key: str = "action_30hz"
    norm_output_range: tuple[float, float] = (-1.0, 1.0)


class ZarrDataset(Dataset):
    """
    Policy training dataset.

    Init:
        Store(RAM) -> WindowIndex(full) -> DatasetNormalizer.fit(full windows)

    __getitem__:
        slice -> action_repr transform -> normalize -> batch dict
    """

    def __init__(self, config: DataConfig | Dict[str, Any]) -> None:
        if isinstance(config, dict):
            norm = config.pop("norm", {}) or {}
            output_range = tuple(norm.get("output_range", [-1.0, 1.0]))
            config = DataConfig(
                **config,
                norm_output_range=(float(output_range[0]), float(output_range[1])),
            )
        self.config = config

        self.store: ReplayBufferStore | None = None
        self.windows: WindowIndex | None = None
        self.normalizer: DatasetNormalizer | None = None
        raise NotImplementedError("See planning/architecture.md §5")

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        raise NotImplementedError

    def get_episode(self, ep_idx: int, *, normalize: bool = False) -> Dict[str, Any]:
        """Full episode raw arrays; default no normalize."""
        raise NotImplementedError

    @property
    def normalizer_state(self) -> dict[str, Any]:
        """For checkpoint save."""
        raise NotImplementedError


if __name__ == "__main__":
    cfg = DataConfig(root_dir="/mnt/oss_data/arx/policy/peel/peel_0629")
    print(cfg)
