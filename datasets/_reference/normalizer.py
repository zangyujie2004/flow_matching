"""③ DatasetNormalizer: fit on full windows at init; no preprocess normalizer.pth."""

from __future__ import annotations

from typing import Any

import torch

from .store import ReplayBufferStore
from .window_index import WindowIndex


class DatasetNormalizer:
    """Policy-owned normalizer; checkpoint via state_dict."""

    @classmethod
    def build(
        cls,
        store: ReplayBufferStore,
        windows: WindowIndex,
        *,
        action_type: str,
        action_representation: str,
        output_range: tuple[float, float] = (-1.0, 1.0),
    ) -> "DatasetNormalizer":
        raise NotImplementedError("See planning/architecture.md §4")

    def normalize_state(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def normalize_action(
        self, x: torch.Tensor, *, anchor_state: torch.Tensor | None = None
    ) -> torch.Tensor:
        raise NotImplementedError

    def normalize_tactile(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def unnormalize_action(
        self, x: torch.Tensor, *, anchor_state: torch.Tensor | None = None
    ) -> torch.Tensor:
        raise NotImplementedError

    def state_dict(self) -> dict[str, Any]:
        raise NotImplementedError

    @classmethod
    def load_state_dict(cls, state: dict[str, Any]) -> "DatasetNormalizer":
        raise NotImplementedError
