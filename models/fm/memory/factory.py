"""Factory for memory encoders (fusion | multi-scale)."""

from __future__ import annotations

from typing import Any

from .fusion import MemoryEncoder
from .multi_scale import FastQueryMemoryEncoder


def normalize_memory_method(method: str) -> str:
    return str(method).lower().replace("_", "-")


def build_memory_encoder(method: str, **kwargs: Any):
    method = normalize_memory_method(method)
    if method == "fusion":
        return MemoryEncoder(**kwargs)
    if method in {"multi-scale", "multiscale"}:
        return FastQueryMemoryEncoder(**kwargs)
    raise ValueError(f"unsupported memory method={method!r}; expected 'fusion' or 'multi-scale'")
