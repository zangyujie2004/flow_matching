from __future__ import annotations

from typing import Any, Callable, Dict

import torch


def dict_apply(x: Dict[str, Any], func: Callable[[Any], Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in x.items():
        if isinstance(value, dict):
            out[key] = dict_apply(value, func)
        else:
            out[key] = func(value)
    return out


def move_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    def _to(value: Any) -> Any:
        if torch.is_tensor(value):
            return value.to(device, non_blocking=True)
        return value

    return dict_apply(batch, _to)
