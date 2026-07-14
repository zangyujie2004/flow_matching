from __future__ import annotations

from typing import Iterable

import torch.nn as nn

from models.fm import FlowMatchingPolicy

PARTIAL_FT_PRESETS = ("action_head", "full")


def _count_params(parameters: Iterable[nn.Parameter]) -> int:
    return sum(param.numel() for param in parameters)


def _set_requires_grad(module: nn.Module, requires_grad: bool) -> None:
    for param in module.parameters():
        param.requires_grad = requires_grad


def apply_partial_ft(policy: FlowMatchingPolicy, preset: str) -> list[nn.Parameter]:
    preset = str(preset)
    if preset not in PARTIAL_FT_PRESETS:
        raise ValueError(
            f"Unsupported partial_ft_preset={preset!r}. "
            f"Expected one of {PARTIAL_FT_PRESETS}"
        )

    if preset == "full":
        trainable = [param for param in policy.parameters() if param.requires_grad]
        frozen = _count_params(param for param in policy.parameters() if not param.requires_grad)
        print(f"[partial_ft] preset=full (no extra freezing, DINO may already be frozen)")
        print(f"[partial_ft] trainable params: {len(trainable):,} tensors, {_count_params(trainable):,} values")
        if frozen:
            print(f"[partial_ft] frozen params: {frozen:,} values")
        return trainable

    if preset == "action_head":
        _set_requires_grad(policy, False)
        _set_requires_grad(policy.model, True)

        trainable = [param for param in policy.parameters() if param.requires_grad]
        frozen = _count_params(param for param in policy.parameters() if not param.requires_grad)
        print("[partial_ft] preset=action_head")
        print("[partial_ft] frozen: condition_encoder + non-UNet modules")
        print("[partial_ft] trainable: model (ConditionalUnet1D)")
        print(f"[partial_ft] trainable params: {len(trainable):,} tensors, {_count_params(trainable):,} values")
        print(f"[partial_ft] frozen params: {frozen:,} values")
        return trainable

    raise RuntimeError(f"Unhandled partial_ft preset: {preset}")


def resolve_partial_ft_preset(finetune_cfg: dict) -> str | None:
    if not bool(finetune_cfg.get("partial_ft", False)):
        return None
    return str(finetune_cfg.get("partial_ft_preset", "action_head"))
