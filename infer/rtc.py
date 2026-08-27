from __future__ import annotations

import math
from collections.abc import Callable

import torch


RTC_SCHEDULES = frozenset({"zeros", "ones", "linear", "exp"})


def latency_steps(elapsed_s: float, hz: float) -> int:
    """Convert wall-clock latency to conservative control steps."""
    elapsed_s = float(elapsed_s)
    hz = float(hz)
    if elapsed_s < 0:
        raise ValueError("elapsed_s must be non-negative")
    if hz <= 0:
        raise ValueError("hz must be positive")
    return int(math.ceil(elapsed_s / (1.0 / hz)))


def prefix_weights(
    *,
    inference_delay: int,
    prefix_attention_horizon: int,
    action_horizon: int,
    schedule: str,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Official RTC weights: locked delay, then decay to the available prefix end."""
    total = int(action_horizon)
    if total <= 0:
        raise ValueError("action_horizon must be positive")
    end = min(max(int(prefix_attention_horizon), 0), total)
    start = min(max(int(inference_delay), 0), end)
    schedule = str(schedule).strip().lower()
    if schedule not in RTC_SCHEDULES:
        raise ValueError(f"unsupported RTC prefix_attention_schedule={schedule!r}")

    weights = torch.zeros(total, device=device, dtype=dtype)
    if schedule == "ones":
        weights[:end] = 1
        return weights
    if schedule == "zeros":
        weights[:start] = 1
        return weights

    weights[:start] = 1
    decay_steps = end - start
    if decay_steps <= 0:
        return weights
    linear = torch.linspace(
        1.0,
        0.0,
        decay_steps + 2,
        device=device,
        dtype=dtype,
    )[1:-1]
    if schedule == "exp":
        linear = linear * torch.expm1(linear) / (math.e - 1.0)
    weights[start:end] = linear
    return weights


def guided_velocity(
    *,
    x_t: torch.Tensor,
    time: float,
    denoise_fn: Callable[[torch.Tensor], torch.Tensor],
    prev_actions: torch.Tensor,
    weights: torch.Tensor,
    max_guidance_weight: float,
) -> torch.Tensor:
    """Apply RTC VJP guidance for a flow integrated from noise t=0 to action t=1."""
    if x_t.ndim != 3:
        raise ValueError(f"x_t must be [B,T,A], got {tuple(x_t.shape)}")
    if prev_actions.ndim != 3:
        raise ValueError(
            f"prev_actions must be [B,T,A], got {tuple(prev_actions.shape)}"
        )
    if weights.ndim != 1 or weights.shape[0] != x_t.shape[1]:
        raise ValueError(
            f"weights must be [{x_t.shape[1]}], got {tuple(weights.shape)}"
        )
    max_guidance_weight = float(max_guidance_weight)
    if max_guidance_weight <= 0:
        raise ValueError("max_guidance_weight must be positive")

    x_g = x_t.detach().requires_grad_(True)
    velocity = denoise_fn(x_g)
    remaining_time = max(1.0 - float(time), 1e-6)
    endpoint = x_g + remaining_time * velocity

    prefix_steps = min(prev_actions.shape[1], endpoint.shape[1])
    prefix_dim = min(prev_actions.shape[2], endpoint.shape[2])
    error = torch.zeros_like(endpoint)
    if prefix_steps > 0 and prefix_dim > 0:
        prefix_error = (
            prev_actions[:, :prefix_steps, :prefix_dim]
            - endpoint[:, :prefix_steps, :prefix_dim]
        )
        error[:, :prefix_steps, :prefix_dim] = (
            prefix_error * weights[:prefix_steps].view(1, prefix_steps, 1)
        )

    correction = torch.autograd.grad(
        endpoint,
        x_g,
        grad_outputs=error.detach(),
        retain_graph=False,
        create_graph=False,
    )[0]
    guidance_weight = _guidance_weight(
        time=float(time),
        max_guidance_weight=max_guidance_weight,
    )
    return velocity.detach() + guidance_weight * correction.detach()


def _guidance_weight(*, time: float, max_guidance_weight: float) -> float:
    clean = min(max(float(time), 1e-6), 1.0 - 1e-6)
    noise = 1.0 - clean
    inv_r2 = (clean * clean + noise * noise) / (noise * noise)
    coefficient = noise / clean
    return min(coefficient * inv_r2, float(max_guidance_weight))
