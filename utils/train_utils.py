from __future__ import annotations

import os
import random
import re
from copy import deepcopy
from typing import Any, Mapping

import numpy as np
import torch
import yaml

_PLACEHOLDER = re.compile(r"\$\{([^}]+)\}")
_MISSING = object()


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def sync_fm_action_horizon_from_data(
    fm_cfg: Mapping[str, Any],
    data_cfg: Mapping[str, Any],
) -> dict[str, Any]:
    """Single source of truth: ``data.action_horizon`` (and optional ``data.n_action_steps``).

    Injects into a copy of ``models.fm`` so Dataset and Policy stay aligned.
    Legacy resolved configs that only set ``models.fm.action_horizon`` still work.
    """
    resolved = dict(fm_cfg)
    if "action_horizon" in data_cfg and data_cfg["action_horizon"] is not None:
        horizon = int(data_cfg["action_horizon"])
    elif "action_horizon" in resolved and resolved["action_horizon"] is not None:
        horizon = int(resolved["action_horizon"])
    else:
        horizon = 32
    if horizon < 1:
        raise ValueError(f"action_horizon must be >= 1, got {horizon}")

    if "n_action_steps" in data_cfg and data_cfg["n_action_steps"] is not None:
        n_steps = int(data_cfg["n_action_steps"])
    elif "action_horizon" not in data_cfg and resolved.get("n_action_steps") is not None:
        # Legacy resolved_config: horizon lived only under models.fm
        n_steps = int(resolved["n_action_steps"])
    else:
        n_steps = horizon

    if n_steps < 1:
        raise ValueError(f"n_action_steps must be >= 1, got {n_steps}")
    if n_steps > horizon:
        raise ValueError(
            f"n_action_steps ({n_steps}) cannot exceed action_horizon ({horizon})"
        )

    resolved["action_horizon"] = horizon
    resolved["n_action_steps"] = n_steps
    return resolved


def _get_by_path(cfg: Mapping[str, Any], path: str, default: Any = _MISSING) -> Any:
    cur: Any = cfg
    for part in path.split("."):
        if not isinstance(cur, Mapping) or part not in cur:
            if default is _MISSING:
                raise KeyError(path)
            return default
        cur = cur[part]
    return cur


def _set_by_path(cfg: dict[str, Any], path: str, value: Any) -> None:
    parts = path.split(".")
    cur = cfg
    for part in parts[:-1]:
        nxt = cur.get(part)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[part] = nxt
        cur = nxt
    cur[parts[-1]] = value


def cfg_get(cfg: Mapping[str, Any], path: str, default: Any = None) -> Any:
    return _get_by_path(cfg, path, default)


def _expand_cfg_strings_inplace(cfg: dict[str, Any]) -> None:
    def expand_str(value: str) -> str:
        def repl(match: re.Match[str]) -> str:
            key = match.group(1).strip()
            resolved = _get_by_path(cfg, key, _MISSING)
            if resolved is _MISSING:
                raise KeyError(f"Unknown config placeholder '${{{key}}}'")
            if isinstance(resolved, (dict, list)) or resolved is None:
                raise TypeError(
                    f"Placeholder '${{{key}}}' must resolve to scalar, got {type(resolved)}"
                )
            return str(resolved)

        return _PLACEHOLDER.sub(repl, value)

    def walk(node: Any) -> Any:
        if isinstance(node, str) and "${" in node:
            return expand_str(node)
        if isinstance(node, dict):
            for key in list(node.keys()):
                node[key] = walk(node[key])
            return node
        if isinstance(node, list):
            for index in range(len(node)):
                node[index] = walk(node[index])
            return node
        return node

    walk(cfg)


def load_config(path: str, overrides: Mapping[str, Any] | None = None) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if not isinstance(cfg, dict):
        raise TypeError(f"Config at {path} must be a YAML mapping.")
    cfg = deepcopy(cfg)
    if overrides:
        for path_key, value in overrides.items():
            if value is not None:
                _set_by_path(cfg, path_key, value)
    _expand_cfg_strings_inplace(cfg)
    return cfg


def log_hparams_to_tensorboard(writer, cfg: Mapping[str, Any], log_dir: str) -> None:
    if writer is None:
        return
    writer.add_text("meta/log_dir", log_dir, 0)
    for key, value in cfg.items():
        if isinstance(value, (dict, list)):
            text = yaml.safe_dump(value, sort_keys=False)
        else:
            text = str(value)
        writer.add_text(f"hparams/{key}", text, 0)


def detach_scalar_dict(metrics: Mapping[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, value in metrics.items():
        if torch.is_tensor(value):
            if value.numel() == 1:
                out[key] = float(value.detach().item())
            else:
                out[key] = float(value.detach().float().mean().item())
        else:
            out[key] = float(value)
    return out
