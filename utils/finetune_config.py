from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import yaml

from models.fm.partial_ft import PARTIAL_FT_PRESETS
from utils.train_utils import _expand_cfg_strings_inplace, cfg_get, load_config

_NORMALIZER_MODES = ("refit", "keep_base")

_DATA_CONSTRAINT_KEYS = (
    "action_type",
    "action_representation",
    "window_size",
    "stride",
    "n_image_steps",
    "action_horizon",
    "use_tactile",
    "camera_views",
    "image_size",
    "image_as_uint8",
)


def _deep_merge(base: dict[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in overlay.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, Mapping)
        ):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _normalize_views(views: Any) -> tuple[str, ...]:
    if views is None:
        return tuple()
    return tuple(str(v) for v in views)


def _resolve_path(path: str | Path, policy_root: Path) -> Path:
    path = Path(path).expanduser()
    if not path.is_absolute():
        path = policy_root / path
    return path.resolve()


def load_base_config(base_run_dir: str | Path, *, policy_root: Path) -> dict[str, Any]:
    base_run_dir = _resolve_path(base_run_dir, policy_root)
    config_path = base_run_dir / "resolved_config.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"Base resolved config not found: {config_path}")
    with open(config_path, encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if not isinstance(cfg, dict):
        raise TypeError(f"Base config must be a mapping: {config_path}")
    cfg = deepcopy(cfg)
    cfg.setdefault("finetune", {})["base_run_dir"] = str(base_run_dir)
    return cfg


def validate_finetune_constraints(base_cfg: Mapping[str, Any], merged_cfg: Mapping[str, Any]) -> None:
    base_data = base_cfg.get("data") or {}
    merged_data = merged_cfg.get("data") or {}
    mismatches: list[str] = []

    for key in _DATA_CONSTRAINT_KEYS:
        base_value = base_data.get(key)
        merged_value = merged_data.get(key)
        if key == "camera_views":
            if _normalize_views(base_value) != _normalize_views(merged_value):
                mismatches.append(
                    f"data.{key}: base={list(base_value)!r} finetune={list(merged_value)!r}"
                )
            continue
        if base_value != merged_value:
            mismatches.append(f"data.{key}: base={base_value!r} finetune={merged_value!r}")

    if mismatches:
        raise ValueError(
            "Finetune data constraints must match base resolved_config:\n"
            + "\n".join(f"  - {line}" for line in mismatches)
        )


def resolve_run_dir(cfg: Mapping[str, Any], *, policy_root: Path) -> Path:
    finetune_cfg = cfg.get("finetune") or {}
    base_run_dir = _resolve_path(finetune_cfg["base_run_dir"], policy_root)
    run_name = str(cfg_get(cfg, "output.run_name", "fine-tune"))
    return base_run_dir / run_name


def merge_finetune_config(
    base_cfg: Mapping[str, Any],
    finetune_overlay: Mapping[str, Any],
    *,
    policy_root: Path,
) -> dict[str, Any]:
    overlay = {k: v for k, v in finetune_overlay.items() if k != "finetune"}
    merged = _deep_merge(dict(base_cfg), overlay)

    finetune_cfg = dict(finetune_overlay.get("finetune") or {})
    finetune_cfg.setdefault("normalizer_mode", "keep_base")
    finetune_cfg.setdefault("reset_optimizer", True)
    finetune_cfg.setdefault("reset_epoch_counter", True)
    if bool(finetune_cfg.get("partial_ft", False)):
        finetune_cfg.setdefault("partial_ft_preset", "action_head")

    base_run_dir = _resolve_path(finetune_cfg["base_run_dir"], policy_root)
    finetune_cfg["base_run_dir"] = str(base_run_dir)
    if "base_checkpoint" in finetune_cfg:
        finetune_cfg["base_checkpoint"] = str(
            _resolve_path(finetune_cfg["base_checkpoint"], policy_root)
            if not Path(str(finetune_cfg["base_checkpoint"])).is_absolute()
            else Path(finetune_cfg["base_checkpoint"]).expanduser().resolve()
        )

    merged["finetune"] = finetune_cfg
    merged.setdefault("output", {})
    merged["output"]["root_dir"] = str(base_run_dir)
    merged["output"]["run_name"] = str(
        cfg_get(finetune_overlay, "output.run_name", cfg_get(merged, "output.run_name", "fine-tune"))
    )

    # finetune must not use train.resume_path semantics
    train_cfg = dict(merged.get("train") or {})
    train_cfg["resume_path"] = None
    merged["train"] = train_cfg

    _apply_finetune_data_defaults(merged, finetune_overlay)
    _apply_finetune_precompute_defaults(merged, finetune_overlay)
    _apply_finetune_normalizer_mode(merged)
    _validate_partial_ft_config(merged)
    validate_finetune_constraints(base_cfg, merged)
    return merged


def _validate_partial_ft_config(merged: dict[str, Any]) -> None:
    finetune_cfg = merged.get("finetune") or {}
    if not bool(finetune_cfg.get("partial_ft", False)):
        return
    preset = str(finetune_cfg.get("partial_ft_preset", "action_head"))
    if preset not in PARTIAL_FT_PRESETS:
        raise ValueError(
            f"finetune.partial_ft_preset must be one of {PARTIAL_FT_PRESETS}, got {preset!r}"
        )
    finetune_cfg["partial_ft_preset"] = preset
    merged["finetune"] = finetune_cfg


def _apply_finetune_normalizer_mode(merged: dict[str, Any]) -> None:
    finetune_cfg = merged.get("finetune") or {}
    mode = str(finetune_cfg.get("normalizer_mode", "keep_base"))
    if mode not in _NORMALIZER_MODES:
        raise ValueError(
            f"finetune.normalizer_mode must be one of {_NORMALIZER_MODES}, got {mode!r}"
        )
    finetune_cfg["normalizer_mode"] = mode
    merged["finetune"] = finetune_cfg

    data = dict(merged.get("data") or {})
    data["fit_normalizer"] = mode == "refit"
    merged["data"] = data


def _apply_finetune_precompute_defaults(
    merged: dict[str, Any],
    finetune_overlay: Mapping[str, Any],
) -> None:
    pre_overlay = finetune_overlay.get("precompute")
    if isinstance(pre_overlay, Mapping) and "overwrite" in pre_overlay:
        return
    pre = dict(merged.get("precompute") or {})
    pre["overwrite"] = False
    merged["precompute"] = pre


def _apply_finetune_data_defaults(
    merged: dict[str, Any],
    finetune_overlay: Mapping[str, Any],
) -> None:
    data_overlay = finetune_overlay.get("data") or {}
    if not isinstance(data_overlay, Mapping) or "root_dir" not in data_overlay:
        return

    data = dict(merged.get("data") or {})
    from tools.latent_cache import default_latent_cache_root_dir

    fm_cfg = (merged.get("models") or {}).get("fm") or {}
    data["latent_cache_root_dir"] = default_latent_cache_root_dir(
        data["root_dir"],
        fm_cfg=fm_cfg,
    )

    finetune_mode = str((merged.get("finetune") or {}).get("normalizer_mode", "keep_base"))
    if finetune_mode != "refit":
        merged["data"] = data
        return

    norm_overlay = data_overlay.get("norm") if isinstance(data_overlay.get("norm"), Mapping) else {}
    if "max_windows" not in norm_overlay:
        norm = dict(data.get("norm") or {})
        norm["max_windows"] = 5000
        data["norm"] = norm

    merged["data"] = data


def resolve_full_config(
    finetune_config_path: str | Path,
    *,
    policy_root: Path | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    policy_root = policy_root or Path(__file__).resolve().parents[1]
    finetune_config_path = Path(finetune_config_path)
    if not finetune_config_path.is_absolute():
        finetune_config_path = policy_root / finetune_config_path

    finetune_overlay = load_config(str(finetune_config_path), overrides=overrides)
    finetune_cfg = finetune_overlay.get("finetune") or {}
    if "base_run_dir" not in finetune_cfg:
        raise KeyError("finetune.base_run_dir is required")
    if "base_checkpoint" not in finetune_cfg:
        raise KeyError("finetune.base_checkpoint is required")

    base_cfg = load_base_config(finetune_cfg["base_run_dir"], policy_root=policy_root)
    merged = merge_finetune_config(base_cfg, finetune_overlay, policy_root=policy_root)
    _expand_cfg_strings_inplace(merged)

    checkpoint_path = Path(str(merged["finetune"]["base_checkpoint"]))
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Base checkpoint not found: {checkpoint_path}")

    data_root = Path(str(cfg_get(merged, "data.root_dir", "")))
    if not data_root.is_dir():
        raise FileNotFoundError(f"Dataset root_dir not found: {data_root}")

    return merged


def write_finetune_manifest(
    run_dir: Path,
    cfg: Mapping[str, Any],
    *,
    finetune_config_path: Path | None = None,
) -> Path:
    finetune_cfg = cfg.get("finetune") or {}
    manifest = {
        "base_run_dir": finetune_cfg.get("base_run_dir"),
        "base_checkpoint": finetune_cfg.get("base_checkpoint"),
        "dataset_root_dir": cfg_get(cfg, "data.root_dir"),
        "output_dir": str(run_dir.resolve()),
        "run_name": cfg_get(cfg, "output.run_name"),
        "normalizer_mode": finetune_cfg.get("normalizer_mode", "keep_base"),
        "partial_ft": bool(finetune_cfg.get("partial_ft", False)),
        "partial_ft_preset": finetune_cfg.get("partial_ft_preset"),
        "reset_optimizer": bool(finetune_cfg.get("reset_optimizer", True)),
        "reset_epoch_counter": bool(finetune_cfg.get("reset_epoch_counter", True)),
        "finetune_config": str(finetune_config_path.resolve()) if finetune_config_path else None,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    path = run_dir / "finetune_manifest.yaml"
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(manifest, handle, sort_keys=False)
    return path
