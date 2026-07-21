from __future__ import annotations

import os
from typing import Any, Mapping

FRAME_CACHE_BASENAME = "frame_backbone.zarr"
LEGACY_CACHE_BASENAME = "policy_latent_cache.zarr"
LATENT_CACHE_BASENAME = FRAME_CACHE_BASENAME  # preferred write/read name
DEFAULT_CACHE_SUBDIR = os.path.join("latent_cache", "dinov2_s14")
DEFAULT_DINOV2_MODEL = "vit_small_patch14_dinov2.lvd142m"
FRAME_CACHE_VERSION = 2


def default_latent_cache_root_dir(data_root: str) -> str:
    return os.path.join(str(data_root), "latent_cache", "dinov2_s14")


def resolve_frame_backbone_zarr_path(cache_root_dir: str) -> str:
    """Canonical write path for frame-only cache (scheme A)."""
    return os.path.join(str(cache_root_dir), FRAME_CACHE_BASENAME)


def resolve_latent_cache_zarr_path(cache_root_dir: str) -> str:
    """Resolve existing cache for reading; prefer frame_backbone, else legacy."""
    root = str(cache_root_dir)
    frame_path = os.path.join(root, FRAME_CACHE_BASENAME)
    if os.path.isdir(frame_path):
        return frame_path
    legacy = os.path.join(root, LEGACY_CACHE_BASENAME)
    if os.path.isdir(legacy):
        return legacy
    return frame_path


def normalize_image_encoder_name(name: str | None) -> str:
    key = str(name or "").strip().lower()
    if key in {"dinov2_small", "dinov2-small", "dino_small", "dino", "dinov2"}:
        return "dinov2"
    if not key:
        return "dinov2"
    return key


def expected_cache_identity(fm_cfg: Mapping[str, Any]) -> dict[str, str]:
    return {
        "image_encoder_name": normalize_image_encoder_name(
            fm_cfg.get("image_encoder_name", "dinov2")
        ),
        "image_model_name": str(fm_cfg.get("dino_model_name", DEFAULT_DINOV2_MODEL)),
    }


def cache_identity_from_attrs(attrs: Mapping[str, Any]) -> dict[str, str]:
    model_name = attrs.get("image_model_name") or attrs.get("dino_model_name")
    encoder_name = attrs.get("image_encoder_name")
    if encoder_name is None and model_name:
        encoder_name = "dinov2"
    return {
        "image_encoder_name": normalize_image_encoder_name(str(encoder_name or "")),
        "image_model_name": str(model_name or ""),
    }


def validate_latent_cache_identity(
    cache_attrs: Mapping[str, Any],
    expected: Mapping[str, Any],
    *,
    cache_path: str,
) -> None:
    actual = cache_identity_from_attrs(cache_attrs)
    exp = expected_cache_identity(expected)

    if not actual["image_model_name"]:
        raise ValueError(
            f"Latent cache missing image model metadata at {cache_path}. "
            "Rebuild with ./scripts/precompute.sh."
        )

    if actual["image_encoder_name"] != exp["image_encoder_name"]:
        raise ValueError(
            "Latent cache image_encoder_name mismatch: "
            f"cache={actual['image_encoder_name']!r}, expected={exp['image_encoder_name']!r}. "
            f"Point data.latent_cache_root_dir to the correct cache directory ({cache_path})."
        )

    if actual["image_model_name"] != exp["image_model_name"]:
        raise ValueError(
            "Latent cache image_model_name mismatch: "
            f"cache={actual['image_model_name']!r}, expected={exp['image_model_name']!r}. "
            f"Rebuild cache for the current vision encoder ({cache_path})."
        )


def write_latent_cache_identity_attrs(root_group: Any, fm_cfg: Mapping[str, Any]) -> dict[str, str]:
    identity = expected_cache_identity(fm_cfg)
    root_group.attrs["image_encoder_name"] = identity["image_encoder_name"]
    root_group.attrs["image_model_name"] = identity["image_model_name"]
    # Backward compatibility for older readers.
    root_group.attrs["dino_model_name"] = identity["image_model_name"]
    return identity


def frame_cache_matches(
    cache_path: str,
    *,
    fm_cfg: Mapping[str, Any],
    source_zarr_path: str,
    image_size: int,
    camera_views: tuple[str, ...] | list[str],
    total_frames: int,
    color_order: str = "rgb",
) -> bool:
    """True if existing frame cache can be reused (scheme A skip)."""
    if not os.path.isdir(cache_path):
        return False
    try:
        import zarr

        root = zarr.open_group(cache_path, mode="r")
    except Exception:
        return False
    if "data" not in root or "frame_image_backbone_feat" not in root["data"]:
        return False
    attrs = dict(getattr(root, "attrs", {}) or {})
    try:
        version = int(attrs.get("cache_version", 0))
    except (TypeError, ValueError):
        return False
    if version < int(FRAME_CACHE_VERSION):
        return False
    try:
        validate_latent_cache_identity(attrs, fm_cfg, cache_path=cache_path)
    except ValueError:
        return False
    if str(attrs.get("source_zarr_path", "")) != str(source_zarr_path):
        return False
    if attrs.get("image_size") is None or int(attrs["image_size"]) != int(image_size):
        return False
    if attrs.get("color_order") is None or str(attrs["color_order"]).lower() != str(color_order).lower():
        return False
    cache_views = str(attrs.get("camera_views", "")).strip()
    if not cache_views:
        return False
    expected_views = ",".join(camera_views)
    if cache_views != expected_views:
        return False
    feat = root["data"]["frame_image_backbone_feat"]
    if int(feat.shape[0]) != int(total_frames):
        return False
    return True
