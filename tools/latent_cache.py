from __future__ import annotations

import os
from typing import Any, Literal, Mapping

FRAME_CACHE_BASENAME = "frame_backbone.zarr"
FRAME_CACHE_BASE_REMOVE_HAND_BASENAME = "frame_backbone_base_remove_hand.zarr"
LEGACY_CACHE_BASENAME = "policy_latent_cache.zarr"
LATENT_CACHE_BASENAME = FRAME_CACHE_BASENAME  # preferred write/read name
DEFAULT_DINOV2_MODEL = "vit_small_patch14_dinov2.lvd142m"
DINOV2_BASE_MODEL = "vit_base_patch14_dinov2.lvd142m"
# v3 supports either CLS (T,V,D) or all tokens (T,V,257,D).
FRAME_CACHE_VERSION = 3
DINOV2_NUM_TOKENS = 257
TOKEN_MODE_CLS = "cls"
TOKEN_MODE_ALL = "all"
TokenMode = Literal["cls", "all"]
CACHE_SUBDIR_SMALL = "dinov2_s14"
CACHE_SUBDIR_BASE = "dinov2_b14"
DEFAULT_CACHE_SUBDIR = os.path.join("latent_cache", CACHE_SUBDIR_SMALL)
CAMERA_BASE_REMOVE_HAND_KEY = "camera_base_remove_hand"


def normalize_token_mode(value: Any, *, default: TokenMode = TOKEN_MODE_ALL) -> TokenMode:
    if value is None or str(value).strip() == "":
        return default
    key = str(value).strip().lower()
    if key in {"cls", "cls_only", "class", "1"}:
        return TOKEN_MODE_CLS
    if key in {"all", "full", "tokens", "257", "patch", "patches"}:
        return TOKEN_MODE_ALL
    raise ValueError(f"invalid token_mode={value!r}; expected 'cls' or 'all'")


def token_mode_num_tokens(token_mode: TokenMode | str) -> int:
    mode = normalize_token_mode(token_mode)
    return 1 if mode == TOKEN_MODE_CLS else int(DINOV2_NUM_TOKENS)


def infer_token_mode_from_attrs_and_shape(
    attrs: Mapping[str, Any],
    feat_shape: tuple[int, ...],
) -> TokenMode:
    """Resolve cache token layout from metadata, with legacy shape fallback."""
    raw = attrs.get("token_mode")
    if raw is not None and str(raw).strip():
        return normalize_token_mode(raw)
    num_tokens = attrs.get("image_num_tokens")
    if num_tokens is not None:
        try:
            count = int(num_tokens)
        except (TypeError, ValueError):
            count = -1
        if count == 1:
            return TOKEN_MODE_CLS
        if count == DINOV2_NUM_TOKENS:
            return TOKEN_MODE_ALL
    if len(feat_shape) == 3:
        return TOKEN_MODE_CLS
    if len(feat_shape) == 4 and int(feat_shape[2]) == 1:
        return TOKEN_MODE_CLS
    if len(feat_shape) == 4 and int(feat_shape[2]) == DINOV2_NUM_TOKENS:
        return TOKEN_MODE_ALL
    raise ValueError(f"cannot infer token_mode from attrs/shape={feat_shape}")


def normalize_image_encoder_name(name: str | None) -> str:
    key = str(name or "").strip().lower()
    if key in {"dinov2_base", "dinov2-base", "dino_base", "dinobase"}:
        return "dinov2_base"
    if key in {"dinov2_small", "dinov2-small", "dino_small", "dino", "dinov2"}:
        return "dinov2"
    if not key:
        return "dinov2"
    return key


def cache_subdir_for_vision(
    *,
    image_encoder_name: str | None = None,
    dino_model_name: str | None = None,
) -> str:
    """latent_cache/{dinov2_s14|dinov2_b14} depending on small vs base."""
    enc = normalize_image_encoder_name(image_encoder_name)
    model = str(dino_model_name or "").lower()
    if enc == "dinov2_base" or "base" in model:
        return CACHE_SUBDIR_BASE
    return CACHE_SUBDIR_SMALL


def default_latent_cache_root_dir(
    data_root: str,
    *,
    image_encoder_name: str | None = None,
    dino_model_name: str | None = None,
    fm_cfg: Mapping[str, Any] | None = None,
) -> str:
    if fm_cfg is not None:
        image_encoder_name = image_encoder_name or fm_cfg.get("image_encoder_name")
        dino_model_name = dino_model_name or fm_cfg.get("dino_model_name")
    subdir = cache_subdir_for_vision(
        image_encoder_name=image_encoder_name,
        dino_model_name=dino_model_name,
    )
    return os.path.join(str(data_root), "latent_cache", subdir)


def resolve_latent_cache_root_dir(cfg: Mapping[str, Any]) -> str:
    """Resolve data.latent_cache_root_dir; auto-fill from small/base when missing/placeholder."""
    data = cfg.get("data") or {}
    fm = (cfg.get("models") or {}).get("fm") or {}
    root = data.get("root_dir")
    if root is None:
        raise KeyError("data.root_dir is required to resolve latent_cache_root_dir")

    current = data.get("latent_cache_root_dir")
    auto = default_latent_cache_root_dir(str(root), fm_cfg=fm)
    if current is None or str(current).strip() == "":
        return auto

    text = str(current)
    # Placeholders: {auto}, {dinov2_s14}, {dinov2_b14}, or literal braces typos
    if "{auto}" in text or "{dinov2_s14}" in text or "{dinov2_b14}" in text or "{dinov2" in text:
        return auto
    return text


def apply_resolved_latent_cache_root_dir(cfg: dict[str, Any]) -> dict[str, Any]:
    """In-place set data.latent_cache_root_dir from vision encoder (small/base)."""
    data = dict(cfg.get("data") or {})
    data["latent_cache_root_dir"] = resolve_latent_cache_root_dir(cfg)
    cfg["data"] = data
    return cfg


def resolve_frame_backbone_zarr_path(cache_root_dir: str) -> str:
    """Canonical write path for frame-only cache (scheme A)."""
    return os.path.join(str(cache_root_dir), FRAME_CACHE_BASENAME)


def resolve_frame_backbone_base_remove_hand_zarr_path(cache_root_dir: str) -> str:
    """Canonical compact cache path for the remove-hand base camera."""
    return os.path.join(str(cache_root_dir), FRAME_CACHE_BASE_REMOVE_HAND_BASENAME)


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


def expected_cache_identity(fm_cfg: Mapping[str, Any]) -> dict[str, str]:
    enc = normalize_image_encoder_name(fm_cfg.get("image_encoder_name", "dinov2"))
    model = str(fm_cfg.get("dino_model_name") or "")
    if not model:
        model = DINOV2_BASE_MODEL if enc == "dinov2_base" else DEFAULT_DINOV2_MODEL
    return {
        "image_encoder_name": enc,
        "image_model_name": model,
    }


def cache_identity_from_attrs(attrs: Mapping[str, Any]) -> dict[str, str]:
    model_name = attrs.get("image_model_name") or attrs.get("dino_model_name")
    encoder_name = attrs.get("image_encoder_name")
    if encoder_name is None and model_name:
        encoder_name = "dinov2_base" if "base" in str(model_name).lower() else "dinov2"
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


def write_token_mode_attrs(root_group: Any, token_mode: TokenMode | str) -> TokenMode:
    mode = normalize_token_mode(token_mode)
    root_group.attrs["token_mode"] = mode
    root_group.attrs["image_num_tokens"] = token_mode_num_tokens(mode)
    return mode


def _feat_shape_matches_token_mode(feat_shape: tuple[int, ...], token_mode: TokenMode) -> bool:
    if token_mode == TOKEN_MODE_CLS:
        return len(feat_shape) == 3 or (
            len(feat_shape) == 4 and int(feat_shape[2]) == 1
        )
    return len(feat_shape) == 4 and int(feat_shape[2]) == DINOV2_NUM_TOKENS


def frame_cache_matches(
    cache_path: str,
    *,
    fm_cfg: Mapping[str, Any],
    source_zarr_path: str,
    image_size: int,
    camera_views: tuple[str, ...] | list[str],
    total_frames: int,
    token_mode: TokenMode | str = TOKEN_MODE_ALL,
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
    wanted = normalize_token_mode(token_mode)
    try:
        actual = infer_token_mode_from_attrs_and_shape(
            attrs, tuple(int(x) for x in feat.shape)
        )
    except ValueError:
        return False
    return actual == wanted and _feat_shape_matches_token_mode(
        tuple(int(x) for x in feat.shape), wanted
    )


def remove_hand_frame_cache_matches(
    cache_path: str,
    *,
    fm_cfg: Mapping[str, Any],
    source_zarr_path: str,
    image_size: int,
    total_frames: int,
    token_mode: TokenMode | str = TOKEN_MODE_ALL,
    color_order: str = "rgb",
) -> bool:
    """Return whether a compact base_0 remove-hand cache can be reused."""
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
    if version < FRAME_CACHE_VERSION:
        return False
    try:
        validate_latent_cache_identity(attrs, fm_cfg, cache_path=cache_path)
    except ValueError:
        return False
    if str(attrs.get("source_zarr_path", "")) != str(source_zarr_path):
        return False
    if attrs.get("image_size") is None or int(attrs["image_size"]) != int(image_size):
        return False
    if str(attrs.get("color_order", "")).lower() != str(color_order).lower():
        return False
    if str(attrs.get("camera_views", "")).strip() != "base_0":
        return False
    if attrs.get("compact") not in (True, 1, "1", "true", "True"):
        return False
    if str(attrs.get("ties_to", "")) != CAMERA_BASE_REMOVE_HAND_KEY:
        return False
    if attrs.get("base_remove_hand") not in (
        True,
        1,
        "1",
        "true",
        "True",
        "present",
    ):
        return False
    feat = root["data"]["frame_image_backbone_feat"]
    if int(feat.shape[0]) != int(total_frames) or int(feat.shape[1]) != 1:
        return False
    wanted = normalize_token_mode(token_mode)
    try:
        actual = infer_token_mode_from_attrs_and_shape(
            attrs, tuple(int(x) for x in feat.shape)
        )
    except ValueError:
        return False
    return actual == wanted and _feat_shape_matches_token_mode(
        tuple(int(x) for x in feat.shape), wanted
    )
