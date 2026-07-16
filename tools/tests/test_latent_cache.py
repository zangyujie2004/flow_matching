from __future__ import annotations

import os
import sys

import pytest
import zarr

_POLICY_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _POLICY_ROOT not in sys.path:
    sys.path.insert(0, _POLICY_ROOT)

from tools.latent_cache import (
    cache_identity_from_attrs,
    default_latent_cache_root_dir,
    expected_cache_identity,
    resolve_latent_cache_zarr_path,
    validate_latent_cache_identity,
    write_latent_cache_identity_attrs,
)


def test_default_latent_cache_root_dir() -> None:
    root = "/data/chahua_all"
    assert default_latent_cache_root_dir(root) == "/data/chahua_all/latent_cache/dinov2_s14"
    assert resolve_latent_cache_zarr_path(default_latent_cache_root_dir(root)).endswith(
        "policy_latent_cache.zarr"
    )


def test_validate_latent_cache_identity_ok() -> None:
    fm_cfg = {
        "image_encoder_name": "dinov2_small",
        "dino_model_name": "vit_small_patch14_dinov2.lvd142m",
    }
    cache_attrs = {
        "image_encoder_name": "dinov2",
        "image_model_name": "vit_small_patch14_dinov2.lvd142m",
    }
    validate_latent_cache_identity(cache_attrs, fm_cfg, cache_path="/tmp/cache.zarr")


def test_validate_latent_cache_identity_legacy_dino_model_name() -> None:
    fm_cfg = expected_cache_identity({"image_encoder_name": "dinov2"})
    cache_attrs = {"dino_model_name": fm_cfg["image_model_name"]}
    assert cache_identity_from_attrs(cache_attrs)["image_encoder_name"] == "dinov2"
    validate_latent_cache_identity(cache_attrs, fm_cfg, cache_path="/tmp/cache.zarr")


def test_validate_latent_cache_identity_mismatch() -> None:
    fm_cfg = expected_cache_identity({"image_encoder_name": "dinov2"})
    cache_attrs = {
        "image_encoder_name": "dinov2",
        "image_model_name": "wrong_model",
    }
    with pytest.raises(ValueError, match="image_model_name mismatch"):
        validate_latent_cache_identity(cache_attrs, fm_cfg, cache_path="/tmp/cache.zarr")


def test_write_latent_cache_identity_attrs(tmp_path) -> None:
    root = zarr.open_group(str(tmp_path / "cache.zarr"), mode="w")
    identity = write_latent_cache_identity_attrs(
        root,
        {"image_encoder_name": "dinov2", "dino_model_name": "vit_small_patch14_dinov2.lvd142m"},
    )
    assert identity["image_encoder_name"] == "dinov2"
    assert root.attrs["image_model_name"] == "vit_small_patch14_dinov2.lvd142m"
    assert root.attrs["dino_model_name"] == "vit_small_patch14_dinov2.lvd142m"
