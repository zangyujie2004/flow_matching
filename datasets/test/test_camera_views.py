from __future__ import annotations

import numpy as np
import pytest

from datasets.zarr_dataset import (
    CAMERA_BUNDLE_ORDER,
    cache_view_indices,
    camera_channel_indices,
    camera_view_indices,
    parse_cache_camera_views,
    resolve_camera_views,
)


def test_resolve_camera_views_defaults_to_all_zarr_views():
    views = resolve_camera_views(None, n_zarr_views=3)
    assert views == CAMERA_BUNDLE_ORDER


def test_resolve_camera_views_selects_wrist_pair():
    views = resolve_camera_views(["left_wrist_0", "right_wrist_0"], n_zarr_views=3)
    assert views == ("left_wrist_0", "right_wrist_0")


def test_camera_channel_indices_for_wrist_pair():
    indices = camera_channel_indices(["left_wrist_0", "right_wrist_0"])
    assert indices == (3, 4, 5, 6, 7, 8)


def test_camera_view_indices():
    assert camera_view_indices(["left_wrist_0", "right_wrist_0"]) == (1, 2)


def test_resolve_camera_views_rejects_unknown_bundle():
    with pytest.raises(ValueError, match="unknown camera_views"):
        resolve_camera_views(["base_0", "unknown_cam"], n_zarr_views=3)


def test_resolve_camera_views_rejects_missing_bundle_in_zarr():
    with pytest.raises(ValueError, match="not available in zarr"):
        resolve_camera_views(["base_0", "left_wrist_0", "right_wrist_0"], n_zarr_views=2)


def test_parse_cache_camera_views():
    assert parse_cache_camera_views("base_0,left_wrist_0,right_wrist_0") == CAMERA_BUNDLE_ORDER


def test_cache_view_indices_from_full_cache():
    cache_views = CAMERA_BUNDLE_ORDER
    assert cache_view_indices(["left_wrist_0", "right_wrist_0"], cache_views) == (1, 2)
    assert cache_view_indices(CAMERA_BUNDLE_ORDER, cache_views) == (0, 1, 2)


def test_cache_view_indices_from_wrist_only_cache():
    cache_views = ("left_wrist_0", "right_wrist_0")
    assert cache_view_indices(cache_views, cache_views) == (0, 1)


def test_cache_view_indices_rejects_missing_view():
    with pytest.raises(ValueError, match="missing requested views"):
        cache_view_indices(["base_0"], ("left_wrist_0", "right_wrist_0"))


def test_slice_wrist_channels_from_zarr_camera():
    camera = np.arange(2 * 2 * 2 * 9, dtype=np.uint8).reshape(2, 2, 2, 9)
    indices = camera_channel_indices(("left_wrist_0", "right_wrist_0"))
    selected = camera[..., indices]
    assert selected.shape == (2, 2, 2, 6)
    np.testing.assert_array_equal(selected[..., 0:3], camera[..., 3:6])
    np.testing.assert_array_equal(selected[..., 3:6], camera[..., 6:9])
