from __future__ import annotations

import numpy as np
import pytest

from datasets.image_augment import apply_photometric_augment
from datasets.zarr_dataset import resolve_camera_data_config


def test_photometric_augment_uint8_range() -> None:
    rng = np.random.default_rng(0)
    img = np.random.randint(0, 256, size=(2, 32, 32, 6), dtype=np.uint8)
    out = apply_photometric_augment(img, rng)
    assert out.shape == img.shape
    assert out.dtype == np.uint8
    assert out.min() >= 0
    assert out.max() <= 255


def test_photometric_augment_sync_views() -> None:
    img = np.full((1, 4, 4, 6), 100, dtype=np.uint8)
    out = apply_photometric_augment(img, np.random.default_rng(42))
    assert np.array_equal(out[0, 0, 0, :3], out[0, 0, 0, 3:6])


def test_photometric_augment_rejects_non_uint8() -> None:
    img = np.zeros((4, 4, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="uint8"):
        apply_photometric_augment(img, np.random.default_rng(0))


def test_camera_augmentation_disables_latent_cache() -> None:
    resolved = resolve_camera_data_config(
        {
            "camera_augmentation": True,
            "use_camera_latent": True,
            "root_dir": "/tmp/unused",
        }
    )
    assert resolved["camera_augmentation"] is True
    assert resolved["use_camera_latent"] is False


def test_resolve_camera_data_config_keeps_latent_when_aug_disabled() -> None:
    resolved = resolve_camera_data_config(
        {
            "camera_augmentation": False,
            "use_camera_latent": True,
            "root_dir": "/tmp/unused",
        }
    )
    assert resolved["use_camera_latent"] is True


def test_zarr_dataset_rejects_augment_with_latent() -> None:
    from datasets.zarr_dataset import ZarrDataset

    with pytest.raises(ValueError, match="incompatible"):
        ZarrDataset(
            root_dir="/tmp/unused",
            camera_augmentation=True,
            use_camera_latent=True,
        )
