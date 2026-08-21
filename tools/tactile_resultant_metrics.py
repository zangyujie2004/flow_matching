"""Physical-space tactile L1 and tangential resultant metrics."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from tools.tactile_feat import TACTILE_BUNDLE_ORDER


AXES_PER_SENSOR = 3


def reshape_tactile_sensors(
    deformation: np.ndarray,
    *,
    num_sensors: int = len(TACTILE_BUNDLE_ORDER),
) -> np.ndarray:
    """Convert (..., H, W, S*3) into (..., H, W, S, 3)."""
    value = np.asarray(deformation, dtype=np.float64)
    expected_channels = int(num_sensors) * AXES_PER_SENSOR
    if value.ndim < 4 or value.shape[-1] != expected_channels:
        raise ValueError(
            "expected tactile deformation (...,H,W,"
            f"{expected_channels}), got {value.shape}"
        )
    return value.reshape(*value.shape[:-1], int(num_sensors), AXES_PER_SENSOR)


def compute_contact_mask(
    target: np.ndarray,
    *,
    threshold: float = 0.005,
) -> np.ndarray:
    """Return the GT contact mask (..., H, W, S) from absolute dz."""
    if float(threshold) < 0:
        raise ValueError("contact threshold must be non-negative")
    sensors = reshape_tactile_sensors(target)
    return np.abs(sensors[..., 2]) > float(threshold)


def compute_xy_resultant(
    deformation: np.ndarray,
    *,
    contact_mask: np.ndarray,
) -> np.ndarray:
    """Sum signed dx/dy over a shared contact mask, yielding (..., S, 2)."""
    sensors = reshape_tactile_sensors(deformation)
    mask = np.asarray(contact_mask, dtype=bool)
    if mask.shape != sensors.shape[:-1]:
        raise ValueError(
            f"contact mask shape {mask.shape} != tactile shape {sensors.shape[:-1]}"
        )
    return np.where(mask[..., None], sensors[..., :2], 0.0).sum(axis=(-4, -3))


def compute_resultant_cosine(
    prediction: np.ndarray,
    target: np.ndarray,
    *,
    contact_dz_threshold: float = 0.005,
    eps: float = 1e-12,
) -> dict[str, np.ndarray]:
    """Compute per-frame, per-sensor XY resultant and direction similarity.

    Both resultants use the GT contact mask. Frames with negligible GT
    resultant are invalid. A valid GT with negligible predicted resultant gets
    cosine 0 so a model cannot improve its score by predicting no force.
    """
    pred = np.asarray(prediction, dtype=np.float64)
    gt = np.asarray(target, dtype=np.float64)
    if pred.shape != gt.shape:
        raise ValueError(f"prediction shape {pred.shape} != target shape {gt.shape}")
    if float(eps) <= 0:
        raise ValueError("eps must be positive")

    gt_contact = compute_contact_mask(gt, threshold=contact_dz_threshold)
    pred_contact = compute_contact_mask(pred, threshold=contact_dz_threshold)
    gt_resultant = compute_xy_resultant(gt, contact_mask=gt_contact)
    pred_resultant = compute_xy_resultant(pred, contact_mask=gt_contact)
    gt_norm = np.linalg.norm(gt_resultant, axis=-1)
    pred_norm = np.linalg.norm(pred_resultant, axis=-1)
    valid_gt = gt_norm > float(eps)
    valid_pred = pred_norm > float(eps)
    both_valid = valid_gt & valid_pred
    cosine = np.zeros_like(gt_norm, dtype=np.float64)
    dot = np.sum(pred_resultant * gt_resultant, axis=-1)
    cosine[both_valid] = dot[both_valid] / (
        pred_norm[both_valid] * gt_norm[both_valid]
    )
    cosine = np.clip(cosine, -1.0, 1.0)
    return {
        "gt_contact": gt_contact,
        "pred_contact": pred_contact,
        "gt_contact_count": gt_contact.sum(axis=(-3, -2), dtype=np.int64),
        "pred_contact_count": pred_contact.sum(axis=(-3, -2), dtype=np.int64),
        "gt_resultant": gt_resultant,
        "pred_resultant": pred_resultant,
        "gt_resultant_norm": gt_norm,
        "pred_resultant_norm": pred_norm,
        "cosine": cosine,
        "valid_gt": valid_gt,
        "pred_zero_on_valid_gt": valid_gt & ~valid_pred,
    }


class PhysicalTactileMetricAccumulator:
    """Streaming physical L1/contact L1/resultant metrics with horizon bins."""

    def __init__(
        self,
        *,
        num_horizons: int,
        num_sensors: int = len(TACTILE_BUNDLE_ORDER),
        contact_dz_threshold: float = 0.005,
        eps: float = 1e-12,
    ) -> None:
        self.num_horizons = int(num_horizons)
        self.num_sensors = int(num_sensors)
        self.contact_dz_threshold = float(contact_dz_threshold)
        self.eps = float(eps)
        if self.num_horizons <= 0:
            raise ValueError("num_horizons must be positive")
        if self.num_sensors <= 0:
            raise ValueError("num_sensors must be positive")
        if self.contact_dz_threshold < 0:
            raise ValueError("contact_dz_threshold must be non-negative")
        if self.eps <= 0:
            raise ValueError("eps must be positive")

        self.abs_sum = 0.0
        self.count = 0
        self.contact_abs_sum = 0.0
        self.contact_count = 0
        self.sensor_cosine_sum = np.zeros(self.num_sensors, dtype=np.float64)
        self.sensor_cosine_count = np.zeros(self.num_sensors, dtype=np.int64)
        self.sensor_weighted_cosine_sum = np.zeros(
            self.num_sensors, dtype=np.float64
        )
        self.sensor_weight_sum = np.zeros(self.num_sensors, dtype=np.float64)
        self.sensor_pred_zero_count = np.zeros(self.num_sensors, dtype=np.int64)

        self.horizon_abs_sum = np.zeros(self.num_horizons, dtype=np.float64)
        self.horizon_count = np.zeros(self.num_horizons, dtype=np.int64)
        self.horizon_contact_abs_sum = np.zeros(
            self.num_horizons, dtype=np.float64
        )
        self.horizon_contact_count = np.zeros(self.num_horizons, dtype=np.int64)
        self.horizon_cosine_sum = np.zeros(self.num_horizons, dtype=np.float64)
        self.horizon_cosine_count = np.zeros(self.num_horizons, dtype=np.int64)
        self.horizon_weighted_cosine_sum = np.zeros(
            self.num_horizons, dtype=np.float64
        )
        self.horizon_weight_sum = np.zeros(self.num_horizons, dtype=np.float64)
        self.horizon_pred_zero_count = np.zeros(
            self.num_horizons, dtype=np.int64
        )

    def update(self, prediction: np.ndarray, target: np.ndarray) -> None:
        pred = np.asarray(prediction, dtype=np.float64)
        gt = np.asarray(target, dtype=np.float64)
        if pred.shape != gt.shape or pred.ndim != 5:
            raise ValueError(
                "expected matching tactile arrays (B,T,H,W,S*3), got "
                f"prediction={pred.shape}, target={gt.shape}"
            )
        if pred.shape[1] != self.num_horizons:
            raise ValueError(
                f"horizon mismatch: {pred.shape[1]} != {self.num_horizons}"
            )
        if pred.shape[-1] != self.num_sensors * AXES_PER_SENSOR:
            raise ValueError(
                f"channel mismatch: {pred.shape[-1]} != "
                f"{self.num_sensors * AXES_PER_SENSOR}"
            )

        abs_diff = np.abs(pred - gt)
        self.abs_sum += float(abs_diff.sum())
        self.count += int(abs_diff.size)
        self.horizon_abs_sum += abs_diff.sum(axis=(0, 2, 3, 4))
        self.horizon_count += np.prod(
            (pred.shape[0], pred.shape[2], pred.shape[3], pred.shape[4]),
            dtype=np.int64,
        )

        resultants = compute_resultant_cosine(
            pred,
            gt,
            contact_dz_threshold=self.contact_dz_threshold,
            eps=self.eps,
        )
        contact = resultants["gt_contact"]
        sensor_abs_diff = reshape_tactile_sensors(abs_diff)
        contact_abs = np.where(contact[..., None], sensor_abs_diff, 0.0)
        self.contact_abs_sum += float(contact_abs.sum())
        self.contact_count += int(contact.sum()) * AXES_PER_SENSOR
        self.horizon_contact_abs_sum += contact_abs.sum(axis=(0, 2, 3, 4, 5))
        self.horizon_contact_count += (
            contact.sum(axis=(0, 2, 3, 4), dtype=np.int64) * AXES_PER_SENSOR
        )

        cosine = resultants["cosine"]
        valid = resultants["valid_gt"]
        weights = resultants["gt_resultant_norm"]
        pred_zero = resultants["pred_zero_on_valid_gt"]
        self.sensor_cosine_sum += np.where(valid, cosine, 0.0).sum(axis=(0, 1))
        self.sensor_cosine_count += valid.sum(axis=(0, 1), dtype=np.int64)
        self.sensor_weighted_cosine_sum += np.where(
            valid, cosine * weights, 0.0
        ).sum(axis=(0, 1))
        self.sensor_weight_sum += np.where(valid, weights, 0.0).sum(axis=(0, 1))
        self.sensor_pred_zero_count += pred_zero.sum(axis=(0, 1), dtype=np.int64)

        self.horizon_cosine_sum += np.where(valid, cosine, 0.0).sum(axis=(0, 2))
        self.horizon_cosine_count += valid.sum(axis=(0, 2), dtype=np.int64)
        self.horizon_weighted_cosine_sum += np.where(
            valid, cosine * weights, 0.0
        ).sum(axis=(0, 2))
        self.horizon_weight_sum += np.where(valid, weights, 0.0).sum(axis=(0, 2))
        self.horizon_pred_zero_count += pred_zero.sum(axis=(0, 2), dtype=np.int64)

    def merge(self, other: "PhysicalTactileMetricAccumulator") -> None:
        if (
            self.num_horizons != other.num_horizons
            or self.num_sensors != other.num_sensors
            or self.contact_dz_threshold != other.contact_dz_threshold
            or self.eps != other.eps
        ):
            raise ValueError("cannot merge incompatible tactile metric accumulators")
        self.load_reduction_vector(
            self.reduction_vector() + other.reduction_vector()
        )

    def reduction_vector(self) -> np.ndarray:
        return np.concatenate(
            [
                np.asarray(
                    [
                        self.abs_sum,
                        float(self.count),
                        self.contact_abs_sum,
                        float(self.contact_count),
                    ],
                    dtype=np.float64,
                ),
                self.sensor_cosine_sum,
                self.sensor_cosine_count.astype(np.float64),
                self.sensor_weighted_cosine_sum,
                self.sensor_weight_sum,
                self.sensor_pred_zero_count.astype(np.float64),
                self.horizon_abs_sum,
                self.horizon_count.astype(np.float64),
                self.horizon_contact_abs_sum,
                self.horizon_contact_count.astype(np.float64),
                self.horizon_cosine_sum,
                self.horizon_cosine_count.astype(np.float64),
                self.horizon_weighted_cosine_sum,
                self.horizon_weight_sum,
                self.horizon_pred_zero_count.astype(np.float64),
            ]
        )

    def load_reduction_vector(self, vector: np.ndarray) -> None:
        values = np.asarray(vector, dtype=np.float64)
        expected = 4 + 5 * self.num_sensors + 9 * self.num_horizons
        if values.shape != (expected,):
            raise ValueError(
                f"invalid metric reduction vector {values.shape}; expected ({expected},)"
            )
        offset = 0
        self.abs_sum = float(values[offset])
        self.count = int(round(float(values[offset + 1])))
        self.contact_abs_sum = float(values[offset + 2])
        self.contact_count = int(round(float(values[offset + 3])))
        offset += 4

        sensor_fields = (
            ("sensor_cosine_sum", np.float64),
            ("sensor_cosine_count", np.int64),
            ("sensor_weighted_cosine_sum", np.float64),
            ("sensor_weight_sum", np.float64),
            ("sensor_pred_zero_count", np.int64),
        )
        for name, dtype in sensor_fields:
            chunk = values[offset : offset + self.num_sensors]
            setattr(
                self,
                name,
                np.rint(chunk).astype(dtype) if dtype == np.int64 else chunk.copy(),
            )
            offset += self.num_sensors

        horizon_fields = (
            ("horizon_abs_sum", np.float64),
            ("horizon_count", np.int64),
            ("horizon_contact_abs_sum", np.float64),
            ("horizon_contact_count", np.int64),
            ("horizon_cosine_sum", np.float64),
            ("horizon_cosine_count", np.int64),
            ("horizon_weighted_cosine_sum", np.float64),
            ("horizon_weight_sum", np.float64),
            ("horizon_pred_zero_count", np.int64),
        )
        for name, dtype in horizon_fields:
            chunk = values[offset : offset + self.num_horizons]
            setattr(
                self,
                name,
                np.rint(chunk).astype(dtype) if dtype == np.int64 else chunk.copy(),
            )
            offset += self.num_horizons

    def state_dict(self) -> dict[str, Any]:
        return {
            "num_horizons": self.num_horizons,
            "num_sensors": self.num_sensors,
            "contact_dz_threshold": self.contact_dz_threshold,
            "eps": self.eps,
            "reduction_vector": self.reduction_vector().tolist(),
        }

    @classmethod
    def from_state_dict(
        cls, state: Mapping[str, Any]
    ) -> "PhysicalTactileMetricAccumulator":
        result = cls(
            num_horizons=int(state["num_horizons"]),
            num_sensors=int(state["num_sensors"]),
            contact_dz_threshold=float(state["contact_dz_threshold"]),
            eps=float(state["eps"]),
        )
        result.load_reduction_vector(
            np.asarray(state["reduction_vector"], dtype=np.float64)
        )
        return result

    @staticmethod
    def _safe_ratio(numerator: float, denominator: float) -> float | None:
        if denominator <= 0:
            return None
        return float(numerator / denominator)

    def summary(self) -> dict[str, Any]:
        sensor_rows: dict[str, Any] = {}
        for sensor, name in enumerate(TACTILE_BUNDLE_ORDER[: self.num_sensors]):
            valid = int(self.sensor_cosine_count[sensor])
            sensor_rows[name] = {
                "resultant_cosine": self._safe_ratio(
                    self.sensor_cosine_sum[sensor], valid
                ),
                "resultant_cosine_magnitude_weighted": self._safe_ratio(
                    self.sensor_weighted_cosine_sum[sensor],
                    self.sensor_weight_sum[sensor],
                ),
                "valid_resultant_frames": valid,
                "pred_zero_on_valid_gt_fraction": self._safe_ratio(
                    self.sensor_pred_zero_count[sensor], valid
                ),
            }

        horizon_rows = []
        for horizon in range(self.num_horizons):
            valid = int(self.horizon_cosine_count[horizon])
            horizon_rows.append(
                {
                    "horizon": horizon,
                    "l1": self._safe_ratio(
                        self.horizon_abs_sum[horizon], self.horizon_count[horizon]
                    ),
                    "contact_l1": self._safe_ratio(
                        self.horizon_contact_abs_sum[horizon],
                        self.horizon_contact_count[horizon],
                    ),
                    "resultant_cosine": self._safe_ratio(
                        self.horizon_cosine_sum[horizon], valid
                    ),
                    "resultant_cosine_magnitude_weighted": self._safe_ratio(
                        self.horizon_weighted_cosine_sum[horizon],
                        self.horizon_weight_sum[horizon],
                    ),
                    "valid_resultant_frames": valid,
                    "pred_zero_on_valid_gt_fraction": self._safe_ratio(
                        self.horizon_pred_zero_count[horizon], valid
                    ),
                }
            )

        total_valid = int(self.sensor_cosine_count.sum())
        return {
            "l1": self._safe_ratio(self.abs_sum, self.count),
            "contact_l1": self._safe_ratio(
                self.contact_abs_sum, self.contact_count
            ),
            "resultant_cosine": self._safe_ratio(
                float(self.sensor_cosine_sum.sum()), total_valid
            ),
            "resultant_cosine_magnitude_weighted": self._safe_ratio(
                float(self.sensor_weighted_cosine_sum.sum()),
                float(self.sensor_weight_sum.sum()),
            ),
            "valid_resultant_frames": total_valid,
            "pred_zero_on_valid_gt_fraction": self._safe_ratio(
                int(self.sensor_pred_zero_count.sum()), total_valid
            ),
            "contact_dz_threshold": self.contact_dz_threshold,
            "per_sensor": sensor_rows,
            "per_horizon": horizon_rows,
        }
