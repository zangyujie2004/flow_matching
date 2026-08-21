from __future__ import annotations

import numpy as np

from tools.eval_tactile_reconstruction import (
    EpisodeReconstructionAccumulator,
    EvalWindow,
    select_plot_window_keys,
)
from tools.tactile_resultant_metrics import (
    PhysicalTactileMetricAccumulator,
    compute_resultant_cosine,
)


def _field(dx: float, dy: float, dz: float, *, horizon: int = 2) -> np.ndarray:
    value = np.zeros((1, horizon, 2, 2, 12), dtype=np.float32)
    for sensor in range(4):
        value[..., sensor * 3 + 0] = dx
        value[..., sensor * 3 + 1] = dy
        value[..., sensor * 3 + 2] = dz
    return value


def test_identical_prediction_has_zero_l1_and_unit_cosine() -> None:
    target = _field(1.0, 0.0, 0.01)
    metrics = PhysicalTactileMetricAccumulator(num_horizons=2)
    metrics.update(target, target)
    summary = metrics.summary()
    assert summary["l1"] == 0.0
    assert summary["contact_l1"] == 0.0
    assert np.isclose(summary["resultant_cosine"], 1.0)
    assert summary["valid_resultant_frames"] == 8


def test_opposite_and_orthogonal_resultants() -> None:
    target = _field(1.0, 0.0, 0.01)
    opposite = _field(-1.0, 0.0, 0.01)
    orthogonal = _field(0.0, 1.0, 0.01)
    opposite_result = compute_resultant_cosine(opposite, target)
    orthogonal_result = compute_resultant_cosine(orthogonal, target)
    np.testing.assert_allclose(opposite_result["cosine"], -1.0)
    np.testing.assert_allclose(orthogonal_result["cosine"], 0.0, atol=1e-12)


def test_no_contact_is_excluded_and_zero_prediction_is_penalized() -> None:
    no_contact = _field(1.0, 0.0, 0.001)
    result = compute_resultant_cosine(no_contact, no_contact)
    assert not result["valid_gt"].any()

    target = _field(1.0, 0.0, 0.01)
    prediction = _field(0.0, 0.0, 0.01)
    metrics = PhysicalTactileMetricAccumulator(num_horizons=2)
    metrics.update(prediction, target)
    summary = metrics.summary()
    assert summary["resultant_cosine"] == 0.0
    assert summary["pred_zero_on_valid_gt_fraction"] == 1.0


def test_state_round_trip_and_merge_preserve_metrics() -> None:
    target = _field(1.0, 2.0, 0.01)
    prediction = _field(0.5, 2.5, 0.02)
    first = PhysicalTactileMetricAccumulator(num_horizons=2)
    first.update(prediction, target)
    restored = PhysicalTactileMetricAccumulator.from_state_dict(first.state_dict())
    np.testing.assert_allclose(restored.reduction_vector(), first.reduction_vector())

    merged = PhysicalTactileMetricAccumulator(num_horizons=2)
    merged.merge(first)
    merged.merge(restored)
    np.testing.assert_allclose(merged.reduction_vector(), 2 * first.reduction_vector())


def test_fixed_plot_window_selection_is_even_and_deterministic() -> None:
    windows = [
        EvalWindow(episode=episode, anchor=episode * 100 + index)
        for episode in (0, 2)
        for index in range(5)
    ]
    selected = select_plot_window_keys(
        windows,
        episode_ids=[2, 0],
        samples_per_episode=3,
    )
    assert selected == {
        (0, 0),
        (0, 2),
        (0, 4),
        (2, 200),
        (2, 202),
        (2, 204),
    }


def test_episode_reconstruction_stitches_overlap_and_nearest_horizon() -> None:
    accumulator = EpisodeReconstructionAccumulator(
        episode_length=6,
        tactile_shape=(2, 2, 12),
    )
    first = np.stack(
        [np.full((2, 2, 12), value, dtype=np.float32) for value in (1, 2, 3)]
    )
    second = np.stack(
        [np.full((2, 2, 12), value, dtype=np.float32) for value in (10, 20, 30)]
    )
    accumulator.update(first, local_start=1)
    accumulator.update(second, local_start=2)
    overlap, nearest, covered, horizons = accumulator.finalize()

    np.testing.assert_array_equal(covered, [False, True, True, True, True, False])
    np.testing.assert_array_equal(accumulator.prediction_count, [0, 1, 2, 2, 1, 0])
    np.testing.assert_allclose(overlap[1:5, 0, 0, 0], [1, 6, 11.5, 30])
    np.testing.assert_allclose(nearest[1:5, 0, 0, 0], [1, 10, 20, 30])
    np.testing.assert_array_equal(horizons, [-1, 0, 0, 1, 2, -1])
