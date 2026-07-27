"""Tests for incremental drift correction and minimum tile size selection."""

from __future__ import annotations

import math

from core.calibration import NormalizedPoint, PolynomialRidgeCalibration, make_grid
from core.drift import BiasRecalibration, DriftCorrector, min_tile_side_px


def _calibrated_strategy() -> PolynomialRidgeCalibration:
    """Generates a reliably trained mock strategy for testing."""
    s = PolynomialRidgeCalibration(grid=make_grid(3))
    for target in s.target_sequence():
        batch = [[target.x + 0.001 * k, target.y, target.x, target.y] for k in range(5)]
        s.add_samples(target, batch)
    s.train()
    return s


def test_drift_correction_pulls_prediction_toward_tile() -> None:
    s = _calibrated_strategy()
    tile = NormalizedPoint(0.25, 0.75)

    # Simulate drift: features that should map to the tile now predict elsewhere.
    drifted_vec = [tile.x + 0.06, tile.y, tile.x + 0.06, tile.y]
    before = s.predict(drifted_vec)
    assert before is not None
    err_before = math.hypot(before.x - tile.x, before.y - tile.y)

    corr = DriftCorrector(s, frames_per_activation=8)
    for _ in range(8):
        corr.observe(drifted_vec)

    result = corr.on_activation(tile)
    assert result is not None

    added, pre_error_norm, post_error_norm = result
    assert added == 8

    after = s.predict(drifted_vec)
    assert after is not None
    err_after = math.hypot(after.x - tile.x, after.y - tile.y)

    # Model must move toward the labeled truth
    assert err_after < err_before 

    # Batch error against the tile center must not grow after the refit
    assert post_error_norm <= pre_error_norm


def test_drift_correction_caps_and_guards() -> None:
    s = _calibrated_strategy()
    corr = DriftCorrector(s, frames_per_activation=8, max_corrections=2)
    vec = [0.5, 0.5, 0.5, 0.5]

    for _ in range(2):
        corr.observe(vec)

    # <3 frames: refuse
    assert corr.on_activation(NormalizedPoint(0.5, 0.5)) is None 

    for _ in range(3):
        for _ in range(8):
            corr.observe(vec)
        corr.on_activation(NormalizedPoint(0.5, 0.5))

    # Ensure it's capped at max_corrections limit
    assert corr.corrections_applied == 2


def test_tile_size_from_measured_error_with_floor() -> None:
    assert min_tile_side_px(85.0) == 170  # 2 x measured error
    assert min_tile_side_px(20.0) == 80   # ergonomic floor
    assert min_tile_side_px(None) == 80
    assert min_tile_side_px(float("nan")) == 80


def test_bias_recalibration_requires_min_frames() -> None:
    recal = BiasRecalibration()
    target = NormalizedPoint(0.5, 0.5)

    for _ in range(BiasRecalibration.MIN_FRAMES - 1):
        recal.feed(NormalizedPoint(0.6, 0.6))

    # below MIN_FRAMES: no offset
    assert recal.compute(target) is None

    recal.feed(NormalizedPoint(0.6, 0.6))
    assert recal.n_fed == BiasRecalibration.MIN_FRAMES
    assert recal.compute(target) is not None


def test_bias_recalibration_computes_offset_from_shifted_predictions() -> None:
    recal = BiasRecalibration()
    target = NormalizedPoint(0.5, 0.5)

    # Predictions land consistently 0.08 right / 0.05 low of the target.
    # The offset must cancel that shift.
    for _ in range(20):
        recal.feed(NormalizedPoint(0.58, 0.55))
        
    offset = recal.compute(target)
    assert offset is not None
    off_x, off_y = offset
    assert math.isclose(off_x, -0.08, abs_tol=1e-9)
    assert math.isclose(off_y, -0.05, abs_tol=1e-9)


def test_bias_recalibration_reset_clears_predictions() -> None:
    recal = BiasRecalibration()
    for _ in range(15):
        recal.feed(NormalizedPoint(0.6, 0.6))
        
    recal.reset()
    assert recal.n_fed == 0
    assert recal.compute(NormalizedPoint(0.5, 0.5)) is None