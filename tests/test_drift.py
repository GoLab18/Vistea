"""Tests for drift correction (DriftCorrector) and minimum tile size selection."""

import math

from core.calibration import NormalizedPoint, PolynomialRidgeCalibration, make_grid
from core.drift import DriftCorrector, min_tile_side_px


# drift correction

def _calibrated_strategy():
    s = PolynomialRidgeCalibration(grid=make_grid(3))
    for target in s.target_sequence():
        batch = [[target.x + 0.001 * k, target.y, target.x, target.y] for k in range(5)]
        s.add_samples(target, batch)
    s.train()
    return s


def test_drift_correction_pulls_prediction_toward_tile():
    s = _calibrated_strategy()
    tile = NormalizedPoint(0.25, 0.75)

    # Simulate drift: features that SHOULD map to the tile now predict elsewhere.
    drifted_vec = [tile.x + 0.06, tile.y, tile.x + 0.06, tile.y]
    before = s.predict(drifted_vec)
    err_before = math.hypot(before.x - tile.x, before.y - tile.y)

    corr = DriftCorrector(s, frames_per_activation=8)
    for _ in range(8):
        corr.observe(drifted_vec)
    added = corr.on_activation(tile)
    assert added == 8

    after = s.predict(drifted_vec)
    err_after = math.hypot(after.x - tile.x, after.y - tile.y)
    assert err_after < err_before # model moved toward the labeled truth


def test_drift_correction_caps_and_guards():
    s = _calibrated_strategy()
    corr = DriftCorrector(s, frames_per_activation=8, max_corrections=2)
    vec = [0.5, 0.5, 0.5, 0.5]
    for _ in range(2):
        corr.observe(vec)
    assert corr.on_activation(NormalizedPoint(0.5, 0.5)) == 0 # <3 frames: refuse
    for _ in range(3):
        for _ in range(8):
            corr.observe(vec)
        corr.on_activation(NormalizedPoint(0.5, 0.5))
    assert corr.corrections_applied == 2  # capped

# tile sizing

def test_tile_size_from_measured_error_with_floor():
    assert min_tile_side_px(85.0) == 170 # 2 x measured error
    assert min_tile_side_px(20.0) == 80 # ergonomic floor
    assert min_tile_side_px(None) == 80
    assert min_tile_side_px(float("nan")) == 80
