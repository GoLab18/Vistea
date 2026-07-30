"""Tests for incremental drift correction and minimum tile size selection."""

from __future__ import annotations

import math

from core.calibration import NormalizedPoint, PolynomialRidgeCalibration, make_grid
from core.drift import (
    BiasRecalibration,
    DriftCorrector,
    RecalibrationSession,
    RecalPhase,
    min_tile_side_px,
)


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


CENTER = NormalizedPoint(0.5, 0.5)


def _run_recal_session(session: RecalibrationSession, gaze, fixating, t0, steps, dt=0.05):
    status = None
    for i in range(steps):
        status = session.update(gaze, fixating, t0 + i * dt)
    assert status is not None
    return status


def test_session_shows_a_prompt_before_measuring_anything() -> None:
    session = RecalibrationSession(CENTER, 0.0)
    status = session.update(CENTER, True, 0.5)

    assert status.phase is RecalPhase.PROMPT
    assert status.message
    assert session.n_frames == 0


def test_prompt_gives_way_to_an_armed_target() -> None:
    session = RecalibrationSession(CENTER, 0.0)
    session.update(None, False, 0.5)
    status = session.update(None, False, RecalibrationSession.PROMPT_S + 0.01)

    assert status.phase is RecalPhase.ARMED
    assert session.n_frames == 0


def test_collection_starts_only_after_a_held_fixation_on_the_target() -> None:
    session = RecalibrationSession(CENTER, 0.0)
    t = RecalibrationSession.PROMPT_S + 0.01
    session.update(None, False, t)

    # Fixating, but not long enough to be deliberate
    status = session.update(CENTER, True, t + 0.1)
    assert status.phase is RecalPhase.ARMED
    assert session.n_frames == 0

    status = _run_recal_session(session, CENTER, True, t + 0.2, steps=4, dt=0.2)
    assert status.phase in (RecalPhase.COLLECTING, RecalPhase.DONE)


def test_large_bias_is_corrected_not_refused() -> None:
    session = RecalibrationSession(CENTER, 0.0)
    t = RecalibrationSession.PROMPT_S + 0.01
    session.update(None, False, t)

    # A badly degraded model maps the screen center far into the corner
    far = NormalizedPoint(0.18, 0.83)
    status = _run_recal_session(session, far, True, t, steps=40, dt=0.05)

    assert status.phase is RecalPhase.DONE
    assert status.offset is not None
    assert math.isclose(status.offset[0], 0.32, abs_tol=1e-9)
    assert math.isclose(status.offset[1], -0.33, abs_tol=1e-9)


def test_wandering_gaze_does_not_produce_an_offset() -> None:
    session = RecalibrationSession(CENTER, 0.0)
    t = RecalibrationSession.PROMPT_S + 0.01
    session.update(None, False, t)

    status = None
    for i in range(40):
        drifting = NormalizedPoint(0.30 + 0.02 * i, 0.30 + 0.02 * i)
        status = session.update(drifting, True, t + i * 0.05)

    assert status is not None
    assert status.phase is RecalPhase.COLLECTING
    assert status.offset is None


def test_session_cancels_when_the_target_is_never_fixated() -> None:
    session = RecalibrationSession(CENTER, 0.0)
    session.update(None, False, RecalibrationSession.PROMPT_S + 0.01)

    status = session.update(None, False, RecalibrationSession.PROMPT_S + 20.0)
    assert status.phase is RecalPhase.CANCELLED
    assert status.finished
    assert status.offset is None


def test_completed_session_yields_the_offset_that_cancels_the_shift() -> None:
    session = RecalibrationSession(CENTER, 0.0)
    t = RecalibrationSession.PROMPT_S + 0.01
    session.update(None, False, t)

    shifted = NormalizedPoint(0.56, 0.46)  # predictions land right and high
    status = _run_recal_session(session, shifted, True, t, steps=40, dt=0.05)

    assert status.phase is RecalPhase.DONE
    assert status.offset is not None
    assert math.isclose(status.offset[0], -0.06, abs_tol=1e-9)
    assert math.isclose(status.offset[1], 0.04, abs_tol=1e-9)


def _armed_and_collecting(steps: int = 14) -> tuple[RecalibrationSession, float]:
    session = RecalibrationSession(CENTER, 0.0)
    t = RecalibrationSession.PROMPT_S + 0.01
    session.update(None, False, t)
    _run_recal_session(session, CENTER, True, t, steps=steps, dt=0.05)
    assert session.phase is RecalPhase.COLLECTING
    return session, t + steps * 0.05


def test_a_blink_mid_collection_does_not_discard_the_samples() -> None:
    session, t = _armed_and_collecting()
    before = session.n_frames
    assert before > 0

    # Blink, inside the grace window
    session.update(None, False, t + 0.05)
    status = session.update(CENTER, True, t + 0.10)

    assert session.n_frames > before
    assert status.phase is not RecalPhase.ARMED


def test_losing_the_gaze_for_too_long_rearms_and_drops_partial_samples() -> None:
    session, t = _armed_and_collecting()
    assert session.n_frames > 0

    session.update(None, False, t + 0.05)
    status = session.update(None, False, t + 0.05 + RecalibrationSession.LOST_GRACE_S + 0.1)

    assert status.phase is RecalPhase.ARMED
    assert session.n_frames == 0