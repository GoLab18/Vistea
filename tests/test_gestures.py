"""Edge scroll detector tests, including grace, acceleration, suppression."""

from __future__ import annotations

import pytest

from core.gestures import (
    MIN_ZONE,
    SCROLL_NONE,
    EdgeScrollDetector,
    zone_from_error,
)


def test_ignores_zone_entry_if_duration_under_delay() -> None:
    det = EdgeScrollDetector(zone_height_frac=0.12, entry_delay_s=0.4)
    ticks = sum(det.update(0.05, t * 0.05) for t in range(7))
    ticks += det.update(0.5, 0.40)
    assert ticks == 0


def test_activates_and_emits_ticks_after_delay_is_met() -> None:
    det = EdgeScrollDetector(zone_height_frac=0.12, entry_delay_s=0.4, tick_interval_s=0.12)
    total = 0
    t = 0.0
    while t <= 2.0:
        total += det.update(0.05, t)
        t += 0.02

    assert total >= 14
    assert det.active


def test_emits_negative_ticks_when_in_bottom_zone() -> None:
    det = EdgeScrollDetector(zone_height_frac=0.12, entry_delay_s=0.1, tick_interval_s=0.1)
    total = 0
    t = 0.0
    while t <= 1.0:
        total += det.update(0.95, t)
        t += 0.02
    assert total < 0


def test_hysteresis_boundary_jitter_does_not_toggle() -> None:
    det = EdgeScrollDetector(
        zone_height_frac=0.12,
        entry_delay_s=0.1,
        tick_interval_s=0.1,
        hysteresis_frac=0.05
    )
    t = 0.0
    while t <= 0.3:
        det.update(0.10, t)
        t += 0.02
    assert det.active

    ticks = 0
    while t <= 1.0:
        ticks += det.update(0.15, t)
        t += 0.02

    # Kept scrolling through the jitter
    assert det.active and ticks > 0

    det.update(0.30, t)
    assert det.active, "a single frame outside must not drop the gesture"
    det.update(0.30, t + EdgeScrollDetector().zone_exit_tolerance_s + 0.05)
    assert not det.active, "zone exit tolerance must be respected"


def test_resets_state_when_signal_loss_exceeds_tolerance() -> None:
    det = EdgeScrollDetector(zone_height_frac=0.12, entry_delay_s=0.1, signal_loss_tolerance_s=0.3)
    t = 0.0
    while t <= 0.3:
        det.update(0.05, t)
        t += 0.02
    assert det.active

    det.update(None, 0.35)
    det.update(None, 0.75)

    assert not det.active and det.current_zone == SCROLL_NONE


def test_pauses_tick_emission_during_brief_signal_loss() -> None:
    det = EdgeScrollDetector(
        zone_height_frac=0.12,
        entry_delay_s=0.1,
        tick_interval_s=0.1,
        signal_loss_tolerance_s=0.3
    )
    t = 0.0
    while t <= 0.3:
        t_call = t
        det.update(0.05, t_call)
        t += 0.02
    assert det.active

    ticks_during_fade = det.update(None, t_call + 0.05)
    assert ticks_during_fade == 0
    assert det.active

    ticks_after = det.update(0.05, t_call + 0.06)
    assert ticks_after in (0, 1)
    assert det.active


def test_returns_zero_ticks_when_detector_disabled() -> None:
    det = EdgeScrollDetector(zone_height_frac=0.12, entry_delay_s=0.4, enabled=False)
    assert sum(det.update(0.05, t * 0.1) for t in range(30)) == 0
    assert det.current_zone == SCROLL_NONE
    assert not det.active


def test_clears_state_when_detector_is_disabled() -> None:
    det = EdgeScrollDetector(zone_height_frac=0.12, entry_delay_s=0.4)
    for t in range(30):
        det.update(0.05, t * 0.05)
    assert det.active

    assert det.set_enabled(False) is False
    assert not det.active
    assert det.current_zone == SCROLL_NONE

    assert det.set_enabled(True) is True


def test_ignores_suppressed_frames() -> None:
    det = EdgeScrollDetector(zone_height_frac=0.12, entry_delay_s=0.4)
    ticks = sum(det.update(0.05, t * 0.05, suppressed=True) for t in range(30))

    assert ticks == 0
    assert det.current_zone == SCROLL_NONE


def test_cancels_active_gesture_on_suppressed_frame() -> None:
    det = EdgeScrollDetector(zone_height_frac=0.12, entry_delay_s=0.4, tick_interval_s=0.12)
    t = 0.0
    while t <= 1.0:
        det.update(0.05, t)
        t += 0.02
    assert det.active

    assert det.update(0.05, 1.05, suppressed=True) == 0
    assert not det.active


def test_applies_hysteresis_during_arming_phase() -> None:
    det = EdgeScrollDetector(zone_height_frac=0.10, entry_delay_s=0.4, hysteresis_frac=0.05)
    t = 0.0
    total = 0

    # Alternating just inside and just outside the entry boundary
    while t <= 1.5:
        y = 0.08 if int(t * 50) % 2 else 0.13
        total += det.update(y, t)
        t += 0.02

    assert det.active
    assert total > 0


def test_preserves_countdown_during_brief_zone_exit() -> None:
    det = EdgeScrollDetector(zone_height_frac=0.10, entry_delay_s=0.4, zone_exit_tolerance_s=0.25)
    t = 0.0
    while t < 0.30:
        det.update(0.05, t)
        t += 0.02

    for _ in range(5):
        det.update(0.50, t)
        t += 0.02

    while t < 0.75:
        det.update(0.05, t)
        t += 0.02

    assert det.active, "a wobble should cost time, not the whole countdown"


def test_cancels_arming_when_zone_exit_exceeds_grace() -> None:
    det = EdgeScrollDetector(zone_height_frac=0.10, entry_delay_s=0.1, zone_exit_tolerance_s=0.25)
    t = 0.0
    while t < 0.4:
        det.update(0.05, t)
        t += 0.02
    assert det.active

    while t < 1.2:
        det.update(0.50, t)
        t += 0.02

    assert not det.active
    assert det.current_zone == SCROLL_NONE


def test_scrolling_accelerates_the_longer_it_is_held() -> None:
    det = EdgeScrollDetector(
        zone_height_frac=0.10,
        entry_delay_s=0.1,
        accel_step_s=0.8,
        max_scroll_multiplier=4
    )
    t = 0.0
    first_second = later_second = 0
    while t <= 1.0:
        first_second += abs(det.update(0.05, t))
        t += 0.02
    while t <= 4.0:
        later_second += abs(det.update(0.05, t))
        t += 0.02

    per_second_early = first_second / 0.9
    per_second_late = later_second / 3.0
    assert per_second_late > 2 * per_second_early


def test_acceleration_is_capped() -> None:
    det = EdgeScrollDetector(
        zone_height_frac=0.10,
        entry_delay_s=0.1,
        accel_step_s=0.2,
        max_scroll_multiplier=3
    )
    t = 0.0
    while t <= 10.0:
        det.update(0.05, t)
        t += 0.02
    assert det._scroll_multiplier(t) == 3


def test_zone_scales_with_measured_accuracy() -> None:
    assert zone_from_error(85.0, 1500) > MIN_ZONE
    assert zone_from_error(85.0, 1500) == pytest.approx(2 * 85 / 1500)
