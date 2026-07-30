"""Eyelid stability gating tests (frames discarded while a lid is in motion)."""

from __future__ import annotations

from core.vision import EyelidStabilityGate


def test_first_frame_establishes_the_baseline() -> None:
    gate = EyelidStabilityGate()
    assert gate.update(0.30) is True
    assert gate.baseline == 0.30


def test_open_eye_frames_stay_stable() -> None:
    gate = EyelidStabilityGate()
    gate.update(0.30)
    assert all(gate.update(0.29 + 0.01 * (i % 3)) for i in range(20))


def test_closing_lid_is_flagged_before_the_blink_threshold() -> None:
    gate = EyelidStabilityGate(stable_fraction=0.8)
    gate.update(0.30)
    assert gate.update(0.21) is False


def test_baseline_is_not_dragged_down_by_a_blink() -> None:
    gate = EyelidStabilityGate(stable_fraction=0.8, adapt_rate=0.5)
    gate.update(0.30)

    for _ in range(6):
        gate.update(0.05)  # eyes shut

    assert gate.baseline == 0.30
    assert gate.update(0.30) is True


def test_baseline_follows_a_genuinely_narrower_eye() -> None:
    gate = EyelidStabilityGate(stable_fraction=0.8, adapt_rate=0.5)
    gate.update(0.30)

    # A user settling into a squint: each step stays within the stable band,
    # so the baseline tracks it instead of permanently rejecting the frames.
    for openness in (0.27, 0.24, 0.22, 0.20, 0.18):
        assert gate.update(openness) is True
    assert gate.baseline is not None and gate.baseline < 0.24


def test_reset_forgets_the_baseline() -> None:
    gate = EyelidStabilityGate()
    gate.update(0.30)
    gate.reset()
    assert gate.baseline is None
    assert gate.update(0.10) is True
