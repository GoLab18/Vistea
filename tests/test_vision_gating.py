"""Eyelid stability gating tests (frames discarded while a lid is in motion)."""

from __future__ import annotations

from core.vision import EyelidStabilityGate

FPS = 30.0
DT = 1.0 / FPS


def _settle(gate: EyelidStabilityGate, openness: float = 0.30, seconds: float = 2.0) -> float:
    t = 0.0
    while t < seconds:
        gate.update(openness, t)
        t += DT
    return t


def _ramp(start: float, end: float, seconds: float) -> list[float]:
    """Openness samples for a linear lid movement at the camera frame rate."""

    n = max(1, round(seconds * FPS))
    return [start + (end - start) * (i + 1) / n for i in range(n)]


def _feed(gate: EyelidStabilityGate, values: list[float], t0: float) -> list[bool]:
    return [gate.update(v, t0 + i * DT) for i, v in enumerate(values)]


def test_first_frame_establishes_the_baseline() -> None:
    gate = EyelidStabilityGate()
    assert gate.update(0.30, 0.0) is True
    assert gate.baseline == 0.30


def test_open_eye_frames_stay_stable() -> None:
    gate = EyelidStabilityGate()
    gate.update(0.30, 0.0)
    assert all(gate.update(0.29 + 0.01 * (i % 3), (i + 1) * DT) for i in range(20))


def test_closing_lid_is_flagged_before_the_blink_threshold() -> None:
    # 0.21 > MIN_EYE_OPENNESS = 0.10, so the blink filter would pass it,
    # yet the lid is mid-travel.
    gate = EyelidStabilityGate(stable_fraction=0.8)
    t = _settle(gate)
    assert gate.update(0.21, t) is False


def test_baseline_is_not_dragged_down_by_a_blink() -> None:
    gate = EyelidStabilityGate(stable_fraction=0.8, adapt_rate=0.5)
    t = _settle(gate)
    before = gate.baseline

    for i in range(6):
        gate.update(0.05, t + i * DT)  # eyes shut

    assert gate.baseline == before
    assert gate.update(0.30, t + 6 * DT) is True


def test_whole_blink_is_flagged_in_both_directions() -> None:
    gate = EyelidStabilityGate()
    t = _settle(gate)

    blink = _ramp(0.30, 0.03, 0.10) + _ramp(0.03, 0.30, 0.10)
    flags = _feed(gate, blink, t)

    assert flags.count(False) >= len(blink) - 1


def test_settled_narrow_eye_is_not_treated_as_lid_motion() -> None:
    gate = EyelidStabilityGate()
    t = _settle(gate)

    # Fast downward saccade, then the gaze is held there
    descent = _ramp(0.30, 0.15, 0.10)
    plateau = [0.15] * int(0.6 * FPS)
    flags = _feed(gate, descent + plateau, t)

    # A couple of transitional frames may be withheld, but not the plateau
    assert flags.count(False) <= 3
    assert all(flags[len(descent):])


def test_baseline_follows_a_sustained_downward_gaze() -> None:
    gate = EyelidStabilityGate()
    t = _settle(gate)

    held = _ramp(0.30, 0.18, 0.15) + [0.18] * int(3 * FPS)
    _feed(gate, held, t)

    assert gate.baseline is not None and gate.baseline < 0.22


def test_blink_is_still_detected_from_a_narrowed_baseline() -> None:
    gate = EyelidStabilityGate()
    t = _settle(gate)
    held = _ramp(0.30, 0.18, 0.15) + [0.18] * int(3 * FPS)
    _feed(gate, held, t)
    t += len(held) * DT

    blink = _ramp(0.18, 0.03, 0.10) + _ramp(0.03, 0.18, 0.10)
    flags = _feed(gate, blink, t)

    assert flags.count(False) >= len(blink) - 1


def test_gate_cannot_withhold_frames_indefinitely() -> None:
    gate = EyelidStabilityGate(max_hold_s=0.5)
    t = _settle(gate)

    # Openness oscillating fast enough to always look like motion
    values = [0.20 if i % 2 else 0.12 for i in range(int(1.5 * FPS))]
    flags = _feed(gate, values, t)

    assert flags[0] is False
    assert True in flags


def test_shut_eye_never_becomes_the_new_baseline() -> None:
    gate = EyelidStabilityGate(max_hold_s=0.2)
    t = _settle(gate)
    before = gate.baseline

    # Resting with eyes closed for far longer than the hold cap.
    flags = _feed(gate, [0.03] * int(3 * FPS), t)

    assert not any(flags)
    assert gate.baseline == before


def test_reset_forgets_the_baseline() -> None:
    gate = EyelidStabilityGate()
    _settle(gate)
    gate.reset()
    assert gate.baseline is None
    assert gate.update(0.10, 0.0) is True
