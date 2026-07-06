"""I-DT fixation detector tests (dispersion, saccade interruption, blink survival)."""

import math

from core.fixation import IDTFixationDetector


# I-DT

def _feed(det, points):
    state = None
    for t, x, y in points:
        state = det.update(x, y, t)
    return state


def test_idt_stable_gaze_becomes_fixation():
    det = IDTFixationDetector(dispersion_threshold=0.04, min_duration=0.12)
    pts = [(i * 0.02, 0.5 + 0.005 * math.sin(i), 0.5 + 0.005 * math.cos(i)) for i in range(20)]
    state = _feed(det, pts)
    assert state.fixating
    assert abs(state.centroid[0] - 0.5) < 0.01 and abs(state.centroid[1] - 0.5) < 0.01
    assert state.duration_s > 0.2


def test_idt_saccade_breaks_fixation_then_new_one_forms():
    det = IDTFixationDetector()
    _feed(det, [(i * 0.02, 0.3, 0.3) for i in range(15)])
    state = det.update(0.8, 0.8, 0.32)  # saccade jump
    assert not state.fixating
    state = _feed(det, [(0.34 + i * 0.02, 0.8, 0.8) for i in range(12)])
    assert state.fixating and abs(state.centroid[0] - 0.8) < 0.01


def test_idt_survives_blink_gap_shorter_than_reset():
    det = IDTFixationDetector(gap_reset=0.5)
    _feed(det, [(i * 0.02, 0.5, 0.5) for i in range(15)])
    state = det.update(None, None, 0.40) # blink frame
    assert not state.fixating
    state = det.update(0.5, 0.5, 0.45) # gaze returns to same spot
    assert state.fixating # window survived the gap
