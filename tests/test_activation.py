"""Activation strategy tests: dwell, blink, ActivationKind, and telemetry serialization."""

import json

from core.activation import (
    ACTIVATION_REGISTRY,
    ActivationKind,
    ActivationStrategy,
    BlinkActivation,
    DwellActivation,
    extract_blink_score,
)


# dwell

def test_dwell_fires_after_dwell_time_and_reports_progress():
    act = DwellActivation(dwell_time=1.0, cooldown=0.5)
    fired = []
    for i in range(60):  # 60 frames x 25 ms = 1.5 s
        ev = act.update("key:a", True, 0.0, i * 0.025)
        if ev:
            fired.append((ev, i * 0.025))
    assert len(fired) == 1
    ev, t_fire = fired[0]
    assert ev.target_id == "key:a" and ev.kind == ActivationKind.DWELL
    assert 0.95 <= t_fire <= 1.1


def test_dwell_repeats_in_place_after_cooldown_and_repeat_dwell():
    # First activation: plain dwell_time (0.5 s). The gaze never leaves the
    # key, so it repeats - but each repeat needs cooldown + dwell*repeat_factor.
    act = DwellActivation(dwell_time=0.5, cooldown=0.2, repeat_factor=1.25)
    fire_times = []
    for i in range(200):  # 5 s glued to the same key
        t = i * 0.025
        ev = act.update("key:a", True, 0.0, t)
        if ev:
            fire_times.append(t)
    assert len(fire_times) >= 2 # in-place repeats happen (no leave needed)

    # first at ~0.5 s (plain dwell), the second only after cooldown + repeat dwell
    assert fire_times[0] <= 0.6
    gap = fire_times[1] - fire_times[0]
    assert gap >= 0.2 + 0.5 * 1.25 - 0.05 # cooldown + repeat threshold


def test_dwell_leaving_and_returning_uses_plain_threshold_again():
    act = DwellActivation(dwell_time=0.5, cooldown=0.2, repeat_factor=1.25)

    # fire once
    for i in range(30):
        act.update("key:a", True, 0.0, i * 0.025) # 0.75 s -> fires at 0.5 s
    # leave the key: repeat state resets
    act.update(None, False, 0.0, 1.0)
    act.update("key:b", True, 0.0, 1.05)  # somewhere else
    # return: the next activation is a PLAIN dwell again, not the repeat dwell
    fired = []
    for i in range(24): # 0.6 s, enough for plain 0.5 s but not repeat 0.625 s
        t = 2.0 + i * 0.025
        ev = act.update("key:a", True, 0.0, t)
        if ev:
            fired.append(t)
    assert len(fired) == 1
    assert fired[0] <= 2.0 + 0.55 # plain threshold, not the longer repeat one


def test_dwell_repeat_progress_reaches_one():
    act = DwellActivation(dwell_time=0.5, cooldown=0.2, repeat_factor=1.25)
    # first fire
    for i in range(30):
        act.update("key:a", True, 0.0, i * 0.025)

    # accumulate a repeat with fine steps and record progress just before it
    # fires (progress resets to 0 on the firing frame).
    peak = 0.0
    for i in range(400):
        t = 0.75 + i * 0.005
        ev = act.update("key:a", True, 0.0, t)
        if ev: # a repeat fired: the arc had filled to ~1.0 the frame before
            break
        peak = max(peak, act.progress)
    assert peak >= 0.99 # the arc fills (relative to the repeat threshold) for repeats too


def test_dwell_target_change_resets_progress():
    act = DwellActivation(dwell_time=1.0)
    for i in range(30):
        act.update("key:a", True, 0.0, i * 0.025)
    assert act.progress > 0.5
    act.update("key:b", True, 0.0, 0.80)
    assert act.progress < 0.1


def test_dwell_grace_survives_short_blink_but_not_long_break():
    act = DwellActivation(dwell_time=1.0, grace=0.25)
    for i in range(20):
        act.update("key:a", True, 0.0, i * 0.025) # 0.5 s accumulated
    p_before = act.progress
    act.update(None, False, 1.0, 0.55) # blink 0.1 s < grace
    act.update("key:a", True, 0.0, 0.60)
    assert act.progress >= p_before * 0.9 # progress held
    act.update(None, False, 0.0, 0.70)
    act.update(None, False, 0.0, 1.10) # 0.4 s > grace
    assert act.progress == 0.0


# blink

class _Cat:
    def __init__(self, name, score):
        self.category_name, self.score = name, score


def test_extract_blink_score_averages_both_eyes():
    bs = [_Cat("eyeBlinkLeft", 0.8), _Cat("eyeBlinkRight", 0.6), _Cat("jawOpen", 1.0)]
    assert abs(extract_blink_score(bs) - 0.7) < 1e-9
    assert extract_blink_score(None) is None


def test_deliberate_blink_fires_natural_blink_ignored():
    act = BlinkActivation(min_blink=0.35, max_blink=1.2)

    # natural blink: 0.15 s closed
    act.update("key:a", True, 0.0, 0.0)
    act.update(None, False, 0.9, 0.10)
    ev = act.update("key:a", True, 0.0, 0.25)
    assert ev is None

    # deliberate blink: 0.5 s closed
    act.update("key:a", True, 0.0, 0.30)
    act.update(None, False, 0.9, 0.40)
    ev = None
    ev = act.update("key:a", True, 0.05, 0.90) or ev
    assert ev is not None and ev.kind == ActivationKind.BLINK and ev.target_id == "key:a"


def test_overlong_closure_is_rest_not_intent():
    act = BlinkActivation(min_blink=0.35, max_blink=1.2)
    act.update("key:a", True, 0.0, 0.0)
    act.update(None, False, 0.9, 0.1)
    ev = act.update("key:a", True, 0.0, 1.5) # 1.4 s closed > max
    assert ev is None


def test_registry_strategies_satisfy_protocol():
    for factory in ACTIVATION_REGISTRY.values():
        assert isinstance(factory(), ActivationStrategy)


# kind telemetry contract

def test_activation_event_kind_is_enum_member():
    act = DwellActivation(dwell_time=0.1)
    ev = act.update("key:a", True, 0.0, 0.0)
    ev = act.update("key:a", True, 0.0, 0.2) # crosses dwell_time
    assert isinstance(ev.kind, ActivationKind)
    assert ev.kind is ActivationKind.DWELL

    act = BlinkActivation(min_blink=0.35, max_blink=1.2)
    act.update("key:a", True, 0.0, 0.0)
    act.update(None, False, 0.9, 0.1)
    ev = act.update("key:a", True, 0.05, 0.5) # 0.4 s closed: deliberate
    assert isinstance(ev.kind, ActivationKind)
    assert ev.kind is ActivationKind.BLINK


def test_activation_kind_serializes_to_plain_string_not_repr():
    payload = json.dumps({"kind": ActivationKind.DWELL, "target": "key:a"})
    assert json.loads(payload)["kind"] == "dwell"
    assert "ActivationKind" not in payload # not the repr form
