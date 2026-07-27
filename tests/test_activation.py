"""Activation strategy tests: dwell, blink, ActivationKind, and telemetry serialization."""

from __future__ import annotations

import json

from core.activation import (
    ACTIVATION_REGISTRY,
    ActivationKind,
    ActivationStrategy,
    BlinkActivation,
    DwellActivation,
    extract_blink_score,
)


def test_dwell_fires_after_dwell_time_and_reports_progress() -> None:
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


def test_dwell_repeats_in_place_after_cooldown_and_repeat_dwell() -> None:
    act = DwellActivation(dwell_time=0.5, cooldown=0.2, repeat_factor=1.25)
    fire_times = []
    for i in range(200):  # 5 s glued to the same key
        t = i * 0.025
        ev = act.update("key:a", True, 0.0, t)
        if ev:
            fire_times.append(t)
    assert len(fire_times) >= 2 # in-place repeats happen (no leave needed)

    assert fire_times[0] <= 0.6
    gap = fire_times[1] - fire_times[0]
    assert gap >= 0.2 + 0.5 * 1.25 - 0.05 # cooldown + repeat threshold


def test_dwell_leaving_and_returning_uses_plain_threshold_again() -> None:
    act = DwellActivation(dwell_time=0.5, cooldown=0.2, repeat_factor=1.25)

    for i in range(30):
        act.update("key:a", True, 0.0, i * 0.025)

    act.update(None, False, 0.0, 1.0)
    act.update("key:b", True, 0.0, 1.05)

    fired = []
    for i in range(24):
        t = 2.0 + i * 0.025
        ev = act.update("key:a", True, 0.0, t)
        if ev:
            fired.append(t)
    assert len(fired) == 1
    assert fired[0] <= 2.0 + 0.55 


def test_dwell_repeat_progress_reaches_one() -> None:
    act = DwellActivation(dwell_time=0.5, cooldown=0.2, repeat_factor=1.25)
    for i in range(30):
        act.update("key:a", True, 0.0, i * 0.025)

    peak = 0.0
    for i in range(400):
        t = 0.75 + i * 0.005
        ev = act.update("key:a", True, 0.0, t)
        if ev:
            break
        peak = max(peak, act.progress)
    assert peak >= 0.99


def test_dwell_target_change_resets_progress() -> None:
    act = DwellActivation(dwell_time=1.0)
    for i in range(30):
        act.update("key:a", True, 0.0, i * 0.025)
    assert act.progress > 0.5
    act.update("key:b", True, 0.0, 0.80)
    assert act.progress < 0.1


def test_dwell_grace_survives_short_blink_but_not_long_break() -> None:
    act = DwellActivation(dwell_time=1.0, grace=0.25)
    for i in range(20):
        act.update("key:a", True, 0.0, i * 0.025)
    p_before = act.progress

    act.update(None, False, 1.0, 0.55) # blink 0.1 s < grace
    act.update("key:a", True, 0.0, 0.60)
    assert act.progress >= p_before * 0.9

    act.update(None, False, 0.0, 0.70)
    act.update(None, False, 0.0, 1.10) # 0.4 s > grace
    assert act.progress == 0.0


class _Cat:
    """Mock for testing BlendshapeCategory duck-typing."""

    def __init__(self, name: str, score: float) -> None:
        self.category_name = name
        self.score = score


def test_extract_blink_score_averages_both_eyes() -> None:
    bs = [_Cat("eyeBlinkLeft", 0.8), _Cat("eyeBlinkRight", 0.6), _Cat("jawOpen", 1.0)]
    assert abs(extract_blink_score(bs) - 0.7) < 1e-9 # type: ignore
    assert extract_blink_score(None) is None


def test_deliberate_blink_fires_natural_blink_ignored() -> None:
    act = BlinkActivation(min_blink=0.35, max_blink=1.2)

    # Natural blink: 0.15 s closed
    act.update("key:a", True, 0.0, 0.0)
    act.update(None, False, 0.9, 0.10)
    ev = act.update("key:a", True, 0.0, 0.25)
    assert ev is None

    # Deliberate blink: 0.5 s closed
    act.update("key:a", True, 0.0, 0.30)
    act.update(None, False, 0.9, 0.40)
    ev = None
    ev = act.update("key:a", True, 0.05, 0.90) or ev
    assert ev is not None and ev.kind == ActivationKind.BLINK and ev.target_id == "key:a"


def test_overlong_closure_is_rest_not_intent() -> None:
    act = BlinkActivation(min_blink=0.35, max_blink=1.2)
    act.update("key:a", True, 0.0, 0.0)
    act.update(None, False, 0.9, 0.1)
    ev = act.update("key:a", True, 0.0, 1.5) # 1.4 s closed > max
    assert ev is None


def test_registry_strategies_satisfy_protocol() -> None:
    for factory in ACTIVATION_REGISTRY.values():
        assert isinstance(factory(), ActivationStrategy)


def test_activation_event_kind_is_enum_member() -> None:
    act = DwellActivation(dwell_time=0.1)
    ev = act.update("key:a", True, 0.0, 0.0)
    ev = act.update("key:a", True, 0.0, 0.2)
    assert ev is not None
    assert isinstance(ev.kind, ActivationKind)
    assert ev.kind is ActivationKind.DWELL

    act = BlinkActivation(min_blink=0.35, max_blink=1.2)
    act.update("key:a", True, 0.0, 0.0)
    act.update(None, False, 0.9, 0.1)
    ev = act.update("key:a", True, 0.05, 0.5)
    assert ev is not None
    assert isinstance(ev.kind, ActivationKind)
    assert ev.kind is ActivationKind.BLINK


def test_activation_kind_serializes_to_plain_string_not_repr() -> None:
    payload = json.dumps({"kind": ActivationKind.DWELL, "target": "key:a"})
    assert json.loads(payload)["kind"] == "dwell"
    assert "ActivationKind" not in payload