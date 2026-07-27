"""Tests for NullActuator, SystemActuator protocol compliance, type_text, and press_key."""

from __future__ import annotations

from core.actuator import NullActuator, SystemActuator


def test_null_actuator_records_and_satisfies_protocol() -> None:
    act = NullActuator()
    assert isinstance(act, SystemActuator)
    act.move(10, 20); act.click(); act.scroll(-3); act.hotkey("alt", "left")
    assert act.calls == [("move", 10, 20), ("click",), ("scroll", -3), ("hotkey", "alt", "left")]


def test_null_actuator_records_type_text_and_press_key() -> None:
    act = NullActuator()
    assert isinstance(act, SystemActuator)
    act.type_text("a"); act.press_key("backspace")
    assert act.calls == [("type_text", "a"), ("press_key", "backspace")]