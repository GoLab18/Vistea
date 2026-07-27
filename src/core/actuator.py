"""System actuation: OS cursor movement, clicking, scrolling, and typing.

The gaze UI executes OS-level actions without taking window focus, injecting
input directly into the user's active application.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal, Protocol, runtime_checkable

type SpecialKey = Literal["enter", "backspace", "ctrl", "alt", "left", "right"]


@runtime_checkable
class SystemActuator(Protocol):
    name: str
    available: bool

    def move(self, x_px: int, y_px: int) -> None: ...
    def click(self) -> None: ...
    def scroll(self, ticks: int) -> None: ...
    def hotkey(self, *keys: SpecialKey) -> None: ...

    def type_text(self, text: str) -> None:
        """Type literal text into the OS-focused field (unicode-safe)."""
        ...

    def press_key(self, key: SpecialKey) -> None:
        """Press-and-release a named special key.
        Single characters are typed literally."""
        ...


class NullActuator:
    """Records calls instead of performing them (tests, missing pynput)."""

    name = "null"
    available = False

    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def move(self, x_px: int, y_px: int) -> None:
        self.calls.append(("move", x_px, y_px))

    def click(self) -> None:
        self.calls.append(("click",))

    def scroll(self, ticks: int) -> None:
        self.calls.append(("scroll", ticks))

    def hotkey(self, *keys: SpecialKey) -> None:
        self.calls.append(("hotkey", *keys))

    def type_text(self, text: str) -> None:
        self.calls.append(("type_text", text))

    def press_key(self, key: SpecialKey) -> None:
        self.calls.append(("press_key", key))


class PynputActuator:
    name = "pynput"
    available = True

    def __init__(self) -> None:
        import pynput.keyboard
        import pynput.mouse
        
        self._mouse = pynput.mouse.Controller()
        self._keyboard = pynput.keyboard.Controller()
        self._Button = pynput.mouse.Button

        self._key_map: Mapping[str, pynput.keyboard.Key] = {
            "enter": pynput.keyboard.Key.enter,
            "backspace": pynput.keyboard.Key.backspace,
            "ctrl": pynput.keyboard.Key.ctrl,
            "alt": pynput.keyboard.Key.alt,
            "left": pynput.keyboard.Key.left,
            "right": pynput.keyboard.Key.right,
        }

    def move(self, x_px: int, y_px: int) -> None:
        self._mouse.position = (x_px, y_px)

    def click(self) -> None:
        self._mouse.click(self._Button.left, 1)

    def scroll(self, ticks: int) -> None:
        if ticks:
            self._mouse.scroll(0, ticks)

    def hotkey(self, *keys: str) -> None:
        resolved = [self._key_map[k] for k in keys]

        for k in resolved:
            self._keyboard.press(k)
        for k in reversed(resolved):
            self._keyboard.release(k)

    def type_text(self, text: str) -> None:
        if text:
            self._keyboard.type(text)

    def press_key(self, key: str) -> None:
        resolved = self._key_map[key]
        self._keyboard.press(resolved)
        self._keyboard.release(resolved)


def make_actuator() -> SystemActuator:
    try:
        return PynputActuator()
    except ImportError:
        return NullActuator()