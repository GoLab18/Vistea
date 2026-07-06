"""System actuation: moving the OS cursor, clicking, scrolling, typing.

The gaze UI needs to act on the operating system: navigation mode moves the
real cursor and clicks, edge gestures emulate the mouse wheel and the on-screen
keyboard injects text and key presses into whatever field currently holds OS
focus (the gaze windows never take focus themselves, so the user's target
field keeps it). That capability is isolated behind a Protocol for two
reasons: (1) testability - all interaction logic is verified against a
recording ``NullActuator`` without touching the OS, (2) graceful degradation -
if the optional ``pynput`` dependency is missing, the keyboard still works
in-preview and only OS-level actions become no-ops (with a log notice).

``pynput`` was chosen over pyautogui (heavier, screenshot-oriented) and raw
platform APIs (three codepaths): a single small, pure-Python, cross-platform
dependency covering mouse + keyboard.
"""

from __future__ import annotations

from typing import List, Protocol, Tuple, runtime_checkable


@runtime_checkable
class SystemActuator(Protocol):
    name: str
    available: bool

    def move(self, x_px: int, y_px: int) -> None: ...
    def click(self) -> None: ...
    def scroll(self, ticks: int) -> None: ...
    def hotkey(self, *keys: str) -> None: ...

    def type_text(self, text: str) -> None:
        """Type literal text into the OS-focused field (unicode-safe)."""
        ...

    def press_key(self, key: str) -> None:
        """Press-and-release a named special key (``"enter"``,
        ``"backspace"``, ...); single characters are typed literally."""
        ...


class NullActuator:
    """Records calls instead of performing them (tests / missing pynput)."""

    name = "null"
    available = False

    def __init__(self):
        self.calls: List[Tuple] = []

    def move(self, x_px: int, y_px: int) -> None:
        self.calls.append(("move", x_px, y_px))

    def click(self) -> None:
        self.calls.append(("click",))

    def scroll(self, ticks: int) -> None:
        self.calls.append(("scroll", ticks))

    def hotkey(self, *keys: str) -> None:
        self.calls.append(("hotkey", *keys))

    def type_text(self, text: str) -> None:
        self.calls.append(("type_text", text))

    def press_key(self, key: str) -> None:
        self.calls.append(("press_key", key))


class PynputActuator:
    name = "pynput"
    available = True

    def __init__(self):
        from pynput.keyboard import Controller as KeyboardController, Key
        from pynput.mouse import Button, Controller as MouseController

        self._mouse = MouseController()
        self._keyboard = KeyboardController()
        self._Button = Button
        self._Key = Key

    def move(self, x_px: int, y_px: int) -> None:
        self._mouse.position = (x_px, y_px)

    def click(self) -> None:
        self._mouse.click(self._Button.left, 1)

    def scroll(self, ticks: int) -> None:
        if ticks:
            self._mouse.scroll(0, ticks)

    def hotkey(self, *keys: str) -> None:
        resolved = [getattr(self._Key, k) if hasattr(self._Key, k) else k for k in keys]
        for k in resolved:
            self._keyboard.press(k)
        for k in reversed(resolved):
            self._keyboard.release(k)

    def type_text(self, text: str) -> None:
        if text:
            self._keyboard.type(text)

    def press_key(self, key: str) -> None:
        resolved = getattr(self._Key, key) if hasattr(self._Key, key) else key
        self._keyboard.press(resolved)
        self._keyboard.release(resolved)


def make_actuator() -> SystemActuator:
    try:
        return PynputActuator()
    except Exception:
        return NullActuator()