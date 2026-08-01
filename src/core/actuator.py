"""System actuation: OS cursor movement, clicking, scrolling, and typing.

The gaze UI executes OS-level actions without taking window focus, injecting
input directly into the user's active application.
"""

from __future__ import annotations

import sys
from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable

# Named keys the actuator understands. Anything else is treated as a literal
# character, which is what makes chords such as ("cmd", "[") expressible.
type SpecialKey = str


@runtime_checkable
class SystemActuator(Protocol):
    name: str
    available: bool

    def move(self, x_px: int, y_px: int) -> None: ...
    def click(self) -> None: ...
    def scroll(self, ticks: int) -> None: ...

    def hotkey(self, *keys: SpecialKey) -> bool:
        """Press a chord together, release in reverse order.
        Returns False when a key cannot be expressed on the system."""
        ...

    def type_text(self, text: str) -> None:
        """Type literal text into the OS-focused field (unicode-safe)."""
        ...

    def press_key(self, key: SpecialKey) -> bool:
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

    def hotkey(self, *keys: SpecialKey) -> bool:
        self.calls.append(("hotkey", *keys))
        return True

    def type_text(self, text: str) -> None:
        self.calls.append(("type_text", text))

    def press_key(self, key: SpecialKey) -> bool:
        self.calls.append(("press_key", key))
        return True


# Dedicated browser navigation keys. Windows addresses them by virtual-key code,
# X11 by keysym and macOS has no equivalent, so its profile never asks for them.
_BROWSER_VK: Mapping[str, Mapping[str, int]] = {
    "win32": {"browser_back": 0xA6, "browser_forward": 0xA7},
    "linux": {"browser_back": 0x1008FF26, "browser_forward": 0x1008FF27},
}


class PynputActuator:
    name = "pynput"
    available = True

    def __init__(self) -> None:
        import pynput.keyboard
        import pynput.mouse

        self._mouse = pynput.mouse.Controller()
        self._keyboard = pynput.keyboard.Controller()
        self._Button = pynput.mouse.Button
        self._KeyCode = pynput.keyboard.KeyCode

        key = pynput.keyboard.Key
        self._key_map: Mapping[str, Any] = {
            "enter": key.enter,
            "backspace": key.backspace,
            "delete": key.delete,
            "tab": key.tab,
            "esc": key.esc,
            "space": key.space,
            "ctrl": key.ctrl,
            "alt": key.alt,
            "shift": key.shift,
            "cmd": key.cmd,
            "left": key.left,
            "right": key.right,
            "up": key.up,
            "down": key.down,
            "home": key.home,
            "end": key.end,
            "page_up": key.page_up,
            "page_down": key.page_down,
        }

        platform = "win32" if sys.platform == "win32" else (
            "linux" if sys.platform.startswith("linux") else sys.platform
        )
        for name, code in _BROWSER_VK.get(platform, {}).items():
            self._key_map = {**self._key_map, name: self._KeyCode.from_vk(code)}

    def move(self, x_px: int, y_px: int) -> None:
        self._mouse.position = (x_px, y_px)

    def click(self) -> None:
        self._mouse.click(self._Button.left, 1)

    def scroll(self, ticks: int) -> None:
        if ticks:
            self._mouse.scroll(0, ticks)

    def _resolve(self, name: str) -> Any | None:
        """Map a key name to a pynput key, or a single character to itself."""

        known = self._key_map.get(name)
        if known is not None:
            return known
        if len(name) == 1:
            return self._KeyCode.from_char(name)
        return None

    def hotkey(self, *keys: str) -> bool:
        resolved = [self._resolve(k) for k in keys]
        if any(k is None for k in resolved):
            return False

        for k in resolved:
            self._keyboard.press(k)
        for k in reversed(resolved):
            self._keyboard.release(k)
        return True

    def type_text(self, text: str) -> None:
        if text:
            self._keyboard.type(text)

    def press_key(self, key: str) -> bool:
        resolved = self._resolve(key)
        if resolved is None:
            return False
        self._keyboard.press(resolved)
        self._keyboard.release(resolved)
        return True


def make_actuator() -> SystemActuator:
    try:
        return PynputActuator()
    except ImportError:
        return NullActuator()