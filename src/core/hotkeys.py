"""
Portable key bindings for navigation actions.

To handle platform differences, each action maps to an ordered list of fallback chords
that are tried in sequence and the first one able to be resolved is emitted.
A user-local JSON configuration file can override any default bindings.
"""

from __future__ import annotations

import json
import logging
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Final, Protocol

from core import research_log as rlog

# One key combination, pressed together and released in reverse order
type KeyChord = tuple[str, ...]


class NavAction(str, Enum):
    """A navigation intent, independent of how any platform spells it."""

    BACK = "back"
    FORWARD = "forward"
    DELETE_WORD = "delete_word"


class ChordSender(Protocol):
    """The slice of the system actuator that hotkey dispatch needs."""

    def hotkey(self, *keys: str) -> bool:
        """Emit one chord. Returns False when a key could not be resolved."""
        ...


@dataclass(frozen=True)
class HotkeyProfile:
    """Ordered chord fallbacks per action for one platform."""

    name: str
    chords: Mapping[NavAction, tuple[KeyChord, ...]]

    def chords_for(self, action: NavAction) -> tuple[KeyChord, ...]:
        return self.chords.get(action, ())


# `browser_back` / `browser_forward` are the dedicated navigation keys
# (VK_BROWSER_BACK on Windows, XF86Back on X11). They work regardless of which
# modifier combination the browser has been configured to accept, so they lead.
WINDOWS_PROFILE: Final[HotkeyProfile] = HotkeyProfile(
    "windows",
    {
        NavAction.BACK: (("browser_back",), ("alt", "left")),
        NavAction.FORWARD: (("browser_forward",), ("alt", "right")),
        NavAction.DELETE_WORD: (("ctrl", "backspace"),),
    },
)

LINUX_PROFILE: Final[HotkeyProfile] = HotkeyProfile(
    "linux",
    {
        NavAction.BACK: (("browser_back",), ("alt", "left")),
        NavAction.FORWARD: (("browser_forward",), ("alt", "right")),
        NavAction.DELETE_WORD: (("ctrl", "backspace"),),
    },
)

# macOS has no dedicated browser keys, hence the Command+[ is the system-wide
# "back" and Command+Left is the browser-specific alternative.
MACOS_PROFILE: Final[HotkeyProfile] = HotkeyProfile(
    "darwin",
    {
        NavAction.BACK: (("cmd", "["), ("cmd", "left")),
        NavAction.FORWARD: (("cmd", "]"), ("cmd", "right")),
        NavAction.DELETE_WORD: (("alt", "backspace"),),
    },
)

USER_HOTKEYS_PATH: Final[Path] = Path.home() / ".vistea" / "hotkeys.json"


def default_profile(platform: str = sys.platform) -> HotkeyProfile:
    """Pick the built-in profile matching the running platform."""

    if platform == "win32":
        return WINDOWS_PROFILE
    if platform == "darwin":
        return MACOS_PROFILE
    return LINUX_PROFILE


def _parse_overrides(raw: object) -> dict[NavAction, tuple[KeyChord, ...]]:
    """
    Read `{"back": [["alt", "left"], ["browser_back"]]}` into chord tuples.

    Unknown action names and malformed entries are skipped, because a typo
    in a hand-edited config must not stop the application from starting.
    """

    if not isinstance(raw, dict):
        return {}

    parsed: dict[NavAction, tuple[KeyChord, ...]] = {}
    for key, value in raw.items():
        try:
            action = NavAction(str(key).strip().lower())
        except ValueError:
            continue
        if not isinstance(value, Sequence) or isinstance(value, str):
            continue

        chords: list[KeyChord] = []
        for chord in value:
            if isinstance(chord, str):
                chords.append((chord,))
            elif isinstance(chord, Sequence):
                keys = tuple(str(k) for k in chord if isinstance(k, str) and k)
                if keys:
                    chords.append(keys)
        if chords:
            parsed[action] = tuple(chords)
    return parsed


def load_profile(
    path: Path = USER_HOTKEYS_PATH, platform: str = sys.platform
) -> HotkeyProfile:
    """Built-in profile for the platform, with user overrides merged over it."""

    base = default_profile(platform)
    try:
        overrides = _parse_overrides(json.loads(Path(path).read_text(encoding="utf-8")))
    except (OSError, ValueError):
        return base

    if not overrides:
        return base

    rlog.event(
        "hotkey_profile_overridden",
        profile=base.name,
        actions=sorted(a.value for a in overrides),
    )
    return HotkeyProfile(f"{base.name}+user", {**base.chords, **overrides})


def send(action: NavAction, sender: ChordSender, profile: HotkeyProfile) -> KeyChord | None:
    """
    Emit the first chord that can be produced for `action`.

    Returns the chord that was sent, or None when the profile offered nothing
    this system can express.
    """

    for chord in profile.chords_for(action):
        if sender.hotkey(*chord):
            return chord

    rlog.event(
        "hotkey_unsupported",
        action=action.value,
        profile=profile.name,
        level=logging.WARNING,
    )
    return None
