"""Portable navigation hotkey profile tests."""

from __future__ import annotations

import json
from pathlib import Path

from core.hotkeys import (
    LINUX_PROFILE,
    MACOS_PROFILE,
    WINDOWS_PROFILE,
    HotkeyProfile,
    NavAction,
    default_profile,
    load_profile,
    send,
)


class RecordingSender:
    """Accepts only the chords it was told it can express."""

    def __init__(self, supported: set[tuple[str, ...]] | None = None) -> None:
        self.supported = supported
        self.attempts: list[tuple[str, ...]] = []

    def hotkey(self, *keys: str) -> bool:
        self.attempts.append(keys)
        return self.supported is None or keys in self.supported


def test_default_profile_matches_platform() -> None:
    assert default_profile("win32") is WINDOWS_PROFILE
    assert default_profile("darwin") is MACOS_PROFILE
    assert default_profile("linux") is LINUX_PROFILE
    assert default_profile("freebsd13") is LINUX_PROFILE


def test_every_profile_covers_every_action() -> None:
    for profile in (WINDOWS_PROFILE, MACOS_PROFILE, LINUX_PROFILE):
        for action in NavAction:
            assert profile.chords_for(action), f"{profile.name} lacks {action.value}"


def test_first_expressible_chord_wins() -> None:
    sender = RecordingSender()
    chord = send(NavAction.BACK, sender, WINDOWS_PROFILE)

    assert chord == ("browser_back",)
    assert sender.attempts == [("browser_back",)]


def test_unsupported_chord_falls_back_to_the_next_one() -> None:
    sender = RecordingSender(supported={("alt", "left")})
    chord = send(NavAction.BACK, sender, WINDOWS_PROFILE)

    assert chord == ("alt", "left")
    assert sender.attempts == [("browser_back",), ("alt", "left")]


def test_action_with_no_expressible_chord_reports_failure() -> None:
    sender = RecordingSender(supported=set())
    assert send(NavAction.FORWARD, sender, MACOS_PROFILE) is None
    assert len(sender.attempts) == len(MACOS_PROFILE.chords_for(NavAction.FORWARD))


def test_missing_override_file_yields_the_builtin_profile(tmp_path: Path) -> None:
    profile = load_profile(tmp_path / "absent.json", platform="win32")
    assert profile is WINDOWS_PROFILE


def test_user_overrides_replace_only_the_named_action(tmp_path: Path) -> None:
    path = tmp_path / "hotkeys.json"
    path.write_text(json.dumps({"back": [["ctrl", "["]]}), encoding="utf-8")

    profile = load_profile(path, platform="win32")

    assert profile.chords_for(NavAction.BACK) == (("ctrl", "["),)
    assert profile.chords_for(NavAction.FORWARD) == WINDOWS_PROFILE.chords_for(NavAction.FORWARD)


def test_malformed_override_entries_are_skipped_not_fatal(tmp_path: Path) -> None:
    path = tmp_path / "hotkeys.json"
    path.write_text(
        json.dumps({"nonsense": [["a"]], "forward": "cmd+]", "back": ["browser_back"]}),
        encoding="utf-8",
    )

    profile = load_profile(path, platform="linux")

    assert profile.chords_for(NavAction.BACK) == (("browser_back",),)
    assert profile.chords_for(NavAction.FORWARD) == LINUX_PROFILE.chords_for(NavAction.FORWARD)


def test_broken_json_leaves_the_builtin_profile_intact(tmp_path: Path) -> None:
    path = tmp_path / "hotkeys.json"
    path.write_text("{not json", encoding="utf-8")

    assert load_profile(path, platform="darwin") is MACOS_PROFILE


def test_profile_is_immutable_data() -> None:
    profile = HotkeyProfile("t", {NavAction.BACK: (("alt", "left"),)})
    assert profile.chords_for(NavAction.FORWARD) == ()
