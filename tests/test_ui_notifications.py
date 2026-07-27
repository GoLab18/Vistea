"""Tests for the gaze UI toast notifications and floor state signals."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtCore import QRect
from PySide6.QtWidgets import QApplication

from core import research_log as rlog
from ui_gaze import GazeOverlay, KeyboardWindow, ToastMessage

_app = QApplication.instance() or QApplication([])


class FakeScreen:
    def geometry(self) -> QRect:
        return QRect(0, 0, 1920, 1080)


@pytest.fixture(autouse=True)
def _mute_research_log(monkeypatch: pytest.MonkeyPatch) -> None:
    # omit rlog logs (the research record)
    monkeypatch.setattr(rlog, "event", lambda *a, **k: None)


@pytest.fixture
def overlay() -> GazeOverlay:
    return GazeOverlay(FakeScreen()) # type: ignore


def test_notify_adds_a_toast(overlay: GazeOverlay) -> None:
    overlay.notify("hello", "info")
    assert len(overlay._toasts) == 1
    assert overlay._toasts[0].text == "hello"
    assert overlay._toasts[0].level == "info"


def test_notify_trims_to_3_most_recent(overlay: GazeOverlay) -> None:
    for i in range(5):
        overlay.notify(f"msg{i}")
    assert len(overlay._toasts) == 3
    texts = [t.text for t in overlay._toasts]
    assert texts == ["msg2", "msg3", "msg4"]


def test_expired_toasts_are_pruned(overlay: GazeOverlay) -> None:
    overlay.notify("stale", ttl=3.5)
    assert len(overlay._toasts) == 1

    toast = overlay._toasts[0]
    overlay._toasts[0] = ToastMessage(text=toast.text, level=toast.level, expiry=0.0)

    overlay._prune_toasts()
    assert overlay._toasts == []


def test_fresh_toast_survives_prune(overlay: GazeOverlay) -> None:
    overlay.notify("fresh", ttl=3.5)
    overlay._prune_toasts()
    assert len(overlay._toasts) == 1


def test_floor_unmet_changed_emitted_on_state_change() -> None:
    kb = KeyboardWindow(FakeScreen(), 80) # type: ignore
    received = []
    kb.floor_unmet_changed.connect(received.append)
    kb.min_tile_px = 400
    kb._layout_tiles()
    assert True in received


def test_floor_unmet_changed_not_emitted_when_state_unchanged() -> None:
    kb = KeyboardWindow(FakeScreen(), 80) # type: ignore
    received = []
    kb.floor_unmet_changed.connect(received.append)
    kb._layout_tiles() # relayout without changing the floor state
    assert received == []