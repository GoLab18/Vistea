import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtCore import QRect
from PySide6.QtWidgets import QApplication

from core import research_log as rlog
from ui_gaze import GazeOverlay, KeyboardWindow

_app = QApplication.instance() or QApplication([])


class FakeScreen:
    def geometry(self):
        return QRect(0, 0, 1920, 1080)


@pytest.fixture(autouse=True)
def _mute_research_log(monkeypatch):
    # omit rlog logs (the research record)
    monkeypatch.setattr(rlog, "event", lambda *a, **k: None)


@pytest.fixture
def overlay():
    return GazeOverlay(FakeScreen())


# toasts

def test_notify_adds_a_toast(overlay):
    overlay.notify("hello", "info")
    assert len(overlay._toasts) == 1
    assert overlay._toasts[0][0] == "hello"
    assert overlay._toasts[0][1] == "info"


def test_notify_trims_to_3_most_recent(overlay):
    for i in range(5):
        overlay.notify(f"msg{i}")
    assert len(overlay._toasts) == 3
    texts = [t[0] for t in overlay._toasts]
    assert texts == ["msg2", "msg3", "msg4"]


def test_expired_toasts_are_pruned(overlay):
    overlay.notify("stale", ttl=3.5)
    assert len(overlay._toasts) == 1
    text, level, _expiry = overlay._toasts[0]
    overlay._toasts[0] = (text, level, 0.0) # force into the past
    overlay._prune_toasts()
    assert overlay._toasts == []


def test_fresh_toast_survives_prune(overlay):
    overlay.notify("fresh", ttl=3.5)
    overlay._prune_toasts()
    assert len(overlay._toasts) == 1


# KeyboardWindow.floor_unmet_changed signal

def test_floor_unmet_changed_emitted_on_state_change():
    kb = KeyboardWindow(FakeScreen(), 80)
    received = []
    kb.floor_unmet_changed.connect(received.append)
    kb.min_tile_px = 400
    kb._layout_tiles()
    assert True in received


def test_floor_unmet_changed_not_emitted_when_state_unchanged():
    kb = KeyboardWindow(FakeScreen(), 80)
    received = []
    kb.floor_unmet_changed.connect(received.append)
    kb._layout_tiles() # relayout without changing the floor state
    assert received == []
