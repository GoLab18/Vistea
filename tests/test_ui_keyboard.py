import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtCore import QRect
from PySide6.QtWidgets import QApplication

from core import research_log as rlog
from core.actuator import NullActuator
from ui_gaze import (
    BrowserBar,
    Ctl,
    DockSide,
    KEY_PREFIX,
    KeyboardMode,
    KeyboardWindow,
    ModeAction,
    NavigationMode,
    Page,
    to_physical,
)

_app = QApplication.instance() or QApplication([])


class FakeScreen:
    def geometry(self):
        return QRect(0, 0, 1920, 1080)


@pytest.fixture(autouse=True)
def _mute_research_log(monkeypatch):
    # omit rlog logs (the research record)
    monkeypatch.setattr(rlog, "event", lambda *a, **k: None)


@pytest.fixture
def kb():
    return KeyboardWindow(FakeScreen(), 80)


# language layouts

def test_en_layout_has_29_letter_keys_in_3_rows(kb):
    assert kb.layout.code == "EN"
    ids = [t.tile_id for t in kb.tiles]
    assert len([i for i in ids if i.startswith(KEY_PREFIX)]) == 29
    assert len(kb.layout.letter_rows) == 3


def test_pl_layout_has_38_letter_keys_in_4_rows_with_diacritics_row(kb):
    kb.cycle_language()
    assert kb.layout.code == "PL"
    ids = [t.tile_id for t in kb.tiles]
    assert len([i for i in ids if i.startswith(KEY_PREFIX)]) == 38
    assert len(kb.layout.letter_rows) == 4
    for ch in "ąćęłńóśźż":
        assert f"{KEY_PREFIX}{ch}" in ids


def test_language_switching_is_cyclic(kb):
    assert kb.layout.code == "EN"
    kb.cycle_language()
    assert kb.layout.code == "PL"
    kb.cycle_language()
    assert kb.layout.code == "EN" # wraps back around


# pages

def test_toggle_symbols_switches_page_and_back(kb):
    assert kb.page is Page.LETTERS
    kb.toggle_symbols()
    assert kb.page is Page.SYMBOLS
    ids = [t.tile_id for t in kb.tiles]
    assert f"{KEY_PREFIX}7" in ids
    kb.toggle_symbols()
    assert kb.page is Page.LETTERS


def test_toggle_options_switches_page_and_back(kb):
    kb.toggle_options()
    assert kb.page is Page.OPTIONS
    kb.toggle_options()
    assert kb.page is Page.LETTERS


def test_control_row_present_on_every_page(kb):
    control_ids = {tid for tid, _, _ in kb._control_specs()}
    for page in (Page.LETTERS, Page.SYMBOLS, Page.OPTIONS):
        kb.switch_page(page)
        ids = {t.tile_id for t in kb.tiles}
        assert control_ids <= ids, page


# docking

@pytest.mark.parametrize("side", list(DockSide))
def test_tiles_fit_within_screen_geometry_for_every_dock(kb, side):
    geo = FakeScreen().geometry()
    kb.set_dock(side)
    for t in kb.tiles:
        r = t.rect
        assert r.x() >= geo.x() - 1 and r.y() >= geo.y() - 1
        assert r.x() + r.width() <= geo.x() + geo.width() + 1
        assert r.y() + r.height() <= geo.y() + geo.height() + 1


def test_side_dock_width_stays_within_caps_and_scale_responds(kb):
    kb.set_dock(DockSide.RIGHT)

    # The board width is bounded by the screen cap (never the advisory floor)
    geo = kb.geometry()
    assert geo.x() + geo.width() <= 1920 + 1
    assert geo.width() <= 1920 - 20 + 1

    # Scale is responsive: growing then shrinking changes the width both ways
    w0 = kb.geometry().width()
    kb.resize_step(+1)
    assert kb.geometry().width() > w0
    kb.resize_step(-1)
    kb.resize_step(-1)
    assert kb.geometry().width() < w0


def test_dock_top_and_bottom_touch_expected_edges(kb):
    kb.set_dock(DockSide.TOP)
    assert kb.geometry().y() == 10
    kb.set_dock(DockSide.BOTTOM)
    assert kb.geometry().y() + kb.geometry().height() >= 1080 - 20


# hard screen cap (huge floor from a poor calibration)

@pytest.mark.parametrize("page", [Page.LETTERS, Page.SYMBOLS, Page.OPTIONS])
@pytest.mark.parametrize("side", list(DockSide))
def test_huge_floor_still_fits_screen_for_every_page_and_dock(page, side):
    geo = FakeScreen().geometry()
    kb = KeyboardWindow(FakeScreen(), 400)
    kb.switch_page(page)
    kb.set_dock(side)
    for t in kb.tiles:
        r = t.rect
        assert r.x() >= geo.x() - 1 and r.y() >= geo.y() - 1
        assert r.x() + r.width() <= geo.x() + geo.width() + 1
        assert r.y() + r.height() <= geo.y() + geo.height() + 1


def test_repeated_resize_up_still_fits_screen(kb):
    geo = FakeScreen().geometry()
    for _ in range(50):
        kb.resize_step(+1)
    for t in kb.tiles:
        r = t.rect
        assert r.x() >= geo.x() - 1 and r.y() >= geo.y() - 1
        assert r.x() + r.width() <= geo.x() + geo.width() + 1
        assert r.y() + r.height() <= geo.y() + geo.height() + 1


def test_browser_bar_huge_floor_fits_screen_vertically():
    geo = FakeScreen().geometry()
    bar = BrowserBar(FakeScreen(), 400)
    for t in bar.tiles:
        r = t.rect
        assert r.y() >= geo.y() - 1
        assert r.y() + r.height() <= geo.y() + geo.height() + 1


# scale and opacity

def test_resize_step_clamps_to_scale_bounds(kb):
    w0 = kb.geometry().width()
    kb.resize_step(+1)
    assert kb.geometry().width() > w0
    for _ in range(50):
        kb.resize_step(+1)
    assert kb.scale == kb.SCALE_MAX
    for _ in range(50):
        kb.resize_step(-1)
    assert kb.scale == kb.SCALE_MIN


def test_shrinking_stays_above_abs_min_and_flags_floor_unmet(kb):
    # kb fixture uses min_tile_px=80. Shrinking to SCALE_MIN drops the tile
    # below that advisory floor (so _floor_unmet flips True) but never below
    # the absolute pointer-target minimum.
    for _ in range(50):
        kb.resize_step(-1)
    for t in kb.tiles:
        assert min(t.rect.width(), t.rect.height()) >= kb.TILE_ABS_MIN
    smallest = min(min(t.rect.width(), t.rect.height()) for t in kb.tiles)
    if smallest < kb.min_tile_px:
        assert kb._floor_unmet is True


def test_scale_responsive_even_with_huge_accuracy_floor():
    # The live-session bug: with a huge floor (400 px) SIZE- did nothing at
    # all. Now the advisory floor no longer clamps, so a resize step down
    # yields a strictly SMALLER board.
    kb = KeyboardWindow(FakeScreen(), 400)
    w_before = kb.geometry().width()
    kb.resize_step(-1)
    assert kb.geometry().width() < w_before


def test_default_bottom_dock_width_is_at_most_55_percent():
    kb = KeyboardWindow(FakeScreen(), 80)
    kb.set_dock(DockSide.BOTTOM)
    assert kb.geometry().width() <= int(1920 * 0.55)


def test_opacity_step_clamps_to_bounds(kb):
    for _ in range(50):
        kb.opacity_step(-1)
    assert abs(kb.opacity_level - kb.OPACITY_MIN) < 1e-9
    for _ in range(50):
        kb.opacity_step(+1)
    assert kb.opacity_level == 1.0


# lock

def test_lock_restricts_hit_test_to_unlock_tile(kb):
    sp = next(t for t in kb.tiles if t.tile_id == Ctl.SPACE).rect.center()
    assert kb.hit_test(sp.x(), sp.y()) == Ctl.SPACE
    kb.set_locked(True)
    assert kb.hit_test(sp.x(), sp.y()) is None
    uc = kb._unlock_tile.rect.center()
    assert kb.hit_test(uc.x(), uc.y()) == Ctl.UNLOCK


def test_unlock_restores_normal_hit_test(kb):
    kb.set_locked(True)
    kb.set_locked(False)
    sp = next(t for t in kb.tiles if t.tile_id == Ctl.SPACE).rect.center()
    assert kb.hit_test(sp.x(), sp.y()) == Ctl.SPACE


# routing KeyboardMode -> actuator

@pytest.fixture
def mode(kb):
    return KeyboardMode(kb, None, NullActuator())


def test_letter_tile_types_the_character(kb, mode):
    mode.on_activation(f"{KEY_PREFIX}a")
    assert kb.buffer == "a"
    assert ("type_text", "a") in mode._actuator.calls


def test_space_types_a_space(kb, mode):
    mode.on_activation(Ctl.SPACE)
    assert kb.buffer == " "
    assert ("type_text", " ") in mode._actuator.calls


def test_backspace_presses_backspace_key(kb, mode):
    mode.on_activation(f"{KEY_PREFIX}a")
    mode.on_activation(Ctl.BACKSPACE)
    assert kb.buffer == ""
    assert ("press_key", "backspace") in mode._actuator.calls


def test_delete_word_sends_ctrl_backspace_hotkey(kb, mode):
    mode.on_activation(f"{KEY_PREFIX}a")
    mode.on_activation(Ctl.SPACE)
    mode.on_activation(f"{KEY_PREFIX}b")
    mode.on_activation(Ctl.DELETE_WORD)
    assert kb.buffer == "a "
    assert ("hotkey", "ctrl", "backspace") in mode._actuator.calls


def test_enter_commits_preview_and_presses_enter(kb, mode):
    got = []
    kb.committed.connect(got.append)
    mode.on_activation(f"{KEY_PREFIX}a")
    mode.on_activation(Ctl.SPACE)
    ret = mode.on_activation(Ctl.ENTER)
    assert ret is None
    assert kb.buffer == "" # committed and reset
    assert got == ["a "] # committed signal carries the preview content
    assert ("press_key", "enter") in mode._actuator.calls


def test_lock_blocks_activation_until_unlocked(kb, mode):
    kb.set_locked(True)
    assert mode.on_activation(f"{KEY_PREFIX}a") is None
    assert mode._actuator.calls == []
    mode.on_activation(Ctl.UNLOCK)
    assert kb.locked is False
    mode.on_activation(f"{KEY_PREFIX}a")
    assert ("type_text", "a") in mode._actuator.calls


def test_nav_and_exit_return_mode_actions(kb, mode):
    assert mode.on_activation(Ctl.NAV) is ModeAction.TO_NAV
    assert mode.on_activation(Ctl.EXIT) is ModeAction.EXIT


# local preview buffer

def test_preview_buffer_type_backspace_delete_word_clear(kb):
    kb.type_preview("a")
    kb.type_preview(" ")
    kb.type_preview("b")
    assert kb.buffer == "a b"
    kb.backspace_preview()
    assert kb.buffer == "a "
    kb.type_preview("c")
    kb.delete_word_preview()
    assert kb.buffer == "a "
    kb.clear_preview()
    assert kb.buffer == ""


# logical -> physical pixel conversion (DPI scaling)

def test_to_physical_is_identity_at_dpr_1():
    assert to_physical(100, 101, 1.0) == (100, 101)


def test_to_physical_scales_and_rounds_at_dpr_1_5():
    assert to_physical(100, 101, 1.5) == (150, 152)


# navigation mode page scrolling

def test_page_scroll_parks_cursor_at_screen_centre_first():
    act = NullActuator()
    mode = NavigationMode(BrowserBar(FakeScreen(), 80), act, dpr=1.5)
    mode.on_activation(Ctl.PAGE_DOWN)
    assert act.calls[0] == ("move", 1440, 810) # centre (960, 540) x 1.5
    assert act.calls[1] == ("scroll", -12)