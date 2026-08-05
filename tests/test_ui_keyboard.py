"""Tests for the gaze keyboard UI: layouts, docking, scaling, and engine routing."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path
from typing import Final

import pytest
from PySide6.QtCore import QRect
from PySide6.QtWidgets import QApplication

from core import research_log as rlog
from core.actuator import NullActuator
from core.suggestions import SuggestionEngine
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
    SUGG_PREFIX,
    reservation_strip,
    to_physical,
)

_app = QApplication.instance() or QApplication([])


class FakeScreen:
    def geometry(self) -> QRect:
        return QRect(0, 0, 1920, 1080)


@pytest.fixture(autouse=True)
def _mute_research_log(monkeypatch: pytest.MonkeyPatch) -> None:
    # omit rlog logs (the research record)
    monkeypatch.setattr(rlog, "event", lambda *a, **k: None)


@pytest.fixture
def kb() -> KeyboardWindow:
    return KeyboardWindow(FakeScreen(), 80) # type: ignore


def test_en_layout_has_29_letter_keys_in_3_rows(kb: KeyboardWindow) -> None:
    assert kb.layout.code == "EN"
    ids = [t.tile_id for t in kb.tiles]
    assert len([i for i in ids if i.startswith(KEY_PREFIX)]) == 29
    assert len(kb.layout.letter_rows) == 3


def test_pl_layout_has_38_letter_keys_in_4_rows_with_diacritics_row(kb: KeyboardWindow) -> None:
    kb.cycle_language()
    assert kb.layout.code == "PL"
    ids = [t.tile_id for t in kb.tiles]
    assert len([i for i in ids if i.startswith(KEY_PREFIX)]) == 38
    assert len(kb.layout.letter_rows) == 4
    for ch in "ąćęłńóśźż":
        assert f"{KEY_PREFIX}{ch}" in ids


def test_language_switching_is_cyclic(kb: KeyboardWindow) -> None:
    assert kb.layout.code == "EN"
    kb.cycle_language()
    assert kb.layout.code == "PL"
    kb.cycle_language()
    assert kb.layout.code == "EN"


def test_toggle_symbols_switches_page_and_back(kb: KeyboardWindow) -> None:
    assert kb.page is Page.LETTERS
    kb.toggle_symbols()
    assert kb.page is Page.SYMBOLS
    ids = [t.tile_id for t in kb.tiles]
    assert f"{KEY_PREFIX}7" in ids
    kb.toggle_symbols()
    assert kb.page is Page.LETTERS


def test_toggle_options_switches_page_and_back(kb: KeyboardWindow) -> None:
    kb.toggle_options()
    assert kb.page is Page.OPTIONS
    kb.toggle_options()
    assert kb.page is Page.LETTERS


def test_control_row_present_on_every_page(kb: KeyboardWindow) -> None:
    control_ids = {tid for tid, _, _ in kb._control_specs()}
    for page in (Page.LETTERS, Page.SYMBOLS, Page.OPTIONS):
        kb.switch_page(page)
        ids = {t.tile_id for t in kb.tiles}
        assert control_ids <= ids, page


@pytest.mark.parametrize("side", list(DockSide))
def test_tiles_fit_within_screen_geometry_for_every_dock(kb: KeyboardWindow, side: DockSide) -> None:
    geo = FakeScreen().geometry()
    kb.set_dock(side)
    for t in kb.tiles:
        r = t.rect
        assert r.x() >= geo.x() - 1 and r.y() >= geo.y() - 1
        assert r.x() + r.width() <= geo.x() + geo.width() + 1
        assert r.y() + r.height() <= geo.y() + geo.height() + 1


def test_side_dock_width_stays_within_caps_and_scale_responds(kb: KeyboardWindow) -> None:
    kb.set_dock(DockSide.RIGHT)

    geo = kb.geometry()
    assert geo.x() + geo.width() <= 1920 + 1
    assert geo.width() <= 1920 - 20 + 1

    w0 = kb.geometry().width()
    kb.resize_step(+1)
    assert kb.geometry().width() > w0
    kb.resize_step(-1)
    kb.resize_step(-1)
    assert kb.geometry().width() < w0


def test_dock_top_and_bottom_touch_expected_edges(kb: KeyboardWindow) -> None:
    kb.set_dock(DockSide.TOP)
    assert kb.geometry().y() == 10
    kb.set_dock(DockSide.BOTTOM)
    assert kb.geometry().y() + kb.geometry().height() >= 1080 - 20


@pytest.mark.parametrize("page", [Page.LETTERS, Page.SYMBOLS, Page.OPTIONS])
@pytest.mark.parametrize("side", list(DockSide))
def test_huge_floor_still_fits_screen_for_every_page_and_dock(page: Page, side: DockSide) -> None:
    geo = FakeScreen().geometry()
    kb = KeyboardWindow(FakeScreen(), 400) # type: ignore
    kb.switch_page(page)
    kb.set_dock(side)
    for t in kb.tiles:
        r = t.rect
        assert r.x() >= geo.x() - 1 and r.y() >= geo.y() - 1
        assert r.x() + r.width() <= geo.x() + geo.width() + 1
        assert r.y() + r.height() <= geo.y() + geo.height() + 1


def test_repeated_resize_up_still_fits_screen(kb: KeyboardWindow) -> None:
    geo = FakeScreen().geometry()
    for _ in range(50):
        kb.resize_step(+1)
    for t in kb.tiles:
        r = t.rect
        assert r.x() >= geo.x() - 1 and r.y() >= geo.y() - 1
        assert r.x() + r.width() <= geo.x() + geo.width() + 1
        assert r.y() + r.height() <= geo.y() + geo.height() + 1


def test_browser_bar_huge_floor_fits_screen_vertically() -> None:
    geo = FakeScreen().geometry()
    bar = BrowserBar(FakeScreen(), 400) # type: ignore
    for t in bar.tiles:
        r = t.rect
        assert r.y() >= geo.y() - 1
        assert r.y() + r.height() <= geo.y() + geo.height() + 1


def test_resize_step_clamps_to_scale_bounds(kb: KeyboardWindow) -> None:
    w0 = kb.geometry().width()
    kb.resize_step(+1)
    assert kb.geometry().width() > w0
    for _ in range(50):
        kb.resize_step(+1)
    assert kb.scale == kb.SCALE_MAX
    for _ in range(50):
        kb.resize_step(-1)
    assert kb.scale == kb.SCALE_MIN


def test_shrinking_stays_above_abs_min_and_flags_floor_unmet(kb: KeyboardWindow) -> None:
    for _ in range(50):
        kb.resize_step(-1)
    for t in kb.tiles:
        assert min(t.rect.width(), t.rect.height()) >= kb.TILE_ABS_MIN
    smallest = min(min(t.rect.width(), t.rect.height()) for t in kb.tiles)
    if smallest < kb.min_tile_px:
        assert kb.floor_unmet is True


def test_scale_responsive_even_with_huge_accuracy_floor() -> None:
    kb = KeyboardWindow(FakeScreen(), 400) # type: ignore
    w_before = kb.geometry().width()
    kb.resize_step(-1)
    assert kb.geometry().width() < w_before


def test_default_bottom_dock_width_is_at_most_55_percent() -> None:
    kb = KeyboardWindow(FakeScreen(), 80) # type: ignore
    kb.set_dock(DockSide.BOTTOM)
    assert kb.geometry().width() <= int(1920 * 0.55)


def test_opacity_step_clamps_to_bounds(kb: KeyboardWindow) -> None:
    for _ in range(50):
        kb.opacity_step(-1)
    assert abs(kb.opacity_level - kb.OPACITY_MIN) < 1e-9
    for _ in range(50):
        kb.opacity_step(+1)
    assert kb.opacity_level == 1.0


def test_lock_restricts_hit_test_to_unlock_tile(kb: KeyboardWindow) -> None:
    sp = next(t for t in kb.tiles if t.tile_id == Ctl.SPACE).rect.center()
    assert kb.tile_hit_test(sp.x(), sp.y()) == Ctl.SPACE
    kb.set_locked(True)
    assert kb.tile_hit_test(sp.x(), sp.y()) is None
    uc = kb._unlock_tile.rect.center()
    assert kb.tile_hit_test(uc.x(), uc.y()) == Ctl.UNLOCK


def test_unlock_restores_normal_hit_test(kb: KeyboardWindow) -> None:
    kb.set_locked(True)
    kb.set_locked(False)
    sp = next(t for t in kb.tiles if t.tile_id == Ctl.SPACE).rect.center()
    assert kb.tile_hit_test(sp.x(), sp.y()) == Ctl.SPACE


@pytest.fixture
def mode(kb: KeyboardWindow) -> KeyboardMode:
    return KeyboardMode(kb, None, NullActuator())


def test_letter_tile_types_the_character(kb: KeyboardWindow, mode: KeyboardMode) -> None:
    mode.on_activation(f"{KEY_PREFIX}a")
    assert kb.buffer == "a"
    assert ("type_text", "a") in mode._actuator.calls


def test_space_types_a_space(kb: KeyboardWindow, mode: KeyboardMode) -> None:
    mode.on_activation(Ctl.SPACE)
    assert kb.buffer == " "
    assert ("type_text", " ") in mode._actuator.calls


def test_backspace_presses_backspace_key(kb: KeyboardWindow, mode: KeyboardMode) -> None:
    mode.on_activation(f"{KEY_PREFIX}a")
    mode.on_activation(Ctl.BACKSPACE)
    assert kb.buffer == ""
    assert ("press_key", "backspace") in mode._actuator.calls


def test_delete_word_sends_ctrl_backspace_hotkey(kb: KeyboardWindow, mode: KeyboardMode) -> None:
    mode.on_activation(f"{KEY_PREFIX}a")
    mode.on_activation(Ctl.SPACE)
    mode.on_activation(f"{KEY_PREFIX}b")
    mode.on_activation(Ctl.DELETE_WORD)
    assert kb.buffer == "a "
    assert ("hotkey", "ctrl", "backspace") in mode._actuator.calls


def test_enter_commits_preview_and_presses_enter(kb: KeyboardWindow, mode: KeyboardMode) -> None:
    mode.on_activation(f"{KEY_PREFIX}a")
    mode.on_activation(Ctl.SPACE)
    assert kb.buffer == "a "

    ret = mode.on_activation(Ctl.ENTER)

    assert ret is None
    assert kb.buffer == ""
    assert ("press_key", "enter") in mode._actuator.calls


def test_lock_blocks_activation_until_unlocked(kb: KeyboardWindow, mode: KeyboardMode) -> None:
    kb.set_locked(True)
    assert mode.on_activation(f"{KEY_PREFIX}a") is None
    assert mode._actuator.calls == []
    mode.on_activation(Ctl.UNLOCK)
    assert kb.locked is False
    mode.on_activation(f"{KEY_PREFIX}a")
    assert ("type_text", "a") in mode._actuator.calls


def test_nav_and_exit_return_mode_actions(kb: KeyboardWindow, mode: KeyboardMode) -> None:
    assert mode.on_activation(Ctl.NAV) is ModeAction.TO_NAV
    assert mode.on_activation(Ctl.EXIT) is ModeAction.EXIT


def test_pin_tile_present_on_options_page(kb: KeyboardWindow) -> None:
    kb.switch_page(Page.OPTIONS)
    assert Ctl.PIN in {t.tile_id for t in kb.tiles}


def test_recal_tile_present_on_options_page(kb: KeyboardWindow) -> None:
    kb.switch_page(Page.OPTIONS)
    assert Ctl.RECAL in {t.tile_id for t in kb.tiles}


def test_recal_tile_returns_recal_action(kb: KeyboardWindow, mode: KeyboardMode) -> None:
    kb.switch_page(Page.OPTIONS)
    assert mode.on_activation(Ctl.RECAL) is ModeAction.RECAL


def test_pin_tile_toggles_pinned_and_emits_signal(kb: KeyboardWindow, mode: KeyboardMode) -> None:
    got: list[bool] = []
    kb.pin_toggled.connect(got.append)
    assert kb.pinned is False
    mode.on_activation(Ctl.PIN)
    assert kb.pinned is True
    assert got == [True]
    mode.on_activation(Ctl.PIN)
    assert kb.pinned is False
    assert got == [True, False]


SCREEN: Final[tuple[int, int, int, int]] = (0, 0, 1920, 1080)


def test_reservation_strip_bottom_dock_full_width_band_at_dpr_1_5() -> None:
    kb_geo = (400, 900, 1120, 170)
    edge, (x, y, w, h) = reservation_strip(DockSide.BOTTOM, kb_geo, None, SCREEN, 1.5)
    assert edge == "bottom"
    assert (x, w) == (0, round(1920 * 1.5))
    assert h == round(190 * 1.5)
    assert y == round((1080 - 190) * 1.5)


def test_reservation_strip_top_dock_full_width_band() -> None:
    kb_geo = (400, 10, 1120, 180)
    edge, (x, y, w, h) = reservation_strip(DockSide.TOP, kb_geo, None, SCREEN, 1.5)
    assert edge == "top"
    assert (x, y, w) == (0, 0, round(1920 * 1.5))
    assert h == round(200 * 1.5)


def test_reservation_strip_left_dock_full_height_band() -> None:
    kb_geo = (10, 300, 500, 480)
    edge, (x, y, w, h) = reservation_strip(DockSide.LEFT, kb_geo, None, SCREEN, 1.5)
    assert edge == "left"
    assert (x, y, h) == (0, 0, round(1080 * 1.5))
    assert w == round(520 * 1.5)


def test_reservation_strip_right_dock_full_height_band() -> None:
    kb_geo = (1410, 300, 500, 480)
    edge, (x, y, w, h) = reservation_strip(DockSide.RIGHT, kb_geo, None, SCREEN, 1.5)
    assert edge == "right"
    assert (y, h) == (0, round(1080 * 1.5))
    assert w == round(520 * 1.5)
    assert x == round((1920 - 520) * 1.5)


def test_reservation_strip_unions_suggestion_geometry() -> None:
    kb_geo = (400, 900, 1120, 170)
    sugg_geo = (400, 840, 1120, 55)
    _, (_, _, _, h_no) = reservation_strip(DockSide.BOTTOM, kb_geo, None, SCREEN, 1.0)
    _, (_, _, _, h_yes) = reservation_strip(DockSide.BOTTOM, kb_geo, sugg_geo, SCREEN, 1.0)
    assert h_yes > h_no


def test_preview_buffer_type_backspace_delete_word_clear(kb: KeyboardWindow) -> None:
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


def test_to_physical_is_identity_at_dpr_1() -> None:
    assert to_physical(100, 101, 1.0) == (100, 101)


def test_to_physical_scales_and_rounds_at_dpr_1_5() -> None:
    assert to_physical(100, 101, 1.5) == (150, 152)


def test_page_scroll_parks_cursor_at_screen_center_first() -> None:
    act = NullActuator()
    mode = NavigationMode(BrowserBar(FakeScreen(), 80), act, dpr=1.5) # type: ignore
    mode.on_activation(Ctl.PAGE_DOWN)
    assert act.calls[0] == ("move", 1440, 810) 
    assert act.calls[1] == ("scroll", -12)


@pytest.fixture
def sugg_mode(kb: KeyboardWindow, tmp_path: Path) -> KeyboardMode:
    dict_dir = tmp_path / "dict"
    dict_dir.mkdir()
    (dict_dir / "en_50k.txt").write_text("the 100\nthen 80\nthere 60\n", encoding="utf-8")
    m = KeyboardMode(kb, None, NullActuator())
    m.engine = SuggestionEngine(dict_dir, tmp_path / "custom.json")
    return m


def test_suggestion_bar_populates_on_letters_page(kb: KeyboardWindow, sugg_mode: KeyboardMode) -> None:
    kb.type_preview("the")
    sugg_mode._refresh_suggestions()

    assert sugg_mode.suggestions_board is not None
    labels = [t.label for t in sugg_mode.suggestions_board.tiles]
    assert "then" in labels and "there" in labels
    assert "DICT+" in labels


def test_accepting_suggestion_replaces_prefix_in_field_and_buffer(kb: KeyboardWindow, sugg_mode: KeyboardMode) -> None:
    for ch in "the":
        sugg_mode.on_activation(f"{KEY_PREFIX}{ch}")
    act = sugg_mode._actuator
    act.calls.clear()
    sugg_mode.on_activation(f"{SUGG_PREFIX}0") 

    assert act.calls.count(("press_key", "backspace")) == 3
    assert ("type_text", "then ") in act.calls
    assert kb.buffer == "then "


def test_add_word_saves_current_word_to_custom(kb: KeyboardWindow, sugg_mode: KeyboardMode, tmp_path: Path) -> None:
    for ch in "hello":
        kb.type_preview(ch)
    sugg_mode.on_activation(Ctl.ADD_WORD)
    reloaded = SuggestionEngine(tmp_path / "dict", tmp_path / "custom.json")
    assert [s.word for s in reloaded.suggest("hel", "EN", k=3)] == ["hello"]


def test_add_word_saves_whole_buffer_when_multiword(kb: KeyboardWindow, sugg_mode: KeyboardMode, tmp_path: Path) -> None:
    for ch in "hi there":
        kb.type_preview(ch)
    sugg_mode.on_activation(Ctl.ADD_WORD)
    reloaded = SuggestionEngine(tmp_path / "dict", tmp_path / "custom.json")
    assert reloaded.suggest("hi", "EN", k=3)[0].word == "hi there"


def test_suggestion_bar_hidden_off_letters_page(kb: KeyboardWindow, sugg_mode: KeyboardMode) -> None:
    kb.switch_page(Page.SYMBOLS)
    sugg_mode._refresh_suggestions()
    assert sugg_mode.suggestions_board is not None
    assert not sugg_mode.suggestions_board.isVisible()


def test_lock_also_covers_the_suggestion_strip(kb: KeyboardWindow, sugg_mode: KeyboardMode) -> None:
    kb.type_preview("the")
    sugg_mode._refresh_suggestions()
    bar = sugg_mode.suggestions_board
    assert bar is not None and bar.isVisible()

    sugg_mode.on_activation(Ctl.LOCK)
    assert not bar.isVisible()

    act = sugg_mode._actuator
    act.calls.clear()
    sugg_mode.on_activation(f"{SUGG_PREFIX}0")
    assert act.calls == [], "a locked keyboard must not type through its suggestions"

    sugg_mode.on_activation(Ctl.UNLOCK)
    assert bar.isVisible()


def test_suggestion_tiles_are_hit_testable(kb: KeyboardWindow, sugg_mode: KeyboardMode) -> None:
    kb.type_preview("the")
    sugg_mode._refresh_suggestions()
    bar = sugg_mode.suggestions_board

    assert bar is not None
    tile = next(t for t in bar.tiles if t.tile_id == f"{SUGG_PREFIX}0")
    c = tile.rect.center()
    assert bar.tile_hit_test(c.x(), c.y()) == f"{SUGG_PREFIX}0"


def test_reserved_suggestion_row_keeps_board_plus_strip_on_screen(kb: KeyboardWindow) -> None:
    geo = FakeScreen().geometry()
    kb.cycle_language()
    assert kb.layout.code == "PL"
    kb.reserve_suggestion_row = True
    kb.relayout()
    tile = next(t for t in kb.tiles if t.tile_id == Ctl.SPACE).rect.height()
    board = kb.geometry()
    assert board.y() >= geo.y()
    assert board.y() + board.height() <= geo.y() + geo.height() + 1
    assert board.height() + tile <= geo.height() + 1


def test_suggestion_bar_never_extends_past_screen_edges(kb: KeyboardWindow, sugg_mode: KeyboardMode) -> None:
    geo = FakeScreen().geometry()
    kb.cycle_language() 
    kb.set_dock(DockSide.TOP)
    kb.type_preview("the")
    sugg_mode._refresh_suggestions()

    assert sugg_mode.suggestions_board is not None
    r = sugg_mode.suggestions_board.geometry()
    assert r.x() >= geo.x()
    assert r.y() >= geo.y()
    assert r.x() + r.width() <= geo.x() + geo.width() + 1
    assert r.y() + r.height() <= geo.y() + geo.height() + 1