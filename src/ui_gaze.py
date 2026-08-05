"""Gaze interaction layer: overlay, tile boards, modes."""

from __future__ import annotations

import logging
import pathlib
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Final, Literal, Protocol, assert_never

from PySide6.QtCore import QByteArray, QObject, QPoint, QRect, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QPaintEvent, QPainter, QPen, QScreen
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtWidgets import QWidget

from core import research_log as rlog
from core.activation import ActivationStrategy, extract_blink_score
from core.actuator import SystemActuator
from core.calibration import CalibrationStrategy, NormalizedPoint
from core.drift import (
    DriftCorrector,
    PoseResidualCorrector,
    RecalibrationSession,
    RecalPhase,
    min_tile_side_px,
)
from core.fixation import FixationState, IDTFixationDetector
from core.gestures import (
    SCROLL_DOWN,
    SCROLL_NONE,
    SCROLL_UP,
    EdgeScrollDetector,
    ScrollDirection,
    zone_from_error,
)
from core.hotkeys import HotkeyProfile, NavAction, load_profile
from core.hotkeys import send as send_hotkey
from core.suggestions import Suggestion, SuggestionEngine
from core.vision import LandmarksData
from core.workarea import DockSide, WorkAreaReservation, make_reservation

ACCENT: Final[QColor] = QColor(41, 128, 185)
ACCENT_ACTIVE: Final[QColor] = QColor(46, 204, 113)
TILE_BG: Final[QColor] = QColor(30, 30, 35, 235)
TILE_TEXT: Final[QColor] = QColor(240, 240, 240)
LOCK_DIM: Final[QColor] = QColor(10, 10, 12, 170)

type TileId = str

KEY_PREFIX: Final[str] = "key:"   # character keys: "key:<char>"
CTL_PREFIX: Final[str] = "ctl:"   # every control tile, i.e. every `Ctl` member
SUGG_PREFIX: Final[str] = "sugg:" # suggestion tiles: "sugg:<index>"

SCREEN_CLICK: Final[TileId] = "screen:click" # pseudo-target: bare screen (nav mode)


def to_physical(x: int, y: int, dpr: float) -> tuple[int, int]:
    """Convert Qt logical-pixel coordinates to physical-pixel coordinates."""

    return round(x * dpr), round(y * dpr)


def _geo_tuple(rect: QRect) -> tuple[int, int, int, int]:
    """Flatten a Qt QRect to a plain (x, y, w, h) tuple."""

    return (rect.x(), rect.y(), rect.width(), rect.height())


# Reserved strip px margin
PIN_MARGIN: Final[int] = 10


def reservation_strip(
    dock: DockSide,
    kb_geo: tuple[int, int, int, int],
    sugg_geo: tuple[int, int, int, int] | None,
    screen_geo: tuple[int, int, int, int],
    dpr: float,
) -> tuple[DockSide, tuple[int, int, int, int]]:
    """The desktop strip to reserve for the pinned keyboard,
    with suggestion strip considered, in physical pixels."""

    kx, ky, kw, kh = kb_geo
    left, top = kx, ky
    right, bottom = kx + kw, ky + kh

    if sugg_geo is not None:
        sx, sy, sw, sh = sugg_geo
        left = min(left, sx)
        top = min(top, sy)
        right = max(right, sx + sw)
        bottom = max(bottom, sy + sh)

    scr_x, scr_y, scr_w, scr_h = screen_geo

    match dock:
        case DockSide.TOP:
            thickness = (bottom - scr_y) + PIN_MARGIN
            strip = (scr_x, scr_y, scr_w, thickness)
        case DockSide.BOTTOM:
            thickness = (scr_y + scr_h - top) + PIN_MARGIN
            strip = (scr_x, scr_y + scr_h - thickness, scr_w, thickness)
        case DockSide.LEFT:
            thickness = (right - scr_x) + PIN_MARGIN
            strip = (scr_x, scr_y, thickness, scr_h)
        case DockSide.RIGHT:
            thickness = (scr_x + scr_w - left) + PIN_MARGIN
            strip = (scr_x + scr_w - thickness, scr_y, thickness, scr_h)
        case _:
            assert_never(dock)

    px, py = to_physical(strip[0], strip[1], dpr)
    pw, ph = to_physical(strip[2], strip[3], dpr)
    return dock, (px, py, pw, ph)


class Ctl(str, Enum):
    """Identifiers of all non-character tiles."""

    # Keyboard control row

    OPTIONS = "ctl:options"
    LANG = "ctl:lang"
    PAGE = "ctl:page"
    SPACE = "ctl:space"
    BACKSPACE = "ctl:bksp"
    DELETE_WORD = "ctl:delword"
    ENTER = "ctl:enter"
    NAV = "ctl:nav"
    EXIT = "ctl:exit"
    ADD_WORD = "ctl:addword"

    # Options page

    DOCK_TOP = "ctl:dock_top"
    DOCK_BOTTOM = "ctl:dock_bottom"
    DOCK_LEFT = "ctl:dock_left"
    DOCK_RIGHT = "ctl:dock_right"
    SIZE_DOWN = "ctl:size_down"
    SIZE_UP = "ctl:size_up"
    OPACITY_DOWN = "ctl:opacity_down"
    OPACITY_UP = "ctl:opacity_up"
    CLEAR = "ctl:clear"
    LOCK = "ctl:lock"
    UNLOCK = "ctl:unlock"
    PIN = "ctl:pin"
    RECAL = "ctl:recal"

    # Browser bar

    BACK = "ctl:back"
    FORWARD = "ctl:fwd"
    PAGE_UP = "ctl:pgup"
    PAGE_DOWN = "ctl:pgdn"
    KEYBOARD = "ctl:kbd"
    SCROLL_ZONES = "ctl:scrollzones"
    BAR_SIDE = "ctl:barside"


ICON_FILES: Final[dict[TileId, str]] = {
    Ctl.BACK: "arrow-left",
    Ctl.FORWARD: "arrow-right",
    Ctl.PAGE_UP: "chevrons-up",
    Ctl.PAGE_DOWN: "chevrons-down",
    Ctl.BACKSPACE: "delete",
    Ctl.ENTER: "corner-down-left",
    Ctl.EXIT: "x",
    Ctl.KEYBOARD: "keyboard",
    Ctl.NAV: "mouse-pointer",
    Ctl.DOCK_TOP: "arrow-up-to-line",
    Ctl.DOCK_BOTTOM: "arrow-down-to-line",
    Ctl.DOCK_LEFT: "arrow-left-to-line",
    Ctl.DOCK_RIGHT: "arrow-right-to-line",
    Ctl.LOCK: "lock",
    Ctl.UNLOCK: "lock-open",
    Ctl.PIN: "pin",
    Ctl.OPTIONS: "settings",
    Ctl.CLEAR: "eraser",
    Ctl.SIZE_UP: "zoom-in",
    Ctl.SIZE_DOWN: "zoom-out",
    Ctl.RECAL: "target",
}

ICONS_DIR: Final[pathlib.Path] = pathlib.Path(__file__).resolve().parent.parent / "assets" / "icons"

_ICON_RENDERERS: dict[TileId, QSvgRenderer | None] = {}


def _icon_renderer(tile_id: TileId) -> QSvgRenderer | None:
    """Lazily build (and cache) a QSvgRenderer for an icon."""

    if tile_id in _ICON_RENDERERS:
        return _ICON_RENDERERS[tile_id]
    renderer = None
    name = ICON_FILES.get(tile_id)
    if name is not None:
        try:
            svg_path = ICONS_DIR / f"{name}.svg"
            data = svg_path.read_text(encoding="utf-8")
            data = data.replace("currentColor", "#f0f0f0")

            candidate = QSvgRenderer(QByteArray(data.encode("utf-8")))
            if candidate.isValid():
                renderer = candidate
        except (OSError, ImportError):
            renderer = None

    _ICON_RENDERERS[tile_id] = renderer
    return renderer


class ModeAction(Enum):
    """Tile activation outcome that the controller must act on."""

    TO_KEYBOARD = "to_keyboard"
    TO_NAV = "to_nav"
    EXIT = "exit"
    CLICK = "click"
    RECAL = "recal"


class Page(Enum):
    LETTERS = "letters"
    SYMBOLS = "symbols"
    OPTIONS = "options"


DOCK_TILES: Final[dict[TileId, DockSide]] = {
    Ctl.DOCK_TOP: DockSide.TOP,
    Ctl.DOCK_BOTTOM: DockSide.BOTTOM,
    Ctl.DOCK_LEFT: DockSide.LEFT,
    Ctl.DOCK_RIGHT: DockSide.RIGHT,
}


@dataclass(frozen=True)
class KeyboardLayout:
    """A language-specific letter arrangement.."""

    code: str # short badge for the language tile
    label: str
    letter_rows: tuple[str, ...]


SYMBOL_ROWS: tuple[str, ...] = (
    "1234567890",
    "!-:;()\"'@_",
    "+*/=%&#<>",
)

KEYBOARD_LAYOUTS: tuple[KeyboardLayout, ...] = (
    KeyboardLayout("EN", "English", ("qwertyuiop", "asdfghjkl", "zxcvbnm,.?")),
    KeyboardLayout("PL", "Polski", ("qwertyuiop", "asdfghjkl", "zxcvbnm,.?", "ąćęłńóśźż")),
)


@dataclass(frozen=True)
class Tile:
    tile_id: TileId
    label: str
    rect: QRect # In global screen pixels


type ToastLevel = Literal["info", "warning", "error"]

TOAST_BORDER: Final[dict[ToastLevel, QColor]] = {
    "info": ACCENT,
    "warning": QColor(230, 150, 40),
    "error": QColor(200, 60, 50),
}


@dataclass(frozen=True)
class ToastMessage:
    text: str
    level: ToastLevel
    expiry_s: float


class TileBoardWindow(QWidget):
    """Frameless, focus-less, always-on-top panel of gaze-activated tiles."""

    def __init__(self, screen: QScreen) -> None:
        super().__init__()
        self._screen = screen
        self.tiles: list[Tile] = []
        self._highlight: TileId | None = None
        self._progress = 0.0
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
            | Qt.WindowType.WindowDoesNotAcceptFocus
        )
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

    @property
    def screen_geometry(self) -> QRect:
        """The screen this board lives on, in global pixels."""

        return self._screen.geometry()

    def _tile(self, tile_id: TileId) -> Tile | None:
        """Id-addressed tile lookup."""

        return next((t for t in self.tiles if t.tile_id == tile_id), None)

    def tile_hit_test(self, gx: int, gy: int) -> TileId | None:
        return next((t.tile_id for t in self.tiles if t.rect.contains(gx, gy)), None)

    def tile_center_norm(self, tile_id: TileId) -> NormalizedPoint | None:
        tile = self._tile(tile_id)
        if tile is None:
            return None

        geo = self.screen_geometry
        center = tile.rect.center()
        return NormalizedPoint(center.x() / geo.width(), center.y() / geo.height())

    def tile_half_extent_norm(self, tile_id: TileId) -> tuple[float, float] | None:
        """Half the tile's width and height in normalized units."""

        tile = self._tile(tile_id)
        if tile is None:
            return None

        geo = self.screen_geometry
        return (
            tile.rect.width() / (2.0 * geo.width()),
            tile.rect.height() / (2.0 * geo.height()),
        )

    def set_highlight(self, tile_id: TileId | None, progress: float) -> None:
        if tile_id != self._highlight or abs(progress - self._progress) > 0.02:
            self._highlight, self._progress = tile_id, progress
            self.update()

    def _paint_dwell_arc(self, painter: QPainter, local: QRect) -> None:
        """The closing ring that shows how far the dwell has come."""

        painter.setPen(QPen(ACCENT_ACTIVE, 5))
        side = min(local.width(), local.height()) - 14
        arc = QRect(0, 0, side, side)
        arc.moveCenter(local.center())
        painter.drawArc(arc, 90 * 16, -int(360 * 16 * self._progress))

    def _paint_tile(self, painter: QPainter, tile: Tile, origin: QPoint) -> None:
        local = tile.rect.translated(-origin.x(), -origin.y()).adjusted(3, 3, -3, -3)
        active = tile.tile_id == self._highlight

        painter.setPen(QPen(ACCENT_ACTIVE if active else ACCENT, 3 if active else 1))
        painter.setBrush(TILE_BG)
        painter.drawRoundedRect(local, 10, 10)

        if active and self._progress > 0.0:
            self._paint_dwell_arc(painter, local)

        painter.setPen(TILE_TEXT)

        renderer = _icon_renderer(tile.tile_id)
        if renderer is None:
            painter.drawText(local, Qt.AlignmentFlag.AlignCenter, tile.label)
            return

        side = min(local.width(), local.height()) * 0.55
        icon = QRectF(0, 0, side, side)
        icon.moveCenter(QRectF(local).center())
        renderer.render(painter, icon)

    def paintEvent(self, _event: QPaintEvent) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        origin = self.mapToGlobal(self.rect().topLeft())
        painter.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        for tile in self.tiles:
            self._paint_tile(painter, tile, origin)


class KeyboardWindow(TileBoardWindow):
    """Dockable on-screen keyboard with language layouts, pages, window
    controls and a lock mode."""

    floor_unmet_changed = Signal(bool)
    pin_toggled = Signal(bool)

    BUFFER_H: Final[int] = 60
    TILE_MAX: Final[int] = 130
    TILE_ABS_MIN: Final[int] = 48
    SCALE_MIN: Final[float] = 0.5
    SCALE_MAX: Final[float] = 2.5
    SCALE_STEP: Final[float] = 0.1
    OPACITY_MIN: Final[float] = 0.35
    OPACITY_STEP: Final[float] = 0.1
    DEFAULT_WIDTH_FRAC: Final[float] = 0.5
    SIDE_DOCK_WIDTH_FRAC: Final[float] = 0.45

    OPTION_ROWS: Final[tuple[tuple[tuple[str, str], ...], ...]] = (
        ((Ctl.DOCK_TOP, "TOP"), (Ctl.DOCK_BOTTOM, "BOTTOM"),
         (Ctl.DOCK_LEFT, "LEFT"), (Ctl.DOCK_RIGHT, "RIGHT")),
        ((Ctl.SIZE_DOWN, "SIZE -"), (Ctl.SIZE_UP, "SIZE +"),
         (Ctl.OPACITY_DOWN, "OPAC -"), (Ctl.OPACITY_UP, "OPAC +")),
        ((Ctl.CLEAR, "CLR"), (Ctl.LOCK, "LOCK"), (Ctl.PIN, "PIN"), (Ctl.RECAL, "RECAL")),
    )

    def __init__(self, screen: QScreen, min_tile_px: int) -> None:
        super().__init__(screen)
        self.reserve_suggestion_row = False
        self.min_tile_px = min_tile_px
        self.page = Page.LETTERS
        self.dock = DockSide.BOTTOM
        self.scale = 1.0
        self.opacity_level = 1.0
        self.locked = False
        self.pinned = False
        self.buffer = ""
        self._layout_idx = 0
        self._first_key_t: float | None = None
        self._unlock_tile: Tile | None = None
        self._floor_unmet = False
        self.relayout()

    @property
    def layout(self) -> KeyboardLayout:
        return KEYBOARD_LAYOUTS[self._layout_idx]

    @property
    def floor_unmet(self) -> bool:
        """Whether the tiles are smaller than the measured accuracy allows."""

        return self._floor_unmet

    def _rows(self) -> tuple[str, ...] | None:
        if self.page is Page.LETTERS:
            return self.layout.letter_rows
        if self.page is Page.SYMBOLS:
            return SYMBOL_ROWS
        return None # options page has no character rows

    def _control_specs(self) -> list[tuple[str, str, int]]:
        next_layout = KEYBOARD_LAYOUTS[(self._layout_idx + 1) % len(KEYBOARD_LAYOUTS)]
        page_label = "123" if self.page is Page.LETTERS else "ABC"
        return [
            (Ctl.OPTIONS, "OPT", 1),
            (Ctl.LANG, next_layout.code, 1),
            (Ctl.PAGE, page_label, 1),
            (Ctl.SPACE, "SPACE", 3),
            (Ctl.BACKSPACE, "BKSP", 1),
            (Ctl.DELETE_WORD, "DEL W", 1),
            (Ctl.ENTER, "ENTER", 1),
            (Ctl.NAV, "NAV", 1),
            (Ctl.EXIT, "EXIT", 1),
        ]

    def relayout(self) -> None:
        """Rebuild every tile rectangle and the board's own geometry."""

        geo = self.screen_geometry
        rows = self._rows()
        control = self._control_specs()
        control_units = sum(units for _, _, units in control)
        if rows is not None:
            n_rows = len(rows) + 1
            n_cols = max(control_units, max(len(r) for r in rows))
        else:
            n_rows = len(self.OPTION_ROWS) + 1
            n_cols = control_units

        if self.dock in (DockSide.LEFT, DockSide.RIGHT):
            base = min(int(geo.width() * self.SIDE_DOCK_WIDTH_FRAC) // n_cols, self.TILE_MAX)
        else:
            base = min(int(geo.width() * self.DEFAULT_WIDTH_FRAC) // n_cols, self.TILE_MAX)

        cap_w = (geo.width() - 20) // n_cols
        cap_h = (geo.height() - self.BUFFER_H - 20) // n_rows

        if self.reserve_suggestion_row:
            cap_h = (geo.height() - self.BUFFER_H - 20) // (n_rows + 1)

        tile = max(self.TILE_ABS_MIN, min(int(base * self.scale), cap_w, cap_h))

        floor_unmet = tile < self.min_tile_px
        if floor_unmet != self._floor_unmet:
            self._floor_unmet = floor_unmet
            if floor_unmet:
                rlog.event(
                    "keyboard_floor_unmet",
                    tile_px=tile,
                    min_tile_px=self.min_tile_px,
                    level=logging.WARNING
                )
            self.floor_unmet_changed.emit(floor_unmet)

        board_w = tile * n_cols
        board_h = tile * n_rows + self.BUFFER_H

        match self.dock:
            case DockSide.BOTTOM:
                x0 = geo.x() + (geo.width() - board_w) // 2
                y0 = geo.y() + geo.height() - board_h - 10
            case DockSide.TOP:
                x0 = geo.x() + (geo.width() - board_w) // 2
                y0 = geo.y() + 10
            case DockSide.LEFT:
                x0 = geo.x() + 10
                y0 = geo.y() + (geo.height() - board_h) // 2
            case DockSide.RIGHT:
                x0 = geo.x() + geo.width() - board_w - 10
                y0 = geo.y() + (geo.height() - board_h) // 2
            case _:
                assert_never(self.dock)

        x0 = max(geo.x(), x0)
        y0 = max(geo.y(), y0)
        self.setGeometry(x0, y0, board_w, board_h)

        self.tiles = []
        y = y0 + self.BUFFER_H
        if rows is not None:
            for row in rows:
                x = x0 + (board_w - tile * len(row)) // 2
                for ch in row:
                    self.tiles.append(
                        Tile(f"{KEY_PREFIX}{ch}", ch.upper(), QRect(x, y, tile, tile)))
                    x += tile
                y += tile
        else:
            for opt_row in self.OPTION_ROWS:
                units = 2 # options tiles are double-width for readability
                x = x0 + (board_w - tile * units * len(opt_row)) // 2
                for tid, label in opt_row:
                    self.tiles.append(Tile(tid, label, QRect(x, y, tile * units, tile)))
                    x += tile * units
                y += tile

        x = x0 + (board_w - tile * control_units) // 2
        for tid, label, units in control:
            t = Tile(tid, label, QRect(x, y, tile * units, tile))
            self.tiles.append(t)
            if tid is Ctl.EXIT:
                # The corner control tile doubles as the UNLOCK target while
                # the keyboard is locked.
                self._unlock_tile = Tile(Ctl.UNLOCK, "UNLOCK", QRect(t.rect))
            x += tile * units
        self.update()

    def _tile(self, tile_id: TileId) -> Tile | None:
        if tile_id == Ctl.UNLOCK:
            return self._unlock_tile
        return super()._tile(tile_id)

    def tile_hit_test(self, gx: int, gy: int) -> TileId | None:
        if not self.locked:
            return super().tile_hit_test(gx, gy)
        if self._unlock_tile is not None and self._unlock_tile.rect.contains(gx, gy):
            return Ctl.UNLOCK
        return None

    def type_preview(self, text: str) -> None:
        if self._first_key_t is None:
            self._first_key_t = time.monotonic()
        self.buffer += text
        self.update()

    def backspace_preview(self) -> None:
        self.buffer = self.buffer[:-1]
        self.update()

    def delete_word_preview(self) -> None:
        self.buffer = self.buffer.rstrip()
        cut = self.buffer.rfind(" ")
        self.buffer = self.buffer[:cut + 1] if cut >= 0 else ""
        self.update()

    def clear_preview(self) -> None:
        self.buffer = ""
        self._first_key_t = None
        self.update()

    def commit_preview(self) -> tuple[str, float, float]:
        """Close a typing burst: returns (text, WPM, elapsed seconds) and
        resets the strip."""

        text, t0 = self.buffer, self._first_key_t
        elapsed = time.monotonic() - t0 if t0 else 0.0
        wpm = (len(text) / 5.0) / (elapsed / 60.0) if elapsed > 1e-3 else 0.0
        self.buffer, self._first_key_t = "", None
        self.update()
        return text, wpm, elapsed

    def switch_page(self, page: Page) -> None:
        self.page = page
        self.relayout()

    def toggle_symbols(self) -> None:
        self.switch_page(Page.SYMBOLS if self.page is not Page.SYMBOLS else Page.LETTERS)

    def toggle_options(self) -> None:
        self.switch_page(Page.OPTIONS if self.page is not Page.OPTIONS else Page.LETTERS)

    def cycle_language(self) -> str:
        self._layout_idx = (self._layout_idx + 1) % len(KEYBOARD_LAYOUTS)
        if self.page is Page.OPTIONS:
            self.page = Page.LETTERS
        self.relayout()
        return self.layout.code

    def set_dock(self, side: DockSide) -> None:
        self.dock = side
        self.relayout()

    def resize_step(self, direction: int) -> float:
        self.scale = min(self.SCALE_MAX, max(self.SCALE_MIN, self.scale + direction * self.SCALE_STEP))
        self.relayout()
        return self.scale

    def opacity_step(self, direction: int) -> float:
        self.opacity_level = min(1.0, max(self.OPACITY_MIN, self.opacity_level + direction * self.OPACITY_STEP))
        self.setWindowOpacity(self.opacity_level)
        return self.opacity_level

    def set_locked(self, locked: bool) -> None:
        self.locked = locked
        self.update()

    def set_pinned(self, pinned: bool) -> None:
        """Flip the pinned flag and announce it."""

        if pinned == self.pinned:
            return
        self.pinned = pinned
        self.pin_toggled.emit(pinned)

    def paintEvent(self, event: QPaintEvent) -> None:
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        panel = QRect(10, 8, self.width() - 20, self.BUFFER_H - 16)
        painter.setPen(QPen(ACCENT, 1))
        painter.setBrush(QColor(48, 48, 56, 240))
        painter.drawRoundedRect(panel, 10, 10)
        text_rect = panel.adjusted(12, 0, -12, 0)
        painter.setFont(QFont("Arial", 20))
        if self.buffer:
            painter.setPen(TILE_TEXT)
            painter.drawText(text_rect, Qt.AlignmentFlag.AlignVCenter, self.buffer + "▁")
        else:
            painter.setPen(QColor(150, 150, 158))
            painter.drawText(text_rect, Qt.AlignmentFlag.AlignVCenter, "Type with your eyes...")
        painter.setFont(QFont("Arial", 11, QFont.Weight.Bold))
        painter.setPen(QColor(150, 150, 158))
        painter.drawText(text_rect, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, self.layout.code)

        if self.locked:
            painter.setFont(QFont("Arial", 16, QFont.Weight.Bold))
            painter.fillRect(self.rect(), LOCK_DIM)
            painter.setPen(QColor(255, 255, 255, 220))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "KEYBOARD LOCKED")
            origin = self.mapToGlobal(self.rect().topLeft())
            if self._unlock_tile is not None:
                self._paint_tile(painter, self._unlock_tile, origin)


class BrowserBar(TileBoardWindow):
    """Side-docked vertical strip for navigation mode."""

    SPECS: tuple[tuple[str, str], ...] = (
        (Ctl.BACK, "BACK"), (Ctl.FORWARD, "FWD"), (Ctl.PAGE_UP, "PG UP"),
        (Ctl.PAGE_DOWN, "PG DN"), (Ctl.SCROLL_ZONES, "ZONES"),
        (Ctl.BAR_SIDE, "SIDE"), (Ctl.KEYBOARD, "KEYB"), (Ctl.EXIT, "EXIT"),
    )

    def __init__(self, screen: QScreen, min_tile_px: int) -> None:
        super().__init__(screen)
        self._min_tile_px = min_tile_px
        self.side: DockSide = DockSide.RIGHT
        self._relayout()

    def _relayout(self) -> None:
        geo = self._screen.geometry()
        tile = min(max(self._min_tile_px, 90), (geo.height() - 20) // len(self.SPECS))
        if self.side is DockSide.LEFT:
            x0 = geo.x() + 8
        else:
            x0 = geo.x() + geo.width() - tile - 8
        y0 = geo.y() + (geo.height() - tile * len(self.SPECS)) // 2
        self.setGeometry(x0, y0, tile + 4, tile * len(self.SPECS) + 4)
        self.tiles = [
            Tile(tid, label, QRect(x0, y0 + i * tile, tile, tile))
            for i, (tid, label) in enumerate(self.SPECS)
        ]
        self.update()

    def toggle_side(self) -> DockSide:
        """Move the bar to the opposite screen edge, freeing what it covered."""

        self.side = DockSide.LEFT if self.side is DockSide.RIGHT else DockSide.RIGHT
        self._relayout()
        return self.side


class SuggestionBar(TileBoardWindow):
    """Strip with prefix word-completion tiles docked against the keyboard."""

    N_SUGGESTIONS = 3

    def __init__(self, screen: QScreen) -> None:
        super().__init__(screen)
        self._words: list[Suggestion] = []
        self._rect = QRect(0, 0, 0, 0)

    def _keyboard_tile_px(self, keyboard: KeyboardWindow) -> int:
        for tile in keyboard.tiles:
            if tile.tile_id == Ctl.SPACE:
                return tile.rect.height()
        if keyboard.tiles:
            return keyboard.tiles[0].rect.height()
        return keyboard.min_tile_px

    def reposition(self, keyboard: KeyboardWindow) -> None:
        """Anchor the bar against the keyboard's outer geometry and rebuild
        the tile row at the keyboard's tile size."""

        board = keyboard.geometry()
        tile = self._keyboard_tile_px(keyboard)
        n = self.N_SUGGESTIONS + 1

        if keyboard.dock in (DockSide.BOTTOM, DockSide.TOP):
            width = board.width()
            x0 = board.x()
            if keyboard.dock is DockSide.BOTTOM:
                y0 = board.y() - tile - 6
            else:
                y0 = board.y() + board.height() + 6
            cell = width // n
        else:
            width = tile * n
            if keyboard.dock is DockSide.LEFT:
                x0 = board.x() + board.width() + 6
            else:
                x0 = board.x() - width - 6
            y0 = board.y()
            cell = tile

        # Clamp to the screen so the strip is never cut off at an edge.
        geo = self._screen.geometry()
        bar_w = cell * n
        x0 = max(geo.x(), min(x0, geo.x() + geo.width() - bar_w))
        y0 = max(geo.y(), min(y0, geo.y() + geo.height() - tile))
        self._rect = QRect(x0, y0, bar_w, tile)
        self.setGeometry(self._rect.x(), self._rect.y(), self._rect.width(), self._rect.height())
        self._build_tiles(x0, y0, cell, tile)

    def _build_tiles(self, x0: int, y0: int, cell: int, tile: int) -> None:
        self.tiles = []
        x = x0
        for i in range(self.N_SUGGESTIONS):
            word = self._words[i].word if i < len(self._words) else ""
            if word:
                self.tiles.append(Tile(f"{SUGG_PREFIX}{i}", word, QRect(x, y0, cell, tile)))
            x += cell
        self.tiles.append(Tile(Ctl.ADD_WORD, "DICT+", QRect(x, y0, cell, tile)))
        self.update()

    def _paint_tile(self, painter: QPainter, tile: Tile, origin: QPoint) -> None:
        if tile.tile_id == Ctl.ADD_WORD:
            super()._paint_tile(painter, tile, origin)
            return

        local = tile.rect.translated(-origin.x(), -origin.y()).adjusted(3, 3, -3, -3)
        active = tile.tile_id == self._highlight
        painter.setPen(QPen(ACCENT_ACTIVE if active else ACCENT, 3 if active else 1))
        painter.setBrush(QColor(36, 52, 66, 235))
        painter.drawRoundedRect(local, 10, 10)

        if active and self._progress > 0.0:
            painter.setPen(QPen(ACCENT_ACTIVE, 5))
            side = min(local.width(), local.height()) - 14
            arc_rect = QRect(0, 0, side, side)
            arc_rect.moveCenter(local.center())
            painter.drawArc(arc_rect, 90 * 16, -int(360 * 16 * self._progress))

        painter.setPen(TILE_TEXT)
        painter.drawText(local, Qt.AlignmentFlag.AlignCenter, tile.label)

    def suggestion_at(self, index: int) -> Suggestion | None:
        return self._words[index] if 0 <= index < len(self._words) else None

    def set_suggestions(self, suggestions: list[Suggestion]) -> None:
        self._words = suggestions
        rect = self._rect
        cell = rect.width() // (self.N_SUGGESTIONS + 1) if rect.width() else 0
        self._build_tiles(rect.x(), rect.y(), cell, rect.height())


class GazeOverlay(QWidget):
    """Transparent, click-through fullscreen layer: cursor + feedback."""

    MAX_TOASTS: Final[int] = 3
    TOAST_TTL_S: Final[float] = 3.5

    def __init__(self, screen: QScreen) -> None:
        super().__init__()
        self._screen = screen
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
            | Qt.WindowType.WindowTransparentForInput
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setGeometry(screen.geometry())
        self.cursor_px: tuple[int, int] | None = None
        self.fixating = False
        self.progress = 0.0
        self.scroll_zones = False

        self.scroll_active: ScrollDirection = SCROLL_NONE
        self.scroll_zone_frac = 0.12

        self.recal_target_progress: float | None = None
        self.recal_message: str = ""
        self.recal_armed: bool = False

        self._toasts: list[ToastMessage] = []

    def notify(self, text: str, level: ToastLevel = "info", ttl: float | None = None) -> None:
        """Queues a short-lived toast, keeping only the most recent few so a
        burst of state changes cannot pile up on screen."""

        expiry = time.monotonic() + (self.TOAST_TTL_S if ttl is None else ttl)
        self._toasts.append(ToastMessage(text, level, expiry))
        self._toasts = self._toasts[-self.MAX_TOASTS:]
        self.update()

    def _paint_scroll_bands(self, painter: QPainter, w: int, h: int) -> None:
        band = int(self.scroll_zone_frac * h)
        for direction, y in ((SCROLL_UP, 0), (SCROLL_DOWN, h - band)):
            active = self.scroll_active == direction

            fill = QColor(ACCENT_ACTIVE if active else ACCENT)
            fill.setAlpha(70 if active else 10)
            painter.fillRect(0, y, w, band, fill)

            edge = QColor(ACCENT_ACTIVE if active else ACCENT)
            edge.setAlpha(200 if active else 90)
            painter.fillRect(0, 0 if direction == SCROLL_UP else h - 3, w, 3, edge)

    def _prune_toasts(self) -> None:
        """Drops the expired toasts."""

        now = time.monotonic()
        self._toasts = [t for t in self._toasts if t.expiry_s > now]

    def _paint_toasts(self, painter: QPainter, w: int) -> None:
        self._prune_toasts()

        painter.setFont(QFont("Arial", 13))
        metrics = painter.fontMetrics()
        y = 12

        for toast in self._toasts:
            box = QRect(0, y, metrics.horizontalAdvance(toast.text) + 32, metrics.height() + 16)
            box.moveLeft((w - box.width()) // 2)

            painter.setPen(QPen(TOAST_BORDER[toast.level], 2))
            painter.setBrush(TILE_BG)
            painter.drawRoundedRect(box, 8, 8)
            painter.setPen(TILE_TEXT)
            painter.drawText(box, Qt.AlignmentFlag.AlignCenter, toast.text)

            y += box.height() + 8

    def paintEvent(self, _event: QPaintEvent) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()

        if self.scroll_zones:
            self._paint_scroll_bands(painter, w, h)

        self._paint_toasts(painter, w)

        if self.recal_target_progress is not None:
            cx, cy = w // 2, h // 2

            if self.recal_message:
                painter.setPen(QColor(235, 235, 240))
                painter.setFont(QFont("Arial", 15, QFont.Weight.Bold))
                painter.drawText(
                    QRect(0, cy - 110, w, 32),
                    Qt.AlignmentFlag.AlignHCenter,
                    self.recal_message,
                )

            if self.recal_armed:
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.setPen(QPen(ACCENT, 2))
                painter.drawEllipse(cx - 22, cy - 22, 44, 44)
                painter.setBrush(ACCENT_ACTIVE)
                painter.setPen(QPen(ACCENT_ACTIVE, 1))
                painter.drawEllipse(cx - 6, cy - 6, 12, 12)

            prog = max(0.0, min(1.0, self.recal_target_progress))
            if prog > 0.0:
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.setPen(QPen(ACCENT_ACTIVE if self.recal_armed else ACCENT, 4))
                painter.drawArc(cx - 30, cy - 30, 60, 60, 90 * 16, -int(360 * 16 * prog))

        if self.cursor_px is None:
            return
        x, y = self.cursor_px

        color = ACCENT_ACTIVE if self.fixating else QColor(231, 76, 60)
        painter.setPen(QPen(color, 3))
        painter.drawEllipse(x - 10, y - 10, 20, 20)

        if self.fixating and self.progress > 0.0:
            painter.setPen(QPen(ACCENT_ACTIVE, 4))
            painter.drawArc(x - 18, y - 18, 36, 36, 90 * 16, -int(360 * 16 * self.progress))


class GazeMode(Protocol):
    name: str
    board: TileBoardWindow
    edge_scroll: bool
    suggestions_board: TileBoardWindow | None

    def enter(self) -> None: ...
    def exit(self) -> None: ...
    def on_activation(self, tile_id: TileId) -> ModeAction | None:
        """Handles an activated tile."""
        ...


class KeyboardMode:
    """Handles typing into the OS-focused field."""

    name = "KEYBOARD"
    edge_scroll = False

    def __init__(self, keyboard: KeyboardWindow, drift: DriftCorrector | None,
                 actuator: SystemActuator, on_relayout: Callable[[], None] | None = None,
                 hotkeys: HotkeyProfile | None = None) -> None:
        self.board = keyboard
        self._drift = drift
        self._actuator = actuator
        self._on_relayout = on_relayout
        self._hotkeys = hotkeys if hotkeys is not None else load_profile()

        assets = pathlib.Path(__file__).resolve().parent.parent / "assets" / "dict"
        self.engine = SuggestionEngine(assets, pathlib.Path.home() / ".vistea" / "user_dictionary.json")
        self.suggestions_board: SuggestionBar | None = SuggestionBar(keyboard._screen)

        keyboard.reserve_suggestion_row = True
        keyboard.relayout()

    def enter(self) -> None:
        self.board.show()
        self._refresh_suggestions()

    def exit(self) -> None:
        self.board.hide()
        if self.suggestions_board is not None:
            self.suggestions_board.hide()

    def _current_prefix(self) -> str:
        """Currently typed word: the buffer tail after the last space."""

        buffer = self.board.buffer
        cut = buffer.rfind(" ")
        return buffer[cut + 1:] if cut >= 0 else buffer

    def _refresh_suggestions(self) -> None:
        bar = self.suggestions_board
        if bar is None:
            return

        if self.board.locked or self.board.page is not Page.LETTERS:
            bar.hide()
            return

        bar.reposition(self.board)
        bar.set_suggestions(self.engine.suggest(self._current_prefix(), self.board.layout.code))
        bar.show()

    def _apply_drift(self, tile_id: TileId) -> None:
        """Drift correction before mutating layout: the buffered dwell frames
        are labelled by the tile's center, if the batch earns that label."""

        if self._drift is None:
            return
        center = self.board.tile_center_norm(tile_id)
        half_extent = self.board.tile_half_extent_norm(tile_id)
        if center is None or half_extent is None:
            return

        result = self._drift.on_activation(center, half_extent)
        if result is None:
            return

        rlog.event(
            "drift_correction",
            tile=tile_id,
            n_samples=result.n_samples,
            total=self._drift.corrections_applied,
            rejected=self._drift.rejected,
            pre_error_norm=round(result.pre_error_norm, 5),
            post_error_norm=round(result.post_error_norm, 5),
        )

    def _type(self, text: str) -> None:
        """Sends text to the focused application and mirror it in the preview."""

        self.board.type_preview(text)
        self._actuator.type_text(text)
        self._refresh_suggestions()

    def _accept_suggestion(self, index: int) -> None:
        """Replaces the half-typed prefix with the whole suggested word."""

        bar = self.suggestions_board
        suggestion = bar.suggestion_at(index) if bar is not None else None
        if suggestion is None:
            return

        prefix = self._current_prefix()
        for _ in prefix:
            self._actuator.press_key("backspace")
            self.board.backspace_preview()

        self._type(suggestion.word + " ")
        rlog.event(
            "suggestion_accepted",
            word=suggestion.word,
            prefix_len=len(prefix),
            saved_keystrokes=len(suggestion.word) - len(prefix),
            fuzzy=suggestion.fuzzy,
            level=logging.DEBUG,
        )

    def _learn_current_word(self) -> None:
        """Adds what is being typed to the user dictionary: the whole preview
        once it holds a phrase, otherwise just the unfinished word."""

        buffer = self.board.buffer.strip()
        text = buffer if " " in buffer else self._current_prefix()
        if self.engine.add_custom(text):
            rlog.event("custom_word_added", text=text, level=logging.DEBUG)

    def _commit_text(self) -> None:
        text, wpm, elapsed = self.board.commit_preview()
        rlog.event("text_committed", chars=len(text), elapsed_s=round(elapsed, 1), wpm=round(wpm, 1))
        self._actuator.press_key("enter")
        self._refresh_suggestions()

    def on_activation(self, tile_id: TileId) -> ModeAction | None:
        board = self.board

        if board.locked:
            if tile_id == Ctl.UNLOCK:
                board.set_locked(False)
                self._refresh_suggestions()
                rlog.event("keyboard_locked", locked=False)
            return None

        if tile_id.startswith(SUGG_PREFIX):
            self._accept_suggestion(int(tile_id.removeprefix(SUGG_PREFIX)))
            return None

        if tile_id.startswith(KEY_PREFIX):
            self._apply_drift(tile_id)
            self._type(tile_id.removeprefix(KEY_PREFIX))
            return None

        match tile_id:
            case Ctl.SPACE:
                self._apply_drift(tile_id)
                self._type(" ")
            case Ctl.ADD_WORD:
                self._learn_current_word()
            case Ctl.BACKSPACE:
                board.backspace_preview()
                self._actuator.press_key("backspace")
                self._refresh_suggestions()
            case Ctl.DELETE_WORD:
                board.delete_word_preview()
                send_hotkey(NavAction.DELETE_WORD, self._actuator, self._hotkeys)
                self._refresh_suggestions()
            case Ctl.ENTER:
                self._commit_text()
            case Ctl.CLEAR:
                board.clear_preview()
                self._refresh_suggestions()
            case Ctl.PAGE:
                board.toggle_symbols()
                self._refresh_suggestions()
            case Ctl.OPTIONS:
                board.toggle_options()
                self._refresh_suggestions()
            case Ctl.LANG:
                rlog.event("keyboard_layout_changed", layout=board.cycle_language())
                self._refresh_suggestions()
            case Ctl.DOCK_TOP | Ctl.DOCK_BOTTOM | Ctl.DOCK_LEFT | Ctl.DOCK_RIGHT:
                board.set_dock(DOCK_TILES[tile_id])
                rlog.event("keyboard_dock_changed", dock=board.dock.value)
                self._relayout_pin()
            case Ctl.SIZE_DOWN | Ctl.SIZE_UP:
                board.resize_step(+1 if tile_id == Ctl.SIZE_UP else -1)
                self._relayout_pin()
            case Ctl.OPACITY_DOWN:
                board.opacity_step(-1)
            case Ctl.OPACITY_UP:
                board.opacity_step(+1)
            case Ctl.LOCK:
                board.set_locked(True)
                self._refresh_suggestions()
                rlog.event("keyboard_locked", locked=True)
            case Ctl.PIN:
                board.set_pinned(not board.pinned)
                rlog.event("keyboard_pinned", pinned=board.pinned)
            case Ctl.RECAL:
                return ModeAction.RECAL
            case Ctl.NAV:
                return ModeAction.TO_NAV
            case Ctl.EXIT:
                return ModeAction.EXIT

        return None

    def _relayout_pin(self) -> None:
        if self._on_relayout is not None:
            self._on_relayout()


class NavigationMode:
    """Handles dwell-clicks and edge-zone scrolling."""

    name = "NAVIGATION"
    edge_scroll = True

    def __init__(
        self,
        bar: BrowserBar,
        actuator: SystemActuator,
        dpr: float = 1.0,
        hotkeys: HotkeyProfile | None = None,
        on_toggle_zones: Callable[[], bool] | None = None,
    ) -> None:
        self.board = bar
        self._actuator = actuator
        self._dpr = dpr
        self._hotkeys = hotkeys if hotkeys is not None else load_profile()
        self._on_toggle_zones = on_toggle_zones
        self.suggestions_board: TileBoardWindow | None = None

    def enter(self) -> None:
        self.board.show()

    def exit(self) -> None:
        self.board.hide()

    PAGE_TICKS: Final[int] = 12

    def _scroll_at_content(self, ticks: int) -> None:
        geo = self.board.screen_geometry
        center = (geo.x() + geo.width() // 2, geo.y() + geo.height() // 2)
        self._actuator.move(*to_physical(*center, self._dpr))
        self._actuator.scroll(ticks)

    def _navigate(self, action: NavAction) -> None:
        chord = send_hotkey(action, self._actuator, self._hotkeys)
        rlog.event(
            "nav_hotkey",
            action=action.value,
            chord="+".join(chord) if chord else None,
            profile=self._hotkeys.name,
            level=logging.DEBUG,
        )

    def on_activation(self, tile_id: TileId) -> ModeAction | None:
        if tile_id == SCREEN_CLICK:
            return ModeAction.CLICK

        match tile_id:
            case Ctl.BACK:
                self._navigate(NavAction.BACK)
            case Ctl.FORWARD:
                self._navigate(NavAction.FORWARD)
            case Ctl.PAGE_UP:
                self._scroll_at_content(+self.PAGE_TICKS)
            case Ctl.PAGE_DOWN:
                self._scroll_at_content(-self.PAGE_TICKS)
            case Ctl.SCROLL_ZONES:
                if self._on_toggle_zones is not None:
                    rlog.event("scroll_zones_toggled", enabled=self._on_toggle_zones())
            case Ctl.BAR_SIDE:
                rlog.event("nav_bar_moved", side=self.board.toggle_side().value)
            case Ctl.KEYBOARD:
                return ModeAction.TO_KEYBOARD
            case Ctl.EXIT:
                return ModeAction.EXIT

        return None


class GazeInteractionController(QObject):
    """Per-frame interaction pipeline, owned by the main window after calibration."""

    stopped = Signal()

    RECAL_TARGET: Final[NormalizedPoint] = NormalizedPoint(0.5, 0.5)

    # Head-pose drift transitions before RECAL is suggested, once per session
    POSE_DRIFT_HINT_AT: Final[int] = 3

    def __init__(
        self,
        strategy: CalibrationStrategy,
        screen: QScreen,
        activation: ActivationStrategy,
        actuator: SystemActuator,
        mean_error_px: float | None,
    ) -> None:
        super().__init__()
        self._screen = screen
        self._geo = screen.geometry()
        self._dpr = screen.devicePixelRatio()
        tile_px = min_tile_side_px(mean_error_px)

        self.overlay = GazeOverlay(screen)
        self.activation = activation
        self.fixation = IDTFixationDetector()
        self.drift = DriftCorrector(strategy)
        self.keyboard = KeyboardWindow(screen, tile_px)
        self.edge_detector = EdgeScrollDetector(
            zone_height_frac=zone_from_error(mean_error_px, self._geo.height())
        )
        self.overlay.scroll_zone_frac = self.edge_detector.zone_height_frac
        self.actuator = actuator
        self.bar = BrowserBar(screen, tile_px)
        self.hotkeys = load_profile()

        self._modes: dict[ModeAction, GazeMode] = {
            ModeAction.TO_KEYBOARD: KeyboardMode(
                self.keyboard, self.drift, actuator, self.relayout_pin,
                hotkeys=self.hotkeys,
            ),
            ModeAction.TO_NAV: NavigationMode(
                self.bar, actuator, self._dpr,
                hotkeys=self.hotkeys, on_toggle_zones=self.toggle_scroll_zones,
            ),
        }
        self.mode: GazeMode = self._modes[ModeAction.TO_KEYBOARD]

        rlog.event(
            "gaze_mode_started",
            activation=activation.name,
            tile_px=tile_px,
            mean_error_px=mean_error_px,
            tile_meets_2x_error=(mean_error_px is None or tile_px >= 2 * mean_error_px),
            device_pixel_ratio=self._dpr
        )
        self.overlay.show()
        self.mode.enter()

        self.workarea: WorkAreaReservation | None = None
        self.keyboard.pin_toggled.connect(self._on_pin_toggled)

        self._recal: RecalibrationSession | None = None
        self._gaze_bias: NormalizedPoint | None = None

        self.pose_residual = PoseResidualCorrector()
        self._latest_pose: tuple[float, float, float] | None = None

        # Head-pose-drift suggestion: nudge the user toward RECAL once per
        # session after repeated drift transitions.
        self._pose_drift_count = 0
        self._pose_drift_hinted = False

        self.keyboard.floor_unmet_changed.connect(self._on_floor_unmet_changed)
        if self.keyboard.floor_unmet:
            self._on_floor_unmet_changed(True)

    def notify(self, text: str, level: ToastLevel = "info") -> None:
        self.overlay.notify(text, level)

    def _on_floor_unmet_changed(self, floor_unmet: bool) -> None:
        if floor_unmet:
            self.notify("Keyboard below accuracy floor - consider recalibrating", "warning")

    def toggle_scroll_zones(self) -> bool:
        """Switch edge scrolling on or off, and announce the result."""

        enabled = self.edge_detector.set_enabled(not self.edge_detector.enabled)
        self.notify(
            "Scroll zones on" if enabled else "Scroll zones off - edges are clickable"
        )
        return enabled

    def _enter_recal(self) -> None:
        """Begin guided single-point recalibration: normal activation is
        suspended while the session walks the user through the stages."""

        self._recal = RecalibrationSession(self.RECAL_TARGET, time.monotonic())
        self.overlay.recal_target_progress = 0.0
        self.overlay.recal_message = ""
        self.activation.reset()
        self.fixation.reset()
        self.overlay.update()

    def _finish_recal(self, status_message: str, applied: bool) -> None:
        self._recal = None
        self.overlay.recal_target_progress = None
        self.overlay.recal_message = ""
        self.activation.reset()
        self.fixation.reset()
        self.overlay.update()

        self.notify(status_message, "info" if applied else "warning")

    def note_pose_drift(self) -> None:
        """Record a head-pose drift transition, and once they add up, point the
        user to quick recalibration."""

        self._pose_drift_count += 1
        if self._pose_drift_count >= self.POSE_DRIFT_HINT_AT and not self._pose_drift_hinted:
            self._pose_drift_hinted = True
            self.notify("Head pose drifts often - consider RECAL (options page)", "warning")

    def _ensure_workarea(self) -> WorkAreaReservation:
        """Build the reservation backend on first use. Deferred so winId()
        (which forces a native window) is only called once the user pins."""

        if self.workarea is None:
            native = sys.platform == "win32" or sys.platform.startswith("linux")
            handle = self.keyboard.winId() if native else None
            self.workarea = make_reservation(handle)
        return self.workarea

    def _reserve_current_strip(self) -> None:
        """(Re)reserve the desktop strip for the current keyboard geometry,
        taking the suggestion strip into account when it's visible."""

        workarea = self._ensure_workarea()
        kb = self.keyboard
        kb_geo = _geo_tuple(kb.geometry())
        sugg = self.mode.suggestions_board
        sugg_geo = _geo_tuple(sugg.geometry()) if sugg is not None and sugg.isVisible() else None
        edge, rect = reservation_strip(kb.dock, kb_geo, sugg_geo, _geo_tuple(self._geo), self._dpr)
        workarea.reserve(edge, rect)

    def _on_pin_toggled(self, pinned: bool) -> None:
        if pinned:
            self._reserve_current_strip()
            self.notify("Keyboard pinned - windows make room")
        else:
            if self.workarea is not None:
                self.workarea.release()
            self.notify("Keyboard unpinned")

    def relayout_pin(self) -> None:
        """Called after a dock or size change: re-reserves the strip if the
        keyboard is currently pinned."""

        if self.keyboard.pinned:
            self._reserve_current_strip()

    def stop(self) -> None:
        try:
            if self.workarea is not None:
                self.workarea.release()
        finally:
            self.mode.exit()
            self.overlay.hide()
            rlog.event("gaze_mode_stopped")
            self.stopped.emit()

    def _switch(self, action: ModeAction) -> None:
        self.mode.exit()
        self.activation.reset()
        self.fixation.reset()

        if action is ModeAction.TO_NAV and self.workarea is not None:
            self.workarea.release()

        self.mode = self._modes[action]
        self.mode.enter()

        if action is ModeAction.TO_KEYBOARD and self.keyboard.pinned:
            self._reserve_current_strip()

        rlog.event("mode_switched", mode=self.mode.name)

    def _observe_pose_residual(
        self, tile_id: TileId, actual_gaze_pos: tuple[float, float] | None
    ) -> None:
        """Records the gaze error for a given head pose upon a successful tile activation."""

        if actual_gaze_pos is None or self._latest_pose is None:
            return
        center = self.mode.board.tile_center_norm(tile_id)
        if center is None:
            return

        residual = (center.x - actual_gaze_pos[0], center.y - actual_gaze_pos[1])
        if not self.pose_residual.observe(self._latest_pose, residual):
            return

        rlog.event(
            "pose_residual_fit",
            tile=tile_id,
            n_observations=self.pose_residual.n_observations,
            residual_x=round(residual[0], 5),
            residual_y=round(residual[1], 5),
            level=logging.DEBUG,
        )

    def _to_pixels(self, point: tuple[float, float]) -> tuple[int, int]:
        return (
            self._geo.x() + int(point[0] * self._geo.width()),
            self._geo.y() + int(point[1] * self._geo.height()),
        )

    def _update_fixation(self, gaze: NormalizedPoint | None, t: float) -> FixationState:
        x, y = (gaze.x, gaze.y) if gaze is not None else (None, None)
        return self.fixation.update(x, y, t)

    def _correct(self, gaze: NormalizedPoint | None) -> NormalizedPoint | None:
        """Applies dynamic gaze correction."""

        if gaze is None:
            return None
        if self._gaze_bias is not None:
            gaze = NormalizedPoint(gaze.x + self._gaze_bias.x, gaze.y + self._gaze_bias.y)
        return self.pose_residual.correct(gaze, self._latest_pose)

    @staticmethod
    def _clamp_to_screen(gaze: NormalizedPoint | None) -> NormalizedPoint | None:
        if gaze is None:
            return None

        return NormalizedPoint(
            min(1.0, max(0.0, gaze.x)),
            min(1.0, max(0.0, gaze.y)),
        )

    def _run_recalibration(self, gaze: NormalizedPoint | None, t: float) -> None:
        """Drive one frame of the recalibration session."""

        recal = self._recal
        if recal is None:
            return

        fix = self._update_fixation(gaze, t)
        status = recal.update(gaze, fix.fixating, t)

        self.overlay.recal_target_progress = status.progress
        self.overlay.recal_message = status.message
        self.overlay.recal_armed = status.phase is not RecalPhase.PROMPT
        self.overlay.cursor_px = None
        self.overlay.fixating = fix.fixating
        self.overlay.progress = 0.0
        self.overlay.update()

        if not status.finished:
            return

        if status.phase is RecalPhase.DONE and status.offset is not None:
            prev = self._gaze_bias or NormalizedPoint(0.0, 0.0)
            self._gaze_bias = NormalizedPoint(prev.x + status.offset[0], prev.y + status.offset[1])
            rlog.event(
                "drift_recalibration",
                offset_x=round(self._gaze_bias.x, 5),
                offset_y=round(self._gaze_bias.y, 5),
                n_frames=recal.n_frames,
            )
            self._finish_recal(status.message, applied=True)
        else:
            self._finish_recal(status.message, applied=False)

    def _resolve_target(self, probe_px: tuple[int, int] | None) -> TileId | None:
        """Which tile the probe lands on. In navigation mode bare screen is a
        target too, unless a scroll band already owns that spot."""

        if probe_px is None:
            return None

        target = self.mode.board.tile_hit_test(*probe_px)

        if target is None:
            suggestions = self.mode.suggestions_board
            if suggestions is not None and suggestions.isVisible():
                target = suggestions.tile_hit_test(*probe_px)

        if (
            target is None
            and self.mode.edge_scroll
            and self.edge_detector.current_zone == SCROLL_NONE
        ):
            target = SCREEN_CLICK

        return target

    def _apply_mode_action(
        self, action: ModeAction | None, probe_px: tuple[int, int] | None
    ) -> bool:
        """Carry out what the mode asked for. Returns True when the frame ends
        here, because the session stopped or recalibration took the screen."""

        match action:
            case ModeAction.EXIT:
                self.stop()
                return True
            case ModeAction.RECAL:
                self._enter_recal()
                return True
            case ModeAction.TO_NAV | ModeAction.TO_KEYBOARD:
                self._switch(action)
            case ModeAction.CLICK if probe_px is not None:
                self.actuator.move(*to_physical(*probe_px, self._dpr))
                self.actuator.click()
            case _:
                pass

        return False

    def _update_edge_scroll(
        self,
        target: TileId | None,
        probe: tuple[float, float] | None,
        gaze: NormalizedPoint | None,
        t: float,
    ) -> None:
        if not self.mode.edge_scroll:
            self.overlay.scroll_zones = False
            self.overlay.scroll_active = SCROLL_NONE
            return

        was_active = self.edge_detector.active

        # A tile drawn inside a band belongs to the board, not the gesture
        over_widget = target is not None and target != SCREEN_CLICK
        ticks = self.edge_detector.update(
            probe[1] if probe is not None else None, t, suppressed=over_widget
        )

        if ticks and gaze is not None:
            self.actuator.move(*to_physical(*self._to_pixels((gaze.x, gaze.y)), self._dpr))
            self.actuator.scroll(ticks)

        if self.edge_detector.active and not was_active:
            rlog.event(
                "scroll_gesture",
                direction="up" if self.edge_detector.current_zone == SCROLL_UP else "down",
                level=logging.DEBUG,
            )

        self.overlay.scroll_zones = self.edge_detector.enabled
        self.overlay.scroll_active = (
            self.edge_detector.current_zone if self.edge_detector.active else SCROLL_NONE
        )

    def _render_feedback(
        self, gaze: NormalizedPoint | None, fix: FixationState, target: TileId | None
    ) -> None:
        if gaze is not None:
            self.overlay.cursor_px = self._to_pixels((gaze.x, gaze.y))
        self.overlay.fixating = fix.fixating
        self.overlay.progress = self.activation.progress
        self.overlay.update()

        for board in (self.mode.board, self.mode.suggestions_board):
            if board is not None:
                board.set_highlight(target, self.activation.progress)

    def on_gaze(self, gaze: NormalizedPoint | None, data: LandmarksData,
                vec: list[float] | None) -> None:
        """One camera frame's worth of interaction, from a gaze estimate to
        keystrokes, clicks, scrolling and the drawn feedback."""

        t = time.monotonic()

        self._latest_pose = (
            (data.head_pose.yaw, data.head_pose.pitch, data.head_pose.roll)
            if data.head_pose is not None else None
        )
        gaze = self._correct(gaze)

        if self._recal is not None:
            self._run_recalibration(gaze, t)
            return

        gaze = self._clamp_to_screen(gaze)
        fix = self._update_fixation(gaze, t)

        gaze_tuple = (gaze.x, gaze.y) if gaze else None
        probe = fix.centroid if fix.centroid is not None else gaze_tuple
        probe_px = self._to_pixels(probe) if probe is not None else None
        target = self._resolve_target(probe_px)

        if fix.fixating and target is not None and target.startswith((KEY_PREFIX, CTL_PREFIX)):
            self.drift.observe(vec)

        event = self.activation.update(
            target, fix.fixating, extract_blink_score(data.blendshapes), t
        )
        if event is not None:
            rlog.event(
                "tile_activated",
                tile=event.target_id,
                kind=event.kind.value,
                mode=self.mode.name,
            )
            self._observe_pose_residual(event.target_id, probe)
            if self._apply_mode_action(self.mode.on_activation(event.target_id), probe_px):
                return

        self._update_edge_scroll(target, probe, gaze, t)
        self._render_feedback(gaze, fix, target)