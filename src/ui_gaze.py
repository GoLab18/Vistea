"""Gaze interaction layer: overlay, tile boards, modes.

Structure:
- ``GazeOverlay`` - fullscreen, transparent, INPUT-TRANSPARENT window drawing
  the gaze cursor, fixation ring, dwell progress and edge-zone hints. Input
  transparency matters twice: the user's real applications stay clickable
  underneath, and navigation-mode OS clicks pass through to them.
- ``TileBoardWindow`` - base for gaze-activated tile panels. The windows
  never take OS focus (``WindowDoesNotAcceptFocus`` +
  ``WA_ShowWithoutActivating``), which is what makes system-wide typing
  possible: the user's text field KEEPS focus while the gaze keyboard
  injects keystrokes into it through the system actuator.
- ``KeyboardWindow`` - dockable on-screen keyboard: pluggable language
  layouts (``KEYBOARD_LAYOUTS``), a symbols page and an options page
  (docking edge, board scale, opacity, lock), a preview strip with WPM
  measurement, and a lock mode in which only a single UNLOCK tile responds
  to gaze (a pause against the Midas touch problem when the user wants to
  read instead of type).
- ``BrowserBar`` - side strip for navigation mode: back/forward/page
  up/down, switch to keyboard, exit.
- Modes (State Pattern behind the ``GazeMode`` Protocol): ``KeyboardMode``
  types into the OS-focused field (and mirrors it in the preview strip) and
  feeds drift correction, ``NavigationMode`` dwell-clicks and enables
  edge-zone scrolling. The real OS cursor is not dragged along with gaze
  continuously - it is moved only on demand, right before a click and right
  before a scroll tick is emitted, since the OS scroll wheel acts on
  whatever is under the real cursor, the on-screen crosshair (drawn on
  ``GazeOverlay``) is the continuous gaze indicator for the user.
- ``GazeInteractionController`` - the per-frame pipeline: gaze -> I-DT
  fixation -> hit-test on the active mode's board -> activation strategy ->
  mode action (+ drift correction, + gestures), all logged as research
  events (tile_activated, text_committed with WPM, drift_correction,
  scroll_gesture) for the user studies. Coordinates handed to the system
  actuator are converted from Qt's logical pixels to physical pixels
  (``to_physical``), so on-demand cursor moves land correctly under
  Windows display scaling.

Tile geometry stays responsive to the user. The accuracy-derived floor
(``min tile side = 2 x mean validation error``, core.drift.min_tile_side_px)
is now ADVISORY: it is a reference point that flags (``_floor_unmet`` + a
warning/toast) when the current tile falls below it, but it never clamps the
size. An explicit user resize and the screen cap take precedence, so the
scale slider always has a visible effect even after a poor calibration, the
only inviolable lower bound is ``TILE_ABS_MIN`` (absolute pointer-target
minimum). The tile side is still hard-capped to the screen dimensions, so a
high user scale can never lay out a board wider or taller than the display.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Protocol, Tuple

from PySide6.QtCore import QObject, QRect, Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import QWidget

from core import research_log as rlog
from core.activation import ActivationStrategy, extract_blink_score
from core.actuator import SystemActuator
from core.calibration import CalibrationStrategy, NormalizedPoint
from core.drift import DriftCorrector, min_tile_side_px
from core.fixation import IDTFixationDetector
from core.gestures import EdgeScrollDetector
from core.vision import LandmarksData

ACCENT = QColor(41, 128, 185)
ACCENT_ACTIVE = QColor(46, 204, 113)
TILE_BG = QColor(30, 30, 35, 235)
TILE_TEXT = QColor(240, 240, 240)
LOCK_DIM = QColor(10, 10, 12, 170)

KEY_PREFIX = "key:" # tile ids of character keys: "key:<char>"
SCREEN_CLICK = "screen:click" # pseudo-target: dwell on bare screen (nav mode)


def to_physical(x: int, y: int, dpr: float) -> Tuple[int, int]:
    """Convert Qt logical-pixel coordinates to physical-pixel coordinates
    for the system actuator. Qt reports geometry in logical pixels, but
    pynput moves the real OS cursor in physical pixels, so under Windows
    display scaling (devicePixelRatio > 1.0) an unconverted move lands the
    real cursor toward the top-left of the intended target."""

    return round(x * dpr), round(y * dpr)


class Ctl(str, Enum):
    """Identifiers of all non-character tiles. A ``str`` subclass, so tile
    ids remain plain strings for hit-testing, telemetry and the activation
    layer, while the UI code compares against named members instead of
    scattering string literals."""

    # keyboard control row

    OPTIONS = "ctl:options"
    LANG = "ctl:lang"
    PAGE = "ctl:page"
    SPACE = "ctl:space"
    BACKSPACE = "ctl:bksp"
    DELETE_WORD = "ctl:delword"
    ENTER = "ctl:enter"
    NAV = "ctl:nav"
    EXIT = "ctl:exit"

    # options page

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

    # browser bar

    BACK = "ctl:back"
    FORWARD = "ctl:fwd"
    PAGE_UP = "ctl:pgup"
    PAGE_DOWN = "ctl:pgdn"
    KEYBOARD = "ctl:kbd"


import pathlib

ICON_FILES: Dict[str, str] = {
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
    Ctl.OPTIONS: "settings",
    Ctl.CLEAR: "eraser",
    Ctl.SIZE_UP: "zoom-in",
    Ctl.SIZE_DOWN: "zoom-out",
}

ICONS_DIR = pathlib.Path(__file__).resolve().parent.parent / "assets" / "icons"

_ICON_RENDERERS: Dict[str, Optional[object]] = {}


def _icon_renderer(tile_id: str) -> Optional[object]:
    """Lazily build (and cache) a QSvgRenderer for a tile's Lucide icon.
    Must stay lazy/inside a paint path - scripts/smoke_ui.py stubs Qt (no
    PySide6.QtSvg module at all) and never runs paintEvent, so the import
    itself is deferred to this call. A cached None means "no icon for this
    tile" (missing file / bad SVG / QtSvg unavailable) and short-circuits
    the disk read and parse on every subsequent repaint."""

    if tile_id in _ICON_RENDERERS:
        return _ICON_RENDERERS[tile_id]
    renderer = None
    name = ICON_FILES.get(tile_id)
    if name is not None:
        try:
            from PySide6.QtCore import QByteArray
            from PySide6.QtSvg import QSvgRenderer
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
    """Semantic outcome of a tile activation that the controller must act
    on (everything else is handled inside the mode/board)."""

    TO_KEYBOARD = "to_keyboard"
    TO_NAV = "to_nav"
    EXIT = "exit"
    CLICK = "click"


class Page(Enum):
    LETTERS = "letters"
    SYMBOLS = "symbols"
    OPTIONS = "options"


class DockSide(Enum):
    BOTTOM = "bottom"
    TOP = "top"
    LEFT = "left"
    RIGHT = "right"


@dataclass(frozen=True)
class KeyboardLayout:
    """A language-specific letter arrangement. Layouts are first-class data:
    adding a language means adding an entry to ``KEYBOARD_LAYOUTS``, nothing
    else changes. Diacritics are ordinary first-class tiles in their own
    row - no dead keys, no long-press, no separate diacritics page."""

    code: str # short badge shown on the LANG tile
    label: str
    letter_rows: Tuple[str, ...]


SYMBOL_ROWS: Tuple[str, ...] = (
    "1234567890",
    "!-:;()\"'@_",
    "+*/=%&#<>",
)

KEYBOARD_LAYOUTS: Tuple[KeyboardLayout, ...] = (
    KeyboardLayout("EN", "English", ("qwertyuiop", "asdfghjkl", "zxcvbnm,.?")),
    KeyboardLayout("PL", "Polski", ("qwertyuiop", "asdfghjkl", "zxcvbnm,.?", "ąćęłńóśźż")),
)


@dataclass
class Tile:
    tile_id: str
    label: str
    rect: QRect # in GLOBAL screen pixels


class TileBoardWindow(QWidget):
    """Frameless, focus-less, always-on-top panel of gaze-activated tiles."""

    def __init__(self, screen):
        super().__init__()
        self._screen = screen
        self.tiles: List[Tile] = []
        self._highlight: Optional[str] = None
        self._progress = 0.0
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
            | Qt.WindowType.WindowDoesNotAcceptFocus
        )
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

    # gaze interface

    def hit_test(self, gx: int, gy: int) -> Optional[str]:
        for tile in self.tiles:
            if tile.rect.contains(gx, gy):
                return tile.tile_id
        return None

    def tile_center_norm(self, tile_id: str) -> Optional[NormalizedPoint]:
        geo = self._screen.geometry()
        for tile in self.tiles:
            if tile.tile_id == tile_id:
                c = tile.rect.center()
                return NormalizedPoint(c.x() / geo.width(), c.y() / geo.height())
        return None

    def set_highlight(self, tile_id: Optional[str], progress: float) -> None:
        if tile_id != self._highlight or abs(progress - self._progress) > 0.02:
            self._highlight, self._progress = tile_id, progress
            self.update()

    def _paint_tile(self, painter: QPainter, tile: Tile, origin) -> None:
        local = tile.rect.translated(-origin.x(), -origin.y()).adjusted(3, 3, -3, -3)
        active = tile.tile_id == self._highlight
        painter.setPen(QPen(ACCENT_ACTIVE if active else ACCENT, 3 if active else 1))
        painter.setBrush(TILE_BG)
        painter.drawRoundedRect(local, 10, 10)
        if active and self._progress > 0.0: # dwell progress arc on the tile
            painter.setPen(QPen(ACCENT_ACTIVE, 5))
            side = min(local.width(), local.height()) - 14
            arc_rect = QRect(0, 0, side, side)
            arc_rect.moveCenter(local.center())
            painter.drawArc(arc_rect, 90 * 16, -int(360 * 16 * self._progress))
        painter.setPen(TILE_TEXT)

        renderer = _icon_renderer(tile.tile_id)
        if renderer is not None:
            from PySide6.QtCore import QRectF
            side = min(local.width(), local.height()) * 0.55
            icon_rect = QRectF(0, 0, side, side)
            icon_rect.moveCenter(QRectF(local).center())
            renderer.render(painter, icon_rect)
            return

        painter.drawText(local, Qt.AlignmentFlag.AlignCenter, tile.label)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        origin = self.mapToGlobal(self.rect().topLeft())
        painter.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        for tile in self.tiles:
            self._paint_tile(painter, tile, origin)


class KeyboardWindow(TileBoardWindow):
    """Dockable on-screen keyboard with language layouts, pages, window
    controls and a lock mode.

    The preview strip mirrors what has been typed through the gaze keyboard
    since entering the mode. The authoritative text lives in the OS-focused
    field (the user may also edit it by other means), so the strip is a
    feedback aid and the WPM source, not the source of truth.
    """

    committed = Signal(str) # emitted on ENTER with the preview content

    BUFFER_H = 60
    TILE_MAX = 130
    TILE_ABS_MIN = 48 # absolute pointer-target minimum (the only hard floor)
    SCALE_MIN, SCALE_MAX, SCALE_STEP = 0.5, 2.5, 0.1
    OPACITY_MIN, OPACITY_STEP = 0.35, 0.1
    DEFAULT_WIDTH_FRAC = 0.5 # board default width when docked top/bottom (~half screen)
    SIDE_DOCK_WIDTH_FRAC = 0.45 # board width cap when docked left/right

    def __init__(self, screen, min_tile_px: int):
        super().__init__(screen)
        self.min_tile_px = min_tile_px
        self.page = Page.LETTERS
        self.dock = DockSide.BOTTOM
        self.scale = 1.0
        self.opacity_level = 1.0
        self.locked = False
        self._layout_idx = 0
        self.buffer = ""
        self._first_key_t: Optional[float] = None
        self._unlock_tile: Optional[Tile] = None
        self._floor_unmet = False
        self._layout_tiles()

    @property
    def layout(self) -> KeyboardLayout:
        return KEYBOARD_LAYOUTS[self._layout_idx]

    def _rows(self) -> Optional[Tuple[str, ...]]:
        if self.page is Page.LETTERS:
            return self.layout.letter_rows
        if self.page is Page.SYMBOLS:
            return SYMBOL_ROWS
        return None # options page has no character rows

    def _control_specs(self) -> List[Tuple[str, str, int]]:
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

    OPTION_ROWS: Tuple[Tuple[Tuple[str, str], ...], ...] = (
        ((Ctl.DOCK_TOP, "TOP"), (Ctl.DOCK_BOTTOM, "BOTTOM"),
         (Ctl.DOCK_LEFT, "LEFT"), (Ctl.DOCK_RIGHT, "RIGHT")),
        ((Ctl.SIZE_DOWN, "SIZE -"), (Ctl.SIZE_UP, "SIZE +"),
         (Ctl.OPACITY_DOWN, "OPAC -"), (Ctl.OPACITY_UP, "OPAC +")),
        ((Ctl.CLEAR, "CLR"), (Ctl.LOCK, "LOCK")),
    )

    def _layout_tiles(self) -> None:
        geo = self._screen.geometry()
        rows = self._rows()
        control = self._control_specs()
        control_units = sum(units for _, _, units in control)
        if rows is not None:
            n_rows = len(rows) + 1
            n_cols = max(control_units, max(len(r) for r in rows))
        else:
            n_rows = len(self.OPTION_ROWS) + 1
            n_cols = control_units

        # Tile side: fit-to-screen base, scaled by the user. The accuracy
        # floor does NOT clamp - it only flags (below). The board defaults to
        # ~half the screen width, the user scale then always has real effect.
        if self.dock in (DockSide.LEFT, DockSide.RIGHT):
            base = min(int(geo.width() * self.SIDE_DOCK_WIDTH_FRAC) // n_cols, self.TILE_MAX)
        else:
            base = min(int(geo.width() * self.DEFAULT_WIDTH_FRAC) // n_cols, self.TILE_MAX)

        # Hard cap: the board must fit the screen no matter what the user scale
        # demands - a tile outside the display is useless.
        cap_w = (geo.width() - 20) // n_cols
        cap_h = (geo.height() - self.BUFFER_H - 20) // n_rows
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

        board_w = tile * n_cols
        board_h = tile * n_rows + self.BUFFER_H
        if self.dock is DockSide.BOTTOM:
            x0 = geo.x() + (geo.width() - board_w) // 2
            y0 = geo.y() + geo.height() - board_h - 10
        elif self.dock is DockSide.TOP:
            x0 = geo.x() + (geo.width() - board_w) // 2
            y0 = geo.y() + 10
        elif self.dock is DockSide.LEFT:
            x0 = geo.x() + 10
            y0 = geo.y() + (geo.height() - board_h) // 2
        else: # DockSide.RIGHT
            x0 = geo.x() + geo.width() - board_w - 10
            y0 = geo.y() + (geo.height() - board_h) // 2
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

    # gaze interface

    def hit_test(self, gx: int, gy: int) -> Optional[str]:
        if self.locked:
            if self._unlock_tile is not None and self._unlock_tile.rect.contains(gx, gy):
                return Ctl.UNLOCK
            return None
        return super().hit_test(gx, gy)

    def tile_center_norm(self, tile_id: str) -> Optional[NormalizedPoint]:
        if tile_id == Ctl.UNLOCK and self._unlock_tile is not None:
            geo = self._screen.geometry()
            c = self._unlock_tile.rect.center()
            return NormalizedPoint(c.x() / geo.width(), c.y() / geo.height())
        return super().tile_center_norm(tile_id)

    # preview strip ops

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

    def commit_preview(self) -> Tuple[str, float, float]:
        """Close a typing burst: returns (text, WPM, elapsed seconds) and
        resets the strip."""

        text, t0 = self.buffer, self._first_key_t
        elapsed = time.monotonic() - t0 if t0 else 0.0
        wpm = (len(text) / 5.0) / (elapsed / 60.0) if elapsed > 1e-3 else 0.0
        self.buffer, self._first_key_t = "", None
        self.update()
        return text, wpm, elapsed

    # window ops

    def switch_page(self, page: Page) -> None:
        self.page = page
        self._layout_tiles()

    def toggle_symbols(self) -> None:
        self.switch_page(Page.SYMBOLS if self.page is not Page.SYMBOLS else Page.LETTERS)

    def toggle_options(self) -> None:
        self.switch_page(Page.OPTIONS if self.page is not Page.OPTIONS else Page.LETTERS)

    def cycle_language(self) -> str:
        self._layout_idx = (self._layout_idx + 1) % len(KEYBOARD_LAYOUTS)
        if self.page is Page.OPTIONS:
            self.page = Page.LETTERS
        self._layout_tiles()
        return self.layout.code

    def set_dock(self, side: DockSide) -> None:
        self.dock = side
        self._layout_tiles()

    def resize_step(self, direction: int) -> float:
        self.scale = min(self.SCALE_MAX, max(self.SCALE_MIN, self.scale + direction * self.SCALE_STEP))
        self._layout_tiles()
        return self.scale

    def opacity_step(self, direction: int) -> float:
        self.opacity_level = min(1.0, max(self.OPACITY_MIN, self.opacity_level + direction * self.OPACITY_STEP))
        self.setWindowOpacity(self.opacity_level)
        return self.opacity_level

    def set_locked(self, locked: bool) -> None:
        self.locked = locked
        self.update()

    # painting

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Buffer panel: a rounded, slightly-lighter-than-tile card with an
        # accent outline, so the preview reads as a distinct surface.
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
    """Right-docked vertical strip for navigation mode: browser controls."""

    SPECS: Tuple[Tuple[str, str], ...] = (
        (Ctl.BACK, "BACK"), (Ctl.FORWARD, "FWD"), (Ctl.PAGE_UP, "PG UP"),
        (Ctl.PAGE_DOWN, "PG DN"), (Ctl.KEYBOARD, "KEYB"), (Ctl.EXIT, "EXIT"),
    )

    def __init__(self, screen, min_tile_px: int):
        super().__init__(screen)
        geo = screen.geometry()
        tile = min(max(min_tile_px, 90), (geo.height() - 20) // len(self.SPECS))
        x0 = geo.x() + geo.width() - tile - 8
        y0 = geo.y() + (geo.height() - tile * len(self.SPECS)) // 2
        self.setGeometry(x0, y0, tile + 4, tile * len(self.SPECS) + 4)
        self.tiles = [Tile(tid, label, QRect(x0, y0 + i * tile, tile, tile)) for i, (tid, label) in enumerate(self.SPECS)]


class GazeOverlay(QWidget):
    """Transparent, click-through fullscreen layer: cursor + feedback."""

    def __init__(self, screen):
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
        self.cursor_px: Optional[tuple] = None
        self.fixating = False
        self.progress = 0.0
        self.mode_name = ""
        self.scroll_zones = False
        self.scroll_active = 0 # +1 top / -1 bottom / 0
        self.scroll_zone_frac = 0.12

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()

        if self.scroll_zones: # edge-zone hints
            band = int(self.scroll_zone_frac * h)
            for sign, y in ((+1, 0), (-1, h - band)):
                col = QColor(ACCENT_ACTIVE if self.scroll_active == sign else ACCENT)
                col.setAlpha(70 if self.scroll_active == sign else 28)
                painter.fillRect(0, y, w, band, col)

        painter.setPen(QColor(255, 255, 255, 200))
        painter.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        painter.drawText(QRect(0, 6, w, 30), Qt.AlignmentFlag.AlignHCenter, f"GAZE MODE - {self.mode_name}")

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

    def enter(self) -> None: ...
    def exit(self) -> None: ...
    def on_activation(self, tile_id: str) -> Optional[ModeAction]:
        """Handle an activated tile. May return a ModeAction for the
        controller (mode switch / exit / OS click)."""
        ...


class KeyboardMode:
    name = "KEYBOARD"
    edge_scroll = False

    def __init__(self, keyboard: KeyboardWindow, drift: Optional[DriftCorrector],
                 actuator: SystemActuator):
        self.board = keyboard
        self._drift = drift
        self._actuator = actuator

    def enter(self) -> None:
        self.board.show()

    def exit(self) -> None:
        self.board.hide()

    def _apply_drift(self, tile_id: str) -> None:
        # Drift correction BEFORE mutating layout: the dwell frames in the
        # buffer are ground-truth labeled by this tile's centre.
        if self._drift is None:
            return
        center = self.board.tile_center_norm(tile_id)
        if center is not None:
            added = self._drift.on_activation(center)
            if added:
                rlog.event("drift_correction", tile=tile_id, n_samples=added, total=self._drift.corrections_applied)

    def on_activation(self, tile_id: str) -> Optional[ModeAction]:
        board = self.board
        if board.locked:
            if tile_id == Ctl.UNLOCK:
                board.set_locked(False)
                rlog.event("keyboard_locked", locked=False)
            return None

        if tile_id.startswith(KEY_PREFIX) or tile_id == Ctl.SPACE:
            self._apply_drift(tile_id)

        if tile_id.startswith(KEY_PREFIX):
            ch = tile_id[len(KEY_PREFIX):]
            board.type_preview(ch)
            self._actuator.type_text(ch)
        elif tile_id == Ctl.SPACE:
            board.type_preview(" ")
            self._actuator.type_text(" ")
        elif tile_id == Ctl.BACKSPACE:
            board.backspace_preview()
            self._actuator.press_key("backspace")
        elif tile_id == Ctl.DELETE_WORD:
            board.delete_word_preview()

            # Ctrl+Backspace deletes the previous word in most toolkits
            # (macOS uses Alt+Backspace - see the platform notes in README).
            self._actuator.hotkey("ctrl", "backspace")
        elif tile_id == Ctl.ENTER:
            text, wpm, elapsed = board.commit_preview()
            rlog.event(
                "text_committed",
                chars=len(text),
                elapsed_s=round(elapsed, 1),
                wpm=round(wpm, 1)
            )
            board.committed.emit(text)
            self._actuator.press_key("enter")
        elif tile_id == Ctl.PAGE:
            board.toggle_symbols()
        elif tile_id == Ctl.OPTIONS:
            board.toggle_options()
        elif tile_id == Ctl.LANG:
            code = board.cycle_language()
            rlog.event("keyboard_layout_changed", layout=code)
        elif tile_id == Ctl.DOCK_TOP:
            board.set_dock(DockSide.TOP)
        elif tile_id == Ctl.DOCK_BOTTOM:
            board.set_dock(DockSide.BOTTOM)
        elif tile_id == Ctl.DOCK_LEFT:
            board.set_dock(DockSide.LEFT)
        elif tile_id == Ctl.DOCK_RIGHT:
            board.set_dock(DockSide.RIGHT)
        elif tile_id == Ctl.SIZE_DOWN:
            board.resize_step(-1)
        elif tile_id == Ctl.SIZE_UP:
            board.resize_step(+1)
        elif tile_id == Ctl.OPACITY_DOWN:
            board.opacity_step(-1)
        elif tile_id == Ctl.OPACITY_UP:
            board.opacity_step(+1)
        elif tile_id == Ctl.CLEAR:
            board.clear_preview()
        elif tile_id == Ctl.LOCK:
            board.set_locked(True)
            rlog.event("keyboard_locked", locked=True)
        elif tile_id == Ctl.NAV:
            return ModeAction.TO_NAV
        elif tile_id == Ctl.EXIT:
            return ModeAction.EXIT

        if tile_id in (Ctl.DOCK_TOP, Ctl.DOCK_BOTTOM, Ctl.DOCK_LEFT, Ctl.DOCK_RIGHT):
            rlog.event("keyboard_dock_changed", dock=board.dock.value)
        return None


class NavigationMode:
    name = "NAVIGATION"
    edge_scroll = True

    def __init__(self, bar: BrowserBar, actuator: SystemActuator, dpr: float = 1.0):
        self.board = bar
        self._actuator = actuator
        self._dpr = dpr

    def enter(self) -> None:
        self.board.show()

    def exit(self) -> None:
        self.board.hide()

    def _scroll_at_content(self, ticks: int) -> None:
        # The OS wheel acts on the window under the real cursor, which after
        # an on-demand move may sit anywhere (including over this bar), so
        # park it at the screen centre before scrolling.
        geo = self.board._screen.geometry()
        cx, cy = geo.x() + geo.width() // 2, geo.y() + geo.height() // 2
        self._actuator.move(*to_physical(cx, cy, self._dpr))
        self._actuator.scroll(ticks)

    def on_activation(self, tile_id: str) -> Optional[ModeAction]:
        if tile_id == SCREEN_CLICK:
            return ModeAction.CLICK
        if tile_id == Ctl.BACK:
            self._actuator.hotkey("alt", "left")
        elif tile_id == Ctl.FORWARD:
            self._actuator.hotkey("alt", "right")
        elif tile_id == Ctl.PAGE_UP:
            self._scroll_at_content(+12)
        elif tile_id == Ctl.PAGE_DOWN:
            self._scroll_at_content(-12)
        elif tile_id == Ctl.KEYBOARD:
            return ModeAction.TO_KEYBOARD
        elif tile_id == Ctl.EXIT:
            return ModeAction.EXIT
        return None


class GazeInteractionController(QObject):
    """Per-frame interaction pipeline, owned by the main window after
    calibration. Feed it via ``on_gaze`` from the UI update slot."""

    stopped = Signal()

    def __init__(
        self,
        strategy: CalibrationStrategy,
        screen,
        activation: ActivationStrategy,
        actuator: SystemActuator,
        mean_error_px: Optional[float],
    ):
        super().__init__()
        self._screen = screen
        self._geo = screen.geometry()
        self._dpr = float(getattr(screen, "devicePixelRatio", lambda: 1.0)())
        tile_px = min_tile_side_px(mean_error_px)

        self.overlay = GazeOverlay(screen)
        self.activation = activation
        self.fixation = IDTFixationDetector()
        self.drift = DriftCorrector(strategy)
        self.keyboard = KeyboardWindow(screen, tile_px)
        self.edge = EdgeScrollDetector()
        self.overlay.scroll_zone_frac = self.edge.zone
        self.actuator = actuator
        self.bar = BrowserBar(screen, tile_px)

        self._modes: Dict[ModeAction, GazeMode] = {
            ModeAction.TO_KEYBOARD: KeyboardMode(self.keyboard, self.drift, actuator),
            ModeAction.TO_NAV: NavigationMode(self.bar, actuator, self._dpr),
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
        self.overlay.mode_name = self.mode.name
        self.overlay.show()
        self.mode.enter()

    # lifecycle

    def stop(self) -> None:
        self.mode.exit()
        self.overlay.hide()
        rlog.event("gaze_mode_stopped")
        self.stopped.emit()

    def _switch(self, action: ModeAction) -> None:
        self.mode.exit()
        self.activation.reset()
        self.fixation.reset()
        self.mode = self._modes[action]
        self.mode.enter()
        self.overlay.mode_name = self.mode.name
        rlog.event("mode_switched", mode=self.mode.name)

    # per-frame pipeline

    def on_gaze(self, gaze: Optional[NormalizedPoint], data: LandmarksData,
                vec: Optional[list]) -> None:
        t = time.monotonic()
        gx = gy = None
        if gaze is not None:
            gx, gy = gaze.x, gaze.y
        fix = self.fixation.update(gx, gy, t)

        # Hit-test at the fixation centroid when available (stable), else gaze.
        target: Optional[str] = None
        px = py = None
        probe = fix.centroid if fix.fixating else (gx, gy) if gx is not None else None
        if probe is not None:
            px = self._geo.x() + int(probe[0] * self._geo.width())
            py = self._geo.y() + int(probe[1] * self._geo.height())
            target = self.mode.board.hit_test(px, py)
            if target is None and self.mode.edge_scroll and self.edge.current_zone == 0:
                # Navigation mode: dwelling on plain screen is itself a target
                # (an OS click) - without a pseudo-target the dwell timer
                # would never arm off-board.
                target = SCREEN_CLICK

        if fix.fixating and target is not None and target.startswith((KEY_PREFIX, "ctl:")):
            self.drift.observe(vec)

        blink = extract_blink_score(data.blendshapes)
        event = self.activation.update(target, fix.fixating, blink, t)
        if event is not None:
            rlog.event(
                "tile_activated",
                tile=event.target_id,
                kind=event.kind.value,
                mode=self.mode.name
            )
            action = self.mode.on_activation(event.target_id)
            if action is ModeAction.EXIT:
                self.stop()
                return
            if action in (ModeAction.TO_NAV, ModeAction.TO_KEYBOARD):
                self._switch(action)
            elif action is ModeAction.CLICK and px is not None:
                self.actuator.move(*to_physical(px, py, self._dpr))
                self.actuator.click()

        if self.mode.edge_scroll:
            was_active = self.edge.active
            ticks = self.edge.update(gy, t)
            if ticks and gx is not None:
                # The OS scroll wheel acts on whatever is under the real
                # cursor, so move it under the current gaze point first.
                sx = self._geo.x() + int(gx * self._geo.width())
                sy = self._geo.y() + int(gy * self._geo.height())
                self.actuator.move(*to_physical(sx, sy, self._dpr))
                self.actuator.scroll(ticks)
            if self.edge.active and not was_active:
                rlog.event(
                    "scroll_gesture",
                    direction="up" if self.edge.current_zone > 0 else "down",
                    level=logging.DEBUG
                )
            self.overlay.scroll_zones = True
            self.overlay.scroll_active = self.edge.current_zone if self.edge.active else 0
        else:
            self.overlay.scroll_zones = False

        # Feedback rendering
        if gx is not None:
            self.overlay.cursor_px = (
                self._geo.x() + int(gx * self._geo.width()),
                self._geo.y() + int(gy * self._geo.height())
            )
        self.overlay.fixating = fix.fixating
        self.overlay.progress = self.activation.progress
        self.overlay.update()
        self.mode.board.set_highlight(target, self.activation.progress)