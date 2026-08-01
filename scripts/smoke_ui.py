"""Headless smoke test of ui_gaze keyboard logic on a minimal Qt stub.

Verifies (without PySide6): tile layout for both language layouts and all
pages, dock/scale/opacity/lock behavior, hit-testing under lock, and the
KeyboardMode routing into the system actuator.
"""

import sys
import types
from typing import Any

class QPointS:
    def __init__(self, x: int, y: int): self._x, self._y = x, y
    def x(self) -> int: return self._x
    def y(self) -> int: return self._y

class QRect:
    def __init__(self, *a: Any):
        if len(a) == 1 and isinstance(a[0], QRect):
            r = a[0]; self._x, self._y, self._w, self._h = r._x, r._y, r._w, r._h
        else:
            self._x, self._y, self._w, self._h = a
    def contains(self, x: int, y: int) -> bool:
        return self._x <= x < self._x + self._w and self._y <= y < self._y + self._h
    def center(self) -> QPointS: return QPointS(self._x + self._w // 2, self._y + self._h // 2)
    def translated(self, dx: int, dy: int) -> 'QRect': return QRect(self._x + dx, self._y + dy, self._w, self._h)
    def adjusted(self, a: int, b: int, c: int, d: int) -> 'QRect': return QRect(self._x + a, self._y + b, self._w + c - a, self._h + d - b)
    def moveCenter(self, p: QPointS) -> None:
        self._x = p.x() - self._w // 2; self._y = p.y() - self._h // 2
    def topLeft(self) -> QPointS: return QPointS(self._x, self._y)
    def width(self) -> int: return self._w
    def height(self) -> int: return self._h
    def x(self) -> int: return self._x
    def y(self) -> int: return self._y

class _Flags:
    def __getattr__(self, k: str) -> int: return 1
class _Qt:
    def __getattr__(self, k: str) -> _Flags: return _Flags()

class Signal:
    def __init__(self, *a: Any): pass
    def connect(self, *a: Any): pass
    def emit(self, *a: Any): pass

class QWidget:
    def __init__(self, *a: Any, **k: Any): self._geo = QRect(0, 0, 0, 0)
    def setWindowFlags(self, *a: Any): pass
    def setAttribute(self, *a: Any): pass
    def setGeometry(self, x: int, y: int, w: int, h: int): self._geo = QRect(x, y, w, h)
    def geometry(self) -> QRect: return self._geo
    def rect(self) -> QRect: return QRect(0, 0, self._geo.width(), self._geo.height())
    def mapToGlobal(self, p: QPointS) -> QPointS: return QPointS(self._geo.x() + p.x(), self._geo.y() + p.y())
    def width(self) -> int: return self._geo.width()
    def height(self) -> int: return self._geo.height()
    def update(self): pass
    def show(self): self.visible = True
    def hide(self): self.visible = False
    def setWindowOpacity(self, v: float): self.win_opacity = v

class QObject:
    def __init__(self, *a: Any, **k: Any): pass

def _stub(name: str, **attrs: Any) -> types.ModuleType:
    m = types.ModuleType(name)
    for k, v in attrs.items(): setattr(m, k, v)
    sys.modules[name] = m
    return m

_noop = lambda *a, **k: types.SimpleNamespace()


class QSvgRenderer:
    def __init__(self, *a: Any, **k: Any): pass
    def isValid(self) -> bool: return False
    def render(self, *a: Any) -> None: pass


pyside = _stub("PySide6")
pyside.QtCore = _stub(
    "PySide6.QtCore",
    QByteArray=_noop, QObject=QObject, QPoint=QPointS, QRect=QRect,
    QRectF=QRect, Qt=_Qt(), Signal=Signal,
)
pyside.QtGui = _stub(
    "PySide6.QtGui",
    QColor=_noop, QFont=_noop, QPaintEvent=object, QPainter=_noop, QPen=_noop,
    QScreen=object,
)
pyside.QtSvg = _stub("PySide6.QtSvg", QSvgRenderer=QSvgRenderer)
pyside.QtWidgets = _stub("PySide6.QtWidgets", QWidget=QWidget)

class _SLLogger:
    def log(self, *a: Any, **k: Any): pass
_sl = _stub("structlog")
_sl.contextvars = types.SimpleNamespace(
    bind_contextvars=lambda **k: None,
    unbind_contextvars=lambda *a: None,
    clear_contextvars=lambda: None,
    merge_contextvars=None
)
_sl.get_logger = lambda name=None: _SLLogger()

sys.path.insert(0, "src")
variant = sys.argv[1] if len(sys.argv) > 1 else "src"
sys.path.insert(0, variant)

from core import research_log as _rlog
_rlog.configure = lambda *a, **k: None

from ui_gaze import (Ctl, DockSide, KeyboardMode, KeyboardWindow, ModeAction, Page, KEY_PREFIX)
from core.actuator import NullActuator

class FakeScreen:
    def geometry(self) -> QRect: return QRect(0, 0, 1920, 1080)

scr = FakeScreen()
kb = KeyboardWindow(scr, 80) # type: ignore

ids = [t.tile_id for t in kb.tiles]
assert f"{KEY_PREFIX}q" in ids and Ctl.SPACE in ids and Ctl.DELETE_WORD in ids
assert kb.layout.code == "EN" and len([i for i in ids if i.startswith(KEY_PREFIX)]) == 29
nav = Ctl.__members__.get("NAV")
assert (nav in ids) == (nav is not None)

for t in kb.tiles:
    assert min(t.rect.width(), t.rect.height()) >= kb.TILE_ABS_MIN, t.tile_id

q = next(t for t in kb.tiles if t.tile_id == f"{KEY_PREFIX}q")
c = q.rect.center()
assert kb.hit_test(c.x(), c.y()) == f"{KEY_PREFIX}q"
assert kb.hit_test(5, 5) is None
cn = kb.tile_center_norm(f"{KEY_PREFIX}q")
assert cn is not None and 0 < cn.x < 1 and 0 < cn.y < 1

kb.cycle_language()
assert kb.layout.code == "PL"
ids = [t.tile_id for t in kb.tiles]
assert f"{KEY_PREFIX}ą" in ids and f"{KEY_PREFIX}ż" in ids
assert len([i for i in ids if i.startswith(KEY_PREFIX)]) == 38
kb.cycle_language()
assert kb.layout.code == "EN"

kb.toggle_symbols()
assert kb.page is Page.SYMBOLS
assert any(t.tile_id == f"{KEY_PREFIX}7" for t in kb.tiles)
kb.toggle_options()
assert kb.page is Page.OPTIONS
opt_ids = [t.tile_id for t in kb.tiles]
for need in (Ctl.DOCK_TOP, Ctl.SIZE_UP, Ctl.OPACITY_DOWN, Ctl.CLEAR, Ctl.LOCK, Ctl.EXIT):
    assert need in opt_ids, need
kb.toggle_options()
assert kb.page is Page.LETTERS

kb.set_dock(DockSide.TOP)
assert kb.geometry().y() == 10
kb.set_dock(DockSide.RIGHT)
geo = kb.geometry()
units = sum(u for _, _, u in kb._control_specs())

assert geo.x() + geo.width() >= 1920 - 20
assert geo.width() <= 1920 - 20 + 1
kb.set_dock(DockSide.BOTTOM)
assert kb.geometry().y() + kb.geometry().height() >= 1080 - 20

w0 = kb.geometry().width()
kb.resize_step(+1)
assert kb.geometry().width() > w0
for _ in range(20):
    kb.resize_step(-1)
assert kb.scale == kb.SCALE_MIN
for t in kb.tiles:
    assert min(t.rect.width(), t.rect.height()) >= kb.TILE_ABS_MIN
for _ in range(20):
    kb.opacity_step(-1)
assert abs(kb.opacity_level - kb.OPACITY_MIN) < 1e-9 and kb.win_opacity == kb.opacity_level
kb.resize_step(+4 if False else +1)

kb.set_locked(True)
sp = next(t for t in kb.tiles if t.tile_id == Ctl.SPACE).rect.center()
assert kb.hit_test(sp.x(), sp.y()) is None 
uc = kb._unlock_tile.rect.center()
assert kb.hit_test(uc.x(), uc.y()) == Ctl.UNLOCK
assert kb.tile_center_norm(Ctl.UNLOCK) is not None

act = NullActuator()
mode = KeyboardMode(kb, None, act) # type: ignore
assert mode.on_activation(f"{KEY_PREFIX}a") is None 
assert act.calls == []
mode.on_activation(Ctl.UNLOCK)
assert kb.locked is False

mode.on_activation(f"{KEY_PREFIX}a")
mode.on_activation(Ctl.SPACE)
mode.on_activation(f"{KEY_PREFIX}ó")
assert kb.buffer == "a ó"
mode.on_activation(Ctl.BACKSPACE)
assert kb.buffer == "a "
mode.on_activation(f"{KEY_PREFIX}b")
mode.on_activation(Ctl.DELETE_WORD)
assert kb.buffer == "a "
enter_ret = mode.on_activation(Ctl.ENTER)
assert enter_ret is None and kb.buffer == ""
assert ("type_text", "a") in act.calls and ("type_text", " ") in act.calls
assert ("type_text", "ó") in act.calls
assert ("press_key", "backspace") in act.calls
assert ("hotkey", "ctrl", "backspace") in act.calls
assert ("press_key", "enter") in act.calls

mode.on_activation(Ctl.OPTIONS)
assert kb.page is Page.OPTIONS
mode.on_activation(Ctl.DOCK_LEFT)
assert kb.dock is DockSide.LEFT
mode.on_activation(Ctl.CLEAR)
assert kb.buffer == ""

assert Ctl.PIN in {t.tile_id for t in kb.tiles}
assert kb.pinned is False
mode.on_activation(Ctl.PIN)
assert kb.pinned is True
mode.on_activation(Ctl.PIN)
assert kb.pinned is False
assert mode.on_activation(Ctl.EXIT) is ModeAction.EXIT
if nav is not None:
    assert mode.on_activation(nav) is ModeAction.TO_NAV

print(f"SMOKE OK ({variant}): layouts, pages, dock, scale, lock, actuator routing")