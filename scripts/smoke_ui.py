"""Headless smoke test of ui_gaze keyboard logic on a minimal Qt stub.

Verifies (without PySide6): tile layout for both language layouts and all
pages, dock/scale/opacity/lock behavior, hit-testing under lock, and the
KeyboardMode routing into the system actuator (typing into OS focus).
"""

import sys, types

# Qt stub

class QPointS:
    def __init__(self, x, y): self._x, self._y = x, y
    def x(self): return self._x
    def y(self): return self._y

class QRect:
    def __init__(self, *a):
        if len(a) == 1 and isinstance(a[0], QRect):
            r = a[0]; self._x, self._y, self._w, self._h = r._x, r._y, r._w, r._h
        else:
            self._x, self._y, self._w, self._h = a
    def contains(self, x, y):
        return self._x <= x < self._x + self._w and self._y <= y < self._y + self._h
    def center(self): return QPointS(self._x + self._w // 2, self._y + self._h // 2)
    def translated(self, dx, dy): return QRect(self._x + dx, self._y + dy, self._w, self._h)
    def adjusted(self, a, b, c, d): return QRect(self._x + a, self._y + b, self._w + c - a, self._h + d - b)
    def moveCenter(self, p):
        self._x = p.x() - self._w // 2; self._y = p.y() - self._h // 2
    def topLeft(self): return QPointS(self._x, self._y)
    def width(self): return self._w
    def height(self): return self._h
    def x(self): return self._x
    def y(self): return self._y

class _Flags:
    def __getattr__(self, k): return 1
class _Qt:
    def __getattr__(self, k): return _Flags()

class Signal:
    def __init__(self, *a): pass
    def connect(self, *a): pass
    def emit(self, *a): pass

class QWidget:
    def __init__(self, *a, **k): self._geo = QRect(0, 0, 0, 0)
    def setWindowFlags(self, *a): pass
    def setAttribute(self, *a): pass
    def setGeometry(self, x, y, w, h): self._geo = QRect(x, y, w, h)
    def geometry(self): return self._geo
    def rect(self): return QRect(0, 0, self._geo.width(), self._geo.height())
    def mapToGlobal(self, p): return QPointS(self._geo.x() + p.x(), self._geo.y() + p.y())
    def width(self): return self._geo.width()
    def height(self): return self._geo.height()
    def update(self): pass
    def show(self): self.visible = True
    def hide(self): self.visible = False
    def setWindowOpacity(self, v): self.win_opacity = v

class QObject:
    def __init__(self, *a, **k): pass

def _stub(name, **attrs):
    m = types.ModuleType(name)
    for k, v in attrs.items(): setattr(m, k, v)
    sys.modules[name] = m
    return m

_noop = lambda *a, **k: types.SimpleNamespace()
pyside = _stub("PySide6")
pyside.QtCore = _stub("PySide6.QtCore", QObject=QObject, QRect=QRect, Qt=_Qt(), Signal=Signal)
pyside.QtGui = _stub("PySide6.QtGui", QColor=_noop, QFont=_noop, QPainter=_noop, QPen=_noop)
pyside.QtWidgets = _stub("PySide6.QtWidgets", QWidget=QWidget)

# structlog stub (dependency absent in this container, event() no-op)
class _SLLogger:
    def log(self, *a, **k): pass
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
_rlog.configure = lambda *a, **k: None # skip real sink setup in the stubbed run
from ui_gaze import (Ctl, DockSide, KeyboardMode, KeyboardWindow, ModeAction, Page, KEY_PREFIX)
from core.actuator import NullActuator


class FakeScreen:
    def geometry(self): return QRect(0, 0, 1920, 1080)


scr = FakeScreen()
kb = KeyboardWindow(scr, 80)

# layout invariants

ids = [t.tile_id for t in kb.tiles]
assert f"{KEY_PREFIX}q" in ids and Ctl.SPACE in ids and Ctl.DELETE_WORD in ids
assert kb.layout.code == "EN" and len([i for i in ids if i.startswith(KEY_PREFIX)]) == 29
nav = Ctl.__members__.get("NAV")
assert (nav in ids) == (nav is not None)

# every tile is at least the absolute pointer-target minimum on its shorter
# side (the accuracy floor is advisory now, not a hard clamp)
for t in kb.tiles:
    assert min(t.rect.width(), t.rect.height()) >= kb.TILE_ABS_MIN, t.tile_id

# hit test + drift centre

q = next(t for t in kb.tiles if t.tile_id == f"{KEY_PREFIX}q")
c = q.rect.center()
assert kb.hit_test(c.x(), c.y()) == f"{KEY_PREFIX}q"
assert kb.hit_test(5, 5) is None
cn = kb.tile_center_norm(f"{KEY_PREFIX}q")
assert 0 < cn.x < 1 and 0 < cn.y < 1

# language cycling: PL gains a diacritics row

kb.cycle_language()
assert kb.layout.code == "PL"
ids = [t.tile_id for t in kb.tiles]
assert f"{KEY_PREFIX}ą" in ids and f"{KEY_PREFIX}ż" in ids
assert len([i for i in ids if i.startswith(KEY_PREFIX)]) == 38
kb.cycle_language()
assert kb.layout.code == "EN"

# pages

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

# docking

kb.set_dock(DockSide.TOP)
assert kb.geometry().y() == 10
kb.set_dock(DockSide.RIGHT)
geo = kb.geometry()
units = sum(u for _, _, u in kb._control_specs())
# board width is bounded by the screen cap (the accuracy floor is advisory,
# it no longer clamps the size)
assert geo.x() + geo.width() >= 1920 - 20
assert geo.width() <= 1920 - 20 + 1
kb.set_dock(DockSide.BOTTOM)
assert kb.geometry().y() + kb.geometry().height() >= 1080 - 20

# scale / opacity

w0 = kb.geometry().width()
kb.resize_step(+1)
assert kb.geometry().width() > w0
for _ in range(20):
    kb.resize_step(-1)
assert kb.scale == kb.SCALE_MIN
for t in kb.tiles: # tiles stay above the absolute minimum (floor is advisory)
    assert min(t.rect.width(), t.rect.height()) >= kb.TILE_ABS_MIN
for _ in range(20):
    kb.opacity_step(-1)
assert abs(kb.opacity_level - kb.OPACITY_MIN) < 1e-9 and kb.win_opacity == kb.opacity_level
kb.resize_step(+4 if False else +1)

# lock

kb.set_locked(True)
sp = next(t for t in kb.tiles if t.tile_id == Ctl.SPACE).rect.center()
assert kb.hit_test(sp.x(), sp.y()) is None # board is inert
uc = kb._unlock_tile.rect.center()
assert kb.hit_test(uc.x(), uc.y()) == Ctl.UNLOCK # except the unlock tile
assert kb.tile_center_norm(Ctl.UNLOCK) is not None

# KeyboardMode routing into the actuator

act = NullActuator()
mode = KeyboardMode(kb, None, act)
assert mode.on_activation(f"{KEY_PREFIX}a") is None # locked: ignored
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
assert mode.on_activation(Ctl.EXIT) is ModeAction.EXIT
if nav is not None:
    assert mode.on_activation(nav) is ModeAction.TO_NAV

print(f"SMOKE OK ({variant}): layouts, pages, dock, scale, lock, actuator routing")