"""Screen work-area reservation for the pinned on-screen keyboard.

When the keyboard is pinned, other windows that maximize or snap must make room
for it insteadof being covered. That is not a window flag - it requires telling
the OS to shrink the usable work area along one screen edge. The mechanism is
per platform.
"""

from __future__ import annotations

import ctypes
import logging
import sys
from typing import Final, Literal, Protocol, runtime_checkable

from core import research_log as rlog

DockEdge = Literal["top", "bottom", "left", "right"]


@runtime_checkable
class WorkAreaReservation(Protocol):
    name: str
    active: bool

    def reserve(self, edge: DockEdge, rect: tuple[int, int, int, int]) -> bool:
        """Reserve a strip of the desktop along the `edge`.
        `rect` is the strip in physical pixels (x, y, w, h).
        Returns True on success."""
        ...

    def release(self) -> None:
        """Give the reserved strip back to the desktop."""
        ...


class NullReservation:
    """Records calls instead of performing them (tests / unsupported
    platform / missing window handle)."""

    name = "null"

    def __init__(self) -> None:
        self.calls: list[tuple] = []
        self.active: bool = False

    def reserve(self, edge: DockEdge, rect: tuple[int, int, int, int]) -> bool:
        self.calls.append(("reserve", edge, rect))
        self.active = True
        return True

    def release(self) -> None:
        self.calls.append(("release",))
        self.active = False


# Windows shell AppBar constants (SHAppBarMessage messages and edges)
_ABM_NEW: Final[int] = 0x0
_ABM_REMOVE: Final[int] = 0x1
_ABM_QUERYPOS: Final[int] = 0x2
_ABM_SETPOS: Final[int] = 0x3

_ABE_LEFT: Final[int] = 0
_ABE_TOP: Final[int] = 1
_ABE_RIGHT: Final[int] = 2
_ABE_BOTTOM: Final[int] = 3

_EDGE_TO_ABE: Final[dict[DockEdge, int]] = {
    "left": _ABE_LEFT,
    "top": _ABE_TOP,
    "right": _ABE_RIGHT,
    "bottom": _ABE_BOTTOM,
}


class _RECT(ctypes.Structure):
    _fields_ = [
        ("left", ctypes.c_long),
        ("top", ctypes.c_long),
        ("right", ctypes.c_long),
        ("bottom", ctypes.c_long),
    ]


class _APPBARDATA(ctypes.Structure):
    _fields_ = [
        ("cbSize", ctypes.c_ulong),
        ("hWnd", ctypes.c_void_p),
        ("uCallbackMessage", ctypes.c_uint),
        ("uEdge", ctypes.c_uint),
        ("rc", _RECT),
        ("lParam", ctypes.c_ssize_t),
    ]


def _make_appbardata(hwnd: int, edge: int, rect: tuple[int, int, int, int]) -> _APPBARDATA:
    """Build an APPBARDATA for the given window, edge and physical-pixel
    strip (x, y, w, h). Split out so the struct assembly can be tested
    without any real WinAPI call."""
    
    x, y, w, h = rect
    data = _APPBARDATA()
    data.cbSize = ctypes.sizeof(_APPBARDATA)
    data.hWnd = hwnd
    data.uCallbackMessage = 0
    data.uEdge = edge
    data.rc = _RECT(x, y, x + w, y + h)
    data.lParam = 0
    return data


class WindowsAppBarReservation:
    """Reserves a desktop strip through the Windows shell AppBar API."""

    name = "windows_appbar"

    def __init__(self, hwnd: int) -> None:
        self._hwnd: int = hwnd
        self._registered: bool = False
        self.active: bool = False
        self._shell = ctypes.windll.shell32

    def reserve(self, edge: DockEdge, rect: tuple[int, int, int, int]) -> bool:
        abe = _EDGE_TO_ABE.get(edge, _ABE_BOTTOM)
        try:
            if not self._registered:
                new = _make_appbardata(self._hwnd, abe, rect)
                self._shell.SHAppBarMessage(_ABM_NEW, ctypes.byref(new))
                self._registered = True

            data = _make_appbardata(self._hwnd, abe, rect)

            self._shell.SHAppBarMessage(_ABM_QUERYPOS, ctypes.byref(data))
            self._shell.SHAppBarMessage(_ABM_SETPOS, ctypes.byref(data))
            self.active = True
            return True
        except (OSError, AttributeError) as exc:
            rlog.event(
                "workarea_error",
                reason=f"AppBar reservation failed ({exc}); leaving work area unchanged",
                level=logging.WARNING,
            )
            return False

    def release(self) -> None:
        if not self._registered:
            return
        try:
            data = _make_appbardata(self._hwnd, _ABE_BOTTOM, (0, 0, 0, 0))
            self._shell.SHAppBarMessage(_ABM_REMOVE, ctypes.byref(data))
        except (OSError, AttributeError) as exc:
            rlog.event(
                "workarea_error",
                reason=f"AppBar release failed ({exc})",
                level=logging.WARNING,
            )
        finally:
            self._registered = False
            self.active = False


def make_reservation(hwnd: int | None) -> WorkAreaReservation:
    """Pick a reservation backend based on platform capabilities."""

    if sys.platform == "win32" and hwnd is not None:
        try:
            return WindowsAppBarReservation(hwnd)
        except (OSError, AttributeError) as exc:
            rlog.event(
                "workarea_unavailable",
                reason=f"Windows AppBar unavailable ({exc}); using null reservation",
                level=logging.WARNING,
            )
            return NullReservation()
            
    rlog.event(
        "workarea_unavailable",
        reason="not available on this platform; using null reservation",
        level=logging.INFO,
    )
    return NullReservation()