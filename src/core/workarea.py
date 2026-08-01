"""Screen work-area reservation for the pinned on-screen keyboard.

When the keyboard is pinned, other windows that maximize or snap must make room
for it instead of being covered. It requires telling the OS to shrink the usable
work area along one screen edge, and every window system does that differently."""

from __future__ import annotations

import ctypes
import logging
import os
import sys
from collections.abc import Mapping, Sequence
from typing import Final, Literal, Protocol, assert_never, runtime_checkable

from core import research_log as rlog

type DockEdge = Literal["top", "bottom", "left", "right"]


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
                reason=f"AppBar reservation failed ({exc}): leaving work area unchanged",
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


# X11 / EWMH constants for _NET_WM_STRUT_PARTIAL
_XA_CARDINAL: Final[int] = 6
_PROP_MODE_REPLACE: Final[int] = 0
_PROP_FORMAT_32: Final[int] = 32


def strut_values(edge: DockEdge, rect: tuple[int, int, int, int]) -> tuple[int, ...]:
    """Build the twelve cardinals of `_NET_WM_STRUT_PARTIAL`."""

    x, y, w, h = rect

    match edge:
        case "left":
            return (w, 0, 0, 0, y, y + h - 1, 0, 0, 0, 0, 0, 0)
        case "right":
            return (0, w, 0, 0, 0, 0, y, y + h - 1, 0, 0, 0, 0)
        case "top":
            return (0, 0, h, 0, 0, 0, 0, 0, x, x + w - 1, 0, 0)
        case "bottom":
            return (0, 0, 0, h, 0, 0, 0, 0, 0, 0, x, x + w - 1)
        case _:
            assert_never(edge)


def is_wayland_session(env: Mapping[str, str] | None = None) -> bool:
    environ = os.environ if env is None else env
    if environ.get("XDG_SESSION_TYPE", "").lower() == "wayland":
        return True
    return bool(environ.get("WAYLAND_DISPLAY"))


class X11StrutReservation:
    """
    Reserves a desktop strip through the EWMH strut properties.

    The window advertises `_NET_WM_STRUT_PARTIAL` on itself and the window manager
    subtracts that band from the area it gives to maximized windows. The older
    `_NET_WM_STRUT` is set alongside it for managers that never implemented the
    partial variant.
    """

    name = "x11_strut"

    def __init__(self, window_id: int) -> None:
        self._window = window_id
        self.active: bool = False

        self._lib = ctypes.CDLL("libX11.so.6")
        self._lib.XOpenDisplay.restype = ctypes.c_void_p
        self._lib.XOpenDisplay.argtypes = [ctypes.c_char_p]
        self._lib.XInternAtom.restype = ctypes.c_ulong
        self._lib.XInternAtom.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int]
        self._lib.XChangeProperty.argtypes = [
            ctypes.c_void_p, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong,
            ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_int,
        ]
        self._lib.XDeleteProperty.argtypes = [ctypes.c_void_p, ctypes.c_ulong, ctypes.c_ulong]
        self._lib.XFlush.argtypes = [ctypes.c_void_p]
        self._lib.XCloseDisplay.argtypes = [ctypes.c_void_p]

        display = self._lib.XOpenDisplay(None)
        if not display:
            raise OSError("cannot open X display")
        self._display = display

    def _atom(self, name: str) -> int:
        return int(self._lib.XInternAtom(self._display, name.encode("ascii"), False))

    def _set_cardinals(self, atom_name: str, values: Sequence[int]) -> None:
        payload = (ctypes.c_ulong * len(values))(*values)
        self._lib.XChangeProperty(
            self._display,
            self._window,
            self._atom(atom_name),
            _XA_CARDINAL,
            _PROP_FORMAT_32,
            _PROP_MODE_REPLACE,
            ctypes.byref(payload),
            len(values),
        )

    def reserve(self, edge: DockEdge, rect: tuple[int, int, int, int]) -> bool:
        values = strut_values(edge, rect)
        try:
            self._set_cardinals("_NET_WM_STRUT_PARTIAL", values)
            self._set_cardinals("_NET_WM_STRUT", values[:4])
            self._lib.XFlush(self._display)
            self.active = True
            return True
        except (OSError, AttributeError) as exc:
            rlog.event(
                "workarea_error",
                reason=f"X11 strut reservation failed ({exc}): leaving work area unchanged",
                level=logging.WARNING,
            )
            return False

    def release(self) -> None:
        if not self.active:
            return
        try:
            self._lib.XDeleteProperty(self._display, self._window, self._atom("_NET_WM_STRUT_PARTIAL"))
            self._lib.XDeleteProperty(self._display, self._window, self._atom("_NET_WM_STRUT"))
            self._lib.XFlush(self._display)
        except (OSError, AttributeError) as exc:
            rlog.event(
                "workarea_error",
                reason=f"X11 strut release failed ({exc})",
                level=logging.WARNING,
            )
        finally:
            self.active = False


def make_reservation(window_handle: int | None) -> WorkAreaReservation:
    """Pick a reservation backend based on platform capabilities."""

    if window_handle is not None:
        if sys.platform == "win32":
            try:
                return WindowsAppBarReservation(window_handle)
            except (OSError, AttributeError) as exc:
                rlog.event(
                    "workarea_unavailable",
                    reason=f"Windows AppBar unavailable ({exc}): using null reservation",
                    level=logging.WARNING,
                )
                return NullReservation()

        if sys.platform.startswith("linux"):
            if is_wayland_session():
                rlog.event(
                    "workarea_unavailable",
                    reason=(
                        "Wayland session: work-area reservation needs the layer-shell "
                        "protocol, which Qt does not expose: keyboard stays on top"
                    ),
                    level=logging.INFO,
                )
                return NullReservation()
            try:
                return X11StrutReservation(window_handle)
            except (OSError, AttributeError) as exc:
                rlog.event(
                    "workarea_unavailable",
                    reason=f"X11 struts unavailable ({exc}): using null reservation",
                    level=logging.WARNING,
                )
                return NullReservation()

        if sys.platform == "darwin":
            rlog.event(
                "workarea_unavailable",
                reason=(
                    "macOS reserves the work area for the menu bar and Dock only: "
                    "no public API exposes it to applications, so the keyboard "
                    "stays a floating always-on-top panel"
                ),
                level=logging.INFO,
            )
            return NullReservation()

    rlog.event(
        "workarea_unavailable",
        reason="not available on this platform: using null reservation",
        level=logging.INFO,
    )
    return NullReservation()