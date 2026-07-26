"""Screen work-area reservation tests."""

from __future__ import annotations

from core.workarea import (
    NullReservation,
    WorkAreaReservation,
    _APPBARDATA,
    _make_appbardata,
    make_reservation,
)


def test_null_reservation_records_reserve_and_release() -> None:
    res = NullReservation()
    assert res.active is False
    assert res.reserve("bottom", (0, 1000, 1920, 90)) is True
    assert res.active is True
    assert res.calls == [("reserve", "bottom", (0, 1000, 1920, 90))]
    res.release()
    assert res.active is False
    assert res.calls[-1] == ("release",)


def test_null_reservation_satisfies_protocol() -> None:
    assert isinstance(NullReservation(), WorkAreaReservation)


def test_make_reservation_without_hwnd_is_null() -> None:
    res = make_reservation(None)
    assert isinstance(res, NullReservation)


def test_make_appbardata_fills_rect_and_size() -> None:
    data = _make_appbardata(0x1234, 3, (10, 20, 100, 200))
    assert data.cbSize == __import__("ctypes").sizeof(_APPBARDATA)
    assert data.uEdge == 3
    assert data.uCallbackMessage == 0
    assert (data.rc.left, data.rc.top, data.rc.right, data.rc.bottom) == (10, 20, 110, 220)