"""Screen work-area reservation tests."""

from __future__ import annotations

from core.workarea import (
    NullReservation,
    WorkAreaReservation,
    _APPBARDATA,
    _make_appbardata,
    is_wayland_session,
    make_reservation,
    strut_values,
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


def test_strut_values_reserve_only_the_docked_edge() -> None:
    values = strut_values("bottom", (0, 990, 1920, 90))

    assert len(values) == 12
    assert values[:4] == (0, 0, 0, 90)
    assert values[10:] == (0, 1919)  # Inclusive extent along the bottom edge


def test_strut_values_for_each_edge_pick_the_right_thickness() -> None:
    assert strut_values("top", (0, 0, 1920, 60))[:4] == (0, 0, 60, 0)
    assert strut_values("left", (0, 0, 200, 1080))[:4] == (200, 0, 0, 0)
    assert strut_values("right", (1720, 0, 200, 1080))[:4] == (0, 200, 0, 0)


def test_side_struts_span_the_vertical_extent_of_the_strip() -> None:
    values = strut_values("left", (0, 100, 200, 500))
    assert values[4:6] == (100, 599)
    assert values[6:8] == (0, 0)


def test_wayland_detected_from_either_environment_marker() -> None:
    assert is_wayland_session({"XDG_SESSION_TYPE": "wayland"}) is True
    assert is_wayland_session({"WAYLAND_DISPLAY": "wayland-0"}) is True
    assert is_wayland_session({"XDG_SESSION_TYPE": "x11"}) is False
    assert is_wayland_session({}) is False
