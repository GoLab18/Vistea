"""Tests for the post-calibration validation pass."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from collections.abc import Iterable
from typing import Protocol

import numpy as np

from core.calibration import NormalizedPoint, make_5point, make_grid
from core.evaluation import ValidationReport, compute_point_error
from main import VALIDATION_TARGETS


class PointLike(Protocol):
    x: float
    y: float


def _coords(points: Iterable[PointLike]) -> set[tuple[float, float]]:
    return {(round(p.x, 6), round(p.y, 6)) for p in points}


def _all_training_points() -> set[tuple[float, float]]:
    """Union of every training grid the system can fit on: 5-pt, 9-pt, 16-pt."""

    pts: set[tuple[float, float]] = set()
    pts |= _coords(make_5point())
    pts |= _coords(make_grid(3))  # 9-pt
    pts |= _coords(make_grid(4))  # 16-pt
    return pts


def test_validation_targets_disjoint_from_all_training_grids() -> None:
    training = _all_training_points()
    for target in VALIDATION_TARGETS:
        assert (round(target.x, 6), round(target.y, 6)) not in training, target


def test_central_validation_target_present_and_off_center() -> None:
    coords = _coords(VALIDATION_TARGETS)
    assert (0.45, 0.55) in coords
    assert (0.5, 0.5) not in coords


def test_central_target_is_off_the_16pt_grid_axes() -> None:
    axes = set(np.round(np.linspace(0.1, 0.9, 4), 6))
    assert round(0.45, 6) not in axes
    assert round(0.55, 6) not in axes


def test_report_counts_refused_points_instead_of_dropping_them() -> None:
    report = ValidationReport(strategy_name="test", screen_w_px=1000, screen_h_px=1000)
    report.points.append(
        compute_point_error(NormalizedPoint(0.5, 0.5), NormalizedPoint(0.6, 0.5), 1000, 1000)
    )
    report.n_refused = 2

    assert abs(report.mean_error_px - 100.0) < 1e-6  # Mean over what survived
    assert "2 refused" in report.summary()