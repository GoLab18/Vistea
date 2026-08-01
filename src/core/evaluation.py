"""
Accuracy evaluation and reporting metrics.

Calculates errors in pixels, screen percentage, and visual degrees (Holmqvist et al. 2011).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from core.calibration import NormalizedPoint

# Assumed viewing distance for converting on-screen error into visual degrees.
# While monocular webcams can approximate relative depth (e.g., via face bounding box size), 
# absolute depth estimation without IR/LiDAR sensors or camera-specific focal calibration is highly noisy.
# A fixed standard distance (60 cm) is therefore used to ensure metrics remain objective 
# and comparable against literature baselines (Holmqvist et al. 2011).
DEFAULT_VIEWING_DISTANCE_CM: float = 60.0


@dataclass
class PointError:
    target: NormalizedPoint
    predicted: NormalizedPoint
    error_px: float
    error_deg: float | None


@dataclass
class ValidationReport:
    """Accuracy report produced by the post-calibration validation pass."""

    strategy_name: str
    screen_w_px: int
    screen_h_px: int
    points: list[PointError] = field(default_factory=list)

    @property
    def mean_error_px(self) -> float:
        if not self.points:
            return float("nan")
        return sum(p.error_px for p in self.points) / len(self.points)

    @property
    def mean_error_deg(self) -> float | None:
        degs = [p.error_deg for p in self.points if p.error_deg is not None]
        if not degs:
            return None
        return sum(degs) / len(degs)

    @property
    def mean_error_percent(self) -> float:
        diag = math.hypot(self.screen_w_px, self.screen_h_px)
        if not diag:
            return float("nan")
        return 100.0 * self.mean_error_px / diag

    def summary(self) -> str:
        deg = self.mean_error_deg
        deg_txt = f", {deg:.2f} deg" if deg is not None else ""
        return (
            f"[{self.strategy_name}] mean accuracy: {self.mean_error_px:.0f} px "
            f"({self.mean_error_percent:.1f}% of diagonal{deg_txt}) over {len(self.points)} points"
        )


def compute_point_error(
    target: NormalizedPoint,
    predicted: NormalizedPoint,
    screen_w_px: int,
    screen_h_px: int,
    screen_w_mm: float | None = None,
    screen_h_mm: float | None = None,
    viewing_distance_cm: float = DEFAULT_VIEWING_DISTANCE_CM,
) -> PointError:
    """
    Computes spatial distance (magnitude) between a target and the model's prediction.
    Reduces the 2D error vector to a 1D scalar for aggregate reporting.
    """

    dx_px = (predicted.x - target.x) * screen_w_px
    dy_px = (predicted.y - target.y) * screen_h_px
    err_px = math.hypot(dx_px, dy_px)

    error_deg: float | None = None
    if screen_w_mm and screen_h_mm and screen_w_px and screen_h_px:
        mm_per_px_x = screen_w_mm / screen_w_px
        mm_per_px_y = screen_h_mm / screen_h_px
        err_cm = math.hypot(dx_px * mm_per_px_x, dy_px * mm_per_px_y) / 10.0
        error_deg = math.degrees(math.atan2(err_cm, viewing_distance_cm))

    return PointError(
        target=target, 
        predicted=predicted, 
        error_px=err_px, 
        error_deg=error_deg
    )