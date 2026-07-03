import math
from dataclasses import dataclass, field
from typing import List, Optional

from core.calibration import NormalizedPoint

# Assumed viewing distance for converting on-screen error into visual
# degrees - the standard reporting unit in eye-tracking literature
# (e.g. Labvanced reports 1.3 deg, WebGazer ~4.17 deg). Can be overridden.
DEFAULT_VIEWING_DISTANCE_CM: float = 60.0


@dataclass
class PointError:
    target: NormalizedPoint
    predicted: NormalizedPoint
    error_px: float
    error_deg: Optional[float]


@dataclass
class ValidationReport:
    """Accuracy report produced by the post-calibration validation pass."""

    strategy_name: str
    screen_w_px: int
    screen_h_px: int
    points: List[PointError] = field(default_factory=list)

    @property
    def mean_error_px(self) -> float:
        return sum(p.error_px for p in self.points) / len(self.points) if self.points else float("nan")

    @property
    def mean_error_deg(self) -> Optional[float]:
        degs = [p.error_deg for p in self.points if p.error_deg is not None]
        return sum(degs) / len(degs) if degs else None

    @property
    def mean_error_percent(self) -> float:
        diag = math.hypot(self.screen_w_px, self.screen_h_px)
        return 100.0 * self.mean_error_px / diag if diag else float("nan")

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
    screen_w_mm: Optional[float] = None,
    screen_h_mm: Optional[float] = None,
    viewing_distance_cm: float = DEFAULT_VIEWING_DISTANCE_CM,
) -> PointError:
    dx_px = (predicted.x - target.x) * screen_w_px
    dy_px = (predicted.y - target.y) * screen_h_px
    err_px = math.hypot(dx_px, dy_px)

    error_deg: Optional[float] = None
    if screen_w_mm and screen_h_mm and screen_w_px and screen_h_px:
        mm_per_px_x = screen_w_mm / screen_w_px
        mm_per_px_y = screen_h_mm / screen_h_px
        err_cm = math.hypot(dx_px * mm_per_px_x, dy_px * mm_per_px_y) / 10.0
        error_deg = math.degrees(math.atan2(err_cm, viewing_distance_cm))

    return PointError(target=target, predicted=predicted, error_px=err_px, error_deg=error_deg)