import math
from typing import Tuple, Optional, Protocol


class PointFilter(Protocol):
    """Protocol defining the interface for coordinate smoothing filters."""

    def filter(self, current_point: Tuple[float, float], timestamp_ms: int) -> Tuple[float, float]:
        ...

    def reset(self) -> None:
        ...


class NoFilter:
    """Pass-through strategy - useful as an experimental baseline."""

    def filter(self, current_point: Tuple[float, float], timestamp_ms: int) -> Tuple[float, float]:
        return current_point

    def reset(self) -> None:
        pass


class EMAFilter:
    """Applies a simple Exponential Moving Average low-pass filter."""

    def __init__(self, alpha: float = 0.5):
        self.alpha = alpha
        self._prev: Optional[Tuple[float, float]] = None

    def reset(self) -> None:
        self._prev = None

    def filter(self, current_point: Tuple[float, float], timestamp_ms: int) -> Tuple[float, float]:
        if self._prev is None:
            self._prev = current_point
            return current_point

        new_x = current_point[0] * self.alpha + self._prev[0] * (1.0 - self.alpha)
        new_y = current_point[1] * self.alpha + self._prev[1] * (1.0 - self.alpha)

        self._prev = (new_x, new_y)
        return self._prev


class OneEuroFilter:
    """
    Applies the 1 Euro Filter algorithm for noisy input.
    Paper: Casiez et al., "1 Euro Filter: A Simple Speed-based Low-pass Filter
    for Noisy Input in Interactive Systems" (CHI 2012).

    The filter is applied to the *predicted gaze point* in normalized screen
    coordinates (0-1), not to the raw iris features. In these units a saccade
    across the screen produces velocities of several units per second, which
    gives the speed term (beta * |v|) a meaningful dynamic range. Tuning
    procedure recommended by the authors:
      1. Set beta = 0 and adjust min_cutoff until jitter disappears during
         a steady fixation.
      2. Increase beta until lag disappears during fast gaze shifts.
    """

    def __init__(self, min_cutoff: float = 0.8, beta: float = 0.5, d_cutoff: float = 1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff

        self._prev_point: Optional[Tuple[float, float]] = None
        self._dx_prev: Tuple[float, float] = (0.0, 0.0)
        self._prev_timestamp_ms: Optional[int] = None

    def reset(self) -> None:
        self._prev_point = None
        self._dx_prev = (0.0, 0.0)
        self._prev_timestamp_ms = None

    def _get_alpha(self, cutoff: float, dt: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def filter(self, current_point: Tuple[float, float], timestamp_ms: int) -> Tuple[float, float]:
        if self._prev_point is None or self._prev_timestamp_ms is None:
            self._prev_point = current_point
            self._prev_timestamp_ms = timestamp_ms
            return current_point

        dt = (timestamp_ms - self._prev_timestamp_ms) / 1000.0
        if dt <= 0.0:
            return self._prev_point

        dx = (current_point[0] - self._prev_point[0]) / dt
        dy = (current_point[1] - self._prev_point[1]) / dt

        alpha_d = self._get_alpha(self.d_cutoff, dt)
        dx_hat = dx * alpha_d + self._dx_prev[0] * (1.0 - alpha_d)
        dy_hat = dy * alpha_d + self._dx_prev[1] * (1.0 - alpha_d)
        self._dx_prev = (dx_hat, dy_hat)

        velocity_magnitude = math.sqrt(dx_hat**2 + dy_hat**2)
        cutoff = self.min_cutoff + self.beta * velocity_magnitude

        alpha = self._get_alpha(cutoff, dt)
        new_x = current_point[0] * alpha + self._prev_point[0] * (1.0 - alpha)
        new_y = current_point[1] * alpha + self._prev_point[1] * (1.0 - alpha)

        self._prev_point = (new_x, new_y)
        self._prev_timestamp_ms = timestamp_ms
        return self._prev_point