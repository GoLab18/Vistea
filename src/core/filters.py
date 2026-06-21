import math
from typing import Tuple, Optional, Protocol

class PointFilter(Protocol):
    """Protocol defining the interface for coordinate smoothing filters."""

    def filter(self, current_point: Tuple[float, float], timestamp_ms: int) -> Tuple[float, float]:
        ...

class EMAFilter:
    """Applies a simple Exponential Moving Average low-pass filter."""

    def __init__(self, alpha: float = 0.5):
        self.alpha = alpha
        self._prev: Optional[Tuple[float, float]] = None

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
    Paper: Casiez et al., "1€ Filter: A Simple Speed-based Low-pass Filter for Noisy Input in Interactive Systems" (2012).
    
    Implementation details:
    This filter adapts its smoothing level dynamically based on the current velocity of the iris.
    - At low speeds (fixation), it heavily filters spatial jitter using 'min_cutoff'.
    - At high speeds (saccades), it reduces filtering to minimize latency, controlled by 'beta'.
    - The raw velocity is also filtered using 'd_cutoff' to prevent jitter amplification.
    """

    def __init__(self, min_cutoff: float = 0.5, beta: float = 0.01, d_cutoff: float = 1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        
        self._prev_point: Optional[Tuple[float, float]] = None
        self._dx_prev: Tuple[float, float] = (0.0, 0.0)
        self._prev_timestamp_ms: Optional[int] = None

    def _get_alpha(self, cutoff: float, dt: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def filter(self, current_point: Tuple[float, float], timestamp_ms: int) -> Tuple[float, float]:
        # First frame init
        if self._prev_point is None or self._prev_timestamp_ms is None:
            self._prev_point = current_point
            self._prev_timestamp_ms = timestamp_ms
            return current_point

        dt = (timestamp_ms - self._prev_timestamp_ms) / 1000.0
        
        if dt <= 0.0:
            return self._prev_point

        # Raw velocity (px/s) calc
        dx = (current_point[0] - self._prev_point[0]) / dt
        dy = (current_point[1] - self._prev_point[1]) / dt

        # Velocity filtering with d_cutoff
        alpha_d = self._get_alpha(self.d_cutoff, dt)
        dx_hat = dx * alpha_d + self._dx_prev[0] * (1.0 - alpha_d)
        dy_hat = dy * alpha_d + self._dx_prev[1] * (1.0 - alpha_d)
        self._dx_prev = (dx_hat, dy_hat)

        # Dynamic cutoff frequency calc based on velocity magnitude
        velocity_magnitude = math.sqrt(dx_hat**2 + dy_hat**2)
        cutoff_x = self.min_cutoff + self.beta * velocity_magnitude
        cutoff_y = self.min_cutoff + self.beta * velocity_magnitude

        alpha_x = self._get_alpha(cutoff_x, dt)
        alpha_y = self._get_alpha(cutoff_y, dt)

        new_x = current_point[0] * alpha_x + self._prev_point[0] * (1.0 - alpha_x)
        new_y = current_point[1] * alpha_y + self._prev_point[1] * (1.0 - alpha_y)

        self._prev_point = (new_x, new_y)
        self._prev_timestamp_ms = timestamp_ms

        return self._prev_point