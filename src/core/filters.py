"""Coordinate smoothing filters for gaze stabilization."""

from __future__ import annotations

import math
from typing import Protocol


class PointFilter(Protocol):
    """Protocol defining the interface for coordinate smoothing filters."""

    def filter(self, current_point: tuple[float, float], timestamp_ms: int) -> tuple[float, float]: ...
    def reset(self) -> None: ...


class NoFilter:
    """Pass-through strategy used as an experimental baseline."""

    def filter(self, current_point: tuple[float, float], timestamp_ms: int) -> tuple[float, float]:
        return current_point

    def reset(self) -> None:
        pass


class EMAFilter:
    """Applies a simple Exponential Moving Average low-pass filter."""

    def __init__(self, alpha: float = 0.5) -> None:
        self.alpha = alpha
        self._prev: tuple[float, float] | None = None

    def reset(self) -> None:
        self._prev = None

    def filter(self, current_point: tuple[float, float], timestamp_ms: int) -> tuple[float, float]:
        if self._prev is None:
            self._prev = current_point
            return current_point

        new_x = current_point[0] * self.alpha + self._prev[0] * (1.0 - self.alpha)
        new_y = current_point[1] * self.alpha + self._prev[1] * (1.0 - self.alpha)

        self._prev = (new_x, new_y)
        return self._prev


class OneEuroFilter:
    """
    Applies the 1 Euro Filter algorithm (Casiez et al., 2012).
    
    Dynamically adjusts the cutoff frequency based on movement speed:
    - Heavy filtering at low speeds to eliminate jitter (controlled by min_cutoff).
    - Light filtering at high speeds to eliminate latency during saccades (controlled by beta).
    Designed to operate on normalized screen coordinates (0.0 - 1.0).
    """

    def __init__(self, min_cutoff: float = 0.8, beta: float = 0.5, d_cutoff: float = 1.0) -> None:
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff

        self._prev_point: tuple[float, float] | None = None
        self._dx_prev: tuple[float, float] = (0.0, 0.0)
        self._prev_timestamp_ms: int | None = None

    def reset(self) -> None:
        self._prev_point = None
        self._dx_prev = (0.0, 0.0)
        self._prev_timestamp_ms = None

    @staticmethod
    def _get_alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def filter(self, current_point: tuple[float, float], timestamp_ms: int) -> tuple[float, float]:
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