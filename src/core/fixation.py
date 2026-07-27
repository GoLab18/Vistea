"""
I-DT (Dispersion-Threshold) fixation detection.

Online streaming adaptation of Salvucci & Goldberg (2000). Classifies incoming 
normalized gaze coordinates into fixations or saccades to drive dwell activation.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass

# Timestamped coordinate (time_s, x, y)
type TemporalPoint = tuple[float, float, float]


@dataclass(frozen=True)
class FixationState:
    fixating: bool
    centroid: tuple[float, float] | None # Normalized screen coords (0.0 - 1.0)
    duration_s: float
    dispersion: float


class IDTFixationDetector:
    """
    Detects eye fixations by measuring spatial dispersion over a sliding time window.
    Blinks do not interrupt an ongoing fixation unless they exceed gap_reset.
    """

    def __init__(
        self,
        dispersion_threshold: float = 0.04,
        min_duration: float = 0.12,
        max_window: float = 1.5,
        gap_reset: float = 0.5,
    ) -> None:
        """
        Args:
            dispersion_threshold: Max dispersion (Δx + Δy) for a valid fixation. 
                0.04 corresponds to ~2x the measured system accuracy (~1.7 deg).
            min_duration: Minimum time (seconds) to physiologically recognize a fixation.
                The Salvucci & Goldberg (2000) paper notes fixations are usually at least 100ms.
                120ms is a safe lower bound for interactivity.
            max_window: [Custom addition] Caps the window size (1.5s) to allow tracking 
                of slow, natural eye/head drifts without breaking the fixation.
            gap_reset: [Custom addition] Max allowed signal loss (e.g., blinks). 
                Up to 500ms, the window freezes instead of resetting, as fixations 
                physiologically continue across blinks.
        """

        self.dispersion_threshold = dispersion_threshold
        self.min_duration = min_duration
        self.max_window = max_window
        self.gap_reset = gap_reset
        self._window: deque[TemporalPoint] = deque()
        self._fixation_start: float | None = None

    def reset(self) -> None:
        self._window.clear()
        self._fixation_start = None

    @staticmethod
    def _dispersion(window: Iterable[TemporalPoint]) -> float:
        """Calculates the dispersion (D = Δx + Δy) of the given data window."""

        _, xs, ys = zip(*window)
        return (max(xs) - min(xs)) + (max(ys) - min(ys))

    def update(self, x: float | None, y: float | None, t: float) -> FixationState:
        if x is None or y is None:
            # Blink / lost gaze: hold the window but report non-fixating.
            # Dwell logic applies its own grace period downstream.
            if self._window and t - self._window[-1][0] > self.gap_reset:
                self.reset()
            return FixationState(False, None, 0.0, 0.0)

        # Safety reset if there's a massive temporal jump between valid frames
        if self._window and t - self._window[-1][0] > self.gap_reset:
            self.reset()

        self._window.append((t, x, y))

        # Slide the window tail to accommodate slow drifts (custom addition)
        while self._window and t - self._window[0][0] > self.max_window:
            self._window.popleft()

        # Erase the oldest points until the window fits within the dispersion threshold
        while len(self._window) >= 2 and self._dispersion(self._window) > self.dispersion_threshold:
            self._window.popleft()

        time_span = self._window[-1][0] - self._window[0][0]
        if time_span >= self.min_duration:
            if self._fixation_start is None:
                self._fixation_start = self._window[0][0]
            cx = sum(p[1] for p in self._window) / len(self._window)
            cy = sum(p[2] for p in self._window) / len(self._window)
            return FixationState(True, (cx, cy), t - self._fixation_start, self._dispersion(self._window))

        self._fixation_start = None
        return FixationState(False, None, 0.0, self._dispersion(self._window) if self._window else 0.0)