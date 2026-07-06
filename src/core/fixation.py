"""I-DT fixation detection (dispersion threshold), streaming variant.

Salvucci & Goldberg (2000) classify gaze samples into fixations and saccades
by dispersion: a window of samples spanning at least ``min_duration`` whose
dispersion D = (max_x - min_x) + (max_y - min_y) stays below a threshold is a
fixation. This implementation is an online adaptation: points are consumed
one at a time (as the gaze pipeline produces them) and the detector reports
the *current* fixation state after every sample, so the UI can render a
fixation ring and drive dwell activation frame by frame.

Design notes:
- Coordinates are normalized screen units (0..1), the native unit of the
  gaze pipeline; the default threshold 0.04 corresponds to roughly twice the
  measured system accuracy (~1.7 deg) on a typical laptop panel.
- A brief loss of gaze (blink -> ``update(None, ...)``) does NOT clear the
  window: fixations physiologically continue across blinks. Only a gap
  longer than ``gap_reset`` discards the window.
- The window is capped at ``max_window`` seconds and slides, so a slow drift
  of a long fixation keeps being tracked instead of breaking it.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional, Tuple


@dataclass(frozen=True)
class FixationState:
    fixating: bool
    centroid: Optional[Tuple[float, float]] # normalized screen coords
    duration_s: float
    dispersion: float


class IDTFixationDetector:
    def __init__(
        self,
        dispersion_threshold: float = 0.04,
        min_duration: float = 0.12,
        max_window: float = 1.5,
        gap_reset: float = 0.5,
    ):
        self.dispersion_threshold = dispersion_threshold
        self.min_duration = min_duration
        self.max_window = max_window
        self.gap_reset = gap_reset
        self._window: Deque[Tuple[float, float, float]] = deque() # (t, x, y)
        self._fixation_start: Optional[float] = None

    def reset(self) -> None:
        self._window.clear()
        self._fixation_start = None

    @staticmethod
    def _dispersion(window) -> float:
        xs = [p[1] for p in window]
        ys = [p[2] for p in window]
        return (max(xs) - min(xs)) + (max(ys) - min(ys))

    def update(self, x: Optional[float], y: Optional[float], t: float) -> FixationState:
        if x is None or y is None:
            # Blink / lost gaze: hold the window (fixation may resume), but
            # report non-fixating; dwell logic applies its own grace period.
            if self._window and t - self._window[-1][0] > self.gap_reset:
                self.reset()
            return FixationState(False, None, 0.0, 0.0)

        if self._window and t - self._window[-1][0] > self.gap_reset:
            self.reset()

        self._window.append((t, x, y))
        while self._window and t - self._window[0][0] > self.max_window:
            self._window.popleft()

        # Shrink from the left until the window is dispersion-coherent
        while len(self._window) >= 2 and self._dispersion(self._window) > self.dispersion_threshold:
            self._window.popleft()

        span = self._window[-1][0] - self._window[0][0]
        if span >= self.min_duration:
            if self._fixation_start is None:
                self._fixation_start = self._window[0][0]
            cx = sum(p[1] for p in self._window) / len(self._window)
            cy = sum(p[2] for p in self._window) / len(self._window)
            return FixationState(True, (cx, cy), t - self._fixation_start, self._dispersion(self._window))

        self._fixation_start = None
        return FixationState(False, None, 0.0, self._dispersion(self._window) if self._window else 0.0)