"""
Edge-zone scroll gestures for navigation mode.

Converts gaze fixations at the top/bottom of the screen into OS scroll wheel events,
incorporating hysteresis and blink-resilience to prevent unintended activations.
"""

from __future__ import annotations

import math
from typing import Final, Literal

type ScrollDirection = Literal[-1, 0, 1]

SCROLL_UP: Final[ScrollDirection] = 1
SCROLL_DOWN: Final[ScrollDirection] = -1
SCROLL_NONE: Final[ScrollDirection] = 0

# Band width limits, as a fraction of screen height
MIN_ZONE: Final[float] = 0.08
MAX_ZONE: Final[float] = 0.15


def zone_from_error(
    mean_error_px: float | None, screen_h_px: int, factor: float = 2.0
) -> float:
    """
    Calculates the scroll band height based on the user's measured accuracy.
    Bounded by hard screen-fraction limits to keep content reachable.
    """

    if mean_error_px is None or screen_h_px <= 0 or math.isnan(mean_error_px):
        return MIN_ZONE
    return min(MAX_ZONE, max(MIN_ZONE, factor * mean_error_px / screen_h_px))


class EdgeScrollDetector:
    """
    Detects intentional gaze dwells in the screen's margin zones to emit scroll ticks.
    """

    def __init__(
        self,
        zone_height_frac: float = MIN_ZONE,
        entry_delay_s: float = 0.4,
        tick_interval_s: float = 0.12,
        hysteresis_frac: float = 0.05,
        signal_loss_tolerance_s: float = 0.3,
        zone_exit_tolerance_s: float = 0.25,
        accel_step_s: float = 0.8,
        max_scroll_multiplier: int = 4,
        enabled: bool = True,
    ) -> None:
        self.zone_height_frac = zone_height_frac
        self.entry_delay_s = entry_delay_s
        self.tick_interval_s = tick_interval_s
        self.hysteresis_frac = hysteresis_frac
        self.signal_loss_tolerance_s = signal_loss_tolerance_s
        self.zone_exit_tolerance_s = zone_exit_tolerance_s
        self.accel_step_s = accel_step_s
        self.max_scroll_multiplier = max_scroll_multiplier
        self.enabled = enabled

        self._zone_sign: ScrollDirection = SCROLL_NONE
        self._entered_at: float | None = None
        self._active: bool = False
        self._active_since: float | None = None
        self._last_tick: float | None = None
        self._paused_since: float | None = None
        self._pause_limit: float = 0.0

    def reset(self) -> None:
        """Fully resets the gesture state machine."""

        self._zone_sign = SCROLL_NONE
        self._entered_at = None
        self._active = False
        self._active_since = None
        self._last_tick = None
        self._paused_since = None
        self._pause_limit = 0.0

    @property
    def active(self) -> bool:
        """Returns True if the user is currently scrolling."""

        return self._active

    @property
    def current_zone(self) -> ScrollDirection:
        return self._zone_sign

    def _classify_to_zone(self, y: float, expanded: bool) -> ScrollDirection:
        """Classifies the Y coordinate into a zone."""

        margin = self.zone_height_frac + (self.hysteresis_frac if expanded else 0.0)
        if y <= margin:
            return SCROLL_UP
        if y >= 1.0 - margin:
            return SCROLL_DOWN
        return SCROLL_NONE

    def set_enabled(self, enabled: bool) -> bool:
        """Turn the gesture on or off, clearing state on the way out."""

        self.enabled = enabled
        if not enabled:
            self.reset()
        return self.enabled

    def _scroll_multiplier(self, t: float) -> int:
        """
        Wheel notches per tick, growing with how long the gesture has been held.
        """

        if self._active_since is None or self.accel_step_s <= 0.0:
            return 1
        held_for = max(0.0, t - self._active_since)
        return min(self.max_scroll_multiplier, 1 + int(held_for / self.accel_step_s))

    def _pause(self, t: float, limit: float) -> bool:
        """Freezes the gesture. Returns True while still within tolerance."""

        if self._paused_since is None:
            self._paused_since = t
            self._pause_limit = limit
        return t - self._paused_since <= self._pause_limit

    def _resume(self, t: float) -> None:
        """Shift the clocks past a frozen interval, so a pause neither counts
        toward the entry delay nor produces a burst of backdated ticks."""

        if self._paused_since is None:
            return
        gap = t - self._paused_since
        if self._entered_at is not None:
            self._entered_at += gap
        if self._last_tick is not None:
            self._last_tick += gap
        if self._active_since is not None:
            self._active_since += gap
        self._paused_since = None

    def update(self, y: float | None, t: float, suppressed: bool = False) -> int:
        """
        Updates the gesture state machine and returns the number of scroll ticks to emit.

        `suppressed` marks a frame where the gaze belongs to something else
        rather than the gesture.
        """

        if not self.enabled:
            return 0

        if suppressed:
            if self._zone_sign != SCROLL_NONE or self._active:
                self.reset()
            return 0

        if y is None:
            if self._pause(t, self.signal_loss_tolerance_s):
                return 0
            self.reset()
            return 0

        curr_zone = self._classify_to_zone(y, expanded=self._zone_sign != SCROLL_NONE)

        if curr_zone == SCROLL_NONE and self._zone_sign != SCROLL_NONE:
            if self._pause(t, self.zone_exit_tolerance_s):
                return 0
            self.reset()
            return 0

        self._resume(t)

        if curr_zone != self._zone_sign:
            self._zone_sign = curr_zone
            self._entered_at = t if curr_zone != SCROLL_NONE else None
            self._active = False
            self._active_since = None
            self._last_tick = None
            return 0

        if curr_zone == SCROLL_NONE or self._entered_at is None:
            return 0

        if not self._active:
            if t - self._entered_at < self.entry_delay_s:
                return 0
            self._active = True
            self._active_since = t
            self._last_tick = t
            return curr_zone

        ticks = 0
        while self._last_tick is not None and t - self._last_tick >= self.tick_interval_s:
            self._last_tick += self.tick_interval_s
            ticks += curr_zone * self._scroll_multiplier(self._last_tick)

        return ticks
