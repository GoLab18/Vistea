"""
Edge-zone scroll gestures for navigation mode.

Converts gaze fixations at the top/bottom of the screen into OS scroll wheel events,
incorporating hysteresis and blink-resilience to prevent the Midas touch problem.
"""

from __future__ import annotations

from typing import Final, Literal

type ScrollDirection = Literal[-1, 0, 1]

SCROLL_UP: Final[ScrollDirection] = 1
SCROLL_DOWN: Final[ScrollDirection] = -1
SCROLL_NONE: Final[ScrollDirection] = 0


class EdgeScrollDetector:
    """
    Detects intentional gaze dwells in the screen's margin zones to emit scroll ticks.
    """

    def __init__(
        self,
        zone: float = 0.12,          # Band height as a fraction of the screen (0.0 to 1.0)
        entry_delay: float = 0.4,    # Seconds required inside the band before arming
        tick_interval: float = 0.12, # Seconds between consecutive wheel ticks while active
        hysteresis: float = 0.05,    # Extra band depth added to the exit boundary (prevents boundary jitter)
        grace: float = 0.3,          # Seconds of signal loss tolerated without resetting the gesture
    ) -> None:
        self.zone = zone
        self.entry_delay = entry_delay
        self.tick_interval = tick_interval
        self.hysteresis = hysteresis
        self.grace = grace

        self._zone_sign: ScrollDirection = SCROLL_NONE 
        self._entered_at: float | None = None
        self._active: bool = False
        self._last_tick: float | None = None
        self._lost_since: float | None = None

    def reset(self) -> None:
        """Fully resets the gesture state machine."""

        self._zone_sign = SCROLL_NONE 
        self._entered_at = None
        self._active = False
        self._last_tick = None
        self._lost_since = None

    @property
    def active(self) -> bool:
        """Returns True if the user is currently scrolling."""

        return self._active

    @property
    def current_zone(self) -> ScrollDirection:
        return self._zone_sign

    def _classify(self, y: float, expanded: bool) -> ScrollDirection:
        """
        Classifies the Y coordinate into a zone.
        Applies hysteresis (expanded margin) if the gesture is already active.
        """

        margin = self.zone + (self.hysteresis if expanded else 0.0)
        if y <= margin:
            return SCROLL_UP
        if y >= 1.0 - margin:
            return SCROLL_DOWN
        return SCROLL_NONE

    def update(self, y: float | None, t: float) -> int:
        """
        Updates the gesture state machine and returns the number of scroll ticks to emit.
        """

        if y is None:
            if self._lost_since is None:
                self._lost_since = t
            if t - self._lost_since <= self.grace:
                return 0
            self.reset()
            return 0

        if self._lost_since is not None:
            lost_for = t - self._lost_since
            if self._entered_at is not None:
                self._entered_at += lost_for
            if self._last_tick is not None:
                self._last_tick += lost_for
            self._lost_since = None

        sign = self._classify(y, expanded=self._active)

        if sign != self._zone_sign:
            self._zone_sign = sign
            self._entered_at = t if sign != SCROLL_NONE else None
            self._active = False
            self._last_tick = None
            return 0

        if sign == SCROLL_NONE or self._entered_at is None:
            return 0

        if not self._active:
            if t - self._entered_at < self.entry_delay:
                return 0
            self._active = True
            self._last_tick = t
            return sign

        ticks = 0
        while self._last_tick is not None and t - self._last_tick >= self.tick_interval:
            self._last_tick += self.tick_interval
            ticks += sign

        return ticks