"""Edge-zone scroll gestures.

Fixating in the top/bottom band of the screen emulates mouse-wheel scrolling.
Three guards address the Midas touch problem for a *zone* target (which,
unlike a tile, is huge and trivially entered by accident):

- **Entry delay**: the gaze must stay inside the zone for ``entry_delay``
  seconds before the first scroll tick - glancing at the top of a page does
  not scroll it.
- **Hysteresis**: once scrolling is active the zone boundary moves OUTWARD by
  ``hysteresis`` for the purpose of *leaving* it, so gaze jitter exactly on
  the boundary cannot rapidly toggle the gesture on and off.
- **Grace**: a momentary loss of signal (``y`` is ``None`` for less than
  ``grace`` seconds) freezes the zone state instead of resetting it - the
  gesture picks up where it left off once the signal returns, with no burst
  of catch-up ticks and no free progress toward arming.

The detector is Qt-free and clock-driven: ``update(y, t)`` returns the number
of wheel ticks to emit this frame (signed: +1 scroll up per tick in the top
zone, -1 down in the bottom zone), which the UI layer forwards to the system
actuator.
"""

from __future__ import annotations

from typing import Optional


class EdgeScrollDetector:
    def __init__(
        self,
        zone: float = 0.12,          # band height, fraction of screen
        entry_delay: float = 0.4,    # seconds inside the band before ticking
        tick_interval: float = 0.12, # seconds between wheel ticks while active
        hysteresis: float = 0.05,    # extra band depth required to LEAVE
        grace: float = 0.3,          # seconds of lost signal tolerated before reset
    ):
        self.zone = zone
        self.entry_delay = entry_delay
        self.tick_interval = tick_interval
        self.hysteresis = hysteresis
        self.grace = grace
        self.reset()

    def reset(self) -> None:
        self._zone_sign = 0 # +1 top, -1 bottom, 0 none
        self._entered_at: Optional[float] = None
        self._active = False
        self._last_tick: Optional[float] = None
        self._lost_since: Optional[float] = None

    @property
    def active(self) -> bool:
        return self._active

    @property
    def current_zone(self) -> int:
        return self._zone_sign

    def _classify(self, y: float, expanded: bool) -> int:
        margin = self.zone + (self.hysteresis if expanded else 0.0)
        if y <= margin:
            return +1
        if y >= 1.0 - margin:
            return -1
        return 0

    def update(self, y: Optional[float], t: float) -> int:
        if y is None:
            if self._lost_since is None:
                self._lost_since = t
            if t - self._lost_since <= self.grace:
                return 0 # signal freeze: state preserved, no ticks
            self.reset()
            return 0

        if self._lost_since is not None:
            # Signal returned within grace: shift the clocks forward by the
            # fade duration so it counts toward neither entry_delay nor a
            # backlog of ticks.
            lost_for = t - self._lost_since
            if self._entered_at is not None:
                self._entered_at += lost_for
            if self._last_tick is not None:
                self._last_tick += lost_for
            self._lost_since = None

        sign = self._classify(y, expanded=self._active)
        if sign != self._zone_sign:
            # Zone changed (entered / left / jumped across): re-arm.
            self._zone_sign = sign
            self._entered_at = t if sign != 0 else None
            self._active = False
            self._last_tick = None
            return 0

        if sign == 0 or self._entered_at is None:
            return 0

        if not self._active:
            if t - self._entered_at < self.entry_delay:
                return 0
            self._active = True
            self._last_tick = t
            return sign # first tick fires immediately on arming

        ticks = 0
        while self._last_tick is not None and t - self._last_tick >= self.tick_interval:
            self._last_tick += self.tick_interval
            ticks += sign
        return ticks