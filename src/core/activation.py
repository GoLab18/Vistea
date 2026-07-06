"""Gaze activation strategies (the "click" of a gaze UI).

The Midas touch problem (Jacob 1990): the eye is a perceptual organ, not an
actuator - everything the user LOOKS at must not become everything the user
TRIGGERS. Both strategies below convert looking into an explicit, deliberate
act and are interchangeable behind ``ActivationStrategy`` so the thesis can
compare them on the same tasks:

- ``DwellActivation``: holding a fixation on one target for ``dwell_time``
  fires an activation; a progress ring gives continuous feedback; a cooldown
  paces refires. Keeping the gaze on the SAME target repeats the activation
  (typing double letters without looking away), but each repeat requires a
  longer hold (``dwell_time * repeat_factor``) so an unintended series is
  guarded against - the user sees the ring refill and can look away in time.
- ``BlinkActivation``: a deliberately LONG blink (default 0.35-1.0 s) on a
  target fires; natural blinks (~0.1-0.3 s) fall below the window and are
  ignored. Eye closure is read from MediaPipe face blendshapes
  (categories ``eyeBlinkLeft``/``eyeBlinkRight``), which the vision layer
  already emits.

All timing is wall-clock (monotonic seconds supplied by the caller), never
frame counts - the same lesson as the pursuit-lag compensation in the
calibration layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Protocol, runtime_checkable


class ActivationKind(str, Enum):
    """How an activation was produced. A ``str`` subclass so the value
    serializes naturally into JSONL telemetry (``"dwell"``/``"blink"``)."""

    DWELL = "dwell"
    BLINK = "blink"


@dataclass(frozen=True)
class ActivationEvent:
    target_id: str
    kind: ActivationKind
    t: float


@runtime_checkable
class ActivationStrategy(Protocol):
    name: str

    def update(
        self,
        target_id: Optional[str],
        fixating: bool,
        blink_score: Optional[float],
        t: float,
    ) -> Optional[ActivationEvent]:
        """Feed one frame of interaction state; returns an event when fired."""
        ...

    @property
    def progress(self) -> float:
        """0..1 feedback value for the UI (dwell ring / blink hold)."""
        ...

    def reset(self) -> None: ...


def extract_blink_score(blendshapes) -> Optional[float]:
    """Mean of eyeBlinkLeft/eyeBlinkRight scores from MediaPipe blendshapes
    (duck-typed: any iterable of objects with .category_name and .score)."""

    if not blendshapes:
        return None
    scores = [
        c.score for c in blendshapes
        if getattr(c, "category_name", "") in ("eyeBlinkLeft", "eyeBlinkRight")
    ]
    return sum(scores) / len(scores) if scores else None


class DwellActivation:
    """Dwell-to-click with in-place repeats.

    The first activation after the gaze ARRIVES on a target uses the plain
    ``dwell_time``. If the gaze then STAYS on that same target, accumulation
    restarts from zero after the cooldown and fires again once it reaches
    ``dwell_time * repeat_factor`` (and likewise for every further repeat).
    The factor above 1 makes repeats deliberately harder than the first hit,
    so resting the eyes on a key does not spray a series - the ring visibly
    refills and the user can look away. Leaving the target resets the repeat
    state, so the next arrival is a plain first activation again.
    """

    def __init__(
        self,
        dwell_time: float = 1.0, # configurable 0.8-1.5 s
        cooldown: float = 0.5,
        grace: float = 0.25, # tolerate blinks / micro-breaks mid-dwell
        repeat_factor: float = 1.25, # repeats need dwell_time * this
        name: Optional[str] = None,
    ):
        self.dwell_time = dwell_time
        self.cooldown = cooldown
        self.grace = grace
        self.repeat_factor = repeat_factor
        self.name = name or f"dwell_{dwell_time:.1f}s"
        self.reset()

    def reset(self) -> None:
        self._target: Optional[str] = None
        self._accum = 0.0
        self._last_t: Optional[float] = None
        self._last_valid_t: Optional[float] = None
        self._repeating_on: Optional[str] = None # target already fired at least once
        self._cooldown_until = 0.0
        self._progress = 0.0

    @property
    def progress(self) -> float:
        return self._progress

    def _threshold(self) -> float:
        """Hold time required for the pending activation on the current target:
        plain dwell for the first hit, the longer repeat dwell afterwards."""

        if self._target is not None and self._target == self._repeating_on:
            return self.dwell_time * self.repeat_factor
        return self.dwell_time

    def update(self, target_id, fixating, blink_score, t) -> Optional[ActivationEvent]:
        dt = 0.0 if self._last_t is None else max(0.0, t - self._last_t)
        self._last_t = t

        valid = target_id is not None and fixating
        if valid:
            if target_id != self._target:
                # Fresh arrival on a target resets the repeat state.
                self._target, self._accum = target_id, 0.0
                self._repeating_on = None
            elif t >= self._cooldown_until:
                self._accum += dt
            self._last_valid_t = t
        else:
            # Grace: a short gap (blink) freezes progress instead of resetting.
            if self._last_valid_t is None or t - self._last_valid_t > self.grace:
                self._target, self._accum = None, 0.0
                self._repeating_on = None

        threshold = self._threshold()
        self._progress = min(1.0, self._accum / threshold) if self._target else 0.0

        if self._target and self._accum >= threshold:
            event = ActivationEvent(self._target, ActivationKind.DWELL, t)
            self._repeating_on = self._target # next hold uses the repeat threshold
            self._cooldown_until = t + self.cooldown
            self._accum, self._progress = 0.0, 0.0
            return event
        return None


class BlinkActivation:
    """Deliberate long blink on a target. The target is locked at blink
    onset (during closure the gaze signal is invalid by definition)."""

    def __init__(
        self,
        min_blink: float = 0.35,
        max_blink: float = 1.2,
        close_threshold: float = 0.5,
        name: str = "blink_long",
    ):
        self.min_blink = min_blink
        self.max_blink = max_blink
        self.close_threshold = close_threshold
        self.name = name
        self.reset()

    def reset(self) -> None:
        self._closed_since: Optional[float] = None
        self._locked_target: Optional[str] = None
        self._last_target: Optional[str] = None
        self._progress = 0.0

    @property
    def progress(self) -> float:
        return self._progress

    def update(self, target_id, fixating, blink_score, t) -> Optional[ActivationEvent]:
        if target_id is not None:
            self._last_target = target_id

        closed = blink_score is not None and blink_score >= self.close_threshold
        if closed:
            if self._closed_since is None:
                self._closed_since = t
                self._locked_target = self._last_target
            held = t - self._closed_since
            self._progress = min(1.0, held / self.min_blink)
            if held > self.max_blink: # too long: treated as rest, not intent
                self._locked_target = None
                self._progress = 0.0
            return None

        # Eyes (re)opened: evaluate the completed blink
        event: Optional[ActivationEvent] = None
        if self._closed_since is not None:
            held = t - self._closed_since
            if self._locked_target and self.min_blink <= held <= self.max_blink:
                event = ActivationEvent(self._locked_target, ActivationKind.BLINK, t)
        self._closed_since = None
        self._locked_target = None
        self._progress = 0.0
        return event


ACTIVATION_REGISTRY = {
    "Dwell 1.0 s (default)": lambda: DwellActivation(1.0),
    "Dwell 0.8 s": lambda: DwellActivation(0.8),
    "Dwell 1.5 s": lambda: DwellActivation(1.5),
    "Long blink": lambda: BlinkActivation(),
}