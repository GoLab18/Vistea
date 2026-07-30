"""
Gaze activation strategies (the "click" of a gaze UI).

Resolves the Midas touch problem (Jacob 1990) by converting perceptual looking 
into deliberate acts (Dwell or Blink).

All timing relies on monotonic wall-clock seconds, never frame counts.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from enum import Enum
from typing import Final, Protocol, runtime_checkable


class ActivationKind(str, Enum):
    """How an activation was produced."""

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
        target_id: str | None,
        fixating: bool,
        blink_score: float | None,
        t: float,
    ) -> ActivationEvent | None:
        """Feed one frame of interaction state. Returns an event when fired."""
        ...

    @property
    def progress(self) -> float:
        """Feedback value for the UI (0.0 - 1.0)."""
        ...

    def reset(self) -> None: ...


class BlendshapeCategory(Protocol):
    category_name: str
    score: float


def extract_blink_score(blendshapes: Iterable[BlendshapeCategory] | None) -> float | None:
    """
    Mean of eyeBlinkLeft/eyeBlinkRight scores from MediaPipe blendshapes.
    """
    if not blendshapes:
        return None
    scores = [
        c.score for c in blendshapes
        if getattr(c, "category_name", "") in ("eyeBlinkLeft", "eyeBlinkRight")
    ]
    return sum(scores) / len(scores) if scores else None


class DwellActivation:
    """
    Dwell-to-click with in-place repeats and blink resilience.
    
    The first activation requires plain `dwell_time`. Keeping gaze on the same target 
    requires a longer hold (`dwell_time * repeat_factor`) to prevent unintended series.
    """

    def __init__(
        self,
        dwell_time: float = 1.0,     # Configurable base threshold (0.8-1.5 s)
        cooldown: float = 0.5,       # Mandatory wait before accumulating a repeat
        grace: float = 0.25,         # Blinks / micro-breaks tolerated mid-dwell
        repeat_factor: float = 1.25, # Repeats multiply the dwell_time requirement
        name: str | None = None,
    ) -> None:
        self.dwell_time = dwell_time
        self.cooldown = cooldown
        self.grace = grace
        self.repeat_factor = repeat_factor
        self.name = name or f"dwell_{dwell_time:.1f}s"
        
        self._target: str | None = None
        self._accum: float = 0.0
        self._last_t: float | None = None
        self._last_valid_t: float | None = None
        self._repeating_on: str | None = None # Target already fired at least once
        self._cooldown_until: float = 0.0
        self._progress: float = 0.0

    def reset(self) -> None:
        self._target = None
        self._accum = 0.0
        self._last_t = None
        self._last_valid_t = None
        self._repeating_on = None
        self._cooldown_until = 0.0
        self._progress = 0.0

    @property
    def progress(self) -> float:
        return self._progress

    def _threshold(self) -> float:
        """Hold time required: plain dwell for first hit, extended for repeats."""

        if self._target is not None and self._target == self._repeating_on:
            return self.dwell_time * self.repeat_factor
        return self.dwell_time

    def update(
        self,
        target_id: str | None,
        fixating: bool,
        blink_score: float | None,
        t: float,
    ) -> ActivationEvent | None:
        
        dt = 0.0 if self._last_t is None else max(0.0, t - self._last_t)
        self._last_t = t

        valid = target_id is not None and fixating
        if valid:
            if target_id != self._target:
                # Fresh arrival resets repeat state
                self._target, self._accum = target_id, 0.0
                self._repeating_on = None
            elif t >= self._cooldown_until:
                self._accum += dt
            self._last_valid_t = t
        else:
            # Grace: short gap freezes progress instead of resetting
            if self._last_valid_t is None or t - self._last_valid_t > self.grace:
                self._target, self._accum = None, 0.0
                self._repeating_on = None

        threshold = self._threshold()
        self._progress = min(1.0, self._accum / threshold) if self._target else 0.0

        if self._target and self._accum >= threshold:
            event = ActivationEvent(self._target, ActivationKind.DWELL, t)
            self._repeating_on = self._target # Next hold uses the repeat threshold
            self._cooldown_until = t + self.cooldown
            self._accum, self._progress = 0.0, 0.0
            return event

        return None


class BlinkActivation:
    """
    Deliberate long blink on a target.
    Natural blinks fall below `min_blink`, resting eyes exceed `max_blink`.

    The activated target is the one the user was looking at *before* the lid
    started moving, not the one under the gaze point at the moment the blink
    score crosses the closing threshold.
    """

    def __init__(
        self,
        min_blink: float = 0.35,
        max_blink: float = 1.2,
        close_threshold: float = 0.5,
        pre_blink_lag: float = 0.25,
        name: str = "blink_long",
    ) -> None:
        self.min_blink = min_blink
        self.max_blink = max_blink
        self.close_threshold = close_threshold
        self.pre_blink_lag = pre_blink_lag
        self.name = name

        self._closed_since: float | None = None
        self._locked_target: str | None = None
        self._history: deque[tuple[float, str]] = deque()
        self._progress: float = 0.0

    def reset(self) -> None:
        self._closed_since = None
        self._locked_target = None
        self._history.clear()
        self._progress = 0.0

    @property
    def progress(self) -> float:
        return self._progress

    def _remember(self, target_id: str, t: float) -> None:
        """Append a seen target, dropping entries older than twice the lag."""

        self._history.append((t, target_id))
        drop_lim = t - 2.0 * self.pre_blink_lag
        while self._history and self._history[0][0] < drop_lim:
            self._history.popleft()

    def _target_before_blink(self, t: float) -> str | None:
        """The most recent target seen at least `pre_blink_lag` before `t`,
        falling back to the oldest one still remembered."""

        cutoff = t - self.pre_blink_lag
        for stamp, tid in reversed(self._history):
            if stamp <= cutoff:
                return tid
        return self._history[0][1] if self._history else None

    def update(
        self,
        target_id: str | None,
        fixating: bool,
        blink_score: float | None,
        t: float,
    ) -> ActivationEvent | None:
        if target_id is not None:
            self._remember(target_id, t)

        closed = blink_score is not None and blink_score >= self.close_threshold
        if closed:
            if self._closed_since is None:
                self._closed_since = t
                self._locked_target = self._target_before_blink(t)
            held = t - self._closed_since
            self._progress = min(1.0, held / self.min_blink)

            if held > self.max_blink: # Treated as rest, not intent
                self._locked_target = None
                self._progress = 0.0
            return None

        # Eyes (re)opened: evaluate the completed blink duration
        event: ActivationEvent | None = None
        if self._closed_since is not None:
            held = t - self._closed_since
            if self._locked_target and self.min_blink <= held <= self.max_blink:
                event = ActivationEvent(self._locked_target, ActivationKind.BLINK, t)

        self._closed_since = None
        self._locked_target = None
        self._progress = 0.0
        return event


DEFAULT_ACTIVATION_KEY: Final[str] = "Dwell 1.0 s (default)"

ACTIVATION_REGISTRY: dict[str, Callable[[], ActivationStrategy]] = {
    DEFAULT_ACTIVATION_KEY: lambda: DwellActivation(1.0),
    "Dwell 0.8 s": lambda: DwellActivation(0.8),
    "Dwell 1.5 s": lambda: DwellActivation(1.5),
    "Long blink": lambda: BlinkActivation(),
}