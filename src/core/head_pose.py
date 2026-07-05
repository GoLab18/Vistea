"""
Virtual chinrest - the Labvanced-style technique of monitoring
head pose against the pose held during calibration.

Physical eye trackers immobilise the head with a chinrest - a webcam system
cannot, but it can detect when the head has drifted far enough from the
calibration pose that the trained mapping is no longer valid, and warn the
user (or pause the cursor) instead of silently producing wrong gaze points.

The monitor is deliberately UI-free: it consumes HeadPose values,
compares them with a reference captured during calibration,
and answers two questions - "is the pose still OK?" and "should the UI emit
a warning right now?" (rate-limited so a user sitting slightly off-pose is
not spammed every frame).
"""

from __future__ import annotations

import math
import statistics
import time
from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

# Default thresholds: yaw/pitch change the projected eye geometry (and thus
# invalidate the 4D mapping) faster than roll, which the PC-EC features
# largely cancel in-plane - hence the looser roll threshold.
DEFAULT_YAW_PITCH_THRESHOLD_DEG: float = 8.0
DEFAULT_ROLL_THRESHOLD_DEG: float = 10.0
DEFAULT_WARN_INTERVAL_S: float = 2.0


@dataclass(frozen=True)
class DriftState:
    ok: bool
    dyaw_deg: float
    dpitch_deg: float
    droll_deg: float

    def worst_axis(self) -> Tuple[str, float]:
        pairs = [("yaw", self.dyaw_deg), ("pitch", self.dpitch_deg), ("roll", self.droll_deg)]
        return max(pairs, key=lambda p: abs(p[1]))


class HeadPoseMonitor:
    """Compares the current head pose with the calibration-time reference."""

    def __init__(
        self,
        yaw_pitch_threshold_deg: float = DEFAULT_YAW_PITCH_THRESHOLD_DEG,
        roll_threshold_deg: float = DEFAULT_ROLL_THRESHOLD_DEG,
        warn_interval_s: float = DEFAULT_WARN_INTERVAL_S,
    ):
        self.yaw_pitch_threshold_deg = yaw_pitch_threshold_deg
        self.roll_threshold_deg = roll_threshold_deg
        self.warn_interval_s = warn_interval_s
        self._ref: Optional[Tuple[float, float, float]] = None # radians
        self._last_warn_t: float = -math.inf

    def set_reference(self, pose_samples: Iterable[Tuple[float, float, float]]) -> bool:
        """
        Fixes the reference as the per-axis MEDIAN of poses observed while
        the user was calibrating (median = robust to the occasional glance
        away or detector glitch, unlike the mean). Returns False when there
        were no samples - the monitor then stays inactive.
        """

        samples = list(pose_samples)
        if not samples:
            self._ref = None
            return False
        self._ref = (
            statistics.median(s[0] for s in samples),
            statistics.median(s[1] for s in samples),
            statistics.median(s[2] for s in samples),
        )
        return True

    @property
    def has_reference(self) -> bool:
        return self._ref is not None

    def update(self, yaw: float, pitch: float, roll: float) -> DriftState:
        """Deviation of the current pose (radians in) from the reference."""

        if self._ref is None:
            return DriftState(True, 0.0, 0.0, 0.0)
        dyaw = math.degrees(yaw - self._ref[0])
        dpitch = math.degrees(pitch - self._ref[1])
        droll = math.degrees(roll - self._ref[2])
        ok = (
            abs(dyaw) <= self.yaw_pitch_threshold_deg
            and abs(dpitch) <= self.yaw_pitch_threshold_deg
            and abs(droll) <= self.roll_threshold_deg
        )
        return DriftState(ok, dyaw, dpitch, droll)

    def should_warn(self, now: Optional[float] = None) -> bool:
        """
        Rate-limited warning gate: returns True at most once per warn_interval_s.
        Should be called only after update() reported not-ok.
        """

        t = time.monotonic() if now is None else now
        if t - self._last_warn_t >= self.warn_interval_s:
            self._last_warn_t = t
            return True
        return False

    def reset(self) -> None:
        self._ref = None
        self._last_warn_t = -math.inf