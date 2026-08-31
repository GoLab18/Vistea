"""
Post-calibration accuracy maintenance and UI sizing rules.

Implements:
- Incremental model refitting via self-labeling (dwell activations as ground truth).
- Single-point bias recalibration for rapid drift compensation.
- Dynamic UI tile sizing based on the measured validation error.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Final

import numpy as np
from sklearn.linear_model import Ridge

from core.calibration import CalibrationStrategy, FeatureVector, NormalizedPoint

type PoseTuple = tuple[float, float, float]  # (yaw, pitch, roll) in radians

# Minimum tile floor regardless of measured accuracy
TILE_FLOOR_PX: Final[int] = 80


def min_tile_side_px(mean_error_px: float | None, factor: float = 2.0) -> int:
    """Calculates the advisory minimum tile side to match the user's calibration error."""

    if mean_error_px is None or math.isnan(mean_error_px):
        return TILE_FLOOR_PX
    return max(TILE_FLOOR_PX, int(round(factor * mean_error_px)))


@dataclass(frozen=True)
class DriftResult:
    """Outcome of one accepted self-labelled correction."""

    n_samples: int
    pre_error_norm: float
    post_error_norm: float
    mean_prediction: NormalizedPoint
    spread_norm: float


class DriftCorrector:
    """
    Refines the calibration model using UI dwell activations as ground-truth labels.

    To avoid cementing user compensation (where the user looks away from the
    target to counteract a biased cursor), this acts strictly as a local
    refinement mechanism:
    1. Batches are rejected if the model's prediction is already too far from
       the tile center.
    2. Scattered batches are rejected (wandering gaze).
    3. Interaction samples are capped so they never outnumber the initial
       calibration samples, preventing the model from overfitting to the
       keyboard area.
    """

    # How far from the tile center the model may already be for its label to
    # count, as a fraction of the tile's half-extent.
    TRUST_FRACTION: Final[float] = 0.6

    MAX_PREDICTION_SPREAD: Final[float] = 0.04

    def __init__(
        self,
        strategy: CalibrationStrategy,
        frames_per_activation: int = 4,
        max_corrections: int = 25,
        buffer_size: int = 30,
    ) -> None:
        self.strategy = strategy
        self.frames_per_activation = frames_per_activation
        self.max_corrections = max_corrections
        self._buffer: deque[FeatureVector] = deque(maxlen=buffer_size)
        self.corrections_applied = 0
        self.rejected = 0

    def observe(self, vec: FeatureVector | None) -> None:
        """Buffers feature vectors continuously while the user dwells."""

        if vec is not None:
            self._buffer.append(vec)

    def _predictions(self, batch: list[FeatureVector]) -> list[NormalizedPoint]:
        return [p for p in (self.strategy.predict(vec) for vec in batch) if p is not None]

    @staticmethod
    def _mean_point(points: list[NormalizedPoint]) -> NormalizedPoint:
        n = len(points)
        return NormalizedPoint(sum(p.x for p in points) / n, sum(p.y for p in points) / n)

    @staticmethod
    def _spread(points: list[NormalizedPoint]) -> float:
        return max(
            max(p.x for p in points) - min(p.x for p in points),
            max(p.y for p in points) - min(p.y for p in points),
        )

    def _batch_error_norm(self, batch: list[FeatureVector], target: NormalizedPoint) -> float:
        """Mean distance in normalized units between model predictions and target."""

        errs = [
            math.hypot(p.x - target.x, p.y - target.y) for p in self._predictions(batch)
        ]
        return sum(errs) / len(errs) if errs else float("nan")

    def on_activation(
        self,
        tile_center: NormalizedPoint,
        tile_half_extent: tuple[float, float],
    ) -> DriftResult | None:
        """
        Label the buffered frames with the activated tile center and refit.

        `tile_half_extent` is half the tile's width and height.
        Returns None whenever the batch was refused, which is the
        normal outcome for an ambiguous or unsteady activation.
        """

        if self.corrections_applied >= self.max_corrections:
            return None
        if not self.strategy.is_calibrated():
            return None

        batch = list(self._buffer)[-self.frames_per_activation:]
        if len(batch) < self.frames_per_activation:
            return None  # Too little signal

        predictions = self._predictions(batch)
        if len(predictions) < self.frames_per_activation:
            return None

        spread = self._spread(predictions)
        if spread > self.MAX_PREDICTION_SPREAD:
            self.rejected += 1
            self._buffer.clear()
            return None

        mean_prediction = self._mean_point(predictions)
        trust_radius = self.TRUST_FRACTION * min(tile_half_extent)
        offset = math.hypot(
            mean_prediction.x - tile_center.x, mean_prediction.y - tile_center.y
        )
        if offset > trust_radius:
            self.rejected += 1
            self._buffer.clear()
            return None

        # Train-test split to measure true generalization
        train_frames, test_frame_newest = batch[:-1], batch[-1:]

        pre_error_norm = self._batch_error_norm(test_frame_newest, tile_center)

        self.strategy.add_samples(tile_center, train_frames)
        self.strategy.train()

        post_error_norm = self._batch_error_norm(test_frame_newest, tile_center)

        self.corrections_applied += 1
        self._buffer.clear()

        return DriftResult(
            len(train_frames), pre_error_norm, post_error_norm, mean_prediction, spread
        )


class PoseResidualCorrector:
    """
    Tries to cancel the systematic mapping error caused by slight head pose changes.

    Uses a rolling window of recent activations to fit a tiny, separate Ridge
    regression (Pose 3D -> 2D Shift), trying to prevent temporary posture changes 
    from polluting the global calibration model.
    """

    # Caps poses so that the history is fresh
    HISTORY_WINDOW: Final[int] = 60

    MIN_OBSERVATIONS: Final[int] = 8
    MAX_POSE_SHIFT: Final[float] = 0.12
    RIDGE_ALPHA: Final[float] = 0.01

    def __init__(self) -> None:
        self._poses: deque[PoseTuple] = deque(maxlen=self.HISTORY_WINDOW)
        self._residuals: deque[tuple[float, float]] = deque(maxlen=self.HISTORY_WINDOW)
        self._model: Ridge | None = None

    @property
    def n_observations(self) -> int:
        return len(self._poses)

    @property
    def is_active(self) -> bool:
        return self._model is not None

    def reset(self) -> None:
        self._poses.clear()
        self._residuals.clear()
        self._model = None

    def observe(self, pose: PoseTuple, gaze_residual: tuple[float, float]) -> bool:
        """
        Records the gaze error for a given head pose.
        Returns True once the model is fitted.
        """

        self._poses.append(pose)
        self._residuals.append(gaze_residual)

        if len(self._poses) < self.MIN_OBSERVATIONS:
            return False

        model = Ridge(alpha=self.RIDGE_ALPHA)
        model.fit(np.asarray(self._poses, dtype=float), np.asarray(self._residuals, dtype=float))
        self._model = model

        return True

    def correct(self, point: NormalizedPoint, pose: PoseTuple | None) -> NormalizedPoint:
        """Applies the pose-dependent shift to one predicted gaze point."""

        if self._model is None or pose is None:
            return point

        shift = self._model.predict(np.asarray([pose], dtype=float))[0]
        dx = float(np.clip(shift[0], -self.MAX_POSE_SHIFT, self.MAX_POSE_SHIFT))
        dy = float(np.clip(shift[1], -self.MAX_POSE_SHIFT, self.MAX_POSE_SHIFT))

        return NormalizedPoint(point.x + dx, point.y + dy)


class BiasRecalibration:
    """
    Single-point recalibration strategy.
    Gathers offset data from a single target to correct uniform drift without a full model refit.
    """

    MIN_FRAMES: Final[int] = 10

    def __init__(self) -> None:
        self._preds: list[NormalizedPoint] = []

    @property
    def n_fed(self) -> int:
        return len(self._preds)

    @property
    def predictions(self) -> list[NormalizedPoint]:
        return self._preds

    def feed(self, prediction: NormalizedPoint) -> None:
        """Collects one prediction while the user fixates the recalibration target."""

        self._preds.append(prediction)

    def drop_oldest_half(self) -> None:
        """Discard the older half of the batch, keeping the most recent frames."""

        if self._preds:
            self._preds = self._preds[len(self._preds) // 2:]

    def compute(self, target: NormalizedPoint) -> tuple[float, float] | None:
        """Offset (target minus mean prediction) in normalized units, or None
        when fewer than `MIN_FRAMES` predictions have been fed."""

        if len(self._preds) < self.MIN_FRAMES:
            return None

        mean_x = sum(p.x for p in self._preds) / len(self._preds)
        mean_y = sum(p.y for p in self._preds) / len(self._preds)

        return target.x - mean_x, target.y - mean_y

    def reset(self) -> None:
        self._preds.clear()


class RecalPhase(str, Enum):
    """Stage of a guided recalibration session."""

    PROMPT = "prompt"          # Explaining what is about to happen
    ARMED = "armed"            # Target shown, waiting for the user to fixate it
    COLLECTING = "collecting"  # Gathering predictions against the target
    DONE = "done"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class RecalStatus:
    """One frame's view of a recalibration session, for rendering."""

    phase: RecalPhase
    progress: float  # (0.0 - 1.0)
    message: str
    offset: tuple[float, float] | None = None

    @property
    def finished(self) -> bool:
        return self.phase in (RecalPhase.DONE, RecalPhase.CANCELLED)


class RecalibrationSession:
    """Guided single-point recalibration."""

    PROMPT_S: Final[float] = 2.5         # Time to read the instruction
    ARM_TIMEOUT_S: Final[float] = 12.0   # Give up if no fixation ever settles
    HOLD_TO_START_S: Final[float] = 0.4  # Steady fixation before collecting
    LOST_GRACE_S: Final[float] = 0.5     # Fixation gaps tolerated mid-collection
    MAX_SPREAD: Final[float] = 0.05      # Dispersion allowed within the batch

    def __init__(self, target: NormalizedPoint, started_at: float) -> None:
        self.target = target
        self._t0 = started_at
        self._phase = RecalPhase.PROMPT
        self._bias = BiasRecalibration()
        self._held_since: float | None = None
        self._lost_since: float | None = None
        self._offset: tuple[float, float] | None = None

    @property
    def phase(self) -> RecalPhase:
        return self._phase

    def _batch_is_tight(self) -> bool:
        """Whether the collected predictions form one cluster. This is the quality gate
        of full calibration applied to a single point.
        """

        preds = self._bias.predictions
        if len(preds) < 2:
            return False
        return (
            max(p.x for p in preds) - min(p.x for p in preds) <= self.MAX_SPREAD
            and max(p.y for p in preds) - min(p.y for p in preds) <= self.MAX_SPREAD
        )

    def _cancel(self, reason: str) -> RecalStatus:
        self._phase = RecalPhase.CANCELLED
        return RecalStatus(self._phase, 0.0, reason)

    def update(self, gaze: NormalizedPoint | None, fixating: bool, t: float) -> RecalStatus:
        """Advance the session by one frame and describe what to draw."""

        elapsed = t - self._t0

        if self._phase is RecalPhase.PROMPT:
            if elapsed < self.PROMPT_S:
                return RecalStatus(
                    self._phase,
                    min(1.0, elapsed / self.PROMPT_S),
                    "Recalibration - look at the dot in the center of the screen",
                )
            self._phase = RecalPhase.ARMED
            self._t0 = t
            elapsed = 0.0

        settled = gaze is not None and fixating

        if self._phase is RecalPhase.ARMED:
            if not settled:
                self._held_since = None
                if elapsed > self.ARM_TIMEOUT_S:
                    return self._cancel("Recalibration cancelled - gaze never settled")
                return RecalStatus(self._phase, 0.0, "Look at the dot and hold your gaze on it")

            if self._held_since is None:
                self._held_since = t
            held = t - self._held_since
            if held < self.HOLD_TO_START_S:
                return RecalStatus(
                    self._phase, held / self.HOLD_TO_START_S, "Hold your gaze on the dot"
                )

            self._phase = RecalPhase.COLLECTING
            self._lost_since = None

        if settled and gaze is not None:
            self._bias.feed(gaze)
            self._lost_since = None
        else:
            if self._lost_since is None:
                self._lost_since = t
            elif t - self._lost_since > self.LOST_GRACE_S:
                self._phase = RecalPhase.ARMED
                self._t0 = t
                self._held_since = None
                self._bias.reset()
                return RecalStatus(self._phase, 0.0, "Gaze lost - look at the dot again")

        progress = min(1.0, self._bias.n_fed / BiasRecalibration.MIN_FRAMES)
        offset = self._bias.compute(self.target)
        if offset is None:
            return RecalStatus(self._phase, progress, "Keep looking at the dot")

        if not self._batch_is_tight():
            self._bias.drop_oldest_half()
            return RecalStatus(self._phase, 0.0, "Hold still - keep looking at the dot")

        self._offset = offset
        self._phase = RecalPhase.DONE
        return RecalStatus(self._phase, 1.0, "Recalibrated", offset)

    @property
    def n_frames(self) -> int:
        return self._bias.n_fed