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

from core.calibration import CalibrationStrategy, NormalizedPoint

type FeatureVector = list[float]

TILE_FLOOR_PX = 80 # ergonomic minimum regardless of measured accuracy


def min_tile_side_px(mean_error_px: float | None, factor: float = 2.0) -> int:
    """Calculates the advisory minimum tile side to match the user's calibration error."""

    if mean_error_px is None or math.isnan(mean_error_px):
        return TILE_FLOOR_PX
    return max(TILE_FLOOR_PX, int(round(factor * mean_error_px)))


class DriftCorrector:
    """
    Treats deliberate UI dwell activations as ground-truth labels.
    Retrains the calibration model incrementally to fight mapping drift.
    """

    def __init__(
        self,
        strategy: CalibrationStrategy,
        frames_per_activation: int = 8,
        max_corrections: int = 40,
        buffer_size: int = 30,
    ) -> None:
        self.strategy = strategy
        self.frames_per_activation = frames_per_activation
        self.max_corrections = max_corrections
        self._buffer: deque[FeatureVector] = deque(maxlen=buffer_size)
        self.corrections_applied = 0

    def observe(self, vec: FeatureVector | None) -> None:
        """Buffers feature vectors continuously while the user dwells."""

        if vec is not None:
            self._buffer.append(vec)

    def clear(self) -> None:
        self._buffer.clear()

    def _batch_error_norm(self, batch: list[FeatureVector], target: NormalizedPoint) -> float:
        """Mean distance (normalized units) between model predictions and target."""

        errs = []
        for vec in batch:
            pred = self.strategy.predict(vec)
            if pred is not None:
                errs.append(math.hypot(pred.x - target.x, pred.y - target.y))
        return sum(errs) / len(errs) if errs else float("nan")

    def on_activation(self, tile_center: NormalizedPoint) -> tuple[int, float, float] | None:
        """
        Labels the buffered frames with the activated tile center and triggers a model refit.
        Returns: (n_samples, pre_error_norm, post_error_norm) or `None` when not retrained.
        """

        if self.corrections_applied >= self.max_corrections:
            return None
        if not self.strategy.is_calibrated():
            return None

        batch = list(self._buffer)[-self.frames_per_activation:]
        if len(batch) < 3: 
            return None # Too little signal to label safely

        pre_error_norm = self._batch_error_norm(batch, tile_center)

        self.strategy.add_samples(tile_center, batch)
        self.strategy.train()

        post_error_norm = self._batch_error_norm(batch, tile_center)

        self.corrections_applied += 1
        self._buffer.clear()

        return len(batch), pre_error_norm, post_error_norm


class BiasRecalibration:
    """
    Single-point recalibration strategy.
    Gathers offset data from a single target to correct uniform drift without a full model refit.
    """

    MIN_FRAMES = 10

    def __init__(self) -> None:
        self._preds: list[NormalizedPoint] = []

    @property
    def n_fed(self) -> int:
        return len(self._preds)

    def feed(self, prediction: NormalizedPoint) -> None:
        """Collects one prediction while the user fixates the recalibration target."""

        self._preds.append(prediction)

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