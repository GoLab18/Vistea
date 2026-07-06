"""Drift correction from dwell activations + gaze-UI sizing rules.

**Drift correction.** Same-session validation overstates real accuracy
because the mapping drifts over time (posture creep, lighting, fatigue).
WebGazer's answer is self-labeling: every mouse click is assumed to be looked
at and becomes a training sample. Vistea has no mouse - but a *dwell
activation is an even stronger label*: the user provably fixated the tile
for the whole dwell time. ``DriftCorrector`` therefore treats each activated
tile as ground truth: the feature vectors buffered during the dwell are added
to the calibration strategy's sample store with the tile centre as target,
and the model is retrained (Ridge refit on a few hundred samples is
milliseconds). Guards: a cap on total corrections (the store must not be
swamped by interaction data relative to the calibration set) and a per-click
cap on frames.

**Tile sizing.** Targets are sized from the system's own measured error:
minimum tile side = 2x mean validation error in pixels (a fixation landing
anywhere within +-error of the tile centre still hits the tile), with an
ergonomic floor. This closes the loop measurement -> design.
"""

from __future__ import annotations

from collections import deque
from typing import Deque, List, Optional

from core.calibration import CalibrationStrategy, NormalizedPoint

FeatureVector = List[float]

TILE_FLOOR_PX = 80 # ergonomic minimum regardless of measured accuracy


def min_tile_side_px(mean_error_px: Optional[float], factor: float = 2.0) -> int:
    if mean_error_px is None or mean_error_px != mean_error_px: # None/NaN
        return TILE_FLOOR_PX
    return max(TILE_FLOOR_PX, int(round(factor * mean_error_px)))


class DriftCorrector:
    def __init__(
        self,
        strategy: CalibrationStrategy,
        frames_per_activation: int = 8,
        max_corrections: int = 40,
        buffer_size: int = 30,
    ):
        self.strategy = strategy
        self.frames_per_activation = frames_per_activation
        self.max_corrections = max_corrections
        self._buffer: Deque[FeatureVector] = deque(maxlen=buffer_size)
        self.corrections_applied = 0

    def observe(self, vec: Optional[FeatureVector]) -> None:
        """Feed every valid feature vector while the user dwells."""

        if vec is not None:
            self._buffer.append(vec)

    def clear(self) -> None:
        self._buffer.clear()

    def on_activation(self, tile_center: NormalizedPoint) -> int:
        """Label the dwell frames with the activated tile centre and retrain.
        Returns the number of samples added (0 if capped/insufficient)."""

        if self.corrections_applied >= self.max_corrections:
            return 0
        if not self.strategy.is_calibrated():
            return 0
        batch = list(self._buffer)[-self.frames_per_activation:]
        if len(batch) < 3: # too little signal to label safely
            return 0
        self.strategy.add_samples(tile_center, batch)
        self.strategy.train()
        self.corrections_applied += 1
        self._buffer.clear()
        return len(batch)
