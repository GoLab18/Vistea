"""
Frame preprocessing strategies for lighting robustness.

MediaPipe's landmark networks were trained with heavy illumination
augmentation, but training-time robustness cannot undo sensor physics:
in dim light the camera raises exposure time and gain, producing motion blur
and shot noise that degrade iris-centre localisation (measured baseline:
dim x no-preprocessing = 3.96 deg vs ~1.69 deg in daylight). What image-side
preprocessing CAN do is restore local contrast for the detector. This module
makes that a pluggable, *measurable* variable.

Design decisions:
  * Strategy Protocol (same pattern as PointFilter / CalibrationStrategy) so
    variants are swappable at runtime from a UI dropdown, and the active
    variant's ``name`` is bound into the research log — every validation run
    is then automatically labelled with (lighting_condition, preprocessing),
    and the thesis table falls out of a pandas pivot.
  * Luma-only processing: frames are converted to YCrCb and only the Y (luma)
    channel is equalised, then converted back. Equalising BGR channels
    independently would shift hues, feeding the face detector colour
    distributions it never saw in training.
  * CLAHE vs global equalisation: global histogram equalisation is a single
    monotone mapping for the whole frame — with a bright monitor glow on the
    face and a dark room behind, the background dominates the histogram and
    the face gains little, while sensor noise in dark regions is amplified.
    CLAHE (Contrast Limited Adaptive Histogram Equalization) equalises per
    tile and clips the histogram (clip_limit) precisely to bound that noise
    amplification — the textbook remedy for exactly this failure mode
    (Zuiderveld 1994; standard OpenCV implementation).

The swap in GazeEstimator is a single attribute assignment (atomic under the
GIL), so it is safe to change from the UI thread while the camera thread reads.
"""

from __future__ import annotations

from typing import Callable, Final, List, Protocol

import cv2
import numpy as np


class FramePreprocessor(Protocol):
    """Pluggable BGR-frame-in, BGR-frame-out preprocessing step."""

    name: str

    def process(self, frame_bgr: np.ndarray) -> np.ndarray:
        ...


class NoPreprocessing:
    """Identity — the control condition every variant is measured against."""

    name = "none"

    def process(self, frame_bgr: np.ndarray) -> np.ndarray:
        return frame_bgr


class HistogramEqualization:
    """Global histogram equalisation of the luma channel."""

    name = "hist_eq"

    def process(self, frame_bgr: np.ndarray) -> np.ndarray:
        ycrcb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YCrCb)
        ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
        return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)


class ClahePreprocessing:
    """Contrast Limited Adaptive Histogram Equalization on the luma channel.

    clip_limit bounds per-tile contrast gain, i.e. bounds noise amplification
    in dark, low-signal regions; tile_grid localises equalisation so a bright
    monitor-lit face and a dark background are treated separately.
    """

    name = "clahe"

    def __init__(self, clip_limit: float = 2.0, tile_grid: int = 8):
        self._clahe = cv2.createCLAHE(
            clipLimit=clip_limit, tileGridSize=(tile_grid, tile_grid)
        )

    def process(self, frame_bgr: np.ndarray) -> np.ndarray:
        ycrcb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YCrCb)
        ycrcb[:, :, 0] = self._clahe.apply(ycrcb[:, :, 0])
        return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)


PREPROCESSOR_FACTORIES: Final[dict[str, Callable[[], FramePreprocessor]]] = {
    "None (control)": NoPreprocessing,
    "Histogram equalization (global)": HistogramEqualization,
    "CLAHE (adaptive)": ClahePreprocessing,
}

# Manual label chosen by the operator before a measurement session.
# It's bound into the research log as ``lighting_condition`` so every event carries it.
LIGHTING_CONDITIONS: Final[List[str]] = [
    "unknown",
    "daylight",
    "incandescent",   # incandescent / evening artificial
    "side",           # strong side light
    "dim",            # dim (twilight / screen only)
]