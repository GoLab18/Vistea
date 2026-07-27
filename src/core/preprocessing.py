"""
Frame preprocessing strategies for lighting robustness.

Restores local contrast for the detector in adverse lighting conditions.
Operates strictly on the Luma (Y) channel to prevent color hue shifts
that could disrupt the pre-trained neural networks.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Final, Protocol

import cv2
import numpy as np


class FramePreprocessor(Protocol):
    """Pluggable BGR-frame-in, BGR-frame-out preprocessing step."""

    name: str

    def process(self, frame_bgr: np.ndarray) -> np.ndarray: ...


class NoPreprocessing:
    """Identity pass-through. Used as the control condition."""

    name = "none"

    def process(self, frame_bgr: np.ndarray) -> np.ndarray:
        return frame_bgr


class HistogramEqualization:
    """Global histogram equalization of the luma channel."""

    name = "hist_eq"

    def process(self, frame_bgr: np.ndarray) -> np.ndarray:
        ycrcb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YCrCb)
        ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
        return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)


class ClahePreprocessing:
    """
    Contrast Limited Adaptive Histogram Equalization on the luma channel.

    Prevents background noise amplification in low-light conditions by clipping
    per-tile contrast gain (Zuiderveld, 1994).
    """

    name = "clahe"

    def __init__(self, clip_limit: float = 2.0, tile_grid: int = 8) -> None:
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

# Manual labels chosen by the operator before a measurement session.
# Bound into the research log as ``lighting_condition``.
LIGHTING_CONDITIONS: Final[list[str]] = [
    "unknown",
    "daylight",
    "incandescent",   # incandescent / evening artificial
    "side",           # strong side light
    "dim",            # dim (twilight / screen only)
]