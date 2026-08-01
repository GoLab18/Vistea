"""
Eye-tracking calibration models, geometries, and strategies.
Maps raw gaze features to normalized screen coordinates.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import Enum, auto
from typing import Final, Protocol

import cv2
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

type FeatureVector = Sequence[float]


class FeatureSource(str, Enum):
    """Origin of the features used by the calibration model."""

    PCEC = "pcec"  # Pupil-Center - Eye-Corner vector
    L2CS = "l2cs"  # CNN gaze angles (pitch, yaw) in radians


# Sentinel value for distinguishing omitted arguments from explicit None.
_UNSET: Final = object()

# Time (in seconds) the eye naturally lags behind a moving target.
# Used to temporally align captured frames with the correct trajectory position.
PURSUIT_LAG_S: Final[float] = 0.150


@dataclass(frozen=True)
class NormalizedPoint:
    x: float
    y: float


class SegmentKind(Enum):
    """Execution mode for a calibration segment."""

    STATIC = auto()   # Fixed points - collects a batch of frames per static target.
    PURSUIT = auto()  # Moving target - collects frames continuously along a path.


@dataclass(frozen=True)
class Segment:
    """A single declarative step in a calibration plan."""

    kind: SegmentKind
    points: list[NormalizedPoint]
    duration_s: float = 0.0

    def position_at(self, t: float) -> NormalizedPoint:
        """Returns the interpolated trajectory position at time t."""

        if self.duration_s <= 0 or len(self.points) < 2:
            return self.points[0]
        frac = min(max(t / self.duration_s, 0.0), 1.0) * (len(self.points) - 1)
        i = int(frac)
        if i >= len(self.points) - 1:
            return self.points[-1]
        a, b = self.points[i], self.points[i + 1]
        w = frac - i
        return NormalizedPoint(a.x + (b.x - a.x) * w, a.y + (b.y - a.y) * w)


def make_grid(n_per_axis: int = 3, margin: float = 0.1) -> list[NormalizedPoint]:
    """Generates an n x n calibration grid with a safety margin."""

    if n_per_axis < 2:
        n_per_axis = 2
    coords = np.linspace(margin, 1.0 - margin, n_per_axis)
    return [NormalizedPoint(float(x), float(y)) for y in coords for x in coords]


def make_5point(margin: float = 0.1) -> list[NormalizedPoint]:
    """
    Traditional 5-point calibration grid (4 corners + 1 center).
    Historically used as a baseline in simpler eye-tracking systems.
    """

    coords = [margin, 1.0 - margin]
    pts = [NormalizedPoint(float(x), float(y)) for y in coords for x in coords]
    pts.append(NormalizedPoint(0.5, 0.5))
    return pts


def make_lissajous(
    n_points: int = 60, amplitude: float = 0.38, a: int = 3, b: int = 2
) -> list[NormalizedPoint]:
    """
    Generates a smooth, self-crossing Lissajous curve for pursuit segments.
    It is a standard choice for pursuit calibration trajectories.
    amplitude=0.38 keeps it within [0.12, 0.88], mirroring the safety margin of the static grids.
    """

    pts = []
    for i in range(n_points):
        t = (i / max(1, n_points - 1)) * 2 * math.pi
        pts.append(NormalizedPoint(0.5 + amplitude * math.sin(a * t), 0.5 + amplitude * math.sin(b * t)))
    return pts


def calculate_sample_spread(features_batch: Sequence[Sequence[float]]) -> float | None:
    """
    Calculates the scalar dispersion of a per-point calibration batch.
    Used for quality gating (rejecting unstable fixations).
    Returns the RMS of the per-feature-dimension standard deviations.
    """
    
    arr = np.asarray(features_batch, dtype=float)
    if arr.ndim != 2 or arr.shape[0] < 2:
        return None
    return float(np.sqrt(np.mean(np.std(arr, axis=0) ** 2)))


class CalibrationStrategy(Protocol):
    """Maps raw gaze features to normalized screen coordinates (0.0 - 1.0), independent of the physical resolution."""

    name: str
    use_head_pose: bool
    feature_source: FeatureSource

    # Quality gating parameters
    samples_per_point: int
    spread_threshold: float | None

    @property
    def sample_count(self) -> int:
        """Returns the total number of collected training samples."""
        ...

    def target_sequence(self) -> list[NormalizedPoint]:
        """Returns a simple sequence of static calibration targets."""
        ...

    def plan(self) -> list[Segment]:
        """Returns the complete execution plan (mixed static/pursuit segments)."""
        ...

    def add_samples(self, target: NormalizedPoint, features_batch: list[FeatureVector]) -> None:
        """Feeds a batch of captured feature vectors for a specific target."""
        ...

    def train(self) -> None:
        """Fits the underlying mapping model using the collected samples."""
        ...

    def is_calibrated(self) -> bool:
        """Checks whether the model has been successfully trained."""
        ...

    def predict(self, features: FeatureVector) -> NormalizedPoint | None:
        """Maps a single feature vector to a normalized screen coordinate."""
        ...

    def reset(self) -> None:
        """Clears all collected samples and resets the model's trained state."""
        ...


class _SampleStore:
    """Internal sample storage for calibration strategies."""

    def __init__(self) -> None:
        self._samples: list[tuple[FeatureVector, NormalizedPoint]] = []

    def add(self, target: NormalizedPoint, batch: list[FeatureVector]) -> None:
        for vec in batch:
            self._samples.append((list(vec), target))

    def matrices(self) -> tuple[np.ndarray, np.ndarray]:
        x = np.array([vec for vec, _ in self._samples], dtype=float)
        y = np.array([[t.x, t.y] for _, t in self._samples], dtype=float)
        return x, y

    def clear(self) -> None:
        self._samples.clear()

    def __len__(self) -> int:
        return len(self._samples)


class _RidgeMixin:
    """Shared polynomial-Ridge regression logic."""

    samples_per_point: int = 25
    spread_threshold: float | None = 0.006

    @property
    def sample_count(self) -> int:
        return len(self._store)

    def _init_model(self, alpha: float, degree: int) -> None:
        self._store = _SampleStore()
        self._model = make_pipeline(
            PolynomialFeatures(degree=degree, include_bias=False),
            Ridge(alpha=alpha),
        )
        self._trained = False

    def add_samples(self, target: NormalizedPoint, features_batch: list[FeatureVector]) -> None:
        self._store.add(target, features_batch)

    def train(self) -> None:
        x, y = self._store.matrices()
        if len(x) == 0:
            raise ValueError("No calibration samples collected.")
        self._model.fit(x, y)
        self._trained = True

    def is_calibrated(self) -> bool:
        return self._trained

    def predict(self, features: FeatureVector) -> NormalizedPoint | None:
        if not self._trained:
            return None
        pred = self._model.predict(np.asarray(features, dtype=float).reshape(1, -1))[0]
        return NormalizedPoint(
            float(np.clip(pred[0], 0.0, 1.0)),
            float(np.clip(pred[1], 0.0, 1.0)),
        )

    def reset(self) -> None:
        self._store.clear()
        self._trained = False


class PolynomialRidgeCalibration(_RidgeMixin):
    """
    Non-linear calibration: 2nd-degree polynomial expansion of the
    eye-relative feature vector, followed by multi-output Ridge regression.

    Rationale: the eyeball is a sphere projected onto a flat display, so the
    feature-to-screen mapping is non-linear, and quadratic terms absorb
    that curvature. Because every calibration point contributes dozens of
    raw frames rather than one averaged value, the regularized least-squares
    problem is well conditioned and the detector noise acts as natural data
    augmentation.
    """

    name = "ridge_poly2_custom"
    feature_source = FeatureSource.PCEC

    def __init__(
        self,
        alpha: float = 1e-3,
        degree: int = 2,
        margin: float = 0.1,
        grid: list[NormalizedPoint] | None = None,
        use_head_pose: bool = False,
    ) -> None:
        self._targets = grid if grid is not None else make_grid(3, margin)
        self.use_head_pose = use_head_pose
        self._init_model(alpha, degree)

    def target_sequence(self) -> list[NormalizedPoint]:
        return self._targets

    def plan(self) -> list[Segment]:
        return [Segment(SegmentKind.STATIC, list(self._targets))]


class HomographyCalibration:
    """
    Linear baseline strategy: a projective transform (homography) between
    the averaged 2D eye vector and the screen plane.

    This is the closest well-posed equivalent of the piecewise-linear
    interpolation used in earlier webcam eye-tracking projects, and serves
    as the reference point for evaluating the non-linear model.
    """

    name = "homography_custom"
    use_head_pose = False
    feature_source = FeatureSource.PCEC
    samples_per_point: int = 25
    spread_threshold: float | None = 0.006

    def __init__(self, margin: float = 0.1, grid: list[NormalizedPoint] | None = None) -> None:
        self._targets = grid if grid is not None else make_grid(3, margin)
        self._per_target: dict[NormalizedPoint, list[FeatureVector]] = {}
        self._h: np.ndarray | None = None

    @property
    def sample_count(self) -> int:
        return sum(len(batch) for batch in self._per_target.values())

    def target_sequence(self) -> list[NormalizedPoint]:
        return self._targets

    def plan(self) -> list[Segment]:
        return [Segment(SegmentKind.STATIC, list(self._targets))]

    @staticmethod
    def _avg_eye(vec: FeatureVector) -> tuple[float, float]:
        lh, lv, rh, rv = vec[:4]
        return (lh + rh) / 2.0, (lv + rv) / 2.0

    def add_samples(self, target: NormalizedPoint, features_batch: list[FeatureVector]) -> None:
        self._per_target.setdefault(target, []).extend(list(v) for v in features_batch)

    def train(self) -> None:
        src, dst = [], []
        for target, batch in self._per_target.items():
            pts = np.array([self._avg_eye(v) for v in batch], dtype=float)
            src.append(np.median(pts, axis=0))
            dst.append([target.x, target.y])
        if len(src) < 4:
            raise ValueError("Homography requires at least 4 calibration points.")
        h, _ = cv2.findHomography(np.array(src, dtype=np.float32), np.array(dst, dtype=np.float32))
        if h is None:
            raise ValueError("Homography estimation failed (degenerate geometry).")
        self._h = h

    def is_calibrated(self) -> bool:
        return self._h is not None

    def predict(self, features: FeatureVector) -> NormalizedPoint | None:
        if self._h is None:
            return None
        pt = np.array([[self._avg_eye(features)]], dtype=np.float32)
        mapped = cv2.perspectiveTransform(pt, self._h)[0][0]
        return NormalizedPoint(
            float(np.clip(mapped[0], 0.0, 1.0)),
            float(np.clip(mapped[1], 0.0, 1.0)),
        )

    def reset(self) -> None:
        self._per_target.clear()
        self._h = None


class SmoothPursuitCalibration(_RidgeMixin):
    """
    Pursuit Calibration (Pfeuffer et al., UIST 2013): the target moves
    smoothly along a Lissajous trajectory and every valid frame becomes a
    training sample.
    """

    name = "smooth_pursuit_60pt"
    use_head_pose = False
    feature_source = FeatureSource.PCEC

    PATH_RESOLUTION = 480

    def __init__(self, alpha: float = 1e-3, degree: int = 2, duration_s: float = 24.0) -> None:
        self.duration_s = duration_s
        self._targets = make_lissajous(60) # nominal sequence (reporting/tests)
        self._init_model(alpha, degree)

    def target_sequence(self) -> list[NormalizedPoint]:
        return self._targets

    def plan(self) -> list[Segment]:
        return [Segment(SegmentKind.PURSUIT, make_lissajous(self.PATH_RESOLUTION), self.duration_s)]


class HybridCalibration(_RidgeMixin):
    """Combines a static grid with a continuous pursuit trajectory."""

    name = "hybrid_9pt_pursuit"
    use_head_pose = False
    feature_source = FeatureSource.PCEC

    PATH_RESOLUTION = 480

    def __init__(
        self,
        alpha: float = 1e-3,
        degree: int = 2,
        duration_s: float = 24.0,
        margin: float = 0.1,
        use_head_pose: bool = False,
    ) -> None:
        self.duration_s = duration_s
        self._grid = make_grid(3, margin)
        self.use_head_pose = use_head_pose
        self._init_model(alpha, degree)

    def target_sequence(self) -> list[NormalizedPoint]:
        # Anchors + nominal pursuit waypoints (used for grid_size logging)
        return self._grid + make_lissajous(60)

    def plan(self) -> list[Segment]:
        return [
            Segment(SegmentKind.STATIC, list(self._grid)),
            Segment(SegmentKind.PURSUIT, make_lissajous(self.PATH_RESOLUTION), self.duration_s),
        ]


def _configure(
    strategy: CalibrationStrategy,
    *,
    name: str,
    feature_source: FeatureSource | None = None,
    samples_per_point: int | None = None,
    spread_threshold: float | None | object = _UNSET,
) -> CalibrationStrategy:
    """Override a strategy's registry-facing attributes."""

    strategy.name = name
    if feature_source is not None:
        strategy.feature_source = feature_source
    if samples_per_point is not None:
        strategy.samples_per_point = samples_per_point
    if spread_threshold is not _UNSET:
        strategy.spread_threshold = spread_threshold  # type: ignore[assignment]
    return strategy


DEFAULT_STRATEGY_KEY: Final[str] = "Ridge 9-pt + pose (default)"

STRATEGY_REGISTRY: dict[str, Callable[[], CalibrationStrategy]] = {
    "Ridge 5-pt": lambda: _configure(
        PolynomialRidgeCalibration(grid=make_5point()), name="ridge_poly2_5pt"
    ),
    "Ridge 9-pt": lambda: _configure(
        PolynomialRidgeCalibration(grid=make_grid(3)), name="ridge_poly2_9pt"
    ),
    "Ridge 16-pt": lambda: _configure(
        PolynomialRidgeCalibration(grid=make_grid(4)), name="ridge_poly2_16pt"
    ),
    "Ridge 16-pt + pose": lambda: _configure(
        PolynomialRidgeCalibration(grid=make_grid(4), use_head_pose=True),
        name="ridge_poly2_16pt_pose",
    ),
    "Homography (9-pt)": lambda: _configure(
        HomographyCalibration(grid=make_grid(3)), name="homography_9pt"
    ),
    "Smooth Pursuit (24 s)": lambda: SmoothPursuitCalibration(),
    "Hybrid 9-pt + Pursuit": lambda: HybridCalibration(),
    DEFAULT_STRATEGY_KEY: lambda: _configure(
        PolynomialRidgeCalibration(grid=make_grid(3), use_head_pose=True),
        name="ridge_poly2_9pt_pose",
    ),
    "Hybrid + pose": lambda: _configure(
        HybridCalibration(use_head_pose=True), name="hybrid_9pt_pursuit_pose"
    ),

    # Appearance family: L2CS emits gaze angles in RADIANS at a few Hz, so the
    # PC-EC capture budget does not transfer. 12 unique predictions suffice for
    # a 2D linear ridge, and 0.08 rad (~4.6 deg) is a starting point for the
    # spread gate tuned from logs.
    "L2CS-Net + linear (CNN)": lambda: _configure(
        PolynomialRidgeCalibration(grid=make_grid(3), degree=1, alpha=1e-4),
        name="l2cs_linear",
        feature_source=FeatureSource.L2CS,
        samples_per_point=12,
        spread_threshold=0.08,
    ),
}