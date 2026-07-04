import numpy as np
import cv2
from dataclasses import dataclass
from typing import Dict, Final, List, Optional, Protocol, Sequence, Tuple, Callable
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

FeatureVector = Sequence[float]


@dataclass(frozen=True)
class NormalizedPoint:
    x: float
    y: float


def make_grid(n_per_axis: int = 3, margin: float = 0.1) -> List[NormalizedPoint]:
    """
    Standard n x n calibration grid with a safety margin.
    
    The margin keeps extreme targets away from the display edges to avoid
    unnaturally large eyeball rotations and to keep the irises well inside
    the camera frame.
    """

    if n_per_axis < 2:
        n_per_axis = 2
    coords = np.linspace(margin, 1.0 - margin, n_per_axis)
    return [NormalizedPoint(float(x), float(y)) for y in coords for x in coords]


def make_5point(margin: float = 0.1) -> List[NormalizedPoint]:
    """
    Traditional 5-point calibration grid (4 corners + 1 center).
    Historically used as a baseline in simpler eye-tracking systems.
    """

    coords = [margin, 1.0 - margin]
    pts = [NormalizedPoint(float(x), float(y)) for y in coords for x in coords]
    pts.append(NormalizedPoint(0.5, 0.5))
    return pts


class CalibrationStrategy(Protocol):
    """
    Pluggable strategy mapping gaze features onto normalized screen
    coordinates (0-1 on both axes, independent of the physical resolution).

    The lifecycle is: for each target returned by target_sequence() the UI
    collects a *batch* of feature vectors (many consecutive frames) and
    passes it to add_samples(). After the last target train() is called.
    """

    name: str

    def target_sequence(self) -> List[NormalizedPoint]:
        ...

    def add_samples(self, target: NormalizedPoint, features_batch: List[FeatureVector]) -> None:
        ...

    def train(self) -> None:
        ...

    def is_calibrated(self) -> bool:
        ...

    def predict(self, features: FeatureVector) -> Optional[NormalizedPoint]:
        ...

    def reset(self) -> None:
        ...


class _SampleStore:
    """Shared sample bookkeeping for concrete strategies."""

    def __init__(self):
        self._samples: List[Tuple[FeatureVector, NormalizedPoint]] = []

    def add(self, target: NormalizedPoint, batch: List[FeatureVector]) -> None:
        for vec in batch:
            self._samples.append((list(vec), target))

    def matrices(self) -> Tuple[np.ndarray, np.ndarray]:
        x = np.array([vec for vec, _ in self._samples], dtype=float)
        y = np.array([[t.x, t.y] for _, t in self._samples], dtype=float)
        return x, y

    def clear(self) -> None:
        self._samples.clear()

    def __len__(self) -> int:
        return len(self._samples)


class PolynomialRidgeCalibration:
    """
    Non-linear calibration: 2nd-degree polynomial expansion of the
    4D eye-relative feature vector, followed by multi-output Ridge
    regression (scikit-learn's Ridge supports 2D targets natively).

    Rationale: the eyeball is a sphere projected onto a flat display, so the
    feature-to-screen mapping is mildly non-linear; quadratic terms absorb
    that curvature (approach analogous to WebGazer's ridge mapping,
    Papoutsaki et al. 2016). Because every calibration point contributes
    dozens of raw frames rather than one averaged value, the regularized
    least-squares problem is well conditioned and the detector noise acts
    as natural data augmentation.
    """

    name = "ridge_poly2_custom"

    def __init__(self, alpha: float = 1e-3, degree: int = 2, margin: float = 0.1, grid: Optional[List[NormalizedPoint]] = None):
        self._targets = grid if grid is not None else make_grid(3, margin)
        self._store = _SampleStore()
        self._model = make_pipeline(
            PolynomialFeatures(degree=degree, include_bias=False),
            Ridge(alpha=alpha),
        )
        self._trained = False

    def target_sequence(self) -> List[NormalizedPoint]:
        return list(self._targets)

    def add_samples(self, target: NormalizedPoint, features_batch: List[FeatureVector]) -> None:
        self._store.add(target, features_batch)

    def train(self) -> None:
        x, y = self._store.matrices()
        if len(x) == 0:
            raise ValueError("No calibration samples collected.")
        self._model.fit(x, y)
        self._trained = True

    def is_calibrated(self) -> bool:
        return self._trained

    def predict(self, features: FeatureVector) -> Optional[NormalizedPoint]:
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


class HomographyCalibration:
    """
    Linear baseline strategy: a projective transform (homography) between
    the averaged 2D eye vector and the screen plane, estimated with OpenCV
    from the target grid.

    This is the closest well-posed equivalent of the piecewise-linear
    interpolation used in earlier webcam eye-tracking projects, and serves
    as the reference point for evaluating the non-linear model.
    """

    name = "homography_custom"

    def __init__(self, margin: float = 0.1, grid: Optional[List[NormalizedPoint]] = None):
        self._targets = grid if grid is not None else make_grid(3, margin)
        self._per_target: Dict[NormalizedPoint, List[FeatureVector]] = {}
        self._h: Optional[np.ndarray] = None

    def target_sequence(self) -> List[NormalizedPoint]:
        return list(self._targets)

    @staticmethod
    def _avg_eye(vec: FeatureVector) -> Tuple[float, float]:
        lh, lv, rh, rv = vec
        return (lh + rh) / 2.0, (lv + rv) / 2.0

    def add_samples(self, target: NormalizedPoint, features_batch: List[FeatureVector]) -> None:
        self._per_target.setdefault(target, []).extend(list(v) for v in features_batch)

    def train(self) -> None:
        src, dst = [], []
        for target, batch in self._per_target.items():
            pts = np.array([self._avg_eye(v) for v in batch], dtype=float)
            src.append(np.median(pts, axis=0))  # median = outlier-robust
            dst.append([target.x, target.y])
        if len(src) < 4:
            raise ValueError("Homography requires at least 4 calibration points.")
        h, _ = cv2.findHomography(np.array(src, dtype=np.float32), np.array(dst, dtype=np.float32))
        if h is None:
            raise ValueError("Homography estimation failed (degenerate geometry).")
        self._h = h

    def is_calibrated(self) -> bool:
        return self._h is not None

    def predict(self, features: FeatureVector) -> Optional[NormalizedPoint]:
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


def _set_name(strategy: CalibrationStrategy, name: str) -> CalibrationStrategy:
    """Helper to cleanly override the strategy name for logging purposes."""

    strategy.name = name
    return strategy


DEFAULT_STRATEGY_KEY: Final[str] = "Ridge 9-pt (default)"

# Registry using factory lambdas to allow injecting distinct grid sizes
STRATEGY_REGISTRY: dict[str, Callable[[], CalibrationStrategy]] = {
    "Ridge 5-pt": lambda: _set_name(PolynomialRidgeCalibration(grid=make_5point()), "ridge_poly2_5pt"),
    DEFAULT_STRATEGY_KEY: lambda: _set_name(PolynomialRidgeCalibration(grid=make_grid(3)), "ridge_poly2_9pt"),
    "Ridge 16-pt": lambda: _set_name(PolynomialRidgeCalibration(grid=make_grid(4)), "ridge_poly2_16pt"),
    "Homography (9-pt)": lambda: _set_name(HomographyCalibration(grid=make_grid(3)), "homography_9pt"),
}