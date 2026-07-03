import numpy as np
import cv2
from dataclasses import dataclass
from typing import Dict, Final, List, Optional, Protocol, Sequence, Tuple
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

FeatureVector = Sequence[float]


@dataclass(frozen=True)
class NormalizedPoint:
    x: float
    y: float


def make_grid(margin: float = 0.1) -> List[NormalizedPoint]:
    """Standard 3x3 calibration grid with a safety margin.

    The margin keeps extreme targets away from the display edges to avoid
    unnaturally large eyeball rotations and to keep the irises well inside
    the camera frame.
    """
    coords = (margin, 0.5, 1.0 - margin)
    return [NormalizedPoint(x, y) for y in coords for x in coords]


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
    Non-linear 9-point calibration: 2nd-degree polynomial expansion of the
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

    name = "ridge_poly2_9pt"

    def __init__(self, alpha: float = 1e-3, degree: int = 2, margin: float = 0.1):
        self._targets = make_grid(margin)
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
    from the same 9-point grid.

    This is the closest well-posed equivalent of the piecewise-linear
    interpolation used in earlier webcam eye-tracking projects, and serves
    as the reference point for evaluating the non-linear model.
    """

    name = "homography_9pt"

    def __init__(self, margin: float = 0.1):
        self._targets = make_grid(margin)
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


STRATEGY_REGISTRY: Final[dict[str, type[CalibrationStrategy]]] = {
    "Polynomial Ridge (9-point)": PolynomialRidgeCalibration,
    "Homography baseline (9-point)": HomographyCalibration,
}