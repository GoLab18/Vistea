import math
import numpy as np
import cv2
from dataclasses import dataclass, field
from typing import Dict, Final, List, Optional, Protocol, Sequence, Tuple, Callable
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

FeatureVector = Sequence[float]

# Oculomotor pursuit latency: the eye tracks a moving target ~100-200 ms
# behind it (Pfeuffer et al., UIST 2013 use a comparable correction).
# During pursuit segments each captured frame is therefore labelled with the
# target position PURSUIT_LAG_S seconds *before* the capture time. Doing the
# correction time-based at collection (in the overlay interpreter) is exact
# under camera FPS jitter, unlike shifting the sample array by a fixed frame
# count, which silently mislabels data whenever FPS deviates from the assumed
# rate - that was the flaw of the first draft of this strategy.
PURSUIT_LAG_S: Final[float] = 0.150


@dataclass(frozen=True)
class NormalizedPoint:
    x: float
    y: float


@dataclass(frozen=True)
class Segment:
    """
    One declarative step of a calibration plan (Plan-as-Data).

    The strategy declares WHAT should happen; the full-screen overlay is a
    generic interpreter that executes segments without knowing which strategy
    produced them. This keeps UI code strategy-agnostic and makes new
    calibration schemes (hybrid, future variants) pure data recombinations.

    kind = "static":  ``points`` are dwell targets - the user fixates each dot
                      and a fixed-size batch of frames is collected per point
                      (with quality gating).
    kind = "pursuit": ``points`` is a dense polyline approximating a smooth
                      trajectory; the interpreter animates the target along it
                      for ``duration_s`` seconds, collecting every valid frame
                      continuously and labelling it with the trajectory
                      position PURSUIT_LAG_S earlier.
    """

    kind: str
    points: List[NormalizedPoint]
    duration_s: float = 0.0

    def position_at(self, t: float) -> NormalizedPoint:
        """Interpolated trajectory position at time t (pursuit segments)."""

        if self.duration_s <= 0 or len(self.points) < 2:
            return self.points[0]
        frac = min(max(t / self.duration_s, 0.0), 1.0) * (len(self.points) - 1)
        i = int(frac)
        if i >= len(self.points) - 1:
            return self.points[-1]
        a, b = self.points[i], self.points[i + 1]
        w = frac - i
        return NormalizedPoint(a.x + (b.x - a.x) * w, a.y + (b.y - a.y) * w)


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


def make_lissajous(
    n_points: int = 60, amplitude: float = 0.38, a: int = 3, b: int = 2
) -> List[NormalizedPoint]:
    """
    Lissajous figure (x ~ sin(a*t), y ~ sin(b*t)) - a smooth, self-crossing
    curve that sweeps the whole plane, standard choice for pursuit
    calibration trajectories. amplitude=0.38 keeps it within [0.12, 0.88],
    mirroring the safety margin of the static grids.
    """

    pts = []
    for i in range(n_points):
        t = (i / max(1, n_points - 1)) * 2 * math.pi
        pts.append(NormalizedPoint(0.5 + amplitude * math.sin(a * t),
                                   0.5 + amplitude * math.sin(b * t)))
    return pts


class CalibrationStrategy(Protocol):
    """
    Pluggable strategy mapping gaze features onto normalized screen
    coordinates (0-1 on both axes, independent of the physical resolution).

    Static lifecycle: for each target returned by target_sequence() the UI
    collects a *batch* of feature vectors and passes it to add_samples();
    after the last target train() is called.

    Plan lifecycle: the UI interprets plan(), which generalizes the above to mixed static/pursuit schedules.
    The default plan of purely static strategies is a single static segment
    over target_sequence(), so both lifecycles stay equivalent for them.
    """

    name: str
    use_head_pose: bool # expects pose-augmented feature vectors

    def target_sequence(self) -> List[NormalizedPoint]:
        ...

    def plan(self) -> List[Segment]:
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


class _RidgeMixin:
    """Shared polynomial-Ridge fitting/prediction used by three strategies."""

    def _init_model(self, alpha: float, degree: int) -> None:
        self._store = _SampleStore()
        self._model = make_pipeline(
            PolynomialFeatures(degree=degree, include_bias=False),
            Ridge(alpha=alpha),
        )
        self._trained = False

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


class PolynomialRidgeCalibration(_RidgeMixin):
    """
    Non-linear calibration: 2nd-degree polynomial expansion of the
    eye-relative feature vector, followed by multi-output Ridge regression
    (scikit-learn's Ridge supports 2D targets natively).

    Rationale: the eyeball is a sphere projected onto a flat display, so the
    feature-to-screen mapping is mildly non-linear; quadratic terms absorb
    that curvature (approach analogous to WebGazer's ridge mapping,
    Papoutsaki et al. 2016). Because every calibration point contributes
    dozens of raw frames rather than one averaged value, the regularized
    least-squares problem is well conditioned and the detector noise acts
    as natural data augmentation.

    The model is dimension-agnostic: it fits whatever feature length it is
    given, so the same class serves the 4D PC-EC vector and the 7D
    pose-augmented vector - the difference is declared via
    ``use_head_pose`` and resolved at feature-building time in the UI.
    """

    name = "ridge_poly2_custom"

    def __init__(
        self,
        alpha: float = 1e-3,
        degree: int = 2,
        margin: float = 0.1,
        grid: Optional[List[NormalizedPoint]] = None,
        use_head_pose: bool = False,
    ):
        self._targets = grid if grid is not None else make_grid(3, margin)
        self.use_head_pose = use_head_pose
        self._init_model(alpha, degree)

    def target_sequence(self) -> List[NormalizedPoint]:
        return list(self._targets)

    def plan(self) -> List[Segment]:
        return [Segment("static", list(self._targets))]


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
    use_head_pose = False  # 2D->2D transform cannot consume pose

    def __init__(self, margin: float = 0.1, grid: Optional[List[NormalizedPoint]] = None):
        self._targets = grid if grid is not None else make_grid(3, margin)
        self._per_target: Dict[NormalizedPoint, List[FeatureVector]] = {}
        self._h: Optional[np.ndarray] = None

    def target_sequence(self) -> List[NormalizedPoint]:
        return list(self._targets)

    def plan(self) -> List[Segment]:
        return [Segment("static", list(self._targets))]

    @staticmethod
    def _avg_eye(vec: FeatureVector) -> Tuple[float, float]:
        lh, lv, rh, rv = vec[:4]
        return (lh + rh) / 2.0, (lv + rv) / 2.0

    def add_samples(self, target: NormalizedPoint, features_batch: List[FeatureVector]) -> None:
        self._per_target.setdefault(target, []).extend(list(v) for v in features_batch)

    def train(self) -> None:
        src, dst = [], []
        for target, batch in self._per_target.items():
            pts = np.array([self._avg_eye(v) for v in batch], dtype=float)
            src.append(np.median(pts, axis=0)) # median = outlier-robust
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


class SmoothPursuitCalibration(_RidgeMixin):
    """
    Pursuit Calibration (Pfeuffer et al., UIST 2013): the target moves
    smoothly along a Lissajous trajectory and every valid frame becomes a
    training sample, yielding dense planar coverage in a fixed time budget
    (duration_s) instead of a fixed number of fixations.

    Oculomotor-lag correction is NOT done here: the overlay interpreter
    labels each sample with the trajectory position PURSUIT_LAG_S seconds
    before capture time (clock-based, exact under FPS jitter). train() fits
    the labelled samples as-is.
    """

    name = "smooth_pursuit_60pt"
    use_head_pose = False

    # Dense polyline resolution for smooth on-screen animation.
    PATH_RESOLUTION = 480

    def __init__(self, alpha: float = 1e-3, degree: int = 2, duration_s: float = 24.0):
        self.duration_s = duration_s
        self._targets = make_lissajous(60) # nominal sequence (reporting/tests)
        self._init_model(alpha, degree)

    def target_sequence(self) -> List[NormalizedPoint]:
        return list(self._targets)

    def plan(self) -> List[Segment]:
        return [Segment("pursuit", make_lissajous(self.PATH_RESOLUTION), self.duration_s)]


class HybridCalibration(_RidgeMixin):
    """
    Hybrid scheme: a static 9-point grid anchors the regression at the
    canonical reference positions (exact, quality-gated labels), then a
    pursuit sweep densifies coverage between the anchors. Both phases feed
    the same sample store / Ridge model - the strategy is literally the
    concatenation of two Segments, which is the point of Plan-as-Data.
    """

    name = "hybrid_9pt_pursuit"
    use_head_pose = False

    PATH_RESOLUTION = 480

    def __init__(
        self,
        alpha: float = 1e-3,
        degree: int = 2,
        duration_s: float = 24.0,
        margin: float = 0.1,
        use_head_pose: bool = False,
    ):
        self.duration_s = duration_s
        self._grid = make_grid(3, margin)
        self.use_head_pose = use_head_pose
        self._init_model(alpha, degree)

    def target_sequence(self) -> List[NormalizedPoint]:
        # Anchors + nominal pursuit waypoints (used for grid_size logging).
        return list(self._grid) + make_lissajous(60)

    def plan(self) -> List[Segment]:
        return [
            Segment("static", list(self._grid)),
            Segment("pursuit", make_lissajous(self.PATH_RESOLUTION), self.duration_s),
        ]


def _set_name(strategy, name: str):
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
    "Smooth Pursuit (24 s)": lambda: SmoothPursuitCalibration(),
    "Hybrid 9-pt + Pursuit": lambda: HybridCalibration(),
    "Ridge 9-pt + pose": lambda: _set_name(
        PolynomialRidgeCalibration(grid=make_grid(3), use_head_pose=True), "ridge_poly2_9pt_pose"
    ),
    "Hybrid + pose": lambda: _set_name(HybridCalibration(use_head_pose=True), "hybrid_9pt_pursuit_pose"),
}