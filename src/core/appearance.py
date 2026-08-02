"""
Appearance-based gaze estimation: CNN gaze angles from the face crop.

This calibration feature family regresses two gaze angles (pitch and yaw) directly
from a cropped face image using a pretrained convolutional network (L2CS-Net).
The network runs efficiently in a seperate background daemon thread.

Torch and torchvision are optional dependencies (part of the `cnn` dependency group).
Failures (missing libraries, missing weights) gracefully degrade to `None` with log notices.
"""

from __future__ import annotations

import importlib.util
import logging
import math
import threading
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Final, Literal, Protocol

import numpy as np

from core import research_log as rlog
from core.vision import LandmarksData, pose_terms

if TYPE_CHECKING:
    import torch
    import torch.nn

type FaceBoundingBox = tuple[int, int, int, int]  # (x0, y0, x1, y1) in pixels
type AnglesRadian = tuple[float, float]           # (pitch, yaw)
type WorkerState = Literal["idle", "loading", "ready", "failed"]

DEFAULT_WEIGHTS_PATH: Final[Path] = Path("models") / "L2CSNet_gaze360.pkl"

# L2CS classification-head geometry
N_BINS: Final[int] = 90
BIN_WIDTH_DEG: Final[float] = 4.0
OFFSET_DEG: Final[float] = -180.0

# ImageNet normalization used by the ResNet-50 backbone
_IMAGENET_MEAN: Final[tuple[float, float, float]] = (0.485, 0.456, 0.406)
_IMAGENET_STD: Final[tuple[float, float, float]] = (0.229, 0.224, 0.225)

# Network input resolution. L2CS-Net is a ResNet-50 whose published training
# and inference pipeline uses 224 px crops.
_INPUT_SIZE: Final[int] = 224

# How often an inference is reported to the research log
_STATS_EVERY: Final[int] = 30


def decode_angle(
    bin_logits: np.ndarray | list,
    n_bins: int = N_BINS,
    bin_width_deg: float = BIN_WIDTH_DEG,
    offset_deg: float = OFFSET_DEG,
) -> float:
    """
    Decode one L2CS classification head into a continuous angle in radians.
    A softmax turns logits into a probability distribution, the expected bin index is the
    probability-weighted sum of indices, yielding the continuous center angle.
    """

    logits = np.asarray(bin_logits, dtype=float).reshape(-1)

    # Numerically stable softmax
    shifted = logits - np.max(logits)
    exp = np.exp(shifted)
    probs = exp / np.sum(exp)
    indices = np.arange(n_bins, dtype=float)

    expectation = float(np.sum(probs * indices))
    return math.radians(expectation * bin_width_deg + offset_deg)


def bin_confidence(bin_logits: np.ndarray | list) -> float:
    """
    Peak probability of a classification head (0.0 - 1.0).
    Tells how sharply the network commits to one angular bin.
    """

    logits = np.asarray(bin_logits, dtype=float).reshape(-1)
    shifted = logits - np.max(logits)
    exp = np.exp(shifted)
    return float(np.max(exp / np.sum(exp)))


def available(weights_path: Path = DEFAULT_WEIGHTS_PATH) -> bool:
    """Whether appearance-based estimation can run."""

    if importlib.util.find_spec("torch") is None:
        return False
    if importlib.util.find_spec("torchvision") is None:
        return False

    return Path(weights_path).is_file()


class L2CSEstimator:
    """Pretrained L2CS-Net (ResNet-50 / Gaze360) gaze-angle estimator."""

    def __init__(self, weights_path: Path = DEFAULT_WEIGHTS_PATH) -> None:
        self._weights_path = Path(weights_path)

        self._model: torch.nn.Module | None = None
        self._torch: ModuleType | None = None
        self._init_attempted: bool = False
        self._init_failed: bool = False
        self._warned: bool = False
        self.last_confidence: float = 0.0

    def _warn_once(self, reason: str) -> None:
        if not self._warned:
            rlog.event("l2cs_unavailable", reason=reason, level=logging.WARNING)
            self._warned = True

    def _ensure_model(self) -> bool:
        """Build the network on first use. Returns True when usable."""

        if self._init_attempted:
            return not self._init_failed
        self._init_attempted = True

        try:
            import torch
            from torchvision.models import resnet50
        except Exception as exc:
            self._init_failed = True
            self._warn_once(f"torch/torchvision unavailable ({exc}): install with `uv sync --group cnn`")
            return False

        if not self._weights_path.is_file():
            self._init_failed = True
            self._warn_once(f"weights not found at {self._weights_path}: see models/README_L2CS.md")
            return False

        try:
            model = resnet50(weights=None)
            in_features = model.fc.in_features
            model.fc = torch.nn.Identity()
            model.fc_yaw_gaze = torch.nn.Linear(in_features, N_BINS)
            model.fc_pitch_gaze = torch.nn.Linear(in_features, N_BINS)

            state = torch.load(self._weights_path, map_location="cpu", weights_only=False)
            if hasattr(state, "state_dict"):
                state = state.state_dict()

            incompatible = model.load_state_dict(state, strict=False)
            heads_missing = [k for k in incompatible.missing_keys if "gaze" in k]
            if heads_missing:
                self._init_failed = True
                self._warn_once(f"checkpoint lacks the gaze heads ({heads_missing})")
                return False

            model.eval()
        except Exception as exc:
            self._init_failed = True
            self._warn_once(f"could not load checkpoint ({exc})")
            return False

        self._torch = torch
        self._model = model
        return True

    def angles(
        self, frame_bgr: np.ndarray, face_bbox: FaceBoundingBox
    ) -> AnglesRadian | None:
        """Estimate (pitch, yaw) gaze angles in radians from a face crop."""

        if not self._ensure_model():
            return None

        try:
            import cv2

            x0, y0, x1, y1 = face_bbox
            if x1 <= x0 or y1 <= y0:
                return None
            crop = frame_bgr[y0:y1, x0:x1]
            if crop.size == 0:
                return None

            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            rgb = cv2.resize(rgb, (_INPUT_SIZE, _INPUT_SIZE), interpolation=cv2.INTER_LINEAR)

            arr = rgb.astype(np.float32) / 255.0
            arr = (arr - np.array(_IMAGENET_MEAN, dtype=np.float32)) / np.array(_IMAGENET_STD, dtype=np.float32)
            chw = np.transpose(arr, (2, 0, 1))

            assert self._torch is not None and self._model is not None
            tensor = self._torch.from_numpy(chw).unsqueeze(0)

            with self._torch.inference_mode():
                feat = self._model(tensor)
                pitch_logits = self._model.fc_pitch_gaze(feat)[0].cpu().numpy()
                yaw_logits = self._model.fc_yaw_gaze(feat)[0].cpu().numpy()

            # Kept for telemetry: how sharply the network committed to a bin is
            # the only quality signal available without ground truth.
            self.last_confidence = (
                bin_confidence(pitch_logits) + bin_confidence(yaw_logits)
            ) / 2.0

            pitch = decode_angle(pitch_logits)
            yaw = decode_angle(yaw_logits)
            return (pitch, yaw)
        except Exception as exc:
            self._warn_once(f"inference failed ({exc})")
            return None


class L2CSConsumer(Protocol):
    """Turns the worker's latest inference into the value a caller should use.

    The network produces a few results per second while the camera delivers
    tens of frames per second, so the two call sites need opposite policies and
    each owns its own consumer.
    """

    def take(self, angles: AnglesRadian | None, seq: int, t: float) -> AnglesRadian | None:
        """Return the angles to use for this frame, or None to skip it."""
        ...

    def reset(self) -> None: ...


class FreshSampleConsumer:
    """Calibration policy: hand out every inference exactly once."""

    def __init__(self) -> None:
        self._last_seq = -1

    def take(self, angles: AnglesRadian | None, seq: int, t: float) -> AnglesRadian | None:
        if angles is None or seq == self._last_seq:
            return None
        self._last_seq = seq
        return angles

    def reset(self) -> None:
        self._last_seq = -1


class SmoothedStreamConsumer:
    """
    Interaction policy: produce a value on every frame, no matter
    the inference rate, easing toward the newest inference.
    """

    # Time constant (tau) as a fraction of the measured inference interval.
    # At 1/3, the cursor covers ~95% of the distance (3*tau) just as the next result arrives.
    INTERVAL_FRACTION: Final[float] = 0.33

    MIN_TAU_S: Final[float] = 0.03  # Absolute floor (~1 frame at 30 FPS)
    MAX_TAU_S: Final[float] = 0.30  # Usability ceiling

    def __init__(self, time_constant_s: float | None = None) -> None:
        self.time_constant_s = time_constant_s  # None: derive from the stream
        self._value: AnglesRadian | None = None
        self._last_t: float | None = None
        self._last_seq: int | None = None
        self._last_new_t: float | None = None
        self._interval_s: float | None = None

    def _observe_interval(self, seq: int, t: float) -> None:
        """Track how far apart successive inferences arrive."""

        if seq == self._last_seq:
            return
        if self._last_new_t is not None:
            gap = t - self._last_new_t
            if gap > 0.0:
                self._interval_s = gap if self._interval_s is None else (
                    self._interval_s + 0.3 * (gap - self._interval_s)
                )
        self._last_seq = seq
        self._last_new_t = t

    def _tau(self) -> float:
        if self.time_constant_s is not None:
            return self.time_constant_s
        if self._interval_s is None:
            return self.MIN_TAU_S
        return min(self.MAX_TAU_S, max(self.MIN_TAU_S, self._interval_s * self.INTERVAL_FRACTION))

    def take(self, angles: AnglesRadian | None, seq: int, t: float) -> AnglesRadian | None:
        if angles is None:
            return None
        self._observe_interval(seq, t)

        if self._value is None or self._last_t is None:
            self._value = angles
            self._last_t = t
            return angles

        dt = max(0.0, t - self._last_t)
        self._last_t = t
        tau = self._tau()
        if tau <= 0.0:
            self._value = angles
            return angles

        alpha = 1.0 - math.exp(-dt / tau)
        pitch, yaw = self._value
        self._value = (
            pitch + alpha * (angles[0] - pitch),
            yaw + alpha * (angles[1] - yaw),
        )
        return self._value

    def reset(self) -> None:
        self._value = None
        self._last_t = None
        self._last_seq = None
        self._last_new_t = None
        self._interval_s = None


class L2CSWorker:
    """Background driver that runs L2CS inference off the UI thread (Single-slot inbox)."""

    def __init__(self, estimator: L2CSEstimator) -> None:
        self._estimator = estimator
        self._inferences: int = 0
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)

        self._pending: tuple[np.ndarray, FaceBoundingBox] | None = None
        self._latest: AnglesRadian | None = None
        self._seq: int = 0
        self._state: WorkerState = "idle"
        self._stopping: bool = False
        self._warmup_requested: bool = False
        self._thread: threading.Thread | None = None

    def _start_thread_locked(self) -> None:
        self._state = "loading"
        self._thread = threading.Thread(target=self._run, name="l2cs-worker", daemon=True)
        self._thread.start()

    def warmup(self) -> None:
        """Build the network ahead of time without inference."""

        with self._cond:
            self._warmup_requested = True
            if self._state == "idle":
                self._start_thread_locked()
            self._cond.notify()

    def submit(self, frame_bgr: np.ndarray, face_bbox: FaceBoundingBox) -> None:
        """Hand the latest frame to the worker without blocking (overwriting previous)."""

        with self._cond:
            self._pending = (frame_bgr, face_bbox)
            if self._state == "idle":
                self._start_thread_locked()
            self._cond.notify()

    def result(self) -> tuple[AnglesRadian | None, int]:
        """Return `(latest_angles, seq)` safely under lock."""

        with self._lock:
            return (self._latest, self._seq)

    @property
    def state(self) -> WorkerState:
        with self._lock:
            return self._state

    def stop(self) -> None:
        """Signal the worker to finish and join it (bounded wait)."""

        with self._cond:
            self._stopping = True
            self._cond.notify_all()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=2.0)

    def _run(self) -> None:
        """Main execution loop for the daemon thread."""

        while True:
            do_warmup = False

            with self._cond:
                while self._pending is None and not self._stopping:
                    if self._warmup_requested:
                        break
                    self._cond.wait(timeout=0.5)

                if self._stopping:
                    return

                # Handle Warmup execution without processing a frame
                if self._pending is None and self._warmup_requested:
                    self._warmup_requested = False
                    do_warmup = True
                else:
                    # Proceed with regular processing
                    frame_bgr, face_bbox = self._pending
                    self._pending = None

            if do_warmup:
                self._estimator._ensure_model()
                with self._lock:
                    if self._estimator._init_failed:
                        self._state = "failed"
                        return
                    self._state = "ready"
                continue

            angles = self._estimator.angles(frame_bgr, face_bbox)

            if angles is not None:
                self._inferences += 1
                if self._inferences % _STATS_EVERY == 0:
                    rlog.event(
                        "l2cs_inference",
                        n=self._inferences,
                        confidence=round(self._estimator.last_confidence, 4),
                        level=logging.DEBUG,
                    )

            with self._lock:
                if angles is not None:
                    self._latest = angles
                    self._seq += 1
                    self._state = "ready"
                elif self._estimator._init_failed:
                    self._state = "failed"
                    return


def build_appearance_vector(
    worker: L2CSWorker,
    data: LandmarksData,
    consumer: L2CSConsumer | None = None,
    use_head_pose: bool = False,
) -> list[float] | None:
    """
    Assembles the regression input vector for the appearance family.

    2D (default): the CNN's [pitch, yaw] gaze angles in radians.
    5D (pose-augmented): those plus [yaw, pitch, roll] for the head.
    """

    if data.face_bbox is None or data.frame_bgr is None:
        return None

    worker.submit(data.frame_bgr, data.face_bbox)
    latest, seq = worker.result()

    angles = latest if consumer is None else consumer.take(latest, seq, data.timestamp_ms / 1000.0)
    if angles is None:
        return None

    vec = [angles[0], angles[1]]
    if not use_head_pose:
        return vec

    pose = pose_terms(data)
    return None if pose is None else vec + pose
