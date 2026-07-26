"""
Appearance-based gaze estimation: CNN gaze angles from the face crop.

This calibration feature family regresses two gaze angles (pitch and yaw) directly
from a cropped face image using a pretrained convolutional network (L2CS-Net).
The network runs efficiently in a seperate background daemon thread.

Torch and torchvision are optional dependencies (part of the ``cnn`` dependency group).
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

# Network input resolution (square face crop)
_INPUT_SIZE: Final[int] = 448

# Face-crop margin
_CROP_MARGIN: Final[float] = 0.2


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
    angle_deg = expectation * bin_width_deg + offset_deg + bin_width_deg / 2.0
    return math.radians(angle_deg)


def available(weights_path: Path = DEFAULT_WEIGHTS_PATH) -> bool:
    """
    Whether appearance-based estimation can run.
    """

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

            model.load_state_dict(state, strict=False)
            model.eval()
        except Exception as exc:
            self._init_failed = True
            self._warn_once(f"could not load checkpoint ({exc})")
            return False

        self._torch = torch
        self._model = model
        return True

    @staticmethod
    def _crop_face(frame_bgr: np.ndarray, face_bbox: FaceBoundingBox) -> np.ndarray | None:
        """Crop the face with a margin."""

        h, w = frame_bgr.shape[:2]
        x0, y0, x1, y1 = face_bbox
        bw, bh = x1 - x0, y1 - y0
        if bw <= 0 or bh <= 0:
            return None

        mx, my = int(bw * _CROP_MARGIN), int(bh * _CROP_MARGIN)
        x0 = max(0, x0 - mx)
        y0 = max(0, y0 - my)
        x1 = min(w, x1 + mx)
        y1 = min(h, y1 + my)

        if x1 <= x0 or y1 <= y0:
            return None
        return frame_bgr[y0:y1, x0:x1]

    def angles(
        self, frame_bgr: np.ndarray, face_bbox: FaceBoundingBox
    ) -> AnglesRadian | None:
        """Estimate (pitch, yaw) gaze angles in radians from a face crop."""

        if not self._ensure_model():
            return None

        try:
            import cv2

            crop = self._crop_face(frame_bgr, face_bbox)
            if crop is None or crop.size == 0:
                return None

            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            rgb = cv2.resize(rgb, (_INPUT_SIZE, _INPUT_SIZE), interpolation=cv2.INTER_LINEAR)

            arr = rgb.astype(np.float32) / 255.0
            arr = (arr - np.array(_IMAGENET_MEAN, dtype=np.float32)) / np.array(_IMAGENET_STD, dtype=np.float32)
            chw = np.transpose(arr, (2, 0, 1))

            assert self._torch is not None and self._model is not None
            tensor = self._torch.from_numpy(chw).unsqueeze(0)

            with self._torch.no_grad():
                feat = self._model(tensor)
                pitch_logits = self._model.fc_pitch_gaze(feat)[0].cpu().numpy()
                yaw_logits = self._model.fc_yaw_gaze(feat)[0].cpu().numpy()

            pitch = decode_angle(pitch_logits)
            yaw = decode_angle(yaw_logits)
            return (pitch, yaw)
        except Exception as exc:
            self._warn_once(f"inference failed ({exc})")
            return None


class L2CSConsumer(Protocol):
    _l2cs_last_seq: int


class L2CSWorker:
    """Background driver that runs L2CS inference off the UI thread (Single-slot inbox)."""

    def __init__(self, estimator: L2CSEstimator) -> None:
        self._estimator = estimator
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

            with self._lock:
                if angles is not None:
                    self._latest = angles
                    self._seq += 1
                    self._state = "ready"
                elif self._estimator._init_failed:
                    self._state = "failed"
                    return