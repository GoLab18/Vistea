"""
Computer Vision and Feature Extraction Pipeline.

This module processes raw webcam frames using MediaPipe Face Landmarker to extract
gaze-estimation features. It implements the PC-EC (Pupil Center - Eye Corner) 
feature vector (Sesma et al., 2012), which isolates eyeball rotation from head 
movement by anchoring the iris center to the stable eye corners.

Key capabilities:
- Scale, translation, and roll invariance via intra-orbital Euclidean distances.
- Yaw, pitch and roll compensation via a 7D pose-augmented feature vector.
- Blink rejection using the EAR (Eye Aspect Ratio) metric (Soukupová & Čech, 2016).
"""

from __future__ import annotations

import logging
import math
import os
import urllib.request
from dataclasses import dataclass
from typing import Final

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.components.containers.category import Category
from mediapipe.tasks.python.components.containers.landmark import NormalizedLandmark

from core.preprocessing import FramePreprocessor, NoPreprocessing

logger = logging.getLogger(__name__)

type PixelCoord = tuple[int, int]
type BoundingBox = tuple[int, int, int, int]
type FaceBlendshapes = list[Category]
type FaceLandmarks = list[NormalizedLandmark]

MODEL_DIR: Final[str] = "models"
MODEL_NAME: Final[str] = "face_landmarker.task"
MODEL_URL: Final[str] = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)

# MediaPipe Face Landmarker indices.
RIGHT_IRIS: Final[int] = 468
LEFT_IRIS: Final[int] = 473
RIGHT_EYE_CORNERS: Final[tuple[int, int]] = (33, 133)   # outer, inner
LEFT_EYE_CORNERS: Final[tuple[int, int]] = (362, 263)   # inner, outer
RIGHT_EYE_LIDS: Final[tuple[int, int]] = (159, 145)     # upper, lower
LEFT_EYE_LIDS: Final[tuple[int, int]] = (386, 374)      # upper, lower

# EAR (Eye Aspect Ratio) threshold. Below this eye-openness ratio (lid gap / eye width),
# a frame is treated as a blink and excluded from calibration / gaze estimation.
MIN_EYE_OPENNESS: Final[float] = 0.10

# Fraction of the user's own resting openness below which the lid is a candidate
# for being mid-motion (closing or reopening).
LID_STABLE_FRACTION: Final[float] = 0.80

# Openness change per second that separates a travelling lid from an eye that
# is just held narrower.
LID_MOTION_RATE: Final[float] = 1.2

# Longest the gate may keep withholding frames.
LID_MAX_HOLD_S: Final[float] = 0.5


@dataclass(frozen=True)
class GazeFeatures:
    """
    Head-pose-robust gaze feature vector.

    For each eye the iris center is expressed in a local coordinate frame
    spanned by the eye corners:
      * h - signed offset along the inter-corner axis, divided by eye width
      * v - signed offset perpendicular to that axis, divided by eye width

    Dividing by the inter-corner distance makes the features invariant to
    the distance from the camera (scale), while projecting onto the corner
    axis makes them invariant to in-plane head rotation (roll) and to any
    translation of the head. This isolates eyeball rotation from head
    movement far better than normalising against the face bounding box,
    whose size also fluctuates with facial expressions.
    """

    left_h: float
    left_v: float
    right_h: float
    right_v: float

    def as_vector(self) -> list[float]:
        return [self.left_h, self.left_v, self.right_h, self.right_v]


@dataclass(frozen=True)
class HeadPose:
    """
    Head orientation in radians, decomposed from MediaPipe's 4x4
    facial transformation matrix (canonical face -> camera space).
 
    Sign conventions follow the camera frame and are internally consistent,
    both consumers only ever use them consistently: the regression learns
    whatever convention it is given, and the virtual chinrest compares
    against a reference captured in the same convention.
    """

    yaw: float    # rotation about the vertical axis (looking left/right)
    pitch: float  # rotation about the horizontal axis (looking up/down)
    roll: float   # in-plane rotation (head tilt)

    def as_degrees(self) -> tuple[float, float, float]:
        return (math.degrees(self.yaw), math.degrees(self.pitch), math.degrees(self.roll))


def decompose_head_pose(transform_matrix: np.ndarray) -> HeadPose | None:
    """Extracts Tait-Bryan angles from the facial transformation matrix."""

    m = np.asarray(transform_matrix, dtype=float)
    if m.shape != (4, 4):
        return None
    r = m[:3, :3]

    # Guard against a non-rotation block
    if not np.all(np.isfinite(r)):
        return None

    sy = math.sqrt(r[0, 0] ** 2 + r[1, 0] ** 2)
    if sy > 1e-6:
        pitch = math.atan2(r[2, 1], r[2, 2])
        yaw = math.atan2(-r[2, 0], sy)
        roll = math.atan2(r[1, 0], r[0, 0])
    else:
        # gimbal lock: yaw = +-90 deg, roll and pitch are coupled
        pitch = math.atan2(-r[1, 2], r[1, 1])
        yaw = math.atan2(-r[2, 0], sy)
        roll = 0.0

    return HeadPose(yaw=yaw, pitch=pitch, roll=roll)


class EyelidStabilityGate:
    """
    Separates a settled eyelid from one that is in motion.

    When the lid travels down, before hitting the openness threshold, it slides
    the detected iris center out of it's real center position. The baseline is
    adapted with time through `adapt_rate`.
    """

    def __init__(
        self,
        stable_fraction: float = LID_STABLE_FRACTION,
        adapt_rate: float = 0.02,
        motion_rate: float = LID_MOTION_RATE,
        max_hold_s: float = LID_MAX_HOLD_S,
        closed_openness: float = MIN_EYE_OPENNESS,
    ) -> None:
        self.stable_fraction = stable_fraction
        self.adapt_rate = adapt_rate
        self.motion_rate = motion_rate
        self.max_hold_s = max_hold_s
        self.closed_openness = closed_openness

        self._baseline: float | None = None
        self._last: tuple[float, float] | None = None  # (t, openness)
        self._unstable_since: float | None = None

    @property
    def baseline(self) -> float | None:
        return self._baseline

    def reset(self) -> None:
        self._baseline = None
        self._last = None
        self._unstable_since = None

    def _rate(self, openness: float, t: float) -> float | None:
        """Absolute openness change per second since the previous frame."""

        if self._last is None:
            return None
        prev_t, prev_openness = self._last
        dt = t - prev_t
        if dt <= 0.0:
            return None
        return abs(openness - prev_openness) / dt

    def _settle(self, openness: float) -> bool:
        """Accept the frame and let the baseline follow it."""

        self._unstable_since = None
        if self._baseline is not None:
            self._baseline += self.adapt_rate * (openness - self._baseline)
        return True

    def update(self, openness: float, t: float) -> bool:
        """Update current stability based on the given openness ratio and
        timestamp in seconds. Returns True when the lid is settled."""

        rate = self._rate(openness, t)
        self._last = (t, openness)

        if self._baseline is None:
            self._baseline = openness
            return True

        if openness < self.closed_openness:
            self._unstable_since = None
            return False

        if openness >= self._baseline * self.stable_fraction:
            return self._settle(openness)

        # Narrowed: might be still travelling, or being held there
        if rate is not None and rate < self.motion_rate:
            return self._settle(openness)

        if self._unstable_since is None:
            self._unstable_since = t
        elif t - self._unstable_since >= self.max_hold_s:
            self._baseline = openness
            self._unstable_since = None
            return True
        return False


@dataclass
class LandmarksData:
    """Container for a single frame's processed visual metrics."""

    face_detected: bool = False
    gaze_valid: bool = False  # False during blinks or degenerate geometry
    features: GazeFeatures | None = None
    head_pose: HeadPose | None = None

    eye_openness: float | None = None  # Mean lid gap / eye width over both eyes
    lids_stable: bool = True           # False while a lid is opening or closing

    left_iris_px: PixelCoord | None = None
    right_iris_px: PixelCoord | None = None
    face_bbox: BoundingBox | None = None # Consumed by the appearance-based (CNN) feature source

    # The preprocessed frame without the debug overlay, for consumers that read
    # pixels rather than landmarks.
    frame_bgr: np.ndarray | None = None

    blendshapes: FaceBlendshapes | None = None
    timestamp_ms: int = 0


def pose_terms(data: LandmarksData) -> list[float] | None:
    """The three head-orientation terms appended to a pose-augmented vector."""

    if data.head_pose is None:
        return None
    return [data.head_pose.yaw, data.head_pose.pitch, data.head_pose.roll]


def build_feature_vector(data: LandmarksData, use_head_pose: bool = False) -> list[float] | None:
    """
    Assembles the regression input vector from a processed frame.

    4D (default): The standard PC-EC vector [left_h, left_v, right_h, right_v].
    7D (pose-augmented): PC-EC + [yaw, pitch, roll] in radians.

    Returns None when the frame carries no valid gaze (or, in pose mode, no pose).
    """

    if not data.gaze_valid or data.features is None:
        return None

    vec = data.features.as_vector()

    if not use_head_pose:
        return vec

    pose = pose_terms(data)
    return None if pose is None else vec + pose


def _eye_relative(
    iris: np.ndarray, corner_a: np.ndarray, corner_b: np.ndarray
) -> tuple[float, float] | None:
    """
    Projects the iris center onto the local coordinate frame spanned by the eye corners.
    Returns (horizontal_projection, vertical_projection) normalized by eye width.
    """

    axis = corner_b - corner_a
    width = float(np.linalg.norm(axis))
    if width < 1e-6:
        return None
    u = axis / width
    n = np.array([-u[1], u[0]])
    d = iris - (corner_a + corner_b) / 2.0
    return float(np.dot(d, u) / width), float(np.dot(d, n) / width)


class GazeEstimator:
    """
    Wraps the MediaPipe Face Landmarker (Tasks API) and converts raw landmarks 
    into eye-relative gaze features and head pose.
    """

    def __init__(self, draw_debug: bool = True) -> None:
        self.model_path = os.path.join(MODEL_DIR, MODEL_NAME)
        self._ensure_model_exists()
        self.draw_debug = draw_debug

        self.preprocessor: FramePreprocessor = NoPreprocessing()
        self.lid_gate = EyelidStabilityGate()

        base_options = python.BaseOptions(model_asset_path=self.model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            num_faces=1,
        )
        self.landmarker = vision.FaceLandmarker.create_from_options(options)

    def _ensure_model_exists(self) -> None:
        """Downloads the MediaPipe model if it is not present locally."""

        if not os.path.exists(MODEL_DIR):
            os.makedirs(MODEL_DIR, exist_ok=True)

        if not os.path.exists(self.model_path):
            logger.info(f"Missing {MODEL_NAME}. Downloading from Google Storage...")
            try:
                urllib.request.urlretrieve(MODEL_URL, self.model_path)
                logger.info("Model downloaded successfully.")
            except Exception as e:
                logger.error(f"Failed to download the model: {e}")
                raise


    def process_frame(
        self, frame: np.ndarray, timestamp_ms: int
    ) -> tuple[np.ndarray, LandmarksData]:
        frame = self.preprocessor.process(frame)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        detection_result = self.landmarker.detect_for_video(mp_image, timestamp_ms)
        data = LandmarksData(timestamp_ms=timestamp_ms, frame_bgr=frame)
        annotated = np.copy(frame)

        if not detection_result.face_landmarks:
            return annotated, data

        data.face_detected = True

        if detection_result.face_blendshapes:
            data.blendshapes = detection_result.face_blendshapes[0]
        if detection_result.facial_transformation_matrixes:
            data.head_pose = decompose_head_pose(
                np.asarray(detection_result.facial_transformation_matrixes[0])
            )

        lm: FaceLandmarks = detection_result.face_landmarks[0]
        h, w, _ = frame.shape
        aspect = w / h

        # Pixel bounding box for appearance-based CNN cropping - landmarks
        # may sit slightly outside the image.
        xs = [p.x for p in lm]
        ys = [p.y for p in lm]
        data.face_bbox = (
            max(0, int(min(xs) * w)), max(0, int(min(ys) * h)),
            min(w, int(max(xs) * w)), min(h, int(max(ys) * h)),
        )

        def pt(idx: int) -> np.ndarray:
            return np.array([lm[idx].x * aspect, lm[idx].y])

        left = _eye_relative(pt(LEFT_IRIS), pt(LEFT_EYE_CORNERS[0]), pt(LEFT_EYE_CORNERS[1]))
        right = _eye_relative(pt(RIGHT_IRIS), pt(RIGHT_EYE_CORNERS[0]), pt(RIGHT_EYE_CORNERS[1]))

        def openness(lids: tuple[int, int], corners: tuple[int, int]) -> float:
            eye_width = float(np.linalg.norm(pt(corners[0]) - pt(corners[1])))
            lid_gap = float(np.linalg.norm(pt(lids[0]) - pt(lids[1])))
            return lid_gap / eye_width if eye_width > 1e-6 else 0.0

        left_open = openness(LEFT_EYE_LIDS, LEFT_EYE_CORNERS)
        right_open = openness(RIGHT_EYE_LIDS, RIGHT_EYE_CORNERS)
        eyes_open = left_open > MIN_EYE_OPENNESS and right_open > MIN_EYE_OPENNESS

        data.eye_openness = (left_open + right_open) / 2.0
        data.lids_stable = self.lid_gate.update(data.eye_openness, timestamp_ms / 1000.0)

        if left is not None and right is not None and eyes_open:
            data.gaze_valid = True
            data.features = GazeFeatures(left[0], left[1], right[0], right[1])

        data.left_iris_px = (int(lm[LEFT_IRIS].x * w), int(lm[LEFT_IRIS].y * h))
        data.right_iris_px = (int(lm[RIGHT_IRIS].x * w), int(lm[RIGHT_IRIS].y * h))

        if self.draw_debug:
            color = (0, 255, 255) if data.gaze_valid else (0, 0, 255)
            cv2.circle(annotated, data.left_iris_px, 4, color, -1)
            cv2.circle(annotated, data.right_iris_px, 4, color, -1)
            for idx in (*LEFT_EYE_CORNERS, *RIGHT_EYE_CORNERS):
                cv2.circle(annotated, (int(lm[idx].x * w), int(lm[idx].y * h)), 2, (255, 0, 255), -1)

        return annotated, data