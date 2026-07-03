import os
import urllib.request
import cv2
import numpy as np
import mediapipe as mp
from dataclasses import dataclass
from typing import Final, Tuple, Optional, Any, List
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_DIR: Final[str] = "models"
MODEL_NAME: Final[str] = "face_landmarker.task"
MODEL_URL: Final[str] = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)

# MediaPipe Face Landmarker indices.
# "Right eye" refers to the user's right eye (left side of the image).
RIGHT_IRIS: Final[int] = 468
LEFT_IRIS: Final[int] = 473
RIGHT_EYE_CORNERS: Final[Tuple[int, int]] = (33, 133)   # outer, inner
LEFT_EYE_CORNERS: Final[Tuple[int, int]] = (362, 263)   # inner, outer
RIGHT_EYE_LIDS: Final[Tuple[int, int]] = (159, 145)     # upper, lower
LEFT_EYE_LIDS: Final[Tuple[int, int]] = (386, 374)      # upper, lower

# Below this eye-openness ratio (lid gap / eye width) a frame is treated
# as a blink and excluded from calibration / gaze estimation.
MIN_EYE_OPENNESS: Final[float] = 0.10


@dataclass(frozen=True)
class GazeFeatures:
    """
    Head-pose-robust gaze feature vector.

    For each eye the iris centre is expressed in a local coordinate frame
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

    def as_vector(self) -> List[float]:
        return [self.left_h, self.left_v, self.right_h, self.right_v]


@dataclass
class LandmarksData:
    face_detected: bool = False
    gaze_valid: bool = False  # False during blinks / degenerate geometry
    features: Optional[GazeFeatures] = None
    left_iris_px: Optional[Tuple[int, int]] = None
    right_iris_px: Optional[Tuple[int, int]] = None
    blendshapes: Optional[Any] = None
    timestamp_ms: int = 0


def _eye_relative(
    iris: np.ndarray, corner_a: np.ndarray, corner_b: np.ndarray
) -> Optional[Tuple[float, float]]:
    """Projects the iris onto the eye-corner frame. Returns (h, v) or None."""
    axis = corner_b - corner_a
    width = float(np.linalg.norm(axis))
    if width < 1e-6:
        return None
    u = axis / width                      # unit vector along the eye
    n = np.array([-u[1], u[0]])           # unit normal (perpendicular)
    d = iris - (corner_a + corner_b) / 2.0
    return float(np.dot(d, u) / width), float(np.dot(d, n) / width)


class GazeEstimator:
    """
    Wraps the MediaPipe Face Landmarker (Tasks API, VIDEO mode) and converts
    raw landmarks into eye-relative gaze features.

    Note: no temporal filtering happens here on purpose. Calibration needs
    raw, unlagged samples (it averages over many frames itself), and the
    interactive cursor is smoothed at the *output* of the gaze pipeline
    (see PointFilter strategies in core.filters), where the 1 Euro Filter
    parameters can be tuned in screen units.
    """

    def __init__(self, draw_debug: bool = True):
        self.model_path = os.path.join(MODEL_DIR, MODEL_NAME)
        self._ensure_model_exists()
        self.draw_debug = draw_debug

        base_options = python.BaseOptions(model_asset_path=self.model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            num_faces=1,
        )
        self.landmarker = vision.FaceLandmarker.create_from_options(options)

    def _ensure_model_exists(self):
        if not os.path.exists(MODEL_DIR):
            os.makedirs(MODEL_DIR, exist_ok=True)
        if not os.path.exists(self.model_path):
            print(f"Missing {MODEL_NAME}. Downloading...")
            urllib.request.urlretrieve(MODEL_URL, self.model_path)
            print("Model downloaded successfully.")

    def process_frame(
        self, frame: np.ndarray, timestamp_ms: int
    ) -> Tuple[np.ndarray, LandmarksData]:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        detection_result = self.landmarker.detect_for_video(mp_image, timestamp_ms)
        data = LandmarksData(timestamp_ms=timestamp_ms)
        annotated = np.copy(frame)

        if not detection_result.face_landmarks:
            return annotated, data

        data.face_detected = True
        if detection_result.face_blendshapes:
            data.blendshapes = detection_result.face_blendshapes[0]

        lm = detection_result.face_landmarks[0]
        h, w, _ = frame.shape
        aspect = w / h

        def pt(idx: int) -> np.ndarray:
            # Aspect-ratio correction: normalized landmark coords span 0-1 on
            # both axes regardless of the sensor shape, so X is multiplied by
            # the frame aspect ratio to obtain an isotropic distance metric.
            return np.array([lm[idx].x * aspect, lm[idx].y])

        left = _eye_relative(pt(LEFT_IRIS), pt(LEFT_EYE_CORNERS[0]), pt(LEFT_EYE_CORNERS[1]))
        right = _eye_relative(pt(RIGHT_IRIS), pt(RIGHT_EYE_CORNERS[0]), pt(RIGHT_EYE_CORNERS[1]))

        # Blink rejection: lid gap relative to eye width.
        def openness(lids: Tuple[int, int], corners: Tuple[int, int]) -> float:
            width = float(np.linalg.norm(pt(corners[0]) - pt(corners[1])))
            gap = float(np.linalg.norm(pt(lids[0]) - pt(lids[1])))
            return gap / width if width > 1e-6 else 0.0

        eyes_open = (
            openness(LEFT_EYE_LIDS, LEFT_EYE_CORNERS) > MIN_EYE_OPENNESS
            and openness(RIGHT_EYE_LIDS, RIGHT_EYE_CORNERS) > MIN_EYE_OPENNESS
        )

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