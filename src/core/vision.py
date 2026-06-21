import os
import urllib.request
import cv2
import numpy as np
import mediapipe as mp
from dataclasses import dataclass
from typing import Final, Tuple, Optional, Any
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import drawing_utils
from mediapipe.tasks.python.vision import drawing_styles

MODEL_DIR: Final[str] = "models"
MODEL_NAME: Final[str] = "face_landmarker.task"
MODEL_URL: Final[str] = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"

@dataclass
class LandmarksData:
    face_detected: bool = False
    left_iris: Optional[Tuple[int, int]] = None
    right_iris: Optional[Tuple[int, int]] = None
    blendshapes: Optional[Any] = None

class GazeEstimator:
    # Alpha = 0.5 -> moderate smoothing (50% new frame, 50% old frame)
    def __init__(self, alpha: float = 0.5):
        self.model_path = os.path.join(MODEL_DIR, MODEL_NAME)
        self._ensure_model_exists()

        base_options = python.BaseOptions(model_asset_path=self.model_path)
        
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            num_faces=1
        )
        
        self.landmarker = vision.FaceLandmarker.create_from_options(options)
        
        self.alpha = alpha
        self._prev_left_iris: Optional[Tuple[float, float]] = None
        self._prev_right_iris: Optional[Tuple[float, float]] = None

    def _ensure_model_exists(self):
        if not os.path.exists(MODEL_DIR):
            os.makedirs(MODEL_DIR, exist_ok=True)

        if not os.path.exists(self.model_path):
            print(f"Missing {MODEL_NAME}. Downloading...")
            urllib.request.urlretrieve(MODEL_URL, self.model_path)
            print("Model downloaded successfully.")

    def _apply_ema(self, current: Tuple[int, int], previous: Optional[Tuple[float, float]]) -> Tuple[float, float]:
        if previous is None:
            return float(current[0]), float(current[1])
        
        return (
            current[0] * self.alpha + previous[0] * (1.0 - self.alpha),
            current[1] * self.alpha + previous[1] * (1.0 - self.alpha)
        )

    def process_frame(self, frame: np.ndarray, timestamp_ms: int) -> Tuple[np.ndarray, LandmarksData]:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        detection_result = self.landmarker.detect_for_video(mp_image, timestamp_ms)

        landmarks_data = LandmarksData()
        annotated_image = np.copy(frame)

        if detection_result.face_landmarks:
            landmarks_data.face_detected = True
            
            if detection_result.face_blendshapes:
                landmarks_data.blendshapes = detection_result.face_blendshapes[0]

            face_landmarks = detection_result.face_landmarks[0]
            h, w, _ = frame.shape
            
            # Extracts raw left iris coordinates
            raw_l_x = int(face_landmarks[473].x * w)
            raw_l_y = int(face_landmarks[473].y * h)
            
            # Extracts raw right iris coordinates
            raw_r_x = int(face_landmarks[468].x * w)
            raw_r_y = int(face_landmarks[468].y * h)
            
            # Exponential Moving Average filter
            self._prev_left_iris = self._apply_ema((raw_l_x, raw_l_y), self._prev_left_iris)
            self._prev_right_iris = self._apply_ema((raw_r_x, raw_r_y), self._prev_right_iris)
            
            # Casts smoothed floats back to integers for UI usage
            landmarks_data.left_iris = (int(self._prev_left_iris[0]), int(self._prev_left_iris[1]))
            landmarks_data.right_iris = (int(self._prev_right_iris[0]), int(self._prev_right_iris[1]))

            drawing_utils.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks,
                connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=drawing_styles.get_default_face_mesh_tesselation_style()
            )
            drawing_utils.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks,
                connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=drawing_styles.get_default_face_mesh_contours_style()
            )
            drawing_utils.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks,
                connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_LEFT_IRIS,
                landmark_drawing_spec=None,
                connection_drawing_spec=drawing_styles.get_default_face_mesh_iris_connections_style()
            )
            drawing_utils.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks,
                connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_IRIS,
                landmark_drawing_spec=None,
                connection_drawing_spec=drawing_styles.get_default_face_mesh_iris_connections_style()
            )
            
            # Draws solid yellow circles over the filtered coordinates to visualize stability
            cv2.circle(annotated_image, landmarks_data.left_iris, 4, (0, 255, 255), -1)
            cv2.circle(annotated_image, landmarks_data.right_iris, 4, (0, 255, 255), -1)
                
        return annotated_image, landmarks_data