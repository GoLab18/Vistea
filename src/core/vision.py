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

from core.filters import PointFilter, OneEuroFilter

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
    def __init__(self, left_filter: Optional[PointFilter] = None, right_filter: Optional[PointFilter] = None):
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
        
        self.left_filter = left_filter if left_filter else OneEuroFilter()
        self.right_filter = right_filter if right_filter else OneEuroFilter()

    def _ensure_model_exists(self):
        if not os.path.exists(MODEL_DIR):
            os.makedirs(MODEL_DIR, exist_ok=True)

        if not os.path.exists(self.model_path):
            print(f"Missing {MODEL_NAME}. Downloading...")
            urllib.request.urlretrieve(MODEL_URL, self.model_path)
            print("Model downloaded successfully.")

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
            
            # Raw iris coordinates extraction
            raw_l_x = int(face_landmarks[473].x * w)
            raw_l_y = int(face_landmarks[473].y * h)
            
            raw_r_x = int(face_landmarks[468].x * w)
            raw_r_y = int(face_landmarks[468].y * h)
            
            smoothed_left = self.left_filter.filter((raw_l_x, raw_l_y), timestamp_ms)
            smoothed_right = self.right_filter.filter((raw_r_x, raw_r_y), timestamp_ms)
            
            # Casting back to integers for UI usage
            landmarks_data.left_iris = (int(smoothed_left[0]), int(smoothed_left[1]))
            landmarks_data.right_iris = (int(smoothed_right[0]), int(smoothed_right[1]))

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
            
            cv2.circle(annotated_image, landmarks_data.left_iris, 4, (0, 255, 255), -1)
            cv2.circle(annotated_image, landmarks_data.right_iris, 4, (0, 255, 255), -1)
                
        return annotated_image, landmarks_data