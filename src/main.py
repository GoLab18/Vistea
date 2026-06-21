import sys
import cv2
import time
import numpy as np
from PySide6.QtCore import Qt, QThread, Signal, Slot
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QApplication, QLabel, QMainWindow, QVBoxLayout, QWidget

from core.vision import GazeEstimator, LandmarksData

class CameraThread(QThread):
    frame_ready = Signal(np.ndarray, LandmarksData)

    def __init__(self):
        super().__init__()
        self._is_running = True
        self.estimator = GazeEstimator()
        self.start_time_ns = time.time_ns()

    def run(self):
        cap = cv2.VideoCapture(0)
        while self._is_running:
            ret, frame = cap.read()
            if ret:
                timestamp_ms = int((time.time_ns() - self.start_time_ns) / 1_000_000)
                
                processed_frame, landmarks_data = self.estimator.process_frame(frame, timestamp_ms)
                
                self.frame_ready.emit(processed_frame, landmarks_data)
        cap.release()

    def stop(self):
        self._is_running = False
        self.wait()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Vistea")
        self.resize(1024, 768)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        self.video_label = QLabel("Loading camera...")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.video_label, stretch=1)
        
        self.status_label = QLabel("Starting detection...")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setStyleSheet("font-size: 18px; font-weight: bold; padding: 10px;")
        layout.addWidget(self.status_label)

        self.camera_thread = CameraThread()
        self.camera_thread.frame_ready.connect(self.update_ui)
        self.camera_thread.start()

    @Slot(np.ndarray, object)
    def update_ui(self, frame: np.ndarray, landmarks_data: LandmarksData):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame_rgb.shape
        bytes_per_line = ch * w
        
        qt_image = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        
        scaled_pixmap = pixmap.scaled(
            self.video_label.size(), 
            Qt.AspectRatioMode.KeepAspectRatio, 
            Qt.TransformationMode.SmoothTransformation
        )
        self.video_label.setPixmap(scaled_pixmap)

        if landmarks_data.face_detected:
            lx, ly = landmarks_data.left_iris
            rx, ry = landmarks_data.right_iris
            
            self.status_label.setText(
                f"Face Detected | Left Eye (X:{lx}, Y:{ly}) | Right Eye (X:{rx}, Y:{ry})"
            )
            self.status_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #2ecc71; padding: 10px;")
        else:
            self.status_label.setText("Searching for face...")
            self.status_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #e74c3c; padding: 10px;")

    def closeEvent(self, event):
        self.camera_thread.stop()
        super().closeEvent(event)

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()