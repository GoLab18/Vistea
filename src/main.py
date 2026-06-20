import sys
import cv2
import numpy as np
from PySide6.QtCore import Qt, QThread, Signal, Slot
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QApplication, QLabel, QMainWindow, QVBoxLayout, QWidget

class CameraThread(QThread):
    frame_ready = Signal(np.ndarray)

    def __init__(self):
        super().__init__()
        self._is_running = True

    def run(self):
        cap = cv2.VideoCapture(0) # Otwarcie domyślnej kamery
        while self._is_running:
            ret, frame = cap.read()
            if ret:
                # Na ten moment wysyłamy czystą klatkę, 
                # tu później wepniemy MediaPipe
                self.frame_ready.emit(frame)
        cap.release()

    def stop(self):
        self._is_running = False
        self.wait()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Vistea - Test Środowiska")
        self.resize(800, 600)

        # Główny widget i layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        self.video_label = QLabel("Ładowanie kamery...")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.video_label)

        # Konfiguracja wątku kamery
        self.camera_thread = CameraThread()
        self.camera_thread.frame_ready.connect(self.update_image)
        self.camera_thread.start()

    @Slot(np.ndarray)
    def update_image(self, frame):
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

    def closeEvent(self, event):
        self.camera_thread.stop()
        super().closeEvent(event)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())