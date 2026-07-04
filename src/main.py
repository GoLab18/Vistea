import sys
import time
from enum import Enum, auto
from typing import List, Optional

import cv2
import numpy as np
from PySide6.QtCore import Qt, QThread, Signal, Slot, QPoint, QRect
from PySide6.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QKeyEvent, QFont
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from core import research_log as rlog
from core.vision import GazeEstimator, LandmarksData
from core.filters import OneEuroFilter, PointFilter
from core.calibration import (
    DEFAULT_STRATEGY_KEY,
    STRATEGY_REGISTRY,
    CalibrationStrategy,
    NormalizedPoint,
)
from core.evaluation import ValidationReport, compute_point_error

# Number of valid (non-blink) frames captured per calibration target.
# At ~30 FPS this is roughly one second of fixation per point.
SAMPLES_PER_POINT = 25

# Validation targets intentionally lie OFF the training grid, so the report
# measures generalization instead of memorization.
VALIDATION_TARGETS = [
    NormalizedPoint(0.3, 0.3),
    NormalizedPoint(0.7, 0.3),
    NormalizedPoint(0.5, 0.5),
    NormalizedPoint(0.3, 0.7),
    NormalizedPoint(0.7, 0.7),
]


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


class Phase(Enum):
    CALIBRATION = auto()
    VALIDATION = auto()


class CalibrationOverlay(QWidget):
    """
    Frameless full-screen calibration surface.

    Targets are rendered at their true on-screen positions, so the eyeball
    sweeps its full physiological range - this is what gives the regression
    a usable signal (targets confined to a small video widget produce
    sub-noise iris displacements). For every target the user fixates the
    dot and presses SPACE; the overlay then records SAMPLES_PER_POINT valid
    frames (blinks are rejected upstream) while drawing a progress ring.
    A validation phase with off-grid targets follows automatically and
    produces a quantitative accuracy report.
    """

    finished = Signal(object, object)   # (strategy, ValidationReport | None)
    aborted = Signal()

    def __init__(self, strategy: CalibrationStrategy, screen):
        super().__init__()
        self.strategy = strategy
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self.setStyleSheet("background-color: #101014;")
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        self._screen = screen
        self.phase = Phase.CALIBRATION
        self.targets: List[NormalizedPoint] = strategy.target_sequence()
        self.current_idx = 0
        self.collecting = False
        self.batch: List[List[float]] = []
        self.validation_predictions: List[List[float]] = []
        self.report: Optional[ValidationReport] = None

    def start(self):
        self.setGeometry(self._screen.geometry())
        self.showFullScreen()
        self.setFocus()

    @Slot(np.ndarray, object)
    def on_frame(self, _frame: np.ndarray, data: LandmarksData):
        if not self.collecting or not data.gaze_valid or data.features is None:
            return
        self.batch.append(data.features.as_vector())
        if len(self.batch) >= SAMPLES_PER_POINT:
            self._commit_point()
        self.update()

    def _commit_point(self):
        self.collecting = False
        target = self._current_target()

        if self.phase is Phase.CALIBRATION:
            rlog.log_calibration_point(target, len(self.batch), features_batch=self.batch)
            self.strategy.add_samples(target, self.batch)
        else:
            self.validation_predictions.append(self._median_prediction(self.batch))

        QApplication.beep()
        self.batch = []
        self.current_idx += 1

        if self.current_idx >= len(self._phase_targets()):
            self._advance_phase()
        self.update()

    def _median_prediction(self, batch: List[List[float]]) -> List[float]:
        preds = [self.strategy.predict(vec) for vec in batch]
        xs = sorted(p.x for p in preds if p is not None)
        ys = sorted(p.y for p in preds if p is not None)
        return [xs[len(xs) // 2], ys[len(ys) // 2]]

    def _advance_phase(self):
        if self.phase is Phase.CALIBRATION:
            self.strategy.train()
            total_samples = len(self.strategy.target_sequence()) * SAMPLES_PER_POINT
            rlog.event(
                "calibration_trained",
                n_points=len(self.strategy.target_sequence()),
                n_samples=total_samples
            )
            
            self.phase = Phase.VALIDATION
            self.current_idx = 0
        else:
            self._build_report()
            rlog.event(
                "validation_report",
                mean_error_px=self.report.mean_error_px,
                mean_error_deg=self.report.mean_error_deg,
                mean_error_percent=self.report.mean_error_percent,
                n_points=len(self.report.points)
            )
            
            self.finished.emit(self.strategy, self.report)
            self.close()

    def _build_report(self):
        size_mm = self._screen.physicalSize()
        geo = self._screen.geometry()
        report = ValidationReport(
            strategy_name=self.strategy.name,
            screen_w_px=geo.width(),
            screen_h_px=geo.height(),
        )
        for target, pred in zip(VALIDATION_TARGETS, self.validation_predictions):
            pe = compute_point_error(
                target,
                NormalizedPoint(pred[0], pred[1]),
                geo.width(),
                geo.height(),
                size_mm.width() or None,
                size_mm.height() or None,
            )
            rlog.log_validation_point(pe.target, pe.error_px, pe.error_deg)
            report.points.append(pe)
        self.report = report

    def _phase_targets(self) -> List[NormalizedPoint]:
        return self.targets if self.phase is Phase.CALIBRATION else VALIDATION_TARGETS

    def _current_target(self) -> NormalizedPoint:
        return self._phase_targets()[self.current_idx]

    def keyPressEvent(self, event: QKeyEvent):
        if event.isAutoRepeat():
            return
        if event.key() == Qt.Key.Key_Space and not self.collecting:
            self.collecting = True
            self.batch = []
            self.update()
        elif event.key() == Qt.Key.Key_Escape:
            self.aborted.emit()
            self.close()

    def paintEvent(self, _event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()

        target = self._current_target()
        cx, cy = int(target.x * w), int(target.y * h)

        color = QColor(52, 152, 219) if self.phase is Phase.CALIBRATION else QColor(155, 89, 182)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(color)
        painter.drawEllipse(QPoint(cx, cy), 16, 16)
        painter.setBrush(QColor(255, 255, 255))
        painter.drawEllipse(QPoint(cx, cy), 4, 4)

        if self.collecting:
            pen = QPen(QColor(46, 204, 113))
            pen.setWidth(4)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            span = int(360 * 16 * len(self.batch) / SAMPLES_PER_POINT)
            painter.drawArc(QRect(cx - 26, cy - 26, 52, 52), 90 * 16, -span)

        painter.setPen(QColor(220, 220, 220))
        painter.setFont(QFont("Arial", 14))
        phase_name = "CALIBRATION" if self.phase is Phase.CALIBRATION else "VALIDATION"
        total = len(self._phase_targets())
        msg = (
            f"{phase_name}  -  point {self.current_idx + 1}/{total}   |   "
            "Look at the dot and press SPACE. Keep looking until the ring completes.   |   ESC to abort"
        )
        painter.drawText(QRect(0, h - 60, w, 40), Qt.AlignmentFlag.AlignCenter, msg)
        painter.end()


class MainWindow(QMainWindow):
    def __init__(self, screen):
        super().__init__()
        self.setWindowTitle("Vistea")
        self.resize(1024, 800)
        self._screen = screen

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        self.video_label = QLabel("Loading camera...")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(self.video_label, stretch=1)

        self.status_label = QLabel("Choose a strategy and start calibration.")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setStyleSheet("font-size: 16px; font-weight: bold; padding: 8px;")
        main_layout.addWidget(self.status_label)

        controls = QHBoxLayout()
        self.strategy_box = QComboBox()
        self.strategy_box.addItems(STRATEGY_REGISTRY.keys())
        self.strategy_box.setCurrentText(DEFAULT_STRATEGY_KEY)
        self.strategy_box.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        controls.addWidget(self.strategy_box, stretch=1)

        self.btn_calibrate = QPushButton("Start Calibration")
        self.btn_calibrate.setStyleSheet(
            "background-color: #2980b9; color: white; font-weight: bold; padding: 10px;"
        )
        self.btn_calibrate.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.btn_calibrate.clicked.connect(self.start_calibration)
        controls.addWidget(self.btn_calibrate)

        self.btn_reset = QPushButton("Reset")
        self.btn_reset.setStyleSheet(
            "background-color: #e74c3c; color: white; font-weight: bold; padding: 10px;"
        )
        self.btn_reset.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.btn_reset.clicked.connect(self.reset_calibration)
        controls.addWidget(self.btn_reset)
        main_layout.addLayout(controls)

        self.strategy: Optional[CalibrationStrategy] = None
        self.overlay: Optional[CalibrationOverlay] = None

        # Output smoothing (screen-space): tuned per Casiez et al. procedure.
        self.gaze_filter: PointFilter = OneEuroFilter(min_cutoff=0.8, beta=0.5)
        self.latest_gaze: Optional[NormalizedPoint] = None

        self.camera_thread = CameraThread()
        self.camera_thread.frame_ready.connect(self.update_ui)
        self.camera_thread.start()

    def start_calibration(self):
        strategy_factory = STRATEGY_REGISTRY[self.strategy_box.currentText()]
        self.strategy = strategy_factory()
        self.gaze_filter.reset()
        self.latest_gaze = None

        rlog.bind(
            session_id=rlog.new_session_id(),
            strategy=self.strategy.name,
            grid_size=len(self.strategy.target_sequence()),
            lighting_condition="unknown",
        )

        self.overlay = CalibrationOverlay(self.strategy, self._screen)
        self.camera_thread.frame_ready.connect(self.overlay.on_frame)
        self.overlay.finished.connect(self.on_calibration_finished)
        self.overlay.aborted.connect(self.on_calibration_aborted)
        self.overlay.start()

    @Slot(object, object)
    def on_calibration_finished(self, strategy: CalibrationStrategy, report: ValidationReport):
        self._detach_overlay()
        summary = report.summary()
        print(summary)
        for pt in report.points:
            deg = f", {pt.error_deg:.2f} deg" if pt.error_deg is not None else ""
            print(f"  target ({pt.target.x:.1f}, {pt.target.y:.1f}) -> error {pt.error_px:.0f} px{deg}")
        self.status_label.setText(summary)
        self.status_label.setStyleSheet(
            "font-size: 15px; font-weight: bold; color: #2ecc71; padding: 8px;"
        )

    @Slot()
    def on_calibration_aborted(self):
        self._detach_overlay()
        self.strategy = None
        self.status_label.setText("Calibration aborted.")

    def _detach_overlay(self):
        if self.overlay is not None:
            self.camera_thread.frame_ready.disconnect(self.overlay.on_frame)
            self.overlay = None

    def reset_calibration(self):
        self.strategy = None
        self.latest_gaze = None
        self.gaze_filter.reset()
        self.status_label.setText("Calibration cleared. Choose a strategy and start calibration.")
        self.status_label.setStyleSheet("font-size: 16px; font-weight: bold; padding: 8px;")

    @Slot(np.ndarray, object)
    def update_ui(self, frame: np.ndarray, data: LandmarksData):
        if self.strategy is not None and self.strategy.is_calibrated() and data.gaze_valid:
            raw = self.strategy.predict(data.features.as_vector())
            if raw is not None:
                fx, fy = self.gaze_filter.filter((raw.x, raw.y), data.timestamp_ms)
                self.latest_gaze = NormalizedPoint(fx, fy)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame_rgb.shape
        pixmap = QPixmap.fromImage(QImage(frame_rgb.data, w, h, ch * w, QImage.Format.Format_RGB888))
        scaled = pixmap.scaled(
            self.video_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )

        if self.latest_gaze is not None and self.strategy is not None and self.strategy.is_calibrated():
            painter = QPainter(scaled)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            pen = QPen(QColor(255, 0, 0))
            pen.setWidth(4)
            painter.setPen(pen)
            gx = int(self.latest_gaze.x * scaled.width())
            gy = int(self.latest_gaze.y * scaled.height())
            radius = 12
            painter.drawEllipse(QPoint(gx, gy), radius, radius)
            painter.drawLine(gx - radius - 5, gy, gx + radius + 5, gy)
            painter.drawLine(gx, gy - radius - 5, gx, gy + radius + 5)
            painter.end()

        self.video_label.setPixmap(scaled)

    def closeEvent(self, event):
        self.camera_thread.stop()
        super().closeEvent(event)


def main():
    rlog.configure()
    app = QApplication(sys.argv)
    screen = app.primaryScreen()
    window = MainWindow(screen)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()