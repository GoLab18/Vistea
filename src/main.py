import sys
import time
import logging
from enum import Enum, auto
from typing import List, Optional

import cv2
import numpy as np
from PySide6.QtCore import Qt, QThread, QTimer, Signal, Slot, QPoint, QRect
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
    PURSUIT_LAG_S,
    STRATEGY_REGISTRY,
    CalibrationStrategy,
    NormalizedPoint,
    Segment,
)
from core.evaluation import ValidationReport, compute_point_error

# Number of valid (non-blink) frames captured per STATIC calibration target.
# At ~30 FPS this is roughly one second of fixation per point.
SAMPLES_PER_POINT = 25

# Quality gating threshold: maximum acceptable dispersion (spread)
# for a collected batch. Above this threshold, the batch is rejected.
# Applies to static segments only - dispersion of a batch collected while
# tracking a MOVING target reflects the trajectory, not capture quality.
SPREAD_THRESHOLD = 0.006

# Validation targets intentionally lie OFF all training grids (5, 9, and 16-pt),
# so the report measures pure generalization instead of memorization.
VALIDATION_TARGETS = [
    NormalizedPoint(0.2, 0.2),
    NormalizedPoint(0.8, 0.2),
    NormalizedPoint(0.2, 0.8),
    NormalizedPoint(0.8, 0.8),
    NormalizedPoint(0.35, 0.65),
    NormalizedPoint(0.65, 0.35),
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
    Frameless full-screen calibration surface - a generic interpreter of the
    strategy's calibration plan (Plan-as-Data, see core.calibration.Segment).

    Static segments: targets are rendered at their true on-screen positions;
    for every target the user fixates the dot and presses SPACE, then
    SAMPLES_PER_POINT valid frames are recorded (blinks rejected upstream)
    behind a progress ring, with quality gating on the batch spread.

    Pursuit segments: SPACE starts a smooth sweep of the target along the
    declared trajectory; every valid frame is collected continuously and
    labelled with the trajectory position PURSUIT_LAG_S seconds BEFORE the
    capture instant (clock-based oculomotor-lag compensation, exact under
    camera FPS jitter).

    A validation phase with off-grid static targets follows automatically
    and produces a quantitative accuracy report.
    """

    finished = Signal(object, object) # (strategy, ValidationReport | None)
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
        self.plan: List[Segment] = strategy.plan()
        self.seg_idx = 0
        self.point_idx = 0 # index within a static segment / validation
        self.retries = 0
        self.collecting = False # static batch capture in progress
        self.batch: List[List[float]] = []
        self.validation_predictions: List[List[float]] = []
        self.report: Optional[ValidationReport] = None

        # Pursuit segment state
        self.pursuit_active = False
        self.pursuit_t0 = 0.0
        self.pursuit_samples = 0
        self._anim = QTimer(self)
        self._anim.setInterval(16)
        self._anim.timeout.connect(self._on_anim_tick)


    def start(self):
        self.setGeometry(self._screen.geometry())
        self.showFullScreen()
        self.setFocus()

    def _build_vec(self, data: LandmarksData) -> Optional[List[float]]:
        return data.features.as_vector()

    @Slot(np.ndarray, object)
    def on_frame(self, _frame: np.ndarray, data: LandmarksData):
        if not data.gaze_valid or data.features is None:
            return

        if self.pursuit_active:
            self._on_pursuit_frame(data)
            return

        if not self.collecting:
            return
        vec = self._build_vec(data)
        if vec is None:
            return
        self.batch.append(vec)
        if len(self.batch) >= SAMPLES_PER_POINT:
            self._commit_point()
        self.update()

    def _on_pursuit_frame(self, data: LandmarksData):
        segment = self._current_segment()
        elapsed = time.monotonic() - self.pursuit_t0
        if elapsed >= segment.duration_s:
            self._finish_pursuit(segment)
            return
        if elapsed < PURSUIT_LAG_S:
            return # the eye has not locked onto the target yet
        vec = self._build_vec(data)
        if vec is None:
            return
        label = segment.position_at(elapsed - PURSUIT_LAG_S)
        self.strategy.add_samples(label, [vec])
        self.pursuit_samples += 1

    def _on_anim_tick(self):
        if self.pursuit_active:
            segment = self._current_segment()
            if time.monotonic() - self.pursuit_t0 >= segment.duration_s:
                self._finish_pursuit(segment)
        self.update()


    def _commit_point(self):
        target = self._current_target()

        if self.phase is Phase.CALIBRATION:
            # Quality Gating: sample spread measurement before batch acceptance
            spread = rlog.sample_spread(self.batch)
            if spread is not None and spread > SPREAD_THRESHOLD:
                if self.retries < 3:
                    rlog.event(
                        "calibration_point_rejected",
                        point=[target.x, target.y],
                        spread=spread,
                        level=logging.WARNING,
                    )
                    self.retries += 1
                    self.collecting = False
                    self.batch = []
                    self.update()
                    return
                else:
                    rlog.event(
                        "calibration_point_forced",
                        point=[target.x, target.y],
                        spread=spread,
                        level=logging.WARNING,
                    )

            self.retries = 0
            rlog.log_calibration_point(target, len(self.batch), features_batch=self.batch)
            self.strategy.add_samples(target, self.batch)
        else:
            self.validation_predictions.append(self._median_prediction(self.batch))

        self.collecting = False
        QApplication.beep()
        self.batch = []
        self.point_idx += 1

        if self.point_idx >= len(self._phase_targets()):
            self._advance_segment()
        self.update()

    def _finish_pursuit(self, segment: Segment):
        self.pursuit_active = False
        self._anim.stop()
        rlog.event(
            "pursuit_segment_captured",
            n_samples=self.pursuit_samples,
            duration_s=segment.duration_s,
            lag_s=PURSUIT_LAG_S,
        )
        QApplication.beep()
        self._advance_segment()
        self.update()

    def _advance_segment(self):
        if self.phase is Phase.VALIDATION:
            self._finish_validation()
            return
        self.seg_idx += 1
        self.point_idx = 0
        if self.seg_idx >= len(self.plan):
            self._train_and_enter_validation()

    def _train_and_enter_validation(self):
        self.strategy.train()
        rlog.event(
            "calibration_trained",
            n_points=len(self.strategy.target_sequence()),
            n_samples=len(getattr(self.strategy, "_store", [])) or None,
        )


        self.phase = Phase.VALIDATION
        self.point_idx = 0

    def _finish_validation(self):
        self._build_report()
        rlog.event(
            "validation_report",
            mean_error_px=self.report.mean_error_px,
            mean_error_deg=self.report.mean_error_deg,
            mean_error_percent=self.report.mean_error_percent,
            n_points=len(self.report.points),
        )
        self.finished.emit(self.strategy, self.report)
        self.close()

    def _median_prediction(self, batch: List[List[float]]) -> List[float]:
        preds = [self.strategy.predict(vec) for vec in batch]
        xs = sorted(p.x for p in preds if p is not None)
        ys = sorted(p.y for p in preds if p is not None)
        return [xs[len(xs) // 2], ys[len(ys) // 2]]

    def _build_report(self):
        size_mm = self._screen.physicalSize()
        geo = self._screen.geometry()
        report = ValidationReport(
            strategy_name=self.strategy.name,
            screen_w_px=geo.width(),
            screen_h_px=geo.height(),
        )
        for target, pred in zip(VALIDATION_TARGETS, self.validation_predictions):
            report.points.append(
                compute_point_error(
                    target,
                    NormalizedPoint(pred[0], pred[1]),
                    geo.width(),
                    geo.height(),
                    size_mm.width() or None,
                    size_mm.height() or None,
                )
            )
            rlog.log_validation_point(target, report.points[-1].error_px, report.points[-1].error_deg)
        self.report = report

    def _current_segment(self) -> Segment:
        return self.plan[min(self.seg_idx, len(self.plan) - 1)]

    def _phase_targets(self) -> List[NormalizedPoint]:
        if self.phase is Phase.VALIDATION:
            return VALIDATION_TARGETS
        return self._current_segment().points

    def _current_target(self) -> NormalizedPoint:
        return self._phase_targets()[self.point_idx]

    def keyPressEvent(self, event: QKeyEvent):
        if event.isAutoRepeat():
            return
        if event.key() == Qt.Key.Key_Space:
            if self.phase is Phase.CALIBRATION and self._current_segment().kind == "pursuit":
                if not self.pursuit_active:
                    self.pursuit_active = True
                    self.pursuit_t0 = time.monotonic()
                    self.pursuit_samples = 0
                    self._anim.start()
                    self.update()
            elif not self.collecting:
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

        pursuit = self.phase is Phase.CALIBRATION and self._current_segment().kind == "pursuit"

        if pursuit:
            segment = self._current_segment()
            elapsed = (time.monotonic() - self.pursuit_t0) if self.pursuit_active else 0.0
            target = segment.position_at(elapsed)
        else:
            target = self._current_target()
        cx, cy = int(target.x * w), int(target.y * h)

        color = QColor(52, 152, 219) if self.phase is Phase.CALIBRATION else QColor(155, 89, 182)
        if pursuit:
            color = QColor(46, 204, 113)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(color)
        painter.drawEllipse(QPoint(cx, cy), 16, 16)
        painter.setBrush(QColor(255, 255, 255))
        painter.drawEllipse(QPoint(cx, cy), 4, 4)

        if self.collecting and not pursuit:
            pen = QPen(QColor(46, 204, 113))
            pen.setWidth(4)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            span = int(360 * 16 * len(self.batch) / SAMPLES_PER_POINT)
            painter.drawArc(QRect(cx - 26, cy - 26, 52, 52), 90 * 16, -span)

        if pursuit and self.pursuit_active:
            segment = self._current_segment()
            frac = min((time.monotonic() - self.pursuit_t0) / segment.duration_s, 1.0)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(46, 204, 113))
            painter.drawRect(0, h - 8, int(w * frac), 8)

        painter.setPen(QColor(220, 220, 220))
        painter.setFont(QFont("Arial", 14))
        if self.phase is Phase.VALIDATION:
            msg = (
                f"VALIDATION  -  point {self.point_idx + 1}/{len(VALIDATION_TARGETS)}   |   "
                "Look at the dot and press SPACE. Keep looking until the ring completes.   |   ESC to abort"
            )
        elif pursuit:
            msg = (
                "CALIBRATION (pursuit)   |   "
                + ("Follow the moving dot with your eyes." if self.pursuit_active
                 else "Press SPACE, then FOLLOW the moving dot with your eyes.")
                + "   |   ESC to abort"
            )
        else:
            total = len(self._phase_targets())
            seg_info = f"segment {self.seg_idx + 1}/{len(self.plan)}, " if len(self.plan) > 1 else ""
            msg = (
                f"CALIBRATION  -  {seg_info}point {self.point_idx + 1}/{total}   |   "
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
        if (
            self.strategy is not None
            and self.strategy.is_calibrated()
            and data.gaze_valid
        ):
            vec = data.features.as_vector()
            if vec is not None:
                raw = self.strategy.predict(vec)
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