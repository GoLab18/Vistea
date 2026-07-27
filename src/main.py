import logging
import sys
import time
from enum import Enum, auto

import cv2
import numpy as np
from PySide6.QtCore import QPoint, QRect, Qt, QThread, QTimer, Signal, Slot
from PySide6.QtGui import (
    QCloseEvent,
    QColor,
    QFont,
    QImage,
    QKeyEvent,
    QPaintEvent,
    QPainter,
    QPen,
    QPixmap,
    QScreen,
)
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
from core.activation import ACTIVATION_REGISTRY
from core.actuator import make_actuator
from core.appearance import L2CSConsumerMixin, L2CSEstimator, L2CSWorker, WorkerState
from core.appearance import available as l2cs_available
from core.calibration import (
    DEFAULT_STRATEGY_KEY,
    PURSUIT_LAG_S,
    STRATEGY_REGISTRY,
    CalibrationStrategy,
    FeatureSource,
    NormalizedPoint,
    Segment,
    SegmentKind,
    calculate_sample_spread,
)
from core.evaluation import ValidationReport, compute_point_error
from core.filters import OneEuroFilter, PointFilter
from core.head_pose import HeadPoseMonitor
from core.preprocessing import LIGHTING_CONDITIONS, PREPROCESSOR_FACTORIES
from core.vision import GazeEstimator, LandmarksData, build_feature_vector
from ui_gaze import GazeInteractionController
from typing import Final

# Time without a single sample after SPACE before the overlay tells the user the
# gaze signal has not arrived yet (for example the CNN model is still loading).
GAZE_WAIT_HINT_S: Final[float] = 1.0

# Validation targets intentionally lie off all training grids so the report
# measures pure generalization instead of memorization. The grids place
# anchors at {0.1, 0.5, 0.9} (5-pt, 9-pt) and at linspace(0.1, 0.9, 4)
# (16-pt); every target below is disjoint from all of them. The central
# target is (0.45, 0.55) and not (0.5, 0.5) on purpose: (0.5, 0.5) is a
# training anchor of both the 5-pt and 9-pt grids, so validating on it would
# measure memory of a fitted point rather than generalization. (0.45, 0.55)
# probes the same central region while staying off every training grid,
# making the edge-vs-center error contrast measurable within a single
# session.
VALIDATION_TARGETS: Final[list[NormalizedPoint]] = [
    NormalizedPoint(0.2, 0.2),
    NormalizedPoint(0.8, 0.2),
    NormalizedPoint(0.2, 0.8),
    NormalizedPoint(0.8, 0.8),
    NormalizedPoint(0.35, 0.65),
    NormalizedPoint(0.65, 0.35),
    NormalizedPoint(0.45, 0.55),
]


# Lazily-constructed shared L2CS worker
_l2cs_worker: L2CSWorker | None = None


def _get_l2cs_worker() -> L2CSWorker:
    """Return the process-wide L2CS worker, creating it on first use."""

    global _l2cs_worker
    if _l2cs_worker is None:
        _l2cs_worker = L2CSWorker(L2CSEstimator())
    return _l2cs_worker


def build_strategy_vector(
    strategy: CalibrationStrategy,
    frame: np.ndarray,
    data: LandmarksData,
    consumer: L2CSConsumerMixin | None = None,
) -> list[float] | None:
    """
    Assembles the feature vector based on the strategy's feature source.

    - Geometric: Returns the PC-EC vector (optionally pose-augmented).
    - Appearance (L2CS): Returns the latest CNN (pitch, yaw) angles.

      The `consumer` ensures each background inference is returned at most once.
    """

    if strategy.feature_source == FeatureSource.L2CS:
        if data.face_bbox is None:
            return None
        worker = _get_l2cs_worker()
        worker.submit(frame, data.face_bbox)
        latest, seq = worker.result()
        if latest is None:
            return None
        if consumer is not None:
            if seq == consumer._l2cs_last_seq:
                return None
            consumer._l2cs_last_seq = seq
        return [latest[0], latest[1]]
    return build_feature_vector(data, use_head_pose=strategy.use_head_pose)


class CameraThread(QThread):
    frame_ready = Signal(np.ndarray, LandmarksData)

    def __init__(self) -> None:
        super().__init__()
        self._is_running = True
        self.estimator = GazeEstimator()
        self.start_time_ns = time.time_ns()

    def run(self) -> None:
        cap = cv2.VideoCapture(0)
        while self._is_running:
            ret, frame = cap.read()
            if ret:
                timestamp_ms = int((time.time_ns() - self.start_time_ns) / 1_000_000)
                processed_frame, landmarks_data = self.estimator.process_frame(frame, timestamp_ms)
                self.frame_ready.emit(processed_frame, landmarks_data)
        cap.release()

    def stop(self) -> None:
        self._is_running = False
        self.wait()


class Phase(Enum):
    CALIBRATION = auto()
    VALIDATION = auto()


class CalibrationOverlay(QWidget, L2CSConsumerMixin):
    """
    Frameless full-screen calibration surface.

    A validation phase with off-grid static targets follows automatically
    and produces an accuracy report after the initial calibration phase.
    """

    finished = Signal(CalibrationStrategy, ValidationReport)
    aborted = Signal()

    def __init__(self, strategy: CalibrationStrategy, screen: QScreen) -> None:
        super().__init__()
        self.strategy = strategy
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self.setStyleSheet("background-color: #101014;")
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        self._screen = screen
        self.phase = Phase.CALIBRATION
        self.plan: list[Segment] = strategy.plan()
        self.seg_idx = 0
        self.point_idx = 0
        self.retries = 0
        self.collecting = False
        self.collect_t0 = 0.0
        self.batch: list[list[float]] = []
        self.validation_predictions: list[list[float]] = []

        # Pursuit segment state
        self.pursuit_active = False
        self.pursuit_t0 = 0.0
        self.pursuit_samples = 0
        self._anim = QTimer(self)
        self._anim.setInterval(16)
        self._anim.timeout.connect(self._on_anim_tick)

        # Repaints the overlay so that the progress ring shows immediately,
        # when there is a reason for it.
        self._collect_timer = QTimer(self)
        self._collect_timer.setInterval(100)
        self._collect_timer.timeout.connect(self.update)

        self.pose_batch: list[tuple[float, float, float]] = []
        self.head_monitor: HeadPoseMonitor | None = None

    def start(self) -> None:
        self.setGeometry(self._screen.geometry())
        self.showFullScreen()
        self.setFocus()

    def _build_vec(self, frame: np.ndarray, data: LandmarksData) -> list[float] | None:
        return build_strategy_vector(self.strategy, frame, data, consumer=self)

    @Slot(np.ndarray, LandmarksData)
    def on_frame(self, frame: np.ndarray, data: LandmarksData) -> None:
        if not data.gaze_valid or data.features is None:
            return

        if self.pursuit_active:
            self._on_pursuit_frame(frame, data)
            return

        if not self.collecting:
            return
        vec = self._build_vec(frame, data)
        if vec is None:
            return
        self.batch.append(vec)
        if data.head_pose is not None and self.phase is Phase.CALIBRATION:
            self.pose_batch.append((data.head_pose.yaw, data.head_pose.pitch, data.head_pose.roll))
        if len(self.batch) >= self.strategy.samples_per_point:
            self._commit_point()
        self.update()

    def _on_pursuit_frame(self, frame: np.ndarray, data: LandmarksData) -> None:
        segment = self._current_segment()
        elapsed = time.monotonic() - self.pursuit_t0
        if elapsed >= segment.duration_s:
            self._finish_pursuit(segment)
            return
        if elapsed < PURSUIT_LAG_S:
            return
        vec = self._build_vec(frame, data)
        if vec is None:
            return
        label = segment.position_at(elapsed - PURSUIT_LAG_S)
        self.strategy.add_samples(label, [vec])
        self.pursuit_samples += 1
        if data.head_pose is not None:
            self.pose_batch.append((data.head_pose.yaw, data.head_pose.pitch, data.head_pose.roll))

    def _on_anim_tick(self) -> None:
        if self.pursuit_active:
            segment = self._current_segment()
            if time.monotonic() - self.pursuit_t0 >= segment.duration_s:
                self._finish_pursuit(segment)
        self.update()

    def _commit_point(self) -> None:
        target = self._current_target()
        self._collect_timer.stop()

        if self.phase is Phase.CALIBRATION:
            spread = calculate_sample_spread(self.batch)
            threshold = self.strategy.spread_threshold

            if threshold is not None and spread is not None and spread > threshold:
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

    def _finish_pursuit(self, segment: Segment) -> None:
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

    def _advance_segment(self) -> None:
        if self.phase is Phase.VALIDATION:
            self._finish_validation()
            return
        self.seg_idx += 1
        self.point_idx = 0
        if self.seg_idx >= len(self.plan):
            self._train_and_enter_validation()

    def _train_and_enter_validation(self) -> None:
        self.strategy.train()
        rlog.event(
            "calibration_trained",
            n_points=len(self.strategy.target_sequence()),
            n_samples=self.strategy.sample_count or None,
        )

        monitor = HeadPoseMonitor()
        if monitor.set_reference(self.pose_batch):
            self.head_monitor = monitor
            rlog.event("chinrest_reference_set", n_pose_samples=len(self.pose_batch))

        self.phase = Phase.VALIDATION
        self.point_idx = 0

    def _finish_validation(self) -> None:
        report = self._build_report()
        rlog.event(
            "validation_report",
            mean_error_px=report.mean_error_px,
            mean_error_deg=report.mean_error_deg,
            mean_error_percent=report.mean_error_percent,
            n_points=len(report.points),
        )
        self.finished.emit(self.strategy, report)
        self.close()

    def _median_prediction(self, batch: list[list[float]]) -> list[float]:
        preds = [self.strategy.predict(vec) for vec in batch]
        xs = sorted(p.x for p in preds if p is not None)
        ys = sorted(p.y for p in preds if p is not None)
        return [xs[len(xs) // 2], ys[len(ys) // 2]]

    def _build_report(self) -> ValidationReport:
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
        return report

    def _current_segment(self) -> Segment:
        return self.plan[min(self.seg_idx, len(self.plan) - 1)]

    def _phase_targets(self) -> list[NormalizedPoint]:
        if self.phase is Phase.VALIDATION:
            return VALIDATION_TARGETS
        return self._current_segment().points

    def _current_target(self) -> NormalizedPoint:
        return self._phase_targets()[self.point_idx]

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.isAutoRepeat():
            return
        if event.key() == Qt.Key.Key_Space:
            if self.phase is Phase.CALIBRATION and self._current_segment().kind == SegmentKind.PURSUIT:
                if not self.pursuit_active:
                    self.pursuit_active = True
                    self.pursuit_t0 = time.monotonic()
                    self.pursuit_samples = 0
                    self._anim.start()
                    self.update()
            elif not self.collecting:
                self.collecting = True
                self.batch = []
                self.collect_t0 = time.monotonic()
                self._collect_timer.start()
                self.update()
        elif event.key() == Qt.Key.Key_Escape:
            self.aborted.emit()
            self.close()

    def paintEvent(self, _event: QPaintEvent) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()

        pursuit = self.phase is Phase.CALIBRATION and self._current_segment().kind == SegmentKind.PURSUIT

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
            ring_rect = QRect(cx - 26, cy - 26, 52, 52)
            painter.setBrush(Qt.BrushStyle.NoBrush)

            # Ring visible at 0% progress so that the target feels active on SPACE presses
            track = QPen(QColor(70, 90, 80))
            track.setWidth(4)
            painter.setPen(track)
            painter.drawEllipse(ring_rect)

            pen = QPen(QColor(46, 204, 113))
            pen.setWidth(4)
            painter.setPen(pen)
            span = int(360 * 16 * len(self.batch) / self.strategy.samples_per_point)
            painter.drawArc(ring_rect, 90 * 16, -span)
            if not self.batch and (time.monotonic() - self.collect_t0) > GAZE_WAIT_HINT_S:
                painter.setPen(QColor(150, 160, 170))
                painter.setFont(QFont("Arial", 11))
                painter.drawText(
                    QRect(cx - 140, cy + 34, 280, 22),
                    Qt.AlignmentFlag.AlignCenter,
                    "waiting for gaze signal...",
                )

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


class MainWindow(QMainWindow, L2CSConsumerMixin):
    def __init__(self, screen: QScreen) -> None:
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
        controls.setSpacing(10)

        lbl_strategy = QLabel("Strategy:")
        lbl_strategy.setStyleSheet("font-weight: bold;")
        controls.addWidget(lbl_strategy)

        self.strategy_box = QComboBox()
        self.strategy_box.addItems(STRATEGY_REGISTRY.keys())
        self.strategy_box.setCurrentText(DEFAULT_STRATEGY_KEY)
        self.strategy_box.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.strategy_box.currentTextChanged.connect(self._on_strategy_changed)
        controls.addWidget(self.strategy_box, stretch=2)

        lbl_lighting = QLabel("Lighting:")
        lbl_lighting.setStyleSheet("font-weight: bold;")
        controls.addWidget(lbl_lighting)

        self.lighting_box = QComboBox()
        self.lighting_box.addItems(LIGHTING_CONDITIONS)
        self.lighting_box.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        controls.addWidget(self.lighting_box, stretch=1)

        lbl_prep = QLabel("Preprocessing:")
        lbl_prep.setStyleSheet("font-weight: bold;")
        controls.addWidget(lbl_prep)

        self.prep_box = QComboBox()
        self.prep_box.addItems(PREPROCESSOR_FACTORIES.keys())
        self.prep_box.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        controls.addWidget(self.prep_box, stretch=2)

        lbl_activation = QLabel("Activation:")
        lbl_activation.setStyleSheet("font-weight: bold;")
        controls.addWidget(lbl_activation)

        self.activation_box = QComboBox()
        self.activation_box.addItems(ACTIVATION_REGISTRY.keys())
        self.activation_box.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        controls.addWidget(self.activation_box, stretch=2)

        self.btn_calibrate = QPushButton("Start Calibration")
        self.btn_calibrate.setStyleSheet(
            "background-color: #2980b9; color: white; font-weight: bold; padding: 10px;"
        )
        self.btn_calibrate.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.btn_calibrate.clicked.connect(self.start_calibration)
        controls.addWidget(self.btn_calibrate)

        self.btn_gaze = QPushButton("Gaze Mode")
        self.btn_gaze.setStyleSheet(
            "background-color: #27ae60; color: white; font-weight: bold; padding: 10px;"
        )
        self.btn_gaze.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.btn_gaze.setEnabled(False) # requires a calibrated model
        self.btn_gaze.clicked.connect(self.start_gaze_mode)
        controls.addWidget(self.btn_gaze)

        self.btn_reset = QPushButton("Reset")
        self.btn_reset.setStyleSheet(
            "background-color: #e74c3c; color: white; font-weight: bold; padding: 10px;"
        )
        self.btn_reset.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.btn_reset.clicked.connect(self.reset_calibration)
        controls.addWidget(self.btn_reset)

        main_layout.addLayout(controls)

        self.strategy: CalibrationStrategy | None = None
        self.overlay: CalibrationOverlay | None = None
        self.head_monitor: HeadPoseMonitor | None = None
        self._pose_ok = True
        self.gaze_controller: GazeInteractionController | None = None
        self.mean_error_px: float | None = None

        self.gaze_filter: PointFilter = OneEuroFilter(min_cutoff=0.8, beta=0.5)
        self.latest_gaze: NormalizedPoint | None = None

        self._l2cs_status_state: WorkerState | None = None
 
        self.camera_thread = CameraThread()
        self.camera_thread.frame_ready.connect(self.update_ui)
        self.camera_thread.start()

        self.lighting_box.currentTextChanged.connect(
            lambda name: rlog.bind(lighting_condition=name)
        )

        def _swap_prep(label: str) -> None:
            prep = PREPROCESSOR_FACTORIES[label]()
            self.camera_thread.estimator.preprocessor = prep
            rlog.bind(preprocessing=prep.name)
            
        self.prep_box.currentTextChanged.connect(_swap_prep)
        
        _swap_prep(self.prep_box.currentText())
        rlog.bind(lighting_condition=self.lighting_box.currentText())

    def _on_strategy_changed(self, key: str) -> None:
        factory = STRATEGY_REGISTRY.get(key)
        if factory is None:
            return
        candidate = factory()

        if candidate.feature_source != FeatureSource.L2CS:
            return

        if not l2cs_available():
            return
        _get_l2cs_worker().warmup()

    def start_calibration(self) -> None:
        strategy_factory = STRATEGY_REGISTRY[self.strategy_box.currentText()]
        candidate = strategy_factory()

        is_l2cs = candidate.feature_source == FeatureSource.L2CS
        if is_l2cs and not l2cs_available():
            self.status_label.setText(
                "L2CS-Net unavailable: install the CNN extras "
                "(`uv sync --group cnn`) and place the weights - see "
                "models/README_L2CS.md."
            )
            self.status_label.setStyleSheet(
                "font-size: 15px; font-weight: bold; color: #e67e22; padding: 8px;"
            )
            return

        self.strategy = candidate
        self.gaze_filter.reset()
        self.latest_gaze = None
        self.head_monitor = None
        self._pose_ok = True
        self.btn_gaze.setEnabled(False)
        self.mean_error_px = None

        if is_l2cs:
            self._l2cs_status_state = None
            self.status_label.setText(
                "CNN model loads in the background - samples will flow in a few seconds."
            )
            self.status_label.setStyleSheet(
                "font-size: 15px; font-weight: bold; color: #3498db; padding: 8px;"
            )

        rlog.bind(
            session_id=rlog.new_session_id(),
            strategy=self.strategy.name,
            grid_size=len(self.strategy.target_sequence()),
            lighting_condition=self.lighting_box.currentText(),
            preprocessing=self.camera_thread.estimator.preprocessor.name,
        )

        self.overlay = CalibrationOverlay(self.strategy, self._screen)
        self.camera_thread.frame_ready.connect(self.overlay.on_frame)
        self.overlay.finished.connect(self.on_calibration_finished)
        self.overlay.aborted.connect(self.on_calibration_aborted)
        self.overlay.start()

    @Slot(CalibrationStrategy, ValidationReport)
    def on_calibration_finished(self, strategy: CalibrationStrategy, report: ValidationReport) -> None:
        if self.overlay is not None:
            self.head_monitor = self.overlay.head_monitor
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
        self.mean_error_px = report.mean_error_px
        self.btn_gaze.setEnabled(True)

    @Slot()
    def on_calibration_aborted(self) -> None:
        self._detach_overlay()
        self.strategy = None
        self.status_label.setText("Calibration aborted.")

    def _detach_overlay(self) -> None:
        if self.overlay is not None:
            self.camera_thread.frame_ready.disconnect(self.overlay.on_frame)
            self.overlay = None

    def reset_calibration(self) -> None:
        self.stop_gaze_mode()
        self.strategy = None
        self.latest_gaze = None
        self.head_monitor = None
        self._pose_ok = True
        self.mean_error_px = None
        self.btn_gaze.setEnabled(False)
        self.gaze_filter.reset()
        self.status_label.setText("Calibration cleared. Choose a strategy and start calibration.")
        self.status_label.setStyleSheet("font-size: 16px; font-weight: bold; padding: 8px;")

    def start_gaze_mode(self) -> None:
        if self.strategy is None or not self.strategy.is_calibrated():
            return

        activation = ACTIVATION_REGISTRY[self.activation_box.currentText()]()

        self.gaze_controller = GazeInteractionController(
            self.strategy,
            self._screen,
            activation,
            make_actuator(),
            self.mean_error_px,
        )

        self.gaze_controller.stopped.connect(self.on_gaze_mode_stopped)
        self.showMinimized()

    def stop_gaze_mode(self) -> None:
        if self.gaze_controller is not None:
            self.gaze_controller.stop()

    @Slot()
    def on_gaze_mode_stopped(self) -> None:
        self.gaze_controller = None
        self.showMaximized()
        self.raise_()
        self.activateWindow()
        self.status_label.setText("Gaze mode closed. You can recalibrate or start it again.")

    def _check_head_pose(self, data: LandmarksData) -> bool:
        """Virtual chinrest: freeze the cursor and warn while the head has
        drifted from the calibration pose. Returns True when pose is OK."""

        if self.head_monitor is None or data.head_pose is None:
            return True
        state = self.head_monitor.update(
            data.head_pose.yaw, data.head_pose.pitch, data.head_pose.roll
        )
        if not state.ok:
            if self._pose_ok or self.head_monitor.should_warn():
                axis, dev = state.worst_axis()
                rlog.event(
                    "head_pose_drift",
                    dyaw_deg=round(state.dyaw_deg, 1),
                    dpitch_deg=round(state.dpitch_deg, 1),
                    droll_deg=round(state.droll_deg, 1),
                    level=logging.WARNING,
                )
                self.status_label.setText(
                    f"Head moved from calibration pose ({axis} {dev:+.0f} deg) - "
                    "return to position or recalibrate."
                )
                self.status_label.setStyleSheet(
                    "font-size: 15px; font-weight: bold; color: #e67e22; padding: 8px;"
                )
            if self._pose_ok and self.gaze_controller is not None:
                self.gaze_controller.notify("Head out of calibration pose", "warning")
                self.gaze_controller.note_pose_drift()
            self._pose_ok = False
        elif not self._pose_ok:
            self._pose_ok = True
            self.status_label.setText("Head pose OK.")
            self.status_label.setStyleSheet(
                "font-size: 15px; font-weight: bold; color: #2ecc71; padding: 8px;"
            )
            if self.gaze_controller is not None:
                self.gaze_controller.notify("Head pose OK", "info")
        return state.ok

    def _update_l2cs_status(self) -> None:
        """Reflect the background CNN's lifecycle in the status label."""

        if self.strategy is None or self.strategy.feature_source != FeatureSource.L2CS:
            return
        if _l2cs_worker is None:
            return
        state = _l2cs_worker.state
        if state == self._l2cs_status_state:
            return
        self._l2cs_status_state = state
        if state == "ready":
            self.status_label.setText("CNN model ready - collecting samples.")
            self.status_label.setStyleSheet(
                "font-size: 15px; font-weight: bold; color: #2ecc71; padding: 8px;"
            )
        elif state == "failed":
            self.status_label.setText(
                "CNN model failed to load - see the log and pick another strategy."
            )
            self.status_label.setStyleSheet(
                "font-size: 15px; font-weight: bold; color: #e67e22; padding: 8px;"
            )

    @Slot(np.ndarray, LandmarksData)
    def update_ui(self, frame: np.ndarray, data: LandmarksData) -> None:
        pose_ok = self._check_head_pose(data)
        self._update_l2cs_status()
        vec = None
        if (
            self.strategy is not None
            and self.strategy.is_calibrated()
            and data.gaze_valid
        ):
            vec = build_strategy_vector(self.strategy, frame, data, consumer=self)
            if vec is not None and pose_ok: # pause the cursor while drifted
                raw = self.strategy.predict(vec)
                if raw is not None:
                    fx, fy = self.gaze_filter.filter((raw.x, raw.y), data.timestamp_ms)
                    self.latest_gaze = NormalizedPoint(fx, fy)

        if self.gaze_controller is not None:
            # While the head has drifted, the controller receives no gaze point and
            # the dwell progress freezes instead of accumulating on a mislocated target.
            gaze = self.latest_gaze if pose_ok else None
            self.gaze_controller.on_gaze(gaze, data, vec)

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

    def closeEvent(self, event: QCloseEvent) -> None:
        self.stop_gaze_mode()
        self.camera_thread.stop()
        if _l2cs_worker is not None:
            _l2cs_worker.stop()
        super().closeEvent(event)


def main() -> None:
    rlog.configure()
    app = QApplication(sys.argv)
    screen = app.primaryScreen()
    if screen is None:
        return
    window = MainWindow(screen)
    window.showMaximized()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()