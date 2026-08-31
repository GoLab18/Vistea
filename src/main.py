from __future__ import annotations

import logging
import sys
import time
from collections.abc import Callable, Iterable
from enum import Enum, auto
from typing import Final

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
from core.appearance import (
    FreshSampleConsumer,
    build_appearance_vector,
    L2CSConsumer,
    L2CSEstimator,
    L2CSWorker,
    SmoothedStreamConsumer,
    WorkerState,
)
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

# Status-label tints
STATUS_OK: Final[str] = "#2ecc71"
STATUS_WARN: Final[str] = "#e67e22"
STATUS_BUSY: Final[str] = "#3498db"

# Time without a single sample after SPACE before the overlay tells the user the
# gaze signal has not arrived yet (for example the CNN model is still loading).
GAZE_WAIT_HINT_S: Final[float] = 1.0

# Validation targets lie off every training grid, so the report measures
# generalization rather than memory of a fitted point. The grids anchor at
# {0.1, 0.5, 0.9} (5-pt, 9-pt) and at linspace(0.1, 0.9, 4) (16-pt). The
# central target is (0.45, 0.55) rather than (0.5, 0.5).
VALIDATION_TARGETS: Final[list[NormalizedPoint]] = [
    NormalizedPoint(0.2, 0.2),
    NormalizedPoint(0.8, 0.2),
    NormalizedPoint(0.2, 0.8),
    NormalizedPoint(0.8, 0.8),
    NormalizedPoint(0.35, 0.65),
    NormalizedPoint(0.65, 0.35),
    NormalizedPoint(0.45, 0.55),
]


_l2cs_worker: L2CSWorker | None = None


def _get_l2cs_worker() -> L2CSWorker:
    """Return the process-wide L2CS worker, creating it on first use."""

    global _l2cs_worker
    if _l2cs_worker is None:
        _l2cs_worker = L2CSWorker(L2CSEstimator())
    return _l2cs_worker


def build_strategy_vector(
    strategy: CalibrationStrategy,
    data: LandmarksData,
    consumer: L2CSConsumer | None = None,
) -> list[float] | None:
    """
    Assembles the feature vector the strategy's feature source calls for:
    the PC-EC vector, or the CNN gaze angles at the pace set by `consumer`.
    Either family appends head pose when the strategy asks for it.
    """

    if strategy.feature_source == FeatureSource.L2CS:
        return build_appearance_vector(
            _get_l2cs_worker(), data, consumer, strategy.use_head_pose
        )
    return build_feature_vector(data, strategy.use_head_pose)


class CameraThread(QThread):
    """Grabs webcam frames and emits them with their landmarks attached."""

    frame_ready = Signal(np.ndarray, LandmarksData)

    def __init__(self) -> None:
        super().__init__()
        self._is_running = True
        self.estimator = GazeEstimator()
        self.start_time_ns = time.time_ns()

    def run(self) -> None:
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            rlog.event("camera_unavailable", level=logging.ERROR)
            cap.release()
            return

        try:
            while self._is_running:
                ok, frame = cap.read()
                if not ok:
                    continue

                timestamp_ms = (time.time_ns() - self.start_time_ns) // 1_000_000
                processed_frame, landmarks = self.estimator.process_frame(frame, timestamp_ms)
                self.frame_ready.emit(processed_frame, landmarks)
        finally:
            cap.release()

    def stop(self) -> None:
        self._is_running = False
        self.wait()


class Phase(Enum):
    CALIBRATION = auto()
    VALIDATION = auto()


class CalibrationOverlay(QWidget):
    """
    Frameless full-screen calibration surface.

    A validation phase with off-grid static targets follows automatically
    and produces an accuracy report after the initial calibration phase.
    """

    finished = Signal(ValidationReport)
    aborted = Signal()

    # Re-collections allowed on one scattered calibration point
    MAX_RETRIES: Final[int] = 3

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
        self.validation_predictions: list[NormalizedPoint | None] = []

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

        # Independent observations only, so batch spread stays meaningful
        self._l2cs_consumer: L2CSConsumer = FreshSampleConsumer()

    def start(self) -> None:
        self.setGeometry(self._screen.geometry())
        self.showFullScreen()
        self.setFocus()

    def _frame_features(self, data: LandmarksData) -> list[float] | None:
        return build_strategy_vector(self.strategy, data, consumer=self._l2cs_consumer)

    def _record_pose(self, data: LandmarksData) -> None:
        """Feeds the chinrest reference, which is built from calibration poses."""

        if data.head_pose is not None:
            self.pose_batch.append(
                (data.head_pose.yaw, data.head_pose.pitch, data.head_pose.roll)
            )

    @Slot(np.ndarray, LandmarksData)
    def on_frame(self, frame: np.ndarray, data: LandmarksData) -> None:
        if not data.gaze_valid or data.features is None:
            return

        if self.pursuit_active:
            self._on_pursuit_frame(data)
            return

        if not self.collecting:
            return

        vec = self._frame_features(data)
        if vec is None:
            return

        self.batch.append(vec)
        if self.phase is Phase.CALIBRATION:
            self._record_pose(data)

        if len(self.batch) >= self.strategy.samples_per_point:
            self._commit_point()

        self.update()

    def _on_pursuit_frame(self, data: LandmarksData) -> None:
        segment = self._current_segment()
        elapsed = time.monotonic() - self.pursuit_t0

        if elapsed >= segment.duration_s:
            self._finish_pursuit(segment)
            return

        # The eye trails the target, so a frame is labelled with where the
        # target was one lag ago, and the first lag has no such position.
        if elapsed < PURSUIT_LAG_S:
            return

        vec = self._frame_features(data)
        if vec is None:
            return

        self.strategy.add_samples(segment.position_at(elapsed - PURSUIT_LAG_S), [vec])
        self.pursuit_samples += 1
        self._record_pose(data)

    def _on_anim_tick(self) -> None:
        if self.pursuit_active:
            segment = self._current_segment()
            if time.monotonic() - self.pursuit_t0 >= segment.duration_s:
                self._finish_pursuit(segment)
        self.update()

    def _batch_too_scattered(self, target: NormalizedPoint, spread: float | None) -> bool:
        """Whether this batch should be re-collected."""

        threshold = self.strategy.spread_threshold
        if threshold is None or spread is None or spread <= threshold:
            self.retries = 0
            return False

        if self.retries >= self.MAX_RETRIES:
            rlog.event("calibration_point_forced", point=[target.x, target.y],
                       spread=spread, level=logging.WARNING)
            self.retries = 0
            return False

        rlog.event("calibration_point_rejected", point=[target.x, target.y],
                   spread=spread, level=logging.WARNING)
        self.retries += 1
        return True

    def _commit_point(self) -> None:
        target = self._current_target()
        self._collect_timer.stop()
        self.collecting = False

        if self.phase is Phase.CALIBRATION:
            spread = calculate_sample_spread(self.batch)

            if self._batch_too_scattered(target, spread):
                self.batch = []
                self.update()
                return

            rlog.log_calibration_point(
                target, len(self.batch), spread=spread, features_batch=self.batch
            )
            self.strategy.add_samples(target, self.batch)
        else:
            self.validation_predictions.append(self._median_prediction(self.batch))

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
            n_samples=self.strategy.sample_count,
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
            summary=report.summary(),
            mean_error_px=report.mean_error_px,
            mean_error_deg=report.mean_error_deg,
            mean_error_percent=report.mean_error_percent,
            n_points=len(report.points),
            n_refused=report.n_refused,
        )
        self.finished.emit(report)
        self.close()

    def _median_prediction(self, batch: list[list[float]]) -> NormalizedPoint | None:
        """Per-axis median of a validation batch, or None when the model
        refused every frame of it."""

        preds = [p for p in map(self.strategy.predict, batch) if p is not None]
        if not preds:
            return None

        xs = sorted(p.x for p in preds)
        ys = sorted(p.y for p in preds)
        return NormalizedPoint(xs[len(xs) // 2], ys[len(ys) // 2])

    def _build_report(self) -> ValidationReport:
        size_mm = self._screen.physicalSize()
        geo = self._screen.geometry()
        report = ValidationReport(
            strategy_name=self.strategy.name,
            screen_w_px=geo.width(),
            screen_h_px=geo.height(),
        )

        for target, prediction in zip(VALIDATION_TARGETS, self.validation_predictions):
            if prediction is None:
                report.n_refused += 1
                rlog.event("validation_point_refused", point=[target.x, target.y],
                           level=logging.WARNING)
                continue

            error = compute_point_error(
                target,
                prediction,
                geo.width(),
                geo.height(),
                size_mm.width() or None,
                size_mm.height() or None,
            )
            report.points.append(error)
            rlog.log_validation_point(target, error.error_px, error.error_deg)

        return report

    def _current_segment(self) -> Segment:
        return self.plan[min(self.seg_idx, len(self.plan) - 1)]

    def _phase_targets(self) -> list[NormalizedPoint]:
        if self.phase is Phase.VALIDATION:
            return VALIDATION_TARGETS
        return self._current_segment().points

    def _current_target(self) -> NormalizedPoint:
        return self._phase_targets()[self.point_idx]

    def _in_pursuit_segment(self) -> bool:
        return (
            self.phase is Phase.CALIBRATION
            and self._current_segment().kind is SegmentKind.PURSUIT
        )

    def _start_pursuit(self) -> None:
        self.pursuit_active = True
        self.pursuit_t0 = time.monotonic()
        self.pursuit_samples = 0
        self._anim.start()

    def _start_collecting(self) -> None:
        self.collecting = True
        self.batch = []
        self.collect_t0 = time.monotonic()
        self._collect_timer.start()

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.isAutoRepeat():
            return

        if event.key() == Qt.Key.Key_Escape:
            self.aborted.emit()
            self.close()
            return

        if event.key() != Qt.Key.Key_Space:
            return

        if self._in_pursuit_segment():
            if not self.pursuit_active:
                self._start_pursuit()
                self.update()
        elif not self.collecting:
            self._start_collecting()
            self.update()

    def _instruction(self, pursuit: bool) -> str:
        """The bottom-of-screen caption for the current phase."""

        hold = "Look at the dot and press SPACE. Keep looking until the ring completes."
        abort = "ESC to abort"

        if self.phase is Phase.VALIDATION:
            position = f"point {self.point_idx + 1}/{len(VALIDATION_TARGETS)}"
            return f"VALIDATION  -  {position}   |   {hold}   |   {abort}"

        if pursuit:
            follow = (
                "Follow the moving dot with your eyes." if self.pursuit_active
                else "Press SPACE, then follow the moving dot with your eyes."
            )
            return f"CALIBRATION (pursuit)   |   {follow}   |   {abort}"

        segment = f"segment {self.seg_idx + 1}/{len(self.plan)}, " if len(self.plan) > 1 else ""
        position = f"{segment}point {self.point_idx + 1}/{len(self._phase_targets())}"
        return f"CALIBRATION  -  {position}   |   {hold}   |   {abort}"

    def _paint_target(self, painter: QPainter, center: QPoint, pursuit: bool) -> None:
        if pursuit:
            color = QColor(46, 204, 113)
        elif self.phase is Phase.CALIBRATION:
            color = QColor(52, 152, 219)
        else:
            color = QColor(155, 89, 182)

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(color)
        painter.drawEllipse(center, 16, 16)
        painter.setBrush(QColor(255, 255, 255))
        painter.drawEllipse(center, 4, 4)

    def _paint_collection_ring(self, painter: QPainter, center: QPoint) -> None:
        ring = QRect(center.x() - 26, center.y() - 26, 52, 52)
        painter.setBrush(Qt.BrushStyle.NoBrush)

        # Ring visible at 0% progress so that the target feels active on SPACE presses
        painter.setPen(QPen(QColor(70, 90, 80), 4))
        painter.drawEllipse(ring)

        painter.setPen(QPen(QColor(46, 204, 113), 4))
        span = int(360 * 16 * len(self.batch) / self.strategy.samples_per_point)
        painter.drawArc(ring, 90 * 16, -span)

        if not self.batch and time.monotonic() - self.collect_t0 > GAZE_WAIT_HINT_S:
            painter.setPen(QColor(150, 160, 170))
            painter.setFont(QFont("Arial", 11))
            painter.drawText(
                QRect(center.x() - 140, center.y() + 34, 280, 22),
                Qt.AlignmentFlag.AlignCenter,
                "waiting for gaze signal...",
            )

    def paintEvent(self, _event: QPaintEvent) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()

        pursuit = self._in_pursuit_segment()
        segment = self._current_segment()
        elapsed = time.monotonic() - self.pursuit_t0 if self.pursuit_active else 0.0

        target = segment.position_at(elapsed) if pursuit else self._current_target()
        center = QPoint(int(target.x * w), int(target.y * h))

        self._paint_target(painter, center, pursuit)

        if self.collecting and not pursuit:
            self._paint_collection_ring(painter, center)

        if pursuit and self.pursuit_active:
            done = min(elapsed / segment.duration_s, 1.0)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(46, 204, 113))
            painter.drawRect(0, h - 8, int(w * done), 8)

        painter.setPen(QColor(220, 220, 220))
        painter.setFont(QFont("Arial", 14))
        painter.drawText(
            QRect(0, h - 60, w, 40),
            Qt.AlignmentFlag.AlignCenter,
            self._instruction(pursuit),
        )
        painter.end()


class MainWindow(QMainWindow):
    """Camera preview, run configuration, and the entry points into
    calibration and gaze mode."""

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

        self.status_label = QLabel()
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(self.status_label)
        self._set_status("Choose a strategy and start calibration.")

        controls = QHBoxLayout()
        controls.setSpacing(10)

        self.strategy_box = self._add_combo(
            controls, "Strategy:", STRATEGY_REGISTRY.keys(), stretch=2
        )
        self.strategy_box.setCurrentText(DEFAULT_STRATEGY_KEY)
        self.strategy_box.currentTextChanged.connect(self._on_strategy_changed)

        self.lighting_box = self._add_combo(
            controls, "Lighting:", LIGHTING_CONDITIONS, stretch=1
        )
        self.prep_box = self._add_combo(
            controls, "Preprocessing:", PREPROCESSOR_FACTORIES.keys(), stretch=2
        )
        self.activation_box = self._add_combo(
            controls, "Activation:", ACTIVATION_REGISTRY.keys(), stretch=2
        )

        self.btn_calibrate = self._add_button(
            controls, "Start Calibration", "#2980b9", self.start_calibration
        )
        self.btn_gaze = self._add_button(
            controls, "Gaze Mode", "#27ae60", self.start_gaze_mode
        )
        self.btn_gaze.setEnabled(False)  # Requires a calibrated model
        self.btn_reset = self._add_button(
            controls, "Reset", "#e74c3c", self.reset_calibration
        )

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

        # Interaction reads the CNN every frame, eased between inferences
        self._l2cs_stream: L2CSConsumer = SmoothedStreamConsumer()

        self.camera_thread = CameraThread()
        self.camera_thread.frame_ready.connect(self.update_ui)
        self.camera_thread.start()

        self.lighting_box.currentTextChanged.connect(
            lambda name: rlog.bind(lighting_condition=name)
        )
        self.prep_box.currentTextChanged.connect(self._swap_preprocessor)

        self._swap_preprocessor(self.prep_box.currentText())
        rlog.bind(lighting_condition=self.lighting_box.currentText())

    @staticmethod
    def _add_combo(
        row: QHBoxLayout, caption: str, items: Iterable[str], stretch: int
    ) -> QComboBox:
        """Appends a caption and its unfocusable selector to the row."""

        label = QLabel(caption)
        label.setStyleSheet("font-weight: bold;")
        row.addWidget(label)

        box = QComboBox()
        box.addItems(items)
        box.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        row.addWidget(box, stretch=stretch)
        return box

    @staticmethod
    def _add_button(
        row: QHBoxLayout, text: str, color: str, on_click: Callable[[], None]
    ) -> QPushButton:
        """Appends an action button. Focus is refused throughout the row so the
        keyboard never steals it from the application being driven."""

        button = QPushButton(text)
        button.setStyleSheet(
            f"background-color: {color}; color: white; font-weight: bold; padding: 10px;"
        )
        button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        button.clicked.connect(on_click)
        row.addWidget(button)
        return button

    def _set_status(self, text: str, color: str | None = None) -> None:
        """Single point of truth for the status line, tint included."""

        tint = f"color: {color}; " if color is not None else ""
        self.status_label.setText(text)
        self.status_label.setStyleSheet(
            f"font-size: {16 if color is None else 15}px; font-weight: bold; {tint}padding: 8px;"
        )

    def _swap_preprocessor(self, label: str) -> None:
        preprocessor = PREPROCESSOR_FACTORIES[label]()
        self.camera_thread.estimator.preprocessor = preprocessor
        rlog.bind(preprocessing=preprocessor.name)

    def _on_strategy_changed(self, key: str) -> None:
        factory = STRATEGY_REGISTRY.get(key)
        if factory is None or factory().feature_source != FeatureSource.L2CS:
            return
        if l2cs_available():
            _get_l2cs_worker().warmup()

    def start_calibration(self) -> None:
        strategy_factory = STRATEGY_REGISTRY[self.strategy_box.currentText()]
        candidate = strategy_factory()

        is_l2cs = candidate.feature_source == FeatureSource.L2CS
        if is_l2cs and not l2cs_available():
            self._set_status(
                "L2CS-Net unavailable: install the CNN extras "
                "(`uv sync --group cnn`) and place the weights - see "
                "models/README_L2CS.md.",
                STATUS_WARN,
            )
            return

        self.strategy = candidate
        self.gaze_filter.reset()
        self._l2cs_stream.reset()
        self.latest_gaze = None
        self.head_monitor = None
        self._pose_ok = True
        self.btn_gaze.setEnabled(False)
        self.mean_error_px = None

        if is_l2cs:
            self._l2cs_status_state = None
            self._set_status(
                "CNN model loads in the background - samples will flow in a few seconds.",
                STATUS_BUSY,
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

    @Slot(ValidationReport)
    def on_calibration_finished(self, report: ValidationReport) -> None:
        if self.overlay is not None:
            self.head_monitor = self.overlay.head_monitor
        self._detach_overlay()

        self._set_status(report.summary(), STATUS_OK)
        self.mean_error_px = report.mean_error_px
        self.btn_gaze.setEnabled(True)

    @Slot()
    def on_calibration_aborted(self) -> None:
        self._detach_overlay()
        self.strategy = None
        self._set_status("Calibration aborted.")

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
        self._set_status("Calibration cleared. Choose a strategy and start calibration.")

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
        self._set_status("Gaze mode closed. You can recalibrate or start it again.")

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
                axis, deviation = state.worst_axis()
                rlog.event(
                    "head_pose_drift",
                    dyaw_deg=round(state.dyaw_deg, 1),
                    dpitch_deg=round(state.dpitch_deg, 1),
                    droll_deg=round(state.droll_deg, 1),
                    level=logging.WARNING,
                )
                self._set_status(
                    f"Head moved from calibration pose ({axis} {deviation:+.0f} deg) - "
                    "return to position or recalibrate.",
                    STATUS_WARN,
                )
            if self._pose_ok and self.gaze_controller is not None:
                self.gaze_controller.notify("Head out of calibration pose", "warning")
                self.gaze_controller.note_pose_drift()
            self._pose_ok = False

        elif not self._pose_ok:
            self._pose_ok = True
            self._set_status("Head pose OK.", STATUS_OK)
            if self.gaze_controller is not None:
                self.gaze_controller.notify("Head pose OK", "info")

        return state.ok

    def _update_l2cs_status(self) -> None:
        """
        Reflect the background CNN's lifecycle in the status label,
        once per transition.
        """

        if self.strategy is None or self.strategy.feature_source != FeatureSource.L2CS:
            return
        if _l2cs_worker is None:
            return

        state = _l2cs_worker.state
        if state == self._l2cs_status_state:
            return
        self._l2cs_status_state = state

        if state == "ready":
            self._set_status("CNN model ready - collecting samples.", STATUS_OK)
        elif state == "failed":
            self._set_status(
                "CNN model failed to load - see the log and pick another strategy.",
                STATUS_WARN,
            )

    def _is_calibrated(self) -> bool:
        return self.strategy is not None and self.strategy.is_calibrated()

    def _track_gaze(self, data: LandmarksData, pose_ok: bool) -> list[float] | None:
        if self.strategy is None or not self._is_calibrated() or not data.gaze_valid:
            return None

        vec = build_strategy_vector(self.strategy, data, consumer=self._l2cs_stream)

        # Cursor paused while the head has drifted, and while a lid is in motion
        if vec is None or not pose_ok or not data.lids_stable:
            return vec

        raw = self.strategy.predict(vec)
        if raw is None:
            self.latest_gaze = None
        else:
            fx, fy = self.gaze_filter.filter((raw.x, raw.y), data.timestamp_ms)
            self.latest_gaze = NormalizedPoint(fx, fy)

        return vec

    def _render_preview(self, frame: np.ndarray) -> None:
        """Draw the camera frame into the window, with a crosshair on the
        current estimate."""

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, channels = frame_rgb.shape
        pixmap = QPixmap.fromImage(
            QImage(frame_rgb.data, w, h, channels * w, QImage.Format.Format_RGB888)
        )
        scaled = pixmap.scaled(
            self.video_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )

        if self.latest_gaze is not None and self._is_calibrated():
            self._draw_crosshair(scaled, self.latest_gaze)

        self.video_label.setPixmap(scaled)

    @staticmethod
    def _draw_crosshair(pixmap: QPixmap, gaze: NormalizedPoint) -> None:
        x = int(min(1.0, max(0.0, gaze.x)) * pixmap.width())
        y = int(min(1.0, max(0.0, gaze.y)) * pixmap.height())
        radius = 12

        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(QPen(QColor(255, 0, 0), 4))
        painter.drawEllipse(QPoint(x, y), radius, radius)
        painter.drawLine(x - radius - 5, y, x + radius + 5, y)
        painter.drawLine(x, y - radius - 5, x, y + radius + 5)
        painter.end()

    @Slot(np.ndarray, LandmarksData)
    def update_ui(self, frame: np.ndarray, data: LandmarksData) -> None:
        pose_ok = self._check_head_pose(data)
        self._update_l2cs_status()

        vec = self._track_gaze(data, pose_ok)

        if self.gaze_controller is not None:
            gaze = self.latest_gaze if (pose_ok and data.lids_stable) else None
            self.gaze_controller.on_gaze(gaze, data, vec)

        self._render_preview(frame)

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