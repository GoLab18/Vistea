"""Tests for appearance-based gaze estimation (L2CS)."""

from __future__ import annotations

import importlib.util
import math
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from core.appearance import (
    BIN_WIDTH_DEG,
    N_BINS,
    OFFSET_DEG,
    AnglesRadian,
    FaceBoundingBox,
    L2CSWorker,
    available,
    decode_angle,
)


def _wait_until(pred: Callable[[], bool], timeout: float = 3.0) -> bool:
    """Poll `pred` until it is true or `timeout` elapses."""

    deadline = time.time() + timeout
    while time.time() < deadline:
        if pred():
            return True
        time.sleep(0.005)
    return False


class _GatedEstimator:
    """Fake L2CSEstimator whose first `angles` call blocks on a gate."""

    def __init__(self) -> None:
        self.gate = threading.Event()
        self.processed: list[FaceBoundingBox] = []
        self._calls: int = 0
        self._init_failed: bool = False

    def angles(self, frame_bgr: np.ndarray, face_bbox: FaceBoundingBox) -> AnglesRadian:
        self._calls += 1
        if self._calls == 1:
            self.gate.wait(timeout=3.0)
        self.processed.append(face_bbox)
        return (float(face_bbox[0]), float(face_bbox[1]))


class _FailingEstimator:
    """Fake estimator whose initialization fails permanently."""

    def __init__(self) -> None:
        self._init_failed: bool = False

    def angles(self, frame_bgr: np.ndarray, face_bbox: FaceBoundingBox) -> None:
        self._init_failed = True
        return None


class _NoFaceThenFaceEstimator:
    """Init succeeds, but the first frame has no face; the second does."""

    def __init__(self) -> None:
        self._init_failed: bool = False
        self._calls: int = 0

    def angles(self, frame_bgr: np.ndarray, face_bbox: FaceBoundingBox) -> AnglesRadian | None:
        self._calls += 1
        if self._calls == 1:
            return None
        return (1.0, 2.0)


class _WarmupEstimator:
    """Fake estimator whose model build succeeds; records build calls."""

    def __init__(self) -> None:
        self._init_failed: bool = False
        self.ensured: int = 0
        self.inferences: int = 0

    def _ensure_model(self) -> bool:
        self.ensured += 1
        return True

    def angles(self, frame_bgr: np.ndarray, face_bbox: FaceBoundingBox) -> AnglesRadian:
        self.inferences += 1
        return (0.0, 0.0)


class _WarmupFailEstimator:
    """Fake estimator whose model build fails permanently."""

    def __init__(self) -> None:
        self._init_failed: bool = False

    def _ensure_model(self) -> bool:
        self._init_failed = True
        return False

    def angles(self, frame_bgr: np.ndarray, face_bbox: FaceBoundingBox) -> None:
        return None


def test_decode_angle_one_hot_matches_manual() -> None:
    k = 60
    logits = [-1e9] * N_BINS
    logits[k] = 0.0  
    expected_deg = k * BIN_WIDTH_DEG + OFFSET_DEG + BIN_WIDTH_DEG / 2.0
    assert math.isclose(decode_angle(logits), math.radians(expected_deg), abs_tol=1e-6)


def test_decode_angle_center_bin_near_zero() -> None:
    logits = [-1e9] * N_BINS
    logits[44] = 0.0
    logits[45] = 0.0
    expected_deg = 44.5 * BIN_WIDTH_DEG + OFFSET_DEG + BIN_WIDTH_DEG / 2.0
    assert math.isclose(expected_deg, 0.0, abs_tol=1e-9)
    assert math.isclose(decode_angle(logits), 0.0, abs_tol=1e-6)


def test_decode_angle_softmax_weights_two_equal_bins() -> None:
    a, b = 10, 30
    logits = [-1e9] * N_BINS
    logits[a] = 0.0
    logits[b] = 0.0
    mean_idx = (a + b) / 2.0
    expected_deg = mean_idx * BIN_WIDTH_DEG + OFFSET_DEG + BIN_WIDTH_DEG / 2.0
    assert math.isclose(decode_angle(logits), math.radians(expected_deg), abs_tol=1e-6)


def test_available_false_without_weights() -> None:
    assert available(Path("models") / "does_not_exist_l2cs.pkl") is False


def test_available_does_not_import_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    had_torch = "torch" in sys.modules
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)

    assert available() is False
    assert ("torch" in sys.modules) == had_torch


def test_worker_latest_wins_and_seq_progression() -> None:
    est = _GatedEstimator()
    worker = L2CSWorker(est) # type: ignore
    assert worker.state == "idle"

    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    worker.submit(frame, (1, 1, 1, 1))
    assert _wait_until(lambda: est._calls == 1)
    assert worker.state == "loading"

    worker.submit(frame, (2, 2, 2, 2))
    worker.submit(frame, (3, 3, 3, 3))

    est.gate.set()

    assert _wait_until(lambda: worker.result()[1] == 2)
    latest, seq = worker.result()
    assert seq == 2
    assert latest == (3.0, 3.0)
    assert worker.state == "ready"
    assert est.processed == [(1, 1, 1, 1), (3, 3, 3, 3)]

    assert worker.result() == (latest, seq)

    worker.stop()
    assert _wait_until(
        lambda: worker._thread is not None and not worker._thread.is_alive()
    )


def test_worker_reports_failed_on_init_failure() -> None:
    worker = L2CSWorker(_FailingEstimator()) # type: ignore
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    worker.submit(frame, (0, 0, 1, 1))

    assert _wait_until(lambda: worker.state == "failed")
    assert worker.result() == (None, 0)
    worker.stop()


def test_worker_none_after_init_is_not_failure() -> None:
    est = _NoFaceThenFaceEstimator()
    worker = L2CSWorker(est) # type: ignore
    frame = np.zeros((4, 4, 3), dtype=np.uint8)

    worker.submit(frame, (0, 0, 1, 1))
    assert _wait_until(lambda: est._calls >= 1)
    assert worker.state == "loading"

    worker.submit(frame, (0, 0, 1, 1))
    assert _wait_until(lambda: worker.state == "ready")
    result = worker.result()[0]
    assert result is not None and result == (1.0, 2.0)
    worker.stop()


def test_worker_warmup_builds_model_without_inference() -> None:
    est = _WarmupEstimator()
    worker = L2CSWorker(est) # type: ignore
    assert worker.state == "idle"

    worker.warmup()
    assert _wait_until(lambda: worker.state == "ready")
    assert est.ensured >= 1
    assert est.inferences == 0
    assert worker.result() == (None, 0)

    worker.warmup()
    assert worker.state == "ready"
    worker.stop()


def test_worker_warmup_reports_failed_build() -> None:
    worker = L2CSWorker(_WarmupFailEstimator()) # type: ignore
    worker.warmup()
    assert _wait_until(lambda: worker.state == "failed")
    assert worker.result() == (None, 0)
    worker.stop()