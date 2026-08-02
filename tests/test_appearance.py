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
    bin_confidence,
    FreshSampleConsumer,
    SmoothedStreamConsumer,
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
    expected_deg = k * BIN_WIDTH_DEG + OFFSET_DEG
    assert math.isclose(decode_angle(logits), math.radians(expected_deg), abs_tol=1e-6)


def test_decode_angle_center_bin_near_zero() -> None:
    logits = [-1e9] * N_BINS
    logits[44] = 0.0
    logits[45] = 0.0
    expected_deg = 44.5 * BIN_WIDTH_DEG + OFFSET_DEG
    assert math.isclose(expected_deg, -2.0, abs_tol=1e-9)
    assert math.isclose(decode_angle(logits), math.radians(expected_deg), abs_tol=1e-6)


def test_decode_angle_softmax_weights_two_equal_bins() -> None:
    a, b = 10, 30
    logits = [-1e9] * N_BINS
    logits[a] = 0.0
    logits[b] = 0.0
    mean_idx = (a + b) / 2.0
    expected_deg = mean_idx * BIN_WIDTH_DEG + OFFSET_DEG
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


def test_fresh_sample_consumer_yields_each_inference_once() -> None:
    consumer = FreshSampleConsumer()

    assert consumer.take((0.1, 0.2), 1, 0.0) == (0.1, 0.2)
    assert consumer.take((0.1, 0.2), 1, 0.03) is None
    assert consumer.take((0.1, 0.2), 1, 0.06) is None
    assert consumer.take((0.3, 0.4), 2, 0.30) == (0.3, 0.4)


def test_fresh_sample_consumer_skips_missing_inferences() -> None:
    consumer = FreshSampleConsumer()
    assert consumer.take(None, 0, 0.0) is None


def test_smoothed_stream_consumer_returns_a_value_every_frame() -> None:
    consumer = SmoothedStreamConsumer(time_constant_s=0.2)
    consumer.take((0.0, 0.0), 1, 0.0)

    values = [consumer.take((0.0, 0.0), 1, 0.03 * i) for i in range(1, 10)]
    assert all(v is not None for v in values)


def test_smoothed_stream_consumer_eases_toward_a_new_inference() -> None:
    consumer = SmoothedStreamConsumer(time_constant_s=0.2)
    consumer.take((0.0, 0.0), 1, 0.0)

    step_targets = (1.0, -1.0)
    first = consumer.take(step_targets, 2, 0.033)
    assert first is not None

    # A single frame must not jump the whole way
    assert 0.0 < first[0] < 0.5
    assert -0.5 < first[1] < 0.0

    value = first
    t = 0.033
    for _ in range(60):
        t += 0.033
        value = consumer.take(step_targets, 2, t)
        assert value is not None

    assert math.isclose(value[0], 1.0, abs_tol=0.02)
    assert math.isclose(value[1], -1.0, abs_tol=0.02)


def test_smoothed_stream_consumer_reset_restarts_from_the_next_value() -> None:
    consumer = SmoothedStreamConsumer(time_constant_s=0.2)
    consumer.take((0.0, 0.0), 1, 0.0)
    consumer.reset()

    assert consumer.take((0.9, 0.9), 2, 0.033) == (0.9, 0.9)


def test_both_consumers_satisfy_the_protocol() -> None:
    for consumer in (FreshSampleConsumer(), SmoothedStreamConsumer()):
        assert hasattr(consumer, "take") and hasattr(consumer, "reset")


def test_stream_consumer_derives_its_time_constant_from_the_stream() -> None:
    consumer = SmoothedStreamConsumer()
    t = 0.0
    seq = 1

    # Inferences arriving every ~0.3 s, camera frames every ~0.033 s
    for _ in range(6):
        for i in range(9):
            consumer.take((0.1, 0.2), seq, t)
            t += 0.033
        seq += 1

    tau = consumer._tau()
    assert SmoothedStreamConsumer.MIN_TAU_S <= tau <= SmoothedStreamConsumer.MAX_TAU_S
    assert 0.05 < tau < 0.20


def test_stream_consumer_honours_an_explicit_time_constant() -> None:
    consumer = SmoothedStreamConsumer(time_constant_s=0.25)
    consumer.take((0.0, 0.0), 1, 0.0)
    assert consumer._tau() == 0.25


def test_bin_confidence_is_one_for_a_certain_head() -> None:
    logits = [-1e9] * N_BINS
    logits[40] = 0.0
    assert math.isclose(bin_confidence(logits), 1.0, abs_tol=1e-9)


def test_bin_confidence_is_lowest_for_a_flat_head() -> None:
    # A flat distribution means the crop is unlike the training data
    assert math.isclose(bin_confidence([0.0] * N_BINS), 1.0 / N_BINS, abs_tol=1e-9)
