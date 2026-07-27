"""Tests for head pose decomposition, 4D/7D feature vectors and chinrest monitor."""

from __future__ import annotations

import math

import numpy as np
import pytest

from core.calibration import NormalizedPoint, PolynomialRidgeCalibration, make_grid
from core.head_pose import HeadPoseMonitor, PoseTuple
from core.vision import GazeFeatures, HeadPose, LandmarksData, build_feature_vector, decompose_head_pose


def _rot(yaw: float, pitch: float, roll: float) -> np.ndarray:
    """R = Rz(roll) @ Ry(yaw) @ Rx(pitch) - the convention decompose assumes."""

    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cr, sr = math.cos(roll), math.sin(roll)

    rx = np.array([[1.0, 0.0, 0.0], [0.0, cp, -sp], [0.0, sp, cp]])
    ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    rz = np.array([[cr, -sr, 0.0], [sr, cr, 0.0], [0.0, 0.0, 1.0]])

    m = np.eye(4)
    m[:3, :3] = rz @ ry @ rx
    return m


@pytest.mark.parametrize(
    "yaw, pitch, roll",
    [(15.0, 0.0, 0.0), (0.0, -12.0, 0.0), (0.0, 0.0, 20.0), (10.0, -8.0, 5.0), (-25.0, 14.0, -9.0)],
)
def test_pose_decomposition_roundtrip(yaw: float, pitch: float, roll: float) -> None:
    m = _rot(math.radians(yaw), math.radians(pitch), math.radians(roll))
    pose = decompose_head_pose(m)

    assert pose is not None
    got = pose.as_degrees()

    for expected, actual in zip((yaw, pitch, roll), got):
        assert abs(expected - actual) < 0.5, f"expected {(yaw, pitch, roll)}, got {got}"


def test_pose_decomposition_rejects_garbage() -> None:
    assert decompose_head_pose(np.zeros((3, 3))) is None
    assert decompose_head_pose(np.full((4, 4), np.nan)) is None


def _frame(h_shift: float, yaw: float) -> LandmarksData:
    f = GazeFeatures(0.1 + h_shift, 0.02, 0.1 + h_shift, 0.02)
    return LandmarksData(
        face_detected=True, gaze_valid=True, features=f,
        head_pose=HeadPose(yaw=yaw, pitch=0.0, roll=0.0),
    )


def test_feature_vector_dimensions() -> None:
    data = _frame(0.0, 0.1)

    vec_4d = build_feature_vector(data)
    assert vec_4d is not None and len(vec_4d) == 4

    vec_7d = build_feature_vector(data, use_head_pose=True)
    assert vec_7d is not None and len(vec_7d) == 7

    data_no_pose = LandmarksData(gaze_valid=True, features=GazeFeatures(0.0, 0.0, 0.0, 0.0))
    assert build_feature_vector(data_no_pose, use_head_pose=True) is None
    assert build_feature_vector(LandmarksData()) is None


def test_pose_features_reduce_yaw_error() -> None:
    """
    Synthetic head-rotation experiment: the true screen position depends 
    on PC-EC features and on yaw. Evaluation happens at a yaw the 4D 
    model must misattribute to gaze.
    """

    rng = np.random.default_rng(42)
    yaw_gain = 0.35 # How strongly yaw leaks into the horizontal eye feature

    def make_sample(p: NormalizedPoint, yaw: float) -> tuple[list[float], list[float]]:
        h = 0.25 * (p.x - 0.5) + yaw_gain * yaw + float(rng.normal(0, 0.002))
        v = 0.18 * (p.y - 0.5) + float(rng.normal(0, 0.002))
        feats_4d = [h, v, h + 0.01, v - 0.01]
        feats_7d = feats_4d + [yaw + float(rng.normal(0, 0.002)), 0.0, 0.0]
        return feats_4d, feats_7d

    grid = make_grid(3)
    model_4d = PolynomialRidgeCalibration(grid=grid)
    model_7d = PolynomialRidgeCalibration(grid=grid, use_head_pose=True)

    for p in grid:
        for _ in range(25):
            yaw = float(rng.uniform(-0.12, 0.12)) # Calibration covers slight motion
            f4, f7 = make_sample(p, yaw)
            model_4d.add_samples(p, [f4])
            model_7d.add_samples(p, [f7])

    model_4d.train()
    model_7d.train()

    screen_w, screen_h = 1920, 1080
    test_yaw = math.radians(10.0)
    errors_4d: list[float] = []
    errors_7d: list[float] = []

    for p in make_grid(3, margin=0.25):
        f4, f7 = make_sample(p, test_yaw)
        p4, p7 = model_4d.predict(f4), model_7d.predict(f7)

        assert p4 is not None and p7 is not None

        errors_4d.append(math.hypot((p4.x - p.x) * screen_w, (p4.y - p.y) * screen_h))
        errors_7d.append(math.hypot((p7.x - p.x) * screen_w, (p7.y - p.y) * screen_h))

    e4 = float(np.mean(errors_4d))
    e7 = float(np.mean(errors_7d))

    print(f"\n4D error under 10 deg yaw: {e4:.0f} px | 7D: {e7:.0f} px")
    assert e4 > 60.0, f"4D model should be visibly displaced by yaw, got {e4:.0f}px"
    assert e7 < e4 / 4.0, f"pose features should cut error >=4x ({e4:.0f} -> {e7:.0f})"


def test_chinrest_monitor_thresholds_and_rate_limit() -> None:
    mon = HeadPoseMonitor(warn_interval_s=2.0)
    assert not mon.has_reference
    assert mon.update(1.0, 1.0, 1.0).ok # Inactive without reference

    # Reference list with a blatant outlier
    ref: list[PoseTuple] = [(0.02, -0.01, 0.005)] * 20 + [(0.5, 0.5, 0.5)] 
    assert mon.set_reference(ref)

    ok_state = mon.update(0.02, -0.01, 0.005)
    assert ok_state.ok and abs(ok_state.dyaw_deg) < 0.1 # Median killed the outlier

    drift_state = mon.update(0.02 + math.radians(9.0), -0.01, 0.005) # 9 deg > 8 deg
    assert not drift_state.ok and drift_state.worst_axis()[0] == "yaw"

    roll_only_state = mon.update(0.02, -0.01, 0.005 + math.radians(9.0)) # 9 < 10 deg
    assert roll_only_state.ok

    # Rate limiting: 1st warn passes, immediate 2nd blocked, after 2 s passes.
    assert mon.should_warn(now=100.0)
    assert not mon.should_warn(now=100.5)
    assert mon.should_warn(now=102.1)