"""Tests for head pose decomposition, 4D/7D feature vectors and chinrest monitor."""

import math

import numpy as np
import pytest

from core.calibration import (
    NormalizedPoint,
    PolynomialRidgeCalibration,
    make_grid,
)
from core.head_pose import HeadPoseMonitor
from core.vision import GazeFeatures, HeadPose, LandmarksData, build_feature_vector, decompose_head_pose


def _rot(yaw: float, pitch: float, roll: float) -> np.ndarray:
    """R = Rz(roll) @ Ry(yaw) @ Rx(pitch) - the convention decompose assumes."""

    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cr, sr = math.cos(roll), math.sin(roll)
    rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rz = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]])
    m = np.eye(4)
    m[:3, :3] = rz @ ry @ rx
    return m


@pytest.mark.parametrize(
    "yaw,pitch,roll",
    [(15, 0, 0), (0, -12, 0), (0, 0, 20), (10, -8, 5), (-25, 14, -9)],
)
def test_pose_decomposition_roundtrip(yaw, pitch, roll):
    m = _rot(math.radians(yaw), math.radians(pitch), math.radians(roll))
    pose = decompose_head_pose(m)
    got = pose.as_degrees()
    for expected, actual in zip((yaw, pitch, roll), got):
        assert abs(expected - actual) < 0.5, f"expected {(yaw, pitch, roll)}, got {got}"


def test_pose_decomposition_rejects_garbage():
    assert decompose_head_pose(np.zeros((3, 3))) is None
    assert decompose_head_pose(np.full((4, 4), np.nan)) is None


def _frame(h_shift: float, yaw: float) -> LandmarksData:
    f = GazeFeatures(0.1 + h_shift, 0.02, 0.1 + h_shift, 0.02)
    return LandmarksData(
        face_detected=True, gaze_valid=True, features=f,
        head_pose=HeadPose(yaw=yaw, pitch=0.0, roll=0.0),
    )


def test_feature_vector_dimensions():
    d = _frame(0.0, 0.1)
    assert len(build_feature_vector(d)) == 4
    assert len(build_feature_vector(d, use_head_pose=True)) == 7
    d_no_pose = LandmarksData(gaze_valid=True, features=GazeFeatures(0, 0, 0, 0))
    assert build_feature_vector(d_no_pose, use_head_pose=True) is None
    assert build_feature_vector(LandmarksData()) is None


def test_pose_features_reduce_yaw_error():
    """
    Synthetic head-rotation experiment: the true screen
    position depends on PC-EC features AND on yaw (turning the head while
    fixating the same point shifts the projected iris geometry). Calibration
    covers slight head motion; evaluation happens at a yaw the 4D model must
    misattribute to gaze. The 7D model should cut the error by an order of
    magnitude (expected direction ~110 px -> ~20 px at 1920x1080).
    """

    rng = np.random.default_rng(42)
    YAW_GAIN = 0.35 # how strongly yaw leaks into the horizontal eye feature

    def make_sample(p: NormalizedPoint, yaw: float):
        h = 0.25 * (p.x - 0.5) + YAW_GAIN * yaw + rng.normal(0, 0.002)
        v = 0.18 * (p.y - 0.5) + rng.normal(0, 0.002)
        feats = [h, v, h + 0.01, v - 0.01]
        return feats, feats + [yaw + rng.normal(0, 0.002), 0.0, 0.0]

    grid = make_grid(3)
    model4 = PolynomialRidgeCalibration(grid=grid)
    model7 = PolynomialRidgeCalibration(grid=grid, use_head_pose=True)
    for p in grid:
        for _ in range(25):
            yaw = rng.uniform(-0.12, 0.12) # calibration covers slight motion
            f4, f7 = make_sample(p, yaw)
            model4.add_samples(p, [f4])
            model7.add_samples(p, [f7])
    model4.train()
    model7.train()

    W, H = 1920, 1080
    test_yaw = math.radians(10.0)
    errs4, errs7 = [], []
    for p in make_grid(3, margin=0.25):
        f4, f7 = make_sample(p, test_yaw)
        p4, p7 = model4.predict(f4), model7.predict(f7)
        errs4.append(math.hypot((p4.x - p.x) * W, (p4.y - p.y) * H))
        errs7.append(math.hypot((p7.x - p.x) * W, (p7.y - p.y) * H))
    e4, e7 = np.mean(errs4), np.mean(errs7)
    print(f"\n4D error under 10 deg yaw: {e4:.0f} px | 7D: {e7:.0f} px")
    assert e4 > 60, f"4D model should be visibly displaced by yaw, got {e4:.0f}px"
    assert e7 < e4 / 4, f"pose features should cut error >=4x ({e4:.0f} -> {e7:.0f})"


def test_chinrest_monitor_thresholds_and_rate_limit():
    mon = HeadPoseMonitor(warn_interval_s=2.0)
    assert not mon.has_reference
    assert mon.update(1.0, 1.0, 1.0).ok # inactive without reference

    ref = [(0.02, -0.01, 0.005)] * 20 + [(0.5, 0.5, 0.5)] # one outlier glance
    assert mon.set_reference(ref)

    ok = mon.update(0.02, -0.01, 0.005)
    assert ok.ok and abs(ok.dyaw_deg) < 0.1 # median killed the outlier

    drift = mon.update(0.02 + math.radians(9.0), -0.01, 0.005) # 9 deg > 8 deg
    assert not drift.ok and drift.worst_axis()[0] == "yaw"

    roll_only = mon.update(0.02, -0.01, 0.005 + math.radians(9.0)) # 9 < 10 deg
    assert roll_only.ok

    # Rate limiting: 1st warn passes, immediate 2nd blocked, after 2 s passes.
    assert mon.should_warn(now=100.0)
    assert not mon.should_warn(now=100.5)
    assert mon.should_warn(now=102.1)
