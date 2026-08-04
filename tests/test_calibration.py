"""Tests for calibration strategies: plans, interpolation, lag compensation, hybrid."""

from __future__ import annotations

import math

from core.calibration import (
    PURSUIT_LAG_S,
    FeatureSource,
    HybridCalibration,
    PREDICTION_MARGIN,
    NormalizedPoint,
    PolynomialRidgeCalibration,
    Segment,
    SegmentKind,
    SmoothPursuitCalibration,
    STRATEGY_REGISTRY,
    _as_prediction,
    make_grid,
)


def test_registry_has_16pt_pose_variant() -> None:
    """The dense 16-point grid is also available in a pose-augmented variant,
    targeting the elevated edge error that the sparser 9-point grid leaves
    under-anchored."""

    assert "Ridge 16-pt + pose" in STRATEGY_REGISTRY
    s = STRATEGY_REGISTRY["Ridge 16-pt + pose"]()
    assert s.use_head_pose is True
    assert len(s.target_sequence()) == 16  # 4x4 grid
    assert s.name == "ridge_poly2_16pt_pose"


def test_registry_has_l2cs_linear_variant() -> None:
    """The appearance family is registered as a linear (degree-1) Ridge over
    the L2CS gaze-angle feature source."""

    key = "L2CS-Net + linear (CNN)"
    assert key in STRATEGY_REGISTRY
    s = STRATEGY_REGISTRY[key]()
    assert s.feature_source == FeatureSource.L2CS
    assert s.name == "l2cs_linear"

    assert s.samples_per_point == 20
    assert s.spread_threshold == 0.08

    poly = s._model.steps[0][1]
    assert poly.degree == 1


def test_all_non_l2cs_strategies_use_pcec() -> None:
    """Every geometric strategy declares the PC-EC feature source and the shared
    static-collection budget; only the appearance strategy overrides them."""

    for key, factory in STRATEGY_REGISTRY.items():
        s = factory()
        if s.feature_source is FeatureSource.L2CS:
            continue
        assert s.feature_source == FeatureSource.PCEC, key
        assert s.samples_per_point == 25, key
        assert s.spread_threshold == 0.006, key


def test_plans_declared() -> None:
    """Every registered strategy exposes a well-formed Plan-as-Data plan."""

    for factory in STRATEGY_REGISTRY.values():
        s = factory()
        plan = s.plan()
        assert len(plan) >= 1
        for seg in plan:
            assert seg.kind in (SegmentKind.STATIC, SegmentKind.PURSUIT)
            assert len(seg.points) >= 1
            if seg.kind == SegmentKind.PURSUIT:
                assert seg.duration_s > 0


def test_pursuit_position_interpolation() -> None:
    seg = Segment(SegmentKind.PURSUIT, [NormalizedPoint(0.0, 0.0), NormalizedPoint(1.0, 1.0)], duration_s=10.0)
    mid = seg.position_at(5.0)
    assert math.isclose(mid.x, 0.5, abs_tol=1e-9) and math.isclose(mid.y, 0.5, abs_tol=1e-9)
    assert seg.position_at(-1.0) == seg.points[0]
    assert seg.position_at(999.0) == seg.points[-1]


def test_pursuit_lag_labelling_recovers_mapping() -> None:
    """
    Simulates the interpreter's clock-based lag compensation.
    """

    def run(compensate: bool) -> float:
        strategy = SmoothPursuitCalibration(duration_s=24.0)
        seg = strategy.plan()[0]
        true_map = lambda p: [0.2 * p.x - 0.1, 0.15 * p.y, 0.2 * p.x, 0.15 * p.y - 0.05]
        t, dt = PURSUIT_LAG_S, 1.0 / 30.0
        while t < seg.duration_s:
            gaze_at = seg.position_at(t - PURSUIT_LAG_S)
            vec = true_map(gaze_at)
            label = seg.position_at(t - PURSUIT_LAG_S) if compensate else seg.position_at(t)
            strategy.add_samples(label, [vec])
            t += dt
        strategy.train()
        errs = []
        for p in make_grid(3, margin=0.2):
            pred = strategy.predict(true_map(p))
            if pred is not None:
                errs.append(math.hypot(pred.x - p.x, pred.y - p.y))
        return sum(errs) / len(errs)

    err_comp = run(True)
    err_naive = run(False)
    assert err_comp < 0.01, f"compensated labelling should recover mapping, got {err_comp:.4f}"
    assert err_naive > 2 * err_comp, (
        f"uncompensated labelling should be clearly worse ({err_naive:.4f} vs {err_comp:.4f})"
    )


def test_hybrid_combines_static_and_pursuit() -> None:
    s = HybridCalibration()
    plan = s.plan()
    assert [seg.kind for seg in plan] == [SegmentKind.STATIC, SegmentKind.PURSUIT]

    fake = lambda p: [0.2 * p.x, 0.15 * p.y, 0.2 * p.x, 0.15 * p.y]
    for p in plan[0].points:
        s.add_samples(p, [fake(p)] * 5)
    for i in range(100):
        t = plan[1].duration_s * i / 100
        p = plan[1].position_at(t)
        s.add_samples(p, [fake(p)])
    s.train()
    pred = s.predict(fake(NormalizedPoint(0.5, 0.5)))
    assert pred is not None and math.hypot(pred.x - 0.5, pred.y - 0.5) < 0.02


def test_wild_extrapolation_is_refused_not_saturated() -> None:
    s = PolynomialRidgeCalibration(grid=make_grid(3), degree=1)
    for value in (0.0, 0.5, 1.0):
        s.add_samples(NormalizedPoint(value, value), [[value, value, value, value]])
    s.train()

    assert s.predict([500.0, 500.0, 500.0, 500.0]) is None


def test_the_gate_admits_the_margin_and_refuses_non_finite_output() -> None:
    assert _as_prediction(1.0 + PREDICTION_MARGIN, -PREDICTION_MARGIN) is not None
    assert _as_prediction(1.0 + PREDICTION_MARGIN + 0.01, 0.5) is None

    assert _as_prediction(float("inf"), 0.5) is None
    assert _as_prediction(0.5, float("nan")) is None
