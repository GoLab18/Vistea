import math

from core.calibration import (
    PURSUIT_LAG_S,
    HybridCalibration,
    NormalizedPoint,
    Segment,
    SmoothPursuitCalibration,
    STRATEGY_REGISTRY,
    make_grid,
)

def test_plans_declared():
    """Every registered strategy exposes a well-formed Plan-as-Data plan."""

    for factory in STRATEGY_REGISTRY.values():
        s = factory()
        plan = s.plan()
        assert len(plan) >= 1
        for seg in plan:
            assert seg.kind in ("static", "pursuit")
            assert len(seg.points) >= 1
            if seg.kind == "pursuit":
                assert seg.duration_s > 0


def test_pursuit_position_interpolation():
    seg = Segment("pursuit", [NormalizedPoint(0.0, 0.0), NormalizedPoint(1.0, 1.0)], duration_s=10.0)
    mid = seg.position_at(5.0)
    assert math.isclose(mid.x, 0.5, abs_tol=1e-9) and math.isclose(mid.y, 0.5, abs_tol=1e-9)
    assert seg.position_at(-1.0) == seg.points[0] # clamped
    assert seg.position_at(999.0) == seg.points[-1] # clamped


def test_pursuit_lag_labelling_recovers_mapping():
    """
    Simulates the interpreter's clock-based lag compensation: the 'eye'
    follows the target PURSUIT_LAG_S behind. Labelling each sample with the
    trajectory position at (t - lag) must recover the true feature->screen
    mapping; labelling with the position at t (no compensation) must not.
    """

    def run(compensate: bool) -> float:
        strategy = SmoothPursuitCalibration(duration_s=24.0)
        seg = strategy.plan()[0]
        true_map = lambda p: [0.2 * p.x - 0.1, 0.15 * p.y, 0.2 * p.x, 0.15 * p.y - 0.05]
        t, dt = PURSUIT_LAG_S, 1.0 / 30.0
        while t < seg.duration_s:
            gaze_at = seg.position_at(t - PURSUIT_LAG_S) # eye lags the dot
            vec = true_map(gaze_at)
            label = seg.position_at(t - PURSUIT_LAG_S) if compensate else seg.position_at(t)
            strategy.add_samples(label, [vec])
            t += dt
        strategy.train()
        errs = []
        for p in make_grid(3, margin=0.2):
            pred = strategy.predict(true_map(p))
            errs.append(math.hypot(pred.x - p.x, pred.y - p.y))
        return sum(errs) / len(errs)

    err_comp = run(True)
    err_naive = run(False)
    assert err_comp < 0.01, f"compensated labelling should recover mapping, got {err_comp:.4f}"
    assert err_naive > 2 * err_comp, (
        f"uncompensated labelling should be clearly worse ({err_naive:.4f} vs {err_comp:.4f})"
    )


def test_hybrid_combines_static_and_pursuit():

    s = HybridCalibration()
    plan = s.plan()
    assert [seg.kind for seg in plan] == ["static", "pursuit"]

    # Feed both segments into the shared store and train.
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