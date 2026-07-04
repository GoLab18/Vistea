import math
from core.calibration import make_grid, make_5point, SmoothPursuitCalibration

# Target coordinates matching VALIDATION_TARGETS in main.py
VALIDATION_TARGETS = [
    (0.2, 0.2), (0.8, 0.2), (0.2, 0.8), (0.8, 0.8), (0.35, 0.65), (0.65, 0.35)
]

def test_grids_sizes():
    """Verify that grid generators yield the expected number of coordinates."""

    assert len(make_grid(4)) == 16
    assert len(make_grid(3)) == 9
    assert len(make_5point()) == 5

def test_validation_disjoint():
    """
    Ensure no validation targets overlap with ANY of the 5, 9, or 16-point 
    training grids. Overlaps would artificially inflate accuracy scores.
    """

    grids = make_grid(4) + make_grid(3) + make_5point()
    for vx, vy in VALIDATION_TARGETS:
        for pt in grids:
            overlap = math.isclose(vx, pt.x, abs_tol=1e-6) and math.isclose(vy, pt.y, abs_tol=1e-6)
            assert not overlap, f"Validation point {vx},{vy} overlaps with grid point {pt.x},{pt.y}"

def test_smooth_pursuit():
    """
    Validate the baseline mechanics of the Smooth Pursuit strategy,
    including sequence generation and simulated model training.
    """

    strategy = SmoothPursuitCalibration()
    
    # Verify Lissajous curve generates the correct number of points
    assert len(strategy.target_sequence()) == 60
    
    # Train using synthetic data (simulating batches collected by UI)
    for pt in strategy.target_sequence():
        # Using an arbitrary uniform 4D feature vector as a mock payload
        batch = [[0.1, 0.2, 0.1, 0.2]] * 10
        strategy.add_samples(pt, batch)
        
    strategy.train()
    assert strategy.is_calibrated()
    
    # Perform a mock prediction to test the active inference path
    pred = strategy.predict([0.1, 0.2, 0.1, 0.2])
    assert pred is not None
    assert 0.0 <= pred.x <= 1.0
    assert 0.0 <= pred.y <= 1.0