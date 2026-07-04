import math
from core.calibration import make_grid, make_5point

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