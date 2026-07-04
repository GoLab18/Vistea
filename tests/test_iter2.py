from core.calibration import make_grid, make_5point

def test_grids_sizes():
    """Verify that grid generators yield the expected number of coordinates."""

    assert len(make_grid(4)) == 16
    assert len(make_grid(3)) == 9
    assert len(make_5point()) == 5