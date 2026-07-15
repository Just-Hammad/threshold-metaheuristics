import numpy as np
import pytest

from threshmh.repair import allow_collapse, repair_distinct

L = 256


@pytest.mark.parametrize("C", [2, 3, 5, 8])
def test_distinct_is_strictly_increasing_and_in_bounds(C):
    rng = np.random.default_rng(1)
    for _ in range(500):
        x = rng.uniform(-50, 300, size=C - 1)  # deliberately out of bounds
        t = np.array(repair_distinct(x, L, C))
        assert t.min() >= 0 and t.max() <= L - 2
        assert np.all(np.diff(t) > 0), t


@pytest.mark.parametrize("C", [2, 3, 5, 8])
def test_collapse_is_sorted_and_in_bounds(C):
    rng = np.random.default_rng(2)
    for _ in range(500):
        x = rng.uniform(-50, 300, size=C - 1)
        t = np.array(allow_collapse(x, L, C))
        assert t.min() >= 0 and t.max() <= L - 2
        assert np.all(np.diff(t) >= 0)


def test_collapse_actually_permits_duplicates():
    t = allow_collapse(np.array([100.0, 100.2, 100.1]), L, 4)
    assert len(set(t)) < len(t)


def test_distinct_handles_saturation_at_upper_bound():
    """All-max input must still yield a legal strictly increasing vector."""
    t = np.array(repair_distinct(np.full(5, 999.0), L, 6))
    assert np.all(np.diff(t) > 0)
    assert t.max() <= L - 2
