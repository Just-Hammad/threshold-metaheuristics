"""The scalar and vectorised scorers must agree exactly.

If they drift apart, the exact DP solver (which uses the vectorised terms) and
the metaheuristics (which use the scalar path) are optimising different
functions, and every comparison this package makes is void.
"""

import numpy as np
import pytest

from threshmh.histogram import HistogramStats
from threshmh.objectives import class_bounds, score, score_scalar


@pytest.fixture
def stats():
    rng = np.random.default_rng(0)
    hist = rng.integers(0, 500, size=256)
    hist[rng.integers(0, 256, size=40)] = 0  # zero bins exercise the empty-class path
    return HistogramStats(hist)


@pytest.mark.parametrize("criterion", ["otsu", "kapur"])
@pytest.mark.parametrize("C", [2, 3, 4, 6, 8])
def test_scalar_matches_vectorised(stats, criterion, C):
    rng = np.random.default_rng(C)
    for _ in range(200):
        t = np.sort(rng.integers(0, 254, size=C - 1))
        assert score(stats, t, criterion) == pytest.approx(
            score_scalar(stats, t.tolist(), criterion), rel=1e-12, abs=1e-12
        )


@pytest.mark.parametrize("criterion", ["otsu", "kapur"])
def test_duplicate_thresholds_are_handled(stats, criterion):
    """Collapsed classes must not produce NaN or inf in either scorer."""
    t = [50, 50, 50, 200]
    a, b = score(stats, t, criterion), score_scalar(stats, t, criterion)
    assert np.isfinite(a) and np.isfinite(b)
    assert a == pytest.approx(b, rel=1e-12)


def test_class_bounds_partition_the_axis():
    starts, ends = class_bounds([10, 40], L=256)
    assert list(starts) == [0, 11, 41]
    assert list(ends) == [10, 40, 255]


def test_unknown_criterion_rejected(stats):
    with pytest.raises(ValueError):
        score(stats, [100], "definitely-not-a-criterion")
