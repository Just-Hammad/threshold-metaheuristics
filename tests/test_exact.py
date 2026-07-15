"""The exact solver is the reference every other number is measured against.

Three independent checks: brute force on a small instance, agreement with
scikit-image's own exact implementation, and dominance over random sampling.
"""

import itertools

import numpy as np
import pytest
from skimage import data, filters

from threshmh.exact import exact_thresholds
from threshmh.histogram import HistogramStats
from threshmh.objectives import score


def _brute_force(stats, C, criterion):
    best_v, best_t = -np.inf, None
    for t in itertools.combinations(range(stats.L - 1), C - 1):
        v = score(stats, np.array(t), criterion)
        if v > best_v:
            best_v, best_t = v, np.array(t)
    return best_t, best_v


@pytest.mark.parametrize("criterion", ["otsu", "kapur"])
@pytest.mark.parametrize("C", [2, 3, 4])
def test_dp_matches_brute_force(criterion, C):
    """On a 24-bin histogram, exhaustive enumeration is feasible. They must tie."""
    rng = np.random.default_rng(7)
    stats = HistogramStats(rng.integers(1, 200, size=24))
    _, v_dp = exact_thresholds(stats, C, criterion)
    _, v_bf = _brute_force(stats, C, criterion)
    assert v_dp == pytest.approx(v_bf, rel=1e-12)


@pytest.mark.parametrize("C", [2, 3, 4, 5])
def test_dp_matches_skimage_multiotsu(C):
    """Independent exact implementation, same optimum."""
    img = data.camera()
    stats = HistogramStats(np.bincount(img.ravel(), minlength=256))
    _, v_dp = exact_thresholds(stats, C, "otsu")
    v_sk = score(stats, np.asarray(filters.threshold_multiotsu(image=img, classes=C, nbins=256)), "otsu")
    assert v_dp == pytest.approx(v_sk, rel=1e-9)


@pytest.mark.parametrize("criterion", ["otsu", "kapur"])
@pytest.mark.parametrize("C", [3, 5, 7])
def test_dp_dominates_random_sampling(criterion, C):
    """No sampled solution may beat a proven optimum."""
    img = data.coins()
    stats = HistogramStats(np.bincount(img.ravel(), minlength=256))
    _, v_dp = exact_thresholds(stats, C, criterion)
    rng = np.random.default_rng(3)
    for _ in range(5000):
        t = np.sort(rng.choice(255, size=C - 1, replace=False))
        assert score(stats, t, criterion) <= v_dp + 1e-9


def test_thresholds_are_sorted_and_in_range():
    img = data.camera()
    stats = HistogramStats(np.bincount(img.ravel(), minlength=256))
    t, _ = exact_thresholds(stats, 6, "otsu")
    assert t.size == 5
    assert np.all(np.diff(t) > 0)
    assert t.min() >= 0 and t.max() <= 254


def test_rejects_impossible_requests():
    stats = HistogramStats(np.ones(8))
    with pytest.raises(ValueError):
        exact_thresholds(stats, 1, "otsu")
    with pytest.raises(ValueError):
        exact_thresholds(stats, 99, "otsu")
