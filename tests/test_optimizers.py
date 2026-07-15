"""Optimizers must respect the budget and produce legal thresholds.

The budget assertion is the important one: it is checked against the counter
inside the objective, so an optimizer cannot under-report evaluations.
"""

import numpy as np
import pytest
from skimage import data

from threshmh.exact import exact_thresholds
from threshmh.histogram import HistogramStats
from threshmh.objectives import ThresholdProblem
from threshmh.optimizers import OPTIMIZERS
from threshmh.repair import STRATEGIES


@pytest.fixture(scope="module")
def stats():
    return HistogramStats(np.bincount(data.coins().ravel(), minlength=256))


@pytest.mark.parametrize("algo", sorted(OPTIMIZERS))
@pytest.mark.parametrize("repair", sorted(STRATEGIES))
@pytest.mark.parametrize("C", [2, 4, 6])
def test_budget_is_never_exceeded(stats, algo, repair, C):
    problem = ThresholdProblem(stats, C, "otsu", STRATEGIES[repair])
    budget = 300
    OPTIMIZERS[algo](problem, budget, np.random.default_rng(0))
    assert problem.evaluations <= budget


@pytest.mark.parametrize("algo", sorted(OPTIMIZERS))
def test_never_beats_the_proven_optimum(stats, algo):
    """A search result above the exact optimum means the objective is broken."""
    C = 4
    _, optimal = exact_thresholds(stats, C, "otsu")
    problem = ThresholdProblem(stats, C, "otsu", STRATEGIES["distinct"])
    result = OPTIMIZERS[algo](problem, 2000, np.random.default_rng(1))
    assert result.best_score <= optimal + 1e-9


@pytest.mark.parametrize("algo", sorted(OPTIMIZERS))
def test_results_are_well_formed(stats, algo):
    C = 5
    problem = ThresholdProblem(stats, C, "kapur", STRATEGIES["distinct"])
    result = OPTIMIZERS[algo](problem, 1000, np.random.default_rng(2))
    t = np.asarray(result.best_thresholds)
    assert t.size == C - 1
    assert np.all(np.diff(t) > 0)
    assert result.trace.size == problem.evaluations
    assert np.all(np.diff(result.trace) >= 0)  # best-so-far is monotone


def test_same_seed_reproduces_exactly(stats):
    def once():
        p = ThresholdProblem(stats, 4, "otsu", STRATEGIES["distinct"])
        return OPTIMIZERS["de"](p, 800, np.random.default_rng(42)).best_score

    assert once() == once()
