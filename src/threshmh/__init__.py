"""Exact and metaheuristic multilevel image thresholding.

The package exists to answer one question honestly: multilevel Otsu and Kapur
thresholding have an exact ``O(C L^2)`` dynamic-programming solution, so what
is left for a metaheuristic to do?

See :mod:`threshmh.exact` for the solver everything else is measured against.
"""

from .exact import exact_thresholds
from .histogram import HistogramStats
from .objectives import ThresholdProblem, score, score_scalar

__version__ = "0.1.0"
__all__ = [
    "HistogramStats",
    "ThresholdProblem",
    "exact_thresholds",
    "score",
    "score_scalar",
]
