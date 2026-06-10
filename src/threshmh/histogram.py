"""Histogram prefix statistics.

Every objective in this package decomposes as a sum over contiguous histogram
intervals.  Precomputing three prefix arrays makes any interval statistic O(1),
which is what lets both the exact DP solver and the metaheuristics evaluate the
*same* objective cheaply.

Interval convention throughout the package: ``[a, b]`` is **inclusive** on both
ends and indexes histogram bins, not intensities.
"""

from __future__ import annotations

import numpy as np

EPS = 1e-12


class HistogramStats:
    """O(1) interval statistics over a normalised histogram.

    Parameters
    ----------
    hist : array of shape (L,)
        Raw (unnormalised) histogram counts.

    Attributes
    ----------
    L : int
        Number of bins.
    p : ndarray, shape (L,)
        Normalised histogram, sums to 1.
    """

    __slots__ = ("L", "p", "_P", "_S", "_E", "Pl", "Sl", "El")

    def __init__(self, hist: np.ndarray):
        hist = np.asarray(hist, dtype=np.float64)
        if hist.ndim != 1:
            raise ValueError(f"hist must be 1-D, got shape {hist.shape}")
        total = hist.sum()
        if total <= 0:
            raise ValueError("histogram is empty")

        self.L = hist.size
        self.p = hist / total

        # Prefix sums, length L+1 so that prefix[b+1] - prefix[a] covers [a, b].
        self._P = np.concatenate([[0.0], np.cumsum(self.p)])
        idx = np.arange(self.L, dtype=np.float64)
        self._S = np.concatenate([[0.0], np.cumsum(idx * self.p)])
        # p log p, with the standard convention 0 log 0 = 0.
        plogp = np.where(self.p > EPS, self.p * np.log(np.maximum(self.p, EPS)), 0.0)
        self._E = np.concatenate([[0.0], np.cumsum(plogp)])

        # Python-list mirrors of the prefix arrays.  The objective is evaluated
        # millions of times on vectors of length < 10, where NumPy's per-call
        # overhead (a few microseconds) dwarfs the arithmetic.  Plain list
        # indexing is ~30 ns, which is the difference between a 40-minute
        # experiment and a 3-hour one.  See objectives.score_scalar.
        self.Pl = self._P.tolist()
        self.Sl = self._S.tolist()
        self.El = self._E.tolist()

    # -- interval primitives ------------------------------------------------

    def omega(self, a, b):
        """Class probability mass of bins [a, b]."""
        return self._P[np.add(b, 1)] - self._P[a]

    def moment(self, a, b):
        """Sum of ``i * p_i`` over bins [a, b] (the unnormalised first moment)."""
        return self._S[np.add(b, 1)] - self._S[a]

    def entropy_term(self, a, b):
        """Sum of ``p_i * log p_i`` over bins [a, b]. Negative or zero."""
        return self._E[np.add(b, 1)] - self._E[a]

    @property
    def mean(self) -> float:
        """Global mean intensity (bin index units)."""
        return float(self._S[-1])
