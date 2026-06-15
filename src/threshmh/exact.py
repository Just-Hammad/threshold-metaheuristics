"""Exact multilevel thresholding by dynamic programming.

This module is the reason the project exists.

Multilevel Otsu and Kapur thresholding are *not* hard problems.  Both criteria
decompose as a sum of per-class terms over contiguous histogram intervals, so
the optimal partition follows from a textbook interval dynamic program in
``O(C * L^2)`` time -- roughly 500k operations for ``L = 256, C = 8``, i.e.
milliseconds.  Exact DP solutions have been known since Liao, Chen & Chung
(2001) and Luessi et al. (2006).

Note the contrast with ``skimage.filters.threshold_multiotsu``, which is also
exact but searches combinations with complexity ``O(C * h^(C-1) / (C-1)!)``.
That is fine for ``C <= 4`` and becomes punishing beyond it.  The DP here has
no such blow-up, which matters for the central claim: there is no regime in
which multilevel Otsu or Kapur thresholding *needs* a metaheuristic.

Recurrence
----------
Let ``f(c, b)`` be the best total score partitioning bins ``[0..b]`` into ``c``
classes, and ``H(a, b)`` the per-class term from :mod:`threshmh.objectives`::

    f(1, b) = H(0, b)
    f(c, b) = max over a in [c-1 .. b] of  f(c-1, a-1) + H(a, b)

The answer is ``f(C, L-1)``; thresholds come from backtracking.
"""

from __future__ import annotations

import numpy as np

from .histogram import HistogramStats
from .objectives import TERMS, score

NEG_INF = -np.inf


def exact_thresholds(
    stats: HistogramStats, n_classes: int, criterion: str = "otsu"
) -> tuple[np.ndarray, float]:
    """Globally optimal thresholds and their criterion value.

    Parameters
    ----------
    stats : HistogramStats
    n_classes : int
        Number of classes ``C >= 2``.
    criterion : {'otsu', 'kapur'}

    Returns
    -------
    thresholds : ndarray of shape (C - 1,), dtype int64
        Sorted, strictly increasing.
    value : float
        The optimal criterion value. **Higher is better.**

    Notes
    -----
    Runs in ``O(C * L^2)`` time and ``O(C * L)`` memory.  The returned value is
    a proven global optimum, not a best-found -- that is the whole point.
    """
    if n_classes < 2:
        raise ValueError("n_classes must be >= 2")
    if criterion not in TERMS:
        raise ValueError(f"unknown criterion {criterion!r}")

    L = stats.L
    C = n_classes
    if C > L:
        raise ValueError(f"cannot split {L} bins into {C} non-empty classes")

    term = TERMS[criterion]

    # H[a, b] for all intervals, vectorised over b for each a.
    # Upper-triangular; entries with a > b stay -inf and are never selected.
    H = np.full((L, L), NEG_INF, dtype=np.float64)
    idx = np.arange(L)
    for a in range(L):
        b = idx[a:]
        H[a, a:] = term(stats, a, b)

    # f[c][b], with c stored 0-based for c = 1..C.
    f = np.full((C, L), NEG_INF, dtype=np.float64)
    # back[c][b] = the start bin 'a' of the last class, for backtracking.
    back = np.zeros((C, L), dtype=np.int64)

    f[0, :] = H[0, :]

    for c in range(1, C):
        # Candidate split points: the last class starts at a, so the previous
        # c classes must cover [0 .. a-1], requiring a >= c.
        for b in range(c, L):
            a_lo, a_hi = c, b
            prev = f[c - 1, a_lo - 1 : a_hi]          # f(c-1, a-1) for a in [a_lo, a_hi]
            cur = H[a_lo : a_hi + 1, b]               # H(a, b)      for a in [a_lo, a_hi]
            total = prev + cur
            j = int(np.argmax(total))
            f[c, b] = total[j]
            back[c, b] = a_lo + j

    value = float(f[C - 1, L - 1])

    # Backtrack: the start bin of class c is back[c][b]; the threshold between
    # class c-1 and class c is that start minus one.
    thresholds = np.empty(C - 1, dtype=np.int64)
    b = L - 1
    for c in range(C - 1, 0, -1):
        a = int(back[c, b])
        thresholds[c - 1] = a - 1
        b = a - 1

    # The DP value and the public scorer must agree exactly; if they ever
    # disagree the two code paths have drifted apart and every comparison in
    # this package is invalid.
    recomputed = score(stats, thresholds, criterion)
    if not np.isclose(recomputed, value, rtol=1e-9, atol=1e-12):
        raise AssertionError(
            f"DP value {value!r} != score(thresholds) {recomputed!r}; "
            "exact.py and objectives.py have diverged"
        )
    return thresholds, value


def exact_curve(
    stats: HistogramStats, class_counts, criterion: str = "otsu"
) -> dict[int, tuple[np.ndarray, float]]:
    """Exact solutions for several class counts. Convenience wrapper."""
    return {int(c): exact_thresholds(stats, int(c), criterion) for c in class_counts}
