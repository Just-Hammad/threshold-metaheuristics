"""Multilevel thresholding objectives: Otsu and Kapur.

Both criteria decompose as a sum of per-class terms that depend only on the
class interval, which is exactly the property that makes the exact dynamic
program in :mod:`threshmh.exact` possible.

**Both are maximised.**  The optimizers in this package minimise, so
:class:`ThresholdProblem` negates once, in one place.

Critically, the metaheuristics and the exact solver call the *same* per-class
term functions here.  If they did not, the comparison between them would be
meaningless.
"""

from __future__ import annotations

import numpy as np

from math import log as _log

from .histogram import EPS, HistogramStats

CRITERIA = ("otsu", "kapur")


def otsu_term(stats: HistogramStats, a, b):
    r"""Otsu's contribution for the class spanning bins [a, b].

    Maximising total between-class variance :math:`\sigma_B^2` is equivalent to
    maximising :math:`\sum_k \omega_k \mu_k^2`, because

    .. math::
        \sigma_B^2 = \sum_k \omega_k (\mu_k - \mu_T)^2
                   = \sum_k \omega_k \mu_k^2 - \mu_T^2

    and :math:`\mu_T` is constant.  With :math:`m = \sum_{i} i\,p_i` over the
    interval, :math:`\omega \mu^2 = m^2 / \omega`.

    An empty class (:math:`\omega = 0`) contributes 0.
    """
    w = stats.omega(a, b)
    m = stats.moment(a, b)
    return np.where(w > EPS, m * m / np.maximum(w, EPS), 0.0)


def kapur_term(stats: HistogramStats, a, b):
    r"""Kapur's entropy contribution for the class spanning bins [a, b].

    .. math::
        H_k = -\sum_{i \in k} \frac{p_i}{\omega} \log \frac{p_i}{\omega}
            = \log \omega - \frac{1}{\omega} \sum_{i \in k} p_i \log p_i

    An empty class contributes 0 by convention (the limit is undefined; this
    choice is shared by the exact solver and the metaheuristics so the
    comparison stays fair).
    """
    w = stats.omega(a, b)
    e = stats.entropy_term(a, b)
    safe_w = np.maximum(w, EPS)
    return np.where(w > EPS, np.log(safe_w) - e / safe_w, 0.0)


TERMS = {"otsu": otsu_term, "kapur": kapur_term}


def class_bounds(thresholds, L: int):
    """Convert a threshold vector into inclusive ``(starts, ends)`` bin ranges.

    ``C - 1`` thresholds produce ``C`` classes: ``[0, t0], [t0+1, t1], ...,
    [t_{C-2}+1, L-1]``.
    """
    t = np.asarray(thresholds, dtype=np.int64).ravel()
    starts = np.concatenate([[0], t + 1])
    ends = np.concatenate([t, [L - 1]])
    return starts, ends


def score(stats: HistogramStats, thresholds, criterion: str = "otsu") -> float:
    """Total criterion value for a threshold vector. **Higher is better.**"""
    if criterion not in TERMS:
        raise ValueError(f"criterion must be one of {CRITERIA}, got {criterion!r}")
    starts, ends = class_bounds(thresholds, stats.L)
    # Degenerate classes (start > end) arise when thresholds collapse; they
    # contribute nothing, matching the empty-class convention above.
    valid = starts <= ends
    if not valid.any():
        return 0.0
    terms = TERMS[criterion](stats, starts[valid], ends[valid])
    return float(np.sum(terms))


def score_scalar(stats: HistogramStats, thresholds, criterion: str = "otsu") -> float:
    """Pure-Python scorer for the inner evaluation loop. **Higher is better.**

    Mathematically identical to :func:`score`, but written against the list
    mirrors in :class:`~threshmh.histogram.HistogramStats` to avoid NumPy's
    per-call overhead on length-<10 vectors.  ``tests/test_objectives.py``
    asserts the two agree to machine precision on random inputs; if that test
    ever fails, the exact solver and the metaheuristics are no longer
    optimising the same function and every comparison in this package is void.
    """
    P, S = stats.Pl, stats.Sl
    L = stats.L
    total = 0.0

    if criterion == "otsu":
        a = 0
        for t in thresholds:
            b1 = t + 1
            if b1 > a:
                w = P[b1] - P[a]
                if w > EPS:
                    m = S[b1] - S[a]
                    total += m * m / w
            a = b1
        if L > a:
            w = P[L] - P[a]
            if w > EPS:
                m = S[L] - S[a]
                total += m * m / w
        return total

    if criterion == "kapur":
        E = stats.El
        a = 0
        for t in thresholds:
            b1 = t + 1
            if b1 > a:
                w = P[b1] - P[a]
                if w > EPS:
                    total += _log(w) - (E[b1] - E[a]) / w
            a = b1
        if L > a:
            w = P[L] - P[a]
            if w > EPS:
                total += _log(w) - (E[L] - E[a]) / w
        return total

    raise ValueError(f"criterion must be one of {CRITERIA}, got {criterion!r}")


class ThresholdProblem:
    """A thresholding instance presented as a box-constrained minimisation problem.

    This is the seam between the image-domain objective and the generic
    optimizers.  It mirrors the interface the ``mhbench`` spine exposes, so
    swapping in that package later is an import change rather than a rewrite.

    Parameters
    ----------
    stats : HistogramStats
    n_classes : int
        Number of output classes ``C``; the search dimension is ``C - 1``.
    criterion : {'otsu', 'kapur'}
    repair : callable
        Maps a continuous vector in the box to an integer threshold vector.
        See :mod:`threshmh.repair`.

    Notes
    -----
    ``evaluate`` **minimises** (it returns the negated criterion) and counts
    every call in :attr:`evaluations`.  The budget is enforced by the runner
    against this counter, never against the optimizer's own loop.
    """

    def __init__(self, stats: HistogramStats, n_classes: int, criterion: str, repair):
        if n_classes < 2:
            raise ValueError("n_classes must be >= 2")
        self.stats = stats
        self.n_classes = n_classes
        self.criterion = criterion
        self.repair = repair
        self.dim = n_classes - 1
        self.lb = np.zeros(self.dim)
        self.ub = np.full(self.dim, stats.L - 2.0)  # last class keeps >= 1 bin
        self.evaluations = 0
        self.best_score = -np.inf
        self.best_thresholds = None
        self.trace: list[float] = []

    def evaluate(self, x) -> float:
        """Minimised objective: ``-score(repair(x))``.

        Every call is counted here.  The runner enforces the budget against
        :attr:`evaluations`, never against an optimizer's own loop counter --
        an optimizer cannot under-report evaluations it actually made.
        """
        t = self.repair(x, self.stats.L, self.n_classes)
        s = score_scalar(self.stats, t, self.criterion)
        self.evaluations += 1
        if s > self.best_score:
            self.best_score = s
            self.best_thresholds = np.asarray(t, dtype=np.int64)
        self.trace.append(self.best_score)
        return -s

    def reset(self) -> None:
        """Clear counters, best-so-far and trace. Called between runs."""
        self.evaluations = 0
        self.best_score = -np.inf
        self.best_thresholds = None
        self.trace = []
