"""Non-parametric comparison: Friedman -> Holm-corrected Wilcoxon -> Cliff's delta.

The two levels are deliberately kept apart, because conflating them is the most
common statistical error in this literature:

* **Across-suite** (:func:`friedman_ranks`, :func:`posthoc_holm`) operates on an
  ``images x algorithms`` matrix of per-image aggregates.  It answers "is A
  better than B *across the image set*".
* **Within-image** (:func:`cliffs_delta`) operates on the **raw runs** for a
  single image.  It answers "when A and B both run on this image, how often
  does A win".

An effect size computed on the aggregates would just restate the Friedman
ranks. Report both, label them differently.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scikit_posthocs as sp
from scipy.stats import friedmanchisquare

# Romano et al. cut-points.
_MAGNITUDES = ((0.147, "negligible"), (0.330, "small"), (0.474, "medium"))


def friedman_ranks(matrix: pd.DataFrame, lower_is_better: bool = True):
    """Friedman test over a blocked design.

    Parameters
    ----------
    matrix : DataFrame
        Rows are blocks (images); columns are algorithms; values are the
        per-image aggregate (median over runs).
    lower_is_better : bool
        Direction of the metric. Rank 1 is always the best algorithm.

    Returns
    -------
    stat, p, ranks : float, float, Series
        ``ranks`` is the mean rank per algorithm, ascending (best first).
        If ``p`` is not significant there is **no post-hoc** -- stop.
    """
    if matrix.shape[1] < 3:
        raise ValueError("Friedman needs >= 3 groups; use Wilcoxon directly for 2")
    stat, p = friedmanchisquare(*[matrix[c].to_numpy() for c in matrix.columns])
    ranks = matrix.rank(axis=1, ascending=lower_is_better).mean(axis=0)
    return float(stat), float(p), ranks.sort_values()


def posthoc_holm(matrix: pd.DataFrame) -> pd.DataFrame:
    """Pairwise Wilcoxon **signed-rank** with Holm step-down correction.

    Paired because the blocks (images) are shared across algorithms.  Returns a
    symmetric matrix of adjusted p-values, labelled with the column names.
    """
    groups = [matrix[c].to_numpy() for c in matrix.columns]
    out = sp.posthoc_wilcoxon(groups, p_adjust="holm")
    out.index = list(matrix.columns)
    out.columns = list(matrix.columns)
    return out


def cliffs_delta(a, b) -> float:
    """Cliff's delta: stochastic dominance of ``a`` over ``b``.

    Run on **raw runs within one problem**, never on aggregates.
    Returns a value in [-1, 1]; positive means ``a`` tends to exceed ``b``.
    """
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.size == 0 or b.size == 0:
        return float("nan")
    gt = int((a[:, None] > b[None, :]).sum())
    lt = int((a[:, None] < b[None, :]).sum())
    return (gt - lt) / (a.size * b.size)


def magnitude(delta: float) -> str:
    """Romano et al. magnitude label for a Cliff's delta."""
    if np.isnan(delta):
        return "undefined"
    d = abs(delta)
    for cut, name in _MAGNITUDES:
        if d < cut:
            return name
    return "large"


def pairwise_cliffs(runs: dict[str, np.ndarray]) -> pd.DataFrame:
    """Cliff's delta for every ordered pair of algorithms, from raw runs."""
    names = list(runs)
    out = pd.DataFrame(np.zeros((len(names), len(names))), index=names, columns=names)
    for i in names:
        for j in names:
            out.loc[i, j] = 0.0 if i == j else cliffs_delta(runs[i], runs[j])
    return out
