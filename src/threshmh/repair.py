"""Continuous-to-discrete repair strategies.

Multilevel thresholding is a *discrete* problem over sorted integer threshold
vectors, but DE, PSO and GA search a continuous box.  How that gap is closed is
a real design decision that changes results, and the literature almost never
states which one it used.  Both strategies here are compared explicitly in the
experiment.

Every strategy has signature ``(x, L, n_classes) -> ndarray[int]`` and must
return a sorted vector of ``n_classes - 1`` values in ``[0, L - 2]``.
"""

from __future__ import annotations

import numpy as np

# These run inside the innermost evaluation loop, so they are written in plain
# Python against a list rather than with NumPy ops on length-<10 arrays.  They
# return ``list[int]``; use np.asarray at the boundary if an array is needed.


def repair_distinct(x, L: int, n_classes: int) -> list[int]:
    """Round, sort, then force *strictly increasing* thresholds.

    Duplicates are pushed apart, so every class is guaranteed at least one bin
    and the solution always uses all ``C`` classes.
    """
    hi = L - 2
    vals = x.tolist() if hasattr(x, "tolist") else list(x)
    t = sorted(0 if v < 0 else (hi if v > hi else int(v + 0.5)) for v in vals)
    n = len(t)
    # Forward pass: ensure t[i] > t[i-1].
    for i in range(1, n):
        if t[i] <= t[i - 1]:
            t[i] = t[i - 1] + 1
    # The forward pass can overflow the upper bound; fix with a backward pass.
    if n and t[-1] > hi:
        t[-1] = hi
        for i in range(n - 2, -1, -1):
            if t[i] >= t[i + 1]:
                t[i] = t[i + 1] - 1
    # A backward pass can in principle push below zero when C is close to L;
    # clamping keeps the vector legal (classes may then collapse).
    return [0 if v < 0 else (hi if v > hi else v) for v in t]


def allow_collapse(x, L: int, n_classes: int) -> list[int]:
    """Round and sort only. Duplicate thresholds are permitted.

    A duplicated threshold collapses a class to zero width, letting the search
    effectively use *fewer* than ``C`` classes.  Whether that helps or hurts is
    an empirical question this package answers rather than assumes.
    """
    hi = L - 2
    vals = x.tolist() if hasattr(x, "tolist") else list(x)
    return sorted(0 if v < 0 else (hi if v > hi else int(v + 0.5)) for v in vals)


STRATEGIES = {"distinct": repair_distinct, "collapse": allow_collapse}
