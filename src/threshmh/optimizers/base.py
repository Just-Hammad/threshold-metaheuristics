"""Optimizer interface.

Every optimizer has the signature ``(problem, budget, rng, **params) -> Result``
and searches the continuous box ``[problem.lb, problem.ub]``.  Discretisation is
the problem's job (see :mod:`threshmh.repair`), not the optimizer's, so the same
optimizers work unchanged on any box-constrained objective.

This mirrors the interface the ``mhbench`` spine exposes.  To swap in that
package, replace the imports in :mod:`threshmh.optimizers` -- nothing else
changes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Result:
    """Outcome of one optimizer run."""

    best_score: float
    best_thresholds: np.ndarray
    evaluations: int
    trace: np.ndarray = field(repr=False)


def finalize(problem) -> Result:
    return Result(
        best_score=problem.best_score,
        best_thresholds=problem.best_thresholds,
        evaluations=problem.evaluations,
        trace=np.asarray(problem.trace, dtype=np.float64),
    )


def init_population(problem, n: int, rng) -> np.ndarray:
    return rng.uniform(problem.lb, problem.ub, size=(n, problem.dim))


def clip(x, problem):
    return np.clip(x, problem.lb, problem.ub)


def evaluate_population(problem, pop, budget) -> np.ndarray:
    """Evaluate rows of ``pop`` while budget remains; unspent rows get +inf."""
    f = np.full(len(pop), np.inf)
    for i, x in enumerate(pop):
        if problem.evaluations >= budget:
            break
        f[i] = problem.evaluate(x)
    return f
