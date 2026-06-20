"""Differential Evolution, DE/rand/1/bin.

The canonical fully-specified variant: there is no ambiguity about what
"DE/rand/1/bin, F=0.5, CR=0.9" means, which is why it is the reference
algorithm here.  Any table that says only "DE" is under-specified.

Implementation note: all randomness for a generation is drawn in one batch.
``Generator.choice(..., replace=False)`` costs tens of microseconds per call,
which on this objective is more expensive than the objective itself.  The
algorithm is unchanged; only the source of the random numbers is reorganised.
"""

from __future__ import annotations

import numpy as np

from .base import Result, clip, evaluate_population, finalize, init_population


def _distinct_donors(pop_size: int, rng) -> np.ndarray:
    """``(pop_size, 3)`` donor indices, each row distinct and excluding its own row.

    Drawn in bulk and repaired by resampling only the offending entries, which
    keeps the distribution uniform over valid triples.
    """
    idx = rng.integers(0, pop_size, size=(pop_size, 3))
    own = np.arange(pop_size)[:, None]
    for _ in range(32):  # in practice converges in 2-3 passes
        # Column j clashes if it repeats the target row or any earlier column.
        clash = np.zeros_like(idx, dtype=bool)
        clash[:, 0] = (idx[:, 0] == own[:, 0])
        clash[:, 1] = (idx[:, 1] == own[:, 0]) | (idx[:, 1] == idx[:, 0])
        clash[:, 2] = (idx[:, 2] == own[:, 0]) | (idx[:, 2] == idx[:, 0]) | (idx[:, 2] == idx[:, 1])
        if not clash.any():
            break
        idx[clash] = rng.integers(0, pop_size, size=int(clash.sum()))
    return idx


def de_rand_1_bin(
    problem, budget: int, rng, pop_size: int = 20, F: float = 0.5, CR: float = 0.9, **_
) -> Result:
    dim = problem.dim
    pop = init_population(problem, pop_size, rng)
    fit = evaluate_population(problem, pop, budget)

    while problem.evaluations < budget:
        donors = _distinct_donors(pop_size, rng)
        cross = rng.random((pop_size, dim)) < CR
        forced = rng.integers(0, dim, size=pop_size)
        cross[np.arange(pop_size), forced] = True  # trial is never a copy of the target

        mutants = clip(pop[donors[:, 0]] + F * (pop[donors[:, 1]] - pop[donors[:, 2]]), problem)
        trials = np.where(cross, mutants, pop)

        for i in range(pop_size):
            if problem.evaluations >= budget:
                break
            f_trial = problem.evaluate(trials[i])
            if f_trial <= fit[i]:  # greedy selection, minimisation
                pop[i], fit[i] = trials[i], f_trial

    return finalize(problem)
