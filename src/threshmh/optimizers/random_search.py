"""Uniform random search.

The baseline that decides the project.  If a metaheuristic cannot beat uniform
sampling over the same box under the same evaluation budget, its machinery is
contributing nothing -- and on a search space this small that is a live
possibility, not a rhetorical one.  Reported alongside every other algorithm,
always.
"""

from __future__ import annotations

from .base import Result, finalize, init_population


def random_search(problem, budget: int, rng, **_) -> Result:
    while problem.evaluations < budget:
        x = init_population(problem, 1, rng)[0]
        problem.evaluate(x)
    return finalize(problem)
