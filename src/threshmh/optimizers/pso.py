"""Particle Swarm Optimisation with linearly decreasing inertia.

There is no single "PSO"; this is the classic Shi & Eberhart inertia-weight
variant with velocity clamping.  The variant is named in every table caption
for the same reason DE's is.
"""

from __future__ import annotations

import numpy as np

from .base import Result, clip, evaluate_population, finalize, init_population


def pso(
    problem,
    budget: int,
    rng,
    pop_size: int = 20,
    w_start: float = 0.9,
    w_end: float = 0.4,
    c1: float = 2.0,
    c2: float = 2.0,
    **_,
) -> Result:
    pos = init_population(problem, pop_size, rng)
    span = problem.ub - problem.lb
    vel = rng.uniform(-span, span, size=(pop_size, problem.dim)) * 0.1
    v_max = 0.2 * span

    fit = evaluate_population(problem, pos, budget)
    pbest, pbest_fit = pos.copy(), fit.copy()
    g = int(np.argmin(pbest_fit))
    gbest, gbest_fit = pbest[g].copy(), pbest_fit[g]

    while problem.evaluations < budget:
        # Inertia decreases with consumed budget, not with iteration count, so
        # the schedule is identical regardless of population size.
        frac = problem.evaluations / budget
        w = w_start + (w_end - w_start) * frac

        r1 = rng.random((pop_size, problem.dim))
        r2 = rng.random((pop_size, problem.dim))
        vel = w * vel + c1 * r1 * (pbest - pos) + c2 * r2 * (gbest - pos)
        vel = np.clip(vel, -v_max, v_max)
        pos = clip(pos + vel, problem)

        for i in range(pop_size):
            if problem.evaluations >= budget:
                break
            f = problem.evaluate(pos[i])
            if f <= pbest_fit[i]:
                pbest[i], pbest_fit[i] = pos[i].copy(), f
                if f <= gbest_fit:
                    gbest, gbest_fit = pos[i].copy(), f

    return finalize(problem)
