"""Real-coded Genetic Algorithm: tournament selection, SBX, polynomial mutation.

The standard Deb operator set, so the algorithm is reproducible from its
parameter list alone.
"""

from __future__ import annotations

import numpy as np

from .base import Result, clip, evaluate_population, finalize, init_population


def _sbx(p1, p2, rng, eta: float, lb, ub):
    """Simulated binary crossover (Deb & Agrawal)."""
    u = rng.random(p1.shape)
    beta = np.where(u <= 0.5, (2 * u) ** (1 / (eta + 1)), (1 / (2 * (1 - u))) ** (1 / (eta + 1)))
    c1 = 0.5 * ((1 + beta) * p1 + (1 - beta) * p2)
    c2 = 0.5 * ((1 - beta) * p1 + (1 + beta) * p2)
    return np.clip(c1, lb, ub), np.clip(c2, lb, ub)


def _poly_mutate(x, rng, eta: float, lb, ub, p_mut):
    """Polynomial mutation (Deb & Goyal)."""
    y = x.copy()
    m = rng.random(x.shape) < p_mut
    if not m.any():
        return y
    u = rng.random(x.shape)
    delta = np.where(u < 0.5, (2 * u) ** (1 / (eta + 1)) - 1, 1 - (2 * (1 - u)) ** (1 / (eta + 1)))
    y[m] = np.clip(x[m] + delta[m] * (ub - lb)[m], lb[m], ub[m])
    return y


def ga(
    problem,
    budget: int,
    rng,
    pop_size: int = 20,
    eta_c: float = 15.0,
    eta_m: float = 20.0,
    tournament: int = 2,
    **_,
) -> Result:
    p_mut = 1.0 / problem.dim
    pop = init_population(problem, pop_size, rng)
    fit = evaluate_population(problem, pop, budget)

    while problem.evaluations < budget:
        # Tournament selection.
        idx = rng.integers(0, pop_size, size=(pop_size, tournament))
        winners = idx[np.arange(pop_size), np.argmin(fit[idx], axis=1)]
        parents = pop[winners]

        children = []
        for i in range(0, pop_size - 1, 2):
            c1, c2 = _sbx(parents[i], parents[i + 1], rng, eta_c, problem.lb, problem.ub)
            children.extend([c1, c2])
        if len(children) < pop_size:
            children.append(parents[-1].copy())
        children = np.array([
            _poly_mutate(c, rng, eta_m, problem.lb, problem.ub, p_mut) for c in children
        ])
        children = clip(children, problem)

        child_fit = np.full(pop_size, np.inf)
        for i in range(pop_size):
            if problem.evaluations >= budget:
                break
            child_fit[i] = problem.evaluate(children[i])

        # Elitist (mu + lambda) survival.
        allp = np.vstack([pop, children])
        allf = np.concatenate([fit, child_fit])
        keep = np.argsort(allf)[:pop_size]
        pop, fit = allp[keep], allf[keep]

    return finalize(problem)
