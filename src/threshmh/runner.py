"""Experiment execution.

One function runs one (image, criterion, class-count, repair, algorithm, seed)
cell and returns a flat record.  The grid is expanded here, dispatched to a
process pool, and written out as a tidy CSV -- so every downstream table and
figure is a pure function of that one file.

Budget accounting is enforced against ``problem.evaluations`` (the counter
inside the objective), never against an optimizer's own loop, so an optimizer
cannot under-report the evaluations it actually made.
"""

from __future__ import annotations

import itertools
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from .datasets import load_phantoms, load_real
from .exact import exact_thresholds
from .histogram import HistogramStats
from .objectives import ThresholdProblem
from .optimizers import OPTIMIZERS
from .repair import STRATEGIES

# Tolerance for "this run found the proven optimum".  The criterion values are
# O(1e4) for Otsu, so a relative tolerance is the meaningful test.
HIT_RTOL = 1e-9


@dataclass
class RunRecord:
    image: str
    criterion: str
    n_classes: int
    repair: str
    algorithm: str
    seed: int
    budget: int
    evaluations: int
    best_score: float
    optimal_score: float
    gap: float
    rel_gap: float
    hit_optimum: bool
    evals_to_best: int
    seconds: float
    thresholds: str


def histogram_of(img: np.ndarray, n_bins: int = 256) -> HistogramStats:
    return HistogramStats(np.bincount(img.ravel(), minlength=n_bins))


def run_cell(task) -> dict:
    """Execute one (config, seed) cell. Must be top-level for pickling.

    The task carries the 256-bin **histogram**, not the image.  The objective
    depends on the image only through its histogram, and shipping full images
    to worker processes would dominate the runtime (``retina`` alone is ~2 MB,
    times tens of thousands of tasks).
    """
    hist, name, criterion, C, repair_name, algo_name, seed, budget_per_dim, optimal = task

    stats = HistogramStats(hist)
    problem = ThresholdProblem(stats, C, criterion, STRATEGIES[repair_name])
    budget = budget_per_dim * problem.dim
    rng = np.random.default_rng(seed)

    t0 = time.perf_counter()
    result = OPTIMIZERS[algo_name](problem, budget, rng)
    elapsed = time.perf_counter() - t0

    if problem.evaluations > budget:
        raise AssertionError(
            f"{algo_name} used {problem.evaluations} evaluations against a budget of {budget}"
        )

    gap = optimal - result.best_score
    rel_gap = gap / abs(optimal) if optimal != 0 else float("nan")
    # First evaluation index at which the best-so-far reached its final value.
    trace = result.trace
    evals_to_best = int(np.argmax(trace >= result.best_score) + 1) if trace.size else 0

    return asdict(RunRecord(
        image=name,
        criterion=criterion,
        n_classes=C,
        repair=repair_name,
        algorithm=algo_name,
        seed=seed,
        budget=budget,
        evaluations=problem.evaluations,
        best_score=float(result.best_score),
        optimal_score=float(optimal),
        gap=float(gap),
        rel_gap=float(rel_gap),
        hit_optimum=bool(np.isclose(result.best_score, optimal, rtol=HIT_RTOL)),
        evals_to_best=evals_to_best,
        seconds=elapsed,
    thresholds="-".join(map(str, np.asarray(result.best_thresholds).tolist())),
    ))


def build_tasks(images, cfg) -> tuple[list, dict]:
    """Expand the experiment grid and precompute every exact optimum."""
    optima = {}
    tasks = []
    for name, img in images.items():
        hist = np.bincount(img.ravel(), minlength=256)
        stats = HistogramStats(hist)
        for criterion, C in itertools.product(cfg["criteria"], cfg["class_counts"]):
            t_opt, v_opt = exact_thresholds(stats, C, criterion)
            optima[(name, criterion, C)] = (t_opt, v_opt)
            for repair_name, algo_name, run in itertools.product(
                cfg["repairs"], cfg["algorithms"], range(cfg["runs"])
            ):
                # Seeds are derived from a single master seed and recorded in
                # every row, so any individual run can be replayed alone.
                seed = cfg["master_seed"] + run
                tasks.append((
                    hist, name, criterion, C, repair_name, algo_name, seed,
                    cfg["budget_per_dim"], v_opt,
                ))
    return tasks, optima


def run_experiment(cfg) -> tuple[pd.DataFrame, dict]:
    images = {}
    if cfg.get("use_real", True):
        images.update(load_real(cfg.get("real_images")))
    if cfg.get("use_phantoms", True):
        images.update({k: v[0] for k, v in load_phantoms(cfg["class_counts"]).items()})

    tasks, optima = build_tasks(images, cfg)
    print(f"[runner] {len(images)} images, {len(tasks)} runs")

    workers = cfg.get("workers") or None
    records = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, rec in enumerate(pool.map(run_cell, tasks, chunksize=16), 1):
            records.append(rec)
            if i % 2000 == 0:
                print(f"[runner] {i}/{len(tasks)}")

    return pd.DataFrame.from_records(records), optima
