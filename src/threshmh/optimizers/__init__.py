"""Optimizer registry.

To swap in the ``mhbench`` spine's implementations later, replace these imports
with ``from mhbench.algorithms import ...``.  The interface in
:mod:`threshmh.optimizers.base` is deliberately identical.
"""

from .base import Result
from .de import de_rand_1_bin
from .ga import ga
from .pso import pso
from .random_search import random_search

OPTIMIZERS = {
    "random": random_search,
    "de": de_rand_1_bin,
    "pso": pso,
    "ga": ga,
}

LABELS = {
    "random": "Random search",
    "de": "DE/rand/1/bin (F=0.5, CR=0.9)",
    "pso": "PSO (inertia 0.9->0.4, c1=c2=2.0)",
    "ga": "GA (SBX eta=15, poly-mut eta=20)",
}

__all__ = ["OPTIMIZERS", "LABELS", "Result", "de_rand_1_bin", "ga", "pso", "random_search"]
