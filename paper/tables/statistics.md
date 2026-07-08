## Across-suite comparison

Friedman chi-square = 568.34, p = 7.349e-123, over 280 blocks
(block = image x criterion x class count x repair; value = **mean** relative gap over
25 runs -- the median is identically zero for DE/PSO/GA in most cells, because they
reach the exact optimum in over half their runs, so a test on medians would compare ties).

### Mean ranks (1 = best)

| Algorithm | Mean rank |
|---|---|
| PSO (inertia 0.9->0.4, c1=c2=2.0) | 1.739 |
| DE/rand/1/bin (F=0.5, CR=0.9) | 2.048 |
| GA (SBX eta=15, poly-mut eta=20) | 2.413 |
| Random search | 3.800 |

### Pairwise Wilcoxon signed-rank, Holm-corrected

| | random | de | pso | ga |
|---|---|---|---|---|
| **random** | -- | 4.09e-40 | 5.48e-40 | 1.48e-38 |
| **de** | 4.09e-40 | -- | 3.01e-04 | 2.15e-08 |
| **pso** | 5.48e-40 | 3.01e-04 | -- | 2.79e-17 |
| **ga** | 1.48e-38 | 2.15e-08 | 2.79e-17 | -- |

### Cliff's delta vs. random search (raw runs, pooled)

| Algorithm | delta | Magnitude |
|---|---|---|
| DE/rand/1/bin (F=0.5, CR=0.9) | -0.703 | large |
| PSO (inertia 0.9->0.4, c1=c2=2.0) | -0.732 | large |
| GA (SBX eta=15, poly-mut eta=20) | -0.654 | large |

Negative delta means a *smaller* gap than random search, i.e. better.
