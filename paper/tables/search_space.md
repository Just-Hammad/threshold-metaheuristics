Feasible set = binom(255, C-1): every sorted vector of C-1 distinct thresholds.

| Classes | Search space | Budget (FEs) | Budget / space | Budget could enumerate it all |
|---|---|---|---|---|
| 2 | 255 | 1,000 | 3.92e+00 | **yes** |
| 3 | 32,385 | 2,000 | 6.18e-02 | no |
| 4 | 2,731,135 | 3,000 | 1.10e-03 | no |
| 5 | 172,061,505 | 4,000 | 2.32e-05 | no |
| 6 | 8,637,487,551 | 5,000 | 5.79e-07 | no |

The dynamic program does not search this space at all: it is `O(C x 256^2)` regardless of C, which is why its cost is flat.
