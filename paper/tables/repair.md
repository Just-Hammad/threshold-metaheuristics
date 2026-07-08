Two ways of turning a continuous search vector into sorted integer thresholds:

- **distinct** -- duplicates pushed apart, so all C classes are always used
- **collapse** -- duplicates permitted, letting the search use fewer than C classes

### Hit rate (runs reaching the exact optimum)

| Repair | C=2 | C=3 | C=4 | C=5 | C=6 |
|---|---|---|---|---|---|
| collapse | 99.6% | 74.6% | 68.1% | 60.4% | 48.1% |
| distinct | 99.6% | 74.6% | 68.1% | 60.5% | 48.1% |

### Mean relative gap (lower is better)

| Repair | C=2 | C=3 | C=4 | C=5 | C=6 |
|---|---|---|---|---|---|
| collapse | 7.33e-07 | 1.88e-04 | 5.17e-04 | 1.21e-03 | 1.95e-03 |
| distinct | 7.33e-07 | 1.88e-04 | 5.17e-04 | 1.21e-03 | 1.94e-03 |

Overall the **distinct** strategy is marginally better, and the largest difference at any class count is 0.1%.

So the choice is close to immaterial *for these algorithms on this problem* -- which is itself worth reporting, given that papers routinely leave it unstated and a reader cannot tell whether it mattered.
