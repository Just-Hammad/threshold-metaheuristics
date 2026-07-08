Both methods return the **same proven optimum** -- they differ only in how they find it.

CPU time (`time.process_time`) is primary; the machine was not exclusively idle and CPU
time is largely insensitive to unrelated load. Wall-clock is reported alongside it.

| Classes | DP CPU (ms) | DP wall (ms) | skimage CPU (ms) | skimage wall (ms) | Speed-up (CPU) | Optima agree |
|---|---|---|---|---|---|---|
| 2 | 3.35 | 3.47 | 1.2 | 1.2 | 0x | yes |
| 3 | 4.88 | 5.19 | 2.2 | 2.2 | 0x | yes |
| 4 | 4.93 | 5.71 | 73.6 | 289.0 | 15x | yes |
| 5 | 6.38 | 6.90 | 5,003.4 | 19,686.5 | 784x | yes |
| 6 | 6.57 | 7.00 | 269,033.8 | 1,243,476.7 | 40,968x | yes |
| 7 | 6.83 | 6.91 | not run | not run | -- | not run |
| 8 | 7.78 | 9.11 | not run | not run | -- | not run |

The DP is flat in C: 3.4-7.8 ms CPU across C=2..8, consistent with its `O(C x L^2)` bound.
`threshold_multiotsu` searches combinations, with documented complexity
`O(C x h^(C-1) / (C-1)!)`, so its cost rises steeply with C.

**This is not a defect in scikit-image** -- its complexity is documented, and for the
two- and three-class cases it is used for overwhelmingly, it is perfectly adequate.
It matters here because that combinatorial cost is what makes exact multilevel
thresholding appear expensive, which is the premise the metaheuristic literature rests on.

