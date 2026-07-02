"""Experiment 02 -- how the two *exact* methods scale in the number of classes.

Both methods return the same proven optimum; they differ only in how they find
it.  ``skimage.filters.threshold_multiotsu`` searches combinations of
thresholds, with documented complexity ``O(C * h^(C-1) / (C-1)!)``.  The
dynamic program in :mod:`threshmh.exact` is ``O(C * L^2)``, which is
effectively flat in ``C`` at these sizes.

The point is not that scikit-image is badly written -- it is that the
combinatorial framing is what makes exact multilevel thresholding *look*
expensive, and that appearance is the premise the metaheuristic literature
rests on.  Remove it and the premise goes with it.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
from skimage import data, filters

from .exact import exact_thresholds
from .histogram import HistogramStats
from .objectives import score

# Above this, threshold_multiotsu's combinatorial search runs for tens of
# minutes; the trend is already unambiguous by C=6.
DEFAULT_SKIMAGE_MAX = 6


def _time(fn, repeats: int = 3, warmup: bool = True) -> tuple[float, float, object]:
    """Best-of-N timing in milliseconds, as ``(cpu_ms, wall_ms, result)``.

    Both clocks are recorded because this measurement was taken on a machine
    that was not exclusively idle.  ``time.process_time`` counts only CPU time
    actually charged to this process, so it is largely unaffected by unrelated
    load; wall-clock is reported alongside it for honesty and because it is
    what a user experiences.  The minimum over repeats is used, as it is the
    estimate least polluted by scheduling interference.

    ``warmup`` runs one untimed call first so import and JIT costs do not land
    in the measurement.  Disable it when a single call already costs minutes --
    the fixed overhead is then irrelevant and a warm-up doubles the runtime for
    nothing.
    """
    out = fn() if warmup else None
    best_cpu = best_wall = float("inf")
    for _ in range(repeats):
        c0, w0 = time.process_time(), time.perf_counter()
        out = fn()
        best_cpu = min(best_cpu, time.process_time() - c0)
        best_wall = min(best_wall, time.perf_counter() - w0)
    return best_cpu * 1e3, best_wall * 1e3, out


def run_scaling(
    out_path: Path,
    image_name: str = "camera",
    max_classes: int = 8,
    skimage_max: int = DEFAULT_SKIMAGE_MAX,
) -> pd.DataFrame:
    img = getattr(data, image_name)()
    stats = HistogramStats(np.bincount(img.ravel(), minlength=256))

    rows = []
    for C in range(2, max_classes + 1):
        # Best-of-many: the DP runs in milliseconds, so a single scheduling
        # hiccup can inflate a measurement by an order of magnitude. Cheap to
        # repeat, and the minimum is the estimate least polluted by contention.
        dp_cpu, dp_wall, (t_dp, v_dp) = _time(
            lambda c=C: exact_thresholds(stats, c, "otsu"), repeats=15
        )

        if C <= skimage_max:
            # One timed call, no warm-up: at C=6 this already costs minutes,
            # and the module was warmed by the C=2 call.
            sk_cpu, sk_wall, t_sk = _time(
                lambda c=C: filters.threshold_multiotsu(image=img, classes=c, nbins=256),
                repeats=1,
                warmup=(C == 2),
            )
            v_sk = score(stats, np.asarray(t_sk, dtype=np.int64), "otsu")
            agree = bool(np.isclose(v_dp, v_sk, rtol=1e-9))
        else:
            sk_cpu = sk_wall = v_sk = float("nan")
            agree = None

        rows.append({
            "n_classes": C,
            "dp_ms": dp_cpu,
            "dp_wall_ms": dp_wall,
            "skimage_ms": sk_cpu,
            "skimage_wall_ms": sk_wall,
            "dp_value": v_dp,
            "skimage_value": v_sk,
            "values_agree": agree,
            "speedup": sk_cpu / dp_cpu if np.isfinite(sk_cpu) else float("nan"),
            "dp_thresholds": "-".join(map(str, t_dp.tolist())),
        })
        print(f"[scaling] C={C}  DP {dp_cpu:8.2f} ms cpu  skimage {sk_cpu:12.2f} ms cpu  agree={agree}")

    df = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"[scaling] wrote {out_path}")
    return df
