"""Language-agnostic entry point: CSV in, CSV out.

Written for MATLAB R2015a, which predates its Python interface, so the only
portable bridge is a subprocess plus files on disk.  It is equally usable from
R, Julia or a shell script.

Usage::

    python -m threshmh.bridge --image path.png  --classes 4 --criterion otsu --out t.csv
    python -m threshmh.bridge --hist  hist.csv  --classes 4 --criterion kapur --out t.csv

The output CSV has one row per threshold plus a trailing row holding the
criterion value, so a caller that can only read numbers still gets both.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from .exact import exact_thresholds
from .histogram import HistogramStats


def _load_hist(args) -> np.ndarray:
    if args.hist:
        return np.loadtxt(args.hist, delimiter=",").ravel()
    from skimage.io import imread

    img = imread(args.image)
    if img.ndim == 3:
        from skimage.color import rgb2gray
        from skimage.util import img_as_ubyte

        img = img_as_ubyte(rgb2gray(img))
    return np.bincount(np.asarray(img).ravel(), minlength=args.nbins)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="threshmh.bridge")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--image", help="image file readable by skimage.io.imread")
    src.add_argument("--hist", help="CSV of histogram counts, one row")
    p.add_argument("--classes", type=int, required=True)
    p.add_argument("--criterion", default="otsu", choices=["otsu", "kapur"])
    p.add_argument("--nbins", type=int, default=256)
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)

    stats = HistogramStats(_load_hist(args))
    thresholds, value = exact_thresholds(stats, args.classes, args.criterion)

    rows = [*thresholds.tolist(), value]
    Path(args.out).write_text("\n".join(str(v) for v in rows) + "\n")
    print(f"thresholds={thresholds.tolist()} value={value:.10f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
