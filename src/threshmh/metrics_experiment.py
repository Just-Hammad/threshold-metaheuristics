"""Experiment 03 -- do the conventional metrics agree with segmentation quality?

The metaheuristic thresholding literature reports PSNR, SSIM and FSIM.  All
three compare the segmented image against the *original greyscale*, so they
reward preserving intensity detail.  None of them asks whether the pixels were
assigned to the right class.

On the synthetic phantoms the true class map is known, so Dice can be computed
exactly.  This experiment takes the *proven optimal* thresholds for both
criteria -- removing the optimizer from the question entirely -- and asks
whether the metric that the literature reports and the metric a practitioner
cares about ever pick different winners.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .datasets import load_phantoms, load_real
from .exact import exact_thresholds
from .histogram import HistogramStats
from .metrics import best_permutation_dice, fsim, iou, label_map, psnr, reconstruct, ssim

CLASS_COUNTS = (2, 3, 4, 5, 6)
MAX_SIDE = 512  # cap for the image-domain metrics only; the objective is unaffected


def _downscale(img: np.ndarray) -> np.ndarray:
    """FSIM runs 32 FFTs per comparison; cap the side length to keep it tractable."""
    if max(img.shape) <= MAX_SIDE:
        return img
    from skimage.transform import resize

    scale = MAX_SIDE / max(img.shape)
    shape = (int(img.shape[0] * scale), int(img.shape[1] * scale))
    return (resize(img, shape, preserve_range=True, anti_aliasing=True)).astype(np.uint8)


def run_metrics_experiment(out_path: Path) -> pd.DataFrame:
    rows = []

    phantoms = load_phantoms(CLASS_COUNTS)
    reals = load_real()

    # Phantoms: ground truth available, so Dice and IoU are exact.
    for name, (img, truth) in phantoms.items():
        n_true = int(truth.max()) + 1
        small = _downscale(img)
        stats = HistogramStats(np.bincount(img.ravel(), minlength=256))
        for criterion in ("otsu", "kapur"):
            t, v = exact_thresholds(stats, n_true, criterion)
            recon = reconstruct(small, t)
            rows.append({
                "image": name, "kind": "phantom", "criterion": criterion,
                "n_classes": n_true, "optimal_score": v,
                "thresholds": "-".join(map(str, t.tolist())),
                "psnr": psnr(small, recon), "ssim": ssim(small, recon),
                "fsim": fsim(small, recon),
                "dice": best_permutation_dice(label_map(img, t), truth, n_true),
                "iou": iou(label_map(img, t), truth, n_true),
            })

    # Real images: no ground truth, so reconstruction fidelity only.
    for name, img in reals.items():
        small = _downscale(img)
        stats = HistogramStats(np.bincount(img.ravel(), minlength=256))
        for criterion in ("otsu", "kapur"):
            for C in CLASS_COUNTS:
                t, v = exact_thresholds(stats, C, criterion)
                recon = reconstruct(small, t)
                rows.append({
                    "image": name, "kind": "real", "criterion": criterion,
                    "n_classes": C, "optimal_score": v,
                    "thresholds": "-".join(map(str, t.tolist())),
                    "psnr": psnr(small, recon), "ssim": ssim(small, recon),
                    "fsim": fsim(small, recon),
                    "dice": float("nan"), "iou": float("nan"),
                })
        print(f"[metrics] {name} done")

    df = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"[metrics] wrote {len(df)} rows to {out_path}")
    return df
