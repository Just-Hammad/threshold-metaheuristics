"""Image sources.

Two kinds, for two different questions:

* **Real images** from ``skimage.data`` -- used for the optimisation-quality
  and reconstruction-fidelity comparisons.  No ground truth, so no Dice.
* **Synthetic phantoms** with known labels -- used for the segmentation-accuracy
  comparison.  Because the true class map is constructed rather than annotated,
  Dice is exact, which is what makes the PSNR-vs-Dice disagreement measurable
  rather than arguable.
"""

from __future__ import annotations

import numpy as np
from skimage import data, img_as_ubyte
from skimage.color import rgb2gray

# (name, loader) -- kept as callables so an unavailable download fails for one
# image rather than for the whole experiment.
_REAL = {
    "camera": lambda: data.camera(),
    "coins": lambda: data.coins(),
    "page": lambda: data.page(),
    "moon": lambda: data.moon(),
    "text": lambda: data.text(),
    "cell": lambda: data.cell(),
    "human_mitosis": lambda: data.human_mitosis(),
    "microaneurysms": lambda: data.microaneurysms(),
    "retina": lambda: data.retina(),
}


def _to_gray_u8(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3:
        img = rgb2gray(img)
    return img_as_ubyte(img)


def load_real(names=None) -> dict[str, np.ndarray]:
    """Load real greyscale images, skipping any that fail to fetch."""
    names = list(_REAL) if names is None else list(names)
    out = {}
    for n in names:
        try:
            out[n] = _to_gray_u8(_REAL[n]())
        except Exception as exc:  # noqa: BLE001 - a missing image must not abort the run
            print(f"[datasets] skipping {n!r}: {type(exc).__name__}: {exc}")
    return out


def make_phantom(
    n_classes: int, size: int = 256, noise_sigma: float = 12.0, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """A multi-class phantom with known ground-truth labels.

    Built from overlapping random blobs at distinct intensity plateaus, then
    blurred slightly and corrupted with Gaussian noise.  Returns
    ``(image_u8, true_labels)``.

    The plateaus are evenly spaced across the intensity range, so a *correct*
    thresholding recovers the labels exactly and any Dice shortfall is the
    method's, not the phantom's.
    """
    from scipy.ndimage import gaussian_filter

    rng = np.random.default_rng(seed)
    labels = np.zeros((size, size), dtype=np.int64)

    # Smooth random fields, thresholded at evenly spaced quantiles, give
    # connected regions with realistic boundaries.
    field = gaussian_filter(rng.standard_normal((size, size)), sigma=size / 16.0)
    qs = np.quantile(field, np.linspace(0, 1, n_classes + 1)[1:-1])
    labels = np.digitize(field, qs)

    levels = np.linspace(30, 225, n_classes)
    clean = levels[labels]
    blurred = gaussian_filter(clean, sigma=1.0)
    noisy = blurred + rng.normal(0, noise_sigma, size=(size, size))
    return np.clip(noisy, 0, 255).astype(np.uint8), labels


def load_phantoms(class_counts, seed: int = 0) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """One phantom per class count, each with its ground-truth label map."""
    return {
        f"phantom_c{c}": make_phantom(int(c), seed=seed + int(c))
        for c in class_counts
    }
