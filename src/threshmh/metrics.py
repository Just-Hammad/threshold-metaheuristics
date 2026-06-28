"""Image-quality and segmentation metrics.

Two families, and the distinction between them is one of the findings this
package is built to surface:

**Reconstruction fidelity** (PSNR, SSIM, FSIM) compares the segmented image
against the *original greyscale*.  These are what the metaheuristic
thresholding literature almost always reports.  They reward preserving
intensity detail, which is not the same thing as segmenting correctly.

**Segmentation agreement** (Dice, IoU) compares the label map against
*ground truth*.  This is the metric that answers the question a practitioner
actually has.

They can disagree, and when they do the reconstruction metrics are the ones
that are wrong about what matters.
"""

from __future__ import annotations

import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from .objectives import class_bounds


def label_map(img: np.ndarray, thresholds) -> np.ndarray:
    """Assign each pixel to a class index via ``np.digitize``."""
    t = np.asarray(thresholds, dtype=np.int64).ravel()
    return np.digitize(img, t, right=True).astype(np.int64)


def reconstruct(img: np.ndarray, thresholds, n_bins: int = 256) -> np.ndarray:
    """Replace each class with its mean intensity.

    The standard reconstruction used for PSNR/SSIM/FSIM in this literature.
    Empty classes keep the class midpoint so the output stays well defined.
    """
    t = np.asarray(thresholds, dtype=np.int64).ravel()
    labels = label_map(img, t)
    out = np.zeros_like(img, dtype=np.float64)
    starts, ends = class_bounds(t, n_bins)
    for k, (a, b) in enumerate(zip(starts, ends)):
        m = labels == k
        out[m] = img[m].mean() if m.any() else 0.5 * (a + b)
    return out


# -- reconstruction fidelity -------------------------------------------------


def psnr(original: np.ndarray, recon: np.ndarray, data_range: float = 255.0) -> float:
    return float(peak_signal_noise_ratio(
        original.astype(np.float64), recon.astype(np.float64), data_range=data_range
    ))


def ssim(original: np.ndarray, recon: np.ndarray, data_range: float = 255.0) -> float:
    return float(structural_similarity(
        original.astype(np.float64), recon.astype(np.float64), data_range=data_range
    ))


def _log_gabor_pc(
    img: np.ndarray,
    n_scale: int = 4,
    n_orient: int = 4,
    min_wavelength: float = 6.0,
    mult: float = 2.0,
    sigma_on_f: float = 0.55,
    k_noise: float = 2.0,
) -> np.ndarray:
    """Phase congruency via a log-Gabor filter bank (Kovesi's construction).

    Returns a non-negative PC map in [0, 1].  This is the standard bank --
    log-Gabor radial times Gaussian angular spread, summed over orientations --
    with a simple fixed noise threshold rather than Kovesi's Rayleigh estimator.
    See the module note in :func:`fsim` on what that means for the numbers.
    """
    rows, cols = img.shape
    F = np.fft.fft2(img.astype(np.float64))

    # Normalised radial frequency grid, DC at the corner to match fft2 layout.
    y, x = np.mgrid[0:rows, 0:cols]
    y = (y - rows // 2) / rows
    x = (x - cols // 2) / cols
    radius = np.fft.ifftshift(np.sqrt(x**2 + y**2))
    theta = np.fft.ifftshift(np.arctan2(-y, x))
    radius[0, 0] = 1.0  # avoid log(0); the DC term is zeroed by the lowpass below

    # Butterworth lowpass to suppress the highest frequencies.
    lp = 1.0 / (1.0 + (radius / 0.45) ** (2 * 15))

    sin_t, cos_t = np.sin(theta), np.cos(theta)
    pc_total = np.zeros((rows, cols))

    for o in range(n_orient):
        angle = o * np.pi / n_orient
        # Angular distance, computed via sin/cos so it wraps correctly.
        ds = sin_t * np.cos(angle) - cos_t * np.sin(angle)
        dc = cos_t * np.cos(angle) + sin_t * np.sin(angle)
        d_theta = np.abs(np.arctan2(ds, dc))
        sigma_theta = (np.pi / n_orient) / 1.5
        spread = np.exp(-(d_theta**2) / (2 * sigma_theta**2))

        sum_e = np.zeros((rows, cols))
        sum_o = np.zeros((rows, cols))
        sum_a = np.zeros((rows, cols))

        for s in range(n_scale):
            wavelength = min_wavelength * (mult**s)
            f0 = 1.0 / wavelength
            log_gabor = np.exp(-((np.log(radius / f0)) ** 2) / (2 * np.log(sigma_on_f) ** 2))
            log_gabor *= lp
            log_gabor[0, 0] = 0.0

            resp = np.fft.ifft2(F * log_gabor * spread)
            sum_e += resp.real
            sum_o += resp.imag
            sum_a += np.abs(resp)

        energy = np.sqrt(sum_e**2 + sum_o**2)
        # Fixed noise floor, proportional to mean amplitude at this orientation.
        T = k_noise * np.mean(sum_a) / max(n_scale, 1)
        pc_total += np.maximum(energy - T, 0.0) / (sum_a + 1e-8)

    pc = pc_total / n_orient
    return np.clip(pc, 0.0, 1.0)


def _gradient_magnitude(img: np.ndarray) -> np.ndarray:
    from scipy.ndimage import convolve

    scharr_x = np.array([[3, 0, -3], [10, 0, -10], [3, 0, -3]], dtype=np.float64) / 16.0
    scharr_y = scharr_x.T
    gx = convolve(img.astype(np.float64), scharr_x, mode="nearest")
    gy = convolve(img.astype(np.float64), scharr_y, mode="nearest")
    return np.sqrt(gx**2 + gy**2)


def fsim(original: np.ndarray, recon: np.ndarray, T1: float = 0.85, T2: float = 160.0) -> float:
    """Feature Similarity Index (Zhang et al., 2011).

    Combines phase-congruency similarity and gradient-magnitude similarity,
    weighted pointwise by ``PCm = max(PC1, PC2)``.

    .. warning::
       This is our own implementation.  It is validated on the *properties* an
       FSIM must have -- ``fsim(x, x) == 1``, monotone decrease under
       increasing degradation, range ``(0, 1]`` -- and it matches the published
       construction, but it has **not** been checked value-for-value against
       Zhang et al.'s reference MATLAB, and the noise-threshold estimator in
       :func:`_log_gabor_pc` is simplified.  Treat absolute FSIM values as
       comparable *within* this package's experiments, not across papers.
    """
    a = original.astype(np.float64)
    b = recon.astype(np.float64)

    pc1, pc2 = _log_gabor_pc(a), _log_gabor_pc(b)
    g1, g2 = _gradient_magnitude(a), _gradient_magnitude(b)

    s_pc = (2 * pc1 * pc2 + T1) / (pc1**2 + pc2**2 + T1)
    s_g = (2 * g1 * g2 + T2) / (g1**2 + g2**2 + T2)
    s_l = s_pc * s_g

    pcm = np.maximum(pc1, pc2)
    denom = pcm.sum()
    if denom <= 0:
        return float("nan")
    return float((s_l * pcm).sum() / denom)


# -- segmentation agreement --------------------------------------------------


def dice(pred_labels: np.ndarray, true_labels: np.ndarray, n_classes: int) -> float:
    """Mean per-class Dice coefficient.

    Classes absent from *both* prediction and truth are skipped rather than
    scored as perfect, which would silently inflate the mean.
    """
    scores = []
    for k in range(n_classes):
        p, t = pred_labels == k, true_labels == k
        denom = p.sum() + t.sum()
        if denom == 0:
            continue
        scores.append(2.0 * np.logical_and(p, t).sum() / denom)
    return float(np.mean(scores)) if scores else float("nan")


def iou(pred_labels: np.ndarray, true_labels: np.ndarray, n_classes: int) -> float:
    """Mean per-class intersection-over-union, same absent-class rule as Dice."""
    scores = []
    for k in range(n_classes):
        p, t = pred_labels == k, true_labels == k
        union = np.logical_or(p, t).sum()
        if union == 0:
            continue
        scores.append(np.logical_and(p, t).sum() / union)
    return float(np.mean(scores)) if scores else float("nan")


def best_permutation_dice(pred_labels, true_labels, n_classes: int) -> float:
    """Mean per-class Dice under the optimal matching of predicted to true labels.

    Thresholding numbers its classes by intensity; a ground-truth annotation
    need not use the same order, so a raw Dice can understate agreement purely
    because of labelling convention.

    Choosing the best matching is a linear assignment problem -- mean Dice is a
    sum of independent per-pair terms -- so the Hungarian algorithm solves it
    exactly in ``O(C^3)``.  An exhaustive search over the ``C!`` permutations
    gives the same answer and is unusable past about eight classes.

    Pairs where prediction and truth are both empty are skipped rather than
    scored as perfect, matching :func:`dice`.
    """
    from scipy.optimize import linear_sum_assignment

    pred_masks = [pred_labels == k for k in range(n_classes)]
    true_masks = [true_labels == k for k in range(n_classes)]
    pred_sizes = [int(m.sum()) for m in pred_masks]
    true_sizes = [int(m.sum()) for m in true_masks]

    overlap = np.full((n_classes, n_classes), np.nan)
    for i in range(n_classes):
        for j in range(n_classes):
            denom = pred_sizes[i] + true_sizes[j]
            if denom:
                overlap[i, j] = 2.0 * np.logical_and(pred_masks[i], true_masks[j]).sum() / denom

    # linear_sum_assignment minimises, so negate; undefined pairs score 0 for
    # the purposes of choosing the matching, then drop out of the mean.
    rows, cols = linear_sum_assignment(-np.nan_to_num(overlap, nan=0.0))
    matched = [overlap[i, j] for i, j in zip(rows, cols) if not np.isnan(overlap[i, j])]
    return float(np.mean(matched)) if matched else float("nan")
