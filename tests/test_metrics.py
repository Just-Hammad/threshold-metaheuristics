import numpy as np
import pytest
from skimage import data

from threshmh.metrics import (
    best_permutation_dice,
    dice,
    fsim,
    iou,
    label_map,
    psnr,
    reconstruct,
    ssim,
)


@pytest.fixture
def img():
    return data.camera()


def test_label_map_assigns_expected_classes():
    img = np.array([[0, 100, 200]], dtype=np.uint8)
    assert list(label_map(img, [50, 150]).ravel()) == [0, 1, 2]


def test_reconstruct_uses_class_means(img):
    r = reconstruct(img, [128])
    assert set(np.unique(r)).issubset({img[img <= 128].mean(), img[img > 128].mean()})


def test_identity_metrics_are_perfect(img):
    assert np.isinf(psnr(img, img.astype(np.float64)))
    assert ssim(img, img.astype(np.float64)) == pytest.approx(1.0, abs=1e-9)
    assert fsim(img, img) == pytest.approx(1.0, abs=1e-6)


def test_fsim_is_bounded_and_degrades_monotonically(img):
    """More classes means a closer reconstruction, so FSIM must not fall."""
    vals = [fsim(img, reconstruct(img, t)) for t in ([128], [85, 170], [64, 128, 192])]
    assert all(0.0 < v <= 1.0 for v in vals)
    assert vals[0] <= vals[1] <= vals[2] + 1e-9


def test_dice_and_iou_identity():
    rng = np.random.default_rng(0)
    lab = rng.integers(0, 4, size=(64, 64))
    assert dice(lab, lab, 4) == pytest.approx(1.0)
    assert iou(lab, lab, 4) == pytest.approx(1.0)


def test_dice_skips_classes_absent_from_both():
    """A class nobody predicts and nobody has must not count as a free win."""
    a = np.zeros((8, 8), dtype=int)
    assert dice(a, a, n_classes=5) == pytest.approx(1.0)


def test_best_permutation_dice_recovers_relabelling():
    rng = np.random.default_rng(1)
    truth = rng.integers(0, 3, size=(32, 32))
    swapped = np.select([truth == 0, truth == 1, truth == 2], [2, 0, 1])
    assert dice(swapped, truth, 3) < 0.5
    assert best_permutation_dice(swapped, truth, 3) == pytest.approx(1.0)
