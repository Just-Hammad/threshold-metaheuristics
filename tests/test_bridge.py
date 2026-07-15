"""The CSV bridge must produce exactly what the Python API produces.

MATLAB R2015a has no Python interface, so this subprocess-and-files path is the
only way Tawhid's group can call the solver. If it drifts from the library, the
MATLAB caller silently gets different answers.
"""

import numpy as np
import pytest
from skimage import data

from threshmh.bridge import main as bridge_main
from threshmh.exact import exact_thresholds
from threshmh.histogram import HistogramStats


@pytest.mark.parametrize("criterion", ["otsu", "kapur"])
@pytest.mark.parametrize("C", [2, 4, 6])
def test_bridge_matches_library(tmp_path, criterion, C):
    img = data.coins()
    counts = np.bincount(img.ravel(), minlength=256)

    hist_csv = tmp_path / "hist.csv"
    out_csv = tmp_path / "out.csv"
    np.savetxt(hist_csv, counts[None, :], delimiter=",", fmt="%d")

    rc = bridge_main([
        "--hist", str(hist_csv), "--classes", str(C),
        "--criterion", criterion, "--out", str(out_csv),
    ])
    assert rc == 0

    rows = [float(x) for x in out_csv.read_text().split()]
    got_t, got_v = np.array(rows[:-1], dtype=np.int64), rows[-1]

    want_t, want_v = exact_thresholds(HistogramStats(counts), C, criterion)
    assert np.array_equal(got_t, want_t)
    assert got_v == pytest.approx(want_v, rel=1e-12)


def test_bridge_rejects_missing_source(tmp_path):
    with pytest.raises(SystemExit):
        bridge_main(["--classes", "3", "--out", str(tmp_path / "x.csv")])
