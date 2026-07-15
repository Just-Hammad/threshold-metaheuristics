import numpy as np
import pandas as pd

from threshmh.stats.compare import (
    cliffs_delta,
    friedman_ranks,
    magnitude,
    pairwise_cliffs,
    posthoc_holm,
)


def test_cliffs_delta_identity_is_zero():
    a = np.array([1.0, 2.0, 3.0, 4.0])
    assert cliffs_delta(a, a) == 0.0


def test_cliffs_delta_total_dominance_is_one():
    assert cliffs_delta([10, 11, 12], [1, 2, 3]) == 1.0
    assert cliffs_delta([1, 2, 3], [10, 11, 12]) == -1.0


def test_cliffs_delta_is_antisymmetric():
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=30), rng.normal(0.5, size=30)
    assert cliffs_delta(a, b) == -cliffs_delta(b, a)


def test_magnitude_labels_match_romano_cutpoints():
    assert magnitude(0.10) == "negligible"
    assert magnitude(0.20) == "small"
    assert magnitude(0.40) == "medium"
    assert magnitude(0.80) == "large"
    assert magnitude(-0.80) == "large"


def test_friedman_ranks_order_best_first():
    # Column 'a' is uniformly smallest, so with lower-is-better it ranks 1.
    df = pd.DataFrame({"a": [1, 1, 1, 1, 1], "b": [2, 2, 2, 2, 2], "c": [3, 3, 3, 3, 3]})
    _, p, ranks = friedman_ranks(df, lower_is_better=True)
    assert list(ranks.index) == ["a", "b", "c"]
    assert ranks.iloc[0] == 1.0
    assert p < 0.05


def test_posthoc_holm_is_labelled_and_symmetric():
    rng = np.random.default_rng(4)
    df = pd.DataFrame({
        "a": rng.normal(0, 1, 20),
        "b": rng.normal(3, 1, 20),
        "c": rng.normal(6, 1, 20),
    })
    out = posthoc_holm(df)
    assert list(out.columns) == ["a", "b", "c"]
    assert np.allclose(out.to_numpy(), out.to_numpy().T, equal_nan=True)


def test_pairwise_cliffs_diagonal_is_zero():
    runs = {"x": np.arange(10.0), "y": np.arange(10.0) + 5}
    out = pairwise_cliffs(runs)
    assert out.loc["x", "x"] == 0.0
    assert out.loc["x", "y"] < 0
