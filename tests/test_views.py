"""Views (w, w_loc, S1, ST) decide once what the table, the LaTeX export and
the ranking plot show, and they share one rank definition with the diagnosis."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from gcisens import MODERATE_DISCREPANCY, SobolStudy

BOUNDS_4 = np.array([[0.0, 10.0]] * 4)


def additive(coefs):
    coefs = np.asarray(coefs, dtype=float)

    def score(X):
        return np.atleast_2d(X) @ coefs / 10

    return score


def run(score, bounds, weights, reference_point=None, n_samples=64):
    study = SobolStudy(score, bounds=bounds, weights=np.array(weights), n_samples=n_samples, seed=0)
    return study.run(reference_point=reference_point)


def keys(result):
    return [v.key for v in result.views]


def test_reference_point_adds_the_local_view(linear_model):
    score, bounds = linear_model
    without = run(score, bounds, [0.7, 0.3])
    with_ref = run(score, bounds, [0.7, 0.3], reference_point=[5, 5])

    assert keys(without) == ["w", "S1", "ST"]
    assert keys(with_ref) == ["w", "w_loc", "S1", "ST"]
    assert without.local_weights is None
    np.testing.assert_allclose(with_ref.local_weights, [0.7, 0.3], atol=1e-6)
    assert "rho_w_wloc" not in without.correlations
    assert "rho_w_wloc" in with_ref.correlations


@pytest.mark.parametrize("reference_point", [None, [5, 5]])
def test_table_latex_and_ranking_plot_walk_the_same_views(linear_model, reference_point):
    score, bounds = linear_model
    result = run(score, bounds, [0.7, 0.3], reference_point)
    views = result.views
    n = len(views)

    columns = list(result.table().columns)
    assert columns[1 : 1 + n] == [v.key for v in views]
    assert [c for c in columns if c.startswith("Rank_")] == [f"Rank_{v.key}" for v in views]

    header = result.to_latex().splitlines()[6]
    cells = [c.strip() for c in header.split("&")]
    assert cells[1 : 1 + n] == [v.label for v in views]

    ax = result.plot_rankings()
    assert [t.get_text() for t in ax.get_xticklabels()] == [v.label for v in views]
    plt.close(ax.figure)


def test_view_values_and_ranks_are_the_table_columns(linear_model):
    score, bounds = linear_model
    result = run(score, bounds, [0.7, 0.3], reference_point=[5, 5])
    table = result.table()
    for v in result.views:
        np.testing.assert_array_equal(table[v.key], v.values)
        np.testing.assert_array_equal(table[f"Rank_{v.key}"], v.ranks)


def test_tied_weights_share_one_rank_in_table_and_diagnosis():
    result = run(additive([0.4, 0.3, 0.2, 0.1]), BOUNDS_4, [0.25] * 4, n_samples=256)
    table = result.table()
    np.testing.assert_array_equal(table["Rank_w"], [2.5] * 4)
    np.testing.assert_array_equal(table["Rank_S1"], [1, 2, 3, 4])
    # Max displacement from 2.5 is 1.5 < 2: no arbitrary "moderate discrepancy".
    assert MODERATE_DISCREPANCY not in set(table["Category"])


def test_displacement_in_the_diagnosis_is_the_one_in_the_table():
    result = run(additive([0.4, 0.3, 0.2, 0.1]), BOUNDS_4, [0.15, 0.2, 0.3, 0.35], n_samples=256)
    table = result.table()
    detail = result.diagnosis().loc[0, "Detail"]
    rank_w, rank_s1 = table.loc[0, "Rank_w"], table.loc[0, "Rank_S1"]

    assert table.loc[0, "Category"] == MODERATE_DISCREPANCY
    assert (rank_w, rank_s1) == (4, 1)
    assert f"rank(w)={rank_w:g} vs rank(S1)={rank_s1:g}" in detail


def test_ranks_dict_is_a_view_over_views(linear_model):
    score, bounds = linear_model
    result = run(score, bounds, [0.7, 0.3], reference_point=[5, 5])
    assert list(result.ranks) == keys(result)
    for v in result.views:
        np.testing.assert_array_equal(result.ranks[v.key], v.ranks)
