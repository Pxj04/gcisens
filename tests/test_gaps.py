"""Test-gap sweep from the 2026-09-02 audit (Package E, item E9)."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

import gcisens
from gcisens import SobolStudy, compare, esp_spotis
from gcisens.diagnosis import (
    CONFIRMED_TRANSPARENCY,
    HIDDEN_INFLUENCE,
    DiagnosisThresholds,
    classify,
    sweep_thresholds,
)

BOUNDS = np.array([[0.0, 10.0], [0.0, 5.0]])


# ------------------------------------------------------------------ LaTeX
def test_s2_to_latex_limits_rows_and_writes_the_file(tmp_path, record):
    path = tmp_path / "s2.tex"
    text = gcisens.s2_to_latex(record, path, top=2, label="tab:custom")

    assert path.read_text() == text
    assert r"\label{tab:custom}" in text
    assert r"Top 2 second-order Sobol' interaction indices" in text
    data_rows = [line for line in text.splitlines() if line.endswith(r"\\") and " & " in line]
    assert len(data_rows) == 1 + 2  # header + top-2 pairs
    assert data_rows[1].startswith("A & B & 0.0800")


def test_comparison_to_latex_matches_the_method_and_writes_the_file(tmp_path, record):
    comparison = compare({"one": record, "two": record})
    path = tmp_path / "comparison.tex"

    text = comparison.to_latex(path, caption="Caption", label="tab:cmp")

    assert path.read_text() == text
    assert text == gcisens.comparison_to_latex(comparison, caption="Caption", label="tab:cmp")
    assert r"\textbf{Metric} & \textbf{one} & \textbf{two} \\" in text
    assert r"\caption{Caption}" in text
    assert r"\label{tab:cmp}" in text
    assert text.count("\n") + 1 == 8 + len(record.metrics) + 3


# ----------------------------------------------------------------- SPOTIS
def test_spotis_local_weights_at_a_reference_point_equal_declared_weights_for_corner_esp():
    # Score = sum_i w_i |x_i - esp_i| / span_i. With the ESP at the lower
    # corner every one-criterion sweep spans the same normalised distance, so
    # the local weights are the declared weights.
    weights = np.array([0.7, 0.3])
    model = esp_spotis(esp=[0.0, 0.0], bounds=BOUNDS, weights=weights, criteria_names=["A", "B"])

    result = SobolStudy(model, n_samples=32, seed=0).run(reference_point=[5.0, 2.5])

    assert [v.key for v in result.views] == ["w", "w_loc", "S1", "ST"]
    np.testing.assert_allclose(result.local_weights, weights, atol=1e-9)
    assert result.local_weights.sum() == pytest.approx(1.0)
    assert result.correlations["rho_w_wloc"] == pytest.approx(1.0)


def test_spotis_types_do_not_change_scores_when_an_esp_is_set():
    profit = esp_spotis(esp=[7.0, 2.0], bounds=BOUNDS, weights=[0.6, 0.4], types=[1, 1])
    cost = esp_spotis(esp=[7.0, 2.0], bounds=BOUNDS, weights=[0.6, 0.4], types=[1, -1])
    rng = np.random.default_rng(0)
    X = rng.uniform(BOUNDS[:, 0], BOUNDS[:, 1], size=(100, 2))

    np.testing.assert_allclose(
        SobolStudy(profit).adapter.scores(X), SobolStudy(cost).adapter.scores(X)
    )


# ----------------------------------------------------------- construction
def test_callable_adapter_rejects_a_non_callable_model():
    with pytest.raises(TypeError, match="Unsupported model type: int"):
        SobolStudy(42, bounds=BOUNDS, n_samples=8)


def test_sampler_is_validated_at_construction_not_at_run():
    with pytest.raises(ValueError, match="sampler must be one of"):
        SobolStudy(lambda X: X[:, 0], bounds=BOUNDS, n_samples=8, sampler="latin")
    study = SobolStudy(lambda X: X[:, 0], bounds=BOUNDS, n_samples=8, sampler="sobol")
    assert study.sampler == "sobol"


# ------------------------------------------------------ dataclass equality
def test_records_holding_arrays_compare_by_identity(record):
    for value in (record.sobol, record.validation, record):
        same, copied = value, replace(value)
        assert value == same
        assert value != copied  # a field-wise copy, not the same object
    assert {record.sobol, record.validation, record}  # hashable


# ------------------------------------------------------ threshold sweeps
def test_sweep_thresholds_returns_one_row_per_grid_point():
    names = ["C1", "C2", "C3", "C4"]
    w = np.array([0.5, 0.3, 0.19, 0.01])
    s1 = np.array([0.45, 0.28, 0.17, 0.06])
    st = np.array([0.46, 0.29, 0.18, 0.07])

    table = sweep_thresholds(
        names, w, s1, st, hidden_st_excess=[0.03, 0.10], interaction_ratio=[0.3, 0.5, 0.9]
    )

    assert isinstance(table, pd.DataFrame)
    assert list(table.columns) == ["hidden_st_excess", "interaction_ratio", *names]
    assert len(table) == 6
    assert list(table["hidden_st_excess"]) == [0.03, 0.03, 0.03, 0.10, 0.10, 0.10]
    # C4: ST - w = 0.06 clears 0.03 but not 0.10.
    assert set(table.loc[table["hidden_st_excess"] == 0.03, "C4"]) == {HIDDEN_INFLUENCE}
    assert set(table.loc[table["hidden_st_excess"] == 0.10, "C4"]) == {CONFIRMED_TRANSPARENCY}
    # Every cell is a Category constant, so the table keys on the same objects.
    assert all(isinstance(c, gcisens.Category) for c in table[names].to_numpy().ravel())


def test_sweep_thresholds_starts_from_the_base_thresholds():
    names = ["C1", "C2", "C3", "C4"]
    w = np.array([0.4, 0.3, 0.2, 0.1])
    s1 = np.array([0.30, 0.29, 0.21, 0.09])
    st = np.array([0.45, 0.30, 0.21, 0.10])
    base = DiagnosisThresholds(interaction_abs=0.5)  # blocks interaction dominance

    table = sweep_thresholds(names, w, s1, st, base=base, interaction_ratio=[0.1])

    expected = [
        d.category for d in classify(names, w, s1, st, replace(base, interaction_ratio=0.1))
    ]
    assert list(table.loc[0, names]) == expected
    assert list(table.columns) == ["interaction_ratio", *names]


def test_sweep_thresholds_rejects_unknown_threshold_names():
    with pytest.raises(TypeError, match="not_a_threshold"):
        sweep_thresholds(["A"], [1.0], [1.0], [1.0], not_a_threshold=[1])
    with pytest.raises(ValueError, match="at least one threshold"):
        sweep_thresholds(["A"], [1.0], [1.0], [1.0])


def test_study_result_sweeps_its_own_views(record):
    table = record.sweep_thresholds(hidden_st_excess=[0.03, 0.2])

    assert list(table.columns) == ["hidden_st_excess", *record.criteria_names]
    assert list(table.loc[0, record.criteria_names]) == [d.category for d in record.diagnoses]
    assert "sweep_thresholds" in gcisens.__all__
