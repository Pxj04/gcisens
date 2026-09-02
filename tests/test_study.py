import json

import numpy as np
import pandas as pd
import pytest

import gcisens
from gcisens import COMET, SPOTIS, ESPExpert, SobolStudy, compare, esp_comet, esp_spotis


def test_builder_equals_manual_pymcdm_construction(hr_setup):
    criteria, bounds, esp1, _ = hr_setup
    built = esp_comet(esps=esp1, bounds=bounds, criteria_names=criteria)

    expert = ESPExpert(esps=esp1, bounds=bounds)
    manual = COMET(expert.make_cvalues_psi(), expert)

    rng = np.random.default_rng(0)
    X = rng.uniform(bounds[:, 0], bounds[:, 1], size=(50, len(criteria)))
    np.testing.assert_allclose(built(X), manual(X))


def test_manual_pymcdm_model_works_in_study(hr_setup):
    criteria, bounds, esp1, _ = hr_setup
    expert = ESPExpert(esps=esp1, bounds=bounds)
    model = COMET(expert.make_cvalues_psi(), expert)
    result = SobolStudy(model, bounds=bounds, criteria_names=criteria, n_samples=64, seed=0).run()
    assert result.table().shape[0] == len(criteria)


def test_comet_bounds_recovered_from_cvalues(hr_setup):
    _criteria, bounds, esp1, _ = hr_setup
    expert = ESPExpert(esps=esp1, bounds=bounds)
    model = COMET(expert.make_cvalues_psi(), expert)
    result = SobolStudy(model, n_samples=64, seed=0).run()  # no bounds given
    np.testing.assert_allclose(result.adapter.bounds, bounds)


def test_spotis_study_uses_declared_weights(hr_setup):
    criteria, bounds, esp1, _ = hr_setup
    w = np.array([0.3, 0.1, 0.2, 0.1, 0.1, 0.1, 0.1])
    model = esp_spotis(esp=esp1[0], bounds=bounds, weights=w, criteria_names=criteria)
    result = SobolStudy(model, n_samples=64, seed=0).run()
    np.testing.assert_allclose(result.weights, w)
    assert result.weights_source == "declared"
    assert not result.adapter.higher_is_closer


def test_spotis_without_weights_raises(hr_setup):
    _, bounds, esp1, _ = hr_setup
    model = SPOTIS(bounds, esp=esp1[0])
    with pytest.raises(ValueError, match="weights"):
        SobolStudy(model, n_samples=64)


def test_callable_model_fallback(linear_model):
    score, bounds = linear_model
    result = SobolStudy(
        score, bounds=bounds, criteria_names=["A", "B"], n_samples=256, seed=0
    ).run()
    # Without declared weights the study reports regression-based ones.
    assert result.weights_source == "regression (samples)"
    np.testing.assert_allclose(result.weights, [0.7, 0.3], atol=0.01)
    assert result.r2 > 0.99


def test_callable_without_bounds_raises(linear_model):
    score, _ = linear_model
    with pytest.raises(ValueError, match="bounds"):
        SobolStudy(score, n_samples=64)


def test_validation_lift_and_orientation(linear_model):
    score, bounds = linear_model
    result = SobolStudy(
        score, bounds=bounds, weights=np.array([0.7, 0.3]), n_samples=64, seed=0
    ).run()
    rng = np.random.default_rng(3)
    X = rng.uniform(0, 10, size=(200, 2))
    labels = score(X) > np.median(score(X))
    val = result.validate(X, labels, top_k=[20])
    assert val.delta_mean > 0
    assert val.lift.loc[0, "lift"] > 1.5


def test_compare_table(linear_model):
    score, bounds = linear_model
    res = SobolStudy(score, bounds=bounds, weights=np.array([0.7, 0.3]), n_samples=64, seed=0).run()
    cmp = compare({"a": res, "b": res})
    table = cmp.table()
    assert list(table.columns) == ["a", "b"]
    assert "R2" in table.index


def test_exports_roundtrip(tmp_path, linear_model):
    score, bounds = linear_model
    result = SobolStudy(
        score,
        bounds=bounds,
        weights=np.array([0.7, 0.3]),
        n_samples=64,
        seed=0,
        thresholds=gcisens.DiagnosisThresholds(hidden_weight_factor=0.25),
    ).run(reference_point=[5, 5])
    files = result.to_csv(tmp_path)
    assert all(f.exists() for f in files)
    main = pd.read_csv(tmp_path / "results_main.csv")
    assert list(main["Criterion"]) == ["C1", "C2"]

    summary = pd.read_csv(tmp_path / "results_summary.csv", index_col=0)["value"]
    assert json.loads(summary["thresholds"])["hidden_weight_factor"] == 0.25
    assert json.loads(summary["reference_point"]) == [5.0, 5.0]

    s2_matrix = pd.read_csv(tmp_path / "results_s2_matrix.csv", index_col=0)
    assert list(s2_matrix.index) == ["C1", "C2"]
    assert list(s2_matrix.columns) == ["C1", "C2"]
    assert s2_matrix.loc["C1", "C2"] == s2_matrix.loc["C2", "C1"]

    latex = result.to_latex(tmp_path / "table.tex")
    assert latex.startswith(r"\begin{table}")
    assert (tmp_path / "table.tex").exists()

    report = result.to_html(tmp_path / "report.html")
    text = report.read_text()
    assert "Discrepancy diagnosis" in text


def test_latex_exports_escape_user_text(linear_model):
    score, bounds = linear_model
    result = SobolStudy(
        score,
        bounds=bounds,
        criteria_names=["Rate_%", "Cost & Fees"],
        weights=np.array([0.7, 0.3]),
        n_samples=64,
        seed=0,
    ).run()

    caption = r"Report \ & % $ # _ { } ~ ^"
    expected_caption = (
        r"Report \textbackslash{} \& \% \$ \# \_ \{ \} "
        r"\textasciitilde{} \textasciicircum{}"
    )
    main = result.to_latex(caption=caption)
    interactions = gcisens.s2_to_latex(result, caption=caption)
    comparison = gcisens.comparison_to_latex(compare({"ESP_1 & 2": result}), caption=caption)

    for latex in (main, interactions, comparison):
        assert rf"\caption{{{expected_caption}}}" in latex
    assert r"Rate\_\%" in main
    assert r"Cost \& Fees" in main
    assert r"Rate\_\%" in interactions
    assert r"Cost \& Fees" in interactions
    assert r"\textbf{ESP\_1 \& 2}" in comparison


def test_html_export_keeps_matplotlib_backend(tmp_path, linear_model):
    import matplotlib

    score, bounds = linear_model
    result = SobolStudy(
        score, bounds=bounds, weights=np.array([0.7, 0.3]), n_samples=64, seed=0
    ).run()
    original_backend = matplotlib.get_backend()
    try:
        matplotlib.use("svg")
        report = result.to_html(tmp_path / "report.html")
        assert matplotlib.get_backend().lower() == "svg"
        text = report.read_text()
        assert "<img" in text
        assert "Plots skipped" not in text
    finally:
        matplotlib.use(original_backend)


def test_html_export_can_skip_plots(tmp_path, linear_model):
    score, bounds = linear_model
    result = SobolStudy(
        score, bounds=bounds, weights=np.array([0.7, 0.3]), n_samples=64, seed=0
    ).run()

    report = result.to_html(tmp_path / "report.html", include_plots=False)

    assert "<img" not in report.read_text()


def test_html_export_warns_when_plots_fail(tmp_path):
    def score(X):
        return np.atleast_2d(X)[:, 0]

    result = SobolStudy(
        score,
        bounds=np.array([[0.0, 1.0]]),
        weights=np.array([1.0]),
        n_samples=64,
        second_order=False,
        seed=0,
    ).run()

    with pytest.warns(UserWarning, match="Plots skipped: ValueError"):
        report = result.to_html(tmp_path / "report.html")

    assert "Plots skipped" in report.read_text()


def test_latex_exports_are_available_through_public_api(tmp_path, linear_model):
    score, bounds = linear_model
    result = SobolStudy(
        score, bounds=bounds, weights=np.array([0.7, 0.3]), n_samples=64, seed=0
    ).run()

    assert "s2_to_latex" in gcisens.__all__
    assert "comparison_to_latex" in gcisens.__all__
    assert result.s2_to_latex(tmp_path / "s2.tex") == gcisens.s2_to_latex(result)
    assert (tmp_path / "s2.tex").exists()


def test_reexports_are_pymcdm_classes():
    from pymcdm.methods import COMET as PymcdmCOMET

    assert gcisens.COMET is PymcdmCOMET
