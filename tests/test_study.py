import numpy as np
import pytest

import gcisens
from gcisens import COMET, SPOTIS, ESPExpert, SobolStudy, compare, esp_comet, esp_spotis
from gcisens.adapters import CometAdapter


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


def test_comet_adapter_weights_fit_has_no_call_order_dependency(hr_setup):
    criteria, bounds, esp1, _ = hr_setup
    model = esp_comet(esps=esp1, bounds=bounds, criteria_names=criteria)
    adapter = CometAdapter(model)

    declared = adapter.declared_weights()
    fit = adapter.weights_fit

    assert declared.r2 == fit.r2
    assert adapter.weights_fit is fit
    np.testing.assert_allclose(declared.weights, fit.weights)


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
    # The reported weights come from the sample fit, so both R^2 are that fit.
    assert result.r2_samples > 0.99
    assert result.r2_fit == result.r2_samples
    assert result.r2 == result.r2_fit  # gcisens <= 0.1.3 compatibility alias


def test_study_reports_bootstrap_configuration(linear_model):
    score, bounds = linear_model

    result = SobolStudy(
        score,
        bounds=bounds,
        weights=np.array([0.7, 0.3]),
        n_samples=64,
        num_resamples=25,
        conf_level=0.9,
        seed=0,
    ).run()

    assert result.sobol.num_resamples == 25
    assert result.sobol.conf_level == 0.9
    assert result.summary()["num_resamples"] == 25
    assert result.summary()["conf_level"] == 0.9


def test_study_s2_table_uses_diagnosis_thresholds(linear_model):
    score, bounds = linear_model
    thresholds = gcisens.DiagnosisThresholds(s2_significance_factor=2.0)
    result = SobolStudy(
        score,
        bounds=bounds,
        weights=np.array([0.7, 0.3]),
        thresholds=thresholds,
        n_samples=64,
        seed=0,
    ).run()
    result.sobol.S2[0, 1] = 0.03
    result.sobol.S2_conf[0, 1] = 0.02

    assert result.sobol.s2_pairs().loc[0, "significant"]
    assert not result.s2_table().loc[0, "significant"]


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
    study = SobolStudy(score, bounds=bounds, weights=np.array([0.7, 0.3]), n_samples=64, seed=0)
    cmp = compare({"a": study.run(), "b": study.run(reference_point=[5, 5])})
    table = cmp.table()
    assert list(table.columns) == ["a", "b"]
    assert "R2" not in table.index
    assert "r2_samples" in table.index
    assert "rho_S1_ST" in table.index
    # Only "b" has a reference point: the row exists, "a" is n/a in it.
    assert np.isnan(table.loc["rho_w_wloc", "a"])
    assert table.loc["rho_w_wloc", "b"] == pytest.approx(1.0)

    latex = cmp.to_latex()
    assert r"$\rho(S1, ST)$" in latex
    assert r"$\rho(w, w_{\mathrm{loc}})$" in latex


def test_reexports_are_pymcdm_classes():
    from pymcdm.methods import COMET as PymcdmCOMET

    assert gcisens.COMET is PymcdmCOMET


def test_r2_samples_is_reported_for_every_model_and_r2_fit_only_for_fitted_weights(hr_setup):
    criteria, bounds, esp1, _ = hr_setup
    comet = esp_comet(esps=esp1, bounds=bounds, criteria_names=criteria)
    spotis = esp_spotis(
        esp=esp1[0], bounds=bounds, weights=np.full(7, 1 / 7), criteria_names=criteria
    )

    comet_result = SobolStudy(comet, n_samples=64, seed=0).run()
    spotis_result = SobolStudy(spotis, n_samples=64, seed=0).run()

    # COMET: the declared weights come from a regression on characteristic
    # objects, so that fit has its own R^2 (the value printed in the article).
    assert comet_result.r2_fit == pytest.approx(CometAdapter(comet).weights_fit.r2)
    # SPOTIS: declared weights are an input, there is no fit behind them.
    assert spotis_result.r2_fit is None
    # Both models get the sample-based R^2 on the same uniform design.
    assert 0.0 < comet_result.r2_samples <= 1.0
    assert 0.0 < spotis_result.r2_samples <= 1.0
    assert comet_result.r2_samples != comet_result.r2_fit
    # gcisens <= 0.1.3 compatibility alias: the fit when there is one, else the sample.
    assert comet_result.r2 == comet_result.r2_fit
    assert spotis_result.r2 == spotis_result.r2_samples


def test_summary_and_comparison_report_both_r2_definitions(hr_setup):
    criteria, bounds, esp1, _ = hr_setup
    comet = esp_comet(esps=esp1, bounds=bounds, criteria_names=criteria)
    spotis = esp_spotis(
        esp=esp1[0], bounds=bounds, weights=np.full(7, 1 / 7), criteria_names=criteria
    )
    results = {
        "COMET": SobolStudy(comet, n_samples=64, seed=0).run(),
        "SPOTIS": SobolStudy(spotis, n_samples=64, seed=0).run(),
    }

    summary = results["COMET"].summary()
    assert summary["r2_fit"] == results["COMET"].r2_fit
    assert summary["r2_samples"] == results["COMET"].r2_samples
    assert "R2" not in summary.index

    table = compare(results).table()
    assert list(table.index[:2]) == ["r2_fit", "r2_samples"]
    assert table.loc["r2_fit", "COMET"] == results["COMET"].r2_fit
    assert np.isnan(table.loc["r2_fit", "SPOTIS"])
    assert table.loc["r2_samples", "SPOTIS"] == results["SPOTIS"].r2_samples

    latex = gcisens.comparison_to_latex(compare(results))
    assert r"$R^2$ (fit)" in latex
    assert r"$R^2$ (uniform sample)" in latex
    assert "n/a" in latex


def test_r2_sample_size_is_configurable(linear_model):
    score, bounds = linear_model
    weights = np.array([0.7, 0.3])

    default = SobolStudy(score, bounds=bounds, weights=weights, n_samples=64, seed=0).run()
    small = SobolStudy(
        score, bounds=bounds, weights=weights, n_samples=64, seed=0, n_r2_samples=16
    ).run()

    assert default.n_r2_samples == 4096
    assert small.n_r2_samples == 16
    assert default.summary()["n_r2_samples"] == 4096
    # A linear model is fitted exactly whatever the sample size.
    assert small.r2_samples == pytest.approx(1.0)
    with pytest.raises(ValueError, match="n_r2_samples"):
        SobolStudy(score, bounds=bounds, weights=weights, n_samples=64, n_r2_samples=2)
