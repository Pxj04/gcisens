import numpy as np
import pytest

from gcisens.diagnosis import DiagnosisThresholds
from gcisens.sensitivity import SobolIndices, sobol_analysis


def test_additive_model_has_no_interactions(linear_model):
    score, bounds = linear_model
    res = sobol_analysis(score, bounds, ["A", "B"], n_samples=1024, seed=0)
    # Variance shares of an additive model: proportional to coef^2 (equal ranges).
    expected = np.array([0.49, 0.09]) / 0.58
    np.testing.assert_allclose(res.S1, expected, atol=0.02)
    np.testing.assert_allclose(res.ST, expected, atol=0.02)
    assert abs(res.interaction).max() < 0.02
    assert res.n_evaluations == 1024 * (2 * 2 + 2)


def test_interactive_model_shows_interactions():
    def score(X):
        X = np.atleast_2d(X)
        return X[:, 0] * X[:, 1]  # pure interaction on [0,1]^2 still has main effects

    bounds = np.array([[0.0, 1.0], [0.0, 1.0]])
    res = sobol_analysis(score, bounds, ["A", "B"], n_samples=1024, seed=0)
    assert (res.interaction > 0.05).all()
    assert res.S2 is not None
    pairs = res.s2_pairs()
    assert pairs.loc[0, "S2"] > 0.05


def test_s2_significance_uses_diagnosis_thresholds():
    s2 = np.array([[np.nan, 0.03], [np.nan, np.nan]])
    conf = np.array([[np.nan, 0.02], [np.nan, np.nan]])
    indices = SobolIndices(
        S1=np.array([0.4, 0.4]),
        ST=np.array([0.5, 0.5]),
        S1_conf=np.array([0.01, 0.01]),
        ST_conf=np.array([0.01, 0.01]),
        S2=s2,
        S2_conf=conf,
        criteria_names=["A", "B"],
        n_samples=64,
        n_evaluations=384,
        sampler="saltelli",
    )

    assert indices.s2_pairs().loc[0, "significant"]
    strict = DiagnosisThresholds(s2_significance_factor=2.0)
    assert not indices.s2_pairs(strict).loc[0, "significant"]


def test_sobol_indices_keeps_output_statistics_positional_arguments():
    indices = SobolIndices(
        np.array([0.4, 0.4]),
        np.array([0.5, 0.5]),
        np.array([0.01, 0.01]),
        np.array([0.01, 0.01]),
        None,
        None,
        ["A", "B"],
        64,
        256,
        "saltelli",
        1.25,
        0.3,
    )

    assert indices.output_mean == 1.25
    assert indices.output_std == 0.3


def test_second_order_disabled():
    def score(X):
        return np.atleast_2d(X)[:, 0]

    bounds = np.array([[0.0, 1.0], [0.0, 1.0]])
    res = sobol_analysis(score, bounds, ["A", "B"], n_samples=256, second_order=False, seed=0)
    assert res.S2 is None
    assert res.n_evaluations == 256 * (2 + 2)
    with pytest.raises(ValueError, match="Second-order"):
        res.s2_pairs()


def test_sobol_sampler_variant(linear_model):
    score, bounds = linear_model
    res = sobol_analysis(score, bounds, ["A", "B"], n_samples=512, sampler="sobol", seed=7)
    expected = np.array([0.49, 0.09]) / 0.58
    np.testing.assert_allclose(res.S1, expected, atol=0.05)


def test_sobol_analysis_records_bootstrap_configuration(linear_model):
    score, bounds = linear_model

    res = sobol_analysis(
        score,
        bounds,
        ["A", "B"],
        n_samples=64,
        num_resamples=25,
        conf_level=0.9,
        seed=7,
    )

    assert res.num_resamples == 25
    assert res.conf_level == 0.9


def test_unknown_sampler_rejected(linear_model):
    score, bounds = linear_model
    with pytest.raises(ValueError, match="sampler"):
        sobol_analysis(score, bounds, ["A", "B"], sampler="lhs")
