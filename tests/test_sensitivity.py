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


@pytest.mark.parametrize("sampler", ["saltelli", "sobol"])
@pytest.mark.parametrize("seed", [0, 42])
def test_seed_repeats_all_indices_and_confidence_intervals(linear_model, sampler, seed):
    score, bounds = linear_model
    first = sobol_analysis(score, bounds, ["A", "B"], n_samples=64, sampler=sampler, seed=seed)
    second = sobol_analysis(score, bounds, ["A", "B"], n_samples=64, sampler=sampler, seed=seed)
    for name in ("S1", "ST", "S1_conf", "ST_conf", "S2", "S2_conf"):
        np.testing.assert_array_equal(getattr(first, name), getattr(second, name))


def test_positive_seed_preserves_salib_bootstrap_samples(linear_model):
    from SALib.analyze import sobol
    from SALib.sample import sobol as sampler

    score, bounds = linear_model
    problem = {"num_vars": 2, "names": ["A", "B"], "bounds": bounds.tolist()}
    samples = sampler.sample(problem, 64, seed=42)
    expected = sobol.analyze(problem, score(samples), seed=42)
    result = sobol_analysis(score, bounds, ["A", "B"], n_samples=64, sampler="sobol", seed=42)
    for name in ("S1", "ST", "S1_conf", "ST_conf", "S2", "S2_conf"):
        np.testing.assert_array_equal(getattr(result, name), expected[name])


@pytest.mark.parametrize(
    "value, message", [(1.0, "constant"), (np.nan, "finite"), (np.inf, "finite")]
)
def test_undefined_scores_fail_before_diagnosis(value, message):
    with pytest.raises(ValueError, match=message):
        sobol_analysis(lambda X: np.full(len(X), value), [[0, 1]], ["A"], n_samples=8)


@pytest.mark.parametrize(
    "setting, value",
    [
        ("n_samples", 8.5),
        ("n_samples", True),
        ("n_samples", "8"),
        ("num_resamples", 1),
        ("num_resamples", 5.5),
        ("num_resamples", False),
        ("conf_level", 0),
        ("conf_level", 1),
        ("conf_level", np.nan),
        ("seed", -1),
        ("seed", 1.5),
        ("seed", True),
        ("second_order", "false"),
    ],
)
def test_invalid_sampling_settings_fail_before_model_evaluation(setting, value):
    def must_not_run(X):
        raise AssertionError("Model must not be called for invalid settings")

    settings = {"n_samples": 8, setting: value}
    with pytest.raises(ValueError, match=setting):
        sobol_analysis(must_not_run, [[0, 1]], ["A"], **settings)


def test_sobol_indices_own_immutable_arrays_and_names(linear_model):
    from dataclasses import FrozenInstanceError, replace

    score, bounds = linear_model
    result = sobol_analysis(score, bounds, ["A", "B"], n_samples=64, seed=42)
    values = result.S1.copy()
    names = ["A", "B"]
    copied = replace(result, S1=values, criteria_names=names)
    values[:] = 0
    names[0] = "changed"
    np.testing.assert_array_equal(copied.S1, result.S1)
    assert copied.criteria_names == ("A", "B")
    with pytest.raises(ValueError):
        copied.S1[0] = 0
    with pytest.raises(ValueError):
        copied.S1.setflags(write=True)
    with pytest.raises(FrozenInstanceError):
        copied.S1 = values


def test_single_criterion_has_empty_second_order_table():
    result = sobol_analysis(lambda X: X[:, 0], [[0, 1]], ["A"], n_samples=64, seed=42)
    assert result.s2_pairs().empty
    assert list(result.s2_pairs()) == ["criterion_i", "criterion_j", "S2", "S2_conf", "significant"]
