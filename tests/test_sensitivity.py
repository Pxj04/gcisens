import numpy as np
import pytest

from gcisens.sensitivity import sobol_analysis


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


def test_second_order_disabled():
    def score(X):
        return np.atleast_2d(X)[:, 0]

    bounds = np.array([[0.0, 1.0], [0.0, 1.0]])
    res = sobol_analysis(score, bounds, ["A", "B"], n_samples=256,
                         second_order=False, seed=0)
    assert res.S2 is None
    assert res.n_evaluations == 256 * (2 + 2)
    with pytest.raises(ValueError, match="Second-order"):
        res.s2_pairs()


def test_sobol_sampler_variant(linear_model):
    score, bounds = linear_model
    res = sobol_analysis(score, bounds, ["A", "B"], n_samples=512,
                         sampler="sobol", seed=7)
    expected = np.array([0.49, 0.09]) / 0.58
    np.testing.assert_allclose(res.S1, expected, atol=0.05)


def test_unknown_sampler_rejected(linear_model):
    score, bounds = linear_model
    with pytest.raises(ValueError, match="sampler"):
        sobol_analysis(score, bounds, ["A", "B"], sampler="lhs")
