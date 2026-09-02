import numpy as np
import pytest

from gcisens.weights import (
    characteristic_objects_grid,
    comet_global_weights,
    regression_weights,
    sweep_local_weights,
)


def test_regression_weights_recover_linear_coefficients(linear_model):
    score, bounds = linear_model
    rng = np.random.default_rng(0)
    X = rng.uniform(0, 10, size=(500, 2))
    fit = regression_weights(X, score(X), bounds)
    np.testing.assert_allclose(fit.weights, [0.7, 0.3], atol=1e-10)
    assert fit.r2 > 0.999999


def test_regression_weights_use_absolute_coefficients():
    bounds = np.array([[0.0, 1.0], [0.0, 1.0]])
    rng = np.random.default_rng(1)
    X = rng.uniform(0, 1, size=(300, 2))
    y = -0.6 * X[:, 0] + 0.4 * X[:, 1]
    fit = regression_weights(X, y, bounds)
    np.testing.assert_allclose(fit.weights, [0.6, 0.4], atol=1e-10)
    assert fit.coefficients[0] < 0


def test_characteristic_objects_grid_shape():
    cvalues = [np.array([0, 1, 2]), np.array([0, 5])]
    grid = characteristic_objects_grid(cvalues)
    assert grid.shape == (6, 2)
    assert {tuple(row) for row in grid} == {(0, 0), (0, 5), (1, 0), (1, 5), (2, 0), (2, 5)}


def test_comet_global_weights_warns_about_large_mej():
    class LargeGridModel:
        def __init__(self):
            self.cvalues = [np.arange(201), np.arange(100)]

        def __call__(self, X):
            return X[:, 0] + X[:, 1]

    bounds = np.array([[0.0, 200.0], [0.0, 99.0]])

    with pytest.warns(
        UserWarning,
        match=r"20,100 characteristic objects.*770\.6 MiB",
    ):
        comet_global_weights(LargeGridModel(), bounds)


def test_sweep_local_weights_linear(linear_model):
    score, bounds = linear_model
    local = sweep_local_weights(score, [5.0, 5.0], bounds)
    np.testing.assert_allclose(local, [0.7, 0.3], atol=1e-6)


def test_sweep_local_weights_flat_model():
    def flat(X):
        return np.ones(len(np.atleast_2d(X)))

    bounds = np.array([[0.0, 1.0], [0.0, 1.0]])
    local = sweep_local_weights(flat, [0.5, 0.5], bounds)
    np.testing.assert_allclose(local, [0.5, 0.5])


def test_sweep_local_weights_excludes_upper_bound_by_default():
    def nonlinear(X):
        return X[:, 0] ** 8 + X[:, 1]

    bounds = np.array([[0.0, 1.0], [0.0, 1.0]])

    local = sweep_local_weights(nonlinear, [0.5, 0.5], bounds, percent_step=0.01)

    expected_ranges = np.array([0.99**8, 0.99])
    np.testing.assert_allclose(local, expected_ranges / expected_ranges.sum())


def test_sweep_local_weights_can_include_upper_bound():
    def nonlinear(X):
        return X[:, 0] ** 8 + X[:, 1]

    bounds = np.array([[0.0, 1.0], [0.0, 1.0]])

    local = sweep_local_weights(
        nonlinear,
        [0.5, 0.5],
        bounds,
        percent_step=0.01,
        include_upper=True,
    )

    np.testing.assert_allclose(local, [0.5, 0.5])
