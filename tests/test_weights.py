import numpy as np

from gcisens.weights import (
    characteristic_objects_grid,
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
