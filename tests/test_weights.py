import numpy as np
import pytest

from gcisens.weights import (
    characteristic_objects_grid,
    grid_regression_weights,
    regression_weights,
    sweep_local_weights,
    warn_large_grid,
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


def test_warn_large_grid_reports_object_count_and_memory():
    grid_lines = [np.arange(201), np.arange(100)]
    with pytest.warns(UserWarning, match=r"20,100 characteristic objects.*770\.6 MiB"):
        assert warn_large_grid(grid_lines) == 20_100


def test_warn_large_grid_is_silent_below_the_limit(recwarn):
    assert warn_large_grid([np.arange(3), np.arange(3)]) == 9
    assert not recwarn.list


def test_esp_comet_warns_before_pymcdm_allocates_the_matrix(monkeypatch):
    import warnings

    from gcisens import builders, weights

    warnings_seen_at_construction = []

    class RecordingComet:
        def __init__(self, cvalues, expert):
            warnings_seen_at_construction.append(len(caught))

    monkeypatch.setattr(weights, "LARGE_GRID_OBJECTS", 10)
    monkeypatch.setattr(builders, "COMET", RecordingComet)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        builders.esp_comet(esps=[[5.0, 2.5, 0.5]], bounds=[[0, 10], [0, 5], [0, 1]])  # 27 objects
    assert warnings_seen_at_construction == [1]


def test_grid_regression_weights_on_an_additive_model():
    grid_lines = [np.array([0.0, 5.0, 10.0]), np.array([0.0, 2.5, 5.0])]
    bounds = np.array([[0.0, 10.0], [0.0, 5.0]])

    def score(X):
        return 0.8 * X[:, 0] / 10 + 0.2 * X[:, 1] / 5

    fit = grid_regression_weights(score, grid_lines, bounds)
    np.testing.assert_allclose(fit.weights, [0.8, 0.2])
    assert fit.r2 == pytest.approx(1.0)


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


@pytest.mark.parametrize("step", [0, -0.01, np.nan, np.inf, True, 1, 1.1])
def test_local_sweep_rejects_invalid_steps(step, linear_model):
    score, bounds = linear_model
    with pytest.raises(ValueError, match="local_percent_step"):
        sweep_local_weights(score, [5, 5], bounds, percent_step=step)


@pytest.mark.parametrize("point", [[np.nan, 5], [-1, 5], [5, 11]])
def test_local_sweep_rejects_invalid_reference_point(point, linear_model):
    score, bounds = linear_model
    with pytest.raises(ValueError, match="reference point"):
        sweep_local_weights(score, point, bounds)


def test_local_sweep_rejects_nonfinite_model_scores():
    with pytest.raises(ValueError, match="finite score"):
        sweep_local_weights(lambda X: np.full(len(X), np.nan), [0.5], [[0, 1]])
