"""Weight extraction: global (regression-based) and local (range-sweep).

Global weights follow the regression approach of Sałabun et al. (ISD 2025),
eqs. (1)-(2): fit a linear model to preference values over the criteria space
and normalise the absolute coefficients. Local weights follow the range-sweep
algorithm of Więckowski et al. (2023), the same one implemented in
``pymcdm.methods.comet_tools.get_local_weights``.
"""

from __future__ import annotations

import itertools
import math
import warnings
from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LinearRegression


@dataclass
class RegressionWeights:
    """Result of a linear-regression weight fit."""

    weights: np.ndarray
    coefficients: np.ndarray
    intercept: float
    r2: float


def normalize_to_bounds(X: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    """Min-max normalise columns of ``X`` to [0, 1] using ``bounds``."""
    X = np.asarray(X, dtype=float)
    lo, hi = bounds[:, 0], bounds[:, 1]
    rng = hi - lo
    rng = np.where(rng > 0, rng, 1.0)
    return (X - lo) / rng


def regression_weights(X: np.ndarray, y: np.ndarray, bounds: np.ndarray) -> RegressionWeights:
    """Fit ``y ~ X`` (bounds-normalised) and return normalised |coef| weights."""
    Xn = normalize_to_bounds(X, bounds)
    reg = LinearRegression()
    reg.fit(Xn, y)
    abs_coefs = np.abs(reg.coef_)
    total = abs_coefs.sum()
    weights = abs_coefs / total if total > 0 else np.full(len(abs_coefs), 1 / len(abs_coefs))
    return RegressionWeights(
        weights=weights,
        coefficients=reg.coef_.copy(),
        intercept=float(reg.intercept_),
        r2=float(reg.score(Xn, y)),
    )


def characteristic_objects_grid(cvalues) -> np.ndarray:
    """Cartesian product of characteristic values: the CO grid of a COMET model."""
    return np.array(list(itertools.product(*cvalues)), dtype=float)


#: Characteristic-object count above which :func:`warn_large_grid` warns.
LARGE_GRID_OBJECTS = 20_000


def characteristic_objects_count(grid_lines) -> int:
    """Number of characteristic objects: the product of the grid sizes."""
    return math.prod(len(values) for values in grid_lines)


def warn_large_grid(grid_lines, limit: int | None = None) -> int:
    """Warn when a COMET grid is large; returns the characteristic-object count.

    pymcdm stores a float16 judgment matrix with ``count**2`` entries, so its
    memory grows with the square of the count. Call this *before* the COMET
    model is built, as :func:`gcisens.esp_comet` does, because pymcdm
    allocates the matrix in the constructor.
    """
    limit = LARGE_GRID_OBJECTS if limit is None else limit
    count = characteristic_objects_count(grid_lines)
    if count > limit:
        mej_mib = count**2 * np.dtype(np.float16).itemsize / 1024**2
        warnings.warn(
            f"COMET has {count:,} characteristic objects; pymcdm's float16 "
            f"judgment matrix needs about {mej_mib:,.1f} MiB because its size grows with "
            "the square of the grid size",
            UserWarning,
            stacklevel=3,
        )
    return count


def grid_regression_weights(score_fn, grid_lines, bounds: np.ndarray) -> RegressionWeights:
    """Global weights by regression on the Cartesian grid of ``grid_lines``.

    For a COMET model the grid is its characteristic objects and ``score_fn``
    its preference function (Sałabun et al., ISD 2025, eqs. (1)-(2)).
    """
    grid = characteristic_objects_grid(grid_lines)
    return regression_weights(grid, np.asarray(score_fn(grid)).ravel(), bounds)


def sweep_local_weights(
    score_fn,
    point,
    bounds,
    percent_step: float = 0.01,
    include_upper: bool = False,
) -> np.ndarray:
    """Local weights at ``point``: preference range under a one-criterion sweep.

    For each criterion the point is swept over the criterion's full domain
    (holding the others fixed) and the range (max - min) of the model score is
    recorded; ranges are normalised to sum to 1. By default, the sweep excludes
    the upper bound, matching ``pymcdm.methods.comet_tools.get_local_weights``.
    Set ``include_upper=True`` to evaluate both bounds. This implementation
    works for any scoring function, not only COMET.
    """
    point = np.asarray(point, dtype=float).ravel()
    m = bounds.shape[0]
    if point.shape != (m,):
        raise ValueError(f"reference point must have {m} values, got {point.shape}")
    ranges = np.zeros(m)
    for i in range(m):
        lo, hi = bounds[i]
        if include_upper:
            swept = np.linspace(lo, hi, round(1 / percent_step) + 1)
        else:
            step = (hi - lo) * percent_step
            swept = np.arange(lo, hi, step)
        candidates = np.tile(point, (len(swept), 1))
        candidates[:, i] = swept
        scores = score_fn(candidates)
        ranges[i] = np.max(scores) - np.min(scores)
    total = ranges.sum()
    return ranges / total if total > 0 else np.full(m, 1 / m)
