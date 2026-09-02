"""Model adapters: a uniform interface over COMET, SPOTIS and callable models.

The rest of the pipeline (Sobol' analysis, weights, diagnosis, exports) only
talks to :class:`ModelAdapter`. Adding support for another MCDA method means
writing one new adapter here and nothing else.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np
from pymcdm.methods import COMET, SPOTIS

META_ATTR = "_gcisens_meta"


def validate_bounds(bounds) -> np.ndarray:
    """Return criteria bounds after validating their shape and row order."""
    bounds = np.asarray(bounds, dtype=float)
    if bounds.ndim != 2 or bounds.shape[1] != 2:
        raise ValueError("bounds must have shape (m, 2) with [min, max] rows")
    invalid = np.flatnonzero(~np.isfinite(bounds).all(axis=1) | (bounds[:, 0] >= bounds[:, 1]))
    if len(invalid):
        row = int(invalid[0])
        lower, upper = bounds[row]
        raise ValueError(
            f"bounds row {row} must have min < max, got [{float(lower)}, {float(upper)}]"
        )
    return bounds


def validate_weights(weights, n_criteria: int) -> np.ndarray:
    """Return declared weights after validating their shape and normalization."""
    weights = np.asarray(weights, dtype=float)
    if weights.shape != (n_criteria,):
        raise ValueError("weights must be a 1-D array with one value per criterion")
    if (
        not np.all(np.isfinite(weights))
        or np.any(weights < 0)
        or abs(float(weights.sum()) - 1.0) >= 1e-6
    ):
        raise ValueError("weights must be non-negative and sum to 1")
    return weights


def validate_criteria_names(criteria_names, n_criteria: int):
    """Reject a criteria-name list whose length does not match the model."""
    if criteria_names is not None and len(criteria_names) != n_criteria:
        raise ValueError(f"Got {len(criteria_names)} criteria names for {n_criteria} criteria")
    return criteria_names


def validate_esps(esps, n_criteria: int) -> np.ndarray | None:
    """Return the ESPs as a ``(k, m)`` array, or ``None`` when there are none."""
    if esps is None:
        return None
    esps = np.atleast_2d(np.asarray(esps, dtype=float))
    if esps.ndim != 2 or esps.shape[1] != n_criteria:
        raise ValueError(f"esps must have shape (k, {n_criteria})")
    return esps


@dataclass(frozen=True)
class DeclaredWeights:
    """The weights a model reports to stakeholders, with their provenance.

    ``source`` is the label the study prints (``"declared"`` for weights
    given as an input, ``"regression (characteristic objects)"`` for the
    COMET fit). ``r2`` is the fit quality behind the weights, ``None`` when
    there is no fit.
    """

    weights: np.ndarray
    source: str
    r2: float | None = None


@dataclass
class ModelMeta:
    """Metadata attached to a model by the :mod:`gcisens.builders` helpers."""

    bounds: np.ndarray | None = None
    criteria_names: list[str] | None = None
    weights: np.ndarray | None = None
    types: np.ndarray | None = None
    esps: np.ndarray | None = None


class ModelAdapter:
    """Uniform interface over a scoring model.

    Parameters
    ----------
    model : object
        The underlying model (pymcdm object or callable).
    bounds : ndarray of shape (m, 2)
        Criteria domain bounds, ``[min, max]`` per criterion.
    criteria_names : list of str or None
        Criteria names; defaults to ``C1..Cm``.
    esps : ndarray of shape (k, m) or None
        Expected Solution Points the model was built around, used as plot
        markers. Subclasses recover them from the model when they can.
    """

    #: Whether higher scores mean "closer to the expected solution point".
    #: COMET preferences grow towards the ESP; SPOTIS scores are distances,
    #: so they shrink towards it.
    higher_is_closer: bool = True

    def __init__(self, model, bounds, criteria_names=None, esps=None):
        if bounds is None:
            raise ValueError(
                "Criteria bounds are required. Pass bounds=... to SobolStudy "
                "or build the model with gcisens.esp_comet / gcisens.esp_spotis."
            )
        self.model = model
        self.bounds = validate_bounds(bounds)
        m = self.bounds.shape[0]
        if criteria_names is None:
            criteria_names = [f"C{i + 1}" for i in range(m)]
        validate_criteria_names(criteria_names, m)
        self.criteria_names = [str(c) for c in criteria_names]
        #: ESPs as a ``(k, m)`` array, or ``None`` when the model has none.
        self.esps = validate_esps(esps, m)

    @property
    def n_criteria(self) -> int:
        return self.bounds.shape[0]

    def grid_lines(self) -> list[np.ndarray] | None:
        """Per-criterion evaluation grid drawn on decision surfaces.

        COMET returns its characteristic values; models without a grid
        return ``None``.
        """
        return None

    def scores(self, X: np.ndarray) -> np.ndarray:
        """Evaluate the model on a 2-D matrix of alternatives; returns 1-D scores."""
        raise NotImplementedError

    def declared_weights(self) -> DeclaredWeights | None:
        """The weights the model "reports" to stakeholders, with their source.

        Returns ``None`` when the model has no declared weights; the study then
        fits them by linear regression on a uniform sample over the bounds.
        """
        return None

    def local_weights(self, point, percent_step: float = 0.01) -> np.ndarray:
        """Local criteria weights at ``point`` via the range-sweep algorithm.

        One implementation for every model; on a COMET model it gives the
        values of ``pymcdm.methods.comet_tools.get_local_weights``.
        """
        from .weights import sweep_local_weights

        return sweep_local_weights(self.scores, point, self.bounds, percent_step)


class CometAdapter(ModelAdapter):
    """Adapter for :class:`pymcdm.methods.COMET` (including ESP-COMET)."""

    higher_is_closer = True

    def __init__(self, model, bounds=None, criteria_names=None, esps=None):
        self.model_bounds = np.array([[cv[0], cv[-1]] for cv in model.cvalues], dtype=float)
        if bounds is None:
            # The characteristic-value grid spans the whole domain, so the
            # bounds can be recovered from the model itself.
            bounds = self.model_bounds
        if esps is None:
            # An ESPExpert keeps the points it was built from.
            esps = getattr(getattr(model, "expert_function", None), "esps", None)
        super().__init__(model, bounds, criteria_names, esps)

    def grid_lines(self):
        return [np.asarray(cv, dtype=float) for cv in self.model.cvalues]

    def scores(self, X):
        X = np.atleast_2d(np.asarray(X, dtype=float))
        try:
            return np.asarray(self.model(X)).ravel()
        except ValueError as error:
            outside = (X < self.model_bounds[:, 0]) | (X > self.model_bounds[:, 1])
            if outside.any():
                row, column = np.argwhere(outside)[0]
                name = self.criteria_names[column]
                lower, upper = self.model_bounds[column]
                raise ValueError(
                    f"Criterion {name!r} contains value {X[row, column]:g} "
                    f"outside bounds [{float(lower)}, {float(upper)}]"
                ) from error
            raise

    @cached_property
    def weights_fit(self):
        """Regression fit used for declared weights and fit quality."""
        from .weights import grid_regression_weights

        return grid_regression_weights(self.scores, self.grid_lines(), self.bounds)

    def declared_weights(self):
        fit = self.weights_fit
        return DeclaredWeights(fit.weights, "regression (characteristic objects)", fit.r2)


class SpotisAdapter(ModelAdapter):
    """Adapter for :class:`pymcdm.methods.SPOTIS` (including ESP-SPOTIS).

    SPOTIS requires criteria ``weights`` and ``types`` at evaluation time.
    When an ESP is set, ``types`` do not influence the result (the expected
    solution point is given explicitly), so they default to profit criteria.
    """

    higher_is_closer = False

    def __init__(
        self, model, bounds=None, criteria_names=None, weights=None, types=None, esps=None
    ):
        if bounds is None:
            bounds = model.bounds
        if esps is None:
            esps = model.esp
        super().__init__(model, bounds, criteria_names, esps)
        if weights is None:
            raise ValueError(
                "SPOTIS models need declared criteria weights. Pass weights=... "
                "to SobolStudy or build the model with gcisens.esp_spotis."
            )
        self.weights = validate_weights(weights, self.n_criteria)
        if types is None:
            if model.esp is None:
                raise ValueError("SPOTIS without an ESP needs criteria types (1 profit / -1 cost).")
            types = np.ones(self.n_criteria)
        self.types = np.asarray(types, dtype=float)

    def scores(self, X):
        X = np.atleast_2d(np.asarray(X, dtype=float))
        # validation=False: pymcdm's decision-matrix checks (dominance etc.)
        # are meaningless for sensitivity samples and only produce warnings.
        return np.asarray(self.model(X, self.weights, self.types, validation=False)).ravel()

    def declared_weights(self):
        return DeclaredWeights(self.weights, "declared")


class CallableAdapter(ModelAdapter):
    """Fallback adapter for any callable ``f(X) -> scores``.

    Lets arbitrary scoring functions (other pymcdm methods wrapped in a lambda,
    custom models) go through the same pipeline. Declared weights are optional;
    without them the study reports regression weights fitted on a seeded
    uniform sample over the bounds (the sample behind ``r2_samples``), not on
    the Sobol' design.
    """

    def __init__(self, model, bounds=None, criteria_names=None, weights=None, esps=None):
        if not callable(model):
            raise TypeError(f"Unsupported model type: {type(model).__name__}")
        super().__init__(model, bounds, criteria_names, esps)
        self.weights = None if weights is None else validate_weights(weights, self.n_criteria)

    def scores(self, X):
        return np.asarray(self.model(np.atleast_2d(np.asarray(X, dtype=float)))).ravel()

    def declared_weights(self):
        return None if self.weights is None else DeclaredWeights(self.weights, "declared")


def _resolve(name, explicit, from_meta):
    """Return the explicit argument or the builder metadata; reject conflicts."""
    if explicit is None:
        return from_meta
    if from_meta is not None and not _same(explicit, from_meta):
        raise ValueError(
            f"{name} passed to SobolStudy differs from the {name} the model was built "
            f"with; pass it to gcisens.esp_comet / gcisens.esp_spotis instead"
        )
    return explicit


def _same(a, b) -> bool:
    try:
        a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    except (TypeError, ValueError):
        return list(a) == list(b)
    return a.shape == b.shape and np.allclose(a, b)


def make_adapter(model, bounds=None, criteria_names=None, weights=None, types=None, esps=None):
    """Build the right adapter for ``model``.

    Models from :mod:`gcisens.builders` carry their bounds, names, weights,
    types and ESPs; those are handed to the adapter here. An explicit
    argument that differs from the builder metadata raises ``ValueError``.
    """
    meta = getattr(model, META_ATTR, None)
    if meta is None:
        meta = ModelMeta()
        if isinstance(model, COMET):
            # Built by hand, so no builder warned before pymcdm allocated the
            # judgment matrix; explain the memory use at least now.
            from .weights import warn_large_grid

            warn_large_grid(model.cvalues)
    bounds = _resolve("bounds", bounds, meta.bounds)
    criteria_names = _resolve("criteria_names", criteria_names, meta.criteria_names)
    weights = _resolve("weights", weights, meta.weights)
    types = _resolve("types", types, meta.types)
    esps = _resolve("esps", esps, meta.esps)

    if isinstance(model, COMET):
        if weights is not None or types is not None:
            raise ValueError(
                "COMET models do not take weights/types; they are estimated by regression"
            )
        return CometAdapter(model, bounds, criteria_names, esps)
    if isinstance(model, SPOTIS):
        return SpotisAdapter(model, bounds, criteria_names, weights, types, esps)
    return CallableAdapter(model, bounds, criteria_names, weights, esps)
