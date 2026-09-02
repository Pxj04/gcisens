"""One-line builders for ESP-COMET and ESP-SPOTIS models.

These return plain pymcdm objects (fully usable with the pymcdm ecosystem,
e.g. ``pymcdm.visuals``) with gcisens metadata attached, so that
:class:`gcisens.SobolStudy` does not need bounds or names repeated.
"""

from __future__ import annotations

import numpy as np
from pymcdm.methods import COMET, SPOTIS
from pymcdm.methods.comet_tools import ESPExpert

from .adapters import (
    META_ATTR,
    ModelMeta,
    validate_bounds,
    validate_criteria_names,
    validate_esps,
    validate_weights,
)
from .weights import warn_large_grid


def esp_comet(
    esps,
    bounds,
    criteria_names=None,
    cvalues=None,
    expert: ESPExpert | None = None,
) -> COMET:
    """Build an ESP-COMET model in one call.

    Equivalent to the pymcdm idiom::

        expert = ESPExpert(esps=esps, bounds=bounds)
        model = COMET(expert.make_cvalues_psi(), expert)

    Parameters
    ----------
    esps : array-like of shape (k, m) or (m,)
        One or more Expected Solution Points (rows).
    bounds : array-like of shape (m, 2)
        Criteria domain bounds, ``[min, max]`` per criterion.
    criteria_names : list of str, optional
        Names used in all reports and plots.
    cvalues : list of arrays, optional
        Custom characteristic values; defaults to the ESP-driven grid
        ``[min, ESP..., max]`` per criterion (``make_cvalues_psi``).
    expert : ESPExpert, optional
        Custom expert (e.g. different distance function or aggregation);
        defaults to ``ESPExpert(esps, bounds)``.

    Returns
    -------
    COMET
        A plain :class:`pymcdm.methods.COMET` instance with gcisens metadata.

    Notes
    -----
    The characteristic-object count is the product of the numbers of
    characteristic values per criterion. pymcdm allocates a float16 matrix
    with roughly ``2 * count**2`` bytes in the ``COMET`` constructor, so this
    builder warns *before* building the model when the count is above
    20,000. Reduce the number of criteria or ESPs, or pass smaller
    ``cvalues``, when the warning appears.
    """
    bounds = validate_bounds(bounds)
    m = bounds.shape[0]
    esps = validate_esps(esps, m)
    validate_criteria_names(criteria_names, m)
    if expert is None:
        expert = ESPExpert(esps=esps, bounds=bounds)
    if cvalues is None:
        cvalues = expert.make_cvalues_psi()
    warn_large_grid(cvalues)
    model = COMET(cvalues, expert)
    setattr(
        model,
        META_ATTR,
        ModelMeta(bounds=bounds, criteria_names=criteria_names, esps=esps),
    )
    return model


def esp_spotis(
    esp,
    bounds,
    weights=None,
    types=None,
    criteria_names=None,
) -> SPOTIS:
    """Build an ESP-SPOTIS model in one call.

    Equivalent to the pymcdm idiom ``SPOTIS(bounds, esp=esp)``, with the
    declared weights and criteria types stored alongside so evaluation and
    analysis need no extra arguments.

    Parameters
    ----------
    esp : array-like of shape (m,)
        The Expected Solution Point (SPOTIS supports exactly one).
    bounds : array-like of shape (m, 2)
        Criteria domain bounds, ``[min, max]`` per criterion.
    weights : array-like of shape (m,), optional
        Declared criteria weights; default equal weights. Unlike COMET, SPOTIS
        takes weights as an *input* — they are the "reported" importance the
        Sensitivity Discrepancy Report checks against actual influence.
    types : array-like of shape (m,), optional
        Criteria types (1 profit / -1 cost). With an explicit ESP they do not
        affect the result, so they default to profit criteria.
    criteria_names : list of str, optional
        Names used in all reports and plots.

    Returns
    -------
    SPOTIS
        A plain :class:`pymcdm.methods.SPOTIS` instance with gcisens metadata.

    Notes
    -----
    SPOTIS scores are *distances*: lower means closer to the ESP. Validation
    and ranking helpers account for this automatically.
    """
    bounds = validate_bounds(bounds)
    m = bounds.shape[0]
    validate_criteria_names(criteria_names, m)
    esp = np.asarray(esp, dtype=float).ravel()
    if esp.shape != (m,):
        raise ValueError(f"esp must have {m} values (one per criterion), got {esp.shape}")
    weights = np.full(m, 1.0 / m) if weights is None else validate_weights(weights, m)
    types = np.ones(m) if types is None else np.asarray(types, dtype=float)
    model = SPOTIS(bounds, esp=esp)
    setattr(
        model,
        META_ATTR,
        ModelMeta(
            bounds=bounds,
            criteria_names=criteria_names,
            weights=weights,
            types=types,
            esps=np.atleast_2d(esp),
        ),
    )
    return model
