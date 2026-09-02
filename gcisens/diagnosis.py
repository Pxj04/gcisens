"""Sensitivity Discrepancy Report: per-criterion diagnostic classification.

Implements the decision rules of Śniegowski et al. (KES 2026), Section 3.
Each criterion is compared across two views — declared weights and Sobol'
indices — and labelled with the first matching category:

1. ``hidden influence`` — near-zero weight, yet substantial total-order effect;
2. ``interaction dominance`` — interactions dominate the criterion's effect;
3. ``moderate discrepancy`` — displaced ranking, or a dismissed-but-influential
   criterion;
4. ``confirmed transparency`` — the weight is a faithful account of influence.

The thresholds are the article's defaults for preference scores normalised to
[0, 1]; the article explicitly frames them as context-dependent, so they are a
parameter (:class:`DiagnosisThresholds`), not constants.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


class Category(str):
    """A discrepancy category with its presentation.

    The value is the name used in tables and CSV files. It compares and
    hashes as a plain ``str``, so ``HIDDEN_INFLUENCE == "hidden influence"``
    holds. The ``label`` is the display text and ``color`` is the badge
    colour of the HTML report, so renderers never key on the raw name.
    """

    label: str
    color: str

    def __new__(cls, name: str, *, label: str | None = None, color: str):
        obj = super().__new__(cls, name)
        obj.label = label if label is not None else name[:1].upper() + name[1:]
        obj.color = color
        return obj

    def __getnewargs_ex__(self):  # copies and pickles keep label and color
        return (str(self),), {"label": self.label, "color": self.color}

    @classmethod
    def of(cls, name) -> Category:
        """The category constant with this name.

        A :class:`Category` that is not one of the constants (a custom
        category) passes through unchanged.
        """
        for category in CATEGORIES:
            if category == name:
                return category
        if isinstance(name, cls):
            return name
        raise ValueError(f"unknown category {name!r}; expected one of {list(CATEGORIES)}")


HIDDEN_INFLUENCE = Category("hidden influence", color="#e37e7e")
INTERACTION_DOMINANCE = Category("interaction dominance", color="#e3b57e")
MODERATE_DISCREPANCY = Category("moderate discrepancy", color="#e3d97e")
CONFIRMED_TRANSPARENCY = Category("confirmed transparency", color="#a8d8a8")

CATEGORIES = (
    HIDDEN_INFLUENCE,
    INTERACTION_DOMINANCE,
    MODERATE_DISCREPANCY,
    CONFIRMED_TRANSPARENCY,
)


@dataclass
class DiagnosisThresholds:
    """Decision-rule thresholds of the Sensitivity Discrepancy Report.

    Defaults follow Śniegowski et al. (KES 2026), Section 3, for scores and
    indices on a [0, 1] scale with a moderate number of criteria. Recalibrate
    for other decision contexts.
    """

    #: hidden influence: ``w < hidden_weight_factor * (1/m)`` ...
    hidden_weight_factor: float = 0.5
    #: ... and ``ST - w >= hidden_st_excess``.
    hidden_st_excess: float = 0.03
    #: interaction dominance: ``(ST - S1)/ST >= interaction_ratio`` ...
    interaction_ratio: float = 0.30
    #: ... with an absolute gap of at least ``interaction_abs`` (the article
    #: treats the two as equivalent on the [0, 1] scale; requiring both keeps
    #: near-zero-ST criteria from being flagged on ratio noise).
    interaction_abs: float = 0.02
    #: moderate discrepancy: rank displacement ``|rank_w - rank_S1| >= rank_displacement`` ...
    rank_displacement: int = 2
    #: ... or ``w < negligible_weight`` while ``ST >= influential_st``.
    negligible_weight: float = 0.01
    influential_st: float = 0.02
    #: pairwise significance: ``abs(S2) > s2_significance_factor * S2_conf`` ...
    s2_significance_factor: float = 1.0
    #: ... and ``abs(S2) > s2_min_abs``.
    s2_min_abs: float = 0.01

    def is_s2_significant(self, value: float, conf: float) -> bool:
        """Return whether an S2 estimate clears both significance thresholds."""
        return bool(
            abs(value) > self.s2_significance_factor * conf and abs(value) > self.s2_min_abs
        )


@dataclass
class CriterionDiagnosis:
    """Diagnostic label for one criterion.

    ``category`` may be given as a plain name; it is resolved to the
    :class:`Category` constant (``ValueError`` for an unknown name).
    """

    criterion: str
    category: Category
    detail: str

    def __post_init__(self):
        self.category = Category.of(self.category)


def rank_descending(values) -> np.ndarray:
    """Descending ranks (1 = largest value); exact ties share their average rank.

    The single rank definition of the package: the results table, the ranking
    plot and the displacement rule in :func:`classify` all use it, so the
    ranks printed next to a diagnosis are the ranks the diagnosis was made
    from. Tied values (e.g. equal declared weights) never produce arbitrary
    input-order displacements.
    """
    from scipy.stats import rankdata

    return rankdata(-np.asarray(values, dtype=float), method="average")


def classify(
    criteria_names,
    weights: np.ndarray,
    S1: np.ndarray,
    ST: np.ndarray,
    thresholds: DiagnosisThresholds | None = None,
) -> list[CriterionDiagnosis]:
    """Label every criterion with the first matching discrepancy category."""
    t = thresholds or DiagnosisThresholds()
    m = len(criteria_names)
    equal_share = 1.0 / m
    rank_w = rank_descending(weights)
    rank_s1 = rank_descending(S1)

    diagnoses = []
    for i, name in enumerate(criteria_names):
        w, s1, st = weights[i], S1[i], ST[i]
        gap = st - s1
        displacement = abs(rank_w[i] - rank_s1[i])

        if w < t.hidden_weight_factor * equal_share and st - w >= t.hidden_st_excess:
            category = HIDDEN_INFLUENCE
            detail = (
                f"w={w:.4f} < {t.hidden_weight_factor:.2f}/m={t.hidden_weight_factor * equal_share:.4f} "
                f"while ST={st:.4f} (ST-w={st - w:.4f})"
            )
        elif st > 0 and gap / st >= t.interaction_ratio and gap >= t.interaction_abs:
            category = INTERACTION_DOMINANCE
            detail = f"interactions carry {gap / st:.0%} of the total effect (ST-S1={gap:.4f})"
        elif displacement >= t.rank_displacement or (
            w < t.negligible_weight and st >= t.influential_st
        ):
            category = MODERATE_DISCREPANCY
            if displacement >= t.rank_displacement:
                detail = (
                    f"rank(w)={rank_w[i]:g} vs rank(S1)={rank_s1[i]:g} "
                    f"(displaced by {displacement:g})"
                )
            else:
                detail = f"w={w:.4f} dismisses a criterion with ST={st:.4f}"
        else:
            category = CONFIRMED_TRANSPARENCY
            detail = f"w={w:.4f}, S1={s1:.4f}, ST={st:.4f} agree (ST-S1={gap:.4f})"

        diagnoses.append(CriterionDiagnosis(str(name), category, detail))
    return diagnoses


def sweep_thresholds(
    criteria_names,
    weights: np.ndarray,
    S1: np.ndarray,
    ST: np.ndarray,
    base: DiagnosisThresholds | None = None,
    **grid,
) -> pd.DataFrame:
    """Classify the criteria for every combination of threshold values.

    Parameters
    ----------
    criteria_names, weights, S1, ST
        The inputs of :func:`classify`.
    base : DiagnosisThresholds, optional
        Thresholds that stay fixed; defaults to :class:`DiagnosisThresholds`.
    **grid
        Threshold field names mapped to the values to sweep, e.g.
        ``hidden_st_excess=[0.01, 0.03, 0.05]``. At least one is required.

    Returns
    -------
    DataFrame
        One row per grid point (Cartesian product, first name slowest): the
        swept threshold columns followed by one :class:`Category` column per
        criterion. Compare rows to see how sensitive the report is to the
        thresholds.
    """
    import itertools
    from dataclasses import fields, replace

    if not grid:
        raise ValueError("sweep_thresholds needs at least one threshold to sweep")
    known = {f.name for f in fields(DiagnosisThresholds)}
    unknown = sorted(set(grid) - known)
    if unknown:
        raise TypeError(f"unknown thresholds {unknown}; expected names from {sorted(known)}")
    base = base or DiagnosisThresholds()
    names = list(grid)
    rows = []
    for values in itertools.product(*(list(grid[name]) for name in names)):
        settings = dict(zip(names, values))
        diagnoses = classify(criteria_names, weights, S1, ST, replace(base, **settings))
        rows.append({**settings, **{d.criterion: d.category for d in diagnoses}})
    return pd.DataFrame(rows, columns=[*names, *map(str, criteria_names)])


def diagnosis_frame(diagnoses: list[CriterionDiagnosis]) -> pd.DataFrame:
    """Diagnoses as a DataFrame."""
    return pd.DataFrame(
        {
            "Criterion": [d.criterion for d in diagnoses],
            "Category": [d.category for d in diagnoses],
            "Detail": [d.detail for d in diagnoses],
        }
    )
