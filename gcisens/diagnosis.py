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

HIDDEN_INFLUENCE = "hidden influence"
INTERACTION_DOMINANCE = "interaction dominance"
MODERATE_DISCREPANCY = "moderate discrepancy"
CONFIRMED_TRANSPARENCY = "confirmed transparency"

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


@dataclass
class CriterionDiagnosis:
    """Diagnostic label for one criterion."""

    criterion: str
    category: str
    detail: str


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


def diagnosis_frame(diagnoses: list[CriterionDiagnosis]) -> pd.DataFrame:
    """Diagnoses as a DataFrame."""
    return pd.DataFrame(
        {
            "Criterion": [d.criterion for d in diagnoses],
            "Category": [d.category for d in diagnoses],
            "Detail": [d.detail for d in diagnoses],
        }
    )
