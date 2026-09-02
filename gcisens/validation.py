"""Model validation against observed outcomes (labels).

Reproduces the validation of Śniegowski et al. (KES 2026), Table 1: score
separation between label groups and lift@k of the top-scored alternatives.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class ValidationResult:
    """Score separation and lift@k for a labelled dataset."""

    groups: pd.DataFrame
    lift: pd.DataFrame
    delta_mean: float
    scores: np.ndarray
    labels: np.ndarray

    def __repr__(self):  # pragma: no cover - cosmetic
        return (
            f"ValidationResult(delta_mean={self.delta_mean:+.4f}, "
            f"lift@k={dict(zip(self.lift['k'], self.lift['lift'].round(2)))})"
        )


def validate_scores(
    scores: np.ndarray,
    labels,
    top_k=(50, 100),
    ascending: bool = False,
) -> ValidationResult:
    """Compare model scores against binary labels.

    Parameters
    ----------
    scores : ndarray
        Model score per alternative.
    labels : array-like of bool
        True for the positive class (e.g. ``Attrition == "Yes"``).
    top_k : iterable of int
        Cut-offs for the lift table. Values above the sample count are capped;
        the table records the effective cut-off.
    ascending : bool
        Set True when *lower* scores mean higher priority (e.g. SPOTIS
        distances, where lower = closer to the expected profile).
    """
    scores = np.asarray(scores, dtype=float).ravel()
    labels = np.asarray(labels, dtype=bool).ravel()
    if scores.shape != labels.shape:
        raise ValueError("scores and labels must have the same length")

    pos, neg = scores[labels], scores[~labels]
    if len(pos) == 0:
        raise ValueError("labels must contain at least one positive")
    if len(neg) == 0:
        raise ValueError("labels must contain at least one negative")
    pos_std = pos.std(ddof=1) if len(pos) > 1 else np.nan
    neg_std = neg.std(ddof=1) if len(neg) > 1 else np.nan
    groups = pd.DataFrame(
        {
            "group": ["positive", "negative"],
            "n": [len(pos), len(neg)],
            "mean_score": [pos.mean(), neg.mean()],
            "median_score": [np.median(pos), np.median(neg)],
            "std_score": [pos_std, neg_std],
        }
    )
    delta_mean = float(pos.mean() - neg.mean())

    base_rate = labels.mean()
    order = np.argsort(scores if ascending else -scores, kind="stable")
    rows = []
    for k in top_k:
        k = min(int(k), len(labels))
        hits = int(labels[order[:k]].sum())
        rate = hits / k
        rows.append(
            {
                "k": k,
                "positives_in_top_k": hits,
                "rate": rate,
                "base_rate": base_rate,
                "lift": rate / base_rate if base_rate > 0 else np.nan,
            }
        )
    lift = pd.DataFrame(rows)

    return ValidationResult(
        groups=groups, lift=lift, delta_mean=delta_mean, scores=scores, labels=labels
    )
