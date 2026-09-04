"""Model validation against observed outcomes (labels).

Reproduces the validation of Śniegowski et al. (KES 2026), Table 1: score
separation between label groups and lift@k of the top-scored alternatives.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

import numpy as np
import pandas as pd


@dataclass(eq=False, frozen=True, init=False)
class ValidationResult:
    """Score separation and lift@k for a labelled dataset.

    Scores and labels are read-only copies. The table properties return
    detached copies so edits cannot change a stored validation report.
    Instances compare by identity (``eq=False``), as they hold arrays.
    """

    _groups: pd.DataFrame
    _lift: pd.DataFrame
    delta_mean: float
    scores: np.ndarray
    labels: np.ndarray

    def __init__(self, groups, lift, delta_mean, scores, labels):
        object.__setattr__(self, "_groups", groups.copy(deep=True))
        object.__setattr__(self, "_lift", lift.copy(deep=True))
        object.__setattr__(self, "delta_mean", float(delta_mean))
        for name, value, dtype in (("scores", scores, float), ("labels", labels, bool)):
            array = np.asarray(value, dtype=dtype)
            copied = np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)
            object.__setattr__(self, name, copied)

    @property
    def groups(self) -> pd.DataFrame:
        """A copy of the score summary for each label group."""
        return self._groups.copy(deep=True)

    @property
    def lift(self) -> pd.DataFrame:
        """A copy of the lift table."""
        return self._lift.copy(deep=True)

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
    labels : array-like of bool or 0/1
        True or 1 for the positive class (e.g. ``Attrition == "Yes"``).
    top_k : iterable of int
        Cut-offs for the lift table. Values above the sample count are capped;
        the table records the effective cut-off.
    ascending : bool
        Set True when *lower* scores mean higher priority (e.g. SPOTIS
        distances, where lower = closer to the expected profile).
    """
    scores = np.array(scores, dtype=float, copy=True)
    labels = np.asarray(labels)
    if scores.ndim != 1 or labels.ndim != 1:
        raise ValueError("scores and labels must be 1-D arrays")
    if scores.shape != labels.shape:
        raise ValueError("scores and labels must have the same length")
    if not np.isfinite(scores).all():
        raise ValueError("scores must contain only finite values")
    if labels.dtype.kind not in "biuf" or not np.isin(labels, [0, 1]).all():
        raise ValueError("labels must contain only binary values: bool or 0/1")
    labels = labels.astype(bool, copy=True)
    try:
        cutoffs = list(top_k)
    except TypeError as error:
        raise ValueError("top_k must be a non-empty iterable of positive integers") from error
    if not cutoffs or any(
        isinstance(k, (bool, np.bool_)) or not isinstance(k, Integral) or k <= 0 for k in cutoffs
    ):
        raise ValueError("top_k must be a non-empty iterable of positive integers")

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
    for k in cutoffs:
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
