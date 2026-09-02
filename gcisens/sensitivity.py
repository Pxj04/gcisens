"""Variance-based (Sobol') sensitivity analysis of a scoring model.

Thin wrapper around SALib: Saltelli / Sobol' sampling over the criteria
bounds, model evaluation, and index estimation with bootstrap confidence
intervals — Algorithm 1 of Sałabun et al. (ISD 2025).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from SALib.analyze import sobol as sobol_analyze
from SALib.sample import saltelli as saltelli_sample
from SALib.sample import sobol as sobol_sample

SAMPLERS = ("saltelli", "sobol")


@dataclass
class SobolIndices:
    """Sobol' sensitivity indices of a model over its criteria space."""

    S1: np.ndarray
    ST: np.ndarray
    S1_conf: np.ndarray
    ST_conf: np.ndarray
    S2: np.ndarray | None
    S2_conf: np.ndarray | None
    criteria_names: list[str]
    n_samples: int
    n_evaluations: int
    sampler: str
    output_mean: float = field(default=float("nan"))
    output_std: float = field(default=float("nan"))

    @property
    def interaction(self) -> np.ndarray:
        """Interaction share per criterion: ``ST - S1``."""
        return self.ST - self.S1

    def s2_pairs(self) -> pd.DataFrame:
        """Pairwise interactions sorted by absolute ``S2``, with significance."""
        if self.S2 is None:
            raise ValueError("Second-order indices were not computed (second_order=False)")
        rows = []
        names = self.criteria_names
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                s2, conf = self.S2[i, j], self.S2_conf[i, j]
                rows.append(
                    {
                        "criterion_i": names[i],
                        "criterion_j": names[j],
                        "S2": s2,
                        "S2_conf": conf,
                        "significant": bool(abs(s2) > 2 * conf and abs(s2) > 0.01),
                    }
                )
        df = pd.DataFrame(rows)
        return df.reindex(df["S2"].abs().sort_values(ascending=False).index).reset_index(drop=True)


def sobol_analysis(
    score_fn,
    bounds: np.ndarray,
    criteria_names: list[str],
    n_samples: int = 2048,
    second_order: bool = True,
    seed: int | None = None,
    sampler: str = "saltelli",
) -> SobolIndices:
    """Run the full sampling -> evaluation -> analysis pipeline.

    Parameters
    ----------
    score_fn : callable
        ``f(X) -> scores`` for a 2-D matrix of alternatives.
    bounds : ndarray of shape (m, 2)
        Criteria bounds sampled uniformly (Saltelli scheme).
    criteria_names : list of str
        Criteria names for reporting.
    n_samples : int
        Base sample size N; total model evaluations are ``N * (2m + 2)`` with
        second-order indices and ``N * (m + 2)`` without.
    second_order : bool
        Whether to estimate pairwise interaction indices (S2).
    seed : int or None
        Seed for the bootstrap confidence intervals (and for the ``"sobol"``
        sampler's scrambling). The ``"saltelli"`` sampler itself is
        deterministic.
    sampler : {"saltelli", "sobol"}
        ``"saltelli"`` reproduces the sampling used in the source articles;
        ``"sobol"`` is SALib's current recommended (scrambled) sampler.
    """
    if sampler not in SAMPLERS:
        raise ValueError(f"sampler must be one of {SAMPLERS}, got {sampler!r}")
    problem = {
        "num_vars": len(criteria_names),
        "names": list(criteria_names),
        "bounds": np.asarray(bounds, dtype=float).tolist(),
    }
    if sampler == "saltelli":
        with warnings.catch_warnings():
            # The articles used the classic Saltelli sampler; keep it available
            # without surfacing SALib's deprecation notice to users.
            warnings.simplefilter("ignore", DeprecationWarning)
            X = saltelli_sample.sample(problem, n_samples, calc_second_order=second_order)
    else:
        X = sobol_sample.sample(problem, n_samples, calc_second_order=second_order, seed=seed)

    Y = np.asarray(score_fn(X), dtype=float).ravel()
    if Y.shape[0] != X.shape[0]:
        raise ValueError(f"score_fn returned {Y.shape[0]} values for {X.shape[0]} samples")

    Si = sobol_analyze.analyze(
        problem, Y, calc_second_order=second_order, print_to_console=False, seed=seed
    )

    return SobolIndices(
        S1=np.asarray(Si["S1"]),
        ST=np.asarray(Si["ST"]),
        S1_conf=np.asarray(Si["S1_conf"]),
        ST_conf=np.asarray(Si["ST_conf"]),
        S2=np.asarray(Si["S2"]) if second_order else None,
        S2_conf=np.asarray(Si["S2_conf"]) if second_order else None,
        criteria_names=list(criteria_names),
        n_samples=n_samples,
        n_evaluations=X.shape[0],
        sampler=sampler,
        output_mean=float(Y.mean()),
        output_std=float(Y.std()),
    )
