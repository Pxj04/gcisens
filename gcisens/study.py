"""SobolStudy: the orchestrated workflow, and StudyResult: its outcome.

The pipeline (Śniegowski et al., KES 2026; Sałabun et al., ISD 2025):

1. declared/global weights (regression on characteristic objects, or as given),
2. local weights at a reference point (optional),
3. Sobol' indices via Saltelli sampling (S1, ST, S2 + confidence),
4. rankings and Spearman correlations between the two views,
5. per-criterion discrepancy diagnosis.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .adapters import make_adapter
from .diagnosis import (
    CriterionDiagnosis,
    DiagnosisThresholds,
    classify,
    diagnosis_frame,
    rank_descending,
)
from .sensitivity import (
    NON_POWER_OF_TWO_WARNING,
    SobolIndices,
    sobol_analysis,
    validate_n_samples,
    validate_sampler,
)
from .validation import ValidationResult, validate_scores
from .weights import RegressionWeights, regression_weights

logger = logging.getLogger("gcisens")


def _spearman(a, b) -> float:
    """Spearman correlation; NaN (without warnings) for constant inputs."""
    if np.ptp(a) == 0 or np.ptp(b) == 0:
        return float("nan")
    return float(spearmanr(a, b).statistic)


@dataclass
class View:
    """One view of criteria importance: a value per criterion and the ranking
    it induces.

    A study has an ordered set of views — ``w`` (declared / global weights),
    ``w_loc`` (local weights, present only with a reference point), ``S1``
    and ``ST``. The results table, the LaTeX export and the ranking plot all
    walk :attr:`StudyResult.views` in that order, so whether the optional
    view exists is decided once, when the study runs.
    """

    #: Column name in tables and CSV files (``w``, ``w_loc``, ``S1``, ``ST``).
    key: str
    #: Math label shared by LaTeX and matplotlib, e.g. ``$w_{\mathrm{loc}}$``.
    label: str
    values: np.ndarray
    #: Ranks induced by ``values`` (see :func:`gcisens.diagnosis.rank_descending`).
    ranks: np.ndarray = field(init=False)

    def __post_init__(self):
        self.values = np.asarray(self.values, dtype=float)
        self.ranks = rank_descending(self.values)


class SobolStudy:
    """Variance-based sensitivity study of an MCDA scoring model.

    Parameters
    ----------
    model : COMET, SPOTIS or callable
        The scoring model. pymcdm models are recognised automatically; any
        callable ``f(X) -> scores`` works as a fallback (then ``bounds`` and,
        for a discrepancy report, ``weights`` must be given).
    bounds : ndarray of shape (m, 2), optional
        Criteria bounds. Not needed for models built with
        :func:`gcisens.esp_comet` / :func:`gcisens.esp_spotis` (metadata) or
        plain COMET / SPOTIS objects (recovered from the model).
    criteria_names : list of str, optional
        Criteria names for all outputs; defaults to ``C1..Cm``.
    weights : ndarray, optional
        Declared weights — required for SPOTIS, optional for callables.
    types : ndarray, optional
        Criteria types for SPOTIS (ignored when the model has an ESP).
    n_samples : int
        Saltelli base sample size N (total evaluations: ``N * (2m + 2)``).
    second_order : bool
        Estimate pairwise interaction indices (S2).
    seed : int or None
        Seed for bootstrap confidence intervals (and the "sobol" sampler).
    sampler : {"saltelli", "sobol"}
        Sampling scheme; "saltelli" matches the source articles.
    thresholds : DiagnosisThresholds, optional
        Decision-rule thresholds for the discrepancy diagnosis.
    local_percent_step : float
        Step size as a fraction of each criterion range for local-weight
        sweeps. The default is 0.01.

    Examples
    --------
    >>> from gcisens import esp_comet, SobolStudy
    >>> model = esp_comet(esps=esps, bounds=bounds, criteria_names=names)
    >>> result = SobolStudy(model, n_samples=2048, seed=42).run()
    >>> result.table()
    """

    def __init__(
        self,
        model,
        *,
        bounds=None,
        criteria_names=None,
        weights=None,
        types=None,
        n_samples: int = 2048,
        second_order: bool = True,
        seed: int | None = None,
        sampler: str = "saltelli",
        thresholds: DiagnosisThresholds | None = None,
        local_percent_step: float = 0.01,
    ):
        self.adapter = make_adapter(model, bounds, criteria_names, weights, types)
        self.sampler = validate_sampler(sampler)
        self.n_samples = validate_n_samples(n_samples)
        self.second_order = bool(second_order)
        self.seed = seed
        self.thresholds = thresholds or DiagnosisThresholds()
        self.local_percent_step = local_percent_step

    def run(self, reference_point=None) -> StudyResult:
        """Execute the full pipeline and return a :class:`StudyResult`.

        Parameters
        ----------
        reference_point : array-like, optional
            Point in the criteria space at which local weights are computed.
            When omitted, the ``w_loc`` view is left out.
        """
        adapter = self.adapter
        names = adapter.criteria_names
        m = adapter.n_criteria

        logger.info(
            "Sobol study: %d criteria, N=%d (%d evaluations), sampler=%s",
            m,
            self.n_samples,
            self.n_samples * ((2 * m + 2) if self.second_order else (m + 2)),
            self.sampler,
        )

        # 1. Sobol' indices (also provides samples-based regression fallback).
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=NON_POWER_OF_TWO_WARNING.format(n_samples=self.n_samples),
                category=UserWarning,
            )
            sobol = sobol_analysis(
                adapter.scores,
                adapter.bounds,
                names,
                n_samples=self.n_samples,
                second_order=self.second_order,
                seed=self.seed,
                sampler=self.sampler,
            )

        # 2. Declared / global weights + linear-fit quality (R^2).
        declared = adapter.declared_weights()
        if declared is None:
            # No declared weights (bare callable): report regression weights
            # derived from a fresh uniform sample over the bounds.
            fit = self._sample_regression(adapter)
            weights_arr, weights_source, r2 = fit.weights, "regression (samples)", fit.r2
        else:
            weights_arr = np.asarray(declared, dtype=float)
            r2_fn = getattr(adapter, "declared_weights_r2", None)
            r2 = r2_fn() if callable(r2_fn) else None
            weights_source = "regression (characteristic objects)" if r2 is not None else "declared"
            if r2 is None:
                # Declared weights (e.g. SPOTIS): R^2 still reported so the
                # linear-approximation quality is comparable across models.
                r2 = self._sample_regression(adapter).r2

        # 3. Local weights at the reference point.
        local = None
        if reference_point is not None:
            point = np.asarray(reference_point, dtype=float).ravel()
            local = np.asarray(adapter.local_weights(point, percent_step=self.local_percent_step))
        else:
            point = None

        # 4. Views (each ranks itself) + Spearman correlations.
        views = [View("w", "$w$", weights_arr)]
        if local is not None:
            views.append(View("w_loc", r"$w_{\mathrm{loc}}$", local))
        views += [View("S1", "$S1$", sobol.S1), View("ST", "$ST$", sobol.ST)]
        # Tie-aware Spearman on the raw values (equivalent to rank correlation,
        # but exact ties — e.g. two zero weights — get average ranks).
        correlations = {
            "rho_w_S1": _spearman(weights_arr, sobol.S1),
            "rho_w_ST": _spearman(weights_arr, sobol.ST),
            "rho_S1_ST": _spearman(sobol.S1, sobol.ST),
        }
        if local is not None:
            correlations["rho_w_wloc"] = _spearman(weights_arr, local)

        # 5. Discrepancy diagnosis.
        diagnoses = classify(names, weights_arr, sobol.S1, sobol.ST, self.thresholds)

        return StudyResult(
            adapter=adapter,
            weights=weights_arr,
            weights_source=weights_source,
            r2=float(r2),
            local_weights=local,
            reference_point=point,
            sobol=sobol,
            views=views,
            correlations=correlations,
            diagnoses=diagnoses,
            thresholds=self.thresholds,
        )

    def _sample_regression(self, adapter) -> RegressionWeights:
        rng = np.random.default_rng(self.seed)
        X = rng.uniform(adapter.bounds[:, 0], adapter.bounds[:, 1], size=(4096, adapter.n_criteria))
        return regression_weights(X, adapter.scores(X), adapter.bounds)


@dataclass
class StudyResult:
    """All raw outputs of a :class:`SobolStudy` run, with reporting helpers."""

    adapter: object
    weights: np.ndarray
    weights_source: str
    r2: float
    local_weights: np.ndarray | None
    reference_point: np.ndarray | None
    sobol: SobolIndices
    views: list[View]
    correlations: dict
    diagnoses: list[CriterionDiagnosis]
    thresholds: DiagnosisThresholds
    validation: ValidationResult | None = field(default=None)

    @property
    def criteria_names(self) -> list[str]:
        return self.sobol.criteria_names

    @property
    def ranks(self) -> dict[str, np.ndarray]:
        """Ranks per view key (``w``, ``w_loc``, ``S1``, ``ST``).

        A read-only view over :attr:`views`, kept so that code written
        against the ``ranks`` dict of gcisens <= 0.1.2 keeps working.
        """
        return {v.key: v.ranks for v in self.views}

    # ------------------------------------------------------------------ tables
    def table(self) -> pd.DataFrame:
        """Main results table: one value and one ``Rank_`` column per view,
        Sobol' confidence columns, diagnosis category."""
        s = self.sobol
        data = {"Criterion": self.criteria_names}
        for v in self.views:
            data[v.key] = v.values
        data.update({"ST_minus_S1": s.interaction, "S1_conf": s.S1_conf, "ST_conf": s.ST_conf})
        for v in self.views:
            data[f"Rank_{v.key}"] = v.ranks
        data["Category"] = [d.category for d in self.diagnoses]
        return pd.DataFrame(data)

    def s2_table(self) -> pd.DataFrame:
        """Pairwise interaction indices sorted by absolute ``S2`` value."""
        return self.sobol.s2_pairs()

    def diagnosis(self) -> pd.DataFrame:
        """Sensitivity Discrepancy Report: category + rationale per criterion."""
        return diagnosis_frame(self.diagnoses)

    def summary(self) -> pd.Series:
        """Configuration-level metrics (cf. KES 2026, Table 5)."""
        s = self.sobol
        data = {
            "R2": self.r2,
            "sum_S1": float(s.S1.sum()),
            "sum_ST": float(s.ST.sum()),
            "sum_interaction": float(s.interaction.sum()),
            **self.correlations,
            "n_samples": s.n_samples,
            "n_evaluations": s.n_evaluations,
            "sampler": s.sampler,
            "weights_source": self.weights_source,
        }
        return pd.Series(data)

    # -------------------------------------------------------------- validation
    def validate(self, X, labels, top_k=(50, 100), ascending=None) -> ValidationResult:
        """Validate model scores against binary labels (cf. KES 2026, Table 1).

        ``ascending`` defaults to the model's score orientation: False for
        COMET (higher = closer to ESP), True for SPOTIS (lower = closer).
        """
        if ascending is None:
            ascending = not self.adapter.higher_is_closer
        if isinstance(X, pd.DataFrame) and all(name in X.columns for name in self.criteria_names):
            X = X[self.criteria_names]
        X = np.asarray(pd.DataFrame(X).values, dtype=float)
        if X.shape[1] != self.adapter.n_criteria:
            raise ValueError(f"X must have {self.adapter.n_criteria} columns, got {X.shape[1]}")
        scores = self.adapter.scores(X)
        self.validation = validate_scores(scores, labels, top_k=top_k, ascending=ascending)
        return self.validation

    # ------------------------------------------------------------------- plots
    def plot_indices(self, ax=None):
        """Grouped bars: w vs S1 vs ST per criterion, with confidence bars."""
        from . import plots

        return plots.plot_indices(self, ax=ax)

    def plot_s2_heatmap(self, ax=None):
        """Lower-triangular heatmap of pairwise interactions (S2)."""
        from . import plots

        return plots.plot_s2_heatmap(self, ax=ax)

    def plot_rankings(self, ax=None):
        """Bump chart of criteria rankings under w / w_loc / S1 / ST."""
        from . import plots

        return plots.plot_rankings(self, ax=ax)

    def plot_validation(self, ax=None):
        """Score distributions per label group (requires ``validate()`` first)."""
        from . import plots

        return plots.plot_validation(self, ax=ax)

    def plot_surface(self, criteria=None, at=None, esps=None, num=100, ax=None):
        """Decision surface over two criteria with the evaluation grid and ESPs
        (cf. ISD 2025, Figs. 1-2); a 2-D slice for models with more criteria."""
        from . import plots

        return plots.plot_surface(self, criteria=criteria, at=at, esps=esps, num=num, ax=ax)

    # ----------------------------------------------------------------- exports
    def to_csv(self, directory, prefix: str = "results") -> list[Path]:
        """Write results_main / results_s2 / results_validation CSV files."""
        from . import export

        return export.to_csv(self, directory, prefix=prefix)

    def to_latex(self, path=None, caption=None, label=None) -> str:
        """Main results table as a LaTeX table (article layout)."""
        from . import export

        return export.to_latex(self, path=path, caption=caption, label=label)

    def s2_to_latex(self, path=None, top: int = 10, caption=None, label=None) -> str:
        """Pairwise interaction indices as a LaTeX table."""
        from . import export

        return export.s2_to_latex(self, path=path, top=top, caption=caption, label=label)

    def to_html(
        self,
        path,
        title: str = "Sensitivity Discrepancy Report",
        include_plots: bool = True,
    ) -> Path:
        """Standalone dark-theme HTML report with tables, diagnosis and plots."""
        from . import export

        return export.to_html(self, path, title=title, include_plots=include_plots)


class Comparison:
    """Side-by-side comparison of several study results (cf. Table 5)."""

    METRICS = (
        "R2",
        "sum_S1",
        "sum_ST",
        "sum_interaction",
        "rho_w_S1",
        "rho_w_ST",
    )

    def __init__(self, results: dict[str, StudyResult]):
        if not results:
            raise ValueError("compare() needs at least one result")
        self.results = dict(results)

    def table(self) -> pd.DataFrame:
        """Metrics as rows, configurations as columns."""
        cols = {}
        for name, res in self.results.items():
            summary = res.summary()
            cols[name] = [summary[m] for m in self.METRICS]
        return pd.DataFrame(cols, index=list(self.METRICS))

    def to_latex(self, path=None, caption=None, label=None) -> str:
        from . import export

        return export.comparison_to_latex(self, path=path, caption=caption, label=label)

    def to_csv(self, path) -> Path:
        path = Path(path)
        self.table().to_csv(path)
        return path


def compare(results: dict[str, StudyResult]) -> Comparison:
    """Compare several :class:`StudyResult` objects side by side."""
    return Comparison(results)
