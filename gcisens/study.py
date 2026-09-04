"""SobolStudy: the orchestrated workflow, and StudyResult: its outcome.

The pipeline (Śniegowski et al., KES 2026; Sałabun et al., ISD 2025):

1. declared/global weights (regression on characteristic objects, or as given),
2. local weights at a reference point (optional),
3. Sobol' indices via Saltelli sampling (S1, ST, S2 + confidence),
4. rankings and Spearman correlations between the two views,
5. per-criterion discrepancy diagnosis.
"""

from __future__ import annotations

import hashlib
import json
import logging
import platform
import warnings
from dataclasses import asdict, dataclass, field
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .adapters import ModelAdapter, make_adapter
from .diagnosis import (
    CriterionDiagnosis,
    DiagnosisThresholds,
    classify,
    diagnosis_frame,
    rank_descending,
    sweep_thresholds,
)
from .sensitivity import (
    NON_POWER_OF_TWO_WARNING,
    SobolIndices,
    sobol_analysis,
    validate_bootstrap,
    validate_integer,
    validate_n_samples,
    validate_sampler,
    validate_seed,
)
from .validation import ValidationResult, validate_scores
from .weights import RegressionWeights, regression_weights, validate_percent_step

logger = logging.getLogger("gcisens")


def _nan_if_none(value) -> float:
    return float("nan") if value is None else float(value)


def _spearman(a, b) -> float:
    """Spearman correlation; NaN (without warnings) for constant inputs."""
    if np.ptp(a) == 0 or np.ptp(b) == 0:
        return float("nan")
    return float(spearmanr(a, b).statistic)


def _result_digest(views, sobol, point, r2_fit, r2_samples, n_r2_samples, weights_source):
    """Identify numerical data so a copied result cannot retain false provenance."""
    digest = hashlib.sha256()
    for view in views:
        digest.update(view.key.encode())
        digest.update(np.asarray(view.values, dtype="<f8").tobytes())
    for name in ("S1", "ST", "S1_conf", "ST_conf", "S2", "S2_conf"):
        values = getattr(sobol, name)
        if values is not None:
            digest.update(np.asarray(values, dtype="<f8").tobytes())
    digest.update(
        json.dumps(
            {
                "names": list(sobol.criteria_names),
                "point": None if point is None else np.asarray(point).tolist(),
                "r2_fit": r2_fit,
                "r2_samples": r2_samples,
                "n_r2_samples": n_r2_samples,
                "weights_source": weights_source,
                "sampler": sobol.sampler,
                "n_samples": sobol.n_samples,
                "n_evaluations": sobol.n_evaluations,
                "output_mean": sobol.output_mean,
                "output_std": sobol.output_std,
                "num_resamples": sobol.num_resamples,
                "conf_level": sobol.conf_level,
            },
            sort_keys=True,
        ).encode()
    )
    return digest.hexdigest()


@dataclass(frozen=True, eq=False)
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

    def __post_init__(self):
        values = np.asarray(self.values, dtype=float)
        if values.ndim != 1 or not np.isfinite(values).all():
            raise ValueError("view values must be a finite 1-D array")
        object.__setattr__(self, "values", np.frombuffer(values.tobytes(), dtype=float))

    @property
    def ranks(self) -> np.ndarray:
        """Ranks derived from the current values, with average ranks for ties."""
        return rank_descending(self.values)


@dataclass(frozen=True)
class Metric:
    """One configuration-level metric of a study: its summary key, its
    display label and its value.

    The label is defined here, once; :meth:`Comparison.table` and the LaTeX
    comparison writer read it from the results they render.
    """

    #: Key in :meth:`StudyResult.summary`, CSV files and comparison tables.
    key: str
    #: Math label for LaTeX, e.g. ``$\rho(w, S1)$``.
    label: str
    value: float


class SobolStudy:
    """Variance-based sensitivity study of an MCDA scoring model.

    Parameters
    ----------
    model : COMET, SPOTIS or callable
        The scoring model. pymcdm models are recognised automatically; a deterministic
        callable ``f(X) -> scores`` can be used with explicit ``bounds``. Its
        score for a point must not depend on other rows in the batch. Without
        input ``weights``, the study estimates weights by regression.
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
    esps : ndarray of shape (k, m), optional
        Expected Solution Points marked on surface plots. Recovered from
        models built with the builders or from pymcdm ESP models; give them
        for a callable.
    n_samples : int
        Saltelli base sample size N (total evaluations: ``N * (2m + 2)``).
    second_order : bool
        Estimate pairwise interaction indices (S2).
    seed : int or None
        Seed for the bootstrap confidence intervals, the ``"sobol"`` sampler
        and the uniform sample behind ``r2_samples``. The default
        ``"saltelli"`` sampler is deterministic, so ``S1``, ``ST`` and ``S2``
        do not change with the seed.
    sampler : {"saltelli", "sobol"}
        Sampling scheme. The default ``"saltelli"`` matches the source
        articles but SALib deprecates it; use ``"sobol"`` (scrambled Sobol'
        sequence) for new studies.
    num_resamples : int
        Number of bootstrap resamples used for Sobol' confidence intervals.
    conf_level : float
        Confidence level for the Sobol' interval half-widths. The default is
        0.95.
    thresholds : DiagnosisThresholds, optional
        Decision-rule thresholds for the discrepancy diagnosis.
    local_percent_step : float
        Step size as a fraction of each criterion range for local-weight
        sweeps. The default is 0.01.
    n_r2_samples : int
        Number of uniform sample points over ``bounds``. The study computes
        :attr:`StudyResult.r2_samples` on this seeded sample. Must be at
        least the number of criteria plus two. The default is 4096.

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
        esps=None,
        n_samples: int = 2048,
        second_order: bool = True,
        seed: int | None = None,
        sampler: str = "saltelli",
        num_resamples: int = 100,
        conf_level: float = 0.95,
        thresholds: DiagnosisThresholds | None = None,
        local_percent_step: float = 0.01,
        n_r2_samples: int = 4096,
    ):
        self.adapter = make_adapter(model, bounds, criteria_names, weights, types, esps)
        self.n_r2_samples = self._validate_n_r2_samples(n_r2_samples, self.adapter.n_criteria)
        self.sampler = validate_sampler(sampler)
        self.n_samples = validate_n_samples(n_samples)
        if not isinstance(second_order, (bool, np.bool_)):
            raise TypeError("second_order must be a boolean")
        self.second_order = bool(second_order)
        self.seed = validate_seed(seed)
        self.num_resamples, self.conf_level = validate_bootstrap(num_resamples, conf_level)
        self.thresholds = thresholds or DiagnosisThresholds()
        self.local_percent_step = validate_percent_step(local_percent_step)

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
        seed = self.seed
        if seed is None:
            seed = int(np.random.SeedSequence().generate_state(1)[0])

        logger.info(
            "Sobol study: %d criteria, N=%d (%d evaluations), sampler=%s",
            m,
            self.n_samples,
            self.n_samples * ((2 * m + 2) if self.second_order else (m + 2)),
            self.sampler,
        )

        # 1. Sobol' indices. The Saltelli sample is deterministic; ``seed``
        # only drives the bootstrap intervals and the "sobol" sampler.
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
                seed=seed,
                sampler=self.sampler,
                num_resamples=self.num_resamples,
                conf_level=self.conf_level,
            )

        # 2. Declared weights + linear-fit quality (R^2).
        # ``r2_samples`` is a linear fit on one seeded uniform sample over the
        # bounds, computed the same way for every model so that studies can
        # be compared. ``r2_fit`` is the R^2 of the fit that produced the
        # reported weights (None when the weights are declared).
        sample_fit = self._sample_regression(adapter, seed)
        declared = adapter.declared_weights()
        if declared is None:
            # No declared weights (bare callable): report the regression
            # weights from the uniform sample.
            weights_arr, weights_source = sample_fit.weights, "regression (samples)"
            r2_fit = sample_fit.r2
        else:
            weights_arr = np.asarray(declared.weights, dtype=float)
            weights_source, r2_fit = declared.source, declared.r2

        # 3. Local weights at the reference point.
        local = None
        if reference_point is not None:
            point = np.asarray(reference_point, dtype=float).ravel()
            local = np.asarray(adapter.local_weights(point, percent_step=self.local_percent_step))
        else:
            point = None

        # 4. Views (each ranks itself). The record derives the Spearman
        # correlations between them.
        views = [View("w", "$w$", weights_arr)]
        if local is not None:
            views.append(View("w_loc", r"$w_{\mathrm{loc}}$", local))
        views += [View("S1", "$S1$", sobol.S1), View("ST", "$ST$", sobol.ST)]

        # 5. Discrepancy diagnosis.
        diagnoses = classify(names, weights_arr, sobol.S1, sobol.ST, self.thresholds)

        metadata = self._run_metadata(seed)
        metadata["result_sha256"] = _result_digest(
            views, sobol, point, r2_fit, float(sample_fit.r2), self.n_r2_samples, weights_source
        )
        return StudyResult(
            views=views,
            sobol=sobol,
            diagnoses=diagnoses,
            r2_fit=None if r2_fit is None else float(r2_fit),
            r2_samples=float(sample_fit.r2),
            thresholds=self.thresholds,
            weights_source=weights_source,
            reference_point=point,
            n_r2_samples=self.n_r2_samples,
            adapter=adapter,
            _metadata_json=json.dumps(metadata),
        )

    @staticmethod
    def _validate_n_r2_samples(n_r2_samples, n_criteria: int) -> int:
        minimum = n_criteria + 2
        n_r2_samples = validate_integer(n_r2_samples, "n_r2_samples", minimum=minimum)
        return n_r2_samples

    def _sample_regression(self, adapter, seed) -> RegressionWeights:
        """Linear fit on a seeded uniform sample over the bounds."""
        rng = np.random.default_rng(seed)
        X = rng.uniform(
            adapter.bounds[:, 0],
            adapter.bounds[:, 1],
            size=(self.n_r2_samples, adapter.n_criteria),
        )
        return regression_weights(X, adapter.scores(X), adapter.bounds)

    def _run_metadata(self, seed) -> dict:
        """Capture model inputs and environment at run time."""
        adapter = self.adapter
        grid = adapter.grid_lines()
        return {
            "model": {
                "class": adapter.model_identity,
                "higher_is_closer": adapter.higher_is_closer,
                "input_weights": (
                    None if getattr(adapter, "weights", None) is None else adapter.weights.tolist()
                ),
                "types": None if not hasattr(adapter, "types") else adapter.types.tolist(),
                "grid_lines": None if grid is None else [line.tolist() for line in grid],
                "definition": "Keep the model construction script with these results.",
            },
            "bounds": adapter.bounds.tolist(),
            "esps": None if adapter.esps is None else adapter.esps.tolist(),
            "sampling": {"seed": seed, "requested_seed": self.seed},
            "local_weights": {"percent_step": self.local_percent_step, "include_upper": False},
            "versions": {
                "python": platform.python_version(),
                **{
                    name: version(name)
                    for name in (
                        "gcisens",
                        "numpy",
                        "pandas",
                        "scipy",
                        "scikit-learn",
                        "SALib",
                        "pymcdm",
                        "matplotlib",
                    )
                },
            },
            "platform": platform.platform(),
        }


#: Display labels of the summary metrics, defined once; :attr:`StudyResult.metrics`
#: attaches them to the values and every renderer reads them from there.
METRIC_LABELS = {
    "r2_fit": r"$R^2$ (fit)",
    "r2_samples": r"$R^2$ (uniform sample)",
    "sum_S1": r"$\sum S1$",
    "sum_ST": r"$\sum ST$",
    "sum_interaction": r"$\sum (ST - S1)$",
    "rho_w_S1": r"$\rho(w, S1)$",
    "rho_w_ST": r"$\rho(w, ST)$",
    "rho_S1_ST": r"$\rho(S1, ST)$",
    "rho_w_wloc": r"$\rho(w, w_{\mathrm{loc}})$",
}


@dataclass(eq=False, frozen=True)
class StudyResult:
    """A study outcome with read-only numerical data and reporting methods.

    Tables, ranks, correlations and diagnoses use the same Sobol arrays.
    Use ``result.weights.copy()`` or ``result.table()`` for editable data.
    ``validate()`` explicitly attaches a label-validation result. It and
    ``plot_surface()`` require the original model; other reports do not.
    ``metadata()`` returns a detached JSON-compatible run description.
    """

    #: Ordered views: ``w``, ``w_loc`` (only with a reference point), ``S1``, ``ST``.
    views: tuple[View, ...]
    sobol: SobolIndices
    diagnoses: tuple[CriterionDiagnosis, ...]
    #: R^2 of the fit behind the reported weights: the characteristic-object
    #: regression for COMET, the sample regression for a callable without
    #: weights, ``None`` for declared weights (SPOTIS).
    r2_fit: float | None = None
    #: R^2 of a linear fit on one uniform sample over the bounds; computed the
    #: same way for every model, so use it to compare studies. Studies with
    #: the same ``seed`` and ``bounds`` share the sample points.
    r2_samples: float | None = None
    #: Thresholds the diagnoses were made with (also used for S2 significance).
    thresholds: DiagnosisThresholds = field(default_factory=DiagnosisThresholds)
    #: Where the ``w`` view comes from (``declared``, ``regression (...)``).
    weights_source: str = "declared"
    #: Point at which the ``w_loc`` view was computed, if any.
    reference_point: np.ndarray | None = None
    #: Size of the uniform sample behind :attr:`r2_samples`.
    n_r2_samples: int | None = None
    #: Set by :meth:`validate`; ``None`` until then. Exports include the
    #: validation section only when it is set; :meth:`plot_validation`
    #: requires it.
    validation: ValidationResult | None = None
    #: The model handle. Needed only by :meth:`validate` and :meth:`plot_surface`.
    adapter: ModelAdapter | None = None
    _metadata_json: str = field(default="{}", repr=False)

    def __post_init__(self):
        keys = [v.key for v in self.views]
        if not {"w", "S1", "ST"}.issubset(keys):
            raise ValueError(f"views must include 'w', 'S1' and 'ST', got {keys}")
        if len(set(keys)) != len(keys) or set(keys) - {"w", "w_loc", "S1", "ST"}:
            raise ValueError("views must have unique keys from w, w_loc, S1 and ST")
        m = len(self.criteria_names)
        canonical = []
        for view in self.views:
            if len(view.values) != m:
                raise ValueError(
                    f"view {view.key!r} has {len(view.values)} values for {m} criteria"
                )
            if view.key in ("S1", "ST"):
                values = getattr(self.sobol, view.key)
                if not np.array_equal(view.values, values):
                    raise ValueError(f"view {view.key} must match sobol.{view.key}")
                view = View(view.key, view.label, values)
                # The Sobol record is the sole source of these two arrays.
                object.__setattr__(view, "values", values)
            canonical.append(view)
        object.__setattr__(self, "views", tuple(canonical))
        if len(self.diagnoses) != m:
            raise ValueError(f"{len(self.diagnoses)} diagnoses for {m} criteria")
        if [d.criterion for d in self.diagnoses] != self.criteria_names:
            raise ValueError("diagnoses must follow the criteria names in order")
        # Recompute after a dataclasses.replace call as well as a new run.
        object.__setattr__(
            self,
            "diagnoses",
            tuple(
                classify(
                    self.criteria_names, self.weights, self.sobol.S1, self.sobol.ST, self.thresholds
                )
            ),
        )
        if self.reference_point is not None:
            point = np.asarray(self.reference_point, dtype=float)
            if point.shape != (m,) or not np.isfinite(point).all():
                raise ValueError(f"reference_point must contain {m} finite values")
            object.__setattr__(self, "reference_point", np.frombuffer(point.tobytes(), dtype=float))
        # Validate the snapshot now; metadata() always returns a detached copy.
        metadata = json.loads(self._metadata_json)
        recorded = metadata.get("result_sha256")
        if recorded is not None and recorded != _result_digest(
            self.views,
            self.sobol,
            self.reference_point,
            self.r2_fit,
            self.r2_samples,
            self.n_r2_samples,
            self.weights_source,
        ):
            raise ValueError("Numerical data differ from the recorded run; run a new study")
        if self.adapter is not None:
            object.__setattr__(self, "adapter", self.adapter.snapshot())

    # ------------------------------------------------------------- accessors
    @property
    def criteria_names(self) -> list[str]:
        return list(self.sobol.criteria_names)

    def view(self, key: str) -> View | None:
        """The view with the given key (``w``, ``w_loc``, ``S1``, ``ST``) or ``None``."""
        return next((v for v in self.views if v.key == key), None)

    @property
    def weights(self) -> np.ndarray:
        """Values of the ``w`` view."""
        return self.view("w").values

    @property
    def local_weights(self) -> np.ndarray | None:
        """Values of the ``w_loc`` view, or ``None`` without a reference point."""
        local = self.view("w_loc")
        return None if local is None else local.values

    @property
    def r2(self) -> float | None:
        """``r2_fit`` when the weights come from a fit, else ``r2_samples``.

        Kept for code written against gcisens <= 0.1.3, which reported one
        ``r2`` with exactly this meaning. New code should read
        :attr:`r2_fit` or :attr:`r2_samples` explicitly.
        """
        return self.r2_samples if self.r2_fit is None else self.r2_fit

    @property
    def ranks(self) -> dict[str, np.ndarray]:
        """Ranks per view key (``w``, ``w_loc``, ``S1``, ``ST``).

        A read-only view over :attr:`views`, kept so that code written
        against the ``ranks`` dict of gcisens <= 0.1.2 keeps working.
        """
        return {v.key: v.ranks for v in self.views}

    @property
    def correlations(self) -> dict[str, float]:
        """Spearman correlations between the views: ``rho_w_S1``,
        ``rho_w_ST``, ``rho_S1_ST`` and, with a reference point, ``rho_w_wloc``.

        Tie-aware Spearman on the raw values: exact ties (e.g. two zero
        weights) get average ranks. NaN when either view is constant.
        """
        w, s1, st = self.weights, self.sobol.S1, self.sobol.ST
        rho = {
            "rho_w_S1": _spearman(w, s1),
            "rho_w_ST": _spearman(w, st),
            "rho_S1_ST": _spearman(s1, st),
        }
        local = self.local_weights
        if local is not None:
            rho["rho_w_wloc"] = _spearman(w, local)
        return rho

    @property
    def metrics(self) -> list[Metric]:
        """Configuration-level metrics with their display labels: the two R²
        values, the index sums and the view correlations (cf. KES 2026,
        Table 5). ``rho_w_wloc`` is present only with a reference point."""
        s = self.sobol
        values = {
            "r2_fit": _nan_if_none(self.r2_fit),
            "r2_samples": _nan_if_none(self.r2_samples),
            "sum_S1": float(s.S1.sum()),
            "sum_ST": float(s.ST.sum()),
            "sum_interaction": float(s.interaction.sum()),
            **self.correlations,
        }
        return [Metric(key, METRIC_LABELS[key], value) for key, value in values.items()]

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
        return self.sobol.s2_pairs(self.thresholds)

    def diagnosis(self) -> pd.DataFrame:
        """Sensitivity Discrepancy Report: category + rationale per criterion."""
        return diagnosis_frame(self.diagnoses)

    def sweep_thresholds(self, base: DiagnosisThresholds | None = None, **grid) -> pd.DataFrame:
        """Re-classify the criteria over a grid of threshold values.

        ``base`` defaults to :attr:`thresholds`; ``grid`` maps threshold names
        to the values to sweep (see :func:`gcisens.sweep_thresholds`).
        """
        s = self.sobol
        return sweep_thresholds(
            self.criteria_names, self.weights, s.S1, s.ST, base or self.thresholds, **grid
        )

    def metadata(self) -> dict:
        """Return a JSON-compatible copy of the run settings and software versions.

        CSV exports write this as ``{prefix}_metadata.json``. HTML reports
        include the same values. This describes the run; retain the model
        construction script and any input data to repeat it. Hand-built
        records have null values for settings that were not recorded.
        """
        data = json.loads(self._metadata_json)
        s = self.sobol
        data.setdefault("model", None)
        data.setdefault("bounds", None)
        data.setdefault("esps", None)
        data.setdefault("versions", None)
        sampling = data.setdefault("sampling", {"seed": None, "requested_seed": None})
        sampling.update(
            {
                "sampler": s.sampler,
                "n_samples": s.n_samples,
                "n_evaluations": s.n_evaluations,
                "second_order": s.S2 is not None,
                "num_resamples": s.num_resamples,
                "conf_level": s.conf_level,
            }
        )
        local = data.setdefault("local_weights", {"percent_step": None, "include_upper": None})
        local["reference_point"] = (
            None if self.reference_point is None else self.reference_point.tolist()
        )
        data.update(
            {
                "schema_version": 1,
                "criteria_names": self.criteria_names,
                "weights": self.weights.tolist(),
                "weights_source": self.weights_source,
                "thresholds": asdict(self.thresholds),
                "n_r2_samples": self.n_r2_samples,
            }
        )
        return data

    def summary(self) -> pd.Series:
        """Configuration-level metrics (cf. KES 2026, Table 5) followed by
        the run configuration.

        ``r2_fit`` is the R^2 of the fit behind the reported weights (the
        article's value for COMET; NaN for declared weights). ``r2_samples``
        is the R^2 of a linear fit on one uniform sample of ``n_r2_samples``
        points, computed the same way for every model; compare studies on it.

        A Spearman correlation is NaN when either input view is constant,
        because its ranks have zero variance. LaTeX and HTML exports display
        this value as ``n/a``.
        """
        s = self.sobol
        metadata = self.metadata()
        data = {m.key: m.value for m in self.metrics}
        data.update(
            {
                "n_samples": s.n_samples,
                "n_evaluations": s.n_evaluations,
                "n_r2_samples": self.n_r2_samples,
                "sampler": s.sampler,
                "num_resamples": s.num_resamples,
                "conf_level": s.conf_level,
                "weights_source": self.weights_source,
                "seed": metadata["sampling"]["seed"],
                "local_percent_step": metadata["local_weights"]["percent_step"],
            }
        )
        return pd.Series(data)

    # -------------------------------------------------------------- validation
    def _require_adapter(self, operation: str) -> ModelAdapter:
        if self.adapter is None:
            raise ValueError(
                f"{operation} scores new points and needs the model: this result has no "
                "adapter (it was built by hand, not by SobolStudy.run)"
            )
        return self.adapter

    def validate(self, X, labels, top_k=(50, 100), ascending=None) -> ValidationResult:
        """Score ``X`` with the model and validate the scores against binary
        labels (cf. KES 2026, Table 1); stores and returns the result as
        :attr:`validation`.

        ``ascending`` defaults to the model's score orientation: False for
        COMET (higher = closer to ESP), True for SPOTIS (lower = closer).
        Needs :attr:`adapter`.
        """
        adapter = self._require_adapter("validate()")
        if ascending is None:
            ascending = not adapter.higher_is_closer
        if isinstance(X, pd.DataFrame):
            if not X.columns.is_unique:
                raise ValueError("X column names must be unique")
            missing = [name for name in self.criteria_names if name not in X.columns]
            if missing:
                raise ValueError(f"X is missing criteria columns: {missing}")
            if isinstance(labels, pd.Series):
                if not X.index.is_unique or not labels.index.is_unique:
                    raise ValueError("X and labels indexes must be unique for label alignment")
                if len(X) != len(labels) or not X.index.isin(labels.index).all():
                    raise ValueError("labels index must contain exactly the X row labels")
                labels = labels.reindex(X.index)
            X = X[self.criteria_names].to_numpy(dtype=float)
        else:
            X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be a 2-D matrix of alternatives")
        if X.shape[1] != adapter.n_criteria:
            raise ValueError(f"X must have {adapter.n_criteria} columns, got {X.shape[1]}")
        if not np.isfinite(X).all():
            raise ValueError("X must contain only finite values")
        scores = adapter.scores(X)
        validation = validate_scores(scores, labels, top_k=top_k, ascending=ascending)
        object.__setattr__(self, "validation", validation)
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
        """Decision surface over two criteria with the adapter's grid lines and
        ESPs (cf. ISD 2025, Figs. 1-2); a 2-D slice for models with more
        criteria. Needs :attr:`adapter`."""
        from . import plots

        adapter = self._require_adapter("plot_surface()")
        return plots.plot_surface(
            self, adapter, criteria=criteria, at=at, esps=esps, num=num, ax=ax
        )

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
        """Standalone HTML report with settings, tables, diagnosis and plots."""
        from . import export

        return export.to_html(self, path, title=title, include_plots=include_plots)


class Comparison:
    """Side-by-side comparison of several study results (cf. Table 5).

    The rows are the union of the results' :attr:`StudyResult.metrics`, in
    first-seen order; a metric a result lacks (``rho_w_wloc`` without a
    reference point) is NaN in its column. Display labels come from the
    metrics themselves.
    """

    def __init__(self, results: dict[str, StudyResult]):
        if not results:
            raise ValueError("compare() needs at least one result")
        self.results = dict(results)

    def labels(self) -> dict[str, str]:
        """Metric key -> display label, in table row order."""
        labels: dict[str, str] = {}
        for res in self.results.values():
            for metric in res.metrics:
                labels.setdefault(metric.key, metric.label)
        return labels

    def table(self) -> pd.DataFrame:
        """Metrics as rows, configurations as columns."""
        keys = list(self.labels())
        cols = {}
        for name, res in self.results.items():
            values = {m.key: m.value for m in res.metrics}
            cols[name] = [values.get(k, float("nan")) for k in keys]
        return pd.DataFrame(cols, index=keys)

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
