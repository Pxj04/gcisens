"""Variance-based (Sobol') sensitivity analysis of a scoring model.

Thin wrapper around SALib: Saltelli / Sobol' sampling over the criteria
bounds, model evaluation, and index estimation with bootstrap confidence
intervals — Algorithm 1 of Sałabun et al. (ISD 2025).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from numbers import Integral, Real

import numpy as np
import pandas as pd
from SALib.analyze import sobol as sobol_analyze
from SALib.sample import sobol as sobol_sample

from .adapters import validate_bounds, validate_criteria_names
from .diagnosis import DiagnosisThresholds

SAMPLERS = ("saltelli", "sobol")
NON_POWER_OF_TWO_WARNING = (
    "n_samples={n_samples} is not a power of two; Sobol' convergence may be reduced"
)


def validate_integer(value, name: str, minimum: int) -> int:
    """Require an integer without silently rounding or converting text."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer, got {value!r}")  # noqa: TRY004
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}, got {value}")
    return int(value)


def validate_seed(seed) -> int | None:
    """Require a non-negative integer seed, or None."""
    return None if seed is None else validate_integer(seed, "seed", 0)


def validate_bootstrap(num_resamples, conf_level) -> tuple[int, float]:
    """Check the settings used to estimate confidence-interval half-widths."""
    num_resamples = validate_integer(num_resamples, "num_resamples", 2)
    if (
        isinstance(conf_level, (bool, np.bool_))
        or not isinstance(conf_level, Real)
        or not 0 < conf_level < 1
    ):
        raise ValueError("conf_level must be a finite number strictly between 0 and 1")
    return num_resamples, float(conf_level)


def validate_n_samples(n_samples: int) -> int:
    """Return the sample count after checking SALib's minimum."""
    n_samples = validate_integer(n_samples, "n_samples", 2)
    if n_samples & (n_samples - 1):
        warnings.warn(
            NON_POWER_OF_TWO_WARNING.format(n_samples=n_samples),
            UserWarning,
            stacklevel=2,
        )
    return n_samples


def validate_sampler(sampler: str) -> str:
    """Return a supported SALib sampler name."""
    if sampler not in SAMPLERS:
        raise ValueError(f"sampler must be one of {SAMPLERS}, got {sampler!r}")
    return sampler


@dataclass(eq=False, frozen=True)
class SobolIndices:
    """Sobol' sensitivity indices of a model over its criteria space.

    ``S1_conf``, ``ST_conf`` and ``S2_conf`` are bootstrap confidence-interval
    half-widths at :attr:`conf_level`, calculated from :attr:`num_resamples`.
    Instances compare by identity (``eq=False``), as they hold arrays.
    """

    S1: np.ndarray
    ST: np.ndarray
    S1_conf: np.ndarray
    ST_conf: np.ndarray
    S2: np.ndarray | None
    S2_conf: np.ndarray | None
    criteria_names: tuple[str, ...] | list[str]
    n_samples: int
    n_evaluations: int
    sampler: str
    output_mean: float = field(default=float("nan"))
    output_std: float = field(default=float("nan"))
    num_resamples: int = 100
    conf_level: float = 0.95

    def __post_init__(self):
        names = validate_criteria_names(self.criteria_names, len(self.S1))
        if not names:
            raise ValueError("Sobol indices require at least one criterion")
        object.__setattr__(self, "criteria_names", tuple(names))
        m = len(names)
        if (self.S2 is None) != (self.S2_conf is None):
            raise ValueError("S2 and S2_conf must both be present or both be None")
        for name in ("S1", "ST", "S1_conf", "ST_conf", "S2", "S2_conf"):
            value = getattr(self, name)
            if value is None:
                continue
            array = np.asarray(value, dtype=float)
            pairwise = name in ("S2", "S2_conf")
            expected = (m, m) if pairwise else (m,)
            if array.shape != expected:
                raise ValueError(f"{name} must have shape {expected}, got {array.shape}")
            estimates = array[np.triu_indices(m, k=1)] if pairwise else array
            if not np.isfinite(estimates).all():
                raise ValueError(f"{name} estimates must be finite; check model and sample size")
            if name.endswith("_conf") and np.any(estimates < 0):
                raise ValueError(f"{name} confidence half-widths must be non-negative")
            # A bytes buffer also prevents callers from re-enabling WRITEABLE.
            copied = np.frombuffer(array.tobytes(), dtype=float).reshape(array.shape)
            object.__setattr__(self, name, copied)

    @property
    def interaction(self) -> np.ndarray:
        """Interaction share per criterion: ``ST - S1``."""
        return self.ST - self.S1

    def s2_pairs(self, thresholds: DiagnosisThresholds | None = None) -> pd.DataFrame:
        """Pairwise interactions sorted by absolute ``S2``, with significance."""
        if self.S2 is None:
            raise ValueError("Second-order indices were not computed (second_order=False)")
        t = thresholds or DiagnosisThresholds()
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
                        "significant": t.is_s2_significant(s2, conf),
                    }
                )
        df = pd.DataFrame(
            rows, columns=["criterion_i", "criterion_j", "S2", "S2_conf", "significant"]
        )
        return df.reindex(df["S2"].abs().sort_values(ascending=False).index).reset_index(drop=True)


def sobol_analysis(
    score_fn,
    bounds: np.ndarray,
    criteria_names: list[str],
    n_samples: int = 2048,
    second_order: bool = True,
    seed: int | None = None,
    sampler: str = "saltelli",
    num_resamples: int = 100,
    conf_level: float = 0.95,
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
    num_resamples : int
        Number of bootstrap resamples used for confidence intervals.
    conf_level : float
        Confidence level for ``S1_conf``, ``ST_conf`` and ``S2_conf``. These
        values are interval half-widths, not standard errors.
    """
    sampler = validate_sampler(sampler)
    n_samples = validate_n_samples(n_samples)
    bounds = validate_bounds(bounds)
    criteria_names = validate_criteria_names(criteria_names, len(bounds))
    if criteria_names is None:
        raise ValueError("criteria_names must contain one name per criterion")
    seed = validate_seed(seed)
    num_resamples, conf_level = validate_bootstrap(num_resamples, conf_level)
    if not isinstance(second_order, (bool, np.bool_)):
        raise ValueError("second_order must be a boolean")  # noqa: TRY004
    problem = {
        "num_vars": len(criteria_names),
        "names": list(criteria_names),
        "bounds": bounds.tolist(),
    }
    if sampler == "saltelli":
        try:
            from SALib.sample import saltelli as saltelli_sample
        except ImportError as error:
            raise ImportError(
                "The installed SALib has no legacy Saltelli sampler. Use sampler='sobol' "
                "for new analyses, or install the article's pinned environment."
            ) from error
        with warnings.catch_warnings():
            # The articles used the classic Saltelli sampler; keep it available
            # without surfacing SALib's deprecation notice to users.
            warnings.simplefilter("ignore", DeprecationWarning)
            warnings.filterwarnings(
                "ignore",
                message=r"\s*Convergence properties of the Sobol' sequence.*",
                category=UserWarning,
            )
            X = saltelli_sample.sample(problem, n_samples, calc_second_order=second_order)
    else:
        X = sobol_sample.sample(problem, n_samples, calc_second_order=second_order, seed=seed)

    Y = np.asarray(score_fn(X), dtype=float).ravel()
    if Y.shape[0] != X.shape[0]:
        raise ValueError(f"score_fn returned {Y.shape[0]} values for {X.shape[0]} samples")
    if not np.isfinite(Y).all():
        raise ValueError("score_fn must return only finite scores")
    if np.all(Y == Y[0]):
        raise ValueError("Sobol indices are undefined for constant model scores (zero variance)")
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        output_mean, output_std = float(Y.mean()), float(Y.std())
    if not np.isfinite(output_mean) or not np.isfinite(output_std) or output_std == 0:
        raise ValueError(
            "Model score variance is not numerically finite and positive; rescale scores"
        )

    Si = sobol_analyze.analyze(
        problem,
        Y,
        calc_second_order=second_order,
        num_resamples=num_resamples,
        conf_level=conf_level,
        print_to_console=False,
        # SALib 1.5 tests seed truthiness. A SeedSequence keeps seed=0
        # deterministic without changing positive-seed bootstrap samples.
        seed=np.random.SeedSequence(0) if seed == 0 else seed,
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
        num_resamples=num_resamples,
        conf_level=conf_level,
        output_mean=output_mean,
        output_std=output_std,
    )
