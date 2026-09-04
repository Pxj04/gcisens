"""gcisens: variance-based sensitivity analysis and Sensitivity Discrepancy
Report for rule-based MCDA models (ESP-COMET, ESP-SPOTIS).

The workflow follows two articles:

- Śniegowski, Świder, Shekhovtsov, Sałabun — *Detecting Hidden Criterion
  Influence When Weights Mislead in Rule-Based Decision Support Systems*
  (KES 2026): the Sensitivity Discrepancy Report artifact.
- Sałabun, Shekhovtsov, Wątróbski — *Variance-Based Analysis of Global
  Criteria Importance in the ESP-COMET Method* (ISD 2025): Sobol'/Saltelli
  methodology and regression-based global weights.

Quick start::

    from gcisens import esp_comet, SobolStudy

    model = esp_comet(esps=esps, bounds=bounds, criteria_names=names)
    result = SobolStudy(model, n_samples=2048, seed=42).run()
    result.table(); result.diagnosis(); result.to_latex()

Import COMET, SPOTIS and ESPExpert from pymcdm when using existing models.
Older explicit imports of helper records and re-exported pymcdm classes remain
available for compatibility. The primary interface is listed in __all__.
"""

# ruff: noqa: F401

from importlib.metadata import version as _package_version

from pymcdm.methods import COMET, SPOTIS
from pymcdm.methods.comet_tools import ESPExpert, get_local_weights

from .builders import esp_comet, esp_spotis
from .diagnosis import (
    CATEGORIES,
    CONFIRMED_TRANSPARENCY,
    HIDDEN_INFLUENCE,
    INTERACTION_DOMINANCE,
    MODERATE_DISCREPANCY,
    Category,
    DiagnosisThresholds,
    sweep_thresholds,
)
from .export import comparison_to_latex, s2_to_latex
from .sensitivity import SobolIndices, sobol_analysis
from .study import Comparison, Metric, SobolStudy, StudyResult, View, compare
from .validation import ValidationResult, validate_scores

__version__ = _package_version("gcisens")

__all__ = [
    "SobolStudy",
    "StudyResult",
    "compare",
    "Comparison",
    "esp_comet",
    "esp_spotis",
    "DiagnosisThresholds",
]
