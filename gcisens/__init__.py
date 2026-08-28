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

Users familiar with pymcdm can build models the pymcdm way — the relevant
classes are re-exported 1:1 (same classes, not copies), so both styles mix
freely and models stay compatible with the pymcdm ecosystem.
"""
from pymcdm.methods import COMET, SPOTIS
from pymcdm.methods.comet_tools import ESPExpert, get_local_weights

from .builders import esp_comet, esp_spotis
from .diagnosis import (
    CATEGORIES,
    CONFIRMED_TRANSPARENCY,
    HIDDEN_INFLUENCE,
    INTERACTION_DOMINANCE,
    MODERATE_DISCREPANCY,
    DiagnosisThresholds,
)
from .sensitivity import SobolIndices, sobol_analysis
from .study import Comparison, SobolStudy, StudyResult, compare
from .validation import ValidationResult, validate_scores

__version__ = "0.1.1"

__all__ = [
    # workflow
    "SobolStudy",
    "StudyResult",
    "compare",
    "Comparison",
    # builders
    "esp_comet",
    "esp_spotis",
    # diagnosis
    "DiagnosisThresholds",
    "CATEGORIES",
    "HIDDEN_INFLUENCE",
    "INTERACTION_DOMINANCE",
    "MODERATE_DISCREPANCY",
    "CONFIRMED_TRANSPARENCY",
    # building blocks
    "SobolIndices",
    "sobol_analysis",
    "ValidationResult",
    "validate_scores",
    # pymcdm re-exports
    "COMET",
    "SPOTIS",
    "ESPExpert",
    "get_local_weights",
]
