"""ESP-SPOTIS demo: same case study, SPOTIS scoring, COMET comparison.

SPOTIS takes declared weights as an *input* (they are the explanation a
stakeholder would see), which makes it a natural target for the Sensitivity
Discrepancy Report: do the declared weights match the model's actual variance
structure? Note SPOTIS scores are distances; lower means closer to the ESP.

Run from the repository root:  python examples/esp_spotis_demo.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

from gcisens import SobolStudy, compare, esp_comet, esp_spotis

HERE = Path(__file__).parent

df = pd.read_csv(HERE / "data" / "WA_Fn-UseC_-HR-Employee-Attrition.csv")
criteria = [
    "Age",
    "DistanceFromHome",
    "MonthlyIncome",
    "NumCompaniesWorked",
    "PercentSalaryHike",
    "TotalWorkingYears",
    "YearsAtCompany",
]
bounds = np.array([[df[c].min(), df[c].max()] for c in criteria], dtype=float)
labels = df["Attrition"] == "Yes"
esp1 = np.array([25, 25, 2000, 7, 12, 2, 1], dtype=float)

# Declared weights: what the organisation *believes* drives retention risk.
declared = np.array([0.10, 0.10, 0.25, 0.10, 0.15, 0.15, 0.15])

spotis_model = esp_spotis(esp=esp1, bounds=bounds, weights=declared, criteria_names=criteria)
spotis_result = SobolStudy(spotis_model, n_samples=2048, sampler="saltelli", seed=42).run(
    reference_point=[40, 12, 4448, 2, 12, 15, 7]
)
spotis_result.validate(df[criteria], labels=labels, top_k=[50, 100])

print("=== ESP-SPOTIS ===")
print(spotis_result.table().round(4).to_string(index=False))
print()
print(spotis_result.diagnosis().to_string(index=False))
print()
print(spotis_result.validation)

# Same ESP through COMET, side by side.
comet_model = esp_comet(esps=esp1, bounds=bounds, criteria_names=criteria)
comet_result = SobolStudy(comet_model, n_samples=2048, sampler="saltelli", seed=42).run()

print()
print("=== COMET vs SPOTIS (same ESP) ===")
print(
    compare({"ESP-COMET": comet_result, "ESP-SPOTIS": spotis_result}).table().round(4).to_string()
)
