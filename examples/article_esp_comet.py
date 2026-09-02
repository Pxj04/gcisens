"""Reproduce the three experiments of the KES 2026 article.

Śniegowski, Świder, Shekhovtsov, Sałabun — "Detecting Hidden Criterion
Influence When Weights Mislead in Rule-Based Decision Support Systems".

Three ESP-COMET configurations on the IBM HR Attrition dataset:
ESP1 (early-career profile), ESP2 (career-stagnation profile), ESP1+ESP2
(dual-profile). Outputs: tables, diagnosis, LaTeX, CSV and an HTML report
per configuration, plus the cross-configuration comparison (Table 5).

Run from the repository root:  python examples/article_esp_comet.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

from gcisens import SobolStudy, compare, esp_comet
from gcisens.export import comparison_to_latex, s2_to_latex

HERE = Path(__file__).parent
OUT = HERE / "output"

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

esp1 = [25, 25, 2000, 7, 12, 2, 1]  # young, mobile, low pay, short tenure
esp2 = [50, 15, 5000, 3, 12, 25, 20]  # long tenure, limited advancement

# Reference employee for local weights (article, Section 5: IQR-interior).
reference_employee = np.array([40, 12, 4448, 2, 12, 15, 7], dtype=float)

configurations = {
    "ESP1": [esp1],
    "ESP2": [esp2],
    "ESP1+ESP2": [esp1, esp2],
}

results = {}
for name, esps in configurations.items():
    print(f"=== {name} ===")
    model = esp_comet(esps=np.array(esps, dtype=float), bounds=bounds, criteria_names=criteria)
    result = SobolStudy(model, n_samples=2048, second_order=True, seed=42).run(
        reference_point=reference_employee
    )
    result.validate(df[criteria], labels=labels, top_k=[50, 100, 200])
    results[name] = result

    print(result.table().round(4).to_string(index=False))
    print()
    print(result.diagnosis().to_string(index=False))
    print()

    slug = name.lower().replace("+", "_")
    out = OUT / slug
    result.to_csv(out)
    result.to_latex(
        out / "table_main.tex", caption=f"{name}: weights and Sobol' indices.", label=f"tab:{slug}"
    )
    s2_to_latex(result, out / "table_s2.tex")
    result.to_html(out / "report.html", title=f"Sensitivity Discrepancy Report — {name}")

comparison = compare(results)
print("=== Cross-configuration comparison (cf. Table 5) ===")
print(comparison.table().round(4).to_string())
comparison.to_csv(OUT / "comparison.csv")
comparison_to_latex(comparison, OUT / "comparison.tex")
print(f"\nAll outputs written to {OUT}/")
