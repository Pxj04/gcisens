# gcisens

[![Documentation Status](https://readthedocs.org/projects/gcisens/badge/?version=latest)](https://gcisens.readthedocs.io/en/latest/?badge=latest)

Variance-based (Sobol') sensitivity analysis and **Sensitivity Discrepancy
Report** for rule-based MCDA models — ESP-COMET and ESP-SPOTIS.

Criteria weights are the standard explanation of MCDA models, but they capture
only linear effects. `gcisens` checks whether the weights of a model faithfully
represent its global behaviour, and flags criteria whose influence is hidden in
nonlinearities and interactions.

Based on:

- Śniegowski, Świder, Shekhovtsov, Sałabun — *Detecting Hidden Criterion Influence
  When Weights Mislead in Rule-Based Decision Support Systems* (KES 2026)
- Sałabun, Shekhovtsov, Wątróbski — *Variance-Based Analysis of Global Criteria
  Importance in the ESP-COMET Method* (ISD 2025)

## Install

```bash
pip install gcisens
```

## Quick start

```python
import numpy as np
from gcisens import esp_comet, SobolStudy

bounds = np.array([[18, 60], [1, 29], [1009, 19999]], float)
esps = np.array([[25, 25, 2000]])          # expected solution point(s)

model = esp_comet(esps=esps, bounds=bounds,
                  criteria_names=["Age", "Distance", "Income"])

result = SobolStudy(model, n_samples=2048, seed=42).run()

result.table()        # weights, S1, ST, ranks, category per criterion
result.diagnosis()    # Sensitivity Discrepancy Report
result.summary()      # R², ΣS1, ΣST, Spearman correlations
result.plot_indices() # w vs S1 vs ST bar chart
result.to_latex()     # publication-ready table
```

Already know [pymcdm](https://gitlab.com/shekhovtsov/pymcdm)? Build the model
the pymcdm way — the classes are re-exported 1:1 and both styles mix freely:

```python
from gcisens import COMET, ESPExpert, SobolStudy

expert = ESPExpert(esps=esps, bounds=bounds)
model = COMET(expert.make_cvalues_psi(), expert)
result = SobolStudy(model, bounds=bounds).run()
```

## What you get

| Output | Method |
|---|---|
| Global weights (regression on characteristic objects) + R² | `result.summary()` |
| Local weights at a reference point | `result.run(reference_point=...)` |
| Sobol' indices S1 / ST / S2 with confidence intervals | `result.table()`, `result.s2_table()` |
| Per-criterion diagnosis: hidden influence, interaction dominance, moderate discrepancy, confirmed transparency | `result.diagnosis()` |
| Validation against labels (Δmean, lift@k) | `result.validate(X, labels)` |
| Plots (pymcdm style): indices, S2 heatmap, ranking flows, decision surface with ESPs, score distributions | `result.plot_*()` |
| Exports: CSV, LaTeX (article layout), standalone HTML report | `result.to_csv/to_latex/to_html` |
| Side-by-side comparison of configurations | `compare({...}).table()` |

## Supported models

| Model | Declared weights | Notes |
|---|---|---|
| `esp_comet(esps, bounds)` / any `COMET` | extracted by regression | multiple ESPs supported |
| `esp_spotis(esp, bounds, weights)` / any `SPOTIS` | provided by the user | scores are distances (lower = closer) |
| any callable `f(X) -> scores` | optional | fallback for other methods |

Diagnosis thresholds are configurable via `DiagnosisThresholds` (defaults follow
the KES 2026 article).

## Examples

`examples/article_esp_comet.py` reproduces the three experiments of the KES 2026
article on the bundled IBM HR Attrition dataset.

## License

MIT
