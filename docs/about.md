# About

[![CI](https://img.shields.io/github/actions/workflow/status/Pxj04/gcisens/ci.yml?branch=main&label=CI&logo=githubactions&logoColor=white)](https://github.com/Pxj04/gcisens/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/gcisens?label=PyPI&logo=pypi&logoColor=white)](https://pypi.org/project/gcisens/)
[![Python](https://img.shields.io/pypi/pyversions/gcisens?logo=python&logoColor=white)](https://pypi.org/project/gcisens/)
[![Documentation Status](https://img.shields.io/readthedocs/gcisens?logo=readthedocs&logoColor=white)](https://gcisens.readthedocs.io/en/latest/)
[![License](https://img.shields.io/badge/License-MIT-yellow?logo=opensourceinitiative&logoColor=white)](https://github.com/Pxj04/gcisens/blob/main/LICENSE)

`gcisens` is a Python library for analysing how individual criteria influence
the results of MCDA models. It uses Sobol' sensitivity analysis to compare
declared criteria weights with their actual impact, helping identify hidden
nonlinear effects and interactions in ESP-COMET and ESP-SPOTIS models.

**Documentation:** [Read the Docs](https://gcisens.readthedocs.io/en/latest/)

**PyPI:** [gcisens](https://pypi.org/project/gcisens/)

## Features

### Analysis

- **Global sensitivity analysis** — calculates first-order (S1), total-order
  (ST) and pairwise interaction (S2) Sobol' indices with confidence intervals.
- **Global and local criterion importance** — estimates the model's overall
  criterion weights and can examine importance around a selected reference point.
- **Ranking comparison** — compares rankings produced by criterion weights, S1
  and ST, including Spearman rank correlations.

### Diagnosis and validation

- **Sensitivity Discrepancy Report** — identifies hidden criterion influence,
  interaction dominance, moderate discrepancies and agreement between weights
  and observed sensitivity.
- **External validation** — evaluates model scores against known labels using
  group differences and lift at selected cut-offs.
- **Configuration comparison** — presents results from multiple model setups
  side by side.

### Visualisation and reporting

- **Visual analysis** — includes index charts, S2 interaction heatmaps, ranking
  flows, decision surfaces with ESPs and score distributions.
- **Reusable outputs** — exports results to CSV, LaTeX and standalone HTML
  reports.

## Analysis pipeline

The workflow combines `gcisens` code with established implementations from
[SALib](https://github.com/SALib/SALib),
[pymcdm](https://github.com/kotbaton/pymcdm) and
[scikit-learn](https://scikit-learn.org/).

| Step | Function | Implementation |
|---|---|---|
| Complete workflow | `SobolStudy.run(reference_point=None)` | Coordinates sampling, model evaluation, weight analysis, ranking and diagnosis in `gcisens` |
| Sobol' indices | `sobol_analysis(adapter.scores, adapter.bounds, names, ...)` | Wraps SALib's Sobol or Saltelli sampler and Sobol analyser to calculate S1, ST and S2 indices with confidence intervals |
| Global weights | `comet_global_weights(model, bounds)` | Builds the COMET characteristic-object grid and uses scikit-learn's `LinearRegression` to estimate criterion weights |
| Local weights | `adapter.local_weights(point, ...)` | Uses pymcdm's `get_local_weights` for COMET and the `gcisens` range-sweep implementation for SPOTIS or custom models |
| Discrepancy diagnosis | `classify(criteria_names, weights, S1, ST, ...)` | Applies the `gcisens` classification rules configured through `DiagnosisThresholds` |

## Supported models

The `COMET`, `SPOTIS` and `ESPExpert` implementations are provided by pymcdm
and re-exported by `gcisens` for a consistent interface.

| Model | Weight handling | Notes |
|---|---|---|
| `esp_comet(esps, bounds)` or any `COMET` | estimated by regression | supports one or multiple ESPs |
| `esp_spotis(esp, bounds, weights)` or any `SPOTIS` | supplied with the model | uses distance scores, where lower means closer |
| any callable `f(X) -> scores` | optional | enables sensitivity analysis of custom scoring models |

## Installation

You can install `gcisens` from PyPI using pip:

```bash
pip install gcisens
```

### Reproducibility

`gcisens` requires NumPy 2.3 or newer. NumPy 2.3 reproduces the published
multi-ESP article results, which depend on exact floating-point tie detection
in pymcdm 1.4.

## Quick start

The example below builds an ESP-COMET model with three criteria and one expected
solution point, runs a Sobol' sensitivity analysis and shows how to inspect and
export the results:

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

Alternatively, the model can be constructed directly with the re-exported
[pymcdm](https://github.com/kotbaton/pymcdm) classes and passed to `SobolStudy`:

```python
from gcisens import COMET, ESPExpert, SobolStudy

expert = ESPExpert(esps=esps, bounds=bounds)
model = COMET(expert.make_cvalues_psi(), expert)
result = SobolStudy(model, bounds=bounds).run()
```

## Examples

The repository includes ready-to-run examples based on the bundled IBM HR
Attrition dataset:

- [`article_esp_comet.py`](https://github.com/Pxj04/gcisens/blob/main/examples/article_esp_comet.py) - analyses multiple
  ESP-COMET configurations and generates CSV, LaTeX and HTML reports.
- [`esp_spotis_demo.py`](https://github.com/Pxj04/gcisens/blob/main/examples/esp_spotis_demo.py) - compares ESP-SPOTIS
  with ESP-COMET using the same expected solution point.

## References

- Sobol', I. M. (2001). [Global sensitivity indices for nonlinear mathematical
  models and their Monte Carlo estimates](https://doi.org/10.1016/S0378-4754(00)00270-6).
  *Mathematics and Computers in Simulation, 55*(1-3), 271-280.
- Saltelli, A., Annoni, P., Azzini, I., Campolongo, F., Ratto, M., & Tarantola,
  S. (2010). [Variance based sensitivity analysis of model output: Design and
  estimator for the total sensitivity index](https://doi.org/10.1016/j.cpc.2009.09.018).
  *Computer Physics Communications, 181*(2), 259-270.
- Herman, J., & Usher, W. (2017). [SALib: An open-source Python library for
  sensitivity analysis](https://doi.org/10.21105/joss.00097).
  *Journal of Open Source Software, 2*(9), 97.
- Kizielewicz, B., Shekhovtsov, A., & Sałabun, W. (2023). [pymcdm - The
  universal library for solving multi-criteria decision-making
  problems](https://doi.org/10.1016/j.softx.2023.101368).
  *SoftwareX, 22*, 101368.
- Sałabun, W. (2015). [The Characteristic Objects Method: A new distance-based
  approach to multicriteria decision-making problems](https://doi.org/10.1002/mcda.1525).
  *Journal of Multi-Criteria Decision Analysis, 22*(1-2), 37-50.
- Shekhovtsov, A., Kizielewicz, B., & Sałabun, W. (2023). [Advancing individual
  decision-making: An extension of the Characteristic Objects Method using
  Expected Solution Point](https://doi.org/10.1016/j.ins.2023.119456).
  *Information Sciences, 647*, 119456.
- Dezert, J., Tchamova, A., Han, D., & Tacnet, J.-M. (2020). [The SPOTIS rank
  reversal free method for multi-criteria decision-making support](https://doi.org/10.23919/FUSION45008.2020.9190347).
  *2020 IEEE 23rd International Conference on Information Fusion (FUSION)*, 1-8.
- [IBM HR Analytics Employee Attrition & Performance](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset) -
  IBM Watson Analytics sample dataset used in the examples.

## License

`gcisens` is available under the MIT License. You may use, modify and
distribute it under its terms. See the [LICENSE](https://github.com/Pxj04/gcisens/blob/main/LICENSE) file for details.
