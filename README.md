# gcisens

<!-- about-start -->
[![CI](https://img.shields.io/github/actions/workflow/status/Pxj04/gcisens/ci.yml?branch=main&label=CI&logo=githubactions&logoColor=white)](https://github.com/Pxj04/gcisens/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/gcisens?label=PyPI&logo=pypi&logoColor=white)](https://pypi.org/project/gcisens/)
[![Python](https://img.shields.io/pypi/pyversions/gcisens?logo=python&logoColor=white)](https://pypi.org/project/gcisens/)
[![Documentation Status](https://img.shields.io/readthedocs/gcisens?logo=readthedocs&logoColor=white)](https://gcisens.readthedocs.io/en/latest/)
[![License](https://img.shields.io/badge/License-MIT-yellow?logo=opensourceinitiative&logoColor=white)](https://github.com/Pxj04/gcisens/blob/main/LICENSE)

`gcisens` is a Python library for analysing how individual criteria influence
the results of MCDA models. It uses Sobol' sensitivity analysis to compare
declared or estimated criterion weights with their influence on model scores,
helping identify hidden effects and interactions in ESP-COMET and ESP-SPOTIS
models.

**Documentation:** [Read the Docs](https://gcisens.readthedocs.io/en/latest/)

**PyPI:** [gcisens](https://pypi.org/project/gcisens/)

## Features

### Analysis

- **Global sensitivity analysis** — calculates first-order (S1), total-order
  (ST) and pairwise interaction (S2) Sobol' indices with confidence intervals.
- **Global and local criterion importance** — estimates the model's overall
  criterion weights and measures score changes by varying each criterion across
  its bounds while the others stay at a selected reference point.
- **Ranking comparison** — compares rankings produced by criterion weights, S1
  and ST, including Spearman rank correlations.

### Diagnosis and validation

- **Sensitivity Discrepancy Report** — identifies hidden criterion influence,
  interaction dominance, moderate discrepancies and agreement between weights
  and observed sensitivity.
- **Threshold sweep** — re-classifies the criteria over a grid of threshold
  values to show how stable the report is.
- **External validation** — evaluates model scores against known labels using
  group differences and lift at selected cut-offs.
- **Configuration comparison** — presents results from multiple model setups
  side by side.

### Visualisation and reporting

- **Visual analysis** — includes index charts, S2 interaction heatmaps, ranking
  flows, decision surfaces with ESPs and score distributions.
- **Reusable outputs** — exports results to CSV, LaTeX and standalone HTML
  reports, with run metadata for CSV and HTML exports.

## Analysis pipeline

A study samples the criterion domain, evaluates the model, then compares
criterion weights with sensitivity indices. It returns the tables, plots and
diagnosis through one result object. The steps below can also be used separately.

| Step | Function | Implementation |
|---|---|---|
| Complete workflow | `SobolStudy.run(reference_point=None)` | Coordinates sampling, model evaluation, weight analysis, ranking and diagnosis in `gcisens` |
| Sobol' indices | `sobol_analysis(adapter.scores, adapter.bounds, names, ...)` | Wraps SALib's Sobol or Saltelli sampler and Sobol analyser to calculate S1, ST and S2 indices with confidence intervals |
| Global weights | `grid_regression_weights(score_fn, grid_lines, bounds)` | Fits scikit-learn's `LinearRegression` to the scores over the COMET characteristic-object grid; the COMET adapter passes its own scores and grid lines |
| Local weights | `adapter.local_weights(point, ...)` | Varies each criterion across its bounds while holding the others at the reference point, then normalises the score ranges |
| Discrepancy diagnosis | `classify(criteria_names, weights, S1, ST, ...)` | Applies the `gcisens` classification rules configured through `DiagnosisThresholds` |

## Supported models

Use `esp_comet` or `esp_spotis` to build a model from expected solution points
(ESPs), or pass an existing model to `SobolStudy`. Existing `COMET` and `SPOTIS`
models from [pymcdm](https://github.com/kotbaton/pymcdm) are supported directly.

| Model | Weight handling | Score orientation | Notes |
|---|---|---|---|
| `esp_comet(esps, bounds)` or an existing `COMET` | estimated by regression | higher = preferred | supports one or multiple ESPs; warns before building a large characteristic-object grid |
| `esp_spotis(esp, bounds, weights)` or an existing `SPOTIS` | supplied with the model | lower = closer to the ESP | uses distance scores |
| any callable `f(X) -> scores` | optional | higher = better | enables sensitivity analysis of custom scoring models |

## Installation

You can install `gcisens` from PyPI using pip:

```bash
pip install gcisens
```

`gcisens` requires Python 3.11 or newer. pip installs the required dependencies.
For article reproduction, use the pinned environment described below.

## Quick start

The example below builds an ESP-COMET model with three criteria and one expected
solution point, runs a Sobol' sensitivity analysis and shows how to inspect and
export the results:

```python
import numpy as np
from gcisens import esp_comet, SobolStudy

bounds = np.array([[18, 60], [1, 29], [1009, 19999]], float)
esps = np.array([[25, 25, 2000]])          # expected solution point(s)

model = esp_comet(
    esps=esps,
    bounds=bounds,
    criteria_names=["Age", "Distance", "Income"],
)

result = SobolStudy(
    model, n_samples=2048, sampler="sobol", seed=42,
).run()

result.table()        # weights, S1, ST, ranks, category per criterion
result.diagnosis()    # Sensitivity Discrepancy Report
result.summary()      # r2_fit, r2_samples, ΣS1, ΣST, Spearman correlations
result.plot_indices() # w vs S1 vs ST bar chart
result.to_latex()     # publication-ready table
```

Save the results as a CSV bundle or a standalone HTML report:

```python
result.to_csv("results")
result.to_html("report.html")
```

Alternatively, construct the model directly with the re-exported
[pymcdm](https://github.com/kotbaton/pymcdm) classes and pass it to `SobolStudy`:

```python
from gcisens import COMET, ESPExpert, SobolStudy

expert = ESPExpert(esps=esps, bounds=bounds)
model = COMET(expert.make_cvalues_psi(), expert)
result = SobolStudy(
    model, bounds=bounds, n_samples=2048, sampler="sobol", seed=42,
).run()
```

The analysis samples independent, uniform inputs over the supplied bounds.
`S1` measures the share of score variance due to one criterion alone; `ST`
also includes its interactions. The ranks compare criterion importance.
The diagnosis uses configurable thresholds to flag differences between weights
and sensitivity indices.

### What you get

| Call | Output |
|---|---|
| `result.table()` | weights, S1, ST, ST − S1, confidence half-widths, ranks and category per criterion |
| `result.diagnosis()` | Sensitivity Discrepancy Report: category and rationale per criterion |
| `result.summary()` | `r2_fit`, `r2_samples`, ΣS1, ΣST, Spearman correlations and the run configuration |
| `result.metadata()` | run settings, model configuration and dependency versions as a JSON-compatible record |
| `result.s2_table()` | pairwise interaction indices with significance |
| `result.sweep_thresholds(interaction_ratio=[0.2, 0.3, 0.4])` | categories over a grid of threshold values |
| `result.validate(X, labels)` | group differences and lift@k of the scores against known labels |
| `result.plot_indices()`, `plot_s2_heatmap()`, `plot_rankings()`, `plot_surface()`, `plot_validation()` | Matplotlib axes |
| `result.to_csv(dir)`, `to_latex()`, `s2_to_latex()`, `to_html(path)` | CSV bundle with run metadata, LaTeX tables, standalone HTML report |
| `compare({"ESP1": r1, "ESP2": r2})` | configurations side by side, with `.table()`, `.to_latex()` and `.to_csv()` |

## Examples

Start with the [synthetic quick start](https://github.com/Pxj04/gcisens/blob/main/examples/quickstart.py),
which needs no dataset. The repository also includes examples based on the bundled IBM HR
Attrition dataset (a fictional IBM Watson Analytics sample, ODbL; see
[`examples/data/README.md`](https://github.com/Pxj04/gcisens/blob/main/examples/data/README.md)):

- [`article_esp_comet.py`](https://github.com/Pxj04/gcisens/blob/main/examples/article_esp_comet.py) - analyses multiple
  ESP-COMET configurations and generates CSV, LaTeX and HTML reports.
- [`esp_spotis_demo.py`](https://github.com/Pxj04/gcisens/blob/main/examples/esp_spotis_demo.py) - compares ESP-SPOTIS
  with ESP-COMET using the same expected solution point.

To reproduce the article outputs, install the pinned dependencies from
[`requirements-repro.txt`](https://github.com/Pxj04/gcisens/blob/main/requirements-repro.txt)
and run `examples/article_esp_comet.py`. The script saves a reproduction record
with source and data hashes alongside the reports. It explicitly uses the
Saltelli sampler to retain the article setup; the quick start above uses the
Sobol sampler with a fixed seed.

The [advanced guide](https://gcisens.readthedocs.io/en/latest/advanced.html)
shows local weights, validation, configuration comparison and custom scoring
functions. The [methodology page](https://gcisens.readthedocs.io/en/latest/methodology.html)
lists the assumptions behind the indices and shows how the report reacts to
its thresholds. The [troubleshooting page](https://gcisens.readthedocs.io/en/latest/troubleshooting.html)
lists the warnings and errors with their causes and fixes.

## Development and releases

See [CONTRIBUTING.md](https://github.com/Pxj04/gcisens/blob/main/CONTRIBUTING.md)
for the repository map, development checks and release checklist. It explains
which files to update for code changes, documentation changes and new releases.

## Citing gcisens

If `gcisens` contributes to a scientific publication, please cite the archived
software release:

> Świder, A., & Śniegowski, S. (2026). *gcisens* (Version 0.1.3) [Computer
> software]. Zenodo. https://doi.org/10.5281/zenodo.22384207

Or use BibTeX:

```bibtex
@software{swider_gcisens_2026,
  author    = {Adrianna Świder and Szymon Śniegowski},
  title     = {gcisens},
  year      = {2026},
  publisher = {Zenodo},
  version   = {0.1.3},
  doi       = {10.5281/zenodo.22384207},
  url       = {https://doi.org/10.5281/zenodo.22384207}
}
```

DOI: [10.5281/zenodo.22384207](https://doi.org/10.5281/zenodo.22384207)

<!-- about-end -->

## References

The first two entries are the source articles of the workflow.

- Śniegowski, S., Świder, A., Shekhovtsov, A., & Sałabun, W. (2026).
  Detecting Hidden Criterion Influence When Weights Mislead in Rule-Based
  Decision Support Systems. *30th International Conference on Knowledge-Based
  and Intelligent Information & Engineering Systems (KES 2026)*, to appear.
- Sałabun, W., Shekhovtsov, A., & Wątróbski, J. (2025). [Variance-Based
  Analysis of Global Criteria Importance in the ESP-COMET
  Method](https://doi.org/10.62036/ISD.2025.83). In I. Luković et al. (Eds.),
  *Empowering the Interdisciplinary Role of ISD in Addressing Contemporary
  Issues in Digital Transformation (ISD 2025 Proceedings)*. University of
  Gdańsk & University of Belgrade. ISBN 978-83-972632-1-5.
- Więckowski, J., Kizielewicz, B., Shekhovtsov, A., & Sałabun, W. (2023).
  [How do the criteria affect sustainable supplier evaluation? A case study
  using multi-criteria decision analysis methods in a fuzzy environment
  (local and global importance weights)](https://doi.org/10.1142/S0219622022500948).
  *International Journal of Information Technology & Decision Making, 22*(6).
- Shekhovtsov, A., & Sałabun, W. (2024). [Comparing Global and Local Weights
  in Multi-Criteria Decision-Making: A COMET-Based
  Approach](https://doi.org/10.5220/0012360700003636). *Proceedings of the
  16th International Conference on Agents and Artificial Intelligence
  (ICAART 2024)*.

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
distribute it under its terms. See the
[LICENSE](https://github.com/Pxj04/gcisens/blob/main/LICENSE) file for details.
