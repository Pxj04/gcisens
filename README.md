# gcisens

<!-- about-start -->
[![CI](https://img.shields.io/github/actions/workflow/status/Pxj04/gcisens/ci.yml?branch=main&label=CI&logo=githubactions&logoColor=white)](https://github.com/Pxj04/gcisens/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/gcisens?label=PyPI&logo=pypi&logoColor=white)](https://pypi.org/project/gcisens/)
[![Python](https://img.shields.io/pypi/pyversions/gcisens?logo=python&logoColor=white)](https://pypi.org/project/gcisens/)
[![Documentation Status](https://img.shields.io/readthedocs/gcisens?logo=readthedocs&logoColor=white)](https://gcisens.readthedocs.io/en/latest/)
[![License](https://img.shields.io/badge/License-MIT-yellow?logo=opensourceinitiative&logoColor=white)](https://github.com/Pxj04/gcisens/blob/main/LICENSE)

`gcisens` measures how much each criterion really influences an MCDA model.
It runs a Sobol' sensitivity analysis, compares the indices with the declared
or estimated criterion weights, and reports where the two disagree: the
Sensitivity Discrepancy Report. It supports ESP-COMET, ESP-SPOTIS and any
Python scoring function.

| | |
|---|---|
| Documentation | https://gcisens.readthedocs.io |
| PyPI | https://pypi.org/project/gcisens/ |
| Source articles | [References](https://gcisens.readthedocs.io/en/latest/references.html) |

## Installation

```bash
pip install gcisens
```

Requires Python 3.11+, NumPy 2.3+ and pymcdm 1.4+. NumPy 2.3 reproduces the
published multi-ESP results exactly.

## Quick start

```python
import numpy as np

from gcisens import SobolStudy, esp_comet

bounds = np.array([[18, 60], [1, 29], [1009, 19999]], float)
model = esp_comet(
    esps=[[25, 25, 2000]],  # expected solution point(s)
    bounds=bounds,
    criteria_names=["Age", "Distance", "Income"],
)

result = SobolStudy(model, n_samples=2048, seed=42).run()
result.table()  # weights, S1, ST, ranks and category per criterion
result.diagnosis()  # Sensitivity Discrepancy Report
result.to_latex()  # article-ready table
```

## What you get

| Call | Output |
|---|---|
| `result.table()` | weights, S1, ST, ST − S1, confidence half-widths, ranks, category |
| `result.diagnosis()` | category and rationale per criterion: hidden influence, interaction dominance, moderate discrepancy, confirmed transparency |
| `result.summary()` | R² of the linear fits, ΣS1, ΣST, Spearman correlations between weights and indices |
| `result.s2_table()` | pairwise interaction indices with significance |
| `result.validate(X, labels)` | group differences and lift of the scores against known labels |
| `result.plot_indices()`, `plot_s2_heatmap()`, `plot_rankings()`, `plot_surface()`, `plot_validation()` | Matplotlib figures in the pymcdm style |
| `result.to_csv(dir)`, `to_latex()`, `to_html(path)` | CSV bundle, LaTeX tables, standalone HTML report |
| `compare({"ESP1": r1, "ESP2": r2})` | configurations side by side |

## Supported models

| Model | Declared weights | Score orientation |
|---|---|---|
| `esp_comet(esps, bounds)` or any pymcdm `COMET` | estimated by regression on the characteristic objects | higher = closer to the ESP |
| `esp_spotis(esp, bounds, weights)` or any pymcdm `SPOTIS` | given with the model | lower = closer to the ESP |
| any callable `f(X) -> scores` | optional | higher = better |

`COMET`, `SPOTIS` and `ESPExpert` are re-exported from pymcdm, so a model
built the pymcdm way works too: `SobolStudy(COMET(cvalues, expert), bounds=bounds)`.

## Examples

Both scripts use the bundled IBM HR Attrition sample
([origin and licence](https://github.com/Pxj04/gcisens/blob/main/examples/data/README.md)).

- [`examples/article_esp_comet.py`](https://github.com/Pxj04/gcisens/blob/main/examples/article_esp_comet.py) reproduces the ESP-COMET tables of the source article and writes CSV, LaTeX and HTML.
- [`examples/esp_spotis_demo.py`](https://github.com/Pxj04/gcisens/blob/main/examples/esp_spotis_demo.py) compares ESP-SPOTIS with ESP-COMET at the same ESP.

## Citing

Use [`CITATION.cff`](https://github.com/Pxj04/gcisens/blob/main/CITATION.cff)
(GitHub shows "Cite this repository"). The source articles are listed on the
[References](https://gcisens.readthedocs.io/en/latest/references.html) page.

## License

MIT. See [LICENSE](https://github.com/Pxj04/gcisens/blob/main/LICENSE).
