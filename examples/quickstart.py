"""Run a small ESP-COMET study without a dataset.

Run from the repository root: python examples/quickstart.py
Outputs are written to examples/output/quickstart/.
"""

from pathlib import Path

import numpy as np
from pymcdm.methods import COMET
from pymcdm.methods.comet_tools import ESPExpert

from gcisens import SobolStudy

bounds = np.array([[0, 10], [0, 100]], dtype=float)
expert = ESPExpert(esps=np.array([[3, 80]], dtype=float), bounds=bounds)
model = COMET(expert.make_cvalues_psi(), expert)
result = SobolStudy(
    model,
    criteria_names=["Cost", "Quality"],
    n_samples=2048,
    sampler="sobol",
    seed=42,
).run()

print(f"Weight source: {result.weights_source}")
print(result.table().round(4).to_string(index=False))
output = Path(__file__).parent / "output" / "quickstart"
result.to_csv(output)
ax = result.plot_indices()
ax.figure.savefig(output / "indices.png", dpi=150, bbox_inches="tight")
print(f"Outputs written to {output}")
