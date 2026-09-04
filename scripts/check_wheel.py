"""Install and check one built wheel in a fresh environment outside the checkout."""

import argparse
import os
import subprocess
import tempfile
import venv
from pathlib import Path

SMOKE = """
from importlib.metadata import version
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import gcisens
from gcisens import SobolStudy

assert Path(gcisens.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
assert gcisens.__version__ == version("gcisens")
result = SobolStudy(
    lambda X: X[:, 0] + 2 * X[:, 1],
    bounds=[[0, 1], [0, 1]], criteria_names=["A", "B"],
    weights=[1/3, 2/3], sampler="sobol", n_samples=1024, seed=42,
).run()
np.testing.assert_allclose(result.sobol.S1, [0.2, 0.8], atol=0.01)
np.testing.assert_allclose(result.sobol.ST, [0.2, 0.8], atol=0.01)
result.to_csv("report")
assert pd.read_csv("report/results_main.csv").shape[0] == 2
print("Installed wheel: numerical example and CSV export passed")
"""


def check_wheel(wheel: Path) -> None:
    wheel = wheel.resolve(strict=True)
    with tempfile.TemporaryDirectory(prefix="gcisens-wheel-") as directory:
        root = Path(directory)
        environment = root / "venv"
        venv.create(environment, with_pip=True, symlinks=os.name != "nt")
        python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        subprocess.run(
            [
                str(python),
                "-m",
                "pip",
                "install",
                "--quiet",
                "--disable-pip-version-check",
                str(wheel),
            ],
            cwd=root,
            check=True,
        )
        subprocess.run([str(python), "-I", "-c", SMOKE], cwd=root, check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path)
    check_wheel(parser.parse_args().wheel)
