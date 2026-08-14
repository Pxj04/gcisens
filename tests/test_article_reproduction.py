"""Regression tests: reproduce the KES 2026 article (Tables 2-5).

Weights are deterministic (regression on the CO grid) and are checked to 4
decimal places. Sobol' indices come from the deterministic Saltelli sequence
and are checked to the article's printed precision with a small tolerance;
bootstrap confidence intervals are seeded but not compared to the article.

Note on ESP2 rank correlations: DistanceFromHome and YearsAtCompany get
numerically-zero global weights (~1e-16), so which of rho(w,S1) / rho(w,ST)
equals 0.9643 vs 1.0000 depends on float dust below any meaningful precision;
the test accepts either split.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gcisens import SobolStudy, compare, esp_comet
from gcisens.diagnosis import CONFIRMED_TRANSPARENCY, HIDDEN_INFLUENCE

DATA = Path(__file__).parent.parent / "examples" / "data" / "WA_Fn-UseC_-HR-Employee-Attrition.csv"
CRITERIA = [
    "Age",
    "DistanceFromHome",
    "MonthlyIncome",
    "NumCompaniesWorked",
    "PercentSalaryHike",
    "TotalWorkingYears",
    "YearsAtCompany",
]
ESP1 = np.array([[25, 25, 2000, 7, 12, 2, 1]], dtype=float)
ESP2 = np.array([[50, 15, 5000, 3, 12, 25, 20]], dtype=float)
REFERENCE_EMPLOYEE = np.array([40, 12, 4448, 2, 12, 15, 7], dtype=float)

# KES 2026, Table 2 (ESP1): w, w_loc, S1, ST.
TABLE2 = np.array(
    [
        [0.1166, 0.1206, 0.0969, 0.0988],
        [0.1298, 0.1291, 0.1164, 0.1209],
        [0.1648, 0.1638, 0.1749, 0.1804],
        [0.0872, 0.0932, 0.0584, 0.0611],
        [0.1550, 0.1537, 0.1624, 0.1645],
        [0.1667, 0.1607, 0.1783, 0.1852],
        [0.1799, 0.1788, 0.1991, 0.2088],
    ]
)

# KES 2026, Table 4 (ESP1+ESP2): w, w_loc, S1, ST.
TABLE4 = np.array(
    [
        [0.0696, 0.0970, 0.0626, 0.0994],
        [0.0957, 0.0769, 0.0366, 0.0489],
        [0.3056, 0.2638, 0.2877, 0.3015],
        [0.0227, 0.0534, 0.0343, 0.0546],
        [0.3877, 0.3928, 0.4583, 0.4652],
        [0.0176, 0.0571, 0.0160, 0.0482],
        [0.1011, 0.0591, 0.0350, 0.0574],
    ]
)


@pytest.fixture(scope="module")
def hr_bounds():
    df = pd.read_csv(DATA)
    assert df.shape[0] == 1470
    return np.array([[df[c].min(), df[c].max()] for c in CRITERIA], dtype=float)


def run_config(bounds, esps):
    model = esp_comet(esps=esps, bounds=bounds, criteria_names=CRITERIA)
    return SobolStudy(model, n_samples=2048, second_order=True, seed=42).run(
        reference_point=REFERENCE_EMPLOYEE
    )


@pytest.mark.slow
def test_experiment_1_reproduces_table_2(hr_bounds):
    res = run_config(hr_bounds, ESP1)
    np.testing.assert_allclose(res.weights, TABLE2[:, 0], atol=5e-5)
    np.testing.assert_allclose(res.local_weights, TABLE2[:, 1], atol=5e-5)
    np.testing.assert_allclose(res.sobol.S1, TABLE2[:, 2], atol=2e-3)
    np.testing.assert_allclose(res.sobol.ST, TABLE2[:, 3], atol=2e-3)

    summary = res.summary()
    assert summary["R2"] == pytest.approx(0.9421, abs=1e-4)
    assert summary["rho_w_S1"] == pytest.approx(1.0)
    assert summary["rho_w_ST"] == pytest.approx(1.0)
    assert all(d.category == CONFIRMED_TRANSPARENCY for d in res.diagnoses)


@pytest.mark.slow
def test_experiment_3_reproduces_table_4(hr_bounds):
    res = run_config(hr_bounds, np.vstack([ESP1, ESP2]))
    np.testing.assert_allclose(res.weights, TABLE4[:, 0], atol=5e-5)
    np.testing.assert_allclose(res.local_weights, TABLE4[:, 1], atol=5e-5)
    np.testing.assert_allclose(res.sobol.S1, TABLE4[:, 2], atol=2e-3)
    np.testing.assert_allclose(res.sobol.ST, TABLE4[:, 3], atol=2e-3)

    summary = res.summary()
    assert summary["R2"] == pytest.approx(0.7628, abs=1e-4)
    assert summary["sum_interaction"] == pytest.approx(0.1447, abs=2e-3)
    assert summary["rho_w_S1"] == pytest.approx(0.8571, abs=1e-4)

    by_name = {d.criterion: d.category for d in res.diagnoses}
    # Section 5.4: the criteria a weight-only reading would dismiss.
    assert by_name["TotalWorkingYears"] == HIDDEN_INFLUENCE
    assert by_name["NumCompaniesWorked"] == HIDDEN_INFLUENCE


@pytest.mark.slow
def test_cross_configuration_comparison_reproduces_table_5(hr_bounds):
    results = {
        "ESP1": run_config(hr_bounds, ESP1),
        "ESP2": run_config(hr_bounds, ESP2),
        "ESP1+ESP2": run_config(hr_bounds, np.vstack([ESP1, ESP2])),
    }
    table = compare(results).table()

    np.testing.assert_allclose(
        table.loc["R2"], [0.9421, 0.8099, 0.7628], atol=1e-4
    )
    np.testing.assert_allclose(
        table.loc["sum_S1"], [0.9865, 0.9840, 0.9305], atol=2e-3
    )
    np.testing.assert_allclose(
        table.loc["sum_ST"], [1.0197, 1.0128, 1.0752], atol=2e-3
    )
    # ESP2 has two numerically-zero weights; the 0.9643/1.0000 split between
    # rho(w,S1) and rho(w,ST) is float-noise-dependent (see module docstring).
    esp2_rhos = sorted([table.loc["rho_w_S1", "ESP2"], table.loc["rho_w_ST", "ESP2"]])
    np.testing.assert_allclose(esp2_rhos, [0.9643, 1.0000], atol=1e-4)


@pytest.mark.slow
def test_validation_reproduces_table_1(hr_bounds):
    df = pd.read_csv(DATA)
    res = run_config(hr_bounds, ESP1)
    val = res.validate(df[CRITERIA], labels=(df["Attrition"] == "Yes"),
                       top_k=[50, 100])
    # KES 2026, Table 1, ESP1 row.
    assert val.delta_mean == pytest.approx(0.1105, abs=1e-3)
    assert val.lift.loc[0, "lift"] == pytest.approx(2.61, abs=0.02)
    assert val.lift.loc[1, "lift"] == pytest.approx(2.48, abs=0.02)
