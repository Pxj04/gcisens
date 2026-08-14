import numpy as np
import pytest


@pytest.fixture
def linear_model():
    """Additive linear scoring function with known variance decomposition."""

    def score(X):
        X = np.atleast_2d(X)
        return 0.7 * X[:, 0] / 10 + 0.3 * X[:, 1] / 10

    bounds = np.array([[0.0, 10.0], [0.0, 10.0]])
    return score, bounds


@pytest.fixture
def hr_setup():
    """Bounds / ESPs / criteria of the KES 2026 case study (dataset-free)."""
    criteria = [
        "Age",
        "DistanceFromHome",
        "MonthlyIncome",
        "NumCompaniesWorked",
        "PercentSalaryHike",
        "TotalWorkingYears",
        "YearsAtCompany",
    ]
    bounds = np.array(
        [[18, 60], [1, 29], [1009, 19999], [0, 9], [11, 25], [0, 40], [0, 40]],
        dtype=float,
    )
    esp1 = np.array([[25, 25, 2000, 7, 12, 2, 1]], dtype=float)
    esp2 = np.array([[50, 15, 5000, 3, 12, 25, 20]], dtype=float)
    return criteria, bounds, esp1, esp2
