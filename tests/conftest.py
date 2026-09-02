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


@pytest.fixture
def record():
    """A StudyResult built by hand: three criteria, a reference point,
    second-order indices and a validation, but no model or adapter."""
    from gcisens import SobolIndices, StudyResult, View, validate_scores
    from gcisens.diagnosis import classify

    names = ["A", "B", "C"]
    w, w_loc = np.array([0.5, 0.1, 0.4]), np.array([0.45, 0.15, 0.40])
    S1, ST = np.array([0.50, 0.05, 0.30]), np.array([0.60, 0.15, 0.50])
    S2 = np.full((3, 3), np.nan)
    S2[0, 1], S2[0, 2], S2[1, 2] = 0.08, 0.02, 0.005
    sobol = SobolIndices(
        S1=S1, ST=ST, S1_conf=np.full(3, 0.02), ST_conf=np.full(3, 0.03),
        S2=S2, S2_conf=np.full((3, 3), 0.01), criteria_names=names,
        n_samples=64, n_evaluations=512, sampler="saltelli",
    )  # fmt: skip
    views = [
        View("w", "$w$", w),
        View("w_loc", r"$w_{\mathrm{loc}}$", w_loc),
        View("S1", "$S1$", S1),
        View("ST", "$ST$", ST),
    ]
    scores = np.linspace(0, 1, 20)
    return StudyResult(
        views=views,
        sobol=sobol,
        diagnoses=classify(names, w, S1, ST),
        r2_fit=None,
        r2_samples=0.9,
        reference_point=np.array([1.0, 2.0, 3.0]),
        validation=validate_scores(scores, scores > 0.6, top_k=[5]),
    )
