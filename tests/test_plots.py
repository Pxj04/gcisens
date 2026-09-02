import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from gcisens import DiagnosisThresholds, SobolStudy, esp_comet, esp_spotis


@pytest.fixture(scope="module")
def comet_result(request):
    bounds = np.array([[0.0, 10.0], [0.0, 5.0], [0.0, 1.0]])
    model = esp_comet(esps=[[7, 2, 0.5]], bounds=bounds, criteria_names=["A", "B", "C"])
    result = SobolStudy(model, n_samples=64, seed=0).run(reference_point=[5, 2.5, 0.5])
    rng = np.random.default_rng(0)
    X = rng.uniform(bounds[:, 0], bounds[:, 1], size=(100, 3))
    result.validate(X, labels=rng.random(100) > 0.5, top_k=[10])
    return result


def teardown_module():
    plt.close("all")


def test_plot_indices(comet_result):
    ax = comet_result.plot_indices()
    assert len(ax.patches) == 9  # 3 series x 3 criteria


def test_plot_rankings(comet_result):
    ax = comet_result.plot_rankings()
    assert ax.get_ylabel() == "Position in ranking"


def test_plot_s2_heatmap(comet_result):
    ax = comet_result.plot_s2_heatmap()
    assert ax.get_images()


def test_plot_s2_heatmap_uses_diagnosis_thresholds(linear_model):
    score, bounds = linear_model
    result = SobolStudy(
        score,
        bounds=bounds,
        weights=np.array([0.7, 0.3]),
        thresholds=DiagnosisThresholds(s2_significance_factor=0.5),
        n_samples=64,
        seed=0,
    ).run()
    result.sobol.S2[0, 1] = 0.03
    result.sobol.S2_conf[0, 1] = 0.02

    ax = result.plot_s2_heatmap()

    assert any(text.get_text().endswith("*") for text in ax.texts)


def test_plot_validation(comet_result):
    ax = comet_result.plot_validation()
    assert "lift@10" in ax.get_title()


def test_plot_surface_defaults_to_top_st(comet_result):
    ax = comet_result.plot_surface()
    assert ax.get_xlabel() in {"A", "B", "C"}
    assert "Slice at" in ax.get_title()


def test_plot_surface_by_name(comet_result):
    ax = comet_result.plot_surface(criteria=("A", "B"))
    assert ax.get_xlabel() == "A"
    assert ax.get_ylabel() == "B"


def test_plot_surface_two_criteria_model():
    bounds = np.array([[0.0, 1.0], [0.0, 1.0]])
    model = esp_comet(esps=[[0.4, 0.4], [0.8, 0.2]], bounds=bounds, criteria_names=["C1", "C2"])
    result = SobolStudy(model, n_samples=32, seed=0).run()
    ax = result.plot_surface()
    assert ax.get_title() == ""  # no slice note for a true 2-D model


def test_plot_surface_spotis():
    bounds = np.array([[0.0, 10.0], [0.0, 5.0], [0.0, 2.0]])
    model = esp_spotis(
        esp=[7, 2, 1], bounds=bounds, weights=[0.5, 0.3, 0.2], criteria_names=["A", "B", "C"]
    )
    result = SobolStudy(model, n_samples=32, seed=0).run()
    ax = result.plot_surface(criteria=(0, 1))
    # ESP recovered from the SPOTIS model itself and marked on the plot.
    assert any("ESP" in t.get_text() for t in ax.texts)


def _additive(X):
    X = np.atleast_2d(X)
    return 0.7 * X[:, 0] / 10 + 0.3 * X[:, 1] / 5


def test_plot_surface_callable_model_marks_given_esps():
    bounds = np.array([[0.0, 10.0], [0.0, 5.0]])
    result = SobolStudy(_additive, bounds=bounds, n_samples=32, seed=0).run()

    without = result.plot_surface(criteria=(0, 1))
    with_esps = result.plot_surface(criteria=(0, 1), esps=[[7, 2], [3, 4]])

    assert not any("ESP" in t.get_text() for t in without.texts)
    assert [t.get_text() for t in with_esps.texts if "ESP" in t.get_text()] == [
        "$ESP_{1}$",
        "$ESP_{2}$",
    ]


def test_plot_surface_callable_model_uses_study_esps():
    bounds = np.array([[0.0, 10.0], [0.0, 5.0]])
    result = SobolStudy(_additive, bounds=bounds, esps=[[7, 2]], n_samples=32, seed=0).run()

    ax = result.plot_surface()

    assert any("ESP" in t.get_text() for t in ax.texts)
    assert ax.get_title() == ""
