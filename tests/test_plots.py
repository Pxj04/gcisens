from dataclasses import replace

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from gcisens import DiagnosisThresholds, SobolStudy, esp_comet, esp_spotis
from gcisens.plots import plot_surface


@pytest.fixture(scope="module")
def comet_result():
    """A study of a real model: only the decision surface needs one."""
    bounds = np.array([[0.0, 10.0], [0.0, 5.0], [0.0, 1.0]])
    model = esp_comet(esps=[[7, 2, 0.5]], bounds=bounds, criteria_names=["A", "B", "C"])
    return SobolStudy(model, n_samples=64, seed=0).run(reference_point=[5, 2.5, 0.5])


def teardown_module():
    plt.close("all")


def test_plot_indices(record):
    ax = record.plot_indices()
    assert len(ax.patches) == 9  # 3 series x 3 criteria


def test_plot_rankings(record):
    ax = record.plot_rankings()
    assert ax.get_ylabel() == "Position in ranking"
    assert [t.get_text() for t in ax.get_xticklabels()] == [v.label for v in record.views]


def test_plot_s2_heatmap(record):
    ax = record.plot_s2_heatmap()
    assert ax.get_images()


def test_plot_s2_heatmap_uses_diagnosis_thresholds(record):
    # S2 = 0.08 and 0.02 (conf 0.01) clear the defaults; 0.005 does not.
    # Each pair is drawn twice (symmetric matrix).
    starred = [t.get_text() for t in record.plot_s2_heatmap().texts if t.get_text().endswith("*")]
    assert sorted(starred) == ["0.020*", "0.020*", "0.080*", "0.080*"]

    strict = replace(record, thresholds=DiagnosisThresholds(s2_significance_factor=10.0))
    assert not any(t.get_text().endswith("*") for t in strict.plot_s2_heatmap().texts)


def test_plot_validation(record):
    ax = record.plot_validation()
    assert "lift@5" in ax.get_title()


def test_plot_validation_requires_a_validation(record):
    with pytest.raises(ValueError, match="validate"):
        replace(record, validation=None).plot_validation()


def test_plot_surface_takes_the_adapter_explicitly(comet_result):
    ax = plot_surface(comet_result, comet_result.adapter, criteria=("A", "B"))
    assert ax.get_xlabel() == "A"
    with pytest.raises(ValueError, match="adapter"):
        plot_surface(comet_result, None)


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
