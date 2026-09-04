"""Regression cases for result ownership, provenance and labelled validation."""

import json
import re
from dataclasses import FrozenInstanceError, replace

import numpy as np
import pandas as pd
import pytest

import gcisens
from gcisens import DiagnosisThresholds, SobolStudy
from gcisens.diagnosis import classify


def study(seed=42, **kwargs):
    return SobolStudy(
        lambda X: 0.8 * X[:, 0] + 0.2 * X[:, 1],
        bounds=[[0.0, 1.0], [0.0, 1.0]],
        weights=[0.8, 0.2],
        criteria_names=["A", "B"],
        n_samples=64,
        n_r2_samples=64,
        seed=seed,
        **kwargs,
    )


def test_result_arrays_cannot_change_table_or_diagnosis(record):
    before = record.table()
    for values in (record.weights, record.sobol.S1, record.reference_point):
        with pytest.raises(ValueError, match="read-only"):
            values[0] = 99
        with pytest.raises(ValueError):
            values.setflags(write=True)
    with pytest.raises(FrozenInstanceError):
        record.views[0].values = np.ones(3)
    with pytest.raises(FrozenInstanceError):
        record.diagnoses[0].detail = "changed"
    with pytest.raises(FrozenInstanceError):
        record.thresholds = DiagnosisThresholds(hidden_weight_factor=0.25)
    edited_ranks = record.ranks
    edited_ranks["w"][0] = 99
    names = record.criteria_names
    names[0] = "changed"
    pd.testing.assert_frame_equal(record.table(), before)


def test_sobol_views_share_one_canonical_array(record):
    assert record.view("S1").values is record.sobol.S1
    assert record.view("ST").values is record.sobol.ST
    views = [replace(v, values=[0.1, 0.2, 0.3]) if v.key == "S1" else v for v in record.views]
    with pytest.raises(ValueError, match="must match"):
        replace(record, views=views)
    with pytest.raises(ValueError, match="unique keys"):
        replace(record, views=[*record.views, record.views[0]])


def test_replacing_weights_recomputes_diagnosis(record):
    views = [replace(v, values=[0.1, 0.8, 0.1]) if v.key == "w" else v for v in record.views]
    changed = replace(record, views=views)
    expected = classify(
        changed.criteria_names,
        changed.weights,
        changed.sobol.S1,
        changed.sobol.ST,
        changed.thresholds,
    )
    assert changed.diagnoses == tuple(expected)
    assert changed.diagnoses != record.diagnoses
    np.testing.assert_array_equal(changed.table()["Rank_w"], [2.5, 1, 2.5])


def test_input_and_metadata_mutation_does_not_change_saved_result():
    instance = study()
    result = instance.run(reference_point=[0.5, 0.5])
    original = result.metadata()
    instance.adapter.bounds[0, 1] = 5
    instance.seed = 17
    detached = result.metadata()
    detached["bounds"][0][1] = 100
    detached["thresholds"]["hidden_weight_factor"] = 10
    assert result.metadata() == original


def test_result_scoring_keeps_its_adapter_settings():
    model = gcisens.esp_spotis([1.0, 1.0], [[0, 1], [0, 1]], weights=[0.8, 0.2])
    instance = SobolStudy(model, n_samples=64, seed=42)
    result = instance.run()
    X = [[0.1, 0.9], [0.9, 0.1]]
    before = result.validate(X, [True, False], top_k=[1]).scores.copy()
    instance.adapter.weights[:] = [0.2, 0.8]
    after = result.validate(X, [True, False], top_k=[1]).scores
    np.testing.assert_array_equal(before, after)
    with pytest.raises(ValueError, match="read-only"):
        result.adapter.weights[:] = [0.2, 0.8]


def test_recorded_runs_reject_numeric_replacements_with_old_provenance():
    result = study().run()
    views = [replace(v, values=[0.2, 0.8]) if v.key == "w" else v for v in result.views]
    with pytest.raises(ValueError, match="recorded run"):
        replace(result, views=views)
    with pytest.raises(ValueError, match="recorded run"):
        replace(result, sobol=replace(result.sobol, num_resamples=200))
    # Threshold changes reclassify the same numbers and keep valid provenance.
    changed = replace(result, thresholds=DiagnosisThresholds(hidden_weight_factor=0.25))
    assert changed.metadata()["thresholds"]["hidden_weight_factor"] == 0.25


@pytest.mark.parametrize("sampler", ["saltelli", "sobol"])
def test_generated_seed_replays_the_full_study(sampler):
    first = study(seed=None, sampler=sampler).run(reference_point=[0.5, 0.5])
    metadata = first.metadata()
    assert metadata["sampling"]["requested_seed"] is None
    seed = metadata["sampling"]["seed"]
    assert isinstance(seed, int)
    replay = study(seed=seed, sampler=sampler).run(reference_point=[0.5, 0.5])
    pd.testing.assert_frame_equal(first.table(), replay.table())
    assert first.r2_samples == replay.r2_samples


def test_csv_and_html_store_identical_run_metadata(tmp_path):
    result = study(seed=0, conf_level=0.9, local_percent_step=0.05).run(reference_point=[0.5, 0.5])
    written = result.to_csv(tmp_path, prefix="audit")
    path = tmp_path / "audit_metadata.json"
    assert path in written
    csv_metadata = json.loads(path.read_text())
    report = result.to_html(tmp_path / "report.html", include_plots=False).read_text()
    embedded = re.search(r'id="gcisens-metadata">(.*?)</script>', report, re.DOTALL).group(1)
    assert json.loads(embedded) == csv_metadata == result.metadata()
    assert csv_metadata["bounds"] == [[0.0, 1.0], [0.0, 1.0]]
    assert csv_metadata["sampling"]["seed"] == 0
    assert csv_metadata["sampling"]["conf_level"] == 0.9
    assert csv_metadata["local_weights"]["percent_step"] == 0.05
    assert csv_metadata["local_weights"]["reference_point"] == [0.5, 0.5]
    assert csv_metadata["model"]["input_weights"] == [0.8, 0.2]
    assert csv_metadata["versions"]["gcisens"] == gcisens.__version__
    assert "proof of transparency" in report


def test_embedded_metadata_escapes_markup(tmp_path, record):
    malicious = "</script><script>alert(1)</script>"
    record = replace(record, _metadata_json=json.dumps({"model": malicious}))
    report = record.to_html(tmp_path / "report.html", include_plots=False).read_text()
    assert malicious not in report
    embedded = re.search(r'id="gcisens-metadata">(.*?)</script>', report, re.DOTALL).group(1)
    assert json.loads(embedded)["model"] == malicious


@pytest.mark.parametrize("columns", [["B", "WRONG"], ["A", "A"]])
def test_named_validation_rejects_missing_or_duplicate_columns(columns):
    X = pd.DataFrame([[0.1, 0.9], [0.9, 0.1]], columns=columns)
    with pytest.raises(ValueError, match="columns|column names"):
        study().run().validate(X, [True, False], top_k=[1])


def test_validation_aligns_series_labels_by_unique_row_index():
    X = pd.DataFrame({"A": [0.1, 0.9], "B": [0.9, 0.1]}, index=["low", "high"])
    labels = pd.Series([True, False], index=["high", "low"])
    result = study().run()
    validation = result.validate(X, labels, top_k=[1])
    np.testing.assert_array_equal(validation.labels, [False, True])
    assert validation.lift.loc[0, "lift"] == 2
    assert result.validation is validation


@pytest.mark.parametrize("index", [["wrong", "high"], ["high", "high"]])
def test_validation_rejects_unmatched_or_duplicate_row_labels(index):
    X = pd.DataFrame({"A": [0.1, 0.9], "B": [0.9, 0.1]}, index=["low", "high"])
    labels = pd.Series([True, False], index=index)
    with pytest.raises(ValueError, match="index"):
        study().run().validate(X, labels, top_k=[1])


@pytest.mark.parametrize("X", [[0.1, 0.9], [[np.nan, 0.1], [0.9, 0.1]]])
def test_validation_rejects_nonmatrix_or_nonfinite_input(X):
    with pytest.raises(ValueError, match="2-D|finite"):
        study().run().validate(X, [True, False], top_k=[1])


def test_primary_exports_are_the_workflow_and_explicit_legacy_imports_remain():
    assert set(gcisens.__all__) == {
        "SobolStudy",
        "StudyResult",
        "compare",
        "Comparison",
        "esp_comet",
        "esp_spotis",
        "DiagnosisThresholds",
    }
    from pymcdm.methods import COMET as NativeCOMET

    from gcisens import COMET, ESPExpert, Metric, View, sobol_analysis

    assert COMET is NativeCOMET
    assert all(callable(symbol) for symbol in (ESPExpert, Metric, View, sobol_analysis))
