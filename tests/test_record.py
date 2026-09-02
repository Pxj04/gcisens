"""StudyResult is a plain record: renderers and tests need no model."""

import numpy as np
import pytest

from gcisens import (
    CONFIRMED_TRANSPARENCY,
    HIDDEN_INFLUENCE,
    INTERACTION_DOMINANCE,
    StudyResult,
)


def test_record_is_built_without_a_model(record):
    assert isinstance(record, StudyResult)
    assert record.adapter is None
    assert record.criteria_names == ["A", "B", "C"]
    np.testing.assert_array_equal(record.weights, [0.5, 0.1, 0.4])
    np.testing.assert_array_equal(record.local_weights, [0.45, 0.15, 0.40])
    assert record.r2 == 0.9


def test_record_derives_correlations_from_its_views(record):
    # w = [0.5, 0.1, 0.4] ranks 1, 3, 2; S1 = [0.5, 0.05, 0.3] ranks 1, 3, 2.
    assert record.correlations["rho_w_S1"] == pytest.approx(1.0)
    assert record.correlations["rho_w_ST"] == pytest.approx(1.0)
    assert record.correlations["rho_S1_ST"] == pytest.approx(1.0)
    assert record.correlations["rho_w_wloc"] == pytest.approx(1.0)


def test_table_reads_the_record(record):
    table = record.table()
    assert list(table["Criterion"]) == ["A", "B", "C"]
    assert list(table["Rank_w"]) == [1, 3, 2]
    assert list(table["Category"]) == [
        CONFIRMED_TRANSPARENCY,
        HIDDEN_INFLUENCE,
        INTERACTION_DOMINANCE,
    ]


def test_comparison_reads_metric_labels_from_the_records(record):
    from dataclasses import replace

    from gcisens import compare, comparison_to_latex

    without_local = replace(record, views=[v for v in record.views if v.key != "w_loc"])
    comparison = compare({"with ref": record, "no ref": without_local})

    table = comparison.table()
    assert list(table.index) == [m.key for m in record.metrics]
    assert table.loc["rho_w_wloc", "with ref"] == pytest.approx(1.0)
    assert np.isnan(table.loc["rho_w_wloc", "no ref"])
    assert comparison.labels() == {m.key: m.label for m in record.metrics}

    latex = comparison_to_latex(comparison)
    for metric in record.metrics:
        assert metric.label in latex


def test_comparison_of_records_without_a_reference_point_has_no_local_row(record):
    from dataclasses import replace

    from gcisens import compare

    without_local = replace(record, views=[v for v in record.views if v.key != "w_loc"])
    table = compare({"a": without_local}).table()
    assert "rho_w_wloc" not in table.index
    assert "rho_S1_ST" in table.index


def test_plot_surface_and_validate_need_the_adapter(record):
    with pytest.raises(ValueError, match="adapter"):
        record.plot_surface()
    with pytest.raises(ValueError, match="adapter"):
        record.validate(np.zeros((4, 3)), [True, False, True, False])


def test_html_report_of_a_record_embeds_plots_and_skips_the_surface(tmp_path, record):
    text = record.to_html(tmp_path / "report.html").read_text()
    assert "<img" in text
    assert "Plots skipped" not in text
    assert "Decision surface" not in text
    assert "Validation" in text


def test_html_report_of_a_study_includes_the_decision_surface(tmp_path, linear_model):
    from gcisens import SobolStudy

    score, bounds = linear_model
    result = SobolStudy(score, bounds=bounds, weights=np.array([0.7, 0.3]), n_samples=64).run()
    text = result.to_html(tmp_path / "report.html").read_text()
    assert "alt='Decision surface'" in text
    assert "Plots skipped" not in text


def test_categories_carry_label_and_colour():
    from gcisens import CATEGORIES
    from gcisens.diagnosis import CriterionDiagnosis

    assert HIDDEN_INFLUENCE == "hidden influence"
    assert HIDDEN_INFLUENCE.label == "Hidden influence"
    assert len({c.color for c in CATEGORIES}) == len(CATEGORIES)
    # A diagnosis built from the plain name resolves to the constant.
    diagnosis = CriterionDiagnosis("A", "hidden influence", "detail")
    assert diagnosis.category is HIDDEN_INFLUENCE
    with pytest.raises(ValueError, match="category"):
        CriterionDiagnosis("A", "no such category", "detail")


def test_html_badges_use_the_category_colours(tmp_path, record):
    text = record.to_html(tmp_path / "report.html", include_plots=False).read_text()
    for diagnosis in record.diagnoses:
        assert f"background:{diagnosis.category.color}" in text
        assert diagnosis.category.label in text


def test_record_types_are_public():
    import gcisens

    assert "Metric" in gcisens.__all__
    assert "Category" in gcisens.__all__
    assert gcisens.Category is type(HIDDEN_INFLUENCE)
