"""Exports render a StudyResult record; no model is needed."""

import json
from dataclasses import replace

import matplotlib
import numpy as np
import pandas as pd
import pytest

import gcisens
from gcisens import DiagnosisThresholds, compare


def test_exports_roundtrip(tmp_path, record):
    record = replace(record, thresholds=DiagnosisThresholds(hidden_weight_factor=0.25))
    files = record.to_csv(tmp_path)
    assert all(f.exists() for f in files)
    main = pd.read_csv(tmp_path / "results_main.csv")
    assert list(main["Criterion"]) == ["A", "B", "C"]
    assert list(main["Category"]) == [str(d.category) for d in record.diagnoses]

    summary = pd.read_csv(tmp_path / "results_summary.csv", index_col=0)["value"]
    assert json.loads(summary["thresholds"])["hidden_weight_factor"] == 0.25
    assert json.loads(summary["reference_point"]) == [1.0, 2.0, 3.0]
    assert float(summary["rho_w_wloc"]) == pytest.approx(1.0)

    s2_matrix = pd.read_csv(tmp_path / "results_s2_matrix.csv", index_col=0)
    assert list(s2_matrix.index) == ["A", "B", "C"]
    assert s2_matrix.loc["A", "B"] == s2_matrix.loc["B", "A"] == 0.08

    validation = pd.read_csv(tmp_path / "results_validation.csv")
    assert list(validation["group"]) == ["positive", "negative"]

    latex = record.to_latex(tmp_path / "table.tex")
    assert latex.startswith(r"\begin{table}")
    assert (tmp_path / "table.tex").exists()

    text = record.to_html(tmp_path / "report.html").read_text()
    assert "Discrepancy diagnosis" in text
    assert "<img" in text
    assert "Plots skipped" not in text


def test_latex_exports_escape_user_text(record):
    names = ["Rate_%", "Cost & Fees", "Plain"]
    record = replace(
        record,
        sobol=replace(record.sobol, criteria_names=names),
        diagnoses=[replace(d, criterion=n) for d, n in zip(record.diagnoses, names)],
    )

    caption = r"Report \ & % $ # _ { } ~ ^"
    expected_caption = (
        r"Report \textbackslash{} \& \% \$ \# \_ \{ \} \textasciitilde{} \textasciicircum{}"
    )
    main = record.to_latex(caption=caption)
    interactions = gcisens.s2_to_latex(record, caption=caption)
    comparison = gcisens.comparison_to_latex(compare({"ESP_1 & 2": record}), caption=caption)

    for latex in (main, interactions, comparison):
        assert rf"\caption{{{expected_caption}}}" in latex
    assert r"Rate\_\%" in main
    assert r"Cost \& Fees" in main
    assert r"Rate\_\%" in interactions
    assert r"Cost \& Fees" in interactions
    assert r"\textbf{ESP\_1 \& 2}" in comparison


def test_latex_table_walks_the_views(record):
    header = record.to_latex().splitlines()[6]
    cells = [c.strip() for c in header.split("&")]
    assert cells[1:5] == [v.label for v in record.views]


def test_html_export_keeps_matplotlib_backend(tmp_path, record):
    original_backend = matplotlib.get_backend()
    try:
        matplotlib.use("svg")
        report = record.to_html(tmp_path / "report.html")
        assert matplotlib.get_backend().lower() == "svg"
        text = report.read_text()
        assert "<img" in text
        assert "Plots skipped" not in text
    finally:
        matplotlib.use(original_backend)


def test_html_export_can_skip_plots(tmp_path, record):
    report = record.to_html(tmp_path / "report.html", include_plots=False)
    assert "<img" not in report.read_text()


def test_html_export_warns_when_plots_fail(tmp_path, record, monkeypatch):
    def broken(result, ax=None):
        raise ValueError("boom")

    monkeypatch.setattr(gcisens.plots, "plot_rankings", broken)

    with pytest.warns(UserWarning, match="Plots skipped: ValueError"):
        report = record.to_html(tmp_path / "report.html")

    assert "Plots skipped: boom" in report.read_text()


def test_constant_view_correlation_is_shown_as_not_available(tmp_path, record):
    record = replace(
        record,
        views=[replace(v, values=np.full(3, 1 / 3)) if v.key == "w" else v for v in record.views],
    )
    assert np.isnan(record.summary()["rho_w_S1"])
    assert "n/a" in gcisens.comparison_to_latex(compare({"equal weights": record}))
    report = record.to_html(tmp_path / "report.html", include_plots=False)
    assert "<td>n/a</td>" in report.read_text()


def test_latex_exports_are_available_through_public_api(tmp_path, record):
    assert callable(gcisens.s2_to_latex)
    assert callable(gcisens.comparison_to_latex)
    assert record.s2_to_latex(tmp_path / "s2.tex") == gcisens.s2_to_latex(record)
    assert (tmp_path / "s2.tex").exists()


@pytest.mark.parametrize("keep_s2,keep_validation", [(False, True), (True, False), (False, False)])
def test_csv_reexport_removes_only_obsolete_optional_files(
    tmp_path, record, keep_s2, keep_validation
):
    record.to_csv(tmp_path, prefix="audit")
    record.to_csv(tmp_path, prefix="other")
    unrelated = tmp_path / "audit_notes.csv"
    unrelated.write_text("Keep this file.\n")
    updated = replace(
        record,
        sobol=record.sobol if keep_s2 else replace(record.sobol, S2=None, S2_conf=None),
        validation=record.validation if keep_validation else None,
    )

    written = updated.to_csv(tmp_path, prefix="audit")

    for suffix in ("s2.csv", "s2_matrix.csv"):
        assert (tmp_path / f"audit_{suffix}").exists() == keep_s2
        assert (tmp_path / f"other_{suffix}").exists()
    for suffix in ("validation.csv", "lift.csv"):
        assert (tmp_path / f"audit_{suffix}").exists() == keep_validation
        assert (tmp_path / f"other_{suffix}").exists()
    assert unrelated.read_text() == "Keep this file.\n"
    assert all(path.exists() for path in written)
    metadata = json.loads((tmp_path / "audit_metadata.json").read_text())
    assert metadata["sampling"]["second_order"] == keep_s2
