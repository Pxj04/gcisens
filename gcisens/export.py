"""Exports: CSV files, LaTeX tables (article layout), standalone HTML report."""

from __future__ import annotations

import base64
import html
import io
import json
import warnings
from dataclasses import asdict
from pathlib import Path

import numpy as np

_CATEGORY_COLORS = {
    "hidden influence": "#e37e7e",
    "interaction dominance": "#e3b57e",
    "moderate discrepancy": "#e3d97e",
    "confirmed transparency": "#a8d8a8",
}


# --------------------------------------------------------------------- helpers
def _fmt(x, digits: int = 4) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    return f"{x:.{digits}f}"


def _write(path, text: str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


_LATEX_ESCAPES = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}


def _latex_escape(value) -> str:
    """Escape user-provided text for a LaTeX text context."""
    return "".join(_LATEX_ESCAPES.get(character, character) for character in str(value))


def _latex_caption(value, default: str) -> str:
    return _latex_escape(value) if value else default


# ------------------------------------------------------------------------- CSV
def to_csv(result, directory, prefix: str = "results") -> list[Path]:
    """Write the study outputs as CSV files; returns the written paths."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    written = []

    main = directory / f"{prefix}_main.csv"
    result.table().to_csv(main, index=False)
    written.append(main)

    if result.sobol.S2 is not None:
        s2 = directory / f"{prefix}_s2.csv"
        result.s2_table().to_csv(s2, index=False)
        written.append(s2)

        names = result.criteria_names
        matrix = np.asarray(result.sobol.S2, dtype=float).copy()
        lower = np.tril_indices_from(matrix, k=-1)
        matrix[lower] = matrix.T[lower]
        s2_matrix = directory / f"{prefix}_s2_matrix.csv"
        import pandas as pd

        pd.DataFrame(matrix, index=names, columns=names).to_csv(s2_matrix)
        written.append(s2_matrix)

    summary = directory / f"{prefix}_summary.csv"
    summary_values = result.summary()
    summary_values["thresholds"] = json.dumps(asdict(result.thresholds), sort_keys=True)
    summary_values["reference_point"] = json.dumps(
        None if result.reference_point is None else result.reference_point.tolist()
    )
    summary_values.to_csv(summary, header=["value"])
    written.append(summary)

    if result.validation is not None:
        val = directory / f"{prefix}_validation.csv"
        result.validation.groups.to_csv(val, index=False)
        written.append(val)
        lift = directory / f"{prefix}_lift.csv"
        result.validation.lift.to_csv(lift, index=False)
        written.append(lift)

    return written


# ----------------------------------------------------------------------- LaTeX
def to_latex(result, path=None, caption=None, label=None) -> str:
    """Main results table in the layout of KES 2026, Tables 2-4."""
    caption = _latex_caption(
        caption, "Weights and Sobol' indices with the Sensitivity Discrepancy Report."
    )
    label = label or "tab:sobol_indices"
    views = result.views
    s = result.sobol
    extras = [r"$ST - S1$", r"$S1_{\mathrm{conf}}$", r"$ST_{\mathrm{conf}}$"]

    cols = "|l|" + "c|" * (len(views) + len(extras))
    header = " & ".join([r"\textbf{Criterion}", *(v.label for v in views), *extras]) + r" \\"

    lines = [
        r"\begin{table}[h]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        rf"\begin{{tabular}}{{{cols}}}",
        r"\hline",
        header,
        r"\hline",
    ]
    for i, name in enumerate(result.criteria_names):
        row = [
            _latex_escape(name),
            *(_fmt(v.values[i]) for v in views),
            _fmt(s.interaction[i]),
            _fmt(s.S1_conf[i]),
            _fmt(s.ST_conf[i]),
        ]
        lines.append(" & ".join(row) + r" \\")
    lines.append(r"\hline")
    sums = [r"$\sum$", *(_fmt(v.values.sum()) for v in views), _fmt(s.interaction.sum()), "", ""]
    lines.append(" & ".join(sums) + r" \\")
    lines += [r"\hline", r"\end{tabular}", r"\end{table}"]

    text = "\n".join(lines)
    if path is not None:
        _write(path, text)
    return text


def s2_to_latex(result, path=None, top: int = 10, caption=None, label=None) -> str:
    """Top pairwise interactions in the layout of the article's S2 table."""
    caption = _latex_caption(caption, f"Top {top} second-order Sobol' interaction indices ($S2$).")
    label = label or "tab:sobol_s2"
    pairs = result.s2_table().head(top)
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\begin{tabular}{|l|l|c|c|}",
        r"\hline",
        r"\textbf{Criterion $i$} & \textbf{Criterion $j$} & $S2$ & $S2_{\mathrm{conf}}$ \\",
        r"\hline",
    ]
    for _, row in pairs.iterrows():
        lines.append(
            f"{_latex_escape(row['criterion_i'])} & {_latex_escape(row['criterion_j'])} & "
            f"{_fmt(row['S2'])} & {_fmt(row['S2_conf'])} \\\\"
        )
    lines += [r"\hline", r"\end{tabular}", r"\end{table}"]
    text = "\n".join(lines)
    if path is not None:
        _write(path, text)
    return text


def comparison_to_latex(comparison, path=None, caption=None, label=None) -> str:
    """Cross-configuration summary in the layout of KES 2026, Table 5."""
    caption = _latex_caption(caption, "Comparative summary across configurations.")
    label = label or "tab:comparison"
    df = comparison.table()
    pretty = {
        "R2": r"$R^2$ (linear approx.)",
        "sum_S1": r"$\sum S1$",
        "sum_ST": r"$\sum ST$",
        "sum_interaction": r"$\sum (ST - S1)$",
        "rho_w_S1": r"$\rho(w, S1)$",
        "rho_w_ST": r"$\rho(w, ST)$",
    }
    cols = "|l|" + "c|" * len(df.columns)
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        rf"\begin{{tabular}}{{{cols}}}",
        r"\hline",
        r"\textbf{Metric} & "
        + " & ".join(rf"\textbf{{{_latex_escape(c)}}}" for c in df.columns)
        + r" \\",
        r"\hline",
    ]
    for metric in df.index:
        row = [pretty[metric] if metric in pretty else _latex_escape(metric)] + [
            _fmt(v) for v in df.loc[metric]
        ]
        lines.append(" & ".join(row) + r" \\")
    lines += [r"\hline", r"\end{tabular}", r"\end{table}"]
    text = "\n".join(lines)
    if path is not None:
        _write(path, text)
    return text


# ------------------------------------------------------------------------ HTML
_HTML_CSS = """
:root { --bg:#0d0d0d; --fg:#d8d8d8; --dim:#8a8a8a; --accent:#7ec8e3;
        --accent2:#a8d8a8; --line:#2a2a2a; --code:#161616; }
* { box-sizing: border-box; }
body { background:var(--bg); color:var(--fg); max-width:900px; margin:0 auto;
       padding:2.5rem 1.5rem 5rem;
       font:15px/1.6 -apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif; }
h1 { font-size:1.5rem; border-bottom:1px solid var(--line); padding-bottom:.5rem; }
h2 { font-size:1.15rem; color:var(--accent); margin-top:2.5rem;
     border-bottom:1px solid var(--line); padding-bottom:.35rem; }
table { border-collapse:collapse; width:100%; margin:1rem 0; font-size:.9em; }
th,td { border:1px solid var(--line); padding:.45rem .65rem; text-align:right; }
th { background:var(--code); color:var(--accent2); }
td:first-child, th:first-child { text-align:left; }
.dim { color:var(--dim); }
.cat { display:inline-block; padding:.05em .55em; border-radius:10px;
       font-size:.85em; color:#0d0d0d; font-weight:600; }
img { max-width:100%; margin:.5rem 0 1rem; border:1px solid var(--line);
      border-radius:8px; }
"""


def _df_to_html(df, float_digits: int = 4) -> str:
    formatted = df.copy()
    for col in formatted.columns:
        if formatted[col].dtype.kind == "f":
            formatted[col] = formatted[col].map(lambda v: _fmt(v, float_digits))
    return formatted.to_html(index=False, border=0, escape=True)


def _fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def to_html(
    result,
    path,
    title: str = "Sensitivity Discrepancy Report",
    include_plots: bool = True,
) -> Path:
    """Standalone dark-theme HTML report: summary, tables, diagnosis, plots."""
    parts = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'>",
        f"<title>{html.escape(title)}</title>",
        f"<style>{_HTML_CSS}</style></head><body>",
        f"<h1>{html.escape(title)}</h1>",
    ]

    summary = result.summary()
    parts.append("<h2>Configuration summary</h2><table><tbody>")
    for key, value in summary.items():
        shown = _fmt(value) if isinstance(value, float) else html.escape(str(value))
        parts.append(f"<tr><td>{html.escape(str(key))}</td><td>{shown}</td></tr>")
    parts.append("</tbody></table>")

    parts.append("<h2>Weights and Sobol' indices</h2>")
    parts.append(_df_to_html(result.table()))

    parts.append("<h2>Discrepancy diagnosis</h2>")
    diag = result.diagnosis()
    parts.append(
        "<table><thead><tr><th>Criterion</th><th>Category</th><th>Detail</th></tr></thead><tbody>"
    )
    for _, row in diag.iterrows():
        color = _CATEGORY_COLORS.get(row["Category"], "#8a8a8a")
        parts.append(
            f"<tr><td>{html.escape(row['Criterion'])}</td>"
            f"<td style='text-align:left'><span class='cat' "
            f"style='background:{color}'>{html.escape(row['Category'])}</span></td>"
            f"<td style='text-align:left'>{html.escape(row['Detail'])}</td></tr>"
        )
    parts.append("</tbody></table>")

    if result.sobol.S2 is not None:
        parts.append("<h2>Top pairwise interactions (S2)</h2>")
        parts.append(_df_to_html(result.s2_table().head(10)))

    if result.validation is not None:
        parts.append("<h2>Validation</h2>")
        parts.append(_df_to_html(result.validation.groups))
        parts.append(_df_to_html(result.validation.lift))

    if include_plots:
        # A plot failure does not stop report generation.
        try:
            import matplotlib.pyplot as plt

            from . import plots

            sections = [("Indices", plots.plot_indices)]
            if result.sobol.S2 is not None:
                sections.append(("Interactions", plots.plot_s2_heatmap))
            sections.append(("Rankings", plots.plot_rankings))
            sections.append(("Decision surface", plots.plot_surface))
            if result.validation is not None:
                sections.append(("Score distributions", plots.plot_validation))
            parts.append("<h2>Plots</h2>")
            for name, fn in sections:
                ax = fn(result)
                fig = (ax if not isinstance(ax, np.ndarray) else ax.ravel()[0]).figure
                parts.append(
                    f"<img alt='{name}' src='data:image/png;base64,{_fig_to_base64(fig)}'>"
                )
                plt.close(fig)
        except Exception as exc:  # noqa: BLE001 - the report must not fail on plotting
            warnings.warn(f"Plots skipped: {exc!r}", UserWarning, stacklevel=2)
            parts.append(f"<p class='dim'>Plots skipped: {html.escape(str(exc))}</p>")

    parts.append("<p class='dim'>Generated by gcisens.</p></body></html>")
    return _write(path, "\n".join(parts))
