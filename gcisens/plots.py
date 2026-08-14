"""Matplotlib plots for study results, in the pymcdm visual style.

Wherever a ``pymcdm.visuals`` function fits, it is called directly
(``ranking_flows``, ``correlation_heatmap``, ``comet_2d_esp_plot``), so the
output is native pymcdm. The remaining plots have no pymcdm equivalent
(grouped bars with confidence intervals; violins over unequal-size groups)
and follow the same conventions by hand: default matplotlib colour cycle,
black-edged bars, dashed recessive grid, legends above the axes. Each
function returns the matplotlib Axes it drew on.
"""
from __future__ import annotations

import re
import textwrap

import numpy as np
from matplotlib import pyplot as plt
from pymcdm import visuals as pymcdm_visuals
from pymcdm.methods import COMET

from .adapters import META_ATTR

_GRID = {"alpha": 0.5, "linestyle": "--"}


def plot_indices(result, ax=None):
    """Grouped bars: declared weight vs S1 vs ST per criterion, with conf bars."""
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 4.2))
    s = result.sobol
    names = result.criteria_names
    x = np.arange(len(names))
    width = 0.26
    err = {"ecolor": "#333333", "elinewidth": 1, "capsize": 2}

    ax.bar(x - width, result.weights, width * 0.92, label="$w$",
           color="C0", edgecolor="black", linewidth=1)
    ax.bar(x, s.S1, width * 0.92, label="$S1$", yerr=s.S1_conf,
           color="C1", edgecolor="black", linewidth=1, error_kw=err)
    ax.bar(x + width, s.ST, width * 0.92, label="$ST$", yerr=s.ST_conf,
           color="C2", edgecolor="black", linewidth=1, error_kw=err)

    ax.set_xticks(x, names, rotation=20, ha="right")
    ax.set_ylabel("Importance / variance share")
    ax.grid(**_GRID)
    ax.set_axisbelow(True)
    ax.legend(bbox_to_anchor=(0.0, 1.02, 1.0, 0.102), loc="lower left",
              ncol=3, mode="expand", borderaxespad=0.0)
    ax.figure.tight_layout()
    return ax


def plot_rankings(result, ax=None):
    """Criteria rankings across views (w, w_loc, S1, ST).

    Drawn directly with :func:`pymcdm.visuals.ranking_flows`; the ``$A_i$``
    alternative labels it produces are replaced with the criteria names.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 4.6))
    names = result.criteria_names
    views = [v for v in ("w", "w_loc", "S1", "ST") if v in result.ranks]
    view_labels = {"w": "$w$", "w_loc": "$w_{loc}$", "S1": "$S1$", "ST": "$ST$"}
    rankings = np.array([result.ranks[v] for v in views])

    ax = pymcdm_visuals.ranking_flows(
        rankings, labels=[view_labels[v] for v in views], ax=ax
    )

    # ranking_flows labels alternatives $A_1$..$A_m$; swap in criteria names.
    pattern = re.compile(r"^\$A_\{(\d+)\}\$$")
    for text in ax.texts:
        match = pattern.match(text.get_text())
        if match:
            text.set_text(names[int(match.group(1)) - 1])
    ax.set_xlim(-1.6, len(views) - 1 + 1.6)  # room for the name labels
    ax.figure.tight_layout()
    return ax


def plot_s2_heatmap(result, ax=None, cmap="Greens"):
    """Heatmap of pairwise interaction indices (S2).

    Drawn directly with :func:`pymcdm.visuals.correlation_heatmap` on the
    symmetric S2 matrix; the diagonal (undefined) is blanked and significant
    interactions (|S2| > 2 * conf and |S2| > 0.01) are starred.
    """
    s = result.sobol
    if s.S2 is None:
        raise ValueError("Second-order indices were not computed (second_order=False)")
    if ax is None:
        _, ax = plt.subplots(figsize=(6.6, 5.6))
    names = result.criteria_names
    m = len(names)

    # SALib fills the upper triangle; make the matrix symmetric for display.
    values = np.full((m, m), np.nan)
    for i in range(m):
        for j in range(i + 1, m):
            values[i, j] = values[j, i] = s.S2[i, j]

    ax = pymcdm_visuals.correlation_heatmap(
        values, labels=names, float_fmt="%0.3f", cmap=cmap,
        text_kwargs={"fontsize": 7.5}, ax=ax,
    )

    # Post-process the cell texts: blank the diagonal, star significance.
    for text in ax.texts:
        j, i = (round(v) for v in text.get_position())
        if i == j:
            text.set_text("")
        else:
            conf = s.S2_conf[min(i, j), max(i, j)]
            val = values[i, j]
            if abs(val) > 2 * conf and abs(val) > 0.01:
                text.set_text(text.get_text() + "*")

    ax.set_title("Pairwise interactions ($S2$); * = significant", fontsize=10)
    ax.figure.tight_layout()
    return ax


def plot_validation(result, ax=None):
    """Score distributions per label group, with lift@k in the title."""
    if result.validation is None:
        raise ValueError("Run result.validate(X, labels) first")
    if ax is None:
        _, ax = plt.subplots(figsize=(6.8, 4.2))
    val = result.validation
    pos, neg = val.scores[val.labels], val.scores[~val.labels]

    parts = ax.violinplot([neg, pos], positions=[0, 1], showmedians=True,
                          showextrema=False)
    for body, color in zip(parts["bodies"], ("C0", "C1")):
        body.set_facecolor(color)
        body.set_alpha(0.6)
        body.set_edgecolor("black")
    parts["cmedians"].set_color("black")

    ax.set_xticks([0, 1], [f"negative (n={len(neg)})", f"positive (n={len(pos)})"])
    ax.annotate(f"$\\Delta$ mean = {val.delta_mean:+.4f}",
                xy=(0.98, 0.02), xycoords="axes fraction", ha="right", fontsize=9)
    ax.set_title("   ".join(f"lift@{int(r.k)}={r.lift:.2f}$\\times$"
                            for r in val.lift.itertuples()), fontsize=10)
    ax.set_ylabel("Model score")
    ax.grid(**_GRID)
    ax.set_axisbelow(True)
    ax.figure.tight_layout()
    return ax


def _model_esps(result):
    """Best-effort recovery of the ESPs attached to the studied model."""
    meta = getattr(result.adapter.model, META_ATTR, None)
    if meta is not None and meta.esps is not None:
        return np.atleast_2d(meta.esps)
    esp = getattr(result.adapter.model, "esp", None)  # SPOTIS stores it natively
    if esp is not None:
        return np.atleast_2d(esp)
    return None


def plot_surface(result, criteria=None, at=None, esps=None, num=100, ax=None,
                 cmap="Greens", levels=14):
    """Decision surface over two criteria with the evaluation grid and ESPs.

    Reproduces the surface plots of Sałabun et al. (ISD 2025), Figs. 1-2.
    For models with more than two criteria the surface is a 2-D *slice*: the
    remaining criteria are fixed at ``at`` (default: the study's reference
    point, or the middle of the bounds).

    For a two-criteria COMET model the plot is delegated directly to
    :func:`pymcdm.visuals.comet_2d_esp_plot` (the exact article figure).

    Parameters
    ----------
    result : StudyResult
        The study result (provides the model, bounds and names).
    criteria : tuple of int or str, optional
        The two criteria to plot; defaults to the two with the highest ST.
    at : array-like, optional
        Values at which the remaining criteria are fixed (full-length vector;
        the two plotted entries are ignored).
    esps : ndarray, optional
        ESP points to mark; recovered from the model when omitted.
    num : int
        Grid resolution per axis.

    Returns
    -------
    ax : matplotlib Axes
    """
    adapter = result.adapter
    names = result.criteria_names
    m = adapter.n_criteria

    if criteria is None:
        order = np.argsort(-result.sobol.ST)
        criteria = tuple(sorted(order[:2]))
    idx = tuple(names.index(c) if isinstance(c, str) else int(c) for c in criteria)
    if len(idx) != 2 or idx[0] == idx[1]:
        raise ValueError("criteria must select two different criteria")
    ci, cj = idx

    if at is None:
        at = (result.reference_point if result.reference_point is not None
              else adapter.bounds.mean(axis=1))
    at = np.asarray(at, dtype=float).ravel()

    if esps is None:
        esps = _model_esps(result)

    if m == 2 and idx == (0, 1) and isinstance(adapter.model, COMET) and esps is not None:
        # True 2-D COMET: the native pymcdm article figure, no slicing needed.
        if ax is None:
            _, ax = plt.subplots(figsize=(5.2, 4.4))
        ax, _cax = pymcdm_visuals.comet_2d_esp_plot(
            adapter.model, np.atleast_2d(esps), adapter.bounds, ax=ax
        )
        ax.set_xlabel(names[0])
        ax.set_ylabel(names[1])
        ax.figure.tight_layout()
        return ax

    if ax is None:
        _, ax = plt.subplots(figsize=(5.2, 4.4))

    (xlo, xhi), (ylo, yhi) = adapter.bounds[ci], adapter.bounds[cj]
    xs = np.linspace(xlo, xhi, num)
    ys = np.linspace(ylo, yhi, num)
    gx, gy = np.meshgrid(xs, ys)
    points = np.tile(at, (num * num, 1))
    points[:, ci] = gx.ravel()
    points[:, cj] = gy.ravel()
    z = adapter.scores(points).reshape(num, num)

    cf = ax.contourf(gx, gy, z, levels=levels, cmap=cmap)
    ax.figure.colorbar(cf, ax=ax, label="Preference" if adapter.higher_is_closer
                       else "Distance to ESP")

    # Evaluation grid: characteristic values for COMET, dotted like pymcdm.
    cvalues = getattr(adapter.model, "cvalues", None)
    if cvalues is not None:
        for v in cvalues[ci]:
            ax.axvline(v, color="black", linewidth=0.7, linestyle=":", alpha=0.6)
        for v in cvalues[cj]:
            ax.axhline(v, color="black", linewidth=0.7, linestyle=":", alpha=0.6)
        co_x, co_y = np.meshgrid(cvalues[ci], cvalues[cj])
        ax.scatter(co_x, co_y, c="black", s=14, zorder=3)

    if esps is not None:
        esps = np.atleast_2d(esps)
        ax.scatter(esps[:, ci], esps[:, cj], c="orange", marker="*", s=140,
                   zorder=4, edgecolors="black", linewidths=0.5)
        for k, esp in enumerate(esps, 1):
            ax.text(esp[ci] + (xhi - xlo) * 0.03, esp[cj], f"$ESP_{{{k}}}$",
                    color="orange", fontweight="bold", fontsize=11)

    ax.set_xlabel(names[ci])
    ax.set_ylabel(names[cj])
    ax.set_xlim(xlo, xhi)
    ax.set_ylim(ylo, yhi)
    ax.set_xticks(np.linspace(xlo, xhi, 5))
    ax.set_yticks(np.linspace(ylo, yhi, 5))
    if m > 2:
        fixed = ", ".join(f"{names[k]}={at[k]:g}" for k in range(m) if k not in idx)
        ax.set_title("\n".join(textwrap.wrap(f"Slice at {fixed}", width=58)),
                     fontsize=8)
    ax.figure.tight_layout()
    return ax
