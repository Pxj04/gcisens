Plots and exports
=================

Plots
-----

Study results expose Matplotlib helpers and return an ``Axes`` object for
further styling:

.. code-block:: python

   ax = result.plot_indices()
   ax.set_title("Declared importance and observed sensitivity")

Available plots are:

* :meth:`gcisens.StudyResult.plot_indices` for ``w``, ``S1`` and ``ST``;
* :meth:`gcisens.StudyResult.plot_s2_heatmap` for pairwise interactions;
* :meth:`gcisens.StudyResult.plot_rankings` for rank changes across views;
* :meth:`gcisens.StudyResult.plot_surface` for a two-criterion surface or a
  two-dimensional slice of a larger model;
* :meth:`gcisens.StudyResult.plot_validation` after validation has run.

Second-order plots and :meth:`gcisens.StudyResult.s2_table` require
``second_order=True``.

CSV, LaTeX and HTML
-------------------

.. code-block:: python

   paths = result.to_csv("output", prefix="experiment_1")
   latex = result.to_latex(
       "output/experiment_1.tex",
       caption="Sensitivity analysis results",
       label="tab:sensitivity",
   )
   report_path = result.to_html(
       "output/experiment_1.html",
       title="Experiment 1 — Sensitivity Discrepancy Report",
   )

``to_csv`` writes the main results and S2 tables, plus validation data when
available. ``to_latex`` returns the generated string as well as optionally
writing it. ``to_html`` creates a standalone report containing tables,
diagnosis and plots.

Comparisons
-----------

Use :func:`gcisens.compare` to place configuration-level metrics side by side:

.. code-block:: python

   comparison = compare({"baseline": baseline, "alternative": alternative})
   comparison.table()
   comparison.to_csv("output/comparison.csv")
   comparison.to_latex("output/comparison.tex")
