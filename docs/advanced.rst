Advanced use and article reproduction
=====================================

Start with :doc:`example`. The methods here extend the same study result.

Local weights and threshold checks
----------------------------------

Pass ``reference_point=point`` to ``SobolStudy(...).run()`` to add local
weights. For each criterion, the method sweeps its full bounds while holding
all other criteria at the reference point. It does not restrict the sweep to
a small neighbourhood. ``local_percent_step`` on ``SobolStudy`` controls the
sweep resolution.

.. code-block:: python

   result = SobolStudy(model, sampler="sobol", seed=42).run(
       reference_point=point,
   )
   print(result.diagnosis())
   print(result.sweep_thresholds(interaction_ratio=[0.2, 0.3, 0.4]))

Report threshold sensitivity with the diagnosis. The rules use point
estimates, so also inspect confidence intervals and sample-size stability.

Plots, validation and comparison
--------------------------------

All plot methods return a Matplotlib ``Axes``. Use ``plot_s2_heatmap()`` for
pairwise interactions, ``plot_rankings()`` for criterion ranks and
``plot_surface()`` for a two-criterion score slice. The other criteria stay
at a fixed point on that slice.

Keep the underlying model or callable unchanged after the study runs.
``validate()`` and ``plot_surface()`` score new points with that model. The
study records its settings, but it does not serialise or freeze arbitrary
model objects or callable state. To analyse a changed model, run a new study.

``result.validate(X, labels, top_k=[50, 100])`` compares model scores with
binary labels through group statistics and lift. Every cut-off must be a
positive integer; values above the row count use all rows. A DataFrame must contain
each criterion name exactly once; the method selects and orders those columns.
An array uses positional column order. When labels are a pandas Series and
``X`` is a DataFrame, both indexes must be unique and contain the same row
labels. Validation aligns the Series to the DataFrame index. Array and list
labels use positional row order.
Use ``result.plot_validation()`` after validation.

.. code-block:: python

   from gcisens import compare

   comparison = compare({"ESP1": result1, "ESP2": result2})
   print(comparison.table())
   comparison.to_csv("comparison.csv")
   comparison.to_latex("comparison.tex")

Use ``result.to_latex()`` and ``result.s2_to_latex()`` for tables, and
``result.to_html("report.html")`` for a shareable report. CSV and HTML include
the configuration from ``result.metadata()``. They do not store the model's
Python code or input dataset.

Custom scoring functions
------------------------

A callable must accept a matrix and return one finite score per row. It must
be deterministic, and a point's score must not change when other rows change.
Fit the model and fix any normalisation constants before starting the study.
The sampled scores must have nonzero variance.

.. code-block:: python

   def score(X):
       return 0.3 * X[:, 0] + 0.7 * X[:, 1] ** 2

   result = SobolStudy(
       score,
       bounds=[[0, 1], [0, 1]],
       criteria_names=["A", "B"],
       sampler="sobol",
       seed=42,
   ).run()

Without supplied weights, the study estimates weights by regression on a
separate uniform sample. Keep the callable definition with the result.
The library cannot check whether every possible input satisfies the scoring
contract.

Wrapping an arbitrary pymcdm method in a lambda is not enough. For example,
default TOPSIS normalises and compares the current matrix of alternatives.
A point can receive different scores when the other rows change. Such a
wrapper does not define the fixed scoring function required by this study.

Reproduce the existing article case study
-----------------------------------------

``examples/article_esp_comet.py`` reproduces the existing KES 2026 case study.
It is also a starting example for the planned SoftwareX article. The map below
describes the current outputs; assign SoftwareX table and figure numbers when
the manuscript selects its results.

Create the Python 3.11 environment from ``requirements-repro.txt`` using the
instructions in ``CONTRIBUTING.md``. From the repository root, run:

.. code-block:: bash

   python examples/article_esp_comet.py

The script uses explicit ``sampler="saltelli"`` and ``seed=42``. The bundled
fictional HR dataset and its licence are described in
``examples/data/README.md``. Outputs are under ``examples/output/``.

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Result
     - Output from ``article_esp_comet.py``
   * - KES Tables 2, 3 and 4, ESP1, ESP2 and ESP1+ESP2
     - ``esp1/``, ``esp2/`` and ``esp1_esp2/`` each contain
       ``results_main.csv`` and ``table_main.tex``
   * - Pairwise interaction results per configuration
     - ``results_s2.csv``, ``results_s2_matrix.csv`` and ``table_s2.tex``
   * - Fit metrics, validation and lift per configuration
     - ``results_summary.csv``, ``results_validation.csv`` and
       ``results_lift.csv``
   * - Configuration record and report per configuration
     - ``results_metadata.json`` and ``report.html``
   * - KES Table 5, comparison across configurations
     - ``comparison.csv`` and ``comparison.tex``
   * - File identities and run settings
     - ``reproduction.json`` records SHA-256 hashes of the dataset, article
       script and installed gcisens Python source files. It also records the
       hash of ``requirements-repro.txt`` when available and the metadata for
       each configuration.

The HTML reports contain plots for inspection. If the manuscript uses a plot
as a separate figure, save the returned Matplotlib figure to a named file and
add that file to this map. ``tests/test_article_reproduction.py`` records
expected article results and their numerical tolerances.

Archive the exact software tag, input data, model script, pinned environment
and generated outputs used by the submitted manuscript. Verify the journal's
current submission instructions separately.

Compatibility helpers
---------------------

Existing explicit imports of low-level functions, ``View``, ``Metric``,
``Category`` and pymcdm re-exports remain available for compatibility.
New code should obtain results from ``SobolStudy.run()``, call result methods
for exports and plots, and import COMET, SPOTIS and ESPExpert from pymcdm.
Manual result construction and custom display records are advanced contracts;
they are not required for the supported workflow.

Result ownership
----------------

Study numbers, thresholds and validation arrays are read-only. Use
``result.weights.copy()`` or ``result.table()`` for editable data. Validation
summary tables are copies. Run a new study to change a recorded model or its
numerical settings; copying a record with changed numbers cannot preserve the
old run metadata. Use ``result.sweep_thresholds(...)`` to compare thresholds.

Older explicit helper imports remain available. Wildcard imports now contain
only the main workflow objects. See ``CHANGELOG.md`` for migration details.
