Quickstart
==========

This example builds an ESP-COMET model with three criteria and one Expected
Solution Point (ESP), then runs the complete sensitivity workflow.

.. code-block:: python

   import numpy as np
   from gcisens import esp_comet, SobolStudy

   bounds = np.array([[18, 60], [1, 29], [1009, 19999]], float)
   names = ["Age", "Distance", "Income"]
   esps = np.array([[25, 25, 2000]], float)

   model = esp_comet(esps=esps, bounds=bounds, criteria_names=names)
   result = SobolStudy(model, n_samples=2048, seed=42).run()

Inspect the result
------------------

The three most useful entry points return pandas objects:

.. code-block:: python

   result.table()      # one row per criterion
   result.diagnosis()  # diagnostic category and explanation
   result.summary()    # one row of configuration-level metrics

``w`` is the declared or estimated importance, ``S1`` is the criterion's
first-order effect, and ``ST`` is its total effect including interactions.
See :doc:`results` before drawing conclusions from individual values.

Local importance
----------------

Pass a reference point to include local range-sweep weights alongside global
weights and Sobol' indices:

.. code-block:: python

   reference = np.array([30, 10, 5000], float)
   local_result = SobolStudy(model, n_samples=2048, seed=42).run(
       reference_point=reference
   )
   local_result.table()[["Criterion", "w", "w_loc", "S1", "ST"]]

Plot and export
---------------

.. code-block:: python

   result.plot_indices()
   result.plot_s2_heatmap()
   result.to_csv("output")
   result.to_latex("output/results.tex")
   result.to_html("report.html")

See :doc:`reporting` for all plots and export formats.

Compare model configurations
----------------------------

ESP-SPOTIS uses the same study interface. Unlike COMET, its declared weights
are inputs to the model:

.. code-block:: python

   from gcisens import compare, esp_spotis

   spotis_model = esp_spotis(
       esp=[25, 25, 2000],
       bounds=bounds,
       weights=[0.5, 0.2, 0.3],
       criteria_names=names,
   )
   spotis_result = SobolStudy(spotis_model, n_samples=2048, seed=42).run()

   comparison = compare({
       "ESP-COMET": result,
       "ESP-SPOTIS": spotis_result,
   })
   comparison.table()

Choosing a sample size
----------------------

``n_samples`` is the Saltelli base sample size, not the final number of model
evaluations. With ``m`` criteria, the default second-order study evaluates the
model ``N * (2m + 2)`` times. Setting ``second_order=False`` reduces this to
``N * (m + 2)`` when pairwise interactions are not required.

Use a small power of two such as 256 while developing, then increase it and
check that the indices and confidence intervals are stable. The source-study
default is 2048.
