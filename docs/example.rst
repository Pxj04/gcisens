Example
=======

This example shows the main ``gcisens`` workflow: create an ESP-COMET model,
run a Sobol' sensitivity study, inspect the criterion-level results and draw a
comparison plot.

Complete example
----------------

.. code-block:: python

   import matplotlib.pyplot as plt
   import numpy as np

   from gcisens import SobolStudy, esp_comet

   bounds = np.array([
       [18, 60],
       [1, 29],
       [1009, 19999],
   ], dtype=float)

   model = esp_comet(
       esps=[[25, 25, 2000]],
       bounds=bounds,
       criteria_names=["Age", "Distance", "Income"],
   )

   result = SobolStudy(
       model,
       n_samples=2048,
       seed=42,
   ).run()

   print(result.table())

   result.plot_indices()
   plt.show()

Model definition
----------------

``bounds`` defines the analysed range of every criterion. Each row contains
the minimum and maximum value for one criterion and must follow the same order
as ``criteria_names``.

``esps`` contains the Expected Solution Point used to construct the ESP-COMET
model. In this example, the preferred point is an age of 25, a distance of 25
and an income of 2000. Multiple Expected Solution Points can be supplied as
additional rows.

Running the study
-----------------

:class:`gcisens.SobolStudy` evaluates the model across the specified criterion
ranges and estimates Sobol' sensitivity indices. ``n_samples`` controls the
base sample size: larger values generally provide more stable estimates but
require more model evaluations. The default ``"saltelli"`` sampler is
deterministic, so the indices are reproducible without a seed. ``seed`` fixes
the bootstrap confidence intervals, the ``"sobol"`` sampler and the uniform
sample behind ``r2_samples``.

Reading the table
-----------------

:meth:`gcisens.StudyResult.table` returns a pandas ``DataFrame`` with one row
per criterion. Its key columns are:

``w``
   The criterion importance estimated from the model.

``S1``
   The part of output variance explained by the criterion on its own.

``ST``
   The criterion's total influence, including its interactions with other
   criteria.

``ST_minus_S1``
   The difference between total and first-order influence. A larger difference
   indicates a stronger contribution through interactions.

The remaining columns provide confidence estimates, rankings and the
diagnostic category assigned to each criterion.

Plotting the result
-------------------

:meth:`gcisens.StudyResult.plot_indices` displays ``w``, ``S1`` and ``ST``
next to each other for every criterion. This makes it easier to compare the
model's criterion importance with the influence measured by the sensitivity
analysis. The method returns a Matplotlib ``Axes`` object, so the figure can be
customised using standard Matplotlib functions.

ESP-SPOTIS models and custom Python scoring functions can be analysed through
the same :class:`gcisens.SobolStudy` interface. Their constructors and the
remaining result methods are listed in the :doc:`api`.
