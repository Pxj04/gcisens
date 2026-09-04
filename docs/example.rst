Quick start
===========

Use ``SobolStudy(model, ...).run()`` to compare a model's criterion weights
with Sobol' indices. You can pass a pymcdm model directly.

A complete synthetic example
----------------------------

This example needs no external data. It builds a small ESP-COMET model and
saves its results. Bounds, ESP coordinates and names use the same criterion
order.

.. literalinclude:: ../examples/quickstart.py
   :language: python
   :start-at: from pathlib import Path

Run it from the repository root after installing the package:

.. code-block:: bash

   python examples/quickstart.py

The script writes tables, run metadata and a plot to
``examples/output/quickstart/``. For a single-call model constructor, use
:func:`gcisens.esp_comet`. The builder returns a pymcdm COMET object and stores
the study settings with it.

Use an existing SPOTIS model
----------------------------

SPOTIS takes declared weights and criterion types. Pass these to the study
along with your existing model:

.. code-block:: python

   import numpy as np
   from pymcdm.methods import SPOTIS
   from gcisens import SobolStudy

   bounds = np.array([[0, 10], [0, 100]], dtype=float)
   weights = np.array([0.4, 0.6])
   types = np.array([-1, 1])
   model = SPOTIS(bounds, esp=np.array([3, 80]))
   result = SobolStudy(
       model,
       weights=weights,
       types=types,
       criteria_names=["Cost", "Quality"],
       n_samples=2048,
       sampler="sobol",
       seed=42,
   ).run()
   print(result.table())

:func:`gcisens.esp_spotis` stores weights and types so that you can omit them
from ``SobolStudy``. It still returns a native SPOTIS object. To score data
directly with either form, call ``model(X, weights, types)``. The builder does
not change the pymcdm scoring interface. Lower SPOTIS scores mean closer to
the ESP. Study validation accounts for this orientation.

Choose the sample size
----------------------

``n_samples`` is the base sample size, not the total number of model
evaluations. Use a power of two, such as 512 or 2048. Increase it to check
whether estimates and confidence intervals are stable enough for your study.
Use explicit ``sampler="sobol", seed=42`` for new studies.

The default remains ``"saltelli"`` for compatibility. It produces a
deterministic sample design. ``seed`` controls bootstrap confidence intervals,
the scrambled ``"sobol"`` sample design and the sample used for ``r2_samples``.
If ``seed=None``, the study generates a seed and records it in
``result.metadata()["sampling"]["seed"]``. Pass that recorded value to repeat
the run. Reproducibility also depends on the model, data and software versions.

Read the table
--------------

:meth:`gcisens.StudyResult.table` returns a pandas ``DataFrame`` with one row
per criterion.

``w``
   Regression-estimated importance for COMET, declared weights for SPOTIS.
   ``result.weights_source`` states the source.

``S1``
   The share of score variance due to the criterion alone.

``ST``
   Total influence, including interactions with other criteria.

``ST_minus_S1``
   The part of total influence due to interactions.

The remaining columns contain bootstrap confidence half-widths, criterion
ranks and diagnosis categories. Rank 1 means most important. These ranks do
not order the alternatives in a dataset.

The diagnosis uses heuristic thresholds. Its retained ``confirmed
transparency`` label means that the rules found no discrepancy at those
thresholds. Read :doc:`methodology` for the assumptions and limits.

Result arrays are read-only so that weights, ranks and diagnosis stay
consistent. Use ``result.weights.copy()`` for an editable array, or change the
study inputs and run a new study. Tables returned by result methods can be
edited for presentation. ``result.validate(...)`` explicitly adds a validation
result for later plots and exports.

Save and inspect
----------------

``result.plot_indices()`` returns a Matplotlib ``Axes``. Use standard
Matplotlib methods to change labels or save the figure.

``result.to_csv(directory)`` writes tables and ``results_metadata.json``.
``result.metadata()`` returns the same JSON-compatible run record. It includes
model information, bounds, names, ESPs, weights and types when available,
sampling and bootstrap settings, thresholds, local-weight settings and
software versions. ``result.to_html(path)`` also includes this record.

Keep the model script and data with the outputs. Metadata records the run
configuration; it cannot restore an arbitrary Python model. See
:doc:`advanced` for reporting and the article output map.
