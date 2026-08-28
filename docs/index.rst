gcisens documentation
=====================

``gcisens`` explains how criteria influence the output of rule-based MCDA
models. It combines Sobol' global sensitivity analysis with declared or
estimated criterion weights and turns differences between those views into a
Sensitivity Discrepancy Report.

The library supports ESP-COMET, ESP-SPOTIS and custom Python scoring
functions. Use it to find criteria whose real influence is hidden by their
weights, quantify interaction effects, validate scores against observed
outcomes, and export publication-ready results.

Start here
----------

New users should follow :doc:`installation` and then :doc:`quickstart`.
The :doc:`results` guide explains what each reported metric means and how to
interpret the diagnostic categories.

.. toctree::
   :maxdepth: 2
   :caption: User guide

   installation
   quickstart
   models
   results
   reporting
   validation

.. toctree::
   :maxdepth: 2
   :caption: Background and reference

   pymcdm-style
   methodology
   api

At a glance
-----------

.. code-block:: python

   from gcisens import SobolStudy, esp_comet

   model = esp_comet(esps=esps, bounds=bounds, criteria_names=names)
   result = SobolStudy(model, n_samples=2048, seed=42).run()

   result.table()       # weights, Sobol' indices, ranks and diagnosis
   result.diagnosis()   # concise explanation for every criterion
   result.summary()     # configuration-level quality metrics

The main workflow returns ordinary pandas objects and Matplotlib axes, so the
results fit naturally into notebooks and existing analysis pipelines.

Links
-----

* `Source code <https://github.com/Pxj04/gcisens>`_
* `Package on PyPI <https://pypi.org/project/gcisens/>`_
* :doc:`api`
