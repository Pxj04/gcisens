Quickstart
==========

Build an ESP model with one call and run the full workflow:

.. code-block:: python

   import numpy as np
   from gcisens import esp_comet, SobolStudy

   bounds = np.array([[18, 60], [1, 29], [1009, 19999]], float)
   esps = np.array([[25, 25, 2000]])

   model = esp_comet(esps=esps, bounds=bounds,
                     criteria_names=["Age", "Distance", "Income"])
   result = SobolStudy(model, n_samples=2048, seed=42).run()

   result.table()        # weights, S1, ST, ranks, category
   result.diagnosis()    # Sensitivity Discrepancy Report
   result.summary()      # R², ΣS1, ΣST, Spearman correlations
   result.to_latex()     # publication-ready table
   result.to_html("report.html")

ESP-SPOTIS works the same way; its declared weights are an input:

.. code-block:: python

   from gcisens import esp_spotis

   model = esp_spotis(esp=[25, 25, 2000], bounds=bounds,
                      weights=[0.5, 0.2, 0.3],
                      criteria_names=["Age", "Distance", "Income"])
   result = SobolStudy(model, n_samples=2048, seed=42).run()

Compare configurations side by side:

.. code-block:: python

   from gcisens import compare

   cmp = compare({"ESP1": res1, "ESP2": res2})
   cmp.table()
