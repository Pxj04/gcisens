Quickstart
==========

This short example builds an ESP-COMET model, runs a sensitivity analysis and
shows its main result.

1. Create an ESP-COMET model
----------------------------

Define the criteria bounds and an Expected Solution Point, then create the
model with :func:`gcisens.esp_comet`:

.. code-block:: python

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

2. Run a Sobol study
--------------------

Create the study and run the analysis:

.. code-block:: python

   result = SobolStudy(model, n_samples=2048, seed=42).run()

3. View the results table
-------------------------

Display one row per criterion with its weights, Sobol' indices and ranks:

.. code-block:: python

   result.table()

``w`` represents global criterion importance, ``S1`` its independent effect
and ``ST`` its total effect including interactions.

4. Plot sensitivity indices
---------------------------

Compare ``w``, ``S1`` and ``ST`` on a single chart:

.. code-block:: python

   import matplotlib.pyplot as plt

   result.plot_indices()
   plt.show()
