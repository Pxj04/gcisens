Supported models
================

ESP-COMET
---------

:func:`gcisens.esp_comet` constructs a regular pymcdm ``COMET`` model and
attaches the bounds, criterion names and ESPs needed by the analysis. It
accepts one or more ESP rows.

.. code-block:: python

   model = esp_comet(
       esps=[[25, 25, 2000], [35, 8, 8000]],
       bounds=bounds,
       criteria_names=names,
   )

COMET does not take declared criterion weights directly. ``gcisens`` estimates
global weights with linear regression over the model's characteristic-object
preferences and reports the fit quality as :math:`R^2`.

ESP-SPOTIS
----------

:func:`gcisens.esp_spotis` accepts exactly one ESP. Its weights are supplied by
the user and are the declared-importance view compared with the observed Sobol'
effects.

.. code-block:: python

   model = esp_spotis(
       esp=[25, 25, 2000],
       bounds=bounds,
       weights=[0.5, 0.2, 0.3],
       types=[-1, -1, 1],
       criteria_names=names,
   )

SPOTIS returns distances, so lower scores mean closer to the ESP. The
validation helpers detect this orientation automatically.

Custom scoring functions
------------------------

Any callable that accepts a two-dimensional NumPy array and returns one score
per row can be analysed. Bounds are required; declared weights are optional.

.. code-block:: python

   import numpy as np
   from gcisens import SobolStudy

   def score(X):
       return 0.7 * X[:, 0] + 0.3 * X[:, 1] ** 2

   result = SobolStudy(
       score,
       bounds=np.array([[0, 1], [0, 1]], float),
       criteria_names=["Linear", "Nonlinear"],
       weights=np.array([0.7, 0.3]),
       n_samples=1024,
       seed=42,
   ).run()

When weights are omitted, ``gcisens`` estimates regression weights from a
uniform sample. Provide weights when the goal is to audit a declared model
specification.

Input contract
--------------

For every model type:

* ``bounds`` must contain one ``[minimum, maximum]`` row per criterion;
* criterion names must match the number and order of the bounds;
* the scoring function must return one finite scalar per sampled alternative;
* bounds should describe the meaningful decision domain, because Sobol'
  indices are conditional on that domain.
