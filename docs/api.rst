API reference
=============

The package-level objects below form the supported public interface.

Workflow
--------

.. autoclass:: gcisens.SobolStudy
   :members:

.. autoclass:: gcisens.StudyResult
   :members:

.. autoclass:: gcisens.View
   :members:

.. autoclass:: gcisens.Metric
   :members:

.. autoclass:: gcisens.Comparison
   :members:

.. autofunction:: gcisens.compare

Builders
--------

.. autofunction:: gcisens.esp_comet

.. autofunction:: gcisens.esp_spotis

Diagnosis
---------

.. autoclass:: gcisens.DiagnosisThresholds
   :members:

.. autoclass:: gcisens.Category
   :members:

The category constants ``HIDDEN_INFLUENCE``, ``INTERACTION_DOMINANCE``,
``MODERATE_DISCREPANCY`` and ``CONFIRMED_TRANSPARENCY`` (tuple
``CATEGORIES``) are :class:`gcisens.Category` instances.

Validation
----------

.. autoclass:: gcisens.ValidationResult
   :members:

.. autofunction:: gcisens.validate_scores

Exports
-------

.. autofunction:: gcisens.export.to_csv

.. autofunction:: gcisens.export.to_latex

.. autofunction:: gcisens.s2_to_latex

.. autofunction:: gcisens.comparison_to_latex

.. autofunction:: gcisens.export.to_html

Plots
-----

.. autofunction:: gcisens.plots.plot_indices

.. autofunction:: gcisens.plots.plot_s2_heatmap

.. autofunction:: gcisens.plots.plot_rankings

.. autofunction:: gcisens.plots.plot_validation

.. autofunction:: gcisens.plots.plot_surface

Building blocks
---------------

.. autofunction:: gcisens.sobol_analysis

.. autoclass:: gcisens.SobolIndices
   :members:

pymcdm re-exports
-----------------

``gcisens`` re-exports ``COMET``, ``SPOTIS``, ``ESPExpert`` and
``get_local_weights`` from pymcdm. See the
`pymcdm documentation <https://pymcdm.readthedocs.io/>`_ for their complete
API.
