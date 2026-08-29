API reference
=============

The package-level objects below form the supported public interface.

Workflow
--------

.. autoclass:: gcisens.SobolStudy
   :members:

.. autoclass:: gcisens.StudyResult
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

Validation
----------

.. autoclass:: gcisens.ValidationResult
   :members:

.. autofunction:: gcisens.validate_scores

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
