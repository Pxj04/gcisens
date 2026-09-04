API reference
=============

Use the workflow objects below for new code. Start with :doc:`example`;
reporting, comparison and custom scoring are described in :doc:`advanced`.

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

Advanced building blocks
------------------------

The study result provides validation, plotting, export and threshold-sweep
methods. Prefer these methods to standalone helpers that do the same work.

.. autoclass:: gcisens.ValidationResult
   :members:

.. autofunction:: gcisens.sobol_analysis

.. autoclass:: gcisens.SobolIndices
   :members:

Compatibility
-------------

Existing explicit imports of ``View``, ``Metric``, ``Category``, category
constants and standalone plotting/export functions remain available. They are
not needed to construct a study. Results should come from
:meth:`gcisens.SobolStudy.run`, so their weights, indices, ranks and diagnosis
stay consistent.

Import ``COMET`` and ``SPOTIS`` from ``pymcdm.methods`` and ``ESPExpert`` from
``pymcdm.methods.comet_tools`` in new code. The older gcisens re-exports remain
available. See the `pymcdm documentation <https://pymcdm.readthedocs.io/>`_
for those classes.
