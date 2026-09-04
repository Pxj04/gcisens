Troubleshooting
===============

Common input errors raise ``ValueError`` or ``TypeError``. Warnings identify
conditions that need review. Messages below are excerpts; the full message
also identifies the parameter or criterion.

Building the study
------------------

.. list-table::
   :header-rows: 1
   :widths: 34 30 36

   * - Message
     - Cause
     - Fix
   * - ``Criteria bounds are required``
     - The model is a plain callable, so the study cannot recover its domain.
     - Pass ``bounds=`` to ``SobolStudy``, or build the model with
       ``esp_comet`` / ``esp_spotis``.
   * - ``SPOTIS models need declared criteria weights``
     - A pymcdm ``SPOTIS`` was built by hand; SPOTIS takes weights as input.
     - Pass ``weights=`` to ``SobolStudy``, or use ``esp_spotis``.
   * - ``weights must be non-negative and sum to 1``
     - Declared weights are on another scale.
     - Normalise them: ``w / w.sum()``.
   * - Error about ``types``
     - SPOTIS criterion types have the wrong shape or contain values other
       than 1 and -1.
     - Supply one type per criterion: 1 for profit, -1 for cost. Without an
       explicit ESP, types are required.
   * - ``COMET models do not take weights/types``
     - COMET has no declared weights; the study estimates them by
       regression.
     - Drop the arguments. The estimated weights are ``result.weights``.
   * - ``<name> passed to SobolStudy differs from the <name> the model was
       built with``
     - The builder already stored that value on the model.
     - Pass the value to ``esp_comet`` / ``esp_spotis`` only.
   * - ``bounds row i must have min < max``
     - A bounds row is reversed or degenerate.
     - Write every row as ``[min, max]``.
   * - ``esps must have shape (k, m)`` / ``Got n criteria names for m
       criteria``
     - A shape does not match the number of criteria in ``bounds``.
     - Give one ESP row and one name per criterion.
   * - ``n_samples must be an integer`` / ``n_samples must be at least 2``; warning ``n_samples=N is not a
       power of two``
     - The Saltelli design needs ``N = 2**k`` for a balanced sample.
     - Use 256, 512, 1024, 2048, ...
   * - ``sampler must be one of ('saltelli', 'sobol')``
     - Unknown sampler name.
     - Use ``"sobol"`` for new studies; ``"saltelli"`` reproduces the
       articles.
   * - ``AttributeError: 'ESPExpert' object has no attribute
       'make_cvalues_psi'``
     - pymcdm older than 1.4.
     - ``pip install "pymcdm>=1.4"``.

Running and validating
----------------------

.. list-table::
   :header-rows: 1
   :widths: 34 30 36

   * - Message
     - Cause
     - Fix
   * - ``COMET has N characteristic objects; ... MiB`` (warning)
     - pymcdm allocates a float16 matrix of ``N**2`` entries. Seven criteria
       with two ESPs give up to 16,384 objects.
     - Use fewer criteria or ESPs, or pass smaller ``cvalues`` to
       ``esp_comet``. See :doc:`methodology`.
   * - ``Criterion 'X' contains value v outside bounds [lo, hi]``
     - ``validate(X, labels)`` received data outside the model domain; COMET
       cannot score it.
     - Clip the data, or widen ``bounds`` when building the model.
   * - ``X must have m columns, got k``
     - The validation data has the wrong number of columns.
     - Select the criteria columns. A DataFrame must contain every criterion
       name exactly once; valid named columns are reordered automatically.
   * - ``labels must contain at least one positive`` / ``negative``
     - The labels are all equal, so group differences and lift are undefined.
     - Check the label column and the cut-off.
   * - Error about non-finite scores or zero variance
     - The model returns NaN or infinity, or all sampled scores are equal.
     - Fix the scoring function or choose bounds over which it varies.
       A constant model has undefined Sobol' indices.
   * - Error about missing or duplicate criterion names
     - DataFrame columns cannot be matched to the study criteria.
     - Use each study criterion name once. Arrays use positional order.
   * - Error about the label index
     - A pandas label Series has missing, extra or duplicate row labels.
     - Use unique indexes with the same row labels. Validation aligns their order.
   * - Error about ``top_k``
     - A cut-off is not a positive integer.
     - Use positive integers. Values above the row count use all rows.
   * - ``Run result.validate(X, labels) first``
     - ``plot_validation`` needs validation results.
     - Call ``result.validate(X, labels)`` before the plot.
   * - ``plot_surface needs the model adapter`` / ``needs the model: this
       result has no adapter``
     - The result was built by hand, not by ``SobolStudy.run``, so nothing
       can score new points.
     - Use the result of ``SobolStudy.run``, or pass an adapter to
       ``gcisens.plots.plot_surface``.
   * - ``Second-order indices were not computed (second_order=False)``
     - The study ran without pairwise indices.
     - Run with ``second_order=True`` (the default).

Reading the results
-------------------

.. list-table::
   :header-rows: 1
   :widths: 34 30 36

   * - Symptom
     - Cause
     - Fix
   * - Ranks such as ``2.5`` in the table
     - Tied values share their average rank; equal declared weights all get
       the same rank.
     - Expected. The diagnosis uses the same ranks.
   * - ``rho_w_S1`` shows ``n/a`` (``NaN``)
     - Spearman correlation is undefined when one side is constant, for
       example equal declared weights.
     - Expected. Compare ``w`` with ``S1`` in the table instead.
   * - Small negative ``S1`` or ``ST - S1``
     - Estimator noise at small effects.
     - Increase ``n_samples``; treat values inside the confidence half-width
       as zero.
   * - Article values differ in the third decimal
     - NumPy older than 2.3 detects distance ties in pymcdm differently.
     - ``pip install "numpy>=2.3"``.
   * - HTML report says ``Plots skipped: ...`` and a ``UserWarning`` appears
     - A plot failed; the report keeps the tables.
     - Fix the error named in the message, or pass ``include_plots=False``.
