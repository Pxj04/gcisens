Methodology
===========

This page states what the numbers in a :class:`gcisens.StudyResult` mean and
where the method stops. Read it before you interpret a Sensitivity
Discrepancy Report.

Assumptions and limitations
---------------------------

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Topic
     - What holds
   * - Input distribution
     - ``gcisens`` computes Sobol' indices for independent, uniform inputs
       over ``bounds``. The indices describe the model over the criteria
       hyper-rectangle, not over the distribution of real alternatives. A
       criterion that is almost constant in the data can still carry a large
       index.
   * - Weights and ``S1`` scales
     - Weights and first-order indices live on different scales. For an
       additive linear model ``S1`` grows with the square of the weight. The
       ESP-SPOTIS demo shows ``w = 0.25`` next to ``S1 = 0.41`` for a model
       that is perfectly transparent. Rank-based rules (rank displacement,
       Spearman correlations) are robust to this; absolute thresholds on
       ``ST - w`` are heuristics.
   * - Threshold defaults
     - :class:`gcisens.DiagnosisThresholds` defaults come from one case study
       (KES 2026) with scores in [0, 1] and seven criteria. Recalibrate them
       for other score scales and criteria counts, and report a threshold
       sweep (below) with every diagnosis.
   * - COMET weights
     - COMET takes no weights as input. Its declared weights (the ``w``
       view) are regression estimates: ``gcisens`` fits a linear model on
       the characteristic objects, drops the sign of the coefficients and
       normalises their absolute values. ``r2_fit`` reports the fit quality.
       A low ``r2_fit`` means that the linear summary explains little of the
       model.
   * - Negative indices
     - Small negative ``S1`` or ``ST - S1`` values are estimator noise, not
       evidence of a negative effect. Treat values inside the confidence
       half-width as zero.
   * - Confidence columns
     - ``S1_conf``, ``ST_conf`` and ``S2_conf`` are half-widths of bootstrap
       confidence intervals at ``conf_level`` (default 0.95) from
       ``num_resamples`` bootstrap draws (default 100). They are not standard
       errors.
   * - Reproducibility
     - Use NumPy 2.3 or newer to reproduce the article values. pymcdm's
       ``ESPExpert`` detects distance ties with exact float equality, and
       NumPy 2.3 changed float reduction. Older NumPy versions give multi-ESP
       weights that differ by up to 0.0011 and a different ``r2_fit``.
       The ``"saltelli"`` sampler is deterministic. ``seed`` affects only the
       bootstrap intervals, the ``"sobol"`` sampler and ``r2_samples``.
   * - Memory
     - The COMET characteristic-object count is the product of the numbers of
       characteristic values per criterion, and pymcdm stores a float16
       judgment matrix of size ``count**2``. Seven criteria with two ESPs
       give up to ``4**7 = 16,384`` objects. The KES 2026 case study has
       12,288, because the two ESPs share one value, and needs about 290 MiB.
       ``comet_global_weights`` warns above 20,000 objects. Reduce the number
       of criteria or ESPs when the warning appears.

Threshold sensitivity: a worked example
---------------------------------------

The categories of the report depend on ``hidden_st_excess`` (how much
``ST`` must exceed a near-zero weight) and ``interaction_ratio`` (which share
of ``ST`` must come from interactions). :func:`gcisens.sweep_thresholds`
re-classifies the criteria for every combination of values, so the stability
of a diagnosis can be reported next to the diagnosis itself.

The example uses the ESP1+ESP2 configuration of the KES 2026 case study
(``examples/article_esp_comet.py``):

.. code-block:: python

   result = SobolStudy(model, n_samples=2048, seed=42).run()
   sweep = result.sweep_thresholds(
       hidden_st_excess=[0.01, 0.03, 0.05],
       interaction_ratio=[0.2, 0.3, 0.4],
   )
   print(sweep.to_string(index=False))

The default thresholds are ``hidden_st_excess = 0.03`` and
``interaction_ratio = 0.30``. Three criteria (``DistanceFromHome``,
``MonthlyIncome``, ``PercentSalaryHike``) stay *confirmed transparency* over
the whole grid. The other four move:

.. list-table::
   :header-rows: 1

   * - ``hidden_st_excess``
     - ``interaction_ratio``
     - Age
     - NumCompaniesWorked
     - TotalWorkingYears
     - YearsAtCompany
   * - 0.01
     - 0.2, 0.3
     - HI
     - HI
     - HI
     - ID
   * - 0.01
     - 0.4
     - HI
     - HI
     - HI
     - MD
   * - 0.03
     - 0.2, 0.3
     - ID
     - HI
     - HI
     - ID
   * - 0.03
     - 0.4
     - MD
     - HI
     - HI
     - MD
   * - 0.05
     - 0.2, 0.3
     - ID
     - ID
     - ID
     - ID
   * - 0.05
     - 0.4
     - MD
     - CT
     - ID
     - MD

HI = hidden influence, ID = interaction dominance, MD = moderate discrepancy,
CT = confirmed transparency.

How to read it:

- ``NumCompaniesWorked`` and ``TotalWorkingYears`` are flagged at every grid
  point until ``hidden_st_excess`` reaches 0.05. Their ``ST - w`` values are
  0.032 and 0.031, so the *hidden influence* label holds for any excess below
  that. Above it the same criteria are still flagged, now as *interaction
  dominance*, because interactions carry 37% and 67% of their total effect.
- ``Age`` and ``YearsAtCompany`` are never *confirmed transparency*. The
  category changes with the thresholds, the finding does not: both criteria
  are displaced by two ranks between ``w`` and ``S1``, so the fallback rule
  (*moderate discrepancy*) catches them whenever the stricter rules release
  them.
- ``NumCompaniesWorked`` becomes *confirmed transparency* only at
  ``hidden_st_excess = 0.05`` with ``interaction_ratio = 0.4``. That corner is
  the one point of the grid where a weight-only reading of the model would
  pass unchallenged.

A diagnosis is robust when the set of flagged criteria does not change over a
plausible threshold range, even if the category names do. Report the sweep
table, or its summary, together with the default report.
