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
       additive linear model, each variance contribution is proportional to
       the squared coefficient times the input variance. The
       ESP-SPOTIS demo shows ``w = 0.25`` next to ``S1 = 0.41`` for a model
       with no criterion interactions. Rank-based rules compare order rather
       than magnitude; absolute thresholds on ``ST - w`` are heuristics.
   * - Threshold defaults
     - :class:`gcisens.DiagnosisThresholds` defaults come from one case study
       with seven criteria, KES 2026. Check their suitability for another
       domain, input distribution or criterion count, and report a threshold
       sweep with the diagnosis. The thresholds compare dimensionless indices
       and normalised weights. An affine change of score units with a nonzero
       scale factor does not change Sobol' indices.
   * - COMET weights
     - COMET takes no weights as input. Its weights in ``w``
       are regression estimates: ``gcisens`` fits a linear model on
       the characteristic objects, drops the sign of the coefficients and
       normalises their absolute values. ``r2_fit`` reports the fit quality.
       A low ``r2_fit`` means that the linear summary explains little of the
       model.
   * - Negative indices
     - Small negative ``S1`` or ``ST - S1`` values are estimator noise, not
       evidence of a negative effect. Inspect the confidence interval before
       interpreting a small effect.
   * - Confidence columns
     - ``S1_conf``, ``ST_conf`` and ``S2_conf`` are half-widths of bootstrap
       confidence intervals at ``conf_level`` (default 0.95) from
       ``num_resamples`` bootstrap draws (default 100). They are not standard
       errors.
   * - Reproducibility
     - Use the Python 3.11 environment in ``requirements-repro.txt``
       to reproduce the article values. pymcdm's
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
       ``esp_comet`` warns above 20,000 objects before it builds the model.
       Reduce the number of criteria or ESPs, or pass smaller ``cvalues``,
       when the warning appears.

Scoring and diagnosis contracts
-------------------------------

A model must define a fixed, deterministic function of one criterion vector.
Its score for that vector must not depend on other rows in the input matrix.
Each evaluated row must produce one finite score. A constant output has zero
variance, so its Sobol' indices are undefined and the study raises an error.
See :doc:`advanced` before passing a custom callable.

Local weights use a conditional range sweep. Each criterion varies over its
full bounds while the other criteria stay at the reference point. They do not
measure only a small neighbourhood around that point.

The discrepancy rules use point estimates and do not use their bootstrap
confidence intervals. The retained category name ``confirmed transparency``
means only that no rule found a discrepancy at the selected thresholds. It is
not proof that the model is transparent. Inspect uncertainty, threshold
sensitivity and sample-size stability when interpreting a result.

Threshold sensitivity: a worked example
---------------------------------------

The categories of the report depend on ``hidden_st_excess`` (how much
``ST`` must exceed a near-zero weight) and ``interaction_ratio`` (which share
of ``ST`` must come from interactions). :meth:`gcisens.StudyResult.sweep_thresholds`
re-classifies the criteria for every combination of values, so the stability
of a diagnosis can be reported next to the diagnosis itself.

The example uses the ESP1+ESP2 configuration of the KES 2026 case study
(``examples/article_esp_comet.py``):

.. code-block:: python

   result = SobolStudy(
       model, n_samples=2048, sampler="saltelli", seed=42,
   ).run()
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
