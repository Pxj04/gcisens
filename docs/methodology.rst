Methodology
===========

The workflow implements two articles:

1. Śniegowski, Świder, Shekhovtsov, Sałabun — *Detecting Hidden Criterion
   Influence When Weights Mislead in Rule-Based Decision Support Systems*
   (KES 2026): the Sensitivity Discrepancy Report artifact.
2. Sałabun, Shekhovtsov, Wątróbski — *Variance-Based Analysis of Global
   Criteria Importance in the ESP-COMET Method* (ISD 2025): the Sobol'/
   Saltelli methodology and regression-based global weights.

Pipeline
--------

1. **Global weights.** For COMET: linear regression on the characteristic-
   object preferences (normalised to bounds); weights are normalised absolute
   coefficients, and the fit's :math:`R^2` measures how much of the decision
   surface the weights can explain. For SPOTIS: the user-declared weights.
2. **Local weights.** Range-sweep at a reference point (pymcdm's
   ``get_local_weights`` for COMET; a generic equivalent otherwise).
3. **Sobol' indices.** Saltelli sampling over the criteria bounds
   (:math:`N(2m+2)` evaluations), first-order :math:`S1`, total-order
   :math:`ST` and pairwise :math:`S2` indices with bootstrap confidence
   intervals.
4. **Rankings and correlations.** Tie-aware Spearman :math:`\rho` between the
   weight-based and variance-based views.
5. **Diagnosis.** Each criterion gets the first matching label:

   - *hidden influence*: :math:`w < \tfrac{1}{2m}` and :math:`ST - w \ge 0.03`;
   - *interaction dominance*: :math:`(ST - S1)/ST \ge 0.30` and
     :math:`ST-S1 \ge 0.02`;
   - *moderate discrepancy*: rank displacement :math:`\ge 2`, or
     :math:`w < 0.01` while :math:`ST \ge 0.02`;
   - *confirmed transparency*: otherwise.

   Thresholds are the article defaults and are configurable via
   :class:`gcisens.DiagnosisThresholds`.

Sampling assumptions
--------------------

The criteria are sampled independently and uniformly within their bounds.
Consequently, the indices describe the supplied decision domain, not an
empirical population distribution. Correlated or constrained criteria require
careful interpretation because the standard Sobol' decomposition assumes
independent inputs.

With :math:`m` criteria and base sample size :math:`N`, a second-order study
uses :math:`N(2m+2)` model evaluations. Disabling pairwise indices uses
:math:`N(m+2)` evaluations. Confidence intervals come from SALib's bootstrap
estimator.

Weight views
------------

For COMET, global weights are normalised absolute coefficients from a linear
regression over characteristic objects. They are an interpretable summary of
the decision surface, not parameters used by COMET itself. For SPOTIS, the
reported weights are the weights supplied to the model. For a custom callable
without weights, regression weights are estimated from a uniform sample.

Reproduction
------------

``examples/article_esp_comet.py`` reproduces the three experiments of the
KES 2026 article; the same reproduction runs in CI as a regression test
(``pytest -m slow``).

For exploratory runs, use a smaller power-of-two sample. For final results,
increase :math:`N` and verify that estimates and confidence intervals are
stable.
