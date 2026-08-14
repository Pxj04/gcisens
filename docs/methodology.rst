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
   - *interaction dominance*: :math:`(ST - S1)/ST \ge 0.30`;
   - *moderate discrepancy*: rank displacement :math:`\ge 2`, or
     :math:`w < 0.01` while :math:`ST \ge 0.02`;
   - *confirmed transparency*: otherwise.

   Thresholds are the article defaults and are configurable via
   :class:`gcisens.DiagnosisThresholds`.

Reproduction
------------

``examples/article_esp_comet.py`` reproduces the three experiments of the
KES 2026 article; the same reproduction runs in CI as a regression test
(``pytest -m slow``).
