Understanding the results
=========================

Main table
----------

:meth:`gcisens.StudyResult.table` returns one row per criterion. Its principal
columns are:

``w``
   Declared weight for SPOTIS, or regression-based global weight for COMET and
   callables without supplied weights.
``w_loc``
   Optional local range-sweep importance at the reference point passed to
   :meth:`gcisens.SobolStudy.run`.
``S1``
   First-order Sobol' index: the criterion's independent contribution to output
   variance.
``ST``
   Total-order Sobol' index: the criterion's contribution including all
   interactions.
``ST_minus_S1``
   A practical interaction indicator. A large gap suggests the criterion acts
   jointly with other criteria.
``S1_conf``, ``ST_conf``
   Bootstrap confidence half-widths. Large intervals call for a larger sample
   size or more cautious interpretation.
``Rank_*``
   Descending ranks under each view, where 1 is most important.
``Category``
   The first matching Sensitivity Discrepancy Report category.

Diagnostic categories
---------------------

``hidden influence``
   The declared or estimated weight is small, but total-order sensitivity is
   materially larger. The model may rely on a criterion that its weights make
   easy to overlook.
``interaction dominance``
   A substantial share of the criterion's total effect comes from
   interactions rather than its isolated effect.
``moderate discrepancy``
   Weight and sensitivity rankings differ materially, or a nearly zero-weight
   criterion still has a non-negligible total effect.
``confirmed transparency``
   None of the discrepancy rules fired. This means the two views are
   consistent under the configured thresholds; it is not a general proof that
   the model is correct or fair.

Call :meth:`gcisens.StudyResult.diagnosis` to see the numeric reason attached
to every label.

Summary metrics
---------------

The :meth:`gcisens.StudyResult.summary` output includes:

* :math:`R^2`, the quality of a linear approximation of the decision surface;
* :math:`\sum S1` and :math:`\sum ST`, useful indicators of interaction
  structure and estimator stability;
* Spearman correlations between weights and Sobol' views;
* the sample and evaluation counts, sampler, and weight source.

An :math:`R^2` close to one means a linear weight summary explains most of the
model's score variation. A lower value warns that a single global weight
vector is an incomplete description.

Cautions
--------

Sobol' indices describe variation over the supplied bounds under independent,
uniform sampling. Changing bounds changes the question and can change the
indices. Confidence intervals and convergence across increasing sample sizes
should be checked before reporting small differences. Diagnostic thresholds
are defaults from the source workflow and should be calibrated to the decision
context when necessary.
