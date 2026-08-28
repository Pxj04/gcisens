Validation against observed outcomes
====================================

Sensitivity analysis explains the model's internal behaviour. Validation asks
a different question: whether its scores separate known outcome groups.

Given an alternatives matrix and binary labels, call
:meth:`gcisens.StudyResult.validate`:

.. code-block:: python

   labels = dataframe["Attrition"].eq("Yes")
   X = dataframe[names]

   validation = result.validate(X, labels, top_k=(50, 100, 200))
   print(validation.groups)
   print(validation.lift)

``groups`` reports the score count, mean, median and standard deviation for
positive and negative observations. ``lift`` reports how concentrated the
positive class is among the top ``k`` alternatives relative to its overall
base rate.

Score direction
---------------

The method automatically ranks COMET scores from high to low and SPOTIS
distances from low to high. For a custom model, pass ``ascending=True`` when a
lower score represents higher priority:

.. code-block:: python

   validation = result.validate(X, labels, ascending=True)

After validation, :meth:`gcisens.StudyResult.plot_validation` shows the score
distributions and export methods include the validation tables.

Validation does not establish causality and does not replace out-of-sample
evaluation. Labels should be independent of the rules used to construct the
model whenever possible.
