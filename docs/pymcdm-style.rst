For pymcdm users
================

gcisens re-exports the relevant pymcdm classes 1:1 — same classes, not
copies. Build models exactly as in the pymcdm documentation, without
importing pymcdm:

.. code-block:: python

   from gcisens import COMET, SPOTIS, ESPExpert, SobolStudy

   expert = ESPExpert(esps=esps, bounds=bounds)
   cvalues = expert.make_cvalues_psi()   # [min, ESP..., max] grid
   model = COMET(cvalues, expert)

   result = SobolStudy(model, bounds=bounds, criteria_names=names).run()

Both styles produce identical objects and mix freely:

- a model built with plain ``import pymcdm`` also works in ``SobolStudy``;
- models built by :func:`gcisens.esp_comet` / :func:`gcisens.esp_spotis`
  remain plain pymcdm objects, fully compatible with ``pymcdm.visuals``;
- any other scoring model can be analysed through the callable fallback:
  ``SobolStudy(lambda X: scores(X), bounds=..., weights=...)``.
