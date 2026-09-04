# Changelog

This file records notable changes to the project. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

These changes include a smaller wildcard-import interface and read-only result
records. Prepare them as a minor release; the published 0.1.3 tag is unchanged.

### Fixed

- Reject constant or non-finite scores and non-finite Sobol indices before
  diagnosis. Keep finite negative estimates, which can arise from sampling error.
- Make bootstrap intervals reproducible with `seed=0` on SALib 1.5.
- Require valid sample counts, bootstrap settings, ESPs, criteria names, SPOTIS
  types, binary labels and positive integer `top_k`. Reject local steps outside
  `(0, 1)` and reference points outside the domain. Large `top_k` remains capped.
- Select DataFrame criteria by unique names and reject missing names. Align a
  labels Series by matching unique row indexes; array labels remain positional.
- Protect numerical results, thresholds and validation data from mutation.
  Tables and ranks are detached copies. S1/ST views share the canonical Sobol
  arrays. Rebuilding a manual result recalculates its diagnosis; changing the
  numbers of a recorded run requires a new study so its provenance stays valid.
- Snapshot adapter settings for later validation and surface plots. The original
  model remains attached and must not be modified after analysis.
- Remove stale optional CSV files when exporting a result without S2 or label
  validation under the same prefix. Keep unrelated files and other prefixes.
- Return an empty pairwise table for one criterion.

### Added

- `result.metadata()` and `{prefix}_metadata.json` with model inputs, sampling
  settings, actual seed, thresholds, local-weight settings and software versions.
  HTML contains the same JSON. An omitted seed is generated and recorded so the
  full study can be repeated. Keep the model construction script with the files.
- An exact article environment in `requirements-repro.txt`, dataset checksum
  tests, and `reproduction.json` from the article script with input/source hashes
  and each configuration's metadata.
- Shared validation for CI and publication, including minimum dependencies,
  article reproduction, release metadata, and a numerical wheel test outside the
  checkout. Publication uses the checked distributions after all jobs pass.
- A contributor map from each change to its files and checks, plus release and
  archive instructions. A small synthetic example is in `examples/quickstart.py`.

### Changed

- Keep the main documented API to `SobolStudy`, `StudyResult`, `compare`,
  `Comparison`, `esp_comet`, `esp_spotis` and `DiagnosisThresholds`. These are now
  the symbols in `__all__`. Existing explicit imports of helper records and
  pymcdm re-exports still work; import pymcdm classes directly in new code.
- Use a plain light HTML report and state the source of the weights. Explain
  that `confirmed transparency` means no discrepancy under the selected rules,
  not proof of transparency. Category codes and numerical rules are unchanged.
- Describe custom callables as deterministic, row-independent scoring functions.
  Clarify COMET regression weights, full-domain conditional local sweeps and
  dimensionless diagnosis thresholds.
- Keep `saltelli` as the compatibility default and select it explicitly in article
  scripts/tests. New examples select `sobol`. Import the legacy sampler lazily.
- Cite the software while the SoftwareX article is unpublished.

### Migration

Use explicit imports instead of relying on helper symbols from wildcard imports:

```python
from gcisens import SobolStudy
from pymcdm.methods import COMET, SPOTIS
from pymcdm.methods.comet_tools import ESPExpert
```

Use `result.weights.copy()` or `result.table()` to edit values for a separate
analysis. Do not modify a result in place. Run another study to change its model
or numerical settings. Use `result.sweep_thresholds(...)` to compare diagnosis
thresholds. `ValidationResult` tables return copies; use `copy.copy` rather than
`dataclasses.replace` to copy that compatibility record. `SobolIndices.criteria_names`
and result views/diagnoses are tuples; `result.criteria_names` returns a list copy.

## [0.1.3] - 2026-09-02

### Added

- Validate study inputs. COMET models reject `weights` and `types`; declared weights must be non-negative and sum to 1; the builders check `esps`, `criteria_names` and `bounds` (`min < max`); `validate()` selects DataFrame columns by name and checks their count; `validate_scores` raises without positives or negatives and caps `top_k` at the sample size; `n_samples` (at least 2, power-of-two warning) and `sampler` are checked when the study is built.
- `warn_large_grid()` and `grid_regression_weights()` in `gcisens.weights`.
- Documentation pages "Troubleshooting" (messages, causes, fixes) and "References".
- Test the package version against installed distribution metadata.
- Check minimum dependency versions, documentation, formatting and coverage in CI.
- Keep `StudyResult.ranks` as a read-only compatibility view over `views`.
- Allow local-weight sweeps to include the upper bound.
- Expose Sobol' bootstrap and S2 significance settings.
- Report `r2_samples`, a linear-fit R² on one uniform sample of `n_r2_samples` points (default 4096). It is computed the same way for every model, next to the source-specific `r2_fit`.
- Accept `esps` in `SobolStudy` so surface plots of callable models can mark Expected Solution Points.
- `StudyResult.metrics`: summary metrics as `Metric` records that carry their display label; `Comparison.labels()` and the LaTeX comparison writer read it.
- `Category`: the diagnosis category constants carry a display label and a colour; the HTML report reads them.
- `StudyResult.view(key)`.
- `sweep_thresholds()` and `StudyResult.sweep_thresholds()`: re-classify the criteria over a grid of threshold values.
- Documentation page "Methodology": assumptions and limitations, and a worked threshold-sensitivity example.
- `CITATION.cff`, `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `examples/data/README.md` (dataset origin and licence).
- Source-article and local-weights references in the README.

### Changed

- `SobolIndices` and `ValidationResult` compare by identity (`eq=False`), as they hold arrays.
- Corrected the `seed` documentation: the default Saltelli sampler is deterministic, the seed drives the bootstrap intervals, the `sobol` sampler and the `r2_samples` sample.
- The documentation "About" page includes the README instead of copying it; the docs copyright line matches the LICENSE.
- Require NumPy 2.3 or newer for article reproduction.
- Raise compiled dependency bounds to versions compatible with NumPy 2.3.
- Require pymcdm 1.4 or newer and SALib 1.5.1 through 1.x.
- Use one tie-aware rank definition across tables, diagnosis and plots.
- Document `local_percent_step` and recommend the `sobol` sampler for new studies.
- Cache the COMET weight fit. `esp_comet` warns about a large characteristic-object grid *before* pymcdm allocates the judgment matrix; `make_adapter` warns for COMET models built by hand.
- Breaking: `comet_global_weights(model, bounds)` is replaced by `grid_regression_weights(score_fn, grid_lines, bounds)`; the COMET adapter passes its own scores and grid lines, so no module outside `adapters.py` reads pymcdm internals.
- Include all rank correlations in comparison tables.
- Breaking: replace the `R2` summary key and comparison row with `r2_fit` and `r2_samples`. `StudyResult.r2` stays as a compatibility alias.
- Breaking: `ModelAdapter.declared_weights()` returns a `DeclaredWeights` record (weights, source label, optional R²); `CometAdapter.declared_weights_r2()` is removed.
- Adapters expose `esps` and `grid_lines()`; `plot_surface` draws every model, including callables, through the adapter and no longer special-cases two-criteria COMET models.
- Use one range-sweep implementation for local weights of every model (identical values to pymcdm's `get_local_weights` on COMET).
- Raise `ValueError` when an explicit `SobolStudy` argument differs from the metadata of a model built with `esp_comet` or `esp_spotis`.
- Set the package version to 0.1.3.
- `StudyResult` is a plain record: `views`, `sobol`, `diagnoses`, `r2_fit`, `r2_samples`, `thresholds`, `weights_source`, `reference_point`, `n_r2_samples`, `validation` and an optional `adapter`. `weights`, `local_weights`, `correlations` and `ranks` are derived. A record built by hand renders like one produced by a study; only `validate()` and `plot_surface()` need the adapter.
- Breaking: `gcisens.plots.plot_surface(result, adapter, ...)` takes the adapter explicitly; `StudyResult.plot_surface()` passes its own.
- Breaking: `Comparison.METRICS` is removed; comparison rows are the union of the results' metrics, so `rho_w_wloc` appears only when a result has a reference point.
- `StudyResult` compares by identity (`eq=False`), as it holds arrays.
- Export and plot tests run on a hand-built record; the HTML report test asserts embedded images.

## [0.1.2] - 2026-08-29

### Changed

- Expanded and reorganized the documentation.
- Added package author metadata.

## [0.1.1] - 2026-08-28

### Added

- Initial public release of the sensitivity analysis, diagnosis, plotting and export APIs.

[Unreleased]: https://github.com/Pxj04/gcisens/compare/v0.1.3...HEAD
[0.1.3]: https://github.com/Pxj04/gcisens/compare/v0.1.2...v0.1.3
[0.1.2]: https://github.com/Pxj04/gcisens/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/Pxj04/gcisens/releases/tag/v0.1.1
