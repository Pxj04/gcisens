# Changelog

This file records notable changes to the project. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

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

### Changed

- Require NumPy 2.3 or newer for article reproduction.
- Raise compiled dependency bounds to versions compatible with NumPy 2.3.
- Require pymcdm 1.4 or newer and SALib 1.5.1 through 1.x.
- Use one tie-aware rank definition across tables, diagnosis and plots.
- Document `local_percent_step` and recommend the `sobol` sampler for new studies.
- Cache the COMET weight fit and warn when its characteristic-object grid is large.
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

[Unreleased]: https://github.com/Pxj04/gcisens/compare/v0.1.2...HEAD
[0.1.2]: https://github.com/Pxj04/gcisens/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/Pxj04/gcisens/releases/tag/v0.1.1
