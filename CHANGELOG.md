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

### Changed

- Require NumPy 2.3 or newer for article reproduction.
- Raise compiled dependency bounds to versions compatible with NumPy 2.3.
- Require pymcdm 1.4 or newer and SALib 1.5.1 through 1.x.
- Use one tie-aware rank definition across tables, diagnosis and plots.
- Document `local_percent_step` and recommend the `sobol` sampler for new studies.
- Cache the COMET weight fit and warn when its characteristic-object grid is large.
- Include all rank correlations in comparison tables.
- Breaking: replace the `R2` summary key and comparison row with `r2_fit` and `r2_samples`. `StudyResult.r2` stays as a compatibility alias.
- Set the package version to 0.1.3.

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
