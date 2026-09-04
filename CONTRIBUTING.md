# Contributing

Keep the main workflow small: build a model, run `SobolStudy`, inspect the
result, and export it. A new public option needs a use case and an example.
Use SALib for Sobol' analysis and pymcdm for the decision models. Open an issue
before a large change to agree on its scope.

## Development setup

Use Python 3.11 or newer.

```bash
git clone https://github.com/Pxj04/gcisens.git
cd gcisens
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev,docs]" build twine
```

## Where to make a change

Test paths below are relative to `tests/`. Implementation paths are relative
to `gcisens/`.

| Change | Implementation | Tests and documentation |
|---|---|---|
| Sampling or Sobol' indices | `sensitivity.py` | `test_sensitivity.py`, article tests, `docs/methodology.rst` |
| Global or local weights | `weights.py` | `test_weights.py`, agreement with pymcdm, article tests |
| Diagnosis rules or thresholds | `diagnosis.py` | `test_diagnosis.py`, threshold sweeps, methodology, affected article tables |
| Model support or input checks | `adapters.py`, `builders.py` | `test_adapters.py`, `test_validation_inputs.py`, a native pymcdm example |
| Result data or workflow | `study.py` | Study, record and view tests; affected tables, plots and exports |
| Label validation | `validation.py` | Validation tests, including score orientation |
| A plot or export | `plots.py`, `export.py` | Plot or export tests; inspect one output file |
| A public symbol | `__init__.py` | `docs/api.rst`, examples and migration notes |
| A dependency bound | `pyproject.toml` | Minimum versions in `.github/workflows/ci.yml`; article reproduction |
| User introduction | `README.md` | The About page includes the README; do not copy it |
| A release | `pyproject.toml`, `CITATION.cff`, `CHANGELOG.md` | Version, date, comparison links, all checks below |

For each fix:

1. Add a small regression test with a known answer for the changed behaviour.
2. Update the user instructions if the change affects them. Use the terms in
   `CONTEXT.md`.
3. Add an entry under `Unreleased` in `CHANGELOG.md`. State the trigger and the
   corrected behaviour. Keep the version and release date until release preparation.
4. Run the focused tests, then the relevant checks below before review.

Do not change expected scientific values only to make tests pass. Explain why
the values changed and whether the published results need a correction.
Do not add an AI assistant as a commit co-author.

## API changes

Use patch releases for fixes that preserve the public call pattern. Record
corrections to numerical results in the changelog. Use a separate minor release
for removed symbols, changed table keys, changed defaults or new diagnosis
rules. Give a before/after migration example. Keep compatibility aliases until
the announced removal release; keep them out of the main tutorial.

Custom scoring functions must return one finite score per row. A point's score
must not depend on other rows in the batch. A wrapper around a pymcdm method
does not by itself meet this requirement.

## Checks

```bash
ruff check gcisens tests examples scripts
ruff format --check gcisens tests examples scripts
pytest --cov=gcisens --cov-report=term-missing -q
python -m sphinx -b html -W --keep-going docs _build/html
python scripts/check_release.py
python -m build
python -m twine check dist/*
python scripts/check_wheel.py dist/*.whl
```

Build in a fresh checkout or an empty `dist/` directory. The wheel check needs
network access to install dependencies. It creates a temporary environment
outside the checkout, checks a linear model against known Sobol' indices, and
writes a CSV report.

The full pytest run includes the article tests. Use `pytest -m "not slow" -q`
for a quick development run, or `pytest -m slow -q` for article tests only.
CI runs the full suite once per supported Python version, with coverage on 3.11.
It also checks minimum dependency versions, the exact reproduction environment,
documentation, release metadata, and the built wheel.

## Article reproduction

`requirements-repro.txt` pins the runtime dependencies and pytest from the
verified Python 3.11.15 environment on macOS arm64. CI checks the same pins on
Linux x86_64. It is separate from the supported dependency ranges in
`pyproject.toml`. It does not pin build, lint or documentation tools.

Use Python 3.11.15 in a new environment. From the source checkout:

```bash
python3.11 -m venv /tmp/gcisens-repro
source /tmp/gcisens-repro/bin/activate
python -m pip install -r requirements-repro.txt
python -m pip install --no-deps -e .
python -m pip check
pytest -m slow -q
```

These tests reproduce the existing KES article tables. Keep their deterministic
Saltelli sampler when checking those results. A new SoftwareX experiment needs
its own saved configuration and outputs.

For each article release, archive the source commit or tag, Python version,
OS and architecture, dependency file, input data checksum, and output files.
Use the dataset source and licence recorded in `examples/data/README.md`.
Floating-point results can differ across platforms; the article tests use
explicit tolerances. Verify all article outputs in the recorded environment.

When dependencies change, use a new environment, update the exact runtime and
pytest pins, run the article tests, and review numerical differences. Do not
replace an archived environment or alter a published tag. The minimum-version
job tests the package's supported lower bounds; it is not the article environment.

## Release procedure

1. Review `Unreleased` and choose a version. Identify changes to public calls,
   numerical results and dependency requirements.
2. Set the version in `pyproject.toml` and `CITATION.cff`. Set `date-released`
   in `CITATION.cff`. Move changelog entries to the same version and date; leave
   an empty `Unreleased` section.
3. Add the version comparison link in `CHANGELOG.md`. Set its `Unreleased`
   link to compare the new tag with `HEAD`.
4. Run all checks and `python scripts/check_release.py --tag vX.Y.Z`.
   For an article release, also archive the reproduction files listed above.
5. Merge the reviewed changes. Create `vX.Y.Z` on that exact commit, push the
   tag, and publish its GitHub release.
6. The publication workflow calls the same CI workflow on the released commit.
   Publication requires every validation job to pass. It uploads the wheel and
   source archive built and checked by that run, without a second build.
7. Check the PyPI version and the Read the Docs build. Save the archive URL and
   version DOI when available.

Never move a published tag or replace a published package. Fix errors in a new
release. For now, cite the software release. Add the SoftwareX article as a
preferred citation only when its published metadata are available.

A version DOI identifies an exact software release; a project DOI covers all
versions. If Zenodo creates a DOI after publication, a later `CITATION.cff`
edit is not part of the existing tag or archive. Record it without changing the
old tag. To include a DOI inside the release files, reserve it before the final
build through the archive service.
