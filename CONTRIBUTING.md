# Contributing

Thank you for helping with `gcisens`. Open an issue before a large change so
that the scope is agreed first.

## Development setup

```bash
git clone https://github.com/Pxj04/gcisens.git
cd gcisens
python -m venv .venv && source .venv/bin/activate   # Python >= 3.11
pip install -e ".[dev,docs]" build twine
```

## Checks

Run these from the repository root before you open a pull request. CI runs
the lint, test, coverage and documentation checks; the release workflow runs
the package check.

| Check | Command |
|---|---|
| Lint and format | `ruff check gcisens tests examples && ruff format --check gcisens tests examples` |
| Fast tests | `pytest -m "not slow" -q` |
| Article reproduction | `pytest -m slow -q` |
| Coverage | `pytest --cov=gcisens --cov-report=term-missing` |
| Package | `python -m build && twine check dist/*` |
| Documentation | `python -m sphinx -b html -W --keep-going docs _build/html` |

Rules:

- Add or update a test for every behaviour change.
- Keep the public API backwards compatible; record breaking changes in
  `CHANGELOG.md` under "Unreleased".
- Use the terms defined in `CONTEXT.md` (view, declared weight, rank, ...).
- Do not add an AI assistant as a commit co-author.

## Release steps

1. Move the "Unreleased" entries of `CHANGELOG.md` under a new version
   heading with the date.
2. Set the version in `pyproject.toml` and `CITATION.cff` (`version`, add
   `date-released`).
3. Run all checks above; the article reproduction tests must pass.
4. Merge to `main`, then tag: `git tag vX.Y.Z && git push origin vX.Y.Z`.
5. Publish a GitHub release for the tag. This triggers
   `publish-to-pypi.yml`, which checks that the tag matches the package
   version, builds the distributions and uploads them to PyPI.
6. With the GitHub–Zenodo integration enabled for the repository, Zenodo
   archives the published release and mints a DOI. Add the DOI badge to
   `README.md` and the `doi` field to `CITATION.cff`.
7. Check that Read the Docs built the new version.
