"""Prevent publication with mismatched source metadata or unreleased changes."""

import runpy
from pathlib import Path

import pytest

check_release = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "scripts/check_release.py")
)["check_release"]


@pytest.fixture
def release_tree(tmp_path):
    (tmp_path / "pyproject.toml").write_text('[project]\nversion = "1.2.3"\n')
    (tmp_path / "CITATION.cff").write_text('version: "1.2.3"\ndate-released: "2026-09-05"\n')
    (tmp_path / "CHANGELOG.md").write_text(
        "## [Unreleased]\n\n## [1.2.3] - 2026-09-05\n\n### Fixed\n\n- A bug.\n\n"
        "[Unreleased]: https://example.org/compare/v1.2.3...HEAD\n"
        "[1.2.3]: https://example.org/compare/v1.2.2...v1.2.3\n"
    )
    return tmp_path


def test_release_accepts_matching_tag(release_tree):
    assert check_release(release_tree, "v1.2.3") == "1.2.3"


@pytest.mark.parametrize(
    ("file", "before", "after", "message"),
    [
        ("CITATION.cff", 'version: "1.2.3"', 'version: "1.2.2"', "versions differ"),
        ("CITATION.cff", "2026-09-05", "2026-09-06", "changelog version/date"),
        ("CHANGELOG.md", "v1.2.3...HEAD", "v1.2.2...HEAD", "Unreleased comparison"),
        ("CHANGELOG.md", "v1.2.2...v1.2.3", "v1.2.1...v1.2.2", "current version link"),
    ],
)
def test_release_rejects_metadata_mismatch(release_tree, file, before, after, message):
    path = release_tree / file
    path.write_text(path.read_text().replace(before, after))
    with pytest.raises(ValueError, match=message):
        check_release(release_tree)


def test_release_rejects_wrong_tag(release_tree):
    with pytest.raises(ValueError, match="Release tag"):
        check_release(release_tree, "v1.2.2")


def test_unreleased_changes_pass_normal_checks_but_block_publication(release_tree):
    path = release_tree / "CHANGELOG.md"
    path.write_text(
        path.read_text().replace("## [Unreleased]", "## [Unreleased]\n\n- Pending fix.")
    )
    assert check_release(release_tree) == "1.2.3"
    with pytest.raises(ValueError, match="Move Unreleased entries"):
        check_release(release_tree, "v1.2.3")


@pytest.mark.parametrize("version", ["1.2.3rc1", "1.2.3.post1", "1.2.3.dev1", "1.2.3+local.1"])
def test_release_accepts_matching_pep440_versions(release_tree, version):
    for name in ("pyproject.toml", "CITATION.cff", "CHANGELOG.md"):
        path = release_tree / name
        path.write_text(path.read_text().replace("1.2.3", version))
    assert check_release(release_tree, f"v{version}") == version
