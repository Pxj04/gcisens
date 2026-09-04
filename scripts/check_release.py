"""Check the project's release fields without a separate version file."""

import argparse
import re
import tomllib
from datetime import date
from pathlib import Path


def check_release(root: Path, tag: str | None = None) -> str:
    """Return the version, or raise ValueError for inconsistent metadata."""
    version = tomllib.loads((root / "pyproject.toml").read_text())["project"]["version"]
    cff = (root / "CITATION.cff").read_text()
    changelog = (root / "CHANGELOG.md").read_text()

    # Only these two controlled scalar fields are read; this is not a YAML parser.
    def field(name):
        match = re.search(
            rf"^{name}:[ \t]+(?P<quote>[\"']?)(?P<value>[^\"'\s]+)(?P=quote)[ \t]*$",
            cff,
            re.MULTILINE,
        )
        if match is None:
            raise ValueError(f"CITATION.cff needs a valid {name} field")
        return match["value"]

    if field("version") != version:
        raise ValueError("CITATION.cff and pyproject.toml versions differ")
    released = field("date-released")
    date.fromisoformat(released)
    headings = re.findall(
        r"^## \[([^]]+)\](?: - (\d{4}-\d{2}-\d{2}))?\s*$", changelog, re.MULTILINE
    )
    if headings[:2] != [("Unreleased", ""), (version, released)]:
        raise ValueError("The latest changelog version/date must match CITATION.cff")
    links = dict(re.findall(r"^\[([^]]+)\]: (\S+)\s*$", changelog, re.MULTILINE))
    if not links.get("Unreleased", "").endswith(f"/compare/v{version}...HEAD"):
        raise ValueError("Update the Unreleased comparison link in CHANGELOG.md")
    if not links.get(version, "").endswith((f"...v{version}", f"/tag/v{version}")):
        raise ValueError("Add the current version link in CHANGELOG.md")
    if tag is not None:
        if tag != f"v{version}":
            raise ValueError(f"Release tag {tag!r} must be 'v{version}'")
        unreleased = changelog.split("## [Unreleased]", 1)[1].split("\n## [", 1)[0]
        if unreleased.strip():
            raise ValueError("Move Unreleased entries into the release before publishing")
    return version


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", help="Also check the release tag and an empty Unreleased section")
    args = parser.parse_args()
    try:
        checked = check_release(Path(__file__).resolve().parents[1], args.tag)
    except ValueError as error:
        parser.exit(1, f"Release metadata error: {error}\n")
    print(f"Release metadata OK: {checked}")
