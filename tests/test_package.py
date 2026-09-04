from importlib.metadata import version

import gcisens


def test_package_version_matches_installed_metadata():
    assert gcisens.__version__ == version("gcisens")


def test_release_metadata_is_consistent():
    import runpy
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    check_release = runpy.run_path(str(root / "scripts" / "check_release.py"))["check_release"]
    assert check_release(root) == gcisens.__version__
