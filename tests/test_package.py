from importlib.metadata import version

import gcisens


def test_package_version_matches_installed_metadata():
    assert gcisens.__version__ == version("gcisens")
