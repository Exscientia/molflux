import pathlib

import _pytest.config
import pytest

PytestConfig = _pytest.config.Config


def path_to_file(pytestconfig: PytestConfig, *other: str) -> pathlib.Path:
    """Get an absolute path to the given target file."""
    return pytestconfig.rootpath.joinpath(*other)


@pytest.fixture(scope="module")
def fixture_path_to_assets(pytestconfig: PytestConfig) -> pathlib.Path:
    """Get an absolute path to the assets directory."""
    return path_to_file(pytestconfig, "tests", "metrics", "assets")
