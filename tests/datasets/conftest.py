import os
import pathlib
from typing import Type

import _pytest.config
import boto3
import fsspec
import pytest
from moto.server import ThreadedMotoServer

PytestConfig = Type[_pytest.config.Config]


def path_to_file(pytestconfig: PytestConfig, *other: str) -> pathlib.Path:
    """Get an absolute path to the given target file."""
    return pytestconfig.rootpath.joinpath(*other)  # type: ignore


@pytest.fixture(scope="module")
def fixture_path_to_assets(pytestconfig: PytestConfig) -> pathlib.Path:
    """Get an absolute path to the assets directory."""
    return path_to_file(pytestconfig, "tests", "datasets", "assets")


@pytest.fixture(scope="module")
def fixture_path_to_this_test(request) -> pathlib.Path:  # type:ignore[no-untyped-def]
    """Get an absolute path to the module invoking this fixture.

    'request' is a magic pytest fixture which contains all information about
    the current test and fixture construction.
    """
    return pathlib.Path(os.path.dirname(request.module.__file__))


@pytest.fixture(scope="module")
def fixture_path_to_output(pytestconfig: PytestConfig) -> pathlib.Path:
    """Get an absolute path to the output directory"""
    return path_to_file(pytestconfig, "tests", "output")


@pytest.fixture(scope="module")
def _fixture_mock_moto_server():
    """A moto server to mock S3 interactions.

    We cannot use the usual moto decorators because pyarrow's S3 FileSystem
    is not based on boto3 at all.

    References:
        https://issues.apache.org/jira/browse/ARROW-16437
    """

    server = ThreadedMotoServer()
    try:
        server.start()
        yield
    finally:
        server.stop()


@pytest.fixture()
def _fixture_mock_aws_environment(monkeypatch):
    """A mock AWS credentials environment."""
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    monkeypatch.setenv("AWS_SECURITY_TOKEN", "testing")
    monkeypatch.setenv("AWS_SESSION_TOKEN", "testing")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    monkeypatch.setenv("AWS_REGION", "us-east-1")


@pytest.fixture()
def fixture_mock_s3_client(_fixture_mock_aws_environment, _fixture_mock_moto_server):
    """An S3 client to the mocked moto AWS server."""
    client = boto3.client("s3", endpoint_url="http://localhost:5000")
    return client


@pytest.fixture()
def fixture_mock_s3_filesystem(_fixture_mock_aws_environment):
    """An S3 FileSystem connected to the mocked AWS server"""
    fs = fsspec.filesystem("s3", endpoint_override="http://localhost:5000")
    return fs
