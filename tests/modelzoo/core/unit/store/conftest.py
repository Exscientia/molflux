from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

import pytest
from cloudpathlib import implementation_registry
from cloudpathlib.local import LocalS3Client, local_s3_implementation


@pytest.fixture
def monkeypatched_aws_credentials(monkeypatch):
    """Mocked AWS Credentials to make sure we are not accessing real resources."""
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    monkeypatch.setenv("AWS_SECURITY_TOKEN", "testing")
    monkeypatch.setenv("AWS_SESSION_TOKEN", "testing")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "eu-west-1")


@pytest.fixture
def fixture_empty_bucket(monkeypatch, monkeypatched_aws_credentials):
    """Fixture that patches CloudPath dispatch and also sets up test assets in LocalS3Client's
    local storage directory.

    References:
        https://cloudpathlib.drivendata.org/stable/testing_mocked_cloudpathlib/
    """

    monkeypatch.setitem(implementation_registry, "s3", local_s3_implementation)

    mock_bucket: Path = LocalS3Client.get_default_storage_dir() / "mock-bucket"
    mock_bucket.mkdir(exist_ok=True, parents=True)

    yield mock_bucket

    LocalS3Client.reset_default_storage_dir()  # clean up temp directory and replace with new one
