import pytest
from cloudpathlib import implementation_registry
from cloudpathlib.local import LocalS3Client, LocalS3Path, local_s3_implementation


@pytest.fixture
def fixture_test_bucket(monkeypatch):
    """Fixture that patches cloudpathlib to provide a mock S3 bucket.

    References:
        https://cloudpathlib.drivendata.org/stable/testing_mocked_cloudpathlib/
    """

    monkeypatch.setitem(implementation_registry, "s3", local_s3_implementation)

    local_cloud_path = LocalS3Path("s3://cloudpathlib-test-bucket")
    yield local_cloud_path

    # clean up temp directory and replace with new one
    LocalS3Client.reset_default_storage_dir()


@pytest.fixture
def fixture_nested_tmp_path(tmp_path):
    """A nested version of pytest's tmp_path fixture.

    Useful to test operations that should work on non-yet-existent directories.
    """
    yield tmp_path / "sub" / "folder"


@pytest.fixture
def fixture_nested_test_bucket(fixture_test_bucket):
    """A 'nested' test bucket.

    Useful to test operations that should work on non-yet-existent buckets.
    """
    yield fixture_test_bucket / "sub" / "bucket"
