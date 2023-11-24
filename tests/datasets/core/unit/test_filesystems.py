import fsspec

from molflux.datasets.filesystems import S3FileSystem


def test_s3_filesystem_is_fsspec_compliant(
    fixture_mock_s3_client,
    fixture_mock_s3_filesystem,
):
    """That our custom S3FileSystem implements the fsspec protocol."""
    filesystem = S3FileSystem()
    assert isinstance(filesystem, fsspec.AbstractFileSystem)


def test_s3_filesystem_is_registered_with_fsspec(
    fixture_mock_s3_client,
    fixture_mock_s3_filesystem,
):
    """That our custom S3FileSystem is registered as an official fsspec entrypoint.

    References:
        https://filesystem-spec.readthedocs.io/en/latest/developer.html#implementing-a-backend
    """
    filesystem = fsspec.filesystem("s3")
    assert isinstance(filesystem, S3FileSystem)


def test_s3_path_gets_mapped_to_our_s3_filesystem(
    fixture_mock_s3_client,
    fixture_mock_s3_filesystem,
):
    """That fsspec dispatches s3 paths to our custom S3FileSystem."""
    path = "s3://my-bucket/test"
    fs_token_paths = fsspec.get_fs_token_paths(path)
    assert isinstance(fs_token_paths[0], S3FileSystem)
