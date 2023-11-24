import pathlib
import uuid

import cloudpathlib
import fsspec
import pyarrow.parquet
import pytest

import datasets
from molflux.datasets import load_dataset_from_store, save_dataset_to_store


@pytest.fixture(scope="module")
def fixture_mock_dataset() -> datasets.Dataset:
    data = {"col1": [1, 2], "col2": [3, 4]}
    return datasets.Dataset.from_dict(data)


@pytest.fixture(scope="module")
def fixture_mock_dataset_dict() -> datasets.DatasetDict:
    data = datasets.Dataset.from_dict({"col1": [1, 2], "col2": [3, 4]})
    return datasets.DatasetDict(
        {
            "train": data,
            "validation": data,
            "test": data,
        },
    )


@pytest.mark.parametrize(
    ("path", "format"),
    [
        # simple paths
        ("data.csv", "csv"),
        ("data.parquet", "parquet"),
        ("data.json", "json"),
        ("data", "disk"),
        # nested paths
        ("sub/dir/data.csv", "csv"),
        ("sub/dir/data.parquet", "parquet"),
        ("sub/dir/data.json", "json"),
        ("sub/dir/data", "disk"),
        # pathlib paths
        (pathlib.Path("data.csv"), "csv"),
        (pathlib.Path("sub/dir/data.csv"), "csv"),
        # compressed paths
        ("data.csv.gz", "csv"),
        # paths with potentially problematic characters
        ("2023-03-22T17:53:17.245386.parquet", "parquet"),
    ],
)
def test_save_dataset_locally(tmp_path, fixture_mock_dataset, path, format):
    """That can persist a Dataset locally in one of the supported file
    formats.

    We also test that missing parent directories are automatically created if
    needed.
    """
    dataset = fixture_mock_dataset
    path = tmp_path / path

    save_dataset_to_store(dataset, path=str(path), format=format)

    if format == "disk":
        assert path.is_dir()
    else:
        assert path.is_file()


@pytest.mark.parametrize(
    ("path", "format"),
    [
        # simple paths
        ("s3://data.csv", "csv"),
        ("s3://data.parquet", "parquet"),
        ("s3://data.json", "json"),
        ("s3://data", "disk"),
        # nested paths
        ("s3://sub/dir/data.csv", "csv"),
        ("s3://sub/dir/data.parquet", "parquet"),
        ("s3://sub/dir/data.json", "json"),
        ("s3://sub/dir/data", "disk"),
        # cloudpathlib paths
        (cloudpathlib.S3Path("s3://data.csv"), "csv"),
        (cloudpathlib.S3Path("s3://sub/dir/data.csv"), "csv"),
        # compressed paths
        ("s3://data.csv.gz", "csv"),
        # paths with potentially problematic characters
        ("s3://data/2023-03-22T17:53:17.245386.parquet", "parquet"),
    ],
)
def test_save_dataset_to_s3(
    fixture_mock_s3_client,
    fixture_mock_s3_filesystem,
    fixture_mock_dataset,
    path,
    format,
):
    """That can persist a Dataset to the cloud in one of the supported file
    formats.

    We also test that missing parent directories are automatically created if
    needed (even though technically speaking in cloud paths there is not really
    a concept of subdirectories).
    """
    dataset = fixture_mock_dataset

    # Prepare s3 bucket
    client = fixture_mock_s3_client
    root_bucket = str(uuid.uuid4())
    client.create_bucket(Bucket=root_bucket)

    # point at mock bucket
    cloud_path = str(path).replace("s3://", f"s3://{root_bucket}/")
    if isinstance(path, cloudpathlib.CloudPath):
        cloud_path = cloudpathlib.S3Path(cloud_path)  # type: ignore[assignment]

    # Save to s3
    fs = fixture_mock_s3_filesystem
    save_dataset_to_store(dataset, path=cloud_path, format=format, fs=fs)

    if not pathlib.Path(str(cloud_path)).suffix:
        assert fs.isdir(str(cloud_path))
    else:
        assert fs.isfile(str(cloud_path))


@pytest.mark.parametrize(
    ("path", "format", "filesystem"),
    [
        ("s3://my/bucket/data.parquet", "parquet", "file"),
        ("my/data.parquet", "parquet", "s3"),
    ],
)
def test_save_dataset_with_mismatched_filesystem_raises(
    fixture_mock_s3_client,
    fixture_mock_s3_filesystem,
    fixture_mock_dataset,
    path,
    format,
    filesystem,
):
    """That attempting to save a dataset to the cloud but providing a
    local filesystem (local) raises (and vice-versa)."""
    dataset = fixture_mock_dataset
    with pytest.raises(ValueError, match="Incompatible filesystem"):
        save_dataset_to_store(
            dataset,
            path=str(path),
            format=format,
            fs=fsspec.filesystem(filesystem),
        )


@pytest.mark.parametrize(
    ("filename", "wrong_format"),
    [
        # simple paths
        ("data.csv", "disk"),
        ("data.parquet", "json"),
        ("data.json", "parquet"),
        ("data", "csv"),
    ],
)
def test_save_dataset_with_mismatched_file_format_raises(
    tmp_path,
    fixture_mock_dataset,
    filename,
    wrong_format,
):
    """That an error is raised if a file format is provided that does not
    match the data filename.
    """
    dataset = fixture_mock_dataset
    path = tmp_path / filename

    with pytest.raises(ValueError, match="Mismatched file format"):
        save_dataset_to_store(dataset, path=str(path), format=wrong_format)


def test_save_dataset_to_not_implemented_format_raises(fixture_mock_dataset):
    """That attempting to save a dataset to an unimplemented format raises."""
    dataset = fixture_mock_dataset

    with pytest.raises(ValueError, match="Unsupported dataset format"):
        save_dataset_to_store(dataset, path="test", format="invalid")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "filename",
    ["data.csv", "data.parquet", "data.json", "data", "data.csv.gz"],
)
def test_dataset_file_format_is_not_required_and_is_autoresolved(
    tmp_path,
    fixture_mock_dataset,
    filename,
):
    """That can persist a Dataset locally in one of the supported file
    formats without explicitly specifying the file format."""

    dataset = fixture_mock_dataset
    path = tmp_path / filename

    save_dataset_to_store(dataset, path=str(path))

    if not path.suffixes:
        assert path.is_dir()
    else:
        assert path.is_file()


@pytest.mark.parametrize(
    "compression",
    ["NONE", "SNAPPY", "GZIP", "BROTLI", "LZ4", "ZSTD"],
)
def test_save_dataset_as_compressed_parquet(
    tmp_path,
    fixture_mock_dataset,
    compression,
):
    """That can save datasets as compressed parquet files.

    Kwargs are forwarded to the backend huggingface .to_parquet() method.

    References:
        https://huggingface.co/docs/datasets/package_reference/main_classes#datasets.Dataset.to_parquet
        https://arrow.apache.org/docs/python/generated/pyarrow.parquet.ParquetWriter.html
    """
    dataset = fixture_mock_dataset

    path = tmp_path / "compressed.parquet"
    save_dataset_to_store(dataset, path=str(path), compression=compression)
    compression_metadata = (
        pyarrow.parquet.ParquetFile(path).metadata.row_group(0).column(0).compression
    )

    if compression == "NONE":
        assert compression_metadata == "UNCOMPRESSED"
    else:
        assert compression_metadata == compression


@pytest.mark.parametrize(
    ("path", "format"),
    [
        # simple paths
        ("data", "csv"),
        ("data", "parquet"),
        ("data", "json"),
        ("data", "disk"),
        # nested paths
        ("sub/dir/data", "csv"),
        ("sub/dir/data", "parquet"),
        ("sub/dir/data", "json"),
        ("sub/dir/data", "disk"),
    ],
)
def test_save_dataset_dict_locally(tmp_path, fixture_mock_dataset_dict, path, format):
    """That can persist a DatasetDict locally in one of the supported file
    formats.

    We also test that missing parent directories are automatically created if
    needed.
    """
    dataset_dict = fixture_mock_dataset_dict
    path = tmp_path / path

    save_dataset_to_store(dataset_dict, path=str(path), format=format)
    assert path.is_dir()

    # The split name formatting is technically an implementation detail
    # do not hesitate to adapt if code changes
    for split_name in dataset_dict:
        if format == "disk":
            expected_split_path = path / split_name
            assert expected_split_path.is_dir()
        else:
            expected_split_path = path / f"{split_name}.{format}"
            assert expected_split_path.is_file()


def test_save_dataset_dict_defaults_to_parquet(tmp_path, fixture_mock_dataset_dict):
    """That the default persistence format a DatasetDict is saved as is parquet.

    This is because we believe that .parquet is the most sensible persistence
    format for general usage
    """
    dataset_dict = fixture_mock_dataset_dict
    path = tmp_path / "data"

    save_dataset_to_store(dataset_dict, path=str(path))
    assert path.is_dir()

    # The split name formatting is technically an implementation detail
    # do not hesitate to adapt if code changes
    for split_name in dataset_dict:
        expected_split_path = path / f"{split_name}.parquet"
        assert expected_split_path.is_file()


def test_non_directory_like_path_for_dataset_dict_raises(
    tmp_path,
    fixture_mock_dataset_dict,
):
    """That DatasetDicts can only be saved to directories."""
    dataset_dict = fixture_mock_dataset_dict
    path = tmp_path / "data.parquet"
    with pytest.raises(ValueError, match="Invalid path"):
        save_dataset_to_store(dataset_dict, path=str(path), format="parquet")


@pytest.mark.parametrize(
    ("path", "format"),
    [
        ("sub/dir/data.csv", "csv"),
        ("sub/dir/data.parquet", "parquet"),
        ("sub/dir/data.json", "json"),
        ("sub/dir/data", "disk"),
    ],
)
def test_round_trip_dataset_io(tmp_path, fixture_mock_dataset, path, format):
    """That can save and load Datasets in a consistent fashion."""
    dataset = fixture_mock_dataset
    path = tmp_path / path

    save_dataset_to_store(dataset, path=str(path), format=format)
    loaded_dataset = load_dataset_from_store(str(path), format=format)

    assert loaded_dataset.column_names == dataset.column_names
    assert loaded_dataset.shape == dataset.shape


@pytest.mark.parametrize(
    ("path", "format"),
    [
        ("sub/dir/data", "csv"),
        ("sub/dir/data", "parquet"),
        ("sub/dir/data", "json"),
        ("sub/dir/data", "disk"),
        # test that default format is interpreted consistently
        ("sub/dir/data", None),
    ],
)
def test_round_trip_dataset_dict_io(tmp_path, fixture_mock_dataset_dict, path, format):
    """That can save and load DatasetDicts in a consistent fashion."""
    dataset_dict = fixture_mock_dataset_dict
    path = tmp_path / path

    save_dataset_to_store(dataset_dict, path=str(path), format=format)
    loaded_dataset_dict = load_dataset_from_store(str(path), format=format)

    assert loaded_dataset_dict.column_names == dataset_dict.column_names
    assert loaded_dataset_dict.shape == dataset_dict.shape
    assert list(loaded_dataset_dict.keys()) == list(dataset_dict.keys())
