import os
import uuid
from pathlib import Path
from types import SimpleNamespace

import fsspec
import pandas as pd
import pytest
from cloudpathlib import S3Path

import datasets
from molflux.datasets import (
    load_dataset,
    load_dataset_from_store,
    save_dataset_to_store,
)


@pytest.fixture(
    params=[
        # simple path
        "data.parquet",
        # path with potentially problematic characters
        "2023-03-22T17:53:17.245386.parquet",
    ],
)
def fixture_sample_parquet_dataset(tmpdir_factory, request):
    """A temporary sample parquet dataset."""
    data = {"col1": [1, 2], "col2": [3, 4]}
    df = pd.DataFrame(data=data)

    filename = request.param
    path = str(tmpdir_factory.mktemp("data").join(filename))
    df.to_parquet(path, index=False)

    shape = df.shape
    columns = list(data.keys())
    return SimpleNamespace(path=path, shape=shape, columns=columns)


@pytest.fixture(scope="module")
def fixture_collection_of_parquet_datasets(tmpdir_factory):
    """A couple of temporary sample parquet dataset."""
    data = {"col1": [1, 2], "col2": [3, 4]}
    df = pd.DataFrame(data=data)

    filename_one = str(tmpdir_factory.mktemp("data").join("data_one.parquet"))
    df.to_parquet(filename_one, index=False)

    filename_two = str(tmpdir_factory.mktemp("data").join("data_one.parquet"))
    df.to_parquet(filename_two, index=False)

    return filename_one, filename_two


@pytest.fixture(scope="module")
def fixture_sample_parquet_dataset_dict(tmpdir_factory):
    """A temporary sample DatasetDict saved as parquet."""

    data = datasets.Dataset.from_dict({"col1": [1, 2], "col2": [3, 4]})
    dataset_dict = datasets.DatasetDict(
        {
            "train": data,
            "validation": data,
            "test": data,
        },
    )

    dir_name = str(tmpdir_factory.mktemp("data").join("data"))
    save_dataset_to_store(dataset_dict, path=dir_name, format="parquet")
    return dir_name


def test_load_from_stored_parquet(fixture_sample_parquet_dataset):
    """That can load a parquet file as a Dataset."""
    parquet = fixture_sample_parquet_dataset

    # test both against str-paths and Path-like paths
    str_path = str(parquet.path)
    pathlib_path = Path(parquet.path)

    for path in [str_path, pathlib_path]:
        dataset = load_dataset_from_store(path, format="parquet")  # type: ignore[arg-type]
        assert isinstance(dataset, datasets.Dataset)


def test_can_load_from_stored_parquet_without_specifying_format(
    fixture_sample_parquet_dataset,
):
    """That can load a single parquet file as a Dataset without explicitly
    specifying the data 'format'."""
    parquet = fixture_sample_parquet_dataset
    dataset = load_dataset_from_store(parquet.path)
    assert isinstance(dataset, datasets.Dataset)


def test_can_load_from_formatless_file_if_format_provided_explicitly(tmp_path):
    """That explicitly specifying a file format allows loading datasets that
    whose format would not otherwise be automatically resolvable.
    """
    data = {"col1": [1, 2], "col2": [3, 4]}
    df = pd.DataFrame(data=data)
    path = str(tmp_path / "data")
    df.to_parquet(path, index=False)

    dataset = load_dataset_from_store(path, format="parquet")
    assert isinstance(dataset, datasets.Dataset)


def test_load_from_sequence_of_files(fixture_collection_of_parquet_datasets):
    """That can load a sequence of (coherent) files as a single Dataset."""
    path_one, path_two = fixture_collection_of_parquet_datasets

    # test both against str-paths and Path-like paths
    str_data_files = [str(path_one), str(path_two)]
    pathlib_data_files = [Path(path_one), Path(path_two)]

    for data_files in [str_data_files, pathlib_data_files]:
        dataset = load_dataset_from_store(data_files, format="parquet")  # type: ignore[arg-type]
        assert isinstance(dataset, datasets.Dataset)


def test_load_from_mapping_of_files_as_dataset_dict(
    fixture_collection_of_parquet_datasets,
):
    """That can load a mapping of (coherent) files as a single DatasetDict."""
    path_one, path_two = fixture_collection_of_parquet_datasets

    # test against str-paths and Path-like paths
    str_mapping = {
        "train": str(path_one),
        "test": str(path_two),
    }
    pathlib_mapping = {
        "train": Path(path_one),
        "test": Path(path_two),
    }

    for data_files in [str_mapping, pathlib_mapping]:
        dataset = load_dataset_from_store(data_files, format="parquet")  # type: ignore[arg-type]
        assert isinstance(dataset, datasets.DatasetDict)

        dataset = load_dataset_from_store(data_files, format="parquet", split=None)  # type: ignore[arg-type]
        assert isinstance(dataset, datasets.DatasetDict)


def test_load_from_mapping_of_files_as_dataset(
    fixture_collection_of_parquet_datasets,
):
    """That can load a mapping of (coherent) files as a single Dataset."""
    path_one, path_two = fixture_collection_of_parquet_datasets
    mapping = {
        "train": path_one,
        "test": path_two,
    }
    dataset = load_dataset_from_store(mapping, format="parquet", split="all")
    assert isinstance(dataset, datasets.Dataset)

    dataset = load_dataset_from_store(mapping, format="parquet", split="train")
    assert isinstance(dataset, datasets.Dataset)

    dataset = load_dataset_from_store(mapping, format="parquet", split="test")
    assert isinstance(dataset, datasets.Dataset)


def test_load_from_dataset_dict_dir(fixture_sample_parquet_dataset_dict):
    """That can load a persisted DatasetDict."""
    path = fixture_sample_parquet_dataset_dict

    dataset_dict = load_dataset_from_store(path, format="parquet", split=None)
    assert isinstance(dataset_dict, datasets.DatasetDict)

    # Test default returns None splits (i.e. a DatasetDict)
    dataset_dict = load_dataset_from_store(path, format="parquet")
    assert isinstance(dataset_dict, datasets.DatasetDict)

    # Test can load a specific split
    dataset = load_dataset_from_store(path, format="parquet", split="train")
    assert isinstance(dataset, datasets.Dataset)

    # Test can load all splits as single Dataset
    dataset = load_dataset_from_store(path, format="parquet", split="all")
    assert isinstance(dataset, datasets.Dataset)


def test_load_from_dataset_dict_dir_defaults_to_assume_parquet(
    fixture_sample_parquet_dataset_dict,
):
    """That loading a DatasetDict defaults to assuming data has been persisted
    as parquet files."""
    path = fixture_sample_parquet_dataset_dict

    dataset_dict = load_dataset_from_store(path, split=None)
    assert isinstance(dataset_dict, datasets.DatasetDict)


def test_load_from_cloud_parquet(
    fixture_mock_s3_client,
    fixture_mock_s3_filesystem,
    fixture_sample_parquet_dataset,
):
    """That can load a dataset saved as parquet from AWS S3."""

    # Upload the dataset to the mock S3 to simulate having it already there
    parquet = fixture_sample_parquet_dataset
    mock_bucket_name = str(uuid.uuid4())
    client = fixture_mock_s3_client
    client.create_bucket(Bucket=mock_bucket_name)
    client.upload_file(
        Filename=parquet.path,
        Bucket=mock_bucket_name,
        Key="data.parquet",
    )

    # Load dataset

    # test both against str-paths and Path-like paths
    str_path = f"s3://{mock_bucket_name}/data.parquet"
    pathlib_path = S3Path(str_path)

    for path in [str_path, pathlib_path]:
        fs = fixture_mock_s3_filesystem
        dataset = load_dataset_from_store(path, format="parquet", fs=fs)  # type: ignore[arg-type]
        assert isinstance(dataset, datasets.Dataset)


@pytest.fixture(scope="module")
def fixture_dataset_saved_to_disk(tmpdir_factory):
    path = str(tmpdir_factory.mktemp("fixture_dataset"))
    dataset = load_dataset("esol")
    fs = fsspec.filesystem("file")
    save_dataset_to_store(dataset, path=path, format="disk", fs=fs)
    return SimpleNamespace(path=path, shape=dataset.shape, columns=dataset.column_names)


def test_load_from_stored_disk_datadir(fixture_dataset_saved_to_disk):
    """That can load a dataset saved as 'disk' as a Dataset."""
    disk = fixture_dataset_saved_to_disk
    dataset = load_dataset_from_store(disk.path, format="disk")
    assert isinstance(dataset, datasets.Dataset)


def test_load_from_cloud_disk_datadir(
    fixture_mock_s3_client,
    fixture_mock_s3_filesystem,
    fixture_dataset_saved_to_disk,
):
    """That can load a dataset saved as 'disk' from AWS S3."""

    # Upload the dataset to the mock S3 to simulate having it already there
    saved_dataset = fixture_dataset_saved_to_disk
    mock_bucket_name = str(uuid.uuid4())
    client = fixture_mock_s3_client
    client.create_bucket(Bucket=mock_bucket_name)
    for f in os.listdir(saved_dataset.path):
        client.upload_file(
            Filename=os.path.join(saved_dataset.path, f),
            Bucket=mock_bucket_name,
            Key=f"data/{f}",
        )

    # test both against str-paths and Path-like paths
    str_path = f"s3://{mock_bucket_name}/data"
    pathlib_path = S3Path(str_path)

    for path in [str_path, pathlib_path]:
        fs = fixture_mock_s3_filesystem
        dataset = load_dataset_from_store(path, format="disk", fs=fs)  # type: ignore[arg-type]
        assert isinstance(dataset, datasets.Dataset)
