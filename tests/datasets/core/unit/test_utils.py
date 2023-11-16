import pytest

from molflux.datasets.utils import is_cloud_data


def test_is_cloud_data_none():
    """That None returns false"""
    assert is_cloud_data(None) is False


def test_is_cloud_data_list():
    """That lists are handled appropriately"""
    file_list = ["s3://test-bucket", "/local_dir/", "temp_file.csv"]

    assert is_cloud_data(file_list)
    assert is_cloud_data(file_list[0])
    assert is_cloud_data(file_list[1]) is False
    assert is_cloud_data(file_list[2]) is False


def test_is_cloud_data_bad_input():
    """That dictionaries with non string inputs raise an error"""
    with pytest.raises(
        ValueError,
        match=r".*Could not resolve if data files are cloud data files.*",
    ):
        is_cloud_data({"dictionary_with_non_str_values": 100})  # type: ignore
