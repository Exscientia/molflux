import pytest

import datasets
from molflux.datasets.load import load_dataset


def test_implicit_split_returns_single_dataset():
    """That using the default split value returns a single dataset."""
    dataset = load_dataset("esol")
    assert isinstance(dataset, datasets.Dataset)


def test_explicit_split_returns_dataset():
    """That using an explicit split value returns a single dataset."""
    dataset = load_dataset("esol", split="train")
    assert isinstance(dataset, datasets.Dataset)


def test_none_split_returns_dataset_dict():
    """That requesting a None split returns a dataset dict of all splits."""
    dataset = load_dataset("esol", split=None)
    assert isinstance(dataset, datasets.DatasetDict)


def test_bad_dataset_name_raises_an_error():
    """That a name not in the catalogue or online raises an error"""
    with pytest.raises(FileNotFoundError):
        load_dataset("exs_certainly_not_a_dataset")
