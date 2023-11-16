import pytest

import datasets
from molflux.datasets.load import load_from_dict

representative_dataset_name = "esol"


def test_returns_dataset():
    """That loading from a dict returns a Dataset."""
    name = representative_dataset_name
    config = {
        "name": name,
        "config": {},
    }
    dataset = load_from_dict(config)
    assert isinstance(dataset, datasets.Dataset)


def test_none_split_returns_dataset_dict():
    """That loading from a dict can also return a DatasetDict if split None."""
    name = representative_dataset_name
    config = {
        "name": name,
        "config": {
            "split": None,
        },
    }
    dataset = load_from_dict(config)
    assert isinstance(dataset, datasets.DatasetDict)


def test_from_minimal_dict():
    """That can provide a config with only required fields."""
    name = representative_dataset_name
    config = {
        "name": name,
    }
    assert load_from_dict(config)


def test_dict_missing_required_fields_raises():
    """That cannot load a dataset with a config missing required fields."""
    config = {"unknown_key": "value"}
    with pytest.raises(SyntaxError):
        load_from_dict(config)
