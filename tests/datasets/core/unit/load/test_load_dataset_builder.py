import pytest

import datasets
from molflux.datasets.load import load_dataset_builder

representative_builder_name = "esol"


def test_can_load_builder_from_catalogue():
    """That can load a builder from the catalogue."""
    builder = load_dataset_builder(representative_builder_name)
    assert isinstance(builder, datasets.DatasetBuilder)


# TODO: update once we have a multi-config builder to test this better
def test_can_load_specific_builder_config():
    """That can load a specific config of a given builder from the catalogue."""
    builder = load_dataset_builder(representative_builder_name, config_name="default")
    assert isinstance(builder, datasets.DatasetBuilder)


def test_bad_dataset_builder_name_raises_an_error():
    """That a name not in the catalogue raises an error"""
    with pytest.raises(FileNotFoundError):
        load_dataset_builder("exs_certainly_not_a_dataset")
