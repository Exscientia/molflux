import pytest

from datasets.builder import DatasetBuilder
from molflux.datasets.catalogue import (
    get_builder_entrypoint,
    list_datasets,
    register_builder,
)


def test_catalogue_is_filled():
    """That the catalogue is not empty."""
    catalogue = list_datasets()
    assert len(catalogue)


def test_register_builder():
    """That a custom class is registered. TODO make this temporary"""
    kind = "test"
    name = "custom_name"

    @register_builder(kind=kind, name=name)
    class MyClass(DatasetBuilder):
        pass

    catalogue = list_datasets()
    assert kind in catalogue
    assert catalogue[kind] == [name]


def test_get_builder_entrypoint():
    with pytest.raises(NotImplementedError):
        get_builder_entrypoint("exs_endpoint_not_available")
