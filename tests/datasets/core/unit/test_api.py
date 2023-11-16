"""
Tests ensuring desired API objects are part of the top-level namespace.
"""
import pytest

import molflux.datasets


@pytest.mark.parametrize(
    "callable_name",
    [
        "fill_catalogue",
        "list_datasets",
        "load_dataset_from_store",
        "save_dataset_to_store",
        "featurise_dataset",
        "load_dataset",
        "load_dataset_builder",
        "load_from_dict",
        "load_from_yaml",
        "split_dataset",
    ],
)
def test_exports_callable(callable_name):
    """That the package exposes the given callable."""
    assert hasattr(molflux.datasets, callable_name)
    assert callable(getattr(molflux.datasets, callable_name))
