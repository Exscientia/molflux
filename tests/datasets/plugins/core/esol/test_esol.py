from unittest.mock import patch

import pytest

import datasets
from molflux.datasets import load_dataset
from molflux.datasets.builders.esol.esol import ESOL
from molflux.datasets.catalogue import list_datasets

dataset_name = "esol"


@pytest.fixture()
def _fixture_import_main_class_patch():
    """Mocks out the import_main_class from datasets'.

    This function is used when generating a dataset builder in the `datasets` class.
    It is done by copying the file path of a builder script into a temporary folder
    under the huggingface cache directory, and loading the Builder class dynamically
    at runtime. This means that any test for a builder will instead be tested on a
    copy of the builder script, not showing up as covered in any coverage run.
    Instead, we redirect the builder class import to the actual class in alexandria.
    """
    with patch("datasets.load.import_main_class") as mock:
        mock.return_value = ESOL
        yield


pytestmark = pytest.mark.usefixtures("_fixture_import_main_class_patch")


def test_in_catalogue() -> None:
    """That the builder is registered in the catalogue."""
    catalogue = list_datasets()
    all_names = [name for names in catalogue.values() for name in names]
    assert dataset_name in all_names


def test_returns_huggingface_dataset():
    """That the built dataset is a huggingface dataset."""
    dataset = load_dataset(dataset_name, split="train")
    assert isinstance(dataset, datasets.Dataset)


def test_default_config_is_available():
    """That the dataset can be loaded without specifying an explicit config."""
    assert load_dataset(dataset_name)


def test_defines_expected_feature_names():
    """That the dataset defines these feature names.

    This test makes sure that we can catch possibly breaking changes to consumers
    if we rename these features down the line.
    """
    expected_feature_names = [
        "smiles",
        "log_solubility",
    ]
    dataset = load_dataset(dataset_name)

    feature_names = dataset.column_names
    assert feature_names is not None
    assert all(name in feature_names for name in expected_feature_names)
    assert len(feature_names) == len(expected_feature_names)
