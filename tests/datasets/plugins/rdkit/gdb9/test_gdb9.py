from unittest.mock import patch

import pytest
from rdkit import Chem

import datasets
from molflux.datasets import load_dataset
from molflux.datasets.builders.gdb9.gdb9 import GDB9
from molflux.datasets.catalogue import list_datasets

dataset_name = "gdb9"
backend_name = "rdkit"


@pytest.fixture()
def _fixture_import_main_class_patch():
    """Mocks out the import_main_class from datasets'.

    This function is used when generating a dataset builder in the `datasets` class.
    It is done by copying the file path of a builder script into a temporary folder
    under the huggingface cache directory, and loading the Builder class dynamically
    at runtime. This means that any test for a builder will instead be tested on a
    copy of the builder script, not showing up as covered in any coverage run.
    Instead, we redirect the builder class import to the actual class in datasets.
    """
    with patch("datasets.load.import_main_class") as mock:
        mock.return_value = GDB9
        yield


pytestmark = pytest.mark.usefixtures("_fixture_import_main_class_patch")


@pytest.fixture(scope="module")
def fixture_path_to_mock_data(fixture_path_to_this_test):
    """The path to the directory containing mock data for this builder."""
    here = fixture_path_to_this_test
    return here / "mock_data"


@pytest.fixture()
def _fixture_mocked_dataset_asset(fixture_path_to_mock_data):
    """Mocks out the DownloadManager used in 'datasets.load_dataset()'.

    The DownloadManager is used in the builder's '_split_generator()`, which
    is the function that actually downloads all the data for a given dataset.
    In this way we can avoid having to download the real datasets from AWS S3,
    and return paths to local assets.
    """
    with patch("datasets.builder.DownloadManager.download_and_extract") as mock:
        mock.return_value = str(fixture_path_to_mock_data)
        yield


def test_in_catalogue():
    """That the builder is registered in the catalogue."""
    catalogue = list_datasets()
    all_names = [name for names in catalogue.values() for name in names]
    assert dataset_name in all_names


@pytest.mark.usefixtures("_fixture_mocked_dataset_asset")
def test_returns_huggingface_dataset():
    """That the built dataset is a huggingface dataset."""
    dataset = load_dataset(dataset_name, backend_name, split="train")
    assert isinstance(dataset, datasets.Dataset)


@pytest.mark.usefixtures("_fixture_mocked_dataset_asset")
def test_dataset_has_correct_num_rows():
    """That the built dataset has correct num rows."""

    dataset = load_dataset(dataset_name, backend_name)

    assert len(dataset) == 10


@pytest.mark.usefixtures("_fixture_mocked_dataset_asset")
def test_dataset_is_readable_with_rdkit():
    """That rdkit can read the mol bytes"""

    dataset = load_dataset(dataset_name, backend_name)

    for mol_bytes in dataset["mol_bytes"]:
        Chem.Mol(mol_bytes)
