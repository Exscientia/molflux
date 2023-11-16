import pytest

from molflux.features import Representation, list_representations, load_representation
from molflux.features.representations.rdkit.fingerprints.mhfp import MHFP

representation_name = "mhfp"


@pytest.fixture(scope="function")
def fixture_representation() -> Representation:
    return load_representation(representation_name)


def test_representation_in_catalogue():
    """That the representation is registered in the catalogue."""
    catalogue = list_representations()
    all_representation_names = [name for names in catalogue.values() for name in names]
    assert representation_name in all_representation_names


def test_representation_is_mapped_to_correct_class(fixture_representation):
    """That the catalogue name is mapped to the appropriate class."""
    representation = fixture_representation
    assert isinstance(representation, MHFP)


def test_implements_protocol(fixture_representation):
    """That the representation implements the public Representation protocol."""
    representation = fixture_representation
    assert isinstance(representation, Representation)


def test_default_fingerprint_length(fixture_representation):
    """That default featurisation gives fingerptins of expected length."""
    representation = fixture_representation
    samples = [
        "Cc1ccc2c(=O)c3cccc(CC(=O)OC4OC(C(=O)O)[C@@H](O)[C@@H](O)[C@@H]4O)c3oc2c1C",
    ]
    result = representation.featurise(samples=samples)
    assert representation_name in result
    fingerprint = result[representation_name][0]
    assert len(fingerprint) == 2048


def test_custom_fingerprint_length(fixture_representation):
    """That can calculate fingerprints of custom lengths."""
    representation = fixture_representation
    samples = [
        "Cc1ccc2c(=O)c3cccc(CC(=O)OC4OC(C(=O)O)[C@@H](O)[C@@H](O)[C@@H]4O)c3oc2c1C",
    ]
    length = 64
    result = representation.featurise(samples=samples, length=length)
    fingerprint = result[representation_name][0]
    assert len(fingerprint) == length


def test_empty_smiles_fingerprint_length(fixture_representation):
    """That empty SMILES strings are processed correctly and give fingerprints of the expected length."""
    representation = fixture_representation
    samples = [""]
    length = 16
    result = representation.featurise(samples=samples, length=length)
    fingerprint = result[representation_name][0]
    assert len(fingerprint) == length


def test_default_compute(fixture_representation):
    """That default scoring gives expected results."""
    representation = fixture_representation
    samples = [
        "Cc1ccc2c(=O)c3cccc(CC(=O)OC4OC(C(=O)O)[C@@H](O)[C@@H](O)[C@@H]4O)c3oc2c1C",
    ]
    result = representation.featurise(samples=samples, length=12)
    expected_result = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    assert "mhfp" in result
    assert result["mhfp"] == [expected_result]
