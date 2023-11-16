import pytest

from molflux.features import Representation, list_representations, load_representation
from molflux.features.representations.rdkit.fingerprints.avalon import Avalon

representation_name = "avalon"


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
    assert isinstance(representation, Avalon)


def test_implements_protocol(fixture_representation):
    """That the representation implements the public Representation protocol."""
    representation = fixture_representation
    assert isinstance(representation, Representation)


def test_default_fingerprint_length(fixture_representation):
    """That default featurisation gives fingerptins of expected length."""
    representation = fixture_representation
    samples = [
        "Cc1ccc2c(=O)c3cccc(c3oc2c1C)CC(=O)OC4[C@H]([C@@H]([C@@H](C(O4)C(=O)O)O)O)O",
    ]
    result = representation.featurise(samples=samples)
    assert representation_name in result
    fingerprint = result[representation_name][0]
    assert len(fingerprint) == 512


def test_custom_fingerprint_length(fixture_representation):
    """That can calculate fingerprints of custom lengths."""
    representation = fixture_representation
    samples = [
        "Cc1ccc2c(=O)c3cccc(c3oc2c1C)CC(=O)OC4[C@H]([C@@H]([C@@H](C(O4)C(=O)O)O)O)O",
    ]
    length = 64
    result = representation.featurise(samples=samples, n_bits=length)
    fingerprint = result[representation_name][0]
    assert len(fingerprint) == length


def test_empty_smiles_fingerprint_length(fixture_representation):
    """That empty SMILES strings are processed correctly and give fingerprints of the expected length."""
    representation = fixture_representation
    samples = [""]
    length = 16
    result = representation.featurise(samples=samples, n_bits=length)
    fingerprint = result[representation_name][0]
    assert len(fingerprint) == length


def test_basic_compute(fixture_representation):
    """That default (shortened) featurisation gives expected results."""
    representation = fixture_representation
    samples = [
        "Cc1ccc2c(=O)c3cccc(c3oc2c1C)CC(=O)OC4[C@H]([C@@H]([C@@H](C(O4)C(=O)O)O)O)O",
    ]
    result = representation.featurise(samples=samples, n_bits=16)
    expected_result = [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]
    assert representation_name in result
    assert result[representation_name] == expected_result


def test_batch_compute(fixture_representation):
    """That batch scoring gives expected results."""
    representation = fixture_representation
    samples = [
        "Cc1ccc2c(=O)c3cccc(c3oc2c1C)CC(=O)OC4[C@H]([C@@H]([C@@H](C(O4)C(=O)O)O)O)O",
        "CC",
    ]
    result = representation.featurise(samples=samples, n_bits=16)
    expected_result = [
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ]
    assert representation_name in result
    assert result[representation_name] == expected_result
