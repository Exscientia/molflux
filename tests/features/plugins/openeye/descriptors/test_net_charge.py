import pytest

from molflux.features import Representation, list_representations, load_representation
from molflux.features.representations.openeye.descriptors.net_charge import (
    NetCharge,
)

representation_name = "net_charge"


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
    assert isinstance(representation, NetCharge)


def test_implements_protocol(fixture_representation):
    """That the representation implements the public Representation protocol."""
    representation = fixture_representation
    assert isinstance(representation, Representation)


def test_default_compute(fixture_representation):
    """That default scoring gives expected results."""
    representation = fixture_representation
    samples = ["[OH3+]"]
    result = representation.featurise(samples=samples)
    expected_result = [1]
    assert representation_name in result
    assert result[representation_name] == expected_result


def test_batch_compute(fixture_representation):
    """That batch scoring gives expected results."""
    representation = fixture_representation
    samples = ["[OH3+]", "CC"]
    result = representation.featurise(samples=samples)
    expected_result = [1, 0]
    assert representation_name in result
    assert result[representation_name] == expected_result


def test_default_digitise(fixture_representation):
    """Test default digitised featurisation gives expected results."""
    representation = fixture_representation
    samples = ["[OH3+]"]
    result = representation.featurise(samples=samples, digitise=True)
    expected_result = [[0, 0, 0, 1, 0]]
    assert representation_name in result
    assert result[representation_name] == expected_result
