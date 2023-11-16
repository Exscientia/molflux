import pytest

from molflux.features import Representation, list_representations, load_representation
from molflux.features.representations.core.generic.exploded import Exploded

representation_name = "exploded"


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
    assert isinstance(representation, Exploded)


def test_implements_protocol(fixture_representation):
    """That the representation implements the public Representation protocol."""
    representation = fixture_representation
    assert isinstance(representation, Representation)


def test_default_representation_tag_matches_entrypoint_name(fixture_representation):
    """That the default representation tag matches the entrypoint name.

    This is not strictly required, but ensures a more intuitive user experience.
    """
    representation = fixture_representation
    assert representation.tag == representation_name


def test_enumerate_fingerprints(fixture_representation):
    """That works as expected on fingerprint-like inputs."""
    representation = fixture_representation
    samples = [[1, 0, 0, 0], [1, 1, 0, 0], [1, 1, 1, 0], [1, 1, 1, 1]]
    results = representation.featurise(samples=samples)

    # should return a column with all the first molfluxs, one with all the
    # second molfluxs, and so on...
    expected_results = [[1, 1, 1, 1], [0, 1, 1, 1], [0, 0, 1, 1], [0, 0, 0, 1]]
    for actual, expected in zip(results.values(), expected_results):
        assert actual == expected
