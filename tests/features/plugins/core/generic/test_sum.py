import pytest

from molflux.features import Representation, list_representations, load_representation
from molflux.features.representations.core.generic.sum import Sum

representation_name = "sum"


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
    assert isinstance(representation, Sum)


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


@pytest.mark.parametrize(
    "columns,expected_output_column",
    [
        (
            [
                [0, 1, 2],
                [3, 4, 5],
                [6, 7, 8],
                [0, 1, 2],
            ],
            [
                9,
                13,
                17,
            ],
        ),
        (
            [
                [0],
                [1],
                [2],
            ],
            [
                3,
            ],
        ),
        (
            [
                [],
                [],
                [],
            ],
            [],
        ),
    ],
)
def test_sum_features(fixture_representation, columns, expected_output_column):
    """That works as expected on input columns"""
    representation = fixture_representation

    results = representation.featurise(*columns)

    # the only key in the output should be the representation tag
    assert [fixture_representation.tag] == list(results.keys())

    # check that the expected sums match
    assert expected_output_column == results[fixture_representation.tag]
