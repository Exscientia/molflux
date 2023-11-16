import pytest

from molflux.features import load_representation


@pytest.fixture(scope="module")
def fixture_mock_representation():
    return load_representation(name="character_count")


def test_on_array_returns_correct_dimensions(fixture_mock_representation):
    """That array featurisation returns an output of the same shape."""
    representation = fixture_mock_representation
    data = ["cccc", "cc"]
    representation_result = representation.featurise(data)
    featurised_data = representation_result.get(representation.tag)
    assert len(featurised_data) == len(data)


def test_on_single_sample_returns_correct_dimensions(fixture_mock_representation):
    """That featurisation of a single sample returns a single result.

    This is to avoid bugs where single sample iterable is iterated over.
    """
    representation = fixture_mock_representation
    data = "cccc"
    representation_result = representation.featurise(data)
    featurised_data = representation_result.get(representation.tag)
    assert len(featurised_data) == 1


def test_can_featurise_empty_data(fixture_mock_representation):
    """That can featurise an empty array without problems."""
    representation = fixture_mock_representation
    assert representation.featurise([])


def test_invalid_kwarg_raises_value_error(fixture_mock_representation):
    """That attempting featurisation with invalid keyword arguments raises."""
    representation = fixture_mock_representation
    data = ["cccc", "cc"]
    with pytest.raises(ValueError, match=r"Unknown featurisation parameter\(s\)"):
        representation.featurise(data, invalid_kwarg=True)
