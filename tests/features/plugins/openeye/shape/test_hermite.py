import pytest

from molflux.features import Representation, list_representations, load_representation
from molflux.features.representations.openeye.shape.hermite import (
    Hermite,
)

representation_name = "hermite"


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
    assert isinstance(representation, Hermite)


def test_implements_protocol(fixture_representation):
    """That the representation implements the public Representation protocol."""
    representation = fixture_representation
    assert isinstance(representation, Representation)


def test_default_compute(fixture_representation):
    """That default scoring gives expected results."""
    representation = fixture_representation
    samples = [
        "CCCC1=NN(C2=C1NC(=NC2=O)C3=C(C=CC(=C3)S(=O)(=O)N4CCN(CC4)C)OCC)C",
    ]
    result = representation.featurise(samples=samples, n_poly_max=2)
    expected_result = [
        [
            89.11933047322135,
            0.0,
            0.003222152914168836,
            0.0,
            0.0,
            0.0032220784927158943,
            0.0,
            0.0,
            0.0,
            0.003221965510728286,
        ],
    ]
    assert representation_name in result
    assert result[representation_name][0] == pytest.approx(
        expected_result[0],
        abs=0.0005,
    )


def test_batch_compute(fixture_representation):
    """That batch scoring gives expected results."""
    representation = fixture_representation
    samples = [
        "CCCC1=NN(C2=C1NC(=NC2=O)C3=C(C=CC(=C3)S(=O)(=O)N4CCN(CC4)C)OCC)C",
        "CC",
    ]
    result = representation.featurise(samples=samples, n_poly_max=2)
    expected_results = [
        [
            89.11933047322135,
            0.0,
            0.003222152914168836,
            0.0,
            0.0,
            0.0032220784927158943,
            0.0,
            0.0,
            0.0,
            0.003221965510728286,
        ],
        [
            5.3907372015426205,
            0.0,
            -0.0015415925639253993,
            0.0,
            0.0,
            -0.0015415925424404435,
            0.0,
            0.0,
            0.0,
            -0.0015415925330217136,
        ],
    ]
    assert representation_name in result
    for j, exp_res in enumerate(expected_results):
        assert result[representation_name][j] == pytest.approx(exp_res, abs=0.0005)
