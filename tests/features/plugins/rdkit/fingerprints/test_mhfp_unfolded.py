import pytest

from molflux.features import Representation, list_representations, load_representation
from molflux.features.representations.rdkit.fingerprints.mhfp_unfolded import (
    MHFPUnfolded,
)

representation_name = "mhfp_unfolded"


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
    assert isinstance(representation, MHFPUnfolded)


def test_implements_protocol(fixture_representation):
    """That the representation implements the public Representation protocol."""
    representation = fixture_representation
    assert isinstance(representation, Representation)


def test_default_compute(fixture_representation):
    """That default scoring gives expected results.

    This test is based on the original mhfp package test suite.
    """
    representation = fixture_representation
    samples = [
        "Cc1ccc2c(=O)c3cccc(CC(=O)OC4OC(C(=O)O)[C@@H](O)[C@@H](O)[C@@H]4O)c3oc2c1C",
    ]
    result = representation.featurise(samples=samples, n_permutations=32, seed=42)
    expected_result = [
        4020624,
        9980344,
        63689098,
        22015504,
        26474917,
        18400837,
        47464384,
        217139116,
        8836027,
        41227994,
        3838604,
        13228201,
        46316417,
        3763579,
        38514427,
        37657111,
        55411077,
        223018370,
        43121840,
        13101872,
        39382703,
        31834049,
        11797863,
        158268734,
        12619637,
        17796484,
        5277139,
        41610747,
        106667704,
        17532210,
        52062080,
        147129299,
    ]
    assert "mhfp_unfolded" in result
    assert result["mhfp_unfolded"] == [expected_result]
