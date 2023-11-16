import pytest

from molflux.features import Representation, list_representations, load_representation
from molflux.features.representations.openeye.canonical.oemol import (
    CanonicalOemol,
)

representation_name = "canonical_oemol"


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
    assert isinstance(representation, CanonicalOemol)


def test_implements_protocol(fixture_representation):
    """That the representation implements the public Representation protocol."""
    representation = fixture_representation
    assert isinstance(representation, Representation)


def test_default_compute(fixture_representation):
    """That default scoring gives expected results."""
    representation = fixture_representation
    samples = ["C=C"]
    result = representation.featurise(samples=samples)
    expected_result = [
        b"\x0b\xb5\n\x93\x19\x85mol_0\x13\x87\x82CC\x81\x00\x01\x02 \x81\x00\x10"
        b"\x9b\x81\x80?\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
        b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00!\x81\x00",
    ]
    assert representation_name in result
    assert result[representation_name] == expected_result


@pytest.mark.parametrize(
    ["sample", "expected_output_sample", "tautomer_options"],
    [
        (
            "C/C=C(/O)F",
            b'\x0b\xe5\n\x9f\x19\x85mol_0\x13\x93\x85CCCOF\x84\x00\x01\x01\x01\x02"\x02\x03\x01\x02\x04\x01 \x81\x00\x10\xbf\x81\x80?\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00!\x81\x00',
            {"save_stereo": True},
        ),
        (
            "C/C=C(/O)F",
            b"\x0bm\x81\n\xeb\x1f\xb6\x01\xb4\x01?\x00\x8d_stableScore_\x08\x00\x00\x00\x00\x00\x00\x00\xc0\x00\x8f_tautomerScore_\x08\x00\x00\x00\x00\x00@\x8f@\x19\x85mol_0\x13\xa7\x8aCCCOFHHHHH\x89\x00\x01\x01\x01\x02!\x02\x03\x02\x02\x04\x01\x00\x05\x01\x00\x06\x01\x00\x07\x01\x01\x08\x01\x01\t\x01 \x81\x00\x10\xfb\x81\x80?\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00!\x81\x00",
            {},
        ),
    ],
)
def test_with_tautomer_options(
    fixture_representation,
    sample,
    expected_output_sample,
    tautomer_options,
):
    """That tautomer options are correctly applied"""
    representation = fixture_representation
    samples = [sample]
    expected_result = [expected_output_sample]

    result = representation.featurise(
        samples=samples,
        reasonable_tautomer=True,
        tautomer_options=tautomer_options,
    )

    assert representation_name in result
    assert result[representation_name] == expected_result
