from __future__ import annotations

import pytest

from molflux.features import Representation, list_representations, load_representation
from molflux.features.representations.rdkit.fingerprints.toxicophores import (
    Toxicophores,
)

representation_name = "toxicophores"


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
    assert isinstance(representation, Toxicophores)


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


def test_compute_one(fixture_representation):
    """That single scoring gives expected results."""
    representation = fixture_representation
    samples = ["C1NC1"]
    result = representation.featurise(samples=samples)
    expected_result = [
        [
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ],
    ]
    assert result[representation.tag] == expected_result


def test_compute_batch(fixture_representation):
    """That batch scoring gives expected results."""
    representation = fixture_representation
    samples = ["C1NC1", "CC(=CCCC(=CC=O)C)C"]
    result = representation.featurise(samples=samples)
    expected_result = [
        [
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ],
        [
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ],
    ]
    assert result[representation.tag] == expected_result


def test_compute_zero(fixture_representation):
    """That empty scoring gives expected results.

    This should be allowed, and not fail.
    """
    representation = fixture_representation
    samples: list = []
    result = representation.featurise(samples=samples)
    expected_result: list = []
    assert result[representation.tag] == expected_result
