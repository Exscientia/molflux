from __future__ import annotations

import pytest

from molflux.features import Representation, list_representations, load_representation
from molflux.features.representations.rdkit.fingerprints.graphscan import Graphscan

representation_name = "graphscan"


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
    assert isinstance(representation, Graphscan)


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


def test_compute_returns_correct_number_of_features(fixture_representation):
    """That generates the correct number of features"""
    representation = fixture_representation
    samples = [
        "Cc1ccc2c(=O)c3cccc(c3oc2c1C)CC(=O)OC4[C@H]([C@@H]([C@@H](C(O4)C(=O)O)O)O)O",
    ]
    min_dist = 1
    max_dist = 3
    result = representation.featurise(
        samples=samples,
        min_dist=min_dist,
        max_dist=max_dist,
    )

    num_expected_features = max_dist - min_dist + 1
    assert len(result) == num_expected_features


def test_compute_returns_fingerprints_of_expected_length(fixture_representation):
    """That generates fingerprints of the expected (fixed) length."""
    representation = fixture_representation
    samples = [
        "Cc1ccc2c(=O)c3cccc(c3oc2c1C)CC(=O)OC4[C@H]([C@@H]([C@@H](C(O4)C(=O)O)O)O)O",
    ]
    result = representation.featurise(samples=samples)

    # pairwise combinations with replacement of the 6 pharmacophores
    _NUM_PHARMACOPHORES_COMBINATIONS = 21
    expected_fingerprint_length = _NUM_PHARMACOPHORES_COMBINATIONS

    for features in result.values():
        for fingerprint in features:
            assert len(fingerprint) == expected_fingerprint_length


def test_compute_one(fixture_representation):
    """That single scoring gives expected results."""
    representation = fixture_representation
    samples = [
        "Cc1ccc2c(=O)c3cccc(c3oc2c1C)CC(=O)OC4[C@H]([C@@H]([C@@H](C(O4)C(=O)O)O)O)O",
    ]
    min_dist = 1
    max_dist = 2
    result = representation.featurise(
        samples=samples,
        min_dist=min_dist,
        max_dist=max_dist,
    )
    expected_results = [
        [
            [0, 0, 0, 0, 0, 0, 16, 0, 15, 0, 0, 0, 0, 0, 0, 13, 0, 0, 2, 0, 0],
        ],
        [
            [3, 1, 1, 1, 2, 0, 38, 0, 35, 0, 0, 0, 0, 0, 0, 29, 0, 0, 3, 0, 0],
        ],
    ]
    for actual, expected in zip(result.values(), expected_results):
        assert actual == expected


def test_compute_batch(fixture_representation):
    """That batch scoring gives expected results."""
    representation = fixture_representation
    samples = [
        "Cc1ccc2c(=O)c3cccc(c3oc2c1C)CC(=O)OC4[C@H]([C@@H]([C@@H](C(O4)C(=O)O)O)O)O",
        "C1=CC(=CC=C1C[C@@H](C(=O)O)N)N(CCCl)CCCl",
    ]
    result = representation.featurise(samples=samples)
    expected_results = [
        [
            [0, 0, 0, 0, 0, 0, 16, 0, 15, 0, 0, 0, 0, 0, 0, 13, 0, 0, 2, 0, 0],
            [0, 0, 0, 0, 0, 0, 6, 0, 7, 0, 0, 0, 0, 0, 0, 7, 0, 0, 2, 0, 0],
        ],
        [
            [3, 1, 1, 1, 2, 0, 38, 0, 35, 0, 0, 0, 0, 0, 0, 29, 0, 0, 3, 0, 0],
            [1, 0, 1, 0, 1, 0, 12, 0, 15, 0, 0, 0, 0, 0, 0, 15, 1, 1, 3, 1, 0],
        ],
        [
            [10, 4, 7, 4, 4, 0, 59, 0, 53, 0, 0, 2, 0, 0, 0, 43, 0, 0, 3, 0, 0],
            [1, 0, 3, 0, 1, 2, 15, 1, 22, 1, 1, 1, 0, 0, 1, 22, 4, 2, 3, 3, 0],
        ],
        [
            [16, 8, 12, 7, 5, 0, 73, 0, 67, 0, 0, 4, 0, 0, 0, 55, 0, 0, 3, 0, 0],
            [1, 0, 3, 0, 1, 2, 15, 4, 27, 5, 3, 1, 0, 0, 1, 28, 8, 4, 3, 3, 0],
        ],
        [
            [22, 9, 17, 9, 7, 0, 83, 0, 79, 0, 0, 5, 0, 0, 0, 67, 0, 0, 3, 0, 0],
            [1, 0, 3, 0, 1, 2, 15, 8, 31, 11, 5, 1, 0, 0, 1, 32, 14, 6, 3, 3, 0],
        ],
        [
            [25, 9, 20, 9, 8, 0, 89, 1, 87, 0, 0, 6, 0, 0, 0, 75, 1, 0, 3, 0, 0],
            [1, 0, 3, 0, 1, 2, 15, 11, 33, 16, 6, 1, 0, 0, 1, 34, 19, 7, 3, 3, 0],
        ],
    ]
    for actual, expected in zip(result.values(), expected_results):
        assert actual == expected


def test_compute_zero(fixture_representation):
    """That empty scoring gives expected results.

    This should be allowed, and not fail.
    """
    representation = fixture_representation
    samples: list = []
    result = representation.featurise(samples=samples)
    expected_results: list[list] = [
        [],
        [],
        [],
        [],
        [],
        [],
    ]
    for actual, expected in zip(result.values(), expected_results):
        assert actual == expected


def test_raises_on_incompatible_distances(fixture_representation):
    """That an error is raised if requesting incompatible distances."""
    representation = fixture_representation
    samples = [
        "Cc1ccc2c(=O)c3cccc(c3oc2c1C)CC(=O)OC4[C@H]([C@@H]([C@@H](C(O4)C(=O)O)O)O)O",
    ]
    with pytest.raises(ValueError, match="must have min_dist <= max_dist"):
        representation.featurise(samples=samples, max_dist=0)

    with pytest.raises(ValueError, match="must have min_dist <= max_dist"):
        representation.featurise(samples=samples, min_dist=3, max_dist=2)
