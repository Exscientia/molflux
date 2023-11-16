from __future__ import annotations

import pytest
import rdkit

from molflux.features import Representation, list_representations, load_representation
from molflux.features.errors import FeaturisationError
from molflux.features.representations.rdkit.descriptors.rdkit_descriptors_2d import (
    RdkitDescriptors_2d,
    list_available_rdkit_descriptors_2d,
)

representation_name = "rdkit_descriptors_2d"
_EXPECTED_NUM_2D_DESCRIPTORS = 210


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
    assert isinstance(representation, RdkitDescriptors_2d)


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


def test_number_output_features(fixture_representation):
    """That this one-to-many representations generates the expected number
    of features."""
    representation = fixture_representation
    samples = ["CCCC1=NN(C2=C1NC(=NC2=O)C3=C(C=CC(=C3)S(=O)(=O)N4CCN(CC4)C)OCC)C"]
    result = representation.featurise(samples=samples)
    assert len(result) == _EXPECTED_NUM_2D_DESCRIPTORS


def test_list_available_rdkit_descriptors_2d_number_output_features():
    """That the function lists out the expected number of available descriptors."""
    available = list_available_rdkit_descriptors_2d()
    assert len(available) == _EXPECTED_NUM_2D_DESCRIPTORS


def test_list_available_rdkit_descriptors_2d_matches_rdkit_backend():
    """That the function matches the 2D descriptors available in rdkit"""
    available = list_available_rdkit_descriptors_2d()
    source = [descriptor[0] for descriptor in rdkit.Chem.Descriptors._descList]
    # remove duplicates to patch a duplicate SPS descriptor in rdkit.Chem.Descriptors._descList (bug)
    source = list(dict.fromkeys(source))
    assert available == source


def test_include_selects_correct_number_of_computed_features(fixture_representation):
    """That the include kwarg can be used to select a specific number of
    descriptors to be calculated."""
    representation = fixture_representation
    samples = ["CCCC1=NN(C2=C1NC(=NC2=O)C3=C(C=CC(=C3)S(=O)(=O)N4CCN(CC4)C)OCC)C"]
    include = ["AvgIpc", "qed", "Chi0"]
    result = representation.featurise(samples=samples, include=include)
    assert len(result) == len(include)


def test_request_to_include_nonexistent_descriptor_raises(fixture_representation):
    """That an error is raised if requesting an non-existent descriptor."""
    representation = fixture_representation
    samples = ["CCCC1=NN(C2=C1NC(=NC2=O)C3=C(C=CC(=C3)S(=O)(=O)N4CCN(CC4)C)OCC)C"]

    wrong = "WRONG"
    include = ["AvgIpc", wrong, "Chi0"]

    with pytest.raises(
        ValueError,
        match=r".*The following descriptor\(s\) are not available.*",
    ):
        representation.featurise(samples=samples, include=include)


def test_request_to_include_nonexistent_similar_descriptor_raises_with_helpful_message(
    fixture_representation,
):
    """That an informative error message is returned if requesting slightly incorrect descriptors."""
    representation = fixture_representation
    samples = ["CCCC1=NN(C2=C1NC(=NC2=O)C3=C(C=CC(=C3)S(=O)(=O)N4CCN(CC4)C)OCC)C"]

    typo = "qeds"  # instead of qed
    include = ["AvgIpc", typo, "Chi0"]
    with pytest.raises(ValueError, match=r".*You might be looking for one of these.*"):
        representation.featurise(samples=samples, include=include)


def test_exclude_deselects_correct_number_of_computed_features(fixture_representation):
    """That the exclude kwarg can be used to deselect a specific number of
    descriptors to be calculated."""
    representation = fixture_representation
    samples = ["CCCC1=NN(C2=C1NC(=NC2=O)C3=C(C=CC(=C3)S(=O)(=O)N4CCN(CC4)C)OCC)C"]
    exclude = ["AvgIpc", "qed", "Chi0"]
    result = representation.featurise(samples=samples, exclude=exclude)
    assert len(result) == _EXPECTED_NUM_2D_DESCRIPTORS - len(exclude)


def test_no_descriptors_to_calculate_raises(fixture_representation):
    """That an error is raised if no descriptors are left to be calculated.

    This could be because of explicitly not selecting any descriptors, or by
    providing 'select' and 'exclude' arrays that complete filtering out of
    any descriptor.
    """
    representation = fixture_representation
    samples = ["CCCC1=NN(C2=C1NC(=NC2=O)C3=C(C=CC(=C3)S(=O)(=O)N4CCN(CC4)C)OCC)C"]

    include = ["AvgIpc", "qed", "Chi0"]
    exclude = include

    with pytest.raises(ValueError, match=r".*No descriptors to calculate.*"):
        representation.featurise(samples=samples, include=include, exclude=exclude)


def test_default_compute(fixture_representation):
    """That batch scoring gives expected results."""
    representation = fixture_representation
    samples = ["COc1cc2c(cc1OCCCN3CCOCC3)c(ncn2)Sc4nccs4", "c1ccccc1"]

    include = ["MaxEStateIndex", "qed", "MolWt"]
    result = representation.featurise(samples=samples, include=include)

    expected = {
        "rdkit_descriptors_2d::MaxEStateIndex": [6.0566621787603925, 2.0],
        "rdkit_descriptors_2d::qed": [0.407489782854996, 0.4426283718993647],
        "rdkit_descriptors_2d::MolWt": [418.544, 78.11399999999999],
    }

    # we reserve the right to change key naming conventions in the future
    assert list(result.values()) == list(expected.values())


def test_default_compute_one(fixture_representation):
    """That scoring on single input gives expected results."""
    representation = fixture_representation
    samples = ["COc1cc2c(cc1OCCCN3CCOCC3)c(ncn2)Sc4nccs4"]

    include = ["MaxEStateIndex", "qed", "MolWt"]
    result = representation.featurise(samples=samples, include=include)

    expected = {
        "rdkit_descriptors_2d::MaxEStateIndex": [6.0566621787603925],
        "rdkit_descriptors_2d::qed": [0.407489782854996],
        "rdkit_descriptors_2d::MolWt": [418.544],
    }

    # we reserve the right to change key naming conventions in the future
    assert list(result.values()) == list(expected.values())


def test_default_compute_zero(fixture_representation):
    """That scoring on empty inputs is possible and results in a no-op."""
    representation = fixture_representation
    samples: list[float] = []

    include = ["MaxEStateIndex", "qed", "MolWt"]
    result = representation.featurise(samples=samples, include=include)

    expected: dict[str, list[float]] = {
        "rdkit_descriptors_2d::MaxEStateIndex": [],
        "rdkit_descriptors_2d::qed": [],
        "rdkit_descriptors_2d::MolWt": [],
    }

    # we reserve the right to change key naming conventions in the future
    assert list(result.values()) == list(expected.values())


def test_ordered_results(fixture_representation):
    """That output features align with the order of requested descriptors."""
    representation = fixture_representation
    samples = ["COc1cc2c(cc1OCCCN3CCOCC3)c(ncn2)Sc4nccs4", "c1ccccc1"]

    include_a = ["MaxEStateIndex", "qed", "MolWt"]
    result_a = representation.featurise(samples=samples, include=include_a)

    include_b = include_a[::-1]
    result_b = representation.featurise(samples=samples, include=include_b)

    assert list(result_a.values()) == list(result_b.values())[::-1]
    assert list(result_a.keys()) == list(result_b.keys())[::-1]


@pytest.mark.parametrize(
    ["samples", "index_problematic"],
    [(["CC", "Cc1c(c2ncc(c(n2n1)C(=O)NC3CC(C3)(F)F)N$C)C(=O)N"], 1)],  # MLOPS-991
)
def test_raises_featurisation_error_on_invalid_input_samples(
    fixture_representation,
    samples,
    index_problematic,
):
    """That a FeaturisationError is raised when trying to featurise 'invalid' input samples.

    This ensures that a standardised exception is raised by all representations on internal runtime errors.
    """
    representation = fixture_representation
    with pytest.raises(FeaturisationError) as excinfo:
        representation.featurise(samples=samples)

        # check that a reference to the problematic sample is present in the error message
        assert samples[index_problematic] in str(excinfo.value)
