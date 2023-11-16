import json
from typing import List

import pytest
from openeye import oechem

from molflux.features import (
    Representation,
    list_representations,
    load_from_dict,
    load_representation,
)
from molflux.features.errors import FeaturisationError
from molflux.features.representations.openeye.sd.sd_feature import (
    DType,
    SDFeature,
)

representation_name = "sd_feature"


@pytest.fixture(scope="module")
def fixture_molecules() -> List[oechem.OEMolBase]:
    """Fixture to obtain three molecules."""
    smiles = ["c1ccccc1", "c1(CCCC)ccccc1", "CC"]
    mols = [oechem.OEGraphMol() for _ in range(len(smiles))]
    for mol, smi in zip(mols, smiles):
        oechem.OESmilesToMol(mol, smi)
    return mols


@pytest.fixture(scope="module")
def fixture_on_bits() -> List[List[int]]:
    return [[3, 6, 8, 2], [9, 5, 3, 1, 8, 7], [0, 4]]


@pytest.fixture(scope="module")
def fixture_dense_bits() -> List[List[int]]:
    return [
        [0, 0, 1, 1, 0, 0, 1, 0, 1, 0],
        [0, 1, 0, 1, 0, 1, 0, 1, 1, 1],
        [1, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    ]


@pytest.fixture(scope="module")
def fixture_molecules_sd_features(
    fixture_molecules: List[oechem.OEMolBase],
    fixture_on_bits: List[List[int]],
    fixture_dense_bits: List[List[int]],
) -> List[oechem.OEMolBase]:
    """Fixture to append SD tags for particular features onto the sampled molecules."""
    molecules_with_sd_tags = []
    for org_mol, on_bits, dense_bits in zip(
        fixture_molecules,
        fixture_on_bits,
        fixture_dense_bits,
    ):
        mol = org_mol.CreateCopy()
        # simulate newline separated 'on' bits.
        oechem.OESetSDData(
            mol,
            "newline-on-bits",
            "\n".join(
                [*map(str, on_bits)],
            ),
        )
        # simulate comma separated 'on' bits.
        oechem.OESetSDData(
            mol,
            "comma-on-bits",
            ",".join([*map(str, on_bits)]),
        )
        # simulate JSON dense list.
        oechem.OESetSDData(
            mol,
            "json-dense-array",
            json.dumps(dense_bits),
        )
        # simulate JSON sparse list.
        oechem.OESetSDData(
            mol,
            "json-sparse-bits",
            json.dumps(on_bits),
        )
        molecules_with_sd_tags.append(mol)
    return molecules_with_sd_tags


def test_representation_in_catalogue():
    """That the representation is registered in the catalogue."""
    catalogue = list_representations()
    all_representation_names = [name for names in catalogue.values() for name in names]
    assert representation_name in all_representation_names


def test_representation_is_mapped_to_correct_class():
    """That the catalogue name is mapped to the appropriate class."""
    representation = load_representation(representation_name)
    assert isinstance(representation, SDFeature)


def test_implements_protocol():
    """That the representation implements the public Representation protocol."""
    representation = load_representation(representation_name)
    assert isinstance(representation, Representation)


@pytest.mark.parametrize(
    "sd_representation",
    [
        {
            "name": "sd_feature",
            "presets": {"sd_tags": ["newline-on-bits"], "as_json": False},
        },
        {"name": "sd_feature", "presets": {"sd_tags": ["json-dense-array"]}},
        {"name": "sd_feature", "presets": {"sd_tags": ["json-sparse-bits"]}},
        {
            "name": "sd_feature",
            "presets": {"sd_tags": ["json-sparse-bits", "json-dense-array"]},
        },
        {
            "name": "sd_feature",
            "presets": {
                "sd_tags": [
                    "json-sparse-bits",
                    "json-dense-array",
                    "newline-on-bits",
                ],
                "as_json": ["json-sparse-bits", "json-dense-array"],
            },
        },
        {
            "name": "sd_feature",
            "presets": {
                "sd_tags": [
                    "json-sparse-bits",
                    "json-dense-array",
                    "newline-on-bits",
                ],
                "as_json": ["json-sparse-bits", "json-dense-array"],
                "dtype": {"newline-on-bits": "float"},
            },
        },
        {
            "name": "sd_feature",
            "presets": {
                "sd_tags": [
                    "comma-on-bits",
                    "newline-on-bits",
                ],
                "as_json": False,
                "dtype": {"newline-on-bits": "float"},
                "separator": {"comma-on-bits": ","},
            },
        },
    ],
    ids=[
        "newline-separated-features",
        "json-dense-array",
        "json-sparse-array",
        "multiple-json",
        "multiple-mix",
        "dtype-cast",
        "different-separators",
    ],
)
def test_featurises_with_different_sd_tag_parsing(
    sd_representation,
    fixture_molecules_sd_features,
):
    """That the featurisation of an SDFeature representation
    works as expected with different sd tags and parsers used."""
    representation = load_from_dict(sd_representation)

    assert "sd_tags" in representation.state
    assert representation.state["sd_tags"]
    # Obtain features.
    features = representation.featurise(fixture_molecules_sd_features)

    # Check all features are present.
    assert {
        f"{representation.tag}::{sd_tag}" for sd_tag in representation.state["sd_tags"]
    } == {*features}

    # Checks for individual feature types.
    if f"{representation.tag}::newline-on-bits" in features:
        assert isinstance(features[f"{representation.tag}::newline-on-bits"][0], list)
        rep_dtype = representation.state["dtype"]
        dtype = (
            rep_dtype.get("newline-on-bits", "str")
            if isinstance(rep_dtype, dict)
            else rep_dtype
        )
        dtype_type = DType[dtype.upper()].value

        assert isinstance(
            features[f"{representation.tag}::newline-on-bits"][0][0],
            dtype_type,
        )
    if f"{representation.tag}::comma-on-bits" in features:
        assert isinstance(features[f"{representation.tag}::comma-on-bits"][0], list)
        rep_dtype = representation.state["dtype"]
        dtype = (
            rep_dtype.get("comma-on-bits", "str")
            if isinstance(rep_dtype, dict)
            else rep_dtype
        )
        dtype_type = DType[dtype.upper()].value
        assert isinstance(
            features[f"{representation.tag}::comma-on-bits"][0][0],
            dtype_type,
        )
    if f"{representation.tag}::json-dense-array" in features:
        assert isinstance(features[f"{representation.tag}::json-dense-array"][0], list)
        assert isinstance(
            features[f"{representation.tag}::json-dense-array"][0][0],
            int,
        )

    if f"{representation.tag}::json-sparse-bits" in features:
        assert isinstance(features[f"{representation.tag}::json-sparse-bits"][0], list)
        assert isinstance(
            features[f"{representation.tag}::json-sparse-bits"][0][0],
            int,
        )


def test_raises_if_sd_tags_not_present(fixture_molecules):
    """That SD feature featurisation fails without SD tags being present."""
    sd_representation = {
        "name": "sd_feature",
        "presets": {"sd_tags": ["json-dense-array"]},
    }
    representation = load_from_dict(sd_representation)
    with pytest.raises(FeaturisationError):
        representation.featurise(fixture_molecules)

    # double check the cause of the error is what it should be
    # (this is an internal consistency check only, not an invariant of the code)
    try:
        representation.featurise(fixture_molecules)
    except FeaturisationError as e:
        assert isinstance(e.__cause__, ValueError)
        assert "Unable to find SD tag" in str(e.__cause__)


def test_failure_if_no_sd_tags_supplied(fixture_molecules_sd_features):
    """Ensures an error is thrown if no SD tags are specified."""
    representation = load_representation(representation_name)
    # No SD tags supplied.
    assert not representation.state.get("sd_tags", None)
    with pytest.raises(ValueError, match="Must supply SD tags .*"):
        representation.featurise(fixture_molecules_sd_features)
