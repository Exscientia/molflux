import pytest

from molflux.features import Representation, list_representations, load_representation
from molflux.features.errors import FeaturisationError
from molflux.features.representations.openeye.canonical.smiles import (
    CanonicalSmiles,
)

representation_name = "canonical_smiles"


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
    assert isinstance(representation, CanonicalSmiles)


def test_implements_protocol(fixture_representation):
    """That the representation implements the public Representation protocol."""
    representation = fixture_representation
    assert isinstance(representation, Representation)


def test_default_compute(fixture_representation):
    """That default scoring gives expected results."""
    representation = fixture_representation
    samples = ["C#C[C@]1(O)CC[C@H]2[C@@H]3CCC4=Cc5oncc5C[C@]4(C)[C@H]3CC[C@@]21C"]
    result = representation.featurise(samples=samples)
    expected_result = [
        "C[C@]12CC[C@H]3[C@H]([C@@H]1CC[C@]2(C#C)O)CCC4=Cc5c(cno5)C[C@]34C",
    ]
    assert representation_name in result
    assert result[representation_name] == expected_result


def test_with_remove_formal_charges_and_reasonable_protomer_raises(
    fixture_representation,
):
    """That an error is raised if attempting to remove formal charges and
    setting reasonable_protomer."""
    representation = fixture_representation
    samples = ["C#C[C@]1(O)CC[C@H]2[C@@H]3CCC4=Cc5oncc5C[C@]4(C)[C@H]3CC[C@@]21C"]

    with pytest.raises(FeaturisationError):
        representation.featurise(
            samples=samples,
            remove_formal_charges=True,
            reasonable_protomer=True,
        )


def test_with_remove_formal_charges_and_set_neutral_ph_raises(fixture_representation):
    """That a ValueError is raised if attempting to remove formal charges and
    setting a neutral pH."""
    representation = fixture_representation
    samples = ["C#C[C@]1(O)CC[C@H]2[C@@H]3CCC4=Cc5oncc5C[C@]4(C)[C@H]3CC[C@@]21C"]

    with pytest.raises(FeaturisationError):
        representation.featurise(
            samples=samples,
            remove_formal_charges=True,
            set_neutral_ph=True,
        )


def test_with_sd_title_tag(fixture_representation):
    """That can set a custom sd_title_tag."""
    representation = fixture_representation
    samples = ["C#C[C@]1(O)CC[C@H]2[C@@H]3CCC4=Cc5oncc5C[C@]4(C)[C@H]3CC[C@@]21C"]

    result = representation.featurise(samples=samples, sd_title_tag="pytest")
    assert result


def test_with_clear_sd_data(fixture_representation):
    """That can set a custom sd_title_tag."""
    representation = fixture_representation
    samples = ["C#C[C@]1(O)CC[C@H]2[C@@H]3CCC4=Cc5oncc5C[C@]4(C)[C@H]3CC[C@@]21C"]

    result = representation.featurise(samples=samples, clear_sd_data=True)
    assert result


@pytest.mark.parametrize(
    ["sample", "expected_result"],
    [
        ("[Na+].CCC(=O)[O-]", "CCC(=O)[O-]"),
        ("[Fr+].C(=O)[O-]", "C(=O)[O-]"),
        ("[Na+].[Cl-]", "[Na+]"),
        ("[Cl-].[Na+]", "[Cl-]"),
        ("O=C(O)C.O=C(O)CC", "CCC(=O)O"),
    ],
)
def test_with_strip_salts(fixture_representation, sample, expected_result):
    """That can strip salts."""
    representation = fixture_representation

    result = representation.featurise(samples=[sample], strip_salts=True)
    assert representation_name in result
    assert result[representation_name] == [expected_result]


@pytest.mark.parametrize(
    ["sample", "expected_result"],
    [
        ("C", "[H]C([H])([H])[H]"),
        ("C(=O)O", "[H]C(=O)O[H]"),
        ("C(=O)[O-]", "[H]C(=O)[O-]"),
    ],
)
def test_with_explicit_h(fixture_representation, sample, expected_result):
    """That can add explicit hydrogens"""
    representation = fixture_representation

    result = representation.featurise(samples=[sample], explicit_h=True)
    assert representation_name in result
    assert result[representation_name] == [expected_result]


def test_with_reasonable_tautomer(fixture_representation):
    """That can set reasonable tautomer."""
    representation = fixture_representation
    samples = ["CC=C(O)C"]

    result = representation.featurise(samples=samples, reasonable_tautomer=True)
    expected_result = ["CCC(=O)C"]
    assert representation_name in result
    assert result[representation_name] == expected_result


def test_with_neutral_ph(fixture_representation):
    """That can set neutral pH"""
    representation = fixture_representation
    samples = ["CN(C)C"]

    result = representation.featurise(samples=samples, set_neutral_ph=True)
    expected_result = ["C[NH+](C)C"]
    assert representation_name in result
    assert result[representation_name] == expected_result


def test_with_reasonable_protomer(fixture_representation):
    """That can get reasonable protomer"""
    representation = fixture_representation
    samples = ["NC(Cc1ccc(O)cc1)C(O)=O"]

    result = representation.featurise(samples=samples, reasonable_protomer=True)
    expected_result = ["c1cc(ccc1CC(C(=O)[O-])[NH3+])O"]
    assert representation_name in result
    assert result[representation_name] == expected_result


@pytest.mark.parametrize(
    ["sample", "expected_result"],
    [
        ("[O-][H]", "O"),
        ("CC(=O)[O-]", "CC(=O)O"),
        ("C#[N-]", "C#N"),
        ("[NH+]", "[N]"),
        ("C[N+](C)(C)C", "C[N+](C)(C)C"),
    ],
)
def test_with_remove_formal_charges(fixture_representation, sample, expected_result):
    """That can remove formal charges"""
    representation = fixture_representation

    result = representation.featurise(samples=[sample], remove_formal_charges=True)
    assert representation_name in result
    assert result[representation_name] == [expected_result]


@pytest.mark.parametrize(
    ["sample", "expected_result"],
    [
        ("c1cc(ccc1C[C@@H](C(=O)O)N)O", "c1cc(ccc1CC(C(=O)O)N)O"),
        ("c1cc(ccc1C[C@H](C(=O)O)N)O", "c1cc(ccc1CC(C(=O)O)N)O"),
        ("F/C=C/F", "C(=CF)F"),
        (r"F/C=C\F", "C(=CF)F"),
    ],
)
def test_with_remove_stereo(fixture_representation, sample, expected_result):
    """That can remove formal charges"""
    representation = fixture_representation

    result = representation.featurise(samples=[sample], remove_stereo=True)
    assert representation_name in result
    assert result[representation_name] == [expected_result]


@pytest.mark.parametrize(
    ["sample", "expected_result"],
    [
        ("[C@@H]", "[CH]"),
        ("[C@H]", "[CH]"),
        ("F/C(F)=C/F", "C(=C(F)F)F"),
        ("F/C(F)=C([C@H])/F", "[CH]C(=C(F)F)F"),
    ],
)
def test_with_clear_non_chiral_stereo(fixture_representation, sample, expected_result):
    """That can remove formal charges"""
    representation = fixture_representation

    result = representation.featurise(samples=[sample], clear_non_chiral_stereo=True)
    assert representation_name in result
    assert result[representation_name] == [expected_result]


def test_with_remove_non_standard_stereo(fixture_representation):
    """That can remove non standard stereo"""
    representation = fixture_representation
    samples = ["CC[N@@H+](C)F"]

    result = representation.featurise(samples=samples, remove_non_standard_stereo=True)
    expected_result = ["CC[NH+](C)F"]
    assert representation_name in result
    assert result[representation_name] == expected_result


def test_with_rekekulise(fixture_representation):
    """That can rekekulise"""
    representation = fixture_representation
    samples = ["c1cnccn1"]

    result = representation.featurise(samples=samples, rekekulise=True)
    expected_result = ["C1=CN=CC=N1"]
    assert representation_name in result
    assert result[representation_name] == expected_result


def test_combine_flags_with_output_flavors(fixture_representation):
    """That can both rekekulize and assign explicit hydrogen atoms"""
    representation = fixture_representation
    samples = ["c1cnccn1"]

    result = representation.featurise(samples=samples, explicit_h=True, rekekulise=True)
    expected_result = ["[H]C1=C(N=C(C(=N1)[H])[H])[H]"]
    assert representation_name in result
    assert result[representation_name] == expected_result


@pytest.mark.parametrize(
    ["sample", "expected_output_sample", "tautomer_options"],
    [
        ("C/C=C(/O)F", "C/C=C(/O)\\F", {"save_stereo": True}),
        ("C/C=C(/O)F", "CCC(=O)F", {}),
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


@pytest.mark.parametrize(
    ["expected_result", "tautomer_timeouts"],
    [
        (
            "C[C@@]12CCC[C@@]([C@H]1CC[C@@]34C2CC([C@@H](C3)C(=C)C4)O[C@@H]5[C@H](C([C@H]([C@@H](O5)CO)O)O[C@@H]6[C@H]([C@@H]([C@H]([C@@H](O6)CO)O)O)O)O[C@@H]7[C@H]([C@@H]([C@H]([C@@H](O7)CO)O)O)O)(C)C(=O)OC8[C@H]([C@@H]([C@H]([C@@H](O8)CO)O)O)O",
            [0.01],
        ),
        (
            "C[C@@]12CCC[C@@]([C@H]1CC[C@@]34C2CC([C@@H](C3)C(=C)C4)O[C@@H]5[C@H](C([C@H]([C@@H](O5)CO)O)O[C@@H]6[C@H]([C@@H]([C@H]([C@@H](O6)CO)O)O)O)O[C@@H]7[C@H]([C@@H]([C@H]([C@@H](O7)CO)O)O)O)(C)C(=O)OC8[C@H]([C@@H]([C@H]([C@@H](O8)CO)O)O)O",
            [10],
        ),
    ],
)
def test_with_tautomer_timeouts(
    fixture_representation,
    expected_result,
    tautomer_timeouts,
):
    """That tautomer timeouts are correctly applied"""
    representation = fixture_representation
    samples = [
        "C=C1C[C@]23CC[C@H]4[C@@](C)(CCC[C@@]4(C)C(=O)OC4O[C@@H](CO)[C@H](O)[C@@H](O)[C@@H]4O)C2CC(O[C@H]2O[C@@H](CO)[C@H](O)C(O[C@H]4O[C@@H](CO)[C@H](O)[C@@H](O)[C@@H]4O)[C@@H]2O[C@H]2O[C@@H](CO)[C@H](O)[C@@H](O)[C@@H]2O)[C@H]1C3",
    ]
    expected_result = [expected_result]

    result = representation.featurise(
        samples=samples,
        reasonable_tautomer=True,
        tautomer_timeouts=tautomer_timeouts,
    )

    assert representation_name in result
    assert result[representation_name] == expected_result
