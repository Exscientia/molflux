import pytest
from openeye import oechem

from molflux.features.representations.openeye._utils import (
    get_sd_data,
    iterate_mols_or_confs,
    to_oemol,
    to_smiles,
)


def test_smiles_to_smiles():
    """That can convert a SMILES string to_smiles is invariant."""
    molecule = (
        "Cc1ccc2c(=O)c3cccc(CC(=O)OC4OC(C(=O)O)[C@@H](O)[C@@H](O)[C@@H]4O)c3oc2c1C"
    )
    out = to_smiles(molecule)
    assert isinstance(out, str)
    assert out == molecule


def test_oemol_to_smiles():
    """That can convert a OEMol object to SMILES."""
    expected_smiles = (
        "Cc1ccc2c(=O)c3cccc(c3oc2c1C)CC(=O)OC4[C@H]([C@@H]([C@@H](C(O4)C(=O)O)O)O)O"
    )
    oemol = oechem.OEMol()
    oechem.OESmilesToMol(oemol, expected_smiles)

    out = to_smiles(oemol)
    assert isinstance(out, str)
    assert out == expected_smiles


def test_bytes_to_smiles():
    """That can convert a bytes string to SMILES"""
    # Acetic Acid
    bytestring = b"\x0b\xce\n\x94\x13\x8f\x84CCOO\x83\x00\x01\x01\x01\x02\x02\x01\x03\x01 \x81\x00\x10\xb3\x81\x80?\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00!\x81\x00"

    out = to_smiles(bytestring)
    assert isinstance(out, str)
    assert out == "CC(=O)O"


def test_hex_to_smiles():
    """That can convert a hex string to SMILES"""
    # Acetic Acid
    hexstring = "0bce0a94138f8443434f4f8300010101020201030120810010b381803f000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000218100"

    out = to_smiles(hexstring)
    assert isinstance(out, str)
    assert out == "CC(=O)O"


def test_smiles_to_oemol():
    """That can convert a SMILES string to OEMol object."""
    molecule = "CC(=O)O"

    out = to_oemol(molecule)
    assert isinstance(out, oechem.OEMol)

    out_as_smiles = to_smiles(out)
    assert out_as_smiles == molecule


def test_oemol_to_oemol():
    """That converting a OEMol object to_oemol is invariant."""
    expected_smiles = "CC(=O)O"
    oemol = oechem.OEMol()
    oechem.OESmilesToMol(oemol, expected_smiles)

    out = to_oemol(oemol)
    assert isinstance(out, oechem.OEMol)

    out_as_smiles = to_smiles(expected_smiles)
    assert out_as_smiles == expected_smiles


def test_bytes_to_oemol():
    """That can convert a bytes string to OEMol object"""
    # Acetic Acid
    bytestring = b"\x0b\xce\n\x94\x13\x8f\x84CCOO\x83\x00\x01\x01\x01\x02\x02\x01\x03\x01 \x81\x00\x10\xb3\x81\x80?\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00!\x81\x00"

    out = to_oemol(bytestring)
    assert isinstance(out, oechem.OEMol)

    out_as_smiles = to_smiles(out)
    assert out_as_smiles == "CC(=O)O"


@pytest.mark.parametrize(
    ("sd_data", "conformer_sd_data", "dtype", "expected", "sd_tag", "conformer"),
    [
        (
            {"tag": "10"},
            {},
            str,
            "10",
            "tag",
            True,
        ),
        (
            {},
            {"tag": "10"},
            str,
            "10",
            "tag",
            True,
        ),
        (
            {"tag": "10"},
            {"tag": "20"},
            str,
            "20",
            "tag",
            True,
        ),
        (
            {"tag": "10"},
            {"tag": "20"},
            str,
            "10",
            "tag",
            False,
        ),
        (
            {"tag": "10"},
            {"tag": "20"},
            float,
            10.0,
            "tag",
            False,
        ),
        (
            {},
            {},
            float,
            Exception,
            "tag",
            False,
        ),
    ],
    ids=[
        "data-on-mol-not-conf",
        "data-on-conf-not-mol",
        "data-on-both-conf-supplied",
        "data-on-both-mol-supplied",
        "dtype-cast-float",
        "error-no-data",
    ],
)
def test_get_sd_data_multi_conf_mols(
    sd_data,
    conformer_sd_data,
    dtype,
    expected,
    sd_tag,
    conformer,
):
    """Test to check that the `get_sd_data` function works as expected with
    multi conformer molecules. Namely that if a conformer is supplied its SD data
    is retrieved (if present) falling back to the parent mol if not (and vice versa
    for parent molecule and conformer), as well as data type casting and erroring
    if no data is present."""
    mol = oechem.OEMol()
    oechem.OESmilesToMol(mol, "CC")
    for tag, value in sd_data.items():
        oechem.OESetSDData(mol, tag, str(value))
    conf = mol.GetActive()
    for tag, value in conformer_sd_data.items():
        oechem.OESetSDData(conf, tag, str(value))

    if expected is not Exception:
        value = get_sd_data(conf if conformer else mol, tag=sd_tag, dtype=dtype)
        assert value == expected
    else:
        with pytest.raises(ValueError, match=f"Unable to find SD tag {sd_tag!s} .*"):
            get_sd_data(conf if conformer else mol, tag=sd_tag, dtype=dtype)


def test_get_sd_data_graphmol():
    """Test that retrieval of SD data from graphmols works as expected."""
    mol = oechem.OEGraphMol()
    oechem.OESmilesToMol(mol, "CC")
    sd_data = {"tag": "20"}
    for tag, value in sd_data.items():
        oechem.OESetSDData(mol, tag, str(value))

    value = get_sd_data(mol, tag="tag", dtype=float)
    assert value == 20.0


@pytest.mark.parametrize(
    ("mols", "expand_conformers", "num", "expected_type"),
    [
        (
            [
                {"type": oechem.OEMol, "num_confs": 3},
                {"type": oechem.OEMol, "num_confs": 1},
            ],
            True,
            4,
            oechem.OEConfBase,
        ),
        (
            [
                {"type": oechem.OEMol, "num_confs": 3},
                {"type": oechem.OEMol, "num_confs": 1},
            ],
            False,
            2,
            oechem.OEConfBase,
        ),
        (
            [
                {"type": oechem.OEGraphMol},
                {"type": oechem.OEGraphMol},
            ],
            True,
            2,
            oechem.OEGraphMol,
        ),
    ],
    ids=["oemols-expand-each-conf", "oemols-active-conf", "graphmols"],
)
def test_iterate_mols_or_confs(mols, expand_conformers, num, expected_type):
    """Tests that iterating through `oemol` object conformers works as expected."""

    def create_mols(mol_dicts):
        for mol_dict in mol_dicts:
            m = mol_dict["type"]()
            oechem.OESmilesToMol(m, "CC")
            for _ in range(max(mol_dict.get("num_confs", 0) - 1, 0)):
                m.NewConf()
            yield m

    mols = create_mols(mols)

    count = 0
    for expanded_mol in iterate_mols_or_confs(
        mols,
        expand_conformers=expand_conformers,
    ):
        count += 1
        assert isinstance(expanded_mol, expected_type)

    assert count == num


def test_hex_to_oemol():
    """That can convert a hex string to OEMol object"""
    # Acetic Acid
    hexstring = "0bce0a94138f8443434f4f8300010101020201030120810010b381803f000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000218100"

    out = to_oemol(hexstring)
    assert isinstance(out, oechem.OEMol)

    out_as_smiles = to_smiles(out)
    assert out_as_smiles == "CC(=O)O"
