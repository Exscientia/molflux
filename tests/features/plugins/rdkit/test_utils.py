import pytest
from rdkit.Chem import Mol

from molflux.features.representations.rdkit._utils import (
    rdkit_mol_from_smiles,
    to_rdkit_mol,
    to_smiles,
)

representation_name = "avalon"


def test_rdkit_mol_from_smiles_valid_smiles():
    """Test that rdkit_mol_from_smiles works on valid data"""
    valid_smiles = [
        "c1ccccc1",
        "CCC",
        "CCCC1=NN(C2=C1NC(=NC2=O)C3=C(C=CC(=C3)S(=O)(=O)N4CCN(CC4)C)OCC)C",
    ]

    mols = [rdkit_mol_from_smiles(smiles) for smiles in valid_smiles]

    for mol in mols:
        assert mol is not None
        assert isinstance(mol, Mol)


def test_bytes_to_rdkit_mol():
    """That can convert a bytes string to OEMol object"""
    # Acetic Acid
    bytestring = b"\xef\xbe\xad\xde\x00\x00\x00\x00\r\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x04\x00\x00\x00\x03\x00\x00\x00\x80\x01\x06\x00`\x00\x00\x00\x01\x03\x06\x00(\x00\x00\x00\x03\x04\x08\x00(\x00\x00\x00\x03\x02\x08\x00h\x00\x00\x00\x03\x01\x01\x0b\x00\x01\x00\x01\x02(\x02\x01\x03 \x14\x00\x00\x00\x00\x17\x04\x00\x00\x00\x00\x00\x00\x00\x16"

    out = to_rdkit_mol(bytestring)
    assert isinstance(out, Mol)

    out_as_smiles = to_smiles(out)
    assert out_as_smiles == "CC(=O)O"


def test_rdkit_bytes_to_smiles():
    """That can convert a bytes string to SMILES"""
    # Acetic Acid
    bytestring = b"\xef\xbe\xad\xde\x00\x00\x00\x00\r\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x04\x00\x00\x00\x03\x00\x00\x00\x80\x01\x06\x00`\x00\x00\x00\x01\x03\x06\x00(\x00\x00\x00\x03\x04\x08\x00(\x00\x00\x00\x03\x02\x08\x00h\x00\x00\x00\x03\x01\x01\x0b\x00\x01\x00\x01\x02(\x02\x01\x03 \x14\x00\x00\x00\x00\x17\x04\x00\x00\x00\x00\x00\x00\x00\x16"

    out = to_smiles(bytestring)
    assert isinstance(out, str)
    assert out == "CC(=O)O"


def test_rdkit_mol_from_smiles_bad_smiles():
    """Test that a RuntimeError is raised on invalid smiles"""
    bad_smiles = ["not_a_smiles", "HHH", "C1.c2"]

    for smiles in bad_smiles:
        with pytest.raises(RuntimeError):
            rdkit_mol_from_smiles(smiles)


def test_rdkit_mol_from_smiles_bad_types():
    """Test that a TypeError is raised on non-string inputs"""
    bad_types_samples = [1, 3.14, [1, 2]]

    for sample in bad_types_samples:
        with pytest.raises(TypeError):
            rdkit_mol_from_smiles(sample)  # type: ignore


def test_hex_to_smiles():
    """That can convert a hex string to SMILES"""
    # Acetic Acid
    hexstring = "efbeadde000000000d0000000000000002000000040000000300000080010600600000000103060028000000030408002800000003020800680000000301010b00010001022802010320140000000017040000000000000016"

    out = to_smiles(hexstring)
    assert isinstance(out, str)
    assert out == "CC(=O)O"


def test_hex_to_oemol():
    """That can convert a hex string to OEMol object"""
    # Acetic Acid
    hexstring = "efbeadde000000000d0000000000000002000000040000000300000080010600600000000103060028000000030408002800000003020800680000000301010b00010001022802010320140000000017040000000000000016"

    out = to_rdkit_mol(hexstring)
    assert isinstance(out, Mol)

    out_as_smiles = to_smiles(out)
    assert out_as_smiles == "CC(=O)O"
