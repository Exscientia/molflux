from __future__ import annotations

import ctypes
import difflib
import logging
from typing import Any, Callable, Iterable, Iterator

import numpy as np

try:
    from openeye import oechem, oegraphsim
except ImportError as e:
    from molflux.features.errors import ExtrasDependencyImportError

    raise ExtrasDependencyImportError("openeye", e) from None

logger = logging.getLogger(__name__)


def oemol_from_smiles(smiles: str) -> oechem.OEMolBase:
    """Returns a OEMol object representing the input SMILES string."""
    oemol = oechem.OEMol()
    if oechem.OESmilesToMol(oemol, smiles):
        return oemol


def oemol_from_bytes(bytes_molecule: bytes) -> oechem.OEMolBase:
    """Returns a OEMol object representing the input bytes string."""
    oemol = oechem.OEMol()
    if oechem.OEReadMolFromBytes(oemol, ".oeb", bytes_molecule):
        return oemol


def oemol_from_hex(hex_molecule: str) -> oechem.OEMolBase:
    """Returns a OEMol object representing the input hex string."""
    bytes_molecule = bytes.fromhex(hex_molecule)
    return oemol_from_bytes(bytes_molecule)


def smiles_from_oemol(oemol: oechem.OEMolBase, flavor: int | None = None) -> str:
    """
    Returns a SMILES string representing the input OEMol object.
    """
    if flavor is None:
        flavor = oechem.OEGetDefaultOFlavor(oechem.OEFormat_SMI)
    smiles: str = oechem.OEWriteMolToString(oechem.OEFormat_SMI, flavor, False, oemol)
    return smiles.rstrip("\n")


def smiles_from_bytes(bytes_molecule: bytes) -> str:
    """
    Returns a SMILES string representing the input bytes string.
    """
    oemol = oemol_from_bytes(bytes_molecule)
    return smiles_from_oemol(oemol)


def to_oemol(molecule: Any) -> oechem.OEMolBase:
    """
    Converts a single input to an OEMol object. This is done by going over the following cases in order:

    - input is an OEMolBase child, in which case it is returned unchanged
    - input is a string
        - if it can be parsed as a SMILES, it is converted to an OEMol from the SMILES
        - else, if it is a hexadecimal string, it is converted to bytes and then to an OEMol
        - else, raise a TypeError
    - input is a bytes
        - it is converted to an OEMol
    - else, raise an error

    Args:
        molecule: Any

    Returns:
        OEMolBase
    """

    if isinstance(molecule, oechem.OEMolBase):
        return molecule

    if isinstance(molecule, str):
        try:
            oemol = oemol_from_smiles(molecule)
            if oemol is not None:
                return oemol
        except:  # noqa: E722 S110
            pass

        try:
            return oemol_from_hex(molecule)
        except ValueError:
            pass

        raise TypeError(
            f"Unparsable string input sample: {molecule}. Tried to parse as SMILES and HEX.",
        )

    elif isinstance(molecule, bytes):
        return oemol_from_bytes(molecule)

    else:
        raise TypeError(f"Unsupported input sample type: {type(molecule)!r}")


def to_smiles(molecule: Any) -> str:
    """
    Converts a single sample to a SMILES string.
    """

    if isinstance(molecule, str):
        if oechem.OEParseSmiles(
            oechem.OEMol(),
            molecule,
            oechem.OEParseSmilesOptions(quiet=True),
        ):
            return molecule

        try:
            return smiles_from_oemol(oemol_from_hex(molecule))
        except ValueError:
            pass

        raise TypeError(
            f"Unparsable string input sample: {molecule}. Tried to parse as SMILES and HEX.",
        )

    elif isinstance(molecule, oechem.OEMolBase):
        return smiles_from_oemol(molecule)

    elif isinstance(molecule, bytes):
        return smiles_from_bytes(molecule)

    else:
        raise TypeError(f"Unsupported input sample type: {type(molecule)!r}")


def to_hex(molecule: Any) -> str:
    """
    Converts a single sample to bytes and then to a hex string.
    """
    mol = to_oemol(molecule)
    mol_bytes = oechem.OEWriteMolToBytes(".oeb", mol)
    mol_hex = str(mol_bytes.hex())
    return mol_hex


def write_mols(
    mols: Iterable[oechem.OEMolBase],
    output_file: str,
) -> None:
    # dest must be a valid output format
    if not oechem.OEIsWriteable(output_file):
        raise ValueError(
            f"`{output_file}` is not a supported chemical structure format.",
        )

    num_mols = 0
    with oechem.oemolostream(output_file) as oss:
        for mol in mols:
            # written as a copy since OEWriteMolecule can change the object
            oechem.OEWriteMolecule(oss, mol.CreateCopy())
            num_mols += 1


def to_digit(data: Any, start: int, stop: int, num: int) -> np.ndarray:
    bins = np.linspace(start=start, stop=stop, num=num)
    binned = np.digitize([data], bins, right=False)
    digit = np.zeros(bins.size + 1, dtype=int)
    for i in binned:
        digit[i] = 1
    return digit


def iterate_mols_or_confs(
    mols: Iterable[oechem.OEMolBase],
    expand_conformers: bool = False,
) -> Iterator[oechem.OEMolBase]:
    """Iterates over molecules and yields either the molecule if it's an OEGraphMol,
    or the conformers if it's an OEMol.

    Args:
        mols: Iterable of molecule objects.
        expand_conformers: Determines whether to expand out conformations of an
            OEMol object (`True`) or to simply yield the active conformation
            (`False`, default).

    Yields:
        Each molecule or conformation in `mols`.
    """
    for mol in mols:
        if isinstance(mol, oechem.OEMCMolBase):
            yield from mol.GetConfs() if expand_conformers else (mol.GetActive(),)
        else:
            yield mol


def get_sd_data(
    mol: oechem.OEMolBase,
    tag: str,
    *,
    dtype: Callable[[Any], Any] = str,
) -> Any:
    """Function to obtain SD data from a molecule object by `tag`.

    Args:
        mol: The molecule to obtain the SD data for.
        tag: The SD data tag to retrieve.
        dtype: The type to cast the value associated with `tag` to. Default is
            `str`.

    Returns:
        The value from the SD tag of `mol` given by `tag`.

    Notes:
        Note, that if `mol` is a conformer, the `tag` is checked on the parent
        MC molecule object if the tag is not present on the conformer. If the
        `mol` object is an `OEMol` object and the tag is not present it will
        check the active conformer for the presence of the tag (via `.GetActive()`).
    """
    if not oechem.OEHasSDData(mol, tag):
        if isinstance(mol, oechem.OEConfBase) and oechem.OEHasSDData(
            mol.GetMCMol(),
            tag,
        ):
            mol = mol.GetMCMol()
        elif isinstance(mol, oechem.OEMCMolBase) and oechem.OEHasSDData(
            mol.GetActive(),
            tag,
        ):
            mol = mol.GetActive()
        else:
            sd_tags = difflib.get_close_matches(
                tag,
                {
                    sd_data_pair.GetTag()
                    for sd_data_pair in oechem.OEGetSDDataPairs(mol)
                },
                # 0.1 cutoff used to provide users with more possible options.
                # cutoff is based on textual similarity, lower is more lenient.
                cutoff=0.1,
            )
            raise ValueError(
                f"Unable to find SD tag {tag!s} on molecule {mol.GetTitle()}. Closest "
                f"SD tags found: {sd_tags}",
            )
    value = oechem.OEGetSDData(mol, tag)
    try:
        value = dtype(value)
    except Exception as err:
        raise TypeError(
            f"Error occurred when attempting to cast {tag!s}={value} to {dtype=}. Error: {err}",
        ) from err

    return value


def fingerprint_to_bit_vector(fingerprint: oegraphsim.OEFingerPrint) -> list[int]:
    """Casts a chemical fingerprint to a bit vector."""

    # calculate the number of bytes / fingerprint can have any size
    fingerprint_size = fingerprint.GetSize()
    nbytes: int = (fingerprint_size // 8) + (1 if fingerprint_size % 8 else 0)

    # get fp bytes from c pointer
    array_type = ctypes.POINTER(ctypes.c_ubyte * nbytes)
    pointer = ctypes.cast(int(fingerprint.GetData()), array_type)
    bits = np.unpackbits(np.ctypeslib.as_array(pointer, shape=(1,)))

    # get correct fp length back
    return bits.reshape(nbytes, 8)[:, ::-1].ravel()[:fingerprint_size].tolist()  # type: ignore[no-any-return]
