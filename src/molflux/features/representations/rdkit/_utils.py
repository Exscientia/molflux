from typing import Any

try:
    from rdkit import Chem, rdBase

except ImportError as e:
    from molflux.features.errors import ExtrasDependencyImportError

    raise ExtrasDependencyImportError("rdkit", e) from None


class RDKitLogContext:
    """Context manager to disable RDKit logs. By default all logs are disabled."""

    def __init__(
        self,
        mute_errors: bool = True,
        mute_warning: bool = True,
        mute_info: bool = True,
        mute_debug: bool = True,
    ):
        # Get current log state
        self.previous_status = self._get_log_status()

        # Init the desired log state to apply during in the context
        self.desired_status = {}
        self.desired_status["rdApp.error"] = not mute_errors
        self.desired_status["rdApp.warning"] = not mute_warning
        self.desired_status["rdApp.debug"] = not mute_debug
        self.desired_status["rdApp.info"] = not mute_info

    def _get_log_status(self) -> Any:
        """Get the current log status of RDKit logs."""
        log_status = rdBase.LogStatus()
        log_status = {
            st.split(":")[0]: st.split(":")[1] for st in log_status.split("\n")
        }
        log_status = {
            k: True if v == "enabled" else False for k, v in log_status.items()
        }
        return log_status

    def _apply_log_status(self, log_status: Any) -> None:
        """Apply an RDKit log status."""
        for k, v in log_status.items():
            if v is True:
                rdBase.EnableLog(k)
            else:
                rdBase.DisableLog(k)

    def __enter__(self) -> None:
        self._apply_log_status(self.desired_status)

    def __exit__(self, *args: Any, **kwargs: Any) -> None:
        self._apply_log_status(self.previous_status)


def rdkit_mol_from_smiles(smiles: str) -> Chem.Mol:
    """Returns a Chem.Mol object representing the input SMILES string."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise RuntimeError(f"Error parsing {smiles} using RDKit")
        return mol

    except TypeError as e:
        raise TypeError(
            f"{smiles} of type {type(smiles)} not supported by RDKit",
        ) from e


def rdkit_mol_from_bytes(bytes_molecule: bytes) -> Chem.Mol:
    """Returns a Chem.Mol object representing the input bytes string."""
    rdkit_mol = Chem.Mol(bytes_molecule)
    if rdkit_mol:
        return rdkit_mol


def rdkit_mol_from_hex(hex_molecule: str) -> Chem.Mol:
    """Returns a Chem.Mol object representing the input hex string."""
    bytes_molecule = bytes.fromhex(hex_molecule)
    return rdkit_mol_from_bytes(bytes_molecule)


def smiles_from_rdkit_mol(rdkit_mol: Chem.Mol) -> str:
    """
    Returns a SMILES string representing the input Chem.Mol object.
    """
    smiles: str = Chem.MolToSmiles(rdkit_mol)
    return smiles


def smiles_from_bytes(bytes_molecule: bytes) -> str:
    """
    Returns a SMILES string representing the input bytes string.
    """
    rdkit_mol = rdkit_mol_from_bytes(bytes_molecule)
    return smiles_from_rdkit_mol(rdkit_mol)


def to_rdkit_mol(molecule: Any) -> Chem.Mol:
    """
    Converts a single input to an Chem.Mol object. This is done by going over the following cases in order:

    - input is a Chem.Mol, in which case it is returned unchanged
    - input is a string
        - if it can be parsed as a SMILES, it is converted to a Chem.Mol from the SMILES
        - else, if it is a hexadecimal string, it is converted to bytes and then to a Chem.Mol
        - else, raise a TypeError
    - input is a bytes
        - it is converted to an Chem.Mol
    - else, raise an error

    Args:
        molecule: Any

    Returns:
        Chem.Mol
    """
    with RDKitLogContext():
        if isinstance(molecule, Chem.Mol):
            return molecule

        if isinstance(molecule, str):
            try:
                rdkit_mol = rdkit_mol_from_smiles(molecule)
                if rdkit_mol is not None:
                    return rdkit_mol
            except:  # noqa: E722 S110
                pass

            try:
                return rdkit_mol_from_hex(molecule)
            except ValueError:
                pass

            raise TypeError(
                f"Unparsable string input sample: {molecule}. Tried to parse as SMILES and HEX.",
            )

        elif isinstance(molecule, bytes):
            return rdkit_mol_from_bytes(molecule)

        else:
            raise TypeError(f"Unsupported input sample type: {type(molecule)!r}")


def to_smiles(molecule: Any) -> str:
    """
    Converts a single sample to a SMILES string.
    """
    with RDKitLogContext():
        if isinstance(molecule, str):
            mol = Chem.MolFromSmiles(molecule)
            if mol:
                return molecule

            try:
                return smiles_from_rdkit_mol(rdkit_mol_from_hex(molecule))
            except ValueError:
                pass

            raise TypeError(
                f"Unparsable string input sample: {molecule}. Tried to parse as SMILES and HEX.",
            )

        elif isinstance(molecule, Chem.Mol):
            return smiles_from_rdkit_mol(molecule)

        elif isinstance(molecule, bytes):
            return smiles_from_bytes(molecule)

        else:
            raise TypeError(f"Unsupported input sample type: {type(molecule)!r}")


def to_hex(molecule: Any) -> str:
    """
    Converts a single sample to bytes and then to a hex string.
    """

    mol = to_rdkit_mol(molecule)

    mol_bytes = mol.ToBinary(Chem.PropertyPickleOptions.AllProps)

    mol_hex = str(mol_bytes.hex())

    return mol_hex
