from typing import Any

try:
    from rdkit.Chem.MACCSkeys import GenMACCSKeys

    from molflux.features.representations.rdkit._utils import rdkit_mol_from_smiles
except ImportError as e:
    from molflux.features.errors import ExtrasDependencyImportError

    raise ExtrasDependencyImportError("rdkit", e) from None

from molflux.features.bases import RepresentationBase
from molflux.features.info import RepresentationInfo
from molflux.features.representations.rdkit._utils import to_smiles
from molflux.features.typing import Fingerprint, MolArray
from molflux.features.utils import assert_n_positional_args, featurisation_error_harness

_DESCRIPTION = """
MACCS fingerprint. [rdkit]

MACCS keys are 166 bit structural key descriptors in which each bit is
associated with a SMARTS pattern.
"""


class MACCSRdkit(RepresentationBase):
    def _info(self) -> RepresentationInfo:
        return RepresentationInfo(
            description=_DESCRIPTION,
        )

    def _featurise(
        self,
        *columns: MolArray,
        **kwargs: Any,
    ) -> dict[str, list[Fingerprint]]:
        """Generates MACCS fingerprints for each input molecule.

        Args:
            samples: The molecules to be fingerprinted.

        Returns:
            MACCS fingerprints, as lists of bits.

        Examples:
            >>> from molflux.features import load_representation
            >>> representation = load_representation('maccs_rdkit')
            >>> samples = ['c1ccccc1']
            >>> representation.featurise(samples)
            {'maccs_rdkit': [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0]]}
        """
        assert_n_positional_args(*columns, expected_size=1)
        samples = columns[0]
        maccs_fp_list: list[list] = []
        for sample in samples:
            with featurisation_error_harness(sample):
                smile = to_smiles(sample)
                mol = rdkit_mol_from_smiles(smile)
                rd_fp = GenMACCSKeys(mol)

                if rd_fp.GetNumOnBits() == 0:
                    bit_vector = [0] * rd_fp.GetNumBits()
                else:
                    bit_vector = rd_fp.ToList()

                maccs_fp_list.append(bit_vector[1:])

        return {self.tag: maccs_fp_list}
