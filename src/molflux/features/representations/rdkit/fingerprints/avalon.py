from typing import Any

try:
    from rdkit.Avalon.pyAvalonTools import GetAvalonFP

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
Avalon fingerprints from rdkit.
"""


class Avalon(RepresentationBase):
    def _info(self) -> RepresentationInfo:
        return RepresentationInfo(
            description=_DESCRIPTION,
        )

    def _featurise(
        self,
        *columns: MolArray,
        n_bits: int = 512,
        is_query: bool = False,
        reset_vect: bool = False,
        bit_flags: int = 15761407,
        **kwargs: Any,
    ) -> dict[str, list[Fingerprint]]:
        """Generates the Avalon fingerprint for each input molecule.

        Args:
            samples: The molecules to be fingerprinted.
            n_bits:
            is_query:
            reset_vect:
            bit_flags:

        Returns:
            Avalon fingerprints, as lists of bits.

        Examples:
            >>> from molflux.features import load_representation
            >>> representation = load_representation('avalon')
            >>> samples = ['c1ccccc1']
            >>> representation.featurise(samples, n_bits=16)
            {'avalon': [[0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0]]}
        """
        assert_n_positional_args(*columns, expected_size=1)
        samples = columns[0]
        avalon_fp_list: list[list] = []
        for sample in samples:
            with featurisation_error_harness(sample):
                smile = to_smiles(sample)
                mol = rdkit_mol_from_smiles(smile)
                rd_fp = GetAvalonFP(
                    mol,
                    nBits=n_bits,
                    isQuery=is_query,
                    resetVect=reset_vect,
                    bitFlags=bit_flags,
                )
                avalon_fp_list.append(rd_fp.ToList())

        return {self.tag: avalon_fp_list}
