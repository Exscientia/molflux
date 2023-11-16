from typing import Any, Dict, List

try:
    from openeye import oegraphsim
except ImportError as e:
    from molflux.features.errors import ExtrasDependencyImportError

    raise ExtrasDependencyImportError("openeye", e) from None

from molflux.features.bases import RepresentationBase
from molflux.features.info import RepresentationInfo
from molflux.features.representations.openeye._utils import (
    fingerprint_to_bit_vector,
    to_oemol,
)
from molflux.features.typing import Fingerprint, MolArray
from molflux.features.utils import featurisation_error_harness

_DESCRIPTION = """
LINGO similarity search.

LINGO is a very fast text-based molecular similarity search method. It is based
on fragmentation of canonical isomeric SMILES strings into overlapping
substrings.
"""


class Lingo(RepresentationBase):
    def _info(self) -> RepresentationInfo:
        return RepresentationInfo(
            description=_DESCRIPTION,
        )

    def _featurise(
        self,
        samples: MolArray,
        **kwargs: Any,
    ) -> Dict[str, List[Fingerprint]]:
        """Generates lingo fingerprints for each input molecule.

        Args:
            samples: The molecules to featurise

        Returns:
            Lingo fingerprints, as lists of bits.

        Examples:
            >>> from molflux.features import load_representation
            >>> representation = load_representation('lingo')
            >>> samples = ['c1ccccc1']
            >>> representation.featurise(samples=samples)
            {'lingo': [[1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]}
        """
        lingo_fp_list: List[Fingerprint] = []
        for sample in samples:
            with featurisation_error_harness(sample):
                # patch openeye support for fingerprints of empty SMILES
                if sample == "":
                    bit_vector = [0]

                else:
                    mol = to_oemol(sample)
                    fp = oegraphsim.OEFingerPrint()
                    oegraphsim.OEMakeLingoFP(fp, mol)
                    bit_vector = fingerprint_to_bit_vector(fp)

                lingo_fp_list.append(bit_vector)

        return {self.tag: lingo_fp_list}
