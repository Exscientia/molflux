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
MACCS fingerprint. [OpenEye]

MACCS keys are 166 bit structural key descriptors in which each bit is
associated with a SMARTS pattern.
"""


class MACCS(RepresentationBase):
    def _info(self) -> RepresentationInfo:
        return RepresentationInfo(
            description=_DESCRIPTION,
        )

    def _featurise(
        self,
        samples: MolArray,
        **kwargs: Any,
    ) -> Dict[str, List[Fingerprint]]:
        """Generates MACCS fingerprints for each input molecule.

        Args:
            samples: The molecules to featurise

        Returns:
            MACCS fingerprints, as lists of bits.

        Examples:
            >>> from molflux.features import load_representation
            >>> representation = load_representation('maccs')
            >>> samples = ['c1ccccc1']
            >>> representation.featurise(samples=samples)
            {'maccs': [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0]]}
        """

        maccs_fp_list: List[Fingerprint] = []
        for sample in samples:
            with featurisation_error_harness(sample):
                # patch openeye support for fingerprints of empty SMILES
                if sample == "":
                    _MACCS_FINGERPRINT_LENGTH = 166
                    bit_vector = [0] * _MACCS_FINGERPRINT_LENGTH

                else:
                    mol = to_oemol(sample)
                    fp = oegraphsim.OEFingerPrint()
                    oegraphsim.OEMakeMACCS166FP(fp, mol)
                    bit_vector = fingerprint_to_bit_vector(fp)

                maccs_fp_list.append(bit_vector)

        return {self.tag: maccs_fp_list}
