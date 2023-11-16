from typing import Any, Dict, List

try:
    from openeye.oemolprop import OEGetHBondDonorCount
    from openeye.oequacpac import OESetNeutralpHModel
except ImportError as e:
    from molflux.features.errors import ExtrasDependencyImportError

    raise ExtrasDependencyImportError("openeye", e) from None

from molflux.features.bases import RepresentationBase
from molflux.features.info import RepresentationInfo
from molflux.features.representations.openeye._utils import to_digit, to_oemol
from molflux.features.typing import MolArray
from molflux.features.utils import featurisation_error_harness

_DESCRIPTION = """
The number of hydrogen-bond donors in a molecule based on the definition in the
work of Mills and Dean ([MillsDean-1996]) and also in the book by
Jeffrey ([Jeffrey-1997]).

It is defined to be the number of hydrogen atoms on nitrogen, oxygen, or
sulfur atoms.
"""


class NumDonors(RepresentationBase):
    def _info(self) -> RepresentationInfo:
        return RepresentationInfo(
            description=_DESCRIPTION,
        )

    def _featurise(
        self,
        samples: MolArray,
        digitise: bool = False,
        start: int = 0,
        stop: int = 5,
        num: int = 5,
        set_neutral_ph: bool = False,
        **kwargs: Any,
    ) -> Dict[str, List[int]]:
        """Calculates the number of hydrogen-bond donors for each input molecule.

        Args:
            samples: The sample molecules to featurise.
            digitise: Whether to digitise the results. Defaults to `False`.
            start: If `digitise=True`, the start of the binning range. Defaults
                to 0.
            stop: If `digitise=True`, the end of the binning range. Defaults
                to 0.
            num: If `digitise=True`, the number of bins. Defaults to 5.
            set_neutral_ph: Whether to attempt to set mol to the most
                energetically favorable ionization state for pH=7.4 using a
                rule-based system. Multiple states are not enumerated for any
                molecule. In cases where the ionization is near 7.4 and could be
                represented by a mixture of states, the state that is likely to
                be most populated is chosen.

        Returns:
            The number of hydrogen-bond donors of each input molecule.

        Examples:
            >>> from molflux.features import load_representation
            >>> representation = load_representation('num_donors')
            >>> samples = ['c1ccccc1', "NH"]
            >>> representation.featurise(samples=samples)
            {'num_donors': [0, 3]}
        """

        results = []
        for sample in samples:
            with featurisation_error_harness(sample):
                mol = to_oemol(sample)
                if set_neutral_ph:
                    new = mol.CreateCopy()
                    OESetNeutralpHModel(new)
                    result = OEGetHBondDonorCount(mol=new)
                else:
                    result = OEGetHBondDonorCount(mol=mol)

                if digitise:
                    result = to_digit(result, start=start, stop=stop, num=num).tolist()

                results.append(result)

        return {self.tag: results}
