from typing import Any, Dict, List

try:
    from openeye.oechem import OENetCharge
except ImportError as e:
    from molflux.features.errors import ExtrasDependencyImportError

    raise ExtrasDependencyImportError("openeye", e) from None

from molflux.features.bases import RepresentationBase
from molflux.features.info import RepresentationInfo
from molflux.features.representations.openeye._utils import to_digit, to_oemol
from molflux.features.typing import MolArray
from molflux.features.utils import featurisation_error_harness

_DESCRIPTION = """
Determines the net charge on a molecule.

If the molecule has specified partial charges, this featuriser returns the sum
of the partial charges rounded to an integer. Otherwise this featuriser returns
the sum of the formal charges on each atom of the molecule.
"""


class NetCharge(RepresentationBase):
    def _info(self) -> RepresentationInfo:
        return RepresentationInfo(
            description=_DESCRIPTION,
        )

    def _featurise(
        self,
        samples: MolArray,
        digitise: bool = False,
        start: int = -2,
        stop: int = 2,
        num: int = 4,
        **kwargs: Any,
    ) -> Dict[str, List[int]]:
        """Calculates the net charge of each input molecule.

        Args:
            samples: The sample molecules to featurise.
            digitise: Whether to digitise the results. Defaults to `False`.
            start: If `digitise=True`, the start of the binning range. Defaults
                to -2.
            stop: If `digitise=True`, the end of the binning range. Defaults
                to 2.
            num: If `digitise=True`, the number of bins. Defaults to 4.

        Returns:
            The net charge of each input molecule.

        Examples:
            >>> from molflux.features import load_representation
            >>> representation = load_representation('net_charge')
            >>> samples = ['c1ccccc1', '[Na+]Cl']
            >>> representation.featurise(samples)
            {'net_charge': [0, 1]}
        """

        results = []
        for sample in samples:
            with featurisation_error_harness(sample):
                mol = to_oemol(sample)
                result = OENetCharge(mol)

                if digitise:
                    result = to_digit(result, start=start, stop=stop, num=num).tolist()

                results.append(result)

        return {self.tag: results}
