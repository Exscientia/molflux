from typing import Any, Dict, List

try:
    from openeye.oechem import OECalculateMolecularWeight
except ImportError as e:
    from molflux.features.errors import ExtrasDependencyImportError

    raise ExtrasDependencyImportError("openeye", e) from None

from molflux.features.bases import RepresentationBase
from molflux.features.info import RepresentationInfo
from molflux.features.representations.openeye._utils import to_digit, to_oemol
from molflux.features.typing import MolArray
from molflux.features.utils import featurisation_error_harness

_DESCRIPTION = """
The molecular weight of a given molecule.

By default, all atoms are assumed their average atomic weight.
"""


class MolecularWeight(RepresentationBase):
    def _info(self) -> RepresentationInfo:
        return RepresentationInfo(
            description=_DESCRIPTION,
        )

    def _featurise(
        self,
        samples: MolArray,
        digitise: bool = False,
        start: int = 0,
        stop: int = 700,
        num: int = 14,
        isotopic: bool = False,
        **kwargs: Any,
    ) -> Dict[str, List[float]]:
        """Calculates the molecular weight of each input molecule.

        By default, all atoms are assumed their average atomic weight.

        Args:
            samples: The sample molecules to featurise.
            digitise: Whether to digitise the results. Defaults to `False`.
            start: If `digitise=True`, the start of the binning range. Defaults
                to 0.
            stop: If `digitise=True`, the end of the binning range. Defaults
                to 700.
            num: If `digitise=True`, the number of bins. Defaults to 14.
            isotopic: Whether to use the `OEGetIsotopicWeight` function instead
                of `OEGetAverageWeight`. If `True`, atoms must have a specified
                (non-zero) isotopic mass, as returned by
                `OEAtomBase.GetIsotope`. Defaults to `False`.

        Returns:
            The molecular weight of each input molecule.

        Examples:
            >>> from molflux.features import load_representation
            >>> representation = load_representation('molecular_weight')
            >>> samples = ['c1ccccc1', 'C']
            >>> representation.featurise(samples)
            {'molecular_weight': [78.11184, 16.04246]}
        """

        results = []
        for sample in samples:
            with featurisation_error_harness(sample):
                mol = to_oemol(sample)
                result = OECalculateMolecularWeight(mol, isotopic=isotopic)

                if digitise:
                    result = to_digit(result, start=start, stop=stop, num=num).tolist()

                results.append(result)

        return {self.tag: results}
