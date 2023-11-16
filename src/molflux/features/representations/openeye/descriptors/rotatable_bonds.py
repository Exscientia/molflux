from typing import Any, Dict, List

try:
    from openeye.oemolprop import OEGetRotatableBondCount
except ImportError as e:
    from molflux.features.errors import ExtrasDependencyImportError

    raise ExtrasDependencyImportError("openeye", e) from None

from molflux.features.bases import RepresentationBase
from molflux.features.info import RepresentationInfo
from molflux.features.representations.openeye._utils import to_digit, to_oemol
from molflux.features.typing import MolArray
from molflux.features.utils import featurisation_error_harness

_DESCRIPTION = """
The number of rotatable bond counts in a molecule.

Rotatable bond is defined as any single non-ring bond, bounded to nonterminal
heavy (i.e. non-hydrogen) atom. In addition, It considers the carbon-carbon
triple bond in acetylene rotatable. The representation excludes single bonds
observed in a certain groups, such as sulfonamides, esters, and amidine.
"""


class RotatableBonds(RepresentationBase):
    def _info(self) -> RepresentationInfo:
        return RepresentationInfo(
            description=_DESCRIPTION,
        )

    def _featurise(
        self,
        samples: MolArray,
        digitise: bool = False,
        start: int = 0,
        stop: int = 16,
        num: int = 8,
        adjust: bool = False,
        **kwargs: Any,
    ) -> Dict[str, List[int]]:
        """Calculates the number of rotatable bond counts for each input molecule.

        Args:
            samples: The sample molecules to featurise.
            digitise: Whether to digitise the results. Defaults to `False`.
            start: If `digitise=True`, the start of the binning range. Defaults
                to 0.
            stop: If `digitise=True`, the end of the binning range. Defaults
                to 16.
            num: If `digitise=True`, the number of bins. Defaults to 8.
            adjust: This parameter allows optional adjustment for aliphatic
                rings following the method of [Oprea-2000]_. In this case, it
                also estimates the number of flexible bonds in aliphatics rings

        Returns:
            The number of rotatable bond counts of each input molecule.

        References:
            [Oprea-2000]_ Oprea, T., Property Distribution of Drug-Related Chemical Databases, Journal of Computer-Aided Molecular Design, Vol. 14, pp. 251-264, 2000

        Examples:
            >>> from molflux.features import load_representation
            >>> representation = load_representation('rotatable_bonds')
            >>> samples = ['C1=CC=C(C=C1)S(=O)O', 'NH']
            >>> representation.featurise(samples=samples)
            {'rotatable_bonds': [1, 0]}
        """

        results = []
        for sample in samples:
            with featurisation_error_harness(sample):
                mol = to_oemol(sample)
                result = OEGetRotatableBondCount(mol=mol, adjust=adjust)

                if digitise:
                    result = to_digit(result, start=start, stop=stop, num=num).tolist()

                results.append(result)

        return {self.tag: results}
