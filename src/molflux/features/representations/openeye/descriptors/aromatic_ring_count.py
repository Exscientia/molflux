from typing import Any, Dict, List

try:
    from openeye.oemolprop import OEGetAromaticRingCount
except ImportError as e:
    from molflux.features.errors import ExtrasDependencyImportError

    raise ExtrasDependencyImportError("openeye", e) from None

from molflux.features.bases import RepresentationBase
from molflux.features.info import RepresentationInfo
from molflux.features.representations.openeye._utils import to_digit, to_oemol
from molflux.features.typing import MolArray
from molflux.features.utils import featurisation_error_harness

_DESCRIPTION = """
The number of aromatic rings in a molecule.

The terminology 'number of aromatic rings' (or aromatic ring
count) is used generically and encompasses both benzenoid
aromatic rings and heteroaromatics (including, e.g. pyridine and
imidazole). ... Each ring in a fused system is counted
individually; thus, indole and naphthalene are each defined as
having two aromatic rings.
"""


class AromaticRingCount(RepresentationBase):
    def _info(self) -> RepresentationInfo:
        return RepresentationInfo(
            description=_DESCRIPTION,
        )

    def _featurise(
        self,
        samples: MolArray,
        digitise: bool = False,
        start: int = 0,
        stop: int = 7,
        num: int = 7,
        **kwargs: Any,
    ) -> Dict[str, List[int]]:
        """Calculates the number of aromatic rings for each input molecule.

        Args:
            samples: The sample molecules to featurise.
            digitise: Whether to digitise the results. Defaults to `False`.
            start: If `digitise=True`, the start of the binning range. Defaults
                to 0.
            stop: If `digitise=True`, the end of the binning range. Defaults
                to 7.
            num: If `digitise=True`, the number of bins. Defaults to 7.

        Returns:
            The number of aromatic rings for each input molecule.

        Examples:
            >>> from molflux.features import load_representation
            >>> representation = load_representation('aromatic_ring_count')
            >>> samples = ['c1ccccc1', 'C']
            >>> representation.featurise(samples)
            {'aromatic_ring_count': [1, 0]}
        """

        results = []
        for sample in samples:
            with featurisation_error_harness(sample):
                mol = to_oemol(sample)
                result = OEGetAromaticRingCount(mol)

                if digitise:
                    result = to_digit(result, start=start, stop=stop, num=num).tolist()

                results.append(result)

        return {self.tag: results}
