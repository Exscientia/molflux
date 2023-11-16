from typing import Any, Dict, List

try:
    from openeye.oemolprop import OEGetLipinskiAcceptorCount
except ImportError as e:
    from molflux.features.errors import ExtrasDependencyImportError

    raise ExtrasDependencyImportError("openeye", e) from None

from molflux.features.bases import RepresentationBase
from molflux.features.info import RepresentationInfo
from molflux.features.representations.openeye._utils import to_digit, to_oemol
from molflux.features.typing import MolArray
from molflux.features.utils import featurisation_error_harness

_DESCRIPTION = """
The number of acceptors in a molecule based on the definition from the work of
Lipinski ([Lipinski-1997]).

It is defined to be the number of nitrogens or oxygens.
"""


class NumAcceptors(RepresentationBase):
    def _info(self) -> RepresentationInfo:
        return RepresentationInfo(
            description=_DESCRIPTION,
        )

    def _featurise(
        self,
        samples: MolArray,
        digitise: bool = False,
        start: int = 0,
        stop: int = 10,
        num: int = 10,
        **kwargs: Any,
    ) -> Dict[str, List[int]]:
        """Calculates the number of Lipinski acceptors in a molecule.

        Args:
            samples: The sample molecules to featurise.
            digitise: Whether to digitise the results. Defaults to `False`.
            start: If `digitise=True`, the start of the binning range. Defaults to 0.
            stop: If `digitise=True`, the end of the binning range. Defaults to 10.
            num: If `digitise=True`, the number of bins. Defaults to 10.

        Returns:
            The number of acceptors for each input molecule.

        Examples:
            >>> from molflux.features import load_representation
            >>> representation = load_representation('num_acceptors')
            >>> samples = ['c1ccccc1', "NH"]
            >>> representation.featurise(samples=samples)
            {'num_acceptors': [0, 1]}
        """

        results = []
        for sample in samples:
            with featurisation_error_harness(sample):
                mol = to_oemol(sample)
                result = OEGetLipinskiAcceptorCount(mol)

                if digitise:
                    result = to_digit(result, start=start, stop=stop, num=num).tolist()

                results.append(result)

        return {self.tag: results}
