from typing import Any, Dict, List, Optional

try:
    from openeye.oemolprop import OEGet2dPSA
except ImportError as e:
    from molflux.features.errors import ExtrasDependencyImportError

    raise ExtrasDependencyImportError("openeye", e) from None

from molflux.features.bases import RepresentationBase
from molflux.features.info import RepresentationInfo
from molflux.features.representations.openeye._utils import to_digit, to_oemol
from molflux.features.typing import MolArray
from molflux.features.utils import featurisation_error_harness

_DESCRIPTION = """
The Topological polar-surface area (TPSA) for a given molecule.

Topological polar-surface area (TPSA) is based on the algorithm developed by
Ertl et al [Ertl-2000]. In Ertl`s publication, use of TPSA both with and without
accounting for phosphorus and sulfur surface-area is reported. However,
evidence shows that in most PK applications one is better off not counting the
contributions of phosphorus and sulfur atoms toward the total TPSA for a
molecule. This implementation of TPSA allows either inclusion or exclusion of
phosphorus and sulfur surface area.
"""


class TPSA(RepresentationBase):
    def _info(self) -> RepresentationInfo:
        return RepresentationInfo(
            description=_DESCRIPTION,
        )

    def _featurise(
        self,
        samples: MolArray,
        digitise: bool = False,
        start: int = 0,
        stop: int = 140,
        num: int = 14,
        atom_psa: Optional[float] = None,
        s_and_p: bool = False,
        **kwargs: Any,
    ) -> Dict[str, List[float]]:
        """Calculates the Topological polar-surface area (TPSA) for each input
        molecule.

        .. warning::

            TPSA values are mildly sensitive to the protonation state of a molecule.

        Args:
            samples: The sample molecules to featurise.
            digitise: Whether to digitise the results. Defaults to `False`.
            start: If `digitise=True`, the start of the binning range. Defaults
                to 0.
            stop: If `digitise=True`, the end of the binning range. Defaults
                to 140.
            num: If `digitise=True`, the number of bins. Defaults to 14.
            atom_psa: Can be used to retrieve the contribution of each atom to
                the total polar surface area.
            s_and_p: Whether sulfur and phosphorus should be counted towards
                the total surface area. Defaults to `False`.

        Returns:
            The Topological polar-surface area (TPSA) for each input molecule.

        Examples:
            >>> from molflux.features import load_representation
            >>> representation = load_representation('tpsa')
            >>> samples = ['C1=CC=C(C=C1)S(=O)O', 'C']
            >>> representation.featurise(samples=samples)
            {'tpsa': [37.29999923706055, 0.0]}
        """

        results = []
        for sample in samples:
            with featurisation_error_harness(sample):
                mol = to_oemol(sample)
                result = OEGet2dPSA(mol, atomPSA=atom_psa, SandP=s_and_p)

                if digitise:
                    result = to_digit(result, start=start, stop=stop, num=num).tolist()

                results.append(result)

        return {self.tag: results}
