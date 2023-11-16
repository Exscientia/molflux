from typing import Any, Dict, List, Optional

try:
    from openeye.oechem import OEFloatArray
    from openeye.oemolprop import OEGetXLogPResult
    from openeye.oequacpac import OERemoveFormalCharge
except ImportError as e:
    from molflux.features.errors import ExtrasDependencyImportError

    raise ExtrasDependencyImportError("openeye", e) from None

from molflux.features.bases import RepresentationBase
from molflux.features.info import RepresentationInfo
from molflux.features.representations.openeye._utils import to_digit, to_oemol
from molflux.features.typing import MolArray
from molflux.features.utils import featurisation_error_harness

_DESCRIPTION = """
The XLogP algorithm ([Wang-1997-2]) is provided because its atom-type
contribution allows calculation of the XLogP contribution of any fragment in a
molecule and allows minimal corrections in a simple additive form to calculate
the LogP of any molecule made from combinations of fragments.

Further, although the method contains many many free parameters, its simple
linear form allows for ready interpretation of the model and most of the
parameters in the model make rational sense.

Unfortunately, the original algorithm is difficult to implement as published.
First, the internal-hydrogen bond term was calculated using a single
3D conformation. It was found that this was both arbitrary and unnecessary.
This arbitrary 3D calculation has been replaced with a 2D approach to recognize
common internal-hydrogen bonds. In tests, this 2D method worked comparably to
the published 3D algorithm. Next, the training set had a few subtle atom-type
inconsistencies.
"""


class XLogP(RepresentationBase):
    def _info(self) -> RepresentationInfo:
        return RepresentationInfo(
            description=_DESCRIPTION,
        )

    def _featurise(
        self,
        samples: MolArray,
        digitise: bool = False,
        start: int = -3,
        stop: int = 7,
        num: int = 10,
        atom_xlogps: Optional[OEFloatArray] = None,
        remove_formal_charge: bool = True,
        **kwargs: Any,
    ) -> Dict[str, List[float]]:
        """Calculates the XLogP value for each input molecule.

        .. hint::

            XLogP should be calculated on the neutral form of the molecule.
            Therefore, calling it is highly recommended to make sure to set
            `remove_formal_charge` to `True` when using this representation.

        Args:
            samples: The sample molecules to featurise.
            digitise: Whether to digitise the results. Defaults to `False`.
            start: If `digitise=True`, the start of the binning range. Defaults
                to -3.
            stop: If `digitise=True`, the end of the binning range. Defaults
                to 7.
            num: If `digitise=True`, the number of bins. Defaults to 10.
            atom_xlogps: Can be used to retrieve the contribution of each atom
                to the total XLogP.
            remove_formal_charge: Whether to attempt to remove all formal
                charges from mol in a manner that is consistent with adding and
                removing implicit or explicit protons. This method will not
                create radicals and will not attach more protons to an atom
                than is acceptable in that atom`s standard valence form.
                Please note that the formal charge of quaternary amines is not
                removed with this flag. Defaults to `True`.

        Returns:
            The computed XLogP values on each input sample.

        Examples:
            >>> from molflux.features import load_representation
            >>> representation = load_representation('x_log_p')
            >>> samples = ['C1=CC=C(C=C1)S(=O)O', 'C']
            >>> representation.featurise(samples=samples)
            {'x_log_p': [-0.31300002336502075, 0.27699998021125793]}
        """

        results = []
        for sample in samples:
            with featurisation_error_harness(sample):
                mol = to_oemol(sample)
                # XLogP should be calculated on the neutral form of the molecule
                if remove_formal_charge:
                    OERemoveFormalCharge(mol)

                result = OEGetXLogPResult(mol, atom_xlogps).GetValue()

                if digitise:
                    result = to_digit(result, start=start, stop=stop, num=num).tolist()

                results.append(result)

        return {self.tag: results}
