import logging
from typing import Any, Dict, List

try:
    from openeye import oechem, oeshape
except ImportError as e:
    from molflux.features.errors import ExtrasDependencyImportError

    raise ExtrasDependencyImportError("openeye", e) from None

from molflux.features.bases import RepresentationBase
from molflux.features.info import RepresentationInfo
from molflux.features.representations.openeye._utils import to_oemol
from molflux.features.typing import ArrayLike
from molflux.features.utils import featurisation_error_harness

logger = logging.getLogger(__name__)

_DESCRIPTION = """
A representation based on coefficients of a hermite polynomial expansion, as used in methods such as ROCS
Details on hermite expansions and shape theory for molecules can be found here:
https://docs.eyesopen.com/toolkits/python/shapetk/shape_theory.html
"""


class Hermite(RepresentationBase):
    def _info(self) -> RepresentationInfo:
        return RepresentationInfo(
            description=_DESCRIPTION,
        )

    def _featurise(
        self,
        samples: ArrayLike,
        n_poly_max: int = 8,
        use_optimal_lambdas: bool = True,
        orient_by_moments_of_inertia: bool = False,
        **kwargs: Any,
    ) -> Dict[str, List[List[float]]]:
        r"""Featurises the input molecules via a hermite polynomial expansion.

        Args:
            n_poly_max: Controls the level of Hermite expansion. In particular
                the lowest value 0 corresponds to a very inaccurate expansion,
                approximating the entire molecule by a single Gaussian, while
                the largest currently allowed value is equal to 30, which
                corresponds to a list of 5456 Hermite coefficients. In the limit
                `n_poly_max` tending to infinity, we obtain exact equivalence
                of the Hermite expansion to the Gaussian representation of the
                molecule. Defaults to `8`.
            use_optimal_lambdas: If `True`, optimization of lambda parameters
                will be performed via maximizing the self-overlap of each input
                molecule at a given resolution of the Hermite expansion. Please
                note, that even in the case when set to `False`, if the value
                of `n_poly_max` is high enough the Hermite expansion will
                converge to the Gaussian shape. However, in practice at a given
                low resolution of Hermite expansion it is essential to use the
                optimal lambda parameters for faster convergence. Defaults to
                `True`.
            orient_by_moments_of_inertia: If `True`, each input sample is moved
                to the center of its inertial frame before calculating its
                Hermite coefficients. Defaults to `False`.

        Returns:
            Coefficients of a hermite expansion representing the shape of a molecule

        Examples:
            >>> from molflux.features import load_representation
            >>> representation = load_representation('hermite')
            >>> samples = ['C']
            >>> representation.featurise(samples, n_poly_max=1)
            {'hermite': [[2.70..., 0.0, 0.0, 0.0]]}
        """
        hermite_mols = []
        for sample in samples:
            with featurisation_error_harness(sample):
                mol = to_oemol(sample)
                out = _get_hermite_coeffs(
                    mol,
                    n_poly_max=n_poly_max,
                    use_optimal_lambdas=use_optimal_lambdas,
                    orient_by_moments_of_inertia=orient_by_moments_of_inertia,
                )
                hermite_mols.append(list(out))

        return {self.tag: hermite_mols}


def _get_hermite_coeffs(
    mol: oechem.OEMolBase,
    n_poly_max: int,
    *,
    use_optimal_lambdas: bool = True,
    orient_by_moments_of_inertia: bool = False,
) -> oechem.OEDoubleVector:
    """Calculates Hermite coefficients."""

    opts = oeshape.OEHermiteOptions()
    opts.SetNPolyMax(n_poly_max)
    opts.SetUseOptimalLambdas(use_optimal_lambdas)
    hermite = oeshape.OEHermite(opts)

    if orient_by_moments_of_inertia:
        trans = oechem.OETrans()
        oeshape.OEOrientByMomentsOfInertia(mol, trans)

    if not hermite.Setup(mol):
        logger.error("failed to setup molecule {} for OEHermite.", mol.GetTitle())

    # Length of the output vector can be determined as number of terms that meet
    # the condition l+m+n < n_poly_max. See https://docs.eyesopen.com/toolkits/python/shapetk/OEShapeClasses/OEHermite.html#OEShape::OEHermite::GetCoefficients
    basis_size = (n_poly_max + 1) * (n_poly_max + 2) * (n_poly_max + 3) / 6
    basis_size = int(basis_size)

    coeffs = oechem.OEDoubleVector(basis_size)
    hermite.GetCoefficients(coeffs)

    return coeffs
