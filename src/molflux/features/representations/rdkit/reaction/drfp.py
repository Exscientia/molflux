from __future__ import annotations

from typing import TYPE_CHECKING, Any

try:
    from molflux.features.representations.rdkit.reaction._drfp_vendored import (
        DrfpEncoder,
    )
except ImportError as e:
    from molflux.features.errors import ExtrasDependencyImportError

    raise ExtrasDependencyImportError("rdkit", e) from None

from molflux.features.bases import RepresentationBase
from molflux.features.info import RepresentationInfo
from molflux.features.utils import featurisation_error_harness

if TYPE_CHECKING:
    from molflux.features.typing import Fingerprint, SmilesArray

_DESCRIPTION = """
The DRFP algorithm takes a reaction SMILES as an input and creates a binary
fingerprint based on the symmetric difference of two sets containing the
circular molecular n-grams generated from the molecules listed left and right
from the reaction arrow, respectively, without the need for distinguishing
between reactants and reagents.

https://pubs.rsc.org/en/content/articlehtml/2022/dd/d1dd00006c
"""


class DRFP(RepresentationBase):
    def _info(self) -> RepresentationInfo:
        return RepresentationInfo(
            description=_DESCRIPTION,
        )

    def _featurise(
        self,
        samples: SmilesArray,
        length: int = 2048,
        min_radius: int = 0,
        max_radius: int = 3,
        include_rings: bool = True,
        root_central_atom: bool = True,
        include_hydrogens: bool = False,
        **kwargs: Any,
    ) -> dict[str, list[Fingerprint]]:
        """Featurises the input molecules as DRFP (differential
        reaction fingerprint.

        Args:
            samples: The array of samples to featurise.
            length: The folded length of the fingerprint (the parameter for the modulo hashing)
            min_radius: The minimum radius of a substructure (0 includes single atoms)
            max_radius: The maximum radius of a substructure
            include_rings: Whether to include full rings as substructures
            atom_index_mapping: Return the atom indices of mapped substructures for each reaction
            root_central_atom: Whether to root the central atom of substructures when generating SMILES
            include_hydrogens: Whether to explicitly include hydrogens in the molecular graph

        Returns:
            DRFP fingerpints.

        Examples:
            >>> from molflux.features import load_representation
            >>> representation = load_representation("drfp")
            >>> samples = ["CO.O.O=C(NC(=S)Nc1nc(-c2ccccc2)cs1)c1ccccc1.[Na+].[OH-]>>NC(=S)Nc1nc(-c2ccccc2)cs1"]
            >>> representation.featurise(samples, length=4)
            {'drfp': [[1, 1, 1, 1]]}
        """
        del kwargs

        fingerprints: list[Fingerprint] = []
        for sample in samples:
            with featurisation_error_harness(sample):
                fingerprint = DrfpEncoder.encode_one(
                    sample,
                    length=length,
                    min_radius=min_radius,
                    max_radius=max_radius,
                    include_rings=include_rings,
                    root_central_atom=root_central_atom,
                    include_hydrogens=include_hydrogens,
                )
                fingerprints.append(fingerprint)

        return {self.tag: fingerprints}
