from __future__ import annotations

from typing import TYPE_CHECKING, Any

try:
    from mhfp.encoder import MHFPEncoder
except ImportError as e:
    from molflux.features.errors import ExtrasDependencyImportError

    raise ExtrasDependencyImportError("rdkit", e) from None

from molflux.features.bases import RepresentationBase
from molflux.features.info import RepresentationInfo
from molflux.features.representations.rdkit._utils import to_smiles
from molflux.features.utils import featurisation_error_harness

if TYPE_CHECKING:
    from molflux.features.typing import MolArray

_DESCRIPTION = """
MHFP6 (MinHash fingerprint, up to six bonds) is a molecular fingerprint which
encodes detailed substructures using the extended connectivity principle of
ECFP in a fundamentally different manner, increasing the performance of exact
nearest neighbor searches in benchmarking studies and enabling the application
of locality sensitive hashing (LSH) approximate nearest neighbor search
algorithms.

To describe a molecule, MHFP6 extracts the SMILES of all circular substructures
around each atom up to a diameter of six bonds and applies the MinHash method
to the resulting set. MHFP6 outperforms ECFP4 in benchmarking analog recovery
studies. Furthermore, MHFP6 outperforms ECFP4 in approximate nearest neighbor
searches by two orders of magnitude in terms of speed, while decreasing the
error rate

https://jcheminf.biomedcentral.com/articles/10.1186/s13321-018-0321-8
"""


class MHFPUnfolded(RepresentationBase):
    def _info(self) -> RepresentationInfo:
        return RepresentationInfo(
            description=_DESCRIPTION,
        )

    def _featurise(
        self,
        samples: MolArray,
        n_permutations: int = 1024,
        radius: int = 3,
        rings: bool = True,
        kekulise: bool = True,
        sanitise: bool = True,
        seed: int = 42,
        **kwargs: Any,
    ) -> dict[str, list[list[int]]]:
        """Featurises the input molecules as unfolded MinHash fingerprints.

        Args:
            samples: The array of samples to featurise.
            n_permutations: Analogous to the number of bits ECFP fingerprints
                are folded into. Higher is better, lower is less exact. Defaults
                to 1024.
            radius: Analogous to the radius for the Morgan fingerprint. The
                default radius 3 encodes SMILES to MHFP6. Defaults to 3.
            rings: Whether rings in the molecule are included in the fingerprint.
                As a radii of 3 fails to encode rings and there is no way to
                determine ring-membership in a substructure SMILES, this
                considerably increases performance. Defaults to `True`.
            kekulise: Whether or not to kekulise the molecule before extracting
                substructure SMILES. Defaults to `True`.
            sanitise: Whether or not to sanitise the molecule (using RDKit)
                before extracting substructure SMILES. Defaults to `True`.
            seed: The seed for the MinHash operation. Has to be the same for
                comparable fingerprints. Defaults to 42.

        Returns:
            MHFP fingerpints.

        Examples:
            >>> from molflux.features import load_representation
            >>> representation = load_representation("mhfp_unfolded")
            >>> samples = ["CCCC1=NN(C2=C1NC(=NC2=O)C3=C(C=CC(=C3)S(=O)(=O)N4CCN(CC4)C)OCC)C"]
            >>> representation.featurise(samples, n_permutations=4)
            {'mhfp_unfolded': [[15876248, 33988699, 14316170, 98479663]]}
        """
        encoder = MHFPEncoder(n_permutations=n_permutations, seed=seed)

        fingerprints: list[list[int]] = []
        for sample in samples:
            with featurisation_error_harness(sample):
                smiles = to_smiles(sample)
                fingerprint = encoder.encode(
                    smiles,
                    radius=radius,
                    rings=rings,
                    kekulize=kekulise,
                    sanitize=sanitise,
                ).tolist()
                fingerprints.append(fingerprint)

        return {self.tag: fingerprints}
