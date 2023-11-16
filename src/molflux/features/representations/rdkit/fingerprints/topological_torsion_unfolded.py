from typing import Any, Dict, List, Optional

try:
    from rdkit.Chem.AtomPairs.Torsions import GetTopologicalTorsionFingerprint

    from molflux.features.representations.rdkit._utils import rdkit_mol_from_smiles
except ImportError as e:
    from molflux.features.errors import ExtrasDependencyImportError

    raise ExtrasDependencyImportError("rdkit", e) from None

from molflux.features.bases import RepresentationBase
from molflux.features.info import RepresentationInfo
from molflux.features.representations.rdkit._utils import to_smiles
from molflux.features.typing import MolArray
from molflux.features.utils import featurisation_error_harness

_DESCRIPTION = """
Topological-torsion fingerprints, as described in:

The topological torsion (and related atom-pair) fingerprints, are implemented
based on the original papers. Atoms are typed based on atomic number,
number of pi electrons, and the degree of the atom. Optionally information
about atomic chirality can also be integrated into the atom types. Both
fingerprint types can be generated in explicit or sparse form and as bit or
count vectors. These fingerprint types are different from the others in the
RDKit in that bits in the sparse form of the fingerprint can be directly
explained (i.e. the “hashing function” used is fully reversible).

These fingerprints were originally “intended” to be used in count-vectors and
they seem to work better that way. The default behavior of the explicit
bit-vector forms of both fingerprints is to use a “count simulation” procedure
where multiple bits are set for a given feature if it occurs more than once.
The default behavior is to use 4 fingerprint bits for each feature
(so a 2048 bit fingerprint actually stores information about the same number of
features as a 512 bit fingerprint that isn`t using count simulation). The bins
correspond to counts of 1, 2, 4, and 8. As an example of how this works: if a
feature occurs 5 times in a molecule, the bits corresponding to counts 1, 2,
and 4 will be set.

R. Nilakantan, N. Bauman, J. S. Dixon, R. Venkataraghavan;
“Topological Torsion: A New Molecular Descriptor for SAR Applications.
Comparison with Other Descriptors” JCICS 27, 82-85 (1987).
"""


class TopologicalTorsionUnfolded(RepresentationBase):
    def _info(self) -> RepresentationInfo:
        return RepresentationInfo(
            description=_DESCRIPTION,
        )

    def _featurise(
        self,
        samples: MolArray,
        target_size: int = 4,
        from_atoms: Optional[List[int]] = None,
        ignore_atoms: Optional[List[int]] = None,
        atom_invariants: Optional[List[int]] = None,
        include_chirality: bool = False,
        **kwargs: Any,
    ) -> Dict[str, List[Dict[str, int]]]:
        """Generates unfolded topological-torsion fingerprints for each input
        molecule.

        Args:
            samples: The molecules to be fingerprinted.
            target_size: The number of atoms to include in the "torsions".
                Defaults to `4`.
            from_atoms: If provided, only torsions that start or end at the
                specified atoms will be included in the fingerprint. Defaults
                to `None`.
            ignore_atoms: If provided, any torsions that include the specified
                atoms will not be included in the fingerprint. Defaults to
                `None`.
            atom_invariants: A list of invariants to use for the atom hashes.
                Note that only the first 9 bits of each invariant are used.
                Defaults to `None`.
            include_chirality: If set, chirality will be used in the atom
                invariants. Note that this is ignored if `atom_invariants` are
                provided. Defaults to `False`.

        Returns:
            Unfolded (sparse) topological-torsion fingerprints, as dictionaries
            of non-zero molfluxs.

        Examples:
            >>> from molflux.features import load_representation
            >>> representation = load_representation('topological_torsion_unfolded')
            >>> samples = ['c1ccccc1']
            >>> representation.featurise(samples)
            {'topological_torsion_unfolded': [{'5513433129': 6}]}
        """

        torsion_fp_list: List[Dict[str, int]] = []
        for sample in samples:
            with featurisation_error_harness(sample):
                smile = to_smiles(sample)
                mol = rdkit_mol_from_smiles(smile)
                rd_fp = GetTopologicalTorsionFingerprint(
                    mol,
                    targetSize=target_size,
                    fromAtoms=from_atoms or [],
                    ignoreAtoms=ignore_atoms or [],
                    atomInvariants=atom_invariants or [],
                    includeChirality=include_chirality,
                )
                torsion_fp_list.append(
                    {str(k): v for k, v in rd_fp.GetNonzeromolfluxs().items()},
                )

        return {self.tag: torsion_fp_list}
