from typing import Any, Dict, List, Optional

try:
    from rdkit.Chem.AtomPairs.Pairs import GetAtomPairFingerprint

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
The atom-pair fingerprint for a molecule.

The atom-pair (and related topological-torsion) fingerprints, are implemented
based on the original paper. Atoms are typed based on atomic number,
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

The algorithm used is described here: R.E. Carhart, D.H. Smith, R. Venkataraghavan; "Atom Pairs as Molecular Features in Structure-Activity Studies: Definition and Applications" JCICS 25, 64-73 (1985).
"""


class AtomPairUnfolded(RepresentationBase):
    def _info(self) -> RepresentationInfo:
        return RepresentationInfo(
            description=_DESCRIPTION,
        )

    def _featurise(
        self,
        samples: MolArray,
        min_length: int = 1,
        max_length: int = 30,
        from_atoms: Optional[List[int]] = None,
        ignore_atoms: Optional[List[int]] = None,
        atom_invariants: Optional[List[int]] = None,
        include_chirality: bool = False,
        use_2d: bool = True,
        conf_id: int = -1,
        **kwargs: Any,
    ) -> Dict[str, List[Dict[str, int]]]:
        """Generates an atom-pair fingerprint for each input molecule.

        Args:
            samples: The molecules to be fingerprinted.
            min_length: The minimum distance between atoms to be considered in
                a pair. Default is 1 bond.
            max_length: The maximum distance between atoms to be considered in
                a pair. Default is 30 bonds.
            from_atoms: If not None, only atom pairs that involve the specified
                atoms will be included in the fingerprint. Defaults to `None`.
            ignore_atoms: If not None, any atom pairs that include the specified
                atoms will not be included in the fingerprint. Defaults to
                `None`.
            atom_invariants: A list of invariants to use for the atom hashes.
                Note that only the first 9 bits of each invariant are used.
                Defaults to `None`.
            include_chirality: If `True`, chirality will be used in the atom
                invariants. Defaults to `False`. Note that this is ignored if
                `atom_invariants` are provided.
            use_2d: Whether to use the 2D (topological) distance matrix.
                Defaults to `True`.
            conf_id: The conformation to use if 3D distances are being used.
                Defaults to `-1`.

        Returns:
            Unfolded Atom-pair fingerprints, as dictionaries.

        Examples:
            >>> from molflux.features import load_representation
            >>> representation = load_representation('atom_pair_unfolded')
            >>> samples = ['c1ccccc1']
            >>> representation.featurise(samples, max_length=16)
            {'atom_pair_unfolded': [{'689473': 6, '689474': 6, '689475': 3}]}
        """

        atom_pairs_fp_list: List[Dict[str, int]] = []
        for sample in samples:
            with featurisation_error_harness(sample):
                smile = to_smiles(sample)
                mol = rdkit_mol_from_smiles(smile)
                rd_fp = GetAtomPairFingerprint(
                    mol,
                    minLength=min_length,
                    maxLength=max_length,
                    fromAtoms=from_atoms or [],
                    ignoreAtoms=ignore_atoms or [],
                    atomInvariants=atom_invariants or [],
                    includeChirality=include_chirality,
                    use2D=use_2d,
                    confId=conf_id,
                )
                atom_pairs_fp_list.append(
                    {str(k): v for k, v in rd_fp.GetNonzeromolfluxs().items()},
                )

        return {self.tag: atom_pairs_fp_list}
