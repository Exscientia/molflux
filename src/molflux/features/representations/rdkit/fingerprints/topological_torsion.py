from typing import Any, Dict, List, Optional

try:
    from rdkit.Chem import rdFingerprintGenerator

    from molflux.features.representations.rdkit._utils import rdkit_mol_from_smiles
except ImportError as e:
    from molflux.features.errors import ExtrasDependencyImportError

    raise ExtrasDependencyImportError("rdkit", e) from None

from molflux.features.bases import RepresentationBase
from molflux.features.info import RepresentationInfo
from molflux.features.representations.rdkit._utils import to_smiles
from molflux.features.typing import Fingerprint, MolArray
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


class TopologicalTorsion(RepresentationBase):
    def _info(self) -> RepresentationInfo:
        return RepresentationInfo(
            description=_DESCRIPTION,
        )

    def _featurise(
        self,
        samples: MolArray,
        include_chirality: bool = False,
        torsion_atom_count: int = 4,
        count_simulation: bool = True,
        count_bounds: Optional[object] = None,
        fp_size: int = 2048,
        atom_invariants_generator: Optional[object] = None,
        **kwargs: Any,
    ) -> Dict[str, List[Fingerprint]]:
        """
        Generates a folded topological torsion fingerprint for each input molecule from a fingerprint generator.

        Args:
            samples: The molecules to be fingerprinted.
            include_chirality: if set, chirality will be used in the atom
                invariants, this is ignored if atom_invariants_generator is provided.
                Defaults to `False`.
            torsion_atom_count: the number of atoms to include in the "torsions".
                Default is 4.
            count_simulation:  if set, use count simulation while  generating
                the fingerprint. Defaults to `True`.
            count_bounds: boundaries for count simulation, corresponding bit
                will be  set if the count is higher than the number provided
                for that spot. Defaults to `None`.
            fp_size: size of the generated fingerprint, does not affect the
                sparse versions. Defaults is 2048.
            atom_invariants_generator: atom invariants to be used during
                fingerprint generation. Defaults to `None`.

        Returns:
            Folded atom-pair fingerprints, as lists of bits.

        Examples:
            >>> from molflux.features import load_representation
            >>> representation = load_representation('topological_torsion')
            >>> samples = ['CN1C=NC2=C1C(=O)N(C(=O)N2C)C']
            >>> representation.featurise(samples, fp_size=16)
            {'topological_torsion': [[1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0]]}
        """

        ttgen = rdFingerprintGenerator.GetTopologicalTorsionGenerator(
            includeChirality=include_chirality,
            torsionAtomCount=torsion_atom_count,
            countSimulation=count_simulation,
            countBounds=count_bounds,
            fpSize=fp_size,
            atomInvariantsGenerator=atom_invariants_generator,
        )

        topological_torsion_fp_list: List[Fingerprint] = []
        for sample in samples:
            with featurisation_error_harness(sample):
                smiles = to_smiles(sample)
                mol = rdkit_mol_from_smiles(smiles)
                fingerprint = list(ttgen.GetFingerprint(mol))
                topological_torsion_fp_list.append(fingerprint)

        return {self.tag: topological_torsion_fp_list}
