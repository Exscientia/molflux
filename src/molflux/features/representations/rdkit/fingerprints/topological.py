from typing import Any, Dict, List, Optional

try:
    from rdkit.Chem import RDKFingerprint

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
The RDKit topological fingerprint for a molecule.

These are topological (Daylight like) fingerprints for a molecule using an
alternate (faster) hashing algorithm. This RDKit-specific fingerprint is
inspired by (though it differs significantly from) public descriptions of the
Daylight fingerprint 7. The fingerprinting algorithm identifies all subgraphs
in the molecule within a particular range of sizes, hashes each subgraph to
generate a raw bit ID, mods that raw bit ID to fit in the assigned fingerprint
size, and then sets the corresponding bit. Options are available to generate
count-based forms of the fingerprint or “non-folded” forms (using a sparse
representation).

The default scheme for hashing subgraphs is to hash the individual bonds based
on:
* the types of the two atoms. Atom types include the atomic number (mod 128), and whether or not the atom is aromatic.
* the degrees of the two atoms in the path.
* the bond type (or AROMATIC if the bond is marked as aromatic)
"""


class Topological(RepresentationBase):
    def _info(self) -> RepresentationInfo:
        return RepresentationInfo(
            description=_DESCRIPTION,
        )

    def _featurise(
        self,
        samples: MolArray,
        min_path: int = 1,
        max_path: int = 7,
        fp_size: int = 2048,
        n_bits_per_hash: int = 2,
        use_hs: bool = True,
        tgt_density: float = 0,
        min_size: int = 128,
        branched_paths: bool = True,
        use_bond_order: bool = True,
        atom_invariants: Optional[List[int]] = None,
        from_atoms: Optional[List] = None,
        atom_bits: Optional[List] = None,
        bit_info: Optional[Dict] = None,
        **kwargs: Any,
    ) -> Dict[str, List[Fingerprint]]:
        """Generates topological (Daylight like) fingerprints for each input
        molecules.

        Args:
            samples: The molecules to be fingerprinted.
            min_path: The minimum path length (in bonds) to be included.
                Defaults to `1`.
            max_path: The maximum path length (in bonds) to be included.
                Defaults to `7`
            fp_size: The size of the fingerprint. Defaults to `2048`.
            n_bits_per_hash: The number of bits to be set by each path.
                Defaults to `2`.
            use_hs: Toggles inclusion of Hs in paths (if the molecule has
                explicit Hs). Defaults to `True`.
            tgt_density: If the generated fingerprint is below this density,
                it will be folded until the density is reached. Defaults to `0`.
            min_size: The minimum size to which the fingerprint will be folded.
                Defaults to `128`.
            branched_paths: Toggles generation of branched subgraphs, not just
                linear paths. Defaults to `True`.
            use_bond_order: Toggles inclusion of bond orders in the path hashes.
                Defaults to `True`.
            atom_invariants: A vector of atom invariants to use while hashing
                the paths. Defaults to `None`.
            from_atoms: Only paths starting at these atoms will be included.
                Defaults to `None`.
            atom_bits: Used to return the bits that each atom is involved in
                (should be at least mol.numAtoms long). Defaults to `None`.
            bit_info: Defaults to `None`.

        Returns:
            Topological (Daylight like) fingerprints, as lists of bits.

        Examples:
            >>> from molflux.features import load_representation
            >>> representation = load_representation('topological')
            >>> samples = ['c1ccccc1']
            >>> representation.featurise(samples, fp_size=16)
            {'topological': [[0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0]]}
        """
        topo_fp_list: List[List] = []
        for sample in samples:
            with featurisation_error_harness(sample):
                smile = to_smiles(sample)
                mol = rdkit_mol_from_smiles(smile)
                rd_fp = RDKFingerprint(
                    mol,
                    minPath=min_path,
                    maxPath=max_path,
                    fpSize=fp_size,
                    nBitsPerHash=n_bits_per_hash,
                    useHs=use_hs,
                    tgtDensity=tgt_density,
                    minSize=min_size,
                    branchedPaths=branched_paths,
                    useBondOrder=use_bond_order,
                    atomInvariants=atom_invariants or [],
                    fromAtoms=from_atoms or [],
                    atomBits=atom_bits or [],
                    bitInfo=bit_info or {},
                )

                if rd_fp.GetNumOnBits() == 0:
                    topo_fp_list.append([0] * rd_fp.GetNumBits())
                else:
                    topo_fp_list.append(rd_fp.ToList())

        return {self.tag: topo_fp_list}
