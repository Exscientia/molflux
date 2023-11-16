from typing import Any, Dict, List, Optional

try:
    from rdkit.Chem.rdmolops import LayeredFingerprint
    from rdkit.DataStructs.cDataStructs import ExplicitBitVect

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
Layered fingerprint for a molecule.

A topological (Daylight like) fingerprint for a molecule using a layer-based
hashing algorithm. These are another “RDKit original” and were developed with
the intention of using them as a substructure fingerprint. Since the pattern
fingerprint is far simpler and has proven to be quite effective as a
substructure fingerprint, the layered fingerprint hasn`t received much
attention. It may still be interesting for something, so we continue to include
it.

The idea of the fingerprint is generate features using the same subgraph
(or path) enumeration algorithm used in the RDKit fingerprint.
After a subgraph has been generated, it is used to set multiple bits based on
different atom and bond type definitions.
"""


class Layered(RepresentationBase):
    def _info(self) -> RepresentationInfo:
        return RepresentationInfo(
            description=_DESCRIPTION,
        )

    def _featurise(
        self,
        samples: MolArray,
        layer_flags: int = 0xFFFFFFFF,
        min_path: int = 1,
        max_path: int = 7,
        fp_size: int = 2048,
        atom_counts: Optional[List[int]] = None,
        set_only_bits: Optional[ExplicitBitVect] = None,
        branched_paths: bool = True,
        from_atoms: Optional[List] = None,
        **kwargs: Any,
    ) -> Dict[str, List[Fingerprint]]:
        """Generates a layered fingerprint for each input molecule.

        Args:
            samples: The molecules to be fingerprinted.
            layer_flags: The layers to be included (see below)
            min_path: The minimum path length (in bonds) to be included.
                Defaults to `1`.
            max_path: The minimum path length (in bonds) to be included.
                Defaults to `7`.
            fp_size: The size of the fingerprint. Defaults to `2048`.
            atom_counts: If not `None`, this will be used to provide the count
                of the number of paths that set bits each atom is involved in.
                The vector should have at least as many entries as the molecule
                has atoms and is not zeroed out here. Defaults to `None`.
            set_only_bits: If provided, only bits that are set in this bit
                vector will be set in the result. This is essentially the same
                as doing: `(*res) &= (*setOnlyBits)`; but also has an impact on
                the `atom_counts` (if being used).
            branched_paths: Toggles generation of branched subgraphs, not just
                linear paths. Defaults to `None`.
            from_atoms: If provided, only the atoms in the vector will be used
                as centers in the fingerprint. Defaults to `None`.

        Layer definitions:

            * 0x01: pure topology
            * 0x02: bond order
            * 0x04: atom types
            * 0x08: presence of rings
            * 0x10: ring sizes
            * 0x20: aromaticity

        Returns:
            Layered fingerprints, as lists of bits.

        Examples:
            >>> from molflux.features import load_representation
            >>> representation = load_representation('layered')
            >>> samples = ['c1ccccc1']
            >>> representation.featurise(samples, fp_size=16)
            {'layered': [[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1]]}
        """

        layered_fp_list: List[List] = []
        for sample in samples:
            with featurisation_error_harness(sample):
                smile = to_smiles(sample)
                mol = rdkit_mol_from_smiles(smile)
                rd_fp = LayeredFingerprint(
                    mol,
                    layerFlags=layer_flags,
                    minPath=min_path,
                    maxPath=max_path,
                    fpSize=fp_size,
                    atomCounts=atom_counts or [],
                    setOnlyBits=set_only_bits,
                    branchedPaths=branched_paths,
                    fromAtoms=from_atoms or [],
                )

                if rd_fp.GetNumOnBits() == 0:
                    layered_fp_list.append([0] * rd_fp.GetNumBits())
                else:
                    layered_fp_list.append(rd_fp.ToList())

        return {self.tag: layered_fp_list}
