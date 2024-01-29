"""Vendored with minor modifications from DRFP

https://raw.githubusercontent.com/reymond-group/drfp/main/src/drfp/fingerprint.py
https://github.com/reymond-group/drfp/commit/6c5db5cc7d2057e932cb1e1be8d697dc2c3327e1
"""
from hashlib import blake2b
from typing import Any, Dict, Iterable, List, Set, Tuple

import numpy as np
from rdkit.Chem import AllChem
from rdkit.Chem.rdchem import Mol


class NoReactionError(Exception):
    """Raised when the encoder attempts to encode a non-reaction SMILES.

    Attributes:
        message: a message containing the non-reaction SMILES
    """

    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)


class DrfpEncoder:
    """A class for encoding SMILES as drfp fingerprints."""

    @staticmethod
    def shingling_from_mol(
        in_mol: Mol,
        max_radius: int = 3,
        include_rings: bool = True,
        min_radius: int = 0,
        root_central_atom: bool = True,
        include_hydrogens: bool = False,
    ) -> List[bytes]:
        """Creates a molecular shingling from a RDKit molecule (rdkit.Chem.rdchem.Mol).

        Arguments:
            in_mol: A RDKit molecule instance
            max_radius: The drfp radius (a radius of 3 corresponds to drfp6)
            include_rings: Whether or not to include rings in the shingling
            min_radius: The minimum radius that is used to extract n-grams

        Returns:
            The molecular shingling.
        """

        if include_hydrogens:
            in_mol = AllChem.AddHs(in_mol)

        shingling = []

        if include_rings:
            for ring in AllChem.GetSymmSSSR(in_mol):
                bonds = set()
                ring = list(ring)
                indices = set()
                for i in ring:
                    for j in ring:
                        if i != j:
                            indices.add(i)
                            indices.add(j)
                            bond = in_mol.GetBondBetweenAtoms(i, j)
                            if bond is not None:
                                bonds.add(bond.GetIdx())

                ngram = AllChem.MolToSmiles(
                    AllChem.PathToSubmol(in_mol, list(bonds)),
                    canonical=True,
                    allHsExplicit=True,
                ).encode("utf-8")

                shingling.append(ngram)

        if min_radius == 0:
            for atom in in_mol.GetAtoms():
                ngram = atom.GetSmarts().encode("utf-8")
                shingling.append(ngram)

        for index, _ in enumerate(in_mol.GetAtoms()):
            for i in range(1, max_radius + 1):
                p = AllChem.FindAtomEnvironmentOfRadiusN(
                    in_mol,
                    i,
                    index,
                    useHs=include_hydrogens,
                )
                amap: Dict[int, Any] = {}
                submol = AllChem.PathToSubmol(in_mol, p, atomMap=amap)

                if index not in amap:
                    continue

                smiles = ""

                if root_central_atom:
                    smiles = AllChem.MolToSmiles(
                        submol,
                        rootedAtAtom=amap[index],
                        canonical=True,
                        allHsExplicit=True,
                    )
                else:
                    smiles = AllChem.MolToSmiles(
                        submol,
                        canonical=True,
                        allHsExplicit=True,
                    )

                if smiles != "":
                    shingling.append(smiles.encode("utf-8"))

        # Set ensures that the same shingle is not hashed multiple times
        # (which would not change the hash, since there would be no new minima)
        return list(set(shingling))

    @staticmethod
    def internal_encode(
        in_smiles: str,
        max_radius: int = 3,
        min_radius: int = 0,
        include_rings: bool = True,
        root_central_atom: bool = True,
        include_hydrogens: bool = False,
    ) -> Tuple[np.ndarray, List[bytes]]:
        """Creates an drfp array from a reaction SMILES string.

        Arguments:
            in_smiles: A valid reaction SMILES string
            max_radius: The drfp radius (a radius of 3 corresponds to drfp6)
            min_radius: The minimum radius that is used to extract n-grams
            include_rings: Whether or not to include rings in the shingling

        Returns:
            A tuple with two arrays, the first containing the drfp hash values, the second the substructure SMILES
        """

        sides = in_smiles.split(">")
        if len(sides) < 3:
            raise NoReactionError(
                f"The following is not a valid reaction SMILES: '{in_smiles}'",
            )

        if len(sides[1]) > 0:
            sides[0] += "." + sides[1]

        left = sides[0].split(".")
        right = sides[2].split(".")

        left_shingles: Set[bytes] = set()
        right_shingles: Set[bytes] = set()

        for component in left:
            mol = AllChem.MolFromSmiles(component)

            sh = DrfpEncoder.shingling_from_mol(
                mol,
                max_radius=max_radius,
                include_rings=include_rings,
                min_radius=min_radius,
                root_central_atom=root_central_atom,
                include_hydrogens=include_hydrogens,
            )

            for s in sh:
                right_shingles.add(s)

        for component in right:
            mol = AllChem.MolFromSmiles(component)

            sh = DrfpEncoder.shingling_from_mol(
                mol,
                max_radius=max_radius,
                include_rings=include_rings,
                min_radius=min_radius,
                root_central_atom=root_central_atom,
                include_hydrogens=include_hydrogens,
            )

            for s in sh:
                left_shingles.add(s)

        s_diff = right_shingles.symmetric_difference(left_shingles)

        return DrfpEncoder.hash(list(s_diff)), list(s_diff)

    @staticmethod
    def hash(shingling: List[bytes]) -> np.ndarray:
        """Directly hash all the SMILES in a shingling to a 32-bit integerself.

        Arguments:
            shingling: A list of n-grams

        Returns:
            A list of hashed n-grams
        """

        return np.array(
            [int(blake2b(t, digest_size=4).hexdigest(), 16) for t in shingling],
        ).astype(np.int32)

    @staticmethod
    def fold(
        hash_values: np.ndarray,
        length: int = 2048,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Folds the hash values to a binary vector of a given length.

        Arguments:
            hash_value: An array containing the hash values
            length: The length of the folded fingerprint

        Returns:
            A tuple containing the folded fingerprint and the indices of the on bits
        """

        folded = np.zeros(length, dtype=np.uint8)
        on_bits = hash_values % length
        folded[on_bits] = 1

        return folded, on_bits

    @staticmethod
    def encode(
        reactions: Iterable[str],
        length: int = 2048,
        min_radius: int = 0,
        max_radius: int = 3,
        include_rings: bool = True,
        root_central_atom: bool = True,
        include_hydrogens: bool = False,
    ) -> List[List[int]]:
        """Encodes a list of reaction SMILES using the drfp fingerprint.

        Args:
            reactions: An iterable (e.g. list) of reaction SMILES to be encoded
            length: The folded length of the fingerprint (the parameter for the modulo hashing)
            min_radius: The minimum radius of a substructure (0 includes single atoms)
            max_radius: The maximum radius of a substructure
            include_rings: Whether to include full rings as substructures
            root_central_atom: Whether to root the central atom of substructures when generating SMILES

        Returns:
            A list of drfp fingerprints.
        """

        return [
            DrfpEncoder.encode_one(
                reaction,
                length=length,
                min_radius=min_radius,
                max_radius=max_radius,
                include_rings=include_rings,
                root_central_atom=root_central_atom,
                include_hydrogens=include_hydrogens,
            )
            for reaction in reactions
        ]

    @staticmethod
    def encode_one(
        reaction: str,
        length: int,
        min_radius: int,
        max_radius: int,
        include_rings: bool,
        root_central_atom: bool,
        include_hydrogens: bool,
    ) -> List[int]:
        """Encodes a reaction SMILES using the drfp fingerprint.

        Args:
            reaction: A reaction SMILES to be encoded
            length: The folded length of the fingerprint (the parameter for the modulo hashing)
            min_radius: The minimum radius of a substructure (0 includes single atoms)
            max_radius: The maximum radius of a substructure
            include_rings: Whether to include full rings as substructures
            root_central_atom: Whether to root the central atom of substructures when generating SMILES

        Returns:
            A drfp fingerprints.
        """

        hashed_diff, smiles_diff = DrfpEncoder.internal_encode(
            reaction,
            min_radius=min_radius,
            max_radius=max_radius,
            include_rings=include_rings,
            root_central_atom=root_central_atom,
            include_hydrogens=include_hydrogens,
        )

        difference_folded, on_bits = DrfpEncoder.fold(
            hashed_diff,
            length=length,
        )

        return difference_folded.tolist()  # type: ignore[no-any-return]
