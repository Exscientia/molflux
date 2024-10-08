from typing import Any

try:
    from openeye import oegraphsim
except ImportError as e:
    from molflux.features.errors import ExtrasDependencyImportError

    raise ExtrasDependencyImportError("openeye", e) from None

from molflux.features.bases import RepresentationBase
from molflux.features.info import RepresentationInfo
from molflux.features.representations.openeye._utils import (
    fingerprint_to_bit_vector,
    to_oemol,
)
from molflux.features.typing import Fingerprint, MolArray
from molflux.features.utils import assert_n_positional_args, featurisation_error_harness

_DESCRIPTION = """
Tree fingerprint.

A tree fingerprint is generated by exhaustively enumerating all tree fragments
of a molecular graph up to a given size and then hashing these fragments into a
fixed-length bitvector.
"""


class Tree(RepresentationBase):
    def _info(self) -> RepresentationInfo:
        return RepresentationInfo(
            description=_DESCRIPTION,
        )

    def _featurise(
        self,
        *columns: MolArray,
        length: int = 2048,
        diameter: int = 6,
        atom_type: int = oegraphsim.OEFPAtomType_DefaultAtom,
        bond_type: int = oegraphsim.OEFPBondType_DefaultBond,
        **kwargs: Any,
    ) -> dict[str, list[Fingerprint]]:
        """Generates a tree fingerprint for each input molecule.

        Args:
            samples: The molecules to featurise.
            length: The size of the fingerprint in bits. This number has to be
                larger than or equal to 2^4 and smaller than 2^16.
            diameter: The largest tree fragments (in half-bonds) that are
                enumerated during the fingerprint generation. All enumerated
                tree fragments are hashed into the OEFingerPrint object.
            atom_type: Defines which atom properties are encoded during the
                fingerprint generation. This value has to be either a value or
                a set of bitwise OR`d values from the `OEFPAtomType` namespace.
            bond_type: Defines which bond properties are encoded during the
                fingerprint generation. This value has to be either a value or
                a set of bitwise OR`d values from the `OEFPBondType` namespace.

        Returns:
            Tree fingerprints, as lists of bits.

        Examples:
            >>> from molflux.features import load_representation
            >>> representation = load_representation('tree')
            >>> samples = ['c1ccccc1']
            >>> representation.featurise(samples=samples, length=16)
            {'tree': [[0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0]]}
        """
        assert_n_positional_args(*columns, expected_size=1)
        samples = columns[0]
        if not ((length & (length - 1) == 0) and length != 0):
            raise RuntimeError(f"length: {length} must be a power of 2")

        if (length < 2**4) or (length >= 2**16):
            raise ValueError(
                "The fingerprint size has to be larger than or equal to 2^4 and smaller than 2^16",
            )

        if not ((diameter % 2 == 0) and diameter >= 0):
            raise RuntimeError(f"diameter: {diameter} must be even and >= 0")

        tree_fp_list: list[Fingerprint] = []
        for sample in samples:
            with featurisation_error_harness(sample):
                # patch openeye support for fingerprints of empty SMILES
                if sample == "":
                    bit_vector = [0] * length

                else:
                    mol = to_oemol(sample)
                    fp = oegraphsim.OEFingerPrint()
                    oegraphsim.OEMakeTreeFP(
                        fp,
                        mol,
                        length,
                        0,
                        int(diameter / 2),
                        atom_type,
                        bond_type,
                    )
                    bit_vector = fingerprint_to_bit_vector(fp)

                tree_fp_list.append(bit_vector)

        return {self.tag: tree_fp_list}
