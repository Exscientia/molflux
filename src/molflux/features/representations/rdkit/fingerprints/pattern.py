import logging
from typing import Any, Dict, List, Optional

try:
    from rdkit.Chem.rdmolops import PatternFingerprint
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

logger = logging.getLogger(__name__)

_DESCRIPTION = """
A fingerprint using SMARTS patterns.

Topological fingerprint for a molecule are generated using a series of
pre-defined structural patterns.

These fingerprints were designed to be used in substructure screening. These
are, as far as we know, unique to the RDKit. The algorithm identifies features
in the molecule by doing substructure searches using a small number
(12 in the 2019.03 release of the RDKit) of very generic SMARTS patterns -
like `[*]~[*]~[*](~[*])~[*]` or `[R]~1[R]~[R]~[R]~1`, and then hashing each
occurrence of a pattern based on the atom and bond types involved. The fact
that particular pattern matched the molecule at all is also stored by hashing t
he pattern ID and size. If a particular feature contains either a query atom or
a query bond (e.g. something generated from SMARTS), the only information that
is hashed is the fact that the generic pattern matched.

For the 2019.03 release, the atom types use just the atomic number of the atom
and the bond types use the bond type, or AROMATIC for aromatic bonds).

NOTE: Because it plays an important role in substructure screenout, the
internals of this fingerprint (the generic patterns used and/or the details of
the hashing algorithm) may change from one release to the next.
"""


class Pattern(RepresentationBase):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        logging.warning(
            "Pattern fingerprints are experimental and may change with rdkit updates.",
        )

    def _info(self) -> RepresentationInfo:
        return RepresentationInfo(
            description=_DESCRIPTION,
        )

    def _featurise(
        self,
        samples: MolArray,
        fp_size: int = 2048,
        atom_counts: Optional[List[int]] = None,
        set_only_bits: Optional[ExplicitBitVect] = None,
        tautomer_fingerprints: bool = False,
        **kwargs: Any,
    ) -> Dict[str, List[Fingerprint]]:
        """Generates topological fingerprints for each input molecule using a
        series of pre-defined structural patterns.

        Args:
            samples: The molecules to be fingerprinted.
            fp_size: The size of the fingerprint. Defaults to `2048`
            atom_counts: If provided, this will be used to provide the count
                of the number of paths that set bits each atom is involved in.
                The vector should have at least as many entries as the molecule
                has atoms and is not zeroed out here.
            set_only_bits: If provided, only bits that are set in this bit
                vector will be set in the result. This is essentially the same
                as doing: `(*res) &= (*setOnlyBits)`; but also has an impact on
                the `atom_counts` (if being used). Defaults to `None`.
            tautomer_fingerprints: Defaults to `False`.

        Returns:
            Pattern fingerprints, as lists of bits.

        Examples:
            >>> from molflux.features import load_representation
            >>> representation = load_representation('pattern')
            >>> samples = ['c1ccccc1']
            >>> representation.featurise(samples, fp_size=16)  # doctest: +SKIP
            {'pattern': [[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]}
        """

        pattern_fp_list: List[List] = []
        for sample in samples:
            with featurisation_error_harness(sample):
                smile = to_smiles(sample)
                mol = rdkit_mol_from_smiles(smile)
                rd_fp = PatternFingerprint(
                    mol,
                    fpSize=fp_size,
                    atomCounts=atom_counts or [],
                    setOnlyBits=set_only_bits,
                    tautomerFingerprints=tautomer_fingerprints,
                )

                if rd_fp.GetNumOnBits() == 0:
                    pattern_fp_list.append([0] * rd_fp.GetNumBits())
                else:
                    pattern_fp_list.append(rd_fp.ToList())

        return {self.tag: pattern_fp_list}
