from typing import Any, Dict, List, Optional

try:
    from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect

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
Morgan fingerprint.

These fingerprints are similar to the well-known ECFP or FCFP fingerprints,
depending on which invariants are used. These are implemented based on the
original paper. The algorithm follows the description in the paper as
closely as possible with the exception of the chemical feature definitions used
for the “Feature Morgan” fingerprint - the RDKit implementation uses the
feature types Donor, Acceptor, Aromatic, Halogen, Basic, and Acidic with
definitions adapted from those in [1]_. It is possible to provide your
own atom types. The fingerprints are available as either explicit or sparse
count vectors or explicit bit vectors.

The algorithm used is described in the paper
Rogers, D. & Hahn, M. Extended-Connectivity Fingerprints. JCIM 50:742-54 (2010)
https://doi.org/10.1021/ci100050t

The original implementation was done using this paper:
D. Rogers, R.D. Brown, M. Hahn J. Biomol. Screen. 10:682-6 (2005)
and an unpublished technical report:
http://www.ics.uci.edu/~welling/teaching/ICS274Bspring06/David%20Rogers%20-%20ECFP%20Manuscript.doc

[1]_ https://doi.org/10.1002/(SICI)1097-0290(199824)61:1%3C47::AID-BIT9%3E3.0.CO;2-Z
"""


class Morgan(RepresentationBase):
    def _info(self) -> RepresentationInfo:
        return RepresentationInfo(
            description=_DESCRIPTION,
        )

    def _featurise(
        self,
        samples: MolArray,
        radius: int = 3,
        n_bits: int = 2048,
        invariants: Optional[List[int]] = None,
        from_atoms: Optional[List[int]] = None,
        use_chirality: bool = False,
        use_bond_types: bool = True,
        use_features: bool = False,
        bit_info: Optional[Dict] = None,
        include_redundant_environments: bool = False,
        **kwargs: Any,
    ) -> Dict[str, List[Fingerprint]]:
        """Generates Morgan fingerprints for each input molecule.

        Args:
            samples: The molecules to be fingerprinted.
            radius: The number of iterations to grow the fingerprint.
            n_bits: The size of the fingerprint. Defaults to `2048`.
            invariants: The set of atom invariants to be used. Defaults to
                `None`, which corresponds to ECFP-type invariants.
            from_atoms: If provided, only the atoms in the vector will be used
                as centers in the fingerprint. Defaults to `None`.
            use_chirality: If set, additional information will be added to the
                fingerprint when chiral atoms are discovered, generating
                different fingerprints. Defaults to `False`.
            use_bond_types: If set, bond types will be included as part of the
                hash for calculating bits. Defaults to `True`.
            use_features: Defaults to `False`.
            bit_info: Defaults to `None`.
            include_redundant_environments: If not None, the check for redundant
                atom environments will not be done. Defaults to `False`.

        Returns:
            MACCS fingerprints, as lists of bits.

        Examples:
            >>> from molflux.features import load_representation
            >>> representation = load_representation('morgan')
            >>> samples = ['c1ccccc1']
            >>> representation.featurise(samples, n_bits=16)
            {'morgan': [[1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]}
        """

        morgan_fp_list: List[List] = []
        for sample in samples:
            with featurisation_error_harness(sample):
                smile = to_smiles(sample)
                mol = rdkit_mol_from_smiles(smile)
                rd_fp = GetMorganFingerprintAsBitVect(
                    mol,
                    radius=radius,
                    nBits=n_bits,
                    invariants=invariants or [],
                    fromAtoms=from_atoms or [],
                    useChirality=use_chirality,
                    useBondTypes=use_bond_types,
                    useFeatures=use_features,
                    bitInfo=bit_info or {},
                    includeRedundantEnvironments=include_redundant_environments,
                )

                if rd_fp.GetNumOnBits() == 0:
                    morgan_fp_list.append([0] * rd_fp.GetNumBits())
                else:
                    morgan_fp_list.append(rd_fp.ToList())

        return {self.tag: morgan_fp_list}
