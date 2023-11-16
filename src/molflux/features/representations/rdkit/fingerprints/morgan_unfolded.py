from typing import Any, Dict, List, Optional

try:
    from rdkit.Chem.AllChem import GetMorganFingerprint

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
Raw/Unfolded Morgan fingerprint.

See the description of the `morgan` fingerprint for more information.
"""


class MorganUnfolded(RepresentationBase):
    def _info(self) -> RepresentationInfo:
        return RepresentationInfo(
            description=_DESCRIPTION,
        )

    def _featurise(
        self,
        samples: MolArray,
        radius: int = 3,
        invariants: Optional[List] = None,
        from_atoms: Optional[List] = None,
        use_chirality: bool = False,
        use_bond_types: bool = True,
        use_features: bool = False,
        use_counts: bool = False,
        bit_info: Optional[Dict] = None,
        include_redundant_environments: bool = False,
        **kwargs: Any,
    ) -> Dict[str, List[Dict[str, int]]]:
        """Featurises the input molecules as unfolded Morgan fingerprints.

        Args:
            samples: The molecules to be fingerprinted.
            radius: The number of iterations to grow the fingerprint.
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
            use_counts: If set, counts of the features will be used.
            bit_info: Defaults to `None`.
            include_redundant_environments: If not None, the check for redundant
                atom environments will not be done. Defaults to `False`.

        Returns:
            Dict[str, List[Dict]]
                inputs featurised as unfolded (sparse) fingerprint (returned as a dictionary of non-zero molfluxs)

        Examples:
            >>> from molflux.features import load_representation
            >>> representation = load_representation("morgan_unfolded")
            >>> samples = ["CCCC"]
            >>> representation.featurise(samples)
            {'morgan_unfolded': [{'1173125914': 1, '1244535424': 1, '2245384272': 1, '2246728737': 1, '3542456614': 1}]}

            >>> from molflux.features import load_representation
            >>> representation = load_representation("morgan_unfolded")
            >>> samples = ["CCCC"]
            >>> representation.featurise(samples, use_counts=True)
            {'morgan_unfolded': [{'1173125914': 2, '1244535424': 1, '2245384272': 2, '2246728737': 2, '3542456614': 2}]}

            >>> from molflux.features import load_representation
            >>> representation = load_representation("morgan_unfolded")
            >>> samples = ["CCCC"]
            >>> representation.featurise(samples, use_features=True)
            {'morgan_unfolded': [{'0': 1, '2602795547': 1, '3205495869': 1, '3766532888': 1}]}

            >>> from molflux.features import load_representation
            >>> representation = load_representation("morgan_unfolded")
            >>> samples = ["CCCC"]
            >>> representation.featurise(samples, use_features=True, use_counts=True)
            {'morgan_unfolded': [{'0': 4, '2602795547': 1, '3205495869': 2, '3766532888': 2}]}
        """

        unfolded_morgan_fp_list: List[Dict[str, int]] = []
        for sample in samples:
            with featurisation_error_harness(sample):
                smile = to_smiles(sample)
                mol = rdkit_mol_from_smiles(smile)
                rd_fp = GetMorganFingerprint(
                    mol,
                    radius=radius,
                    invariants=invariants or [],
                    fromAtoms=from_atoms or [],
                    useChirality=use_chirality,
                    useBondTypes=use_bond_types,
                    useFeatures=use_features,
                    useCounts=use_counts,
                    bitInfo=bit_info or {},
                    includeRedundantEnvironments=include_redundant_environments,
                )

                unfolded_morgan_fp_list.append(
                    {str(k): v for k, v in rd_fp.GetNonzeromolfluxs().items()},
                )

        return {self.tag: unfolded_morgan_fp_list}
