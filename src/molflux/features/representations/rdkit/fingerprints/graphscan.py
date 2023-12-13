from __future__ import annotations

import os
import re
from collections import Counter, defaultdict
from itertools import combinations_with_replacement
from typing import TYPE_CHECKING, Any

from molflux.features.bases import RepresentationBase
from molflux.features.info import RepresentationInfo
from molflux.features.representations.rdkit._utils import (
    rdkit_mol_from_smiles,
    to_smiles,
)
from molflux.features.utils import featurisation_error_harness

if TYPE_CHECKING:
    from molflux.features.typing import MolArray

try:
    from rdkit import Chem, RDConfig
    from rdkit.Chem import ChemicalFeatures

    fdefName = os.path.join(RDConfig.RDDataDir, "BaseFeatures.fdef")
    factory = ChemicalFeatures.BuildFeatureFactory(fdefName)

except ImportError as e:
    from molflux.features.errors import ExtrasDependencyImportError

    raise ExtrasDependencyImportError("rdkit", e) from None

_PHARMACOPHORE_TYPES = [  # from BaseFeatures.fdef
    "Acceptor",
    "Aromatic",
    "Donor",
    "Hydrophobe",
    "NegIonizable",
    "PosIonizable",
]

_DESCRIPTION = """
Python implementation of graph-based signature using the cutoff scanning matrix
method of Douglas E Pires (J Med Chem. 2015 May 14; 58(9): 4066-4072).
Computes the cumulative distribution of shortest
paths of lengths 1...N (default 6) between pharmacophore atom pairs.
"""


class Graphscan(RepresentationBase):
    def _info(self) -> RepresentationInfo:
        return RepresentationInfo(
            description=_DESCRIPTION,
        )

    def _featurise(
        self,
        samples: MolArray,
        min_dist: int = 1,
        max_dist: int = 6,
        **kwargs: Any,
    ) -> dict[str, list[list[int]]]:
        """
        Calculates cumulative distribution of shortest paths in molecule adjacency matrix.
        Distribution calculated between feature pairs for path lengths between min_dist and max_dist.

        Args:
            samples: The molecules for which to calculate descriptors.
            min_dist: Minimum shortest path length to count.
            max_dist: Maximum shortest path length to count.

        Returns:
            Features of vectors of cumulative path counts for each feature pair,
            at each path length.

        Examples:
            >>> from molflux.features import load_representation
            >>> representation = load_representation("graphscan")
            >>> samples = ["CCOC1=CC2=C(C=C1)N=C(S2)S(=O)(=O)N"]
            >>> representation.featurise(samples, min_dist=1, max_dist=2)
            {'graphscan::1': [[0, 2, 0, 1, 0, 0, 10, 0, 6, 0, 0, 0, 0, 0, 0, 6, 0, 0, 0, 0, 0]], 'graphscan::2': [[1, 5, 2, 3, 0, 0, 23, 1, 12, 0, 0, 0, 0, 0, 0, 12, 0, 0, 0, 0, 0]]}
        """
        if min_dist > max_dist:
            raise ValueError("must have min_dist <= max_dist") from None

        # build a list with all unique pairs of features, of the form:
        #     feature_1_name, feature_1_name
        #     feature_1_name, feature_2_name
        #     ...
        #     feature_1_name, feature_N_name
        #     feature_2_name, feature_2_name
        #     ...
        #     feature_N_name, feature_N_name
        #
        pharmacophore_combinations = list(
            combinations_with_replacement(_PHARMACOPHORE_TYPES, r=2),
        )

        distances = list(range(min_dist, max_dist + 1))
        descriptor_dict: dict[str, list[list[int]]] = {
            f"{self.tag}::{d}": [] for d in distances
        }
        for sample in samples:
            with featurisation_error_harness(sample):
                smiles = to_smiles(sample)
                mol = rdkit_mol_from_smiles(smiles)
                distance_matrix = Chem.GetDistanceMatrix(mol)
                num_heavy_atoms = mol.GetNumHeavyAtoms()
                feature_set = factory.GetFeaturesForMol(mol)

                # create a dictionary mapping features to corresponding atom indices in this molecule
                #     feature_1: [atom_i_1, atom_i_2, ...],
                #     feature_1: [atom_i_1, atom_i_2, ...],
                #     ...
                #     feature_N: [atom_1, atom_2, ... atom_i_1],
                feature_dict: dict[str, Any] = defaultdict(list)
                for feature in feature_set:
                    feature_family = feature.GetFamily()

                    # Count 'LumpedX' family to be of family 'X'
                    feature_family = re.sub("Lumped", "", feature_family)

                    if feature_family in _PHARMACOPHORE_TYPES:
                        for atomIdx in feature.GetAtomIds():
                            if atomIdx not in feature_dict[feature_family]:
                                feature_dict[feature_family].append(atomIdx)

                # for each pair of atoms within min_dist and max_dist, increment the count_dict
                # for each pair of features they match
                for d in distances:
                    # create a counter to collect pharmacophore counts per smiles per distance
                    #     feature_1_name:feature_1_name: 0
                    #     feature_1_name:feature_2_name: 0
                    #     ...
                    #     feature_N_name:feature_N_name: 0
                    #
                    # This will define the positional meaning of individual counts in the
                    # final output vectors
                    counter = Counter(
                        {
                            f"{features[0]}:{features[1]}": 0
                            for features in pharmacophore_combinations
                        },
                    )

                    for i in range(num_heavy_atoms):
                        for j in range(i + 1, num_heavy_atoms):
                            if min_dist <= distance_matrix[i, j] <= d:
                                i_keys = [k for k, v in feature_dict.items() if i in v]
                                j_keys = [k for k, v in feature_dict.items() if j in v]
                                for k_i in i_keys:
                                    for k_j in j_keys:
                                        if f"{k_i}:{k_j}" in counter:
                                            counter[f"{k_i}:{k_j}"] += 1

                    descriptor_dict[f"{self.tag}::{d}"].append(list(counter.values()))

        return descriptor_dict
