"""Tanimoto similarity splitting strategy.

References:
    https://github.com/deepchem/deepchem/blob/master/deepchem/splits/splitters.py#L1193
"""

import logging
from typing import Any, Iterator, List, Optional, Tuple

import numpy as np

try:
    from rdkit import Chem, DataStructs
    from rdkit.Chem import AllChem
except ImportError as e:
    from molflux.splits.errors import ExtrasDependencyImportError

    raise ExtrasDependencyImportError("rdkit", e) from e

from molflux.splits.bases import SplittingStrategyBase
from molflux.splits.info import SplittingStrategyInfo
from molflux.splits.typing import ArrayLike, SplitIndices, Splittable

logger = logging.getLogger(__name__)

_DESCRIPTION = """
Class for doing data splits based on the Tanimoto similarity between ECFP4 fingerprints.

This class tries to split the data such that the molecules in each dataset are
as different as possible from the ones in the other datasets.  This makes it a
very stringent test of models.  Predicting the test and validation sets may
require extrapolating far outside the training data.

The running time for this splitter scales as O(n^2) in the number of samples.
Splitting large datasets can take a long time.

Note:
    This strategy requires rdkit to be installed.
"""


class Tanimoto(SplittingStrategyBase):
    def _info(self) -> SplittingStrategyInfo:
        return SplittingStrategyInfo(
            description=_DESCRIPTION,
        )

    def _split(
        self,
        dataset: Splittable,
        y: Optional[ArrayLike] = None,
        groups: Optional[ArrayLike] = None,
        n_splits: int = 1,
        train_fraction: float = 0.8,
        validation_fraction: float = 0.1,
        test_fraction: float = 0.1,
        **kwargs: Any,
    ) -> Iterator[SplitIndices]:
        """Splits compounds according to the Tanimoto similarity of their ECFP4 fingerprints.

        This splitting algorithm has an O(N^2) run time, where N is the number
        of molfluxs in the dataset.

        Args:
            dataset: The data to be split.
            y (optional): List of smiles to be used for tanimoto split.
            groups (optional): Group labels for the samples used while splitting the dataset.
            n_splits (optional): The number of splits to generate. Must be greater than one. Defaults to 2.
            train_fraction: The proportion of the dataset to include in the train split.
            validation_fraction: The proportion of the dataset to include in the validation split.
            test_fraction: The proportion of the dataset to include in the test split.

        Yields:
            A tuple of train, validation, and test indices.

        Examples:
            >>> from molflux.splits import load_splitting_strategy
            >>> strategy = load_splitting_strategy('tanimoto_rdkit')
            >>> dataset = ['CCCC', 'CC', 'c1ccncc1']
            >>> folds = strategy.split(dataset=dataset, y=dataset)
        """

        if y is None:
            raise ValueError("""y parameter should be provided for tanimoto splits.""")

        np.testing.assert_almost_equal(
            train_fraction + validation_fraction + test_fraction,
            1.0,
        )

        try:
            mols = [Chem.MolFromSmiles(smiles) for smiles in y]
        except TypeError as e:
            raise TypeError(
                "Tanimoto splitting strategy expects a collection of SMILES as input.",
            ) from e

        fingerprints = [AllChem.GetMorganFingerprintAsBitVect(x, 2, 1024) for x in mols]

        # Split into two groups: training set and everything else.

        train_size = int(train_fraction * len(dataset))
        validation_size = int(validation_fraction * len(dataset))
        test_size = len(dataset) - train_size - validation_size
        train_indices, test_validation_indices = _split_fingerprints(
            fingerprints,
            train_size,
            validation_size + test_size,
        )

        # Split the second group into validation and test sets.

        if validation_size == 0:
            validation_indices = []
            test_indices = test_validation_indices
        elif test_size == 0:
            test_indices = []
            validation_indices = test_validation_indices
        else:
            test_valid_fps = [fingerprints[i] for i in test_validation_indices]
            test_indices, validation_indices = _split_fingerprints(
                test_valid_fps,
                test_size,
                validation_size,
            )
            test_indices = [test_validation_indices[i] for i in test_indices]
            validation_indices = [
                test_validation_indices[i] for i in validation_indices
            ]

        for _ in range(n_splits):
            yield train_indices, validation_indices, test_indices


def _split_fingerprints(
    fps: List,
    size1: int,
    size2: int,
) -> Tuple[List[int], List[int]]:
    """Divides a list of fingerprints into two groups."""

    if len(fps) != size1 + size2:
        raise ValueError(
            f"Incompatible fingerprint splitting sizes for dataset of length {len(fps)}.",
        )

    # Begin by assigning the first molecule to the first group.

    fp_in_group = [[fps[0]], []]
    indices_in_group: Tuple[List[int], List[int]] = ([0], [])
    remaining_fp = fps[1:]
    remaining_indices = list(range(1, len(fps)))
    max_similarity_to_group = [
        DataStructs.BulkTanimotoSimilarity(fps[0], remaining_fp),
        [0] * len(remaining_fp),
    ]
    while len(remaining_fp) > 0:
        # Decide which group to assign a molecule to.

        group = 0 if len(fp_in_group[0]) / size1 <= len(fp_in_group[1]) / size2 else 1

        # Identify the unassigned molecule that is least similar to everything in
        # the other group.

        i = np.argmin(max_similarity_to_group[1 - group])

        # Add it to the group.

        fp = remaining_fp[i]
        fp_in_group[group].append(fp)
        indices_in_group[group].append(remaining_indices[i])

        # Update the data on unassigned molecules.

        similarity = DataStructs.BulkTanimotoSimilarity(fp, remaining_fp)
        max_similarity_to_group[group] = np.delete(
            np.maximum(similarity, max_similarity_to_group[group]),
            i,
        )
        max_similarity_to_group[1 - group] = np.delete(
            max_similarity_to_group[1 - group],
            i,
        )
        del remaining_fp[i]
        del remaining_indices[i]

    return indices_in_group
