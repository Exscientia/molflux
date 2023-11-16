from typing import Any, Iterator, Optional

import numpy as np
from sklearn.utils import check_random_state

from molflux.splits.bases import SplittingStrategyBase
from molflux.splits.info import SplittingStrategyInfo
from molflux.splits.typing import ArrayLike, SplitIndices, Splittable
from molflux.splits.utils import partition

_DESCRIPTION = """
Random permutation cross-validator.
"""


class ShuffleSplit(SplittingStrategyBase):
    def _info(self) -> SplittingStrategyInfo:
        return SplittingStrategyInfo(
            description=_DESCRIPTION,
        )

    def _split(
        self,
        dataset: Splittable,
        y: Optional[ArrayLike] = None,
        groups: Optional[ArrayLike] = None,
        *,
        n_splits: int = 1,
        train_fraction: float = 0.8,
        validation_fraction: float = 0.1,
        test_fraction: float = 0.1,
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> Iterator[SplitIndices]:
        """
        Args:
            dataset: The data to be split.
            y (optional): The target variable for supervised learning problems.
            groups (optional): Group labels for the samples used while splitting the dataset.
            n_splits (optional): The number of splits to generate. Defaults to 1.
            train_fraction (optional): The proportion of the dataset to include in the train split.
                Defaults to 0.8.
            validation_fraction: The proportion of the dataset to include in the validation split.
                Defaults to 0.1.
            test_fraction: The proportion of the dataset to include in the test split.
                Defaults to 0.1.
            seed (optional): Controls the shuffling applied to the data before applying
                the split. Pass an int for reproducible output across multiple function calls.
                Defaults to None.
        """
        np.testing.assert_almost_equal(
            train_fraction + validation_fraction + test_fraction,
            1.0,
        )

        train_cutoff, validation_cutoff = partition(
            dataset,
            train_fraction,
            validation_fraction,
        )
        indices = range(len(dataset))

        rng = check_random_state(seed)
        for _ in range(n_splits):
            permutation = rng.permutation(indices)
            train_indices, validation_indices, test_indices = np.split(
                permutation,
                [train_cutoff, validation_cutoff],
            )
            yield train_indices, validation_indices, test_indices
