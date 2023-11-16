from typing import Any, Iterator, Optional

import numpy as np

from molflux.splits.info import SplittingStrategyInfo
from molflux.splits.strategies.core.shuffle_split import ShuffleSplit
from molflux.splits.typing import ArrayLike, SplitIndices, Splittable

_DESCRIPTION = """
Shuffle-Group(s)-Out cross-validation iterator.

Provides randomized train/validation/test indices to split data according to a
third-party provided group. This group information can be used to encode
arbitrary domain specific stratifications of the samples as integers.

For instance the groups could be the year of collection of the samples
and thus allow for cross-validation against time-based splits.

Note: The parameters ``train_fraction``, ``validation_fraction`` and
``test_fraction`` refer to groups, and not to samples, as in ShuffleSplit.
"""


class GroupShuffleSplit(ShuffleSplit):
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
            train_fraction (optional): The proportion of groups to include in the train split.
                Defaults to 0.8.
            validation_fraction: The proportion of groups to include in the validation split.
                Defaults to 0.1.
            test_fraction: The proportion of groups to include in the test split.
                Defaults to 0.1.
            seed (optional): Controls the shuffling applied to the data before applying
                the split. Pass an int for reproducible output across multiple function calls.
                Defaults to None.
        """

        if groups is None:
            raise ValueError("The 'groups' parameter should not be None.")

        classes, group_indices = np.unique(np.array(groups), return_inverse=True)
        for group_train, group_validation, group_test in super()._split(
            dataset=classes,
            y=None,
            groups=None,
            n_splits=n_splits,
            train_fraction=train_fraction,
            validation_fraction=validation_fraction,
            test_fraction=test_fraction,
            seed=seed,
            **kwargs,
        ):
            # these are the indices of classes in the partition
            # invert them into data indices
            train_indices = np.flatnonzero(
                np.in1d(group_indices, group_train),  # type: ignore[arg-type]
            )
            validation_indices = np.flatnonzero(
                np.in1d(group_indices, group_validation),  # type: ignore[arg-type]
            )
            test_indices = np.flatnonzero(
                np.in1d(group_indices, group_test),  # type: ignore[arg-type]
            )

            yield train_indices, validation_indices, test_indices
