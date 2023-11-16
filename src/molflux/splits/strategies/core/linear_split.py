from typing import Any, Iterator, Optional

import numpy as np

from molflux.splits.bases import SplittingStrategyBase
from molflux.splits.info import SplittingStrategyInfo
from molflux.splits.typing import ArrayLike, SplitIndices, Splittable
from molflux.splits.utils import partition

_DESCRIPTION = """
Deterministic linear cross-validator.
"""


class LinearSplit(SplittingStrategyBase):
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
        indices = np.array(range(len(dataset)))

        train_indices, validation_indices, test_indices = np.split(
            indices,
            [train_cutoff, validation_cutoff],
        )

        for _ in range(n_splits):
            yield train_indices, validation_indices, test_indices
