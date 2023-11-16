from typing import Any, Iterable, Iterator, Optional

import sklearn.model_selection

from molflux.splits.bases import SplittingStrategyBase
from molflux.splits.info import SplittingStrategyInfo
from molflux.splits.typing import ArrayLike, SplitIndices, Splittable

_DESCRIPTION = """
Time Series cross-validator

Provides train/validation indices to split time series data samples
that are observed at fixed time intervals, in train/validation sets.
In each split, test indices must be higher than before, and thus shuffling
in cross validator is inappropriate.

This cross-validation object is a variation of :class:`KFold`.
In the kth split, it returns first k folds as train set and the
(k+1)th fold as test set.

Note that unlike standard cross-validation methods, successive
training sets are supersets of those that come before them.
"""


class TimeSeriesSplit(SplittingStrategyBase):
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
        n_splits: int = 2,
        max_train_size: Optional[int] = None,
        test_size: Optional[int] = None,
        gap: Optional[int] = 0,
        **kwargs: Any,
    ) -> Iterator[SplitIndices]:
        """
        Args:
            dataset: The data to be split.
            y (optional): The target variable for supervised learning problems.
            groups (optional): Group labels for the samples used while splitting the dataset.
            n_splits (optional): The number of splits to generate. Must be greater than one. Defaults to 2.
            max_train_size (optional): The maximum size for a single training set.
            test_size (optional): Used to limit the size of the test set. Defaults to
                ``n_samples // (n_splits + 1)``, which is the maximum allowed value
                with ``gap=0``.
            gap (optional): Number of samples to exclude from the end of each train set
                before the test set.
        """
        # Wrap scikit-learn implementation
        cv = sklearn.model_selection.TimeSeriesSplit(
            n_splits=n_splits,
            max_train_size=max_train_size,
            test_size=test_size,
            gap=gap,
        )

        for indices in cv.split(dataset, y=y, groups=groups):
            train_indices, validation_indices = indices
            test_indices: Iterable[int] = []  # no holdout in cross-validation
            yield train_indices, validation_indices, test_indices
