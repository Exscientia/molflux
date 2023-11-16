from typing import Any, Iterable, Iterator, Optional

import sklearn.model_selection

from molflux.splits.bases import SplittingStrategyBase
from molflux.splits.info import SplittingStrategyInfo
from molflux.splits.typing import ArrayLike, SplitIndices, Splittable

_DESCRIPTION = """
K-fold iterator variant with non-overlapping groups.

The same group will not appear in two different folds (the number of
distinct groups has to be at least equal to the number of folds).
The folds are approximately balanced in the sense that the number of
distinct groups is approximately the same in each fold.
"""


class GroupKFold(SplittingStrategyBase):
    def _info(self) -> SplittingStrategyInfo:
        return SplittingStrategyInfo(
            description=_DESCRIPTION,
        )

    def _split(
        self,
        dataset: Splittable,
        y: Optional[ArrayLike] = None,
        groups: Optional[ArrayLike] = None,
        n_splits: int = 2,
        **kwargs: Any,
    ) -> Iterator[SplitIndices]:
        """
        Args:
            dataset: The data to be split.
            y (optional): The target variable for supervised learning problems.
            groups: Group labels for the samples used while splitting the dataset.
            n_splits (optional): The number of splits to generate. Must be greater than one.
                Defaults to 2.
        """
        # Wrap scikit-learn implementation
        cv = sklearn.model_selection.GroupKFold(
            n_splits=n_splits,
        )

        for indices in cv.split(dataset, y=y, groups=groups):
            train_indices, validation_indices = indices
            test_indices: Iterable[int] = []  # no holdout in cross-validation
            yield train_indices, validation_indices, test_indices
