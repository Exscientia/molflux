from typing import Any, Iterable, Iterator, Optional

import sklearn.model_selection

from molflux.splits.bases import SplittingStrategyBase
from molflux.splits.info import SplittingStrategyInfo
from molflux.splits.typing import ArrayLike, SplitIndices, Splittable

_DESCRIPTION = """
Leave One Group Out cross-validator

Provides train/validation indices to split data according to a third-party
provided group. This group information can be used to encode arbitrary
domain specific stratifications of the samples as integers.

For instance the groups could be the year of collection of the samples
and thus allow for cross-validation against time-based splits.

Notes
-----
Splits are ordered according to the index of the group left out. The first
split has validation set consisting of the group whose index in `groups` is
lowest, and so on.
"""


class LeaveOneGroupOut(SplittingStrategyBase):
    def _info(self) -> SplittingStrategyInfo:
        return SplittingStrategyInfo(
            description=_DESCRIPTION,
        )

    def _split(
        self,
        dataset: Splittable,
        y: Optional[ArrayLike] = None,
        groups: Optional[ArrayLike] = None,
        **kwargs: Any,
    ) -> Iterator[SplitIndices]:
        """
        Args:
            dataset: The data to be split.
            groups: Group labels for the samples used while splitting the dataset.

        Examples:
            >>> from molflux.splits import load_splitting_strategy
            >>> strategy = load_splitting_strategy('leave_one_group_out')
            >>> dataset = range(10)
            >>> groups = [1, 1, 2, 2, 2, 3, 3, 3, 3]
            >>> folds = strategy.split(dataset, groups=groups)
        """

        # Wrap scikit-learn implementation
        cv = sklearn.model_selection.LeaveOneGroupOut()

        for indices in cv.split(dataset, y=y, groups=groups):
            train_indices, validation_indices = indices
            test_indices: Iterable[int] = []  # no test in cross-validation
            yield train_indices, validation_indices, test_indices
