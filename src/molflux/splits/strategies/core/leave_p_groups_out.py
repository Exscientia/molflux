from typing import Any, Iterable, Iterator, Optional

import sklearn.model_selection

from molflux.splits.bases import SplittingStrategyBase
from molflux.splits.info import SplittingStrategyInfo
from molflux.splits.typing import ArrayLike, SplitIndices, Splittable

_DESCRIPTION = """
Leave P Group(s) Out cross-validator

Provides train/validation indices to split data according to a third-party
provided group. This group information can be used to encode arbitrary
domain specific stratifications of the samples as integers.

For instance the groups could be the year of collection of the samples
and thus allow for cross-validation against time-based splits.

The difference between leave_p_groups_out and leave_one_group_out is that
the former builds the validation sets with all the samples assigned to
``p`` different values of the groups while the latter uses samples
all assigned the same groups.
"""


class LeavePGroupsOut(SplittingStrategyBase):
    def _info(self) -> SplittingStrategyInfo:
        return SplittingStrategyInfo(
            description=_DESCRIPTION,
        )

    def _split(
        self,
        dataset: Splittable,
        y: Optional[ArrayLike] = None,
        groups: Optional[ArrayLike] = None,
        p: Optional[int] = None,
        **kwargs: Any,
    ) -> Iterator[SplitIndices]:
        """
        Args:
            dataset: The data to be split.
            groups: Group labels for the samples used while splitting the dataset.
            p: Number of groups (``p``) to leave out in the validation split.

        Examples:
            >>> from molflux.splits import load_splitting_strategy
            >>> strategy = load_splitting_strategy('leave_p_groups_out')
            >>> dataset = range(10)
            >>> groups = [1, 1, 2, 2, 2, 3, 3, 3, 3]
            >>> folds = strategy.split(dataset, groups=groups, p=1)
        """
        if p is None:
            raise ValueError("The 'p' parameter should not be None.")

        if p == 0:
            raise ValueError("The 'p' parameter should not be 0.")

        # Wrap scikit-learn implementation
        cv = sklearn.model_selection.LeavePGroupsOut(n_groups=p)

        for indices in cv.split(dataset, y=y, groups=groups):
            train_indices, validation_indices = indices
            test_indices: Iterable[int] = []  # no test in cross-validation
            yield train_indices, validation_indices, test_indices
