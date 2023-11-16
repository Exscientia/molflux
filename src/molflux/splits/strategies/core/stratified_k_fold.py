from typing import Any, Iterable, Iterator, Optional

import sklearn.model_selection

from molflux.splits.bases import SplittingStrategyBase
from molflux.splits.info import SplittingStrategyInfo
from molflux.splits.typing import ArrayLike, SplitIndices, Splittable

_DESCRIPTION = """
Stratified K-Folds cross-validator.

Provides train/validation indices to split data in train/validation sets.
This cross-validation object is a variation of KFold that returns
stratified folds. The folds are made by preserving the percentage of
samples for each class.
"""


class StratifiedKFold(SplittingStrategyBase):
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
        shuffle: bool = False,
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> Iterator[SplitIndices]:
        """
        Args:
            dataset: The data to be split.
            y: The target variable for supervised learning problems.
            groups (optional): Group labels for the samples used while splitting the dataset.
            n_splits (optional): The number of splits to generate. Must be greater than one. Defaults to 2.
            shuffle (optional): Whether to shuffle the data before splitting into batches.
                Note that the samples within each split will not be shuffled.
                Defaults to False.
            seed (optional): When `shuffle` is True, `seed` affects the ordering of the
                indices, which controls the randomness of each fold. Otherwise, this
                parameter has no effect. Pass an int for reproducible output across multiple function calls.
                Defaults to None.
        """
        if y is None:
            raise ValueError("The 'y' parameter should not be None.")

        # Wrap scikit-learn implementation
        cv = sklearn.model_selection.StratifiedKFold(
            n_splits=n_splits,
            shuffle=shuffle,
            random_state=seed,
        )

        for indices in cv.split(dataset, y=y, groups=groups):
            train_indices, validation_indices = indices
            test_indices: Iterable[int] = []  # no holdout in cross-validation
            yield train_indices, validation_indices, test_indices
