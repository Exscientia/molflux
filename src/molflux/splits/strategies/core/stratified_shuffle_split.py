from typing import Any, Iterator, List, Optional

import numpy as np
from sklearn.utils import check_random_state

from molflux.splits.bases import SplittingStrategyBase
from molflux.splits.info import SplittingStrategyInfo
from molflux.splits.typing import ArrayLike, SplitIndices, Splittable
from molflux.splits.utils import partition

_DESCRIPTION = """
Stratified ShuffleSplit.

Provides train/validation/test indices to split data in train/validation/test sets.
This cross-validation object is a merge of StratifiedKFold and
ShuffleSplit, which returns stratified randomized folds. The folds
are made by preserving the percentage of samples for each class.

Note: like the ShuffleSplit strategy, stratified random splits
do not guarantee that all folds will be different, although this is
still very likely for sizeable datasets.

References:
    https://github.com/scikit-learn/scikit-learn/blob/baf828ca126bcb2c0ad813226963621cafe38adb/sklearn/model_selection/_split.py#L1849
"""


class StratifiedShuffleSplit(SplittingStrategyBase):
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
            y: The target variable for supervised learning problems.
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

        References:
            An extension of scikit-learn's StratifiedShuffleSplit source code
            https://github.com/scikit-learn/scikit-learn/blob/364c77e04/sklearn/model_selection/_split.py#L1977
        """
        if y is None:
            raise ValueError("The 'y' parameter should not be None.")

        np.testing.assert_almost_equal(
            train_fraction + validation_fraction + test_fraction,
            1.0,
        )

        train_cutoff, validation_cutoff = partition(
            dataset,
            train_fraction,
            validation_fraction,
        )
        n_train, n_validation, n_test = (
            train_cutoff,
            validation_cutoff - train_cutoff,
            len(dataset) - validation_cutoff,
        )

        y = np.asarray(y)
        if y.ndim == 2:
            # for multi-label y, map each distinct row to a string repr
            # using join because str(row) uses an ellipsis if len(row) > 1000
            y = np.array([" ".join(row.astype("str")) for row in y])

        classes, y_indices = np.unique(y, return_inverse=True)
        n_classes = classes.shape[0]

        class_counts = np.bincount(y_indices)
        if np.min(class_counts) < 2:
            raise ValueError(
                "The least populated class in y has only 1"
                " member, which is too few. The minimum"
                " number of groups for any class cannot"
                " be less than 2.",
            )

        if n_train > 0 and n_train < n_classes:
            raise ValueError(
                "The train_size = %d should be greater or "
                "equal to the number of classes = %d" % (n_train, n_classes),
            )

        if n_validation > 0 and n_validation < n_classes:
            raise ValueError(
                "The validation_size = %d should be greater or "
                "equal to the number of classes = %d" % (n_validation, n_classes),
            )

        if n_test > 0 and n_test < n_classes:
            raise ValueError(
                "The test_size = %d should be greater or "
                "equal to the number of classes = %d" % (n_test, n_classes),
            )

        # Find the sorted list of instances for each class:
        # (np.unique above performs a sort, so code is O(n logn) already)
        class_indices = np.split(
            np.argsort(y_indices, kind="mergesort"),
            np.cumsum(class_counts)[:-1],
        )

        rng = check_random_state(seed)
        for _ in range(n_splits):
            # calculate how many of each class should be in each split:
            # we run it on each split so that if there are ties in the
            # class-counts, we break them anew in each iteration
            n_train_per_class = _approximate_mode(class_counts, n_train, rng)
            class_counts_remaining = class_counts - n_train_per_class
            n_validation_per_class = _approximate_mode(
                class_counts_remaining,
                n_validation,
                rng,
            )

            train: List[int] = []
            validation: List[int] = []
            test: List[int] = []

            for i in range(n_classes):
                permutation = rng.permutation(class_counts[i])
                perm_indices_class_i = class_indices[i].take(permutation, mode="clip")

                class_n_train = n_train_per_class[i]
                class_n_validation = n_validation_per_class[i]

                cutoff_train = class_n_train
                cutoff_validation = class_n_train + class_n_validation

                class_train, class_validation, class_test = np.split(
                    perm_indices_class_i,
                    (cutoff_train, cutoff_validation),
                )

                train.extend(class_train)
                validation.extend(class_validation)
                test.extend(class_test)

            train = rng.permutation(train).tolist()
            validation = rng.permutation(validation).tolist()
            test = rng.permutation(test).tolist()

            yield train, validation, test


def _approximate_mode(
    class_counts: np.ndarray,
    n_draws: int,
    rng: np.random.RandomState,
) -> np.ndarray:
    """Computes approximate mode of multivariate hypergeometric.
    This is an approximation to the mode of the multivariate
    hypergeometric given by class_counts and n_draws.
    It shouldn't be off by more than one.
    It is the mostly likely outcome of drawing n_draws many
    samples from the population given by class_counts.

    References:
        https://github.com/scikit-learn/scikit-learn/blob/baf828ca126bcb2c0ad813226963621cafe38adb/sklearn/utils/__init__.py#L1021
    """

    # this computes a bad approximation to the mode of the
    # multivariate hypergeometric given by class_counts and n_draws
    continuous = class_counts / class_counts.sum() * n_draws
    # floored means we don't overshoot n_samples, but probably undershoot
    floored: np.ndarray = np.floor(continuous)
    # we add samples according to how much "left over" probability
    # they had, until we arrive at n_samples
    need_to_add = int(n_draws - floored.sum())
    if need_to_add > 0:
        remainder = continuous - floored
        values = np.sort(np.unique(remainder))[::-1]
        # add according to remainder, but break ties
        # randomly to avoid biases
        for value in values:
            (inds,) = np.where(remainder == value)
            # if we need_to_add less than what's in inds
            # we draw randomly from them.
            # if we need to add more, we add them all and
            # go to the next value
            add_now = min(len(inds), need_to_add)
            inds = rng.choice(inds, size=add_now, replace=False)
            floored[inds] += 1
            need_to_add -= add_now
            if need_to_add == 0:
                break

    return floored.astype(int)
