from typing import Any, Iterator, List, Optional, Union

from molflux.splits.bases import SplittingStrategyBase
from molflux.splits.info import SplittingStrategyInfo
from molflux.splits.strategies.core._ordered_split_utils import (
    find_begin_end_indices,
    force_min_test_size,
    force_same_set_same_group,
    make_sorted_groups_indices,
    validate_ordered_strategy_arguments,
)
from molflux.splits.typing import ArrayLike, SplitIndices, Splittable

_DESCRIPTION = """
A split based on the ordering of rows defined by the group column. This can be used for any sort of temporal splits.
Effectively, this is a sorting of the dataset by the groups column, and then a linear split.
"""


class OrderedSplit(SplittingStrategyBase):
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
        train_fraction: Union[float, List[float]] = 0.8,
        gap_train_validation_fraction: Optional[Union[float, List[float]]] = None,
        validation_fraction: Union[float, List[float]] = 0.1,
        gap_validation_test_fraction: Optional[Union[float, List[float]]] = None,
        test_fraction: Union[float, List[float]] = 0.1,
        gap_test_end_fraction: Optional[Union[float, List[float]]] = None,
        undefined_groups_in_train: bool = True,
        min_test_size: Optional[Union[int, List[int]]] = None,
        same_group_same_set: bool = True,
        **kwargs: Any,
    ) -> Iterator[SplitIndices]:
        """
        Args:
            dataset: The data to be split.
            groups: Group labels for the samples used while splitting the dataset.
            train_fraction (float or list): The proportion of the dataset to include in the train split.
                If a list is specified, split returns multiple folds with the specified fraction.
                Might not always be possible to get the exact fraction.
                Defaults to 0.8.
            gap_train_validation_fraction (Optional): Fraction of data to leave out between train and validation.
                Must be a float or a list, same as the train, validation fractions.
                Defaults to None
            validation_fraction: The proportion of the dataset to include in the validation split.
                If a list is specified, split returns multiple folds with the specified fraction.
                Might not always be possible to get the exact fraction.
                Defaults to 0.1.
            gap_validation_test_fraction (Optional): Fraction of data to leave out between validation and test.
                Must be a float or a list, same as the validation, test fractions.
                Defaults to None
            test_fraction: The proportion of the dataset to include in the test split.
                If a list is specified, split returns multiple folds with the specified fraction.
                Might not always be possible to get the exact fraction.
                Defaults to 0.1.
            gap_test_end_fraction (Optional): Fraction of data to leave out after test ste.
                Must be a float or a list, same as the test fractions.
                Defaults to None
            undefined_groups_in_train: If True, any row with no group (None) will be added to the train set (split ratio
                might be different from specified). Defaults to True.
            min_test_size: The minimum number of datapoints in the test set. Overrides the ratios.
                Can be an int for all the folds, or a list of ints with the same length as the folds. Defaults to None.
            same_group_same_set: Forces datapoints with the same group to be in the same set. Defaults to True.

            Examples:
            >>> from molflux.splits import load_splitting_strategy
            >>> strategy = load_splitting_strategy("ordered_split")
            >>>
            >>> x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            >>> groups = [0, 0, 0, 2, 2, 3, 1, 1, 1, 1]
            >>>
            >>> next(strategy.split(x, groups=groups))
            ([0, 1, 2, 6, 7, 8, 9], [3, 4], [5])
        """

        # verify that all arguments are valid (and massage them into standard form
        validated_dict = validate_ordered_strategy_arguments(
            groups=groups,
            train_fraction=train_fraction,
            gap_train_validation_fraction=gap_train_validation_fraction,
            validation_fraction=validation_fraction,
            gap_validation_test_fraction=gap_validation_test_fraction,
            test_fraction=test_fraction,
            gap_test_end_fraction=gap_test_end_fraction,
            stratified_bool=False,
            y=None,
            stratify_targets_to_ignore=None,
            min_test_size=min_test_size,
        )

        train_fraction_list = validated_dict["train_fraction_list"]
        gap_train_validation_fraction_list = validated_dict[
            "gap_train_validation_fraction_list"
        ]
        validation_fraction_list = validated_dict["validation_fraction_list"]
        gap_validation_test_fraction_list = validated_dict[
            "gap_validation_test_fraction_list"
        ]
        test_fraction_list = validated_dict["test_fraction_list"]
        gap_test_end_fraction_list = validated_dict["gap_test_end_fraction_list"]
        groups_array = validated_dict["groups_array"]
        index_array = validated_dict["index_array"]
        min_test_size_list = validated_dict["min_test_size_list"]

        # iterate over number of folds
        for ii in range(len(train_fraction_list)):
            # fractions for this split
            split_train_fraction = train_fraction_list[ii]
            split_gap_train_val_fraction = gap_train_validation_fraction_list[ii]
            split_validation_fraction = validation_fraction_list[ii]
            split_gap_val_test_fraction = gap_validation_test_fraction_list[ii]
            split_test_fraction = test_fraction_list[ii]
            split_gap_test_end_fraction = gap_test_end_fraction_list[ii]
            split_min_test_size_list = min_test_size_list[ii]

            sorted_groups_indices_dict = make_sorted_groups_indices(
                groups_array=groups_array,
                index_array=index_array,
                targets_array=None,
                target=None,
            )
            sorted_groups_with_nones = sorted_groups_indices_dict[
                "sorted_groups_with_nones"
            ]
            sorted_indices_without_nones = sorted_groups_indices_dict[
                "sorted_indices_without_nones"
            ]
            sorted_indices_with_nones = sorted_groups_indices_dict[
                "sorted_indices_with_nones"
            ]

            begin_end_dict = find_begin_end_indices(
                sorted_indices_without_nones=sorted_indices_without_nones,
                sorted_indices_with_nones=sorted_indices_with_nones,
                split_train_fraction=split_train_fraction,
                split_gap_train_val_fraction=split_gap_train_val_fraction,
                split_validation_fraction=split_validation_fraction,
                split_gap_val_test_fraction=split_gap_val_test_fraction,
                split_test_fraction=split_test_fraction,
                split_gap_test_end_fraction=split_gap_test_end_fraction,
                undefined_groups_in_train=undefined_groups_in_train,
            )
            train_end = begin_end_dict["train_end"]
            validation_begin = begin_end_dict["validation_begin"]
            validation_end = begin_end_dict["validation_end"]
            test_begin = begin_end_dict["test_begin"]
            test_end = begin_end_dict["test_end"]

            # set required min test size (is specified and test_size is not enough)
            if split_min_test_size_list is not None:
                begin_end_dict = force_min_test_size(
                    min_test_size=split_min_test_size_list,
                    train_end=train_end,
                    validation_begin=validation_begin,
                    validation_end=validation_end,
                    test_begin=test_begin,
                    test_end=test_end,
                )
                train_end = begin_end_dict["train_end"]
                validation_begin = begin_end_dict["validation_begin"]
                validation_end = begin_end_dict["validation_end"]
                test_begin = begin_end_dict["test_begin"]
                test_end = begin_end_dict["test_end"]

            # if same group in same set
            if same_group_same_set:
                # change sets to the nearest whole group
                train_end = force_same_set_same_group(
                    sorted_groups_with_nones,
                    train_end,
                )
                validation_begin = force_same_set_same_group(
                    sorted_groups_with_nones,
                    validation_begin,
                )
                validation_end = force_same_set_same_group(
                    sorted_groups_with_nones,
                    validation_end,
                )
                test_begin = force_same_set_same_group(
                    sorted_groups_with_nones,
                    test_begin,
                )
                test_end = force_same_set_same_group(sorted_groups_with_nones, test_end)

            train_indices = sorted_indices_with_nones[:train_end].tolist()
            validation_indices = sorted_indices_with_nones[
                validation_begin:validation_end
            ].tolist()
            test_indices = sorted_indices_with_nones[test_begin:test_end].tolist()

            yield train_indices, validation_indices, test_indices
