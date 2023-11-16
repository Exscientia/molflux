from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from molflux.splits.typing import ArrayLike
from molflux.splits.utils import partition


def validate_ordered_strategy_arguments(
    stratified_bool: bool,
    y: Optional[ArrayLike],
    groups: Optional[ArrayLike],
    train_fraction: Union[float, List[float]],
    gap_train_validation_fraction: Optional[Union[float, List[float]]],
    validation_fraction: Union[float, List[float]],
    gap_validation_test_fraction: Optional[Union[float, List[float]]],
    test_fraction: Union[float, List[float]],
    gap_test_end_fraction: Optional[Union[float, List[float]]],
    stratify_targets_to_ignore: Optional[List],
    min_test_size: Optional[Union[int, List[int]]],
) -> Dict[str, Any]:
    """
    method to validate that all input data is correct.
        checks that groups exists, have > 2 groups, have same type, are comparable
        checks that train, validation, test, gaps, fractions are same type, same length,
            and fills haps with 0.0 if not defined
        checks that fractions add up to 1
        checks that targets exists (if specified) and finds targets to ignore
    """

    # assert that groups is defined
    if groups is None:
        raise ValueError("Need to define 'groups' for ordered splits.")

    # assert that there are more than 2 groups
    assert len(set(groups)) > 2, RuntimeError(
        "Not enough groups for ordered split. Must be > 2.",
    )

    groups = [x if not pd.isnull(x) else None for x in groups]

    # assert that all groups are of same type (or None)
    type_check = list({type(x) for x in groups})
    if type(None) in type_check:
        type_check.remove(type(None))
    assert len(type_check) == 1, TypeError(
        "Groups are not all of the same type (or None, nan, NaT)",
    )

    # assert that groups support comparison
    assert all(
        ((x is None) or (hasattr(x, "__ge__") and hasattr(x, "__le__"))) for x in groups
    ), TypeError("Group molfluxs cannot be compared.")

    assert (
        isinstance(train_fraction, list)
        and isinstance(validation_fraction, list)
        and isinstance(test_fraction, list)
    ) or (
        isinstance(train_fraction, (int, float))
        and isinstance(validation_fraction, (int, float))
        and isinstance(test_fraction, (int, float))
    ), TypeError(
        f"Type of train, validation, test are not the same {[type(train_fraction), type(validation_fraction), type(test_fraction)]}",
    )

    if isinstance(train_fraction, (int, float)):
        train_fraction_list: List = [train_fraction]
    else:
        train_fraction_list = train_fraction
    if isinstance(validation_fraction, (int, float)):
        validation_fraction_list: List = [validation_fraction]
    else:
        validation_fraction_list = validation_fraction
    if isinstance(test_fraction, (int, float)):
        test_fraction_list: List = [test_fraction]
    else:
        test_fraction_list = test_fraction

    if (
        len(
            {
                len(train_fraction_list),
                len(validation_fraction_list),
                len(test_fraction_list),
            },
        )
        != 1
    ):
        raise TypeError(
            f"Lengths of train, validation, test are not the same "
            f"{[len(train_fraction_list), len(validation_fraction_list), len(test_fraction_list)]}",
        )

    if gap_train_validation_fraction is None:
        gap_train_validation_fraction_list: List = [
            0.0 for _ in range(len(train_fraction_list))
        ]
    else:
        if isinstance(gap_train_validation_fraction, float):
            gap_train_validation_fraction_list = [gap_train_validation_fraction]
        else:
            assert isinstance(gap_train_validation_fraction, List)
            gap_train_validation_fraction_list = gap_train_validation_fraction

    if gap_validation_test_fraction is None:
        gap_validation_test_fraction_list: List = [
            0.0 for _ in range(len(train_fraction_list))
        ]
    else:
        if isinstance(gap_validation_test_fraction, float):
            gap_validation_test_fraction_list = [gap_validation_test_fraction]
        else:
            assert isinstance(gap_validation_test_fraction, List)
            gap_validation_test_fraction_list = gap_validation_test_fraction

    if gap_test_end_fraction is None:
        gap_test_end_fraction_list: List = [
            0.0 for _ in range(len(train_fraction_list))
        ]
    else:
        if isinstance(gap_test_end_fraction, float):
            gap_test_end_fraction_list = [gap_test_end_fraction]
        else:
            assert isinstance(gap_test_end_fraction, List)
            gap_test_end_fraction_list = gap_test_end_fraction

    if min_test_size is None:
        min_test_size_list: List = [0 for _ in range(len(test_fraction_list))]
    else:
        if isinstance(min_test_size, int):
            min_test_size_list = [min_test_size]
        else:
            assert isinstance(min_test_size, List)
            min_test_size_list = min_test_size

    for min_test, test_fraction in zip(min_test_size_list, test_fraction_list):
        if test_fraction == 0.0:
            assert min_test == 0, RuntimeError(
                f"Cannot have non-zero min test size {min_test} when test fraction is {test_fraction}.",
            )

    # assert ratios are equal to 1
    for (
        tra_fr,
        g_tr_val_fr,
        val_fr,
        g_val_tes_fr,
        test_fr,
        g_tes_end_fr,
    ) in zip(
        train_fraction_list,
        gap_train_validation_fraction_list,
        validation_fraction_list,
        gap_validation_test_fraction_list,
        test_fraction_list,
        gap_test_end_fraction_list,
    ):
        np.testing.assert_almost_equal(
            sum([tra_fr, g_tr_val_fr, val_fr, g_val_tes_fr, test_fr, g_tes_end_fr]),
            1.0,
        )

    if stratified_bool:
        # assert that targets is defined
        if y is None:
            raise ValueError("Need to define 'targets' for stratified ordered splits.")

        y = [x if not pd.isnull(x) else None for x in y]

        y_expanded = []
        groups_expanded = []
        indices_expanded = []

        # check same length
        if len(list(y)) != len(list(groups)):
            raise RuntimeError("Length of groups is not equal to length of targets")

        for idx, (yy, group) in enumerate(zip(y, groups)):
            if isinstance(yy, str):
                yy_split = [x.strip() for x in yy.split("|")]
            else:
                yy_split = [yy]

            if len(yy_split) > 1:
                y_expanded += yy_split
                groups_expanded += [group] * len(yy_split)
                indices_expanded += [idx] * len(yy_split)
            else:
                y_expanded.append(yy_split[0])
                groups_expanded.append(group)
                indices_expanded.append(idx)

        y = y_expanded
        groups = groups_expanded
        indices = indices_expanded

        # get set of targets
        set_of_targets: set = set(y)

        if None in set_of_targets:
            set_of_targets.remove(None)

        # if need to ignore some targets
        if stratify_targets_to_ignore is not None:
            assert isinstance(stratify_targets_to_ignore, list), TypeError(
                f"'stratify_targets_to_ignore' {stratify_targets_to_ignore} must be a list of targets to stratify by.",
            )

            set_of_targets = set_of_targets - set(stratify_targets_to_ignore)

        assert len(set_of_targets) > 1, RuntimeError(
            "Need more than 1 targets to stratify with.",
        )
        groups_array: np.ndarray = np.array(groups)
        targets_array: np.ndarray = np.array(y)
        index_array: np.ndarray = np.array(indices)

    else:
        set_of_targets = set()
        # define groups array, targets array, index array
        groups_array = np.array(groups)
        targets_array = np.array(y)
        index_array = np.arange(len(list(groups)))

    validated_dict = {
        "train_fraction_list": train_fraction_list,
        "gap_train_validation_fraction_list": gap_train_validation_fraction_list,
        "validation_fraction_list": validation_fraction_list,
        "gap_validation_test_fraction_list": gap_validation_test_fraction_list,
        "test_fraction_list": test_fraction_list,
        "gap_test_end_fraction_list": gap_test_end_fraction_list,
        "set_of_targets": set_of_targets,
        "groups_array": groups_array,
        "targets_array": targets_array,
        "index_array": index_array,
        "min_test_size_list": min_test_size_list,
    }

    return validated_dict


def make_sorted_groups_indices(
    groups_array: np.ndarray,
    index_array: np.ndarray,
    targets_array: Optional[np.ndarray],
    target: Optional[str],
) -> Dict:
    """
    method to find sorted groups and indices with and without None groups
    """
    if (target is not None) and (targets_array is not None):
        # bool mask of this target and groups Not equal to None
        mask_without_nones = (targets_array == target) * (
            groups_array != None  # noqa: E711
        )

        # bool mask of this target and groups equal to None
        mask_of_nones = (targets_array == target) * (groups_array == None)  # noqa: E711

    else:
        # bool mask of groups Not equal to None
        mask_without_nones = groups_array != None  # noqa: E711

        # bool mask of groups equal to None
        mask_of_nones = groups_array == None  # noqa: E711

    # find target indices and indices of Nones
    indices_without_nones = index_array[mask_without_nones]
    indices_of_nones = index_array[mask_of_nones]

    # find target groups
    groups_without_nones = groups_array[mask_without_nones]
    if len(groups_without_nones) > 0:
        groups_without_nones = groups_without_nones.astype(
            type(groups_without_nones[0]),
        )
    groups_of_nones = groups_array[mask_of_nones]

    # find sorted indices based on groups
    sorted_args_without_nones = np.argsort(groups_without_nones, kind="stable")
    sorted_groups_without_nones = groups_without_nones[sorted_args_without_nones]
    sorted_indices_without_nones = indices_without_nones[sorted_args_without_nones]

    # final sorted groups and indices with nones at the beginning
    sorted_groups_with_nones = np.concatenate(
        [groups_of_nones, sorted_groups_without_nones],
    )
    sorted_indices_with_nones = np.concatenate(
        [indices_of_nones, sorted_indices_without_nones],
    )

    sorted_groups_indices_dict: Dict = {
        "sorted_indices_without_nones": sorted_indices_without_nones,
        "sorted_indices_with_nones": sorted_indices_with_nones,
        "sorted_groups_without_nones": sorted_groups_without_nones,
        "sorted_groups_with_nones": sorted_groups_with_nones,
    }

    return sorted_groups_indices_dict


def find_begin_end_indices(
    sorted_indices_without_nones: np.ndarray,
    sorted_indices_with_nones: np.ndarray,
    split_train_fraction: float,
    split_gap_train_val_fraction: float,
    split_validation_fraction: float,
    split_gap_val_test_fraction: float,
    split_test_fraction: float,
    split_gap_test_end_fraction: float,
    undefined_groups_in_train: bool,
) -> Dict:
    """method to find begin, end, indices of different sets"""

    if undefined_groups_in_train:
        # partitioning is performed on the dataset with None-groups ignored...
        indices = sorted_indices_without_nones
    else:
        indices = sorted_indices_with_nones

    train_end, validation_begin, validation_end, test_begin, test_end = partition(
        indices,
        split_train_fraction,
        split_gap_train_val_fraction,
        split_validation_fraction,
        split_gap_val_test_fraction,
        split_test_fraction,
    )

    cutoffs: Dict[str, int] = {
        "train_end": train_end,
        "validation_begin": validation_begin,
        "validation_end": validation_end,
        "test_begin": test_begin,
        "test_end": test_end,
    }

    # ... and then we inject the Nones in the train section
    # (cascading the offset introduced by the extra items)
    if undefined_groups_in_train:
        n_nones = len(sorted_indices_with_nones) - len(sorted_indices_without_nones)
        cutoffs = {k: v + n_nones for k, v in cutoffs.items()}

    return cutoffs


def force_min_test_size(
    min_test_size: int,
    train_end: int,
    validation_begin: int,
    validation_end: int,
    test_begin: int,
    test_end: int,
) -> Dict:
    """
    method to force test set to have min size. Shifts test begin and modifies
    validation and train such that train is used up first, then validation
    """
    if min_test_size > test_end - test_begin:
        old_test_begin = test_begin

        # If min_test_size is bigger that num_indices, everything goes in test set
        if min_test_size > test_end:
            test_begin = 0
        else:
            test_begin = test_end - min_test_size

        # how much to shift indices of other sets
        shift_indices = old_test_begin - test_begin

        # shift sets and set to 0 if outside of range
        validation_end = max(validation_end - shift_indices, 0)
        validation_begin = max(validation_begin - shift_indices, 0)
        train_end = max(train_end - shift_indices, 0)

    bgn_end_dict: Dict = {
        "train_end": train_end,
        "validation_begin": validation_begin,
        "validation_end": validation_end,
        "test_begin": test_begin,
        "test_end": test_end,
    }

    return bgn_end_dict


def force_same_set_same_group(
    sorted_groups_with_nones: np.ndarray,
    terminal_index: int,
) -> int:
    """
    method to force the beginning/end of sets to correspond to whole groups.
    """

    if terminal_index == 0:
        return terminal_index
    elif terminal_index == len(sorted_groups_with_nones):
        return terminal_index
    else:
        # what is the group
        terminal_group = sorted_groups_with_nones[terminal_index - 1]

        # find group indices
        terminal_group_indices = np.where(sorted_groups_with_nones == terminal_group)[0]

        # pick nearest
        terminal_index = min(
            [
                terminal_group_indices[0],
                terminal_group_indices[-1] + 1,
            ],
            key=lambda x: int(abs(x - terminal_index)),
        )

        return terminal_index
