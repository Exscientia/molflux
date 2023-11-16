import numbers
from typing import Any, Union

import numpy as np

from datasets import Dataset

MAX_INT = np.iinfo(np.int32).max
MIN_INT = -1 * MAX_INT - 1


def check_parameter(
    param: Union[int, float],
    low: Union[int, float] = MIN_INT,
    high: Union[int, float] = MAX_INT,
    param_name: str = "",
    include_left: bool = False,
    include_right: bool = False,
) -> bool:
    """Check if an input is within the defined range.
    Parameters
    ----------
    param : int, float
        The input parameter to check.
    low : int, float
        The lower bound of the range.
    high : int, float
        The higher bound of the range.
    param_name : str, optional (default='')
        The name of the parameter.
    include_left : bool, optional (default=False)
        Whether includes the lower bound (lower bound <=).
    include_right : bool, optional (default=False)
        Whether includes the higher bound (<= higher bound).
    Returns
    -------
    within_range : bool or raise errors
        Whether the parameter is within the range of (low, high)
    """

    # param, low and high should all be numerical
    if not isinstance(param, (numbers.Integral, np.integer, float)):
        raise TypeError(f"{param_name} is set to {param} Not numerical")

    if not isinstance(low, (numbers.Integral, np.integer, float)):
        raise TypeError(f"low is set to {low}. Not numerical")

    if not isinstance(high, (numbers.Integral, np.integer, float)):
        raise TypeError(f"high is set to {high}. Not numerical")

    # at least one of the bounds should be specified
    if low is MIN_INT and high is MAX_INT:
        raise ValueError("Neither low nor high bounds is undefined")

    # if wrong bound values are used
    if low > high:
        raise ValueError("Lower bound > Higher bound")

    # value check under different bound conditions
    if (include_left and include_right) and (param < low or param > high):
        raise ValueError(
            f"{param_name} is set to {param}. " f"Not in the range of [{low}, {high}].",
        )

    elif (include_left and not include_right) and (param < low or param >= high):
        raise ValueError(
            f"{param_name} is set to {param}. " f"Not in the range of [{low}, {high}).",
        )

    elif (not include_left and include_right) and (param <= low or param > high):
        raise ValueError(
            f"{param_name} is set to {param}. " f"Not in the range of ({low}, {high}].",
        )

    elif (not include_left and not include_right) and (param <= low or param >= high):
        raise ValueError(
            f"{param_name} is set to {param}. " f"Not in the range of ({low}, {high}).",
        )
    else:
        return True


def list_diff(first_list: list, second_list: list) -> Union[list, set]:
    """Utility function to calculate list difference (first_list-second_list)
    For duplicated values, both duplicates are removed such that
    eg. >>> list_diff([1,2,2],[1,2])
        []
    Parameters
    ----------
    first_list : list
        First list.
    second_list : list
        Second list.
    Returns
    -------
    diff : different molfluxs.
    """
    second_set = set(second_list)
    return [item for item in first_list if item not in second_set]


def get_split_indices(
    dataset: Dataset,
    n_folds: int = 3,
) -> Any:
    """Utility function to get split indices to then splits the data
    for stacking. The data is split
    into n_folds with roughly equal rough size.
    Parameters
    ----------
    dataset: Dataset
    n_folds : int, optional (default=3)
        The number of splits of the training sample.
    Returns
    -------
    index_lists : list of list
        The list of indexes of each fold regarding the returned X and y.
        For instance, index_lists[0] contains the indexes of fold 0.
    """

    if not isinstance(n_folds, int):
        raise ValueError("n_folds must be an integer variable")
    check_parameter(n_folds, low=2, include_left=True, param_name="n_folds")

    idx_length = dataset.shape[0]
    idx_list = list(range(idx_length))

    avg_length = int(idx_length / n_folds)

    index_lists = []
    for i in range(n_folds - 1):
        index_lists.append(idx_list[i * avg_length : (i + 1) * avg_length])

    index_lists.append(idx_list[(n_folds - 1) * avg_length :])

    return index_lists
