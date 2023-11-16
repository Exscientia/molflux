from typing import List

import pandas as pd
import pytest

from molflux.modelzoo.models.ensemble._combo.utils import (
    check_parameter,
    get_split_indices,
    list_diff,
)

_X_FEATURES = ["X_col_1", "X_col_2"]
_Y_FEATURES = ["y_col"]

train_df = pd.DataFrame(
    [
        [1, 2],
        [1, 3],
        [2, 4],
        [2, 4],
        [2, 5],
    ],
    columns=_X_FEATURES,
)


def test_list_diff_example():
    """test that list diff works"""
    actual = list_diff([1, 2, -9, -3], [1, -3])
    expected = [2, -9]
    assert len(actual) == len(expected)
    assert all(a == b for a, b in zip(actual, expected))


def test_list_diff_duplicates():
    """test that list diff removes duplicates properly"""
    actual = list_diff([1, 2, 2], [1, 2])
    expected: List[int] = []
    assert len(actual) == len(expected)
    assert all(a == b for a, b in zip(actual, expected))


def test_get_split_indices():
    """test that utility function gets split indices with right number of folds and containing all the indices"""
    train_dataset = train_df
    n_folds = 4
    actual = get_split_indices(train_dataset, n_folds=n_folds)
    flat_actual_list = [item for sublist in actual for item in sublist]
    expected = list(range(train_dataset.shape[0]))
    assert len(actual) == n_folds
    assert len(flat_actual_list) == train_dataset.shape[0]
    assert all(a == b for a, b in zip(flat_actual_list, expected))


def test_check_parameter():
    actual = check_parameter(param=5.1, low=4, high=8)
    expected = True
    assert actual == expected
    with pytest.raises(ValueError):
        check_parameter(param=-12, low=0, high=20, param_name="alpha")
