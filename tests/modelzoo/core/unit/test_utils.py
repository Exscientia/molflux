from typing import Any

import numpy as np
import pytest
from numpy.testing import assert_array_equal

import datasets
from molflux.modelzoo.utils import (
    get_concatenated_array,
    raise_on_unrecognised_parameters,
)


def test_raise_on_unrecognised_parameters_raises_on_unrecognised_parameters():
    """That an error is raised if passing parameters not defined by the
    function."""

    def my_callable(a: str, b: int = 2, c: bool = False, **kwargs: Any) -> None:
        ...

    kwargs = {"b": 3, "not specified": "any"}
    with pytest.raises(ValueError, match=r"Unknown my_callable parameter\(s\)"):
        raise_on_unrecognised_parameters(my_callable, **kwargs)


def test_raise_on_unrecognised_parameters_does_not_raise_on_recognised_parameters():
    """That an error is not raised if passing parameters defined by the
    function."""

    def my_callable(a: str, b: int = 2, c: bool = False, **kwargs: Any) -> None:
        ...

    kwargs = {"a": "a", "b": 3}
    raise_on_unrecognised_parameters(my_callable, **kwargs)
    assert True


def test_get_concatenated_array_concatenates_nested_arrays():
    """That the function generates a single feature vector out of features with
    nested arrays.
    """
    dataset = datasets.Dataset.from_dict(
        {"a": [1, 2], "b": [[10, 11, 12], [20, 21, 22]]},
    )
    out = get_concatenated_array(dataset, features=["a", "b"])
    expected = np.array([[1, 10, 11, 12], [2, 20, 21, 22]])
    assert_array_equal(expected, out)


def test_get_concatenated_array_preserves_order():
    """That the function preserves the specified order in the output array"""
    dataset = datasets.Dataset.from_dict(
        {"a": [1, 2], "b": [[10, 11, 12], [20, 21, 22]]},
    )
    out = get_concatenated_array(dataset, features=["b", "a"])
    expected = np.array([[10, 11, 12, 1], [20, 21, 22, 2]])
    assert_array_equal(expected, out)


def test_get_concatenated_array_robust_to_empty_dataset():
    """That the function can handle empty input datasets."""
    dataset = datasets.Dataset.from_dict({"a": [], "b": []})
    out = get_concatenated_array(dataset, features=["a", "b"])
    expected = np.array([]).reshape(-1, 1)
    assert_array_equal(expected, out)
