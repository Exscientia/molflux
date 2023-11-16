import numpy as np
import pandas as pd
import pytest

from molflux.splits.catalogue import list_splitting_strategies
from molflux.splits.load import load_splitting_strategy
from molflux.splits.strategy import SplittingStrategy

strategy_name = "stratified_ordered_split"


@pytest.fixture(scope="module")
def fixture_test_strategy():
    return load_splitting_strategy(strategy_name)


@pytest.fixture(scope="module")
def fixture_sample_inputs():
    n = 100
    dataset = np.random.rand(n)
    groups = np.concatenate([np.arange(n - 20), np.array([None] * 20)])
    y = np.concatenate(
        [
            np.ones((n - 24) // 4),
            2 * np.ones((n - 24) // 4),
            3 * np.ones((n - 24) // 4),
            4 * np.ones((n - 24) // 4),
            np.array([None] * 24),
        ],
    )
    np.random.shuffle(groups)
    return dataset, groups, y


def test_is_in_catalogue():
    """That the strategy is registered in the catalogue."""
    catalogue = list_splitting_strategies()
    all_strategy_names = [name for names in catalogue.values() for name in names]
    assert strategy_name in all_strategy_names


def test_implements_protocol(fixture_test_strategy):
    """That the strategy implements the protocol."""
    strategy = fixture_test_strategy
    assert isinstance(strategy, SplittingStrategy)


def test_requires_groups_and_targets(fixture_test_strategy):
    """That you need to provide groups and targets to use this strategy."""
    dataset = [0, 1, 2, 3, 4, 5]
    strategy = fixture_test_strategy
    with pytest.raises(ValueError):
        next(strategy.split(dataset))

    with pytest.raises(ValueError):
        next(strategy.split(dataset, groups=[0, 1, 3] * len(dataset)))

    with pytest.raises(RuntimeError):
        next(strategy.split(dataset, groups=[1, 2, 3] * len(dataset), y=[0]))


def test_hard_coded_example(fixture_test_strategy):
    """Test to illustrate the split"""
    dataset = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    groups = [2, 7, 5, 6, 3, 1, 0, 8, 9, 4]
    y = ["a", "a", "a", "a", "a", "b", "b", "b", "b", "b"]

    strategy = fixture_test_strategy
    indices = strategy.split(
        dataset=dataset,
        groups=groups,
        y=y,
        train_fraction=0.6,
        validation_fraction=0.2,
        test_fraction=0.2,
        same_group_same_set=False,
    )
    train_indices, validation_indices, test_indices = next(indices)

    assert sorted(train_indices) == [0, 2, 4, 5, 6, 9]
    assert sorted(validation_indices) == [3, 7]
    assert sorted(test_indices) == [1, 8]


def test_example_with_nones(fixture_test_strategy):
    """Test to illustrate the split when there are nones:
    Nones will be added to train and ignored when computing fractions
    """

    dataset = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    groups = [2, 7, 5, None, 3, 1, 0, None, 9, 4]
    y = ["a", "a", "a", "a", "a", "b", "b", "b", "b", "b"]

    strategy = fixture_test_strategy
    indices = strategy.split(
        dataset=dataset,
        groups=groups,
        y=y,
        train_fraction=0.6,
        validation_fraction=0.2,
        test_fraction=0.2,
        same_group_same_set=False,
    )
    train_indices, validation_indices, test_indices = next(indices)

    assert sorted(train_indices) == [0, 3, 4, 5, 6, 7]
    assert sorted(validation_indices) == [2, 9]
    assert sorted(test_indices) == [1, 8]


def test_example_with_nulls_nones(fixture_test_strategy):
    """Test to illustrate the split when there are nones or other null values:
    Nones will be added to train and ignored when computing fractions
    """

    dataset = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    groups = [2, 7, 5, pd.NaT, 3, 1, 0, np.nan, 9, 4]
    y = ["a", "a", "a", "a", "a", "b", "b", "b", "b", "b"]

    strategy = fixture_test_strategy
    indices = strategy.split(
        dataset=dataset,
        groups=groups,
        y=y,
        train_fraction=0.6,
        validation_fraction=0.2,
        test_fraction=0.2,
        same_group_same_set=False,
    )
    train_indices, validation_indices, test_indices = next(indices)

    assert sorted(train_indices) == [0, 3, 4, 5, 6, 7]
    assert sorted(validation_indices) == [2, 9]
    assert sorted(test_indices) == [1, 8]


def test_example_with_non_unique_groups(fixture_test_strategy):
    """Test to illustrate the split when there are non-unique groups."""

    dataset = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    groups = [0, 0, 0, 0, 1, 1, 1, 2, 2, 3]
    y = ["a", "a", "a", "a", "a", "b", "b", "b", "b", "b"]

    strategy = fixture_test_strategy
    indices = strategy.split(
        dataset=dataset,
        groups=groups,
        y=y,
        train_fraction=0.6,
        validation_fraction=0.2,
        test_fraction=0.2,
        same_group_same_set=False,
    )
    train_indices, validation_indices, test_indices = next(indices)

    assert sorted(train_indices) == [0, 1, 2, 5, 6, 7]
    assert sorted(validation_indices) == [3, 8]
    assert sorted(test_indices) == [4, 9]


def test_example_with_unusual_ratios(fixture_test_strategy):
    """Test to illustrate the split with unusual ratios"""

    dataset = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    groups = [2, 7, 5, None, 3, 1, 0, None, 9, 4]
    y = ["a", "a", "a", "a", "a", "b", "b", "b", "b", "b"]

    strategy = fixture_test_strategy
    indices = strategy.split(
        dataset=dataset,
        groups=groups,
        y=y,
        train_fraction=0.2,
        validation_fraction=0.6,
        test_fraction=0.2,
        undefined_groups_in_train=False,
        same_group_same_set=False,
    )
    train_indices, validation_indices, test_indices = next(indices)

    assert sorted(train_indices) == [3, 7]
    assert sorted(validation_indices) == [0, 2, 4, 5, 6, 9]
    assert sorted(test_indices) == [1, 8]


def test_example_same_test_ratios_but_different_train_val(
    fixture_sample_inputs,
    fixture_test_strategy,
):
    """Test to illustrate that the same test ratio with different train val results
    in the same test set size (consistency)
    """

    dataset, groups, y = fixture_sample_inputs
    strategy = fixture_test_strategy

    indices_1 = strategy.split(
        dataset=dataset,
        groups=groups,
        y=y,
        train_fraction=0.6,
        validation_fraction=0.2,
        test_fraction=0.2,
        same_group_same_set=False,
    )
    _, _, test_indices_1 = next(indices_1)

    indices_2 = strategy.split(
        dataset=dataset,
        groups=groups,
        y=y,
        train_fraction=0.3,
        validation_fraction=0.5,
        test_fraction=0.2,
        same_group_same_set=False,
    )
    _, _, test_indices_2 = next(indices_2)

    assert len(test_indices_1) == len(test_indices_2)


def test_n_splits(fixture_test_strategy):
    """Test the n_splits behaviour. should return n_splits folds with the required fractions"""

    n = 100
    dataset = np.random.rand(n)
    groups = np.arange(n)
    np.random.shuffle(groups)
    y = np.concatenate(
        [
            np.ones(n // 5),
            2 * np.ones(n // 5),
            3 * np.ones(n // 5),
            4 * np.ones(n // 5),
            5 * np.ones(n // 5),
        ],
    )

    train_fraction = [0.6, 0.7, 0.8]
    gap_train_validation_fraction = [0.05, 0.05, 0.05]
    validation_fraction = [0.2, 0.1, 0.0]
    gap_validation_test_fraction = [0.05, 0.05, 0.05]
    test_fraction = [0.1, 0.1, 0.1]

    strategy = fixture_test_strategy
    indices = strategy.split(
        dataset=dataset,
        groups=groups,
        y=y,
        train_fraction=train_fraction,
        gap_train_validation_fraction=gap_train_validation_fraction,
        validation_fraction=validation_fraction,
        gap_validation_test_fraction=gap_validation_test_fraction,
        test_fraction=test_fraction,
        same_group_same_set=False,
    )

    for ii, (train_indices, validation_indices, test_indices) in enumerate(indices):
        assert len(train_indices) == round(train_fraction[ii] * len(dataset))
        assert len(validation_indices) == round(validation_fraction[ii] * len(dataset))
        assert len(test_indices) == round(test_fraction[ii] * len(dataset))


def test_splits_are_disjoint(fixture_sample_inputs, fixture_test_strategy):
    """That data is spread across splits without overlap."""

    dataset, groups, y = fixture_sample_inputs
    strategy = fixture_test_strategy
    indices = strategy.split(dataset=dataset, groups=groups, y=y)

    for train_indices, validation_indices, test_indices in indices:
        assert set(train_indices).isdisjoint(set(validation_indices))
        assert set(validation_indices).isdisjoint(set(test_indices))
        assert set(train_indices).isdisjoint(set(test_indices))


def test_deterministic(fixture_sample_inputs, fixture_test_strategy):
    """That repeated splits are deterministic."""
    dataset, groups, y = fixture_sample_inputs
    strategy = fixture_test_strategy

    # Do one iteration of ordered splits
    indices_a = strategy.split(dataset, groups=groups, y=y)
    train_indices_a1, validation_indices_a1, test_indices_a1 = next(indices_a)

    # And again...
    indices_b = strategy.split(dataset, groups=groups, y=y)
    train_indices_b1, validation_indices_b1, test_indices_b1 = next(indices_b)

    # Check that we get the same results across folds
    assert sorted(train_indices_a1) == sorted(train_indices_b1)
    assert sorted(validation_indices_a1) == sorted(validation_indices_b1)
    assert sorted(test_indices_a1) == sorted(test_indices_b1)


def test_ignoring_nones_ratios(fixture_sample_inputs, fixture_test_strategy):
    """That ignoring none for computing split ratios means more data in train and less in val and test"""
    dataset, groups, y = fixture_sample_inputs
    strategy = fixture_test_strategy

    # Do one iteration of ordered splits
    indices_a = strategy.split(
        dataset,
        groups=groups,
        y=y,
        undefined_groups_in_train=True,
    )
    train_indices_a1, validation_indices_a1, test_indices_a1 = next(indices_a)

    indices_b = strategy.split(
        dataset,
        groups=groups,
        y=y,
        undefined_groups_in_train=False,
    )
    train_indices_b1, validation_indices_b1, test_indices_b1 = next(indices_b)

    # assert that there are more in train and less in val and test
    assert len(train_indices_b1) < len(train_indices_a1)
    assert (len(validation_indices_b1) > len(validation_indices_a1)) or (
        len(test_indices_b1) > len(test_indices_a1)
    )


def test_same_groups_same_set(fixture_test_strategy):
    """Test to check that same_groups of same targets are put in the same set (no overlap)"""
    dataset = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    groups = [0, 1, 1, 1, 2, 2, 2, 2, 2, 3]
    y = ["a", "a", "a", "a", "a", "b", "b", "b", "b", "b"]

    strategy = fixture_test_strategy
    indices = strategy.split(dataset=dataset, groups=groups, y=y)
    train_indices, validation_indices, test_indices = next(indices)

    assert sorted(train_indices) == [0, 1, 2, 3, 5, 6, 7, 8]
    assert sorted(validation_indices) == []
    assert sorted(test_indices) == [4, 9]


def test_min_size_test(fixture_test_strategy):
    """Test to check that min_test_size is respected"""

    dataset = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    groups = [0, 1, 1, 1, 2, 2, 2, 2, 2, 3]
    y = ["a", "a", "a", "a", "a", "b", "b", "b", "b", "b"]

    strategy = fixture_test_strategy
    indices = strategy.split(
        dataset=dataset,
        groups=groups,
        y=y,
        min_test_size_per_target=3,
    )
    train_indices, validation_indices, test_indices = next(indices)
    assert len(test_indices) >= (len(set(y)) * 3)


def test_n_splits_with_min_size(fixture_test_strategy):
    """Test the n_splits behaviour with min_test_size list."""

    n = 100
    dataset = np.random.rand(n)
    groups = np.arange(n)
    np.random.shuffle(groups)
    y = np.concatenate(
        [
            np.ones(n // 5),
            2 * np.ones(n // 5),
            3 * np.ones(n // 5),
            4 * np.ones(n // 5),
            5 * np.ones(n // 5),
        ],
    )

    train_fraction = [0.6, 0.7, 0.8]
    gap_train_validation_fraction = [0.05, 0.05, 0.05]
    validation_fraction = [0.2, 0.1, 0.0]
    gap_validation_test_fraction = [0.05, 0.05, 0.05]
    test_fraction = [0.1, 0.1, 0.1]
    min_test_size_per_target = [0, 5, 9]

    strategy = fixture_test_strategy
    indices = strategy.split(
        dataset=dataset,
        groups=groups,
        y=y,
        train_fraction=train_fraction,
        gap_train_validation_fraction=gap_train_validation_fraction,
        validation_fraction=validation_fraction,
        gap_validation_test_fraction=gap_validation_test_fraction,
        test_fraction=test_fraction,
        same_group_same_set=False,
        min_test_size_per_target=min_test_size_per_target,
    )

    for ii, (_train_indices, _validation_indices, test_indices) in enumerate(indices):
        assert len(test_indices) >= min_test_size_per_target[ii]


def test_line_separated_targets_and_groups(fixture_test_strategy):
    """Test functionality of using | separated targets and groups"""

    dataset = list(range(25))
    groups = [
        "2000",
        "2001",
        "2002",
        "2003",
        "2004",
        "2005",
        "2006",
        "2007",
        "2008",
        "2009",
        "2010",
        "2011",
        "2012",
        "2013",
        "2014",
        "2015",
        "2016",
        "2017",
        "2018",
        "2019",
        "2005",
        "2008",
        "2012",
        "2014",
        "2018",
    ]
    y = 10 * ["a"] + 10 * ["b"] + 5 * ["a | b"]

    strategy = fixture_test_strategy
    indices = strategy.split(
        dataset=dataset,
        groups=groups,
        y=y,
        train_fraction=0.6,
        validation_fraction=0.2,
        test_fraction=0.2,
        same_group_same_set=False,
    )
    train_indices, validation_indices, test_indices = next(indices)

    assert sorted(train_indices) == [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        10,
        11,
        12,
        13,
        14,
        20,
    ]
    assert sorted(validation_indices) == [8, 9, 15, 16, 17, 21]
    assert sorted(test_indices) == [18, 19, 22, 23, 24]
