import numpy as np
import pandas as pd
import pytest

from molflux.splits.catalogue import list_splitting_strategies
from molflux.splits.load import load_splitting_strategy
from molflux.splits.strategy import SplittingStrategy

strategy_name = "ordered_split"


@pytest.fixture(scope="module")
def fixture_test_strategy():
    return load_splitting_strategy(strategy_name)


@pytest.fixture(scope="module")
def fixture_sample_inputs():
    n = 100
    dataset = np.random.rand(n)
    groups = np.concatenate([np.arange(n - 20), np.array([None] * 20)])
    np.random.shuffle(groups)
    return dataset, groups


def test_is_in_catalogue():
    """That the strategy is registered in the catalogue."""
    catalogue = list_splitting_strategies()
    all_strategy_names = [name for names in catalogue.values() for name in names]
    assert strategy_name in all_strategy_names


def test_implements_protocol(fixture_test_strategy):
    """That the strategy implements the protocol."""
    strategy = fixture_test_strategy
    assert isinstance(strategy, SplittingStrategy)


def test_requires_groups(fixture_test_strategy):
    """That you need to provide groups to use this strategy."""
    dataset = [0, 1, 2, 3, 4, 5]
    strategy = fixture_test_strategy
    with pytest.raises(ValueError):
        next(strategy.split(dataset))


def test_hard_coded_example(fixture_test_strategy):
    """Test to illustrate the split"""
    dataset = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    groups = [2, 7, 5, 6, 3, 1, 0, 8, 9, 4]

    strategy = fixture_test_strategy
    indices = strategy.split(dataset=dataset, groups=groups)
    train_indices, validation_indices, test_indices = next(indices)

    assert train_indices == [6, 5, 0, 4, 9, 2, 3, 1]
    assert validation_indices == [7]
    assert test_indices == [8]


def test_hard_coded_example_with_nones(fixture_test_strategy):
    """Test to illustrate the split when there are nones:
    Nones will be added to train and ignored when computing fractions
    """

    dataset = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    groups = [3, 0, 1, None, 4, 5, 2, 6, None, 9]

    strategy = fixture_test_strategy
    indices = strategy.split(dataset=dataset, groups=groups, same_group_same_set=False)
    train_indices, validation_indices, test_indices = next(indices)

    assert train_indices == [3, 8, 1, 2, 6, 0, 4, 5]
    assert validation_indices == [7]
    assert test_indices == [9]


def test_example_with_nulls_nones(fixture_test_strategy):
    """Test to illustrate the split when there are nones or other null values:
    Nones will be added to train and ignored when computing fractions
    """

    dataset = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    groups = [3, 0, 1, np.nan, 4, 5, 2, 6, pd.NaT, 9]

    strategy = fixture_test_strategy
    indices = strategy.split(dataset=dataset, groups=groups, same_group_same_set=False)
    train_indices, validation_indices, test_indices = next(indices)

    assert train_indices == [3, 8, 1, 2, 6, 0, 4, 5]
    assert validation_indices == [7]
    assert test_indices == [9]


def test_hard_coded_example_with_non_unique_groups(fixture_test_strategy):
    """Test to illustrate the split when there are nones:
    Nones will be added to train and ignored when computing fractions
    """

    dataset = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    groups = [0, 0, 0, 0, 1, 1, 1, 2, 2, 3]

    strategy = fixture_test_strategy
    indices = strategy.split(dataset=dataset, groups=groups, same_group_same_set=False)
    train_indices, validation_indices, test_indices = next(indices)

    assert train_indices == [0, 1, 2, 3, 4, 5, 6, 7]
    assert validation_indices == [8]
    assert test_indices == [9]


def test_hard_coded_example_with_unusual_ratios(fixture_test_strategy):
    """Test to illustrate the split when there are nones:
    Nones will be added to train and ignored when computing fractions
    """

    dataset = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    groups = [2, 0, 1, None, 2, 5, 2, 5, None, 9]

    strategy = fixture_test_strategy
    indices = strategy.split(
        dataset=dataset,
        groups=groups,
        train_fraction=0.1,
        validation_fraction=0.8,
        test_fraction=0.1,
        undefined_groups_in_train=False,
        same_group_same_set=False,
    )
    train_indices, validation_indices, test_indices = next(indices)

    assert train_indices == [3]
    assert validation_indices == [8, 1, 2, 0, 4, 6, 5, 7]
    assert test_indices == [9]


def test_example_same_test_ratios_but_different_train_val(
    fixture_sample_inputs,
    fixture_test_strategy,
):
    """Test to illustrate that the same test ratio with different train val results
    in the same test set size (consistency)
    """

    dataset, groups = fixture_sample_inputs
    strategy = fixture_test_strategy

    indices_1 = strategy.split(
        dataset=dataset,
        groups=groups,
        train_fraction=0.6,
        validation_fraction=0.2,
        test_fraction=0.2,
        same_group_same_set=False,
    )
    _, _, test_indices_1 = next(indices_1)

    indices_2 = strategy.split(
        dataset=dataset,
        groups=groups,
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

    train_fraction = [0.6, 0.7, 0.8]
    gap_train_validation_fraction = [0.05, 0.05, 0.05]
    validation_fraction = [0.2, 0.1, 0.0]
    gap_validation_test_fraction = [0.05, 0.05, 0.05]
    test_fraction = [0.1, 0.1, 0.1]

    strategy = fixture_test_strategy
    indices = strategy.split(
        dataset=dataset,
        groups=groups,
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
    dataset, groups = fixture_sample_inputs
    strategy = fixture_test_strategy
    indices = strategy.split(
        dataset=dataset,
        groups=groups,
        same_group_same_set=False,
    )
    for train_indices, validation_indices, test_indices in indices:
        assert (
            set(train_indices).isdisjoint(set(validation_indices))
            and set(validation_indices).isdisjoint(set(test_indices))
            and set(train_indices).isdisjoint(set(test_indices))
        )


def test_deterministic(fixture_sample_inputs, fixture_test_strategy):
    """That repeated splits are deterministic."""
    dataset, groups = fixture_sample_inputs
    strategy = fixture_test_strategy

    # Do one iteration of ordered splits
    indices_a = strategy.split(dataset, groups=groups, same_group_same_set=False)
    train_indices_a1, validation_indices_a1, test_indices_a1 = next(indices_a)

    # And again...
    indices_b = strategy.split(dataset, groups=groups, same_group_same_set=False)
    train_indices_b1, validation_indices_b1, test_indices_b1 = next(indices_b)

    # Check that we get the same results across folds
    assert train_indices_a1 == train_indices_b1
    assert validation_indices_a1 == validation_indices_b1
    assert test_indices_a1 == test_indices_b1


def test_ignoring_nones_ratios(fixture_sample_inputs, fixture_test_strategy):
    """That ignoring none for computing split ratios means more data in train and less in val and test"""
    dataset, groups = fixture_sample_inputs
    strategy = fixture_test_strategy

    # Do one iteration of ordered splits
    indices_a = strategy.split(
        dataset,
        groups=groups,
        undefined_groups_in_train=True,
        same_group_same_set=False,
    )
    train_indices_a1, validation_indices_a1, test_indices_a1 = next(indices_a)

    indices_b = strategy.split(
        dataset,
        groups=groups,
        undefined_groups_in_train=False,
        same_group_same_set=False,
    )
    train_indices_b1, validation_indices_b1, test_indices_b1 = next(indices_b)

    # assert that there are more in train and less in val and test
    assert len(train_indices_b1) < len(train_indices_a1)
    assert (len(validation_indices_b1) > len(validation_indices_a1)) or (
        len(test_indices_b1) > len(test_indices_a1)
    )


def test_same_groups_same_set(fixture_test_strategy):
    """Test to check that same_groups are put in the same set (no overlap)"""
    dataset = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    groups = [0, 1, 1, 1, 2, 2, 2, 2, 2, 3]

    strategy = fixture_test_strategy
    indices = strategy.split(dataset=dataset, groups=groups)
    train_indices, validation_indices, test_indices = next(indices)

    assert train_indices == [0, 1, 2, 3, 4, 5, 6, 7, 8]
    assert validation_indices == []
    assert test_indices == [9]


def test_min_size_test(fixture_test_strategy):
    """Test to check that min_test_size is respected"""
    dataset = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    groups = [0, 1, 1, 1, 2, 2, 2, 2, 2, 3]

    strategy = fixture_test_strategy
    indices = strategy.split(dataset=dataset, groups=groups, min_test_size=5)
    train_indices, validation_indices, test_indices = next(indices)

    assert len(test_indices) > 5


def test_n_splits_with_min_test_size(fixture_test_strategy):
    """Test the n_splits behaviour with min_test_size list"""

    n = 100
    dataset = np.random.rand(n)
    groups = np.arange(n)
    np.random.shuffle(groups)

    train_fraction = [0.6, 0.7, 0.8]
    gap_train_validation_fraction = [0.05, 0.05, 0.05]
    validation_fraction = [0.2, 0.1, 0.0]
    gap_validation_test_fraction = [0.05, 0.05, 0.05]
    test_fraction = [0.1, 0.1, 0.1]
    min_test_size = [0, 5, 9]

    strategy = fixture_test_strategy
    indices = strategy.split(
        dataset=dataset,
        groups=groups,
        train_fraction=train_fraction,
        gap_train_validation_fraction=gap_train_validation_fraction,
        validation_fraction=validation_fraction,
        gap_validation_test_fraction=gap_validation_test_fraction,
        test_fraction=test_fraction,
        same_group_same_set=False,
        min_test_size=min_test_size,
    )

    for ii, (_train_indices, _validation_indices, test_indices) in enumerate(indices):
        assert len(test_indices) >= min_test_size[ii]
