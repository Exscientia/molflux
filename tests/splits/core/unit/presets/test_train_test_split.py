from molflux.splits.presets import train_test_split


def test_strategy_generates_train_test_splits():
    """That train, and test splits are generated, but no validation."""
    strategy = train_test_split()
    data = range(100)
    folds = strategy.split(data)
    train_indices, validation_indices, test_indices = next(folds)
    assert len(list(train_indices))
    assert len(list(validation_indices)) == 0
    assert len(list(test_indices))


def test_ensure_empty_validation_on_small_samples():
    """That the validation split is empty, even on small samples.

    (Where rounding errors might be more pronounced)
    """
    strategy = train_test_split()

    data = range(4)
    folds = strategy.split(data)
    train_indices, validation_indices, test_indices = next(folds)
    assert len(list(validation_indices)) == 0

    data = range(3)
    folds = strategy.split(data)
    train_indices, validation_indices, test_indices = next(folds)
    assert len(list(validation_indices)) == 0

    data = range(2)
    folds = strategy.split(data)
    train_indices, validation_indices, test_indices = next(folds)
    assert len(list(validation_indices)) == 0
