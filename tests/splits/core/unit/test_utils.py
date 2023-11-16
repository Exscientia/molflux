from molflux.splits.utils import partition


def test_even_partition():
    """That a dataset with even molfluxs is partitioned correctly."""
    train_fraction = 0.6
    validation_fraction = 0.3
    dataset = range(100)
    train_cutoff, validation_cutoff = partition(
        dataset,
        train_fraction,
        validation_fraction,
    )
    assert train_cutoff == 60
    assert validation_cutoff == 90


def test_odd_partition():
    """That a dataset with odd molfluxs is partitioned correctly."""
    train_fraction = 0.6
    validation_fraction = 0.3
    dataset = range(101)
    train_cutoff, validation_cutoff = partition(
        dataset,
        train_fraction,
        validation_fraction,
    )
    assert train_cutoff == 60
    assert validation_cutoff == 90
    # spill over should go into test dataset


def test_train_test_partition():
    """That a train test split gives an empty validation partition."""
    train_fraction = 0.6
    validation_fraction = 0.0
    dataset = range(100)
    train_cutoff, validation_cutoff = partition(
        dataset,
        train_fraction,
        validation_fraction,
    )
    assert train_cutoff == 60
    assert validation_cutoff == 60
