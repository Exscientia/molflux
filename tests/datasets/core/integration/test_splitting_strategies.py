import molflux.splits
from molflux.datasets.interfaces import SplittingStrategy


def test_integrates_with_molflux_splits():
    """That the SplittingStrategy interface used by datasets is compatible
    with the SplittingStrategy protocol defined by molflux.features.

    That is, we check that datasets works with strategies from molflux.features.
    """
    assert isinstance(molflux.splits.SplittingStrategy, SplittingStrategy)
