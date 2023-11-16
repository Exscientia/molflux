from typing import Any, Iterable, Iterator, Optional, Sized, Tuple

import pytest

from molflux.datasets import load_dataset
from molflux.datasets.interfaces import SplittingStrategy


@pytest.fixture(scope="module")
def fixture_dataset(fixture_path_to_assets):
    """A sample Dataset implementation."""
    return load_dataset("esol").select(range(100))


class SplittingStrategyMock:
    """A mock SplittingStrategy"""

    def split(
        self,
        dataset: Sized,
        y: Optional[Iterable] = None,
        groups: Optional[Iterable] = None,
        **kwargs: Any,
    ) -> Iterator[Tuple[Iterable[int], Iterable[int], Iterable[int]]]:
        """Splits the dataset into thirds."""
        n = len(dataset)
        indices = range(n)
        yield indices[: n // 3], indices[n // 3 : 2 * n // 3], indices[2 * n // 3 :]


@pytest.fixture(scope="module")
def fixture_splitting_strategy_mock():
    strategy = SplittingStrategyMock()
    assert isinstance(strategy, SplittingStrategy)
    return strategy
