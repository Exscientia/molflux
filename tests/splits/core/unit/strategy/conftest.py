import pytest

from molflux.splits.load import load_splitting_strategy


@pytest.fixture(scope="module")
def fixture_mock_splitting_strategy():
    return load_splitting_strategy(name="linear_split")
