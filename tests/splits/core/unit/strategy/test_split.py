import pytest


def test_invalid_kwarg_raises_value_error(fixture_mock_splitting_strategy):
    """That attempting split with invalid keyword arguments raises."""
    strategy = fixture_mock_splitting_strategy
    data = ["CCCC", "CC"] * 10
    with pytest.raises(ValueError, match=r"Unknown split parameter\(s\)"):
        next(strategy.split(data, invalid_kwarg=True))
