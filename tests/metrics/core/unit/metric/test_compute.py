import pytest

from molflux.metrics.load import load_metric


@pytest.fixture(scope="module")
def fixture_mock_metric():
    return load_metric(kind="classification", name="accuracy")


def test_invalid_kwarg_raises_value_error(fixture_mock_metric):
    """That attempting scoring with invalid keyword arguments raises."""
    metric = fixture_mock_metric
    predictions = [1, 0]
    references = [1, 0]
    with pytest.raises(ValueError, match=r"Unknown compute parameter\(s\)"):
        metric.compute(
            predictions=predictions,
            references=references,
            invalid_kwarg=True,
        )
