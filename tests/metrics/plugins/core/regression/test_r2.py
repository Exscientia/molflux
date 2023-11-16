import pytest

from molflux.metrics import Metric, list_metrics, load_metric
from molflux.metrics.regression.r2 import R2


@pytest.fixture()
def fixture_metric():
    return load_metric("r2")


def test_metric_in_catalogue():
    """That the metric is registered in the catalogue."""
    catalogue = list_metrics()
    all_metric_names = [name for names in catalogue.values() for name in names]
    assert "r2" in all_metric_names


def test_metric_is_mapped_to_correct_class(fixture_metric):
    """That the catalogue name is mapped to the appropriate class."""
    metric = fixture_metric
    assert isinstance(metric, R2)


def test_implements_protocol(fixture_metric):
    """That the metric implements the public Metric protocol."""
    metric = fixture_metric
    assert isinstance(metric, Metric)


def test_default_compute(fixture_metric):
    """That default scoring."""
    metric = fixture_metric
    predictions = [2.5, 0.0, 2, 8]
    references = [3, -0.5, 2, 7]
    result = metric.compute(predictions=predictions, references=references)
    assert pytest.approx(result["r2"], 0.01) == 0.948


def test_perfect_score(fixture_metric):
    """Test score on perfect predictions."""
    metric = fixture_metric
    predictions = [3, -0.5, 2, 7]
    references = [3, -0.5, 2, 7]
    result = metric.compute(predictions=predictions, references=references)
    assert result["r2"] == 1


def test_constant_model(fixture_metric):
    """Test null score on a constant model that always predicts the expected value disregarding input features."""
    metric = fixture_metric
    predictions = [1, 2, 3]
    references = [2, 2, 2]
    result = metric.compute(predictions=predictions, references=references)
    assert result["r2"] == 0


def test_multioutpout_variance_weighted(fixture_metric):
    """Multioutput case with errors of all outputs averaged weighted by the variances of each individual output."""
    metric = load_metric("r2", config_name="multioutput")
    predictions = [[0, 2], [-1, 2], [8, -5]]
    references = [[0.5, 1], [-1, 1], [7, -6]]
    result = metric.compute(
        predictions=predictions,
        references=references,
        multioutput="variance_weighted",
    )
    assert result
    assert pytest.approx(result["r2"], 0.01) == 0.938
