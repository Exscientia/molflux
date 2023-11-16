import pytest

from molflux.metrics import Metric, list_metrics, load_metric
from molflux.metrics.regression.explained_variance import ExplainedVariance


@pytest.fixture()
def fixture_metric():
    return load_metric("explained_variance")


def test_metric_in_catalogue():
    """That the metric is registered in the catalogue."""
    catalogue = list_metrics()
    all_metric_names = [name for names in catalogue.values() for name in names]
    assert "explained_variance" in all_metric_names


def test_metric_is_mapped_to_correct_class(fixture_metric):
    """That the catalogue name is mapped to the appropriate class."""
    metric = fixture_metric
    assert isinstance(metric, ExplainedVariance)


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
    assert pytest.approx(result["explained_variance"], 0.01) == 0.957


def test_perfect_score(fixture_metric):
    """Test score on perfect predictions."""
    metric = fixture_metric
    predictions = [3, -0.5, 2, 7]
    references = [3, -0.5, 2, 7]
    result = metric.compute(predictions=predictions, references=references)
    assert result["explained_variance"] == 1.0


def test_multioutpout(fixture_metric):
    """Multioutput case with errors of all outputs averaged with uniform weight."""
    metric = load_metric("explained_variance", config_name="multioutput")
    predictions = [[0, 2], [-1, 2], [8, -5]]
    references = [[0.5, 1], [-1, 1], [7, -6]]
    result = metric.compute(
        predictions=predictions,
        references=references,
        multioutput="uniform_average",
    )
    assert result
    assert pytest.approx(result["explained_variance"], 0.01) == 0.983
