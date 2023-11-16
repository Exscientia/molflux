import numpy as np
import pytest

from molflux.metrics import Metric, list_metrics, load_metric
from molflux.metrics.regression.proportion_within_fold import ProportionWithinFold


@pytest.fixture()
def fixture_metric():
    return load_metric("proportion_within_fold")


def test_metric_in_catalogue():
    """That the metric is registered in the catalogue."""
    catalogue = list_metrics()
    all_metric_names = [name for names in catalogue.values() for name in names]
    assert "proportion_within_fold" in all_metric_names


def test_metric_is_mapped_to_correct_class(fixture_metric):
    """That the catalogue name is mapped to the appropriate class."""
    metric = fixture_metric
    assert isinstance(metric, ProportionWithinFold)


def test_implements_protocol(fixture_metric):
    """That the metric implements the public Metric protocol."""
    metric = fixture_metric
    assert isinstance(metric, Metric)


def test_default_compute(fixture_metric):
    """That default scoring."""
    metric = fixture_metric
    predictions = [2.5, 4.0, 2, 8]
    references = [3, 0.5, 2, 7]
    result = metric.compute(predictions=predictions, references=references)
    assert pytest.approx(result["proportion_within_fold"], 0.1) == 0.75


def test_raises_warning_for_negative_inputs(fixture_metric):
    """Negative inputs should raise a warning as will need to be log transformed."""
    metric = fixture_metric
    predictions = [2.5, -4.0, 2, 8]
    references = [3, 0.5, 2, 7]
    with pytest.raises(RuntimeError):
        metric.compute(predictions=predictions, references=references)


def test_log_scale_gives_result(fixture_metric):
    """Transforming by log10 should give the same result as using the log_scale option."""
    metric = fixture_metric
    predictions = [2.5, 4.0, 2, 8]
    references = [3, 0.5, 2, 7]
    result = metric.compute(predictions=predictions, references=references)
    result_log_scaled = metric.compute(
        predictions=np.log10(predictions),
        references=np.log10(references),
        log_scaled_inputs=True,
    )
    assert (
        pytest.approx(result["proportion_within_fold"], 0.0001)
        == result_log_scaled["proportion_within_fold"]
    )
