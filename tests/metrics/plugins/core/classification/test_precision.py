import numpy as np
import pytest

from molflux.metrics import Metric, list_metrics, load_metric
from molflux.metrics.classification.precision import Precision


@pytest.fixture()
def fixture_metric():
    return load_metric("precision")


def test_metric_in_catalogue():
    """That the metric is registered in the catalogue."""
    catalogue = list_metrics()
    all_metric_names = [name for names in catalogue.values() for name in names]
    assert "precision" in all_metric_names


def test_metric_is_mapped_to_correct_class(fixture_metric):
    """That the catalogue name is mapped to the appropriate class."""
    metric = fixture_metric
    assert isinstance(metric, Precision)


def test_implements_protocol(fixture_metric):
    """That the metric implements the public Metric protocol."""
    metric = fixture_metric
    assert isinstance(metric, Metric)


def test_integer_labels(fixture_metric):
    """That can use integer labels."""
    metric = fixture_metric
    predictions = [1, 1, 0, 0, 0, 1]
    references = [0, 1, 1, 0, 0, 1]
    result = metric.compute(predictions=predictions, references=references)
    assert result["precision"] == 2 / 3


def test_multiclass_macro_average(fixture_metric):
    """Multiclass case with unweighted mean average across classes."""
    metric = fixture_metric
    predictions = [0, 2, 1, 0, 0, 1]
    references = [0, 1, 2, 0, 1, 2]
    result = metric.compute(
        predictions=predictions,
        references=references,
        average="macro",
    )
    assert pytest.approx(result["precision"], 0.01) == 0.22


def test_multiclass_micro_average(fixture_metric):
    """Multiclass case with metrics calculated globally."""
    metric = fixture_metric
    predictions = [0, 2, 1, 0, 0, 1]
    references = [0, 1, 2, 0, 1, 2]
    result = metric.compute(
        predictions=predictions,
        references=references,
        average="micro",
    )
    assert pytest.approx(result["precision"], 0.01) == 0.333


def test_multiclass_weighted_average(fixture_metric):
    """Multiclass case with average across classes weighted by support."""
    metric = fixture_metric
    predictions = [0, 2, 1, 0, 0, 1]
    references = [0, 1, 2, 0, 1, 2]
    result = metric.compute(
        predictions=predictions,
        references=references,
        average="weighted",
    )
    assert pytest.approx(result["precision"], 0.01) == 0.22


def test_multiclass_none_average(fixture_metric):
    """Multiclass case with scores returned for each class."""
    metric = fixture_metric
    predictions = [0, 2, 1, 0, 0, 1]
    references = [0, 1, 2, 0, 1, 2]
    result = metric.compute(
        predictions=predictions,
        references=references,
        average=None,
    )
    np.testing.assert_allclose([2 / 3, 0, 0], result["precision"])


@pytest.mark.filterwarnings(
    "ignore:Precision is ill-defined and being set to 0.0 in labels with no predicted samples:sklearn.exceptions.UndefinedMetricWarning",
)
def test_zero_division_zero(fixture_metric):
    """Returns zero on a zero division (when all predictions amd labels are negative)."""
    metric = fixture_metric
    predictions = [0, 0, 0, 0, 0, 0]
    references = [0, 1, 2, 0, 1, 2]
    result = metric.compute(
        predictions=predictions,
        references=references,
        average=None,
    )
    np.testing.assert_allclose([1 / 3, 0, 0], result["precision"])


def test_zero_division_one(fixture_metric):
    """Can return one on a zero division (when all predictions amd labels are negative)."""
    metric = fixture_metric
    predictions = [0, 0, 0, 0, 0, 0]
    references = [0, 1, 2, 0, 1, 2]
    result = metric.compute(
        predictions=predictions,
        references=references,
        average=None,
        zero_division=1,
    )
    np.testing.assert_allclose([1 / 3, 1, 1], result["precision"])


def test_perfect_score(fixture_metric):
    """That get unitary score for perfect predictions."""
    metric = fixture_metric
    predictions = [0, 1, 1, 0, 0, 1]
    references = [0, 1, 1, 0, 0, 1]
    result = metric.compute(predictions=predictions, references=references)
    assert result["precision"] == 1.0


@pytest.mark.xfail
def test_binary_multilabel():
    """Multilabel case with binary label indicators."""
    metric = load_metric("precision", config_name="multilabel")
    predictions = [[0, 0, 0], [1, 1, 1], [1, 1, 0]]
    references = [[0, 0, 0], [1, 1, 1], [0, 1, 1]]
    result = metric.compute(predictions=predictions, references=references)
    np.testing.assert_allclose([1 / 2, 1, 1], result["precision"])
