import pytest

from molflux.metrics import Metric, list_metrics, load_metric
from molflux.metrics.uncertainty.uncertainty_based_rejection import (
    UncertaintyBasedRejection,
)


@pytest.fixture()
def fixture_metric():
    return load_metric("uncertainty_based_rejection")


def test_metric_in_catalogue():
    """That the metric is registered in the catalogue."""
    catalogue = list_metrics()
    all_metric_names = [name for names in catalogue.values() for name in names]
    assert "uncertainty_based_rejection" in all_metric_names


def test_metric_is_mapped_to_correct_class(fixture_metric):
    """That the catalogue name is mapped to the appropriate class."""
    metric = fixture_metric
    assert isinstance(metric, UncertaintyBasedRejection)


def test_implements_protocol(fixture_metric):
    """That the metric implements the public Metric protocol."""
    metric = fixture_metric
    assert isinstance(metric, Metric)


def test_default_compute(fixture_metric):
    """That default scoring."""
    metric = fixture_metric
    predictions = [3, 0.0, 2, 8]
    references = [3, -0.0, 3, 6]
    uncertainties = [0.0, 0.0, 1, 2]
    metric_name = "mean_squared_error"
    result = metric.compute(
        predictions=predictions,
        references=references,
        uncertainties=uncertainties,
        metric_name=metric_name,
    )

    # The highest threshold should be the same as computing over the entire dataset.
    results_dic = result["uncertainty_based_rejection"]
    threshold_index = max(list(results_dic.keys()))
    hf_metric = load_metric(metric_name)
    result_ind = hf_metric.compute(predictions=predictions, references=references)[
        metric_name
    ]
    assert results_dic[threshold_index][metric_name] == result_ind
    # The entire dataset should be included in the highest threshold.
    assert results_dic[threshold_index]["size_of_thresholded_ds"] == len(predictions)
    assert results_dic[threshold_index]["frac_of_retained_data"] == 1

    # Only the predictions with low uncertainty.
    filtered_pred = predictions[:2]
    filtered_ref = references[:2]
    result_ind = hf_metric.compute(predictions=filtered_pred, references=filtered_ref)[
        metric_name
    ]
    assert results_dic[0][metric_name] == result_ind


def test_different_metric_name(fixture_metric):
    """Change computed kw metric."""
    metric = fixture_metric
    predictions = [3, 0.0, 2, 8]
    references = [3, 0.0, 3, 6]
    uncertainties = [0.0, 0.0, 1, 2]
    metric_name = "mean_absolute_error"
    result = metric.compute(
        predictions=predictions,
        references=references,
        uncertainties=uncertainties,
        metric_name=metric_name,
    )

    # The highest threshold should be the same as computing over the entire dataset.
    results_dic = result["uncertainty_based_rejection"]
    threshold_index = max(list(results_dic.keys()))
    hf_metric = load_metric(metric_name)
    result_ind = hf_metric.compute(predictions=predictions, references=references)[
        metric_name
    ]
    assert results_dic[threshold_index][metric_name] == result_ind


def test_raise_error_different_length(fixture_metric):
    """That default scoring."""
    metric = fixture_metric
    predictions = [3, 0.0, 2]
    references = [3, -0.0, 3, 1]
    uncertainties = [0.0, 0.0, 1]
    metric_name = "mean_squared_error"
    with pytest.raises(RuntimeError):
        metric.compute(
            predictions=predictions,
            references=references,
            uncertainties=uncertainties,
            metric_name=metric_name,
        )


def test_num_of_threshold_steps(fixture_metric):
    """That default scoring."""
    metric = fixture_metric
    predictions = [3, 0.0, 2, 8]
    references = [3, -0.0, 3, 6]
    uncertainties = [0.0, 0.0, 1, 2]
    metric_name = "mean_squared_error"
    result = metric.compute(
        predictions=predictions,
        references=references,
        uncertainties=uncertainties,
        metric_name=metric_name,
        num_of_threshold_steps=2,
    )

    # The number of molfluxs of the dict should be <= num_of_threshold_steps
    # (not necessarily equal since metric will not compute if filtered preds
    # and refs have len < 2
    results_dic = result["uncertainty_based_rejection"]
    assert len(results_dic) <= 2


def test_compute_works_with_prediction_intervals(fixture_metric):
    """Check that using prediction intervals rather than uncertainties directly as input works."""
    metric = fixture_metric
    predictions = [3, 0.0, 2, 8]
    references = [3, -0.0, 3, 6]
    lower_bound = [0.0, 0.0, 1, 2]
    upper_bound = [5.0, 0.9, 4.0, 15.0]
    prediction_intervals = list(zip(lower_bound, upper_bound))
    metric_name = "mean_squared_error"
    result = metric.compute(
        predictions=predictions,
        references=references,
        uncertainties=None,
        prediction_intervals=prediction_intervals,
        metric_name=metric_name,
    )
    assert "uncertainty_based_rejection" in result
