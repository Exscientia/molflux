import pytest

from molflux.metrics.errors import DuplicateKeyError
from molflux.metrics.load import (
    load_from_dict,
    load_from_dicts,
    load_from_yaml,
    load_metric,
    load_metrics,
)
from molflux.metrics.metric import Metric, Metrics

representative_metric_name = "explained_variance"


def test_returns_metric():
    """That loading a metric returns an object of type Metric."""
    name = representative_metric_name
    metric = load_metric(name=name)
    assert isinstance(metric, Metric)


def test_raises_not_implemented_for_unknown_metric():
    """That a NotImplementedError is raised if attempting to load an unknown
    metric."""
    name = "unknown"
    with pytest.raises(NotImplementedError):
        load_metric(name=name)


def test_raised_error_for_unknown_metric_provides_close_matches():
    """That the error raised when attempting to load an unknown metric
    shows possible close matches to the user (if any)."""
    name = "accurcy"
    # This should suggest e.g. ["accuracy", "top_k_accuracy"]
    with pytest.raises(
        NotImplementedError,
        match="You might be looking for one of these",
    ):
        load_metric(name=name)


def test_forwards_init_kwargs_to_builder():
    """That keyword arguments get forwarded to the metric initialiser."""
    name = representative_metric_name
    metric = load_metric(name=name, tag="pytest-tag")
    assert metric.tag == "pytest-tag"


def test_metrics_with_same_tag_in_collection_raises():
    """That adding several metrics with the same tag in a Metrics collection raises."""
    name = representative_metric_name
    with pytest.raises(DuplicateKeyError):
        load_metrics(name, name)


def test_load_metrics_returns_metrics():
    """That the load_metrics function returns a Metrics collection."""
    name = representative_metric_name
    metrics = load_metrics(name, name, tags=["one", "two"])
    assert isinstance(metrics, Metrics)


def test_load_metrics_returns_correct_number_of_metrics():
    """That the load_metrics function returns a Metrics collection of the
    expected size."""
    name = representative_metric_name
    metrics = load_metrics(name, name, tags=["one", "two"])
    assert len(metrics) == 2


def test_from_dict_returns_metric():
    """That loading from a dict returns a Metric."""
    name = representative_metric_name
    config = {
        "name": name,
        "config": {},
        "presets": {},
    }
    metric = load_from_dict(config)
    assert isinstance(metric, Metric)


def test_from_minimal_dict():
    """That can provide a config with only required fields."""
    name = representative_metric_name
    config = {
        "name": name,
    }
    assert load_from_dict(config)


def test_from_dict_forwards_config_to_builder():
    """That config keyword arguments get forwarded to the initialiser."""
    name = representative_metric_name
    config = {
        "name": name,
        "config": {
            "tag": "pytest-tag",
        },
    }
    metric = load_from_dict(config)
    assert metric.tag == "pytest-tag"


def test_from_dict_forwards_presets_to_state():
    """That presets keyword arguments get stored in the metric state."""
    name = representative_metric_name
    config = {
        "name": name,
        "presets": {
            "multioutput": "variance_weighted",
        },
    }
    metric = load_from_dict(config)
    assert metric.state
    assert metric.state.get("multioutput") == "variance_weighted"


def test_dict_missing_required_fields_raises():
    """That cannot load a metric with a config missing required fields."""
    config = {"unknown_key": "value"}
    with pytest.raises(SyntaxError):
        load_from_dict(config)


def test_from_dicts_returns_metrics():
    """That loading from a collection of dicts returns a Metrics object."""
    name = representative_metric_name
    config_one = {
        "name": name,
        "config": {
            "tag": "one",
        },
        "presets": {},
    }
    config_two = {
        "name": name,
        "config": {
            "tag": "two",
        },
        "presets": {},
    }
    configs = [config_one, config_two]
    metrics = load_from_dicts(configs)
    assert isinstance(metrics, Metrics)


def test_from_yaml_returns_metrics(fixture_path_to_assets):
    path = fixture_path_to_assets / "config.yml"
    metrics = load_from_yaml(path=path)
    assert len(metrics) == 2
    assert "explained_variance" in metrics
    assert "custom_metric" in metrics
    assert isinstance(metrics, Metrics)
    assert metrics["custom_metric"].state.get("root") is True
