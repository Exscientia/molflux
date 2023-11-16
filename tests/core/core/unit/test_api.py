"""
Tests ensuring desired API objects are part of the top-level namespace.
"""
import pytest

import molflux.core
import molflux.core.tracking


def test_exports_version():
    """That the package exposes the __version__ variable."""
    assert hasattr(molflux, "__version__")


@pytest.mark.parametrize(
    "callable_name",
    [
        "featurise_dataset",
        "replay_dataset_featurisation",
        "fetch_model_featurisation_metadata",
        "load_featurisation_metadata",
        "get_inputs",
        "get_references",
        "inference",
        "load_model",
        "predict",
        "save_model",
        "compute_scores",
        "invert_scores_hierarchy",
        "merge_scores",
        "score_model",
    ],
)
def test_exports_callable(callable_name):
    """That the package exposes the given callable."""
    assert hasattr(molflux.core, callable_name)
    assert callable(getattr(molflux.core, callable_name))


def test_exports_tracking_api():
    """That the package exposes the .tracking module."""
    assert hasattr(molflux.core, "tracking")


@pytest.mark.parametrize(
    "tracking_api_callable_name",
    [
        "log_dataset",
        "log_dataset_dict",
        "log_featurised_dataset",
        "log_featurisation_metadata",
        "log_fold",
        "log_inputs",
        "log_model_params",
        "log_params",
        "log_predictions",
        "log_pipeline_config",
        "log_references",
        "log_scores",
        "log_splitting_strategy",
    ],
)
def test_exports_tracking_api_callable(tracking_api_callable_name):
    """That the package .tracking API exposes the given callable."""
    assert hasattr(molflux.core.tracking, tracking_api_callable_name)
    assert callable(getattr(molflux.core.tracking, tracking_api_callable_name))
