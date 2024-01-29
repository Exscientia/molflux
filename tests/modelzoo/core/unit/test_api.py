"""
Tests ensuring desired API objects are part of the top-level namespace.
"""

import molflux.modelzoo


def test_exports_classification_mixin():
    """That the package exposes the ClassificationMixin ABC.

    This is exposed as a convenience building block for developers wishing to
    quickly create new modelzoo models in their applications.
    """
    assert hasattr(molflux.modelzoo, "ClassificationMixin")


def test_exports_list_models():
    """That the package exposes the list_models function."""
    assert hasattr(molflux.modelzoo, "list_models")
    assert callable(molflux.modelzoo.list_models)


def test_exports_load_model():
    """That the package exposes the load_model function."""
    assert hasattr(molflux.modelzoo, "load_model")
    assert callable(molflux.modelzoo.load_model)


def test_exports_load_from_dict():
    """That the package exposes the load_from_dict function."""
    assert hasattr(molflux.modelzoo, "load_from_dict")


def test_exports_load_from_dicts():
    """That the package exposes the load_from_dicts function."""
    assert hasattr(molflux.modelzoo, "load_from_dicts")


def test_exports_load_from_yaml():
    """That the package exposes the load_from_yaml function."""
    assert hasattr(molflux.modelzoo, "load_from_yaml")


def test_exports_register_model():
    """That the package exposes the register_model function."""
    assert hasattr(molflux.modelzoo, "register_model")
    assert callable(molflux.modelzoo.register_model)


def test_exports_model():
    """That the package exposes the Model type alias."""
    assert hasattr(molflux.modelzoo, "Model")


def test_exports_models():
    """That the package exposes the Models type alias."""
    assert hasattr(molflux.modelzoo, "Models")


def test_exports_model_base():
    """That the package exposes the ModelBase ABC.

    This is exposed as a convenience building block for developers wishing to
    quickly create new modelzoo models in their applications.
    """
    assert hasattr(molflux.modelzoo, "ModelBase")


def test_exports_supports_classification():
    """That the package exposes the supports_classification function."""
    assert hasattr(molflux.modelzoo, "supports_classification")
    assert callable(molflux.modelzoo.supports_classification)


def test_exports_supports_covariance():
    """That the package exposes the supports_covariance function."""
    assert hasattr(molflux.modelzoo, "supports_covariance")
    assert callable(molflux.modelzoo.supports_covariance)


def test_exports_supports_supports_prediction_interval():
    """That the package exposes the supports_prediction_interval function."""
    assert hasattr(molflux.modelzoo, "supports_prediction_interval")
    assert callable(molflux.modelzoo.supports_prediction_interval)


def test_exports_supports_sampling():
    """That the package exposes the supports_sampling function."""
    assert hasattr(molflux.modelzoo, "supports_sampling")
    assert callable(molflux.modelzoo.supports_sampling)


def test_exports_supports_supports_std():
    """That the package exposes the supports_std function."""
    assert hasattr(molflux.modelzoo, "supports_std")
    assert callable(molflux.modelzoo.supports_std)


def test_exports_supports_uncertainty_calibration():
    """That the package exposes the supports_uncertainty_calibration function."""
    assert hasattr(molflux.modelzoo, "supports_uncertainty_calibration")
    assert callable(molflux.modelzoo.supports_uncertainty_calibration)
