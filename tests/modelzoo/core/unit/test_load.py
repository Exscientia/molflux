import pytest

from molflux.modelzoo.load import load_from_dict, load_model


def test_raises_not_implemented_for_unknown_model():
    """That a NotImplementedError is raised if attempting to load an unknown
    model."""
    name = "unknown"
    with pytest.raises(NotImplementedError):
        load_model(name=name)


def test_raised_error_for_unknown_model_provides_close_matches():
    """That the error raised when attempting to load an unknown model
    shows possible close matches to the user (if any)."""
    name = "random_forst"
    # This should suggest e.g. ["random_forest_classifier", "random_forest_regressor"]
    with pytest.raises(
        NotImplementedError,
        match="You might be looking for one of these",
    ):
        load_model(name=name)


def test_dict_missing_required_fields_raises():
    """That cannot load a model with a config missing required fields."""
    config = {"unknown_key": "value"}
    with pytest.raises(SyntaxError):
        load_from_dict(config)
