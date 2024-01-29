import pytest
from cloudpathlib import AnyPath

import datasets
import molflux.modelzoo
from molflux.core import load_model, save_model

parametrized_output_directory_fixtures = [
    "tmp_path",
    "fixture_test_bucket",
    "fixture_nested_tmp_path",
    "fixture_nested_test_bucket",
]


@pytest.mark.parametrize("output_directory", parametrized_output_directory_fixtures)
def test_save_model_creates_standardised_artefact(output_directory, request):
    """That models are saved into a standardised artefact directory."""
    output_directory = request.getfixturevalue(output_directory)

    dataset = datasets.Dataset.from_dict({"x": [1, 2, 3], "y": [1, 2, 3]})
    featurisation_metadata = {"version": 1, "config": []}
    model = molflux.modelzoo.load_model(
        "linear_regressor",
        x_features=["x"],
        y_features=["y"],
    )

    model.train(dataset)

    artefact_directory = save_model(
        model,
        path=output_directory,
        featurisation_metadata=featurisation_metadata,
    )
    artefact_path = AnyPath(artefact_directory)

    assert artefact_path.is_dir()  # type: ignore[attr-defined]

    ls = list(artefact_path.glob("**/*"))  # type: ignore[attr-defined]

    assert artefact_path / "featurisation_metadata.json" in ls  # type: ignore[operator]

    # optional for now (no clients depend on this)
    assert artefact_path / "requirements.txt" in ls  # type: ignore[operator]

    # these are an implementation detail of molflux.modelzoo.save_model()
    # we test them here just in case but could technically change without worries
    # as long as molflux.modelzoo also knows how to read them back
    assert artefact_path / "model_config.json" in ls  # type: ignore[operator]
    assert artefact_path / "model_artefacts" in ls  # type: ignore[operator]


def test_save_model_with_no_featurisation_metadata_raises_warning(tmp_path):
    """That a warning is raised if saving models without the associated
    featurisation metadata that was used to featurise the training dataset.

    Users that do not want to associate featurisation metadata should be able
    to communicate that by passing in None or an empty dict.
    """

    dataset = datasets.Dataset.from_dict({"x": [1, 2, 3], "y": [1, 2, 3]})
    model = molflux.modelzoo.load_model(
        "linear_regressor",
        x_features=["x"],
        y_features=["y"],
    )

    model.train(dataset)

    with pytest.warns(UserWarning, match="featurisation metadata"):
        save_model(model, path=tmp_path, featurisation_metadata=None)

    with pytest.warns(UserWarning, match="featurisation metadata"):
        save_model(model, path=tmp_path, featurisation_metadata={})


@pytest.mark.parametrize("output_directory", parametrized_output_directory_fixtures)
def test_can_load_back_saved_model(output_directory, request):
    """That can load a model that has been saved."""

    output_directory = request.getfixturevalue(output_directory)
    dataset = datasets.Dataset.from_dict({"x": [1, 2, 3], "y": [1, 2, 3]})
    featurisation_metadata = {"version": 1, "config": []}
    model = molflux.modelzoo.load_model(
        "linear_regressor",
        x_features=["x"],
        y_features=["y"],
    )

    model.train(dataset)
    model_path = save_model(
        model,
        path=output_directory,
        featurisation_metadata=featurisation_metadata,
    )

    reloaded_model = load_model(model_path)

    assert reloaded_model.name == "linear_regressor"
