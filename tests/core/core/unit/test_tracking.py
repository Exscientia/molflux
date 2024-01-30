import itertools
from typing import TYPE_CHECKING

import pytest

import datasets
import molflux.core.tracking
import molflux.modelzoo
import molflux.splits

if TYPE_CHECKING:
    from pathlib import Path

parametrized_output_directory_fixtures = [
    "tmp_path",
    "fixture_test_bucket",
    "fixture_nested_tmp_path",
    "fixture_nested_test_bucket",
]

file_formats = ["parquet", "csv", "json"]


@pytest.mark.parametrize("output_directory", parametrized_output_directory_fixtures)
def test_log_params_creates_json(output_directory, request):
    """That can log arbitrary param payloads to json."""

    output_directory = request.getfixturevalue(output_directory)
    params = {"a": [1, 2, 3], "b": "value"}
    output_path: Path = output_directory / "params.json"

    molflux.core.tracking.log_params(params, path=output_path)
    assert output_path.is_file()


# TODO(avianello) mock the datasets S3 callstack to test saving datasets on s3 too
@pytest.mark.parametrize(
    "output_directory, format",
    itertools.product(parametrized_output_directory_fixtures[:1], file_formats),
)
def test_log_dataset_saves_dataset_as_persisted_file_format(
    output_directory,
    format,
    request,
):
    """That can log an arbitrary dataset as persisted file in a variety of formats."""

    output_directory = request.getfixturevalue(output_directory)
    dataset = datasets.Dataset.from_dict({"x1": [1, 2, 3], "x2": [1, 2, 3]})
    output_path: Path = output_directory / f"data.{format}"

    molflux.core.tracking.log_dataset(dataset, path=output_path)
    assert output_path.is_file()


# TODO(avianello) mock the datasets S3 callstack to test saving datasets on s3 too
@pytest.mark.parametrize("output_directory", parametrized_output_directory_fixtures[:1])
def test_log_featurised_dataset_saves_dataset_as_canonical_file(
    output_directory,
    request,
):
    """That featurised datasets are logged according to standard format."""

    output_directory = request.getfixturevalue(output_directory)
    dataset = datasets.Dataset.from_dict({"x1": [1, 2, 3], "x2": [1, 2, 3]})
    molflux.core.tracking.log_featurised_dataset(dataset, directory=output_directory)

    expected_output_path = output_directory / "featurised_dataset.parquet"
    assert expected_output_path.is_file()


# TODO(avianello) mock the datasets S3 callstack to test saving datasets on s3 too
@pytest.mark.parametrize("output_directory", parametrized_output_directory_fixtures[:1])
def test_log_fold_saves_fold_as_canonical_files(output_directory, request):
    """That folds are logged according to standard format."""

    output_directory = request.getfixturevalue(output_directory)
    split = datasets.Dataset.from_dict({"x1": [1, 2, 3], "x2": [1, 2, 3]})
    fold = datasets.DatasetDict({"train": split, "validation": split, "test": split})
    molflux.core.tracking.log_fold(fold, directory=output_directory)

    expected_output_paths = [
        output_directory / f"{split_name}.parquet" for split_name in fold.keys()
    ]
    for output_path in expected_output_paths:
        assert output_path.is_file()


@pytest.mark.parametrize("output_directory", parametrized_output_directory_fixtures)
def test_log_model_params_creates_canonical_file(output_directory, request):
    """That the log_model_params function saves model metadata in a file
    called model_params.json"""

    output_directory = request.getfixturevalue(output_directory)
    model = molflux.modelzoo.load_model("random_forest_regressor")
    molflux.core.tracking.log_model_params(model, directory=output_directory)

    expected_output_path: Path = output_directory / "model_params.json"
    assert expected_output_path.is_file()


@pytest.mark.parametrize("output_directory", parametrized_output_directory_fixtures)
def test_log_pipeline_config_creates_canonical_file(output_directory, request):
    """That the log_pipeline_config function saves pipeline config metadata
    in a file called pipeline.json"""

    output_directory = request.getfixturevalue(output_directory)
    config = {"a": [1, 2, 3], "b": "value"}
    molflux.core.tracking.log_pipeline_config(config, directory=output_directory)

    expected_output_path: Path = output_directory / "pipeline.json"
    assert expected_output_path.is_file()


@pytest.mark.parametrize("output_directory", parametrized_output_directory_fixtures)
def test_log_scores_creates_canonical_file(output_directory, request):
    """That the log_scores function saves model scores in a file called scores.json"""

    output_directory = request.getfixturevalue(output_directory)
    scores = {
        "train": {"y1": {"r2": 0.99}},
        "validation": {"y1": {"r2": 0.99}},
        "test": {"y1": {"r2": 0.99}},
    }
    molflux.core.tracking.log_scores(scores, directory=output_directory)

    expected_output_path: Path = output_directory / "scores.json"
    assert expected_output_path.is_file()


@pytest.mark.parametrize("output_directory", parametrized_output_directory_fixtures)
def test_log_splitting_strategy_creates_canonical_file(output_directory, request):
    """That the log_splitting_strategy function saves splitting strategy
    metadata in a file called splitting_strategy.json"""

    output_directory = request.getfixturevalue(output_directory)
    splitting_strategy = molflux.splits.load_splitting_strategy("shuffle_split")
    molflux.core.tracking.log_splitting_strategy(
        splitting_strategy,
        directory=output_directory,
    )

    expected_output_path: Path = output_directory / "splitting_strategy.json"
    assert expected_output_path.is_file()
