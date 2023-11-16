import pytest

import datasets
import molflux.metrics
import molflux.modelzoo
from molflux.core import compute_scores, score_model

parametrized_suites = ["regression"]


@pytest.mark.parametrize("suite", parametrized_suites)
def test_can_score_model(suite):
    """That scoring a model returns scores according to standardised metrics format."""

    fold = datasets.DatasetDict(
        {
            "train": datasets.Dataset.from_dict(
                {
                    "x": [1, 2, 3],
                    "y": [3, 2, 1],
                },
            ),
            "validation": datasets.Dataset.from_dict(
                {
                    "x": [],
                    "y": [],
                },
            ),
            "test": datasets.Dataset.from_dict(
                {
                    "x": [1, 3, 2],
                    "y": [3, 1, 2],
                },
            ),
        },
    )

    model = molflux.modelzoo.load_model(
        "average_regressor",
        x_features=["x"],
        y_features=["y"],
    )
    model.train(fold["train"])

    metrics = molflux.metrics.load_suite(suite)

    scores = score_model(model, fold=fold, metrics=metrics)

    for split_name in fold.keys():
        assert split_name in scores.keys()

        if fold[split_name]:
            for y_feature in model.y_features:
                assert y_feature in scores[split_name]

                score_keys = list(scores[split_name][y_feature])

                for metric in metrics:
                    # the exact names given to score keys are not officially
                    # part of the metrics API, but for now they tend to
                    # always have metric.tag in it so do this instead of
                    # assert metric.tag in scores[split_name][y_feature]
                    assert any(metric.tag in score_key for score_key in score_keys)


def test_splits_not_in_same_order():
    """
    Computing scores should be robust to input references and predictions folds
    containing splits not in the same order.
    """

    references = datasets.DatasetDict(
        {
            "train": datasets.Dataset.from_dict(
                {
                    "x": [1, 2, 3],
                    "y": [3, 2, 1],
                },
            ),
            "validation": datasets.Dataset.from_dict(
                {
                    "x": [],
                    "y": [],
                },
            ),
            "test": datasets.Dataset.from_dict(
                {
                    "x": [1, 3, 2],
                    "y": [3, 1, 2],
                },
            ),
        },
    )

    predictions = datasets.DatasetDict(
        {
            "train": datasets.Dataset.from_dict(
                {
                    "x": [1, 2, 3],
                    "y": [3, 2, 1],
                },
            ),
            "test": datasets.Dataset.from_dict(
                {
                    "x": [1, 3, 2],
                    "y": [3, 1, 2],
                },
            ),
            "validation": datasets.Dataset.from_dict(
                {
                    "x": [],
                    "y": [],
                },
            ),
        },
    )

    metrics = molflux.metrics.load_suite("regression")

    scores = compute_scores(
        predictions=predictions,
        references=references,
        metrics=metrics,
    )

    # If scores have been computed correctly, the validation scores should be None
    # but the test scores should be actual values
    for task in scores["validation"]:
        for score in scores["validation"][task].values():
            assert score is None

    for task in scores["test"]:
        for score in scores["test"][task].values():
            assert score is not None


def test_warning_is_raised_on_dissimilar_tasks():
    """That a warning is raised to our users if computing scores across tasks
    with dissimilar names.

    This is allowed as a convenience way of computing scores between e.g.
    'y' and 'my-model::y' without needing to add an explicit renaming step.
    The disadvantage is that it has the risk of silently allowing unrelated
    tasks to be compared for scoring.
    """

    references = datasets.DatasetDict(
        {
            "train": datasets.Dataset.from_dict(
                {
                    "x": [1, 2, 3],
                    "y": [3, 2, 1],
                    "z": [2, 1, 3],
                },
            ),
        },
    )

    predictions = datasets.DatasetDict(
        {
            "train": datasets.Dataset.from_dict(
                {
                    "x": [1, 2, 3],
                    "my-model::y": [3, 2, 1],
                    "another_task": [2, 1, 3],
                },
            ),
        },
    )

    metrics = molflux.metrics.load_suite("regression")

    with pytest.warns(UserWarning, match="dissimilar names: 'z' vs. 'another_task'"):
        compute_scores(predictions=predictions, references=references, metrics=metrics)
