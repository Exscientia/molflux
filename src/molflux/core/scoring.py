import warnings
from collections import defaultdict
from enum import Enum
from typing import Any, DefaultDict, Dict, Final, Mapping, MutableMapping, Optional

import mergedeep
from thefuzz import fuzz

from datasets import Dataset, DatasetDict
from molflux.core.models import get_references, predict
from molflux.core.typing import FoldScores, TasksScores
from molflux.metrics import Metrics
from molflux.modelzoo import Model


def _compute_scores(
    predictions: Dataset,
    references: Dataset,
    metrics: Metrics,
    compute_kwargs: Any,
) -> TasksScores:
    """Computes scores from model predictions and ground truth references.

    Args:
        predictions: The dataset of predictions following the standardised
            predictions format. With predicted features appearing in the same
            order defined by the model architecture.
        references: The corresponding prediction references. Columns are expected
            to match the corresponding order in predictions.
        metrics: The metrics with which to score the model.
        compute_kwargs: A dictionary of extra kwargs to use for computing
            metrics.

    Returns:
        A dictionary with predicted tasks as keys and a dictionary of scores as
        values.
    """

    # some predictions/references might be empty (e.g. from cross-validation strategy splits)
    # return an empty dictionary with empty scores dict for each task
    if not predictions.num_rows or not references.num_rows:
        return dict.fromkeys(references.column_names, {})

    references_dict, predictions_dict = references.to_dict(), predictions.to_dict()

    tasks_scores: TasksScores = {}
    for (task_name, task_references), (predicted_task_name, task_predictions) in zip(
        references_dict.items(),
        predictions_dict.items(),
    ):
        # we are allowing predicted tasks to be named differently than
        # reference tasks (e.g. my-model::y1). It's up to the user to make
        # sure that they are ordered correctly to match their respective
        # reference tasks!
        similarity_score = fuzz.token_set_ratio(task_name, predicted_task_name)
        if similarity_score < 100:  # favour false positives
            warnings.warn(
                f"Scores are being computed between reference and predicted tasks with dissimilar names: {task_name!r} vs. {predicted_task_name!r}. Make sure these are equivalent for results to be valid.",
                UserWarning,
                stacklevel=1,
            )

        tasks_scores[task_name] = metrics.compute(
            predictions=task_predictions,
            references=task_references,
            **compute_kwargs,
        )

    return tasks_scores


def compute_scores(
    predictions: DatasetDict,
    references: DatasetDict,
    metrics: Metrics,
    scoring_kwargs: Optional[Dict[str, Dict[str, Any]]] = None,
) -> FoldScores:
    """Computes scores for a model based on given fold predictions and references.

    Args:
        predictions: The dataset dict of model predictions for each split in the fold.
        references: The dataset dict of model prediction references for each split in the fold.
        metrics: The metrics to score the model with.
        scoring_kwargs: A dictionary of extra kwargs to use for computing
            metrics on each split in the fold.

    Returns:
        A dictionary with split names as keys, and dictionaries of scores for
        each task as values.
    """

    # for each split...
    fold_scores: FoldScores = {}
    for split_name, references_dataset in references.items():
        # make sure malformed kwargs do not go through silently
        if scoring_kwargs and split_name not in scoring_kwargs:
            raise KeyError(f"Please provide scoring kwargs for split {split_name!r}")

        if split_name not in predictions:
            raise KeyError(f"No predictions found for split {split_name!r}")
        predictions_dataset = predictions[split_name]

        kwargs = scoring_kwargs[split_name] if scoring_kwargs else {}
        fold_scores[split_name] = _compute_scores(
            predictions=predictions_dataset,
            references=references_dataset,
            metrics=metrics,
            compute_kwargs=kwargs,
        )

    # Empty splits will have returned empty scores dicts
    # backfill them with metrics names and None scores
    metrics_names = [  # noqa
        list(task_scores.keys())
        for tasks_scores in fold_scores.values()
        for task_scores in tasks_scores.values()
    ][0]
    for split_name, tasks_scores in fold_scores.items():
        for task_name, task_scores in tasks_scores.items():
            if not task_scores:
                fold_scores[split_name][task_name] = dict.fromkeys(metrics_names, None)

    return fold_scores


def score_model(
    model: Model,
    fold: DatasetDict,
    metrics: Metrics,
    prediction_kwargs: Optional[Dict[str, Dict[str, Any]]] = None,
    scoring_kwargs: Optional[Dict[str, Dict[str, Any]]] = None,
) -> FoldScores:
    """Scores a model on a given fold (DatasetDict) based on a suite of metrics.

    This is a higher-level wrapper of 'compute_scores', for when accessing or
    storing prediction results is not needed.

    Args:
        model: The model to score.
        fold: The fold to score the model on.
        metrics: The metrics to score the model with.
        prediction_kwargs: A dictionary of extra prediction kwargs to use for
            each split in the fold.
        scoring_kwargs: A dictionary of extra kwargs to use for computing
            metrics on each split in the fold.

    Returns:
        A dictionary with split names as keys, and dictionaries of scores for
        each task as values.
    """

    # other prediction methods would return a tuple of prediction results,
    # which makes scoring ambiguous. Users should use the more fine-grained
    # compute_scores instead for such cases
    prediction_method: Final = "predict"

    references = get_references(model=model, fold=fold)
    predictions = predict(
        model=model,
        fold=fold,
        prediction_method=prediction_method,
        prediction_kwargs=prediction_kwargs,
    )

    return compute_scores(
        predictions=predictions,
        references=references,
        metrics=metrics,
        scoring_kwargs=scoring_kwargs,
    )


def invert_scores_hierarchy(scores: FoldScores) -> FoldScores:
    """Invert a dictionary of scores per task per split into a dictionary
    of scores per split per task, or vice-versa.

    This is a convenience wrapper to be used in situations where it might be
    easier to iterate over scores across tasks rather than splits.
    """
    flipped: DefaultDict[str, Dict[str, Any]] = defaultdict(dict)
    for key, val in scores.items():
        for subkey, subval in val.items():
            flipped[subkey][key] = subval
    return dict(flipped)


class Strategy(Enum):
    """Merging strategy.

    References:
         https://github.com/clarketm/mergedeep/blob/master/mergedeep/mergedeep.py#L9
    """

    REPLACE = 0
    ADDITIVE = 1
    TYPESAFE = 2
    TYPESAFE_REPLACE = 3
    TYPESAFE_ADDITIVE = 4


def merge_scores(
    destination: MutableMapping,
    *sources: Mapping,
    strategy: Strategy = Strategy.REPLACE,
) -> FoldScores:
    return mergedeep.merge(destination, *sources, strategy=strategy)  # type: ignore[no-any-return]
