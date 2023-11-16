import warnings
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple, Union

import datasets
import molflux.modelzoo
from datasets import Dataset, DatasetDict
from molflux.core.environment import save_python_environment
from molflux.core.featurisation.metadata import save_featurisation_metadata
from molflux.core.typing import AllowedInferenceMethods, PathLike
from molflux.modelzoo import Model


def save_model(
    model: Model,
    path: PathLike,
    featurisation_metadata: Optional[Dict[str, Any]],
) -> str:
    """Packages a trained model to disk into a standard format.

    This standardised format ensures that models can be deployed in production
    and used seamlessly in a variety of downstream tools.

    .. code-block: shell
        model/
        ├── model_config.json
        ├── model_artefacts/
        |   └── model.pkl
        ├── featurisation_metadata.json
        └── requirements.txt

    Args:
        model: The model to save.
        path: The path under which to save the model.
        featurisation_metadata: The featurisation metadata for the features the
            model was trained on.

    Returns:
        A reference to the path the model was saved to.
    """

    # save model to store
    molflux.modelzoo.save_to_store(key=path, model=model)

    # save featurisation config alongside of it
    if not featurisation_metadata:
        warnings.warn(
            "You have saved a model but did not track its featurisation metadata. "
            "Users might not be able to automatically inference your model from unfeaturised inputs",
            UserWarning,
            stacklevel=1,
        )

    save_featurisation_metadata(
        featurisation_metadata=featurisation_metadata,
        path=path,
    )

    # save python environment metadata alongside of it
    save_python_environment(path=path)

    return str(path)


def load_model(path: str) -> Model:
    """Loads a model saved to disk."""
    return molflux.modelzoo.load_from_store(key=path)


def get_references(model: Model, fold: DatasetDict) -> DatasetDict:
    """Extracts the model ground truth references from a fold."""
    return fold.select_columns(column_names=model.y_features)


def get_inputs(model: Model, fold: DatasetDict) -> DatasetDict:
    """Extracts the model's input features from a fold."""
    return fold.select_columns(column_names=model.x_features)


def _dict_of_iterables_to_iterable_of_dicts(
    d: Mapping[str, Iterable[Any]],
) -> Iterable[Dict[str, Any]]:
    """Split dictionary of iterables into iterable of dictionaries.

    References:
        https://stackoverflow.com/a/1780295
    """
    return map(dict, zip(*[[(k, v) for v in value] for k, value in d.items()]))


def predict(
    model: Model,
    fold: DatasetDict,
    prediction_method: AllowedInferenceMethods = "predict",
    prediction_kwargs: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Union[DatasetDict, Tuple[DatasetDict, ...]]:
    """Calculates model predictions for each split in a fold.

    Model predictions follow the molflux.modelzoo predictions output format, where
    predictions are returned for each predicted task.

    Args:
        model: The model to use for predictions.
        fold: The fold of prediction inputs.
        prediction_method: The method to use for predictions. Should be one of
            ["predict", "predict_proba", "predict_with_prediction_interval",
            "predict_with_std", "sample"]. Defaults to "predict".
        prediction_kwargs: A dictionary of extra prediction kwargs to use for
            each split in the fold.

    Returns:
        DatasetDict(s) of prediction results.
    """

    # Initialise the dictionary of prediction outputs for each input split dataset
    # ensuring same ordering of split names for more intuitive behaviour
    fold_prediction_outputs: Dict[str, Tuple[DatasetDict, ...]] = dict.fromkeys(
        fold,
        (DatasetDict(),),
    )

    for split_name, dataset in fold.items():
        # make sure malformed kwargs do not go through silently
        if prediction_kwargs and split_name not in prediction_kwargs:
            raise KeyError(f"Please provide prediction kwargs for split {split_name!r}")

        kwargs = prediction_kwargs[split_name] if prediction_kwargs else {}
        prediction_results = inference(
            model,
            dataset=dataset,
            prediction_method=prediction_method,
            prediction_kwargs=kwargs,
        )

        if not isinstance(prediction_results, tuple):
            prediction_results = (prediction_results,)

        fold_prediction_outputs[split_name] = prediction_results

    # An iterable of dicts of Datasets, one for each prediction_method item
    # {"train": (y, ...), "validation": (y, ...), "test": (y, ...)}
    # ->
    # (DatasetDict({"train": y, "validation": y, "test": y}), ...)
    out = tuple(
        DatasetDict(fold_predictions)
        for fold_predictions in _dict_of_iterables_to_iterable_of_dicts(
            fold_prediction_outputs,
        )
    )
    return out[0] if len(out) == 1 else out


def inference(
    model: Model,
    dataset: Dataset,
    prediction_method: AllowedInferenceMethods = "predict",
    prediction_kwargs: Optional[Dict[str, Any]] = None,
) -> Union[Dataset, Tuple[Dataset, ...]]:
    """Calculates model predictions on a single dataset."""

    predictor = getattr(model, prediction_method)
    prediction_results = predictor(dataset, **(prediction_kwargs or {}))

    if isinstance(prediction_results, tuple):
        return tuple(datasets.Dataset.from_dict(d) for d in prediction_results)
    else:
        return datasets.Dataset.from_dict(prediction_results)
