import os
from abc import ABC, abstractmethod
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Protocol

import joblib
import numpy as np

import datasets
from molflux.modelzoo.errors import NotTrainedError
from molflux.modelzoo.model import ClassificationMixin, ModelBase, _ModelConfigT
from molflux.modelzoo.typing import Classes, PredictionResult
from molflux.modelzoo.utils import (
    get_concatenated_array,
    pick_features,
    validate_features,
)


class _SKLearnBaseAPI(Protocol):
    """The interface of a typical sklearn model.

    We are formalising it ourselves here because sklearn doesn't have a base
    class that defines these two methods.
    """

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None, **kwargs: Any) -> Any:
        ...

    def predict(self, X: np.ndarray, **kwargs: Any) -> np.ndarray:
        ...


class SKLearnModelBase(ModelBase[_ModelConfigT], ABC):
    """Base model for all models based on the sklearn API"""

    @abstractmethod
    def _instantiate_model(self) -> _SKLearnBaseAPI:
        ...

    def _train(
        self,
        train_data: datasets.Dataset,
        **kwargs: Any,
    ) -> Any:
        # validate y features as well
        validate_features(train_data, self.y_features)

        x_data = pick_features(train_data, self.x_features)
        X = get_concatenated_array(x_data, self.x_features)

        y_data = pick_features(train_data, self.y_features)
        y = get_concatenated_array(y_data, self.y_features)

        # reshape (n,1) column-vector into (n,) 1d array, as expected by sklearn
        is_column_vector = (len(y.shape) > 1) and (y.shape[1] == 1)
        if is_column_vector:
            y = y.ravel()

        # instantiate model
        self.model = self._instantiate_model()

        # train
        self.model.fit(X, y, **kwargs)

    def _predict(
        self,
        data: datasets.Dataset,
        **kwargs: Any,
    ) -> PredictionResult:
        display_names = self._predict_display_names

        if not len(data):
            return {display_name: [] for display_name in display_names}

        X = get_concatenated_array(data, self.x_features)
        y_predict: np.ndarray = self.model.predict(X, **kwargs)
        # this could be of several shapes / types:
        # - singletask: (n_samples,)
        # - multitask: (n_tasks, n_samples)
        # lets align...
        if np.ndim(y_predict) == 1:
            y_predict = np.expand_dims(y_predict, axis=1)

        n_samples, n_tasks = np.shape(y_predict)

        # Check that output tasks match what specified
        if n_tasks != len(self.y_features):
            raise ValueError(
                f"Predictions do not have expected number of tasks: {np.shape(y_predict)!r}",
            )

        # (n_tasks, n_samples)
        return {
            display_name: task_predictions.tolist()
            for display_name, task_predictions in zip(display_names, y_predict.T)
        }

    @property
    def config(self) -> Dict[str, Any]:
        return asdict(self.model_config)

    def as_dir(self, directory: str) -> None:
        """Serialises a pre-trained model in a directory."""

        if self.model is None:
            raise NotTrainedError

        filename = os.path.join(directory, "model.joblib")
        joblib.dump(self.model, filename=filename, compress=0)

    def from_dir(self, directory: str) -> None:
        """Deserialises the backend model object stored in a given directory.

        This should undo the process defined in `self.as_dir()`.
        """

        try:
            filename = os.path.join(directory, "model.joblib")
            self.model = joblib.load(filename)
            return
        except FileNotFoundError as err:
            raise FileNotFoundError(
                f"Could not find scikit-learn model binaries: expected {filename!r}",
            ) from err


class _SKLearnClassifierAPI(Protocol):
    """The interface of a typical sklearn classifier.

    We are formalising it ourselves here because sklearn doesn't have a formal
    class that defines it
    """

    classes_: List[str]

    def predict_proba(self, X: np.ndarray, **kwargs: Any) -> np.ndarray:
        ...


class SKLearnClassificationMixin(ClassificationMixin):
    model: _SKLearnClassifierAPI
    tag: str

    @property
    def classes(self) -> Classes:
        if not hasattr(self.model, "classes_"):
            return {}

        sklearn_classes = self.model.classes_
        # this could be of several shapes / types
        # - singletask: (n_classes,)
        # - multitask: (n_tasks, n_classes)

        is_multitask = np.ndim(sklearn_classes) == 2
        classes = sklearn_classes if is_multitask else [sklearn_classes]  # type: ignore[list-item]

        return {
            task: classes.tolist() for task, classes in zip(self.y_features, classes)  # type: ignore[attr-defined]
        }

    def _predict_proba(self, data: datasets.Dataset, **kwargs: Any) -> PredictionResult:
        display_names = self._predict_proba_display_names

        if not len(data):
            return {display_name: [] for display_name in display_names}

        X = get_concatenated_array(data, self.x_features)
        y_predict: np.ndarray = self.model.predict_proba(X, **kwargs)
        # this could be of several shapes / types:
        # - singletask: (n_samples, n_classes)
        # - multitask: (n_tasks, n_samples, n_classes)
        # lets align...
        if np.ndim(y_predict) == 2:
            y_predict = np.expand_dims(y_predict, axis=0)

        n_tasks, n_samples, n_classes = np.shape(y_predict)

        # Check that output tasks match what specified
        if n_tasks != len(self.y_features):
            raise ValueError(
                f"Predictions do not have expected number of tasks: {np.shape(y_predict)!r}",
            )

        results = {}
        for task, task_predictions, display_name in zip(
            self.y_features,
            y_predict,
            display_names,
        ):
            n_samples, n_classes = np.shape(task_predictions)

            # Check that output classes match what specified
            if n_classes != len(self.classes[task]):
                raise ValueError(
                    f"Predictions do not have expected number of classes: {np.shape(task_predictions)!r}",
                )

            results[display_name] = task_predictions.tolist()

        # (n_tasks, n_samples, n_classes)
        return results
