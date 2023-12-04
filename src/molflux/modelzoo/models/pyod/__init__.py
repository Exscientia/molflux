from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Literal, TypeVar

from pydantic.dataclasses import dataclass

import datasets

if TYPE_CHECKING:
    import numpy as np

from molflux.modelzoo.model import ClassificationMixin, ModelBase, ModelConfig
from molflux.modelzoo.typing import Classes, PredictionResult
from molflux.modelzoo.utils import get_concatenated_array, pick_features

try:
    import pyod.models.base
except ImportError as e:
    from molflux.modelzoo.errors import ExtrasDependencyImportError

    raise ExtrasDependencyImportError("pyod", e) from None

ProbabilityConversionMethod = Literal["linear", "unify"]


class Config:
    arbitrary_types_allowed = True
    extra = "forbid"


@dataclass(config=Config)
class PyODModelConfig(ModelConfig):
    def __post_init_post_parse__(self) -> None:
        if self.y_features:
            if len(self.y_features) != 1:
                raise ValueError(
                    f"Unsupervised learning models only accepts an optional y_features "
                    f"list with a single molflux: got {self.y_features}",
                )
        else:
            # TODO (dm 12 Oct 23) Is this sensible?
            self.train_features = self.x_features
            self.y_features = ["outliers"]


_PyODModelConfigT = TypeVar("_PyODModelConfigT", bound=PyODModelConfig)


class PyODModelBase(ModelBase[_PyODModelConfigT], ABC):
    """Base model for all models based on the PyOD API"""

    @abstractmethod
    def _instantiate_model(self) -> pyod.models.base.BaseDetector:
        ...

    def _train(self, train_data: datasets.Dataset, **kwargs: Any) -> Any:
        x_data = pick_features(train_data, self.x_features)
        X = get_concatenated_array(x_data, self.x_features)

        # instantiate model
        self.model = self._instantiate_model()

        # train (unsupervised learning)
        self.model.fit(X)

    def _predict(self, data: datasets.Dataset, **kwargs: Any) -> PredictionResult:
        display_names = self._predict_display_names

        if not len(data):
            return {display_name: [] for display_name in display_names}

        x_data = pick_features(data, self.x_features)
        X = get_concatenated_array(x_data, self.x_features)
        y_predict: np.ndarray = self.model.predict(X, return_confidence=False)

        # Outlier detectors are single task models
        return {display_name: y_predict.tolist() for display_name in display_names}


class PyODClassificationMixin(ClassificationMixin):
    model: pyod.models.base.BaseDetector
    tag: str

    @property
    def classes(self) -> Classes:
        if not hasattr(self.model, "_classes"):
            return {}
        return {self.model.y_features[0]: list(self.model._classes)}

    def _predict_proba(
        self,
        data: datasets.Dataset,
        method: ProbabilityConversionMethod = "linear",
        **kwargs: Any,
    ) -> PredictionResult:
        display_names = self._predict_proba_display_names

        if not len(data):
            return {display_name: [] for display_name in display_names}

        x_data = pick_features(data, self.x_features)
        X = get_concatenated_array(x_data, self.x_features)
        y_proba: np.ndarray = self.model.predict_proba(
            X,
            method=method,
            return_confidence=False,
        )
        # Outlier detectors are single task models
        return {display_name: y_proba.tolist() for display_name in display_names}
