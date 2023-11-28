from dataclasses import asdict
from typing import TYPE_CHECKING, Any, Dict, Literal, Type

from pydantic.dataclasses import dataclass

import datasets
from molflux.modelzoo.info import ModelInfo
from molflux.modelzoo.models.pyod import (
    PyODClassificationMixin,
    PyODModelBase,
    PyODModelConfig,
)
from molflux.modelzoo.typing import PredictionResult
from molflux.modelzoo.utils import get_concatenated_array, pick_features

try:
    if TYPE_CHECKING:
        import numpy as np
    from pyod.models.abod import ABOD
except ImportError as e:
    from molflux.modelzoo.errors import ExtrasDependencyImportError

    raise ExtrasDependencyImportError("pyod", e) from None

ProbabilityConversionMethod = Literal["linear", "unify"]

_DESCRIPTION = """
ABOD class for Angle-base Outlier Detection.

For an observation, the variance of its weighted cosine scores to all
neighbours could be viewed as the outlying score.
See :cite:`kriegel2008angle` for details.

Two version of ABOD are supported:

- Fast ABOD: use k nearest neighbours to approximate.
- Original ABOD: consider all training points with high time complexity at
  O(n^3).
"""

_CONFIG_DESCRIPTION = """
Parameters
----------
contamination : float in (0., 0.5), optional (default=0.1)
    The amount of contamination of the data set, i.e.
    the proportion of outliers in the data set. Used when fitting to
    define the threshold on the decision function.

n_neighbours : int, optional (default=10)
    Number of neighbours to use by default for k neighbours queries.

method: str, optional (default='fast')
    Valid values for metric are:

    - 'fast': fast ABOD. Only consider n_neighbours of training points
    - 'default': original ABOD with all training points, which could be
      slow
"""

Method = Literal["default", "fast"]


class Config:
    arbitrary_types_allowed = True
    extra = "forbid"


@dataclass(config=Config)
class ABODDetectorConfig(PyODModelConfig):
    contamination: float = 0.1
    n_neighbours: int = 10
    method: Method = "fast"


class ABODDetector(PyODClassificationMixin, PyODModelBase[ABODDetectorConfig]):
    @property
    def config(self) -> Dict[str, Any]:
        return asdict(self.model_config)

    @property
    def _config_builder(self) -> Type[ABODDetectorConfig]:
        return ABODDetectorConfig

    def _info(self) -> ModelInfo:
        return ModelInfo(
            model_description=_DESCRIPTION,
            config_description=_CONFIG_DESCRIPTION,
        )

    def _instantiate_model(self) -> ABOD:
        config = self.model_config
        return ABOD(
            contamination=config.contamination,
            n_neighbors=config.n_neighbours,
            method=config.method,
        )

    def _train(self, train_data: datasets.Dataset, **kwargs: Any) -> Any:
        # Developer note - this function requires special handling as it raises
        # an error on input datasets with non-float columns
        x_data = pick_features(train_data, self.x_features)
        X = get_concatenated_array(x_data, self.x_features)

        # ABOD detectors need float vectors
        X_float: np.ndarray = X.astype(float)

        # instantiate model
        self.model = self._instantiate_model()

        # train (unsupervised learning)
        self.model.fit(X_float)

    def _predict(self, data: datasets.Dataset, **kwargs: Any) -> PredictionResult:
        # Developer note - this function requires special handling as it raises
        # an error on input datasets with non-float columns
        display_names = self._predict_display_names

        if not len(data):
            return {display_name: [] for display_name in display_names}

        x_data = pick_features(data, self.x_features)
        X = get_concatenated_array(x_data, self.x_features)

        # ABOD detectors need float vectors
        X_float: np.ndarray = X.astype(float)
        y_predict: np.ndarray = self.model.predict(X_float, return_confidence=False)

        # Outlier detectors are single task models
        return {display_name: y_predict.tolist() for display_name in display_names}

    def _predict_proba(
        self,
        data: datasets.Dataset,
        method: ProbabilityConversionMethod = "linear",
        **kwargs: Any,
    ) -> PredictionResult:
        # Developer note - this function requires special handling as it raises
        # an error on input datasets with non-float columns

        display_names = self._predict_proba_display_names

        if not len(data):
            return {display_name: [] for display_name in display_names}

        x_data = pick_features(data, self.x_features)
        X = get_concatenated_array(x_data, self.x_features)

        # ABOD detectors need float vectors
        X_float: np.ndarray = X.astype(float)

        y_proba: np.ndarray = self.model.predict_proba(
            X_float,
            method=method,
            return_confidence=False,
        )
        # Outlier detectors are single task models
        return {display_name: y_proba.tolist() for display_name in display_names}
