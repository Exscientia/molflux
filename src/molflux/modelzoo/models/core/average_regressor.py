from dataclasses import asdict
from typing import Any, Callable, Dict, Literal, Tuple, Type

import numpy as np
import scipy.stats as st
from pydantic.dataclasses import dataclass

import datasets
from molflux.modelzoo.info import ModelInfo
from molflux.modelzoo.model import (
    ModelBase,
    ModelConfig,
    PredictionIntervalMixin,
    SamplingMixin,
    StandardDeviationMixin,
    UncertaintyCalibrationMixin,
)
from molflux.modelzoo.typing import PredictionResult
from molflux.modelzoo.utils import validate_features

_DESCRIPTION = """
A core model that predicts the average (median/mean) of the dataset for each predicted variable
"""

_CONFIG_DESCRIPTION = """
Parameters
----------
average_method: Literal, default = "median". Method by which to summarise the average of the dataset.
    Available options are median or mean. median is preferred since it is robust to outliers.
deviation_method: Literal, default = "mad". Method by which to summarise the deviation of the dataset.
    Available options are std or mad. Note that mad is the median absolute deviation which is a version
    of standard deviation robust to outlier values.
"""

# mapping from method strings to fns
_AVERAGE_METHOD_DICT: Dict[str, Callable] = {
    "median": np.nanmedian,
    "mean": np.nanmean,
}
_DEVIATION_METHOD_DICT: Dict[str, Callable] = {
    "std": np.nanstd,
    "mad": st.median_abs_deviation,
}


class Config:
    arbitrary_types_allowed = True
    extra = "forbid"


@dataclass(config=Config)
class AverageRegressorConfig(ModelConfig):
    average_method: Literal["median", "mean"] = "median"
    deviation_method: Literal["std", "mad"] = "mad"


class AverageRegressor(
    UncertaintyCalibrationMixin,
    PredictionIntervalMixin,
    StandardDeviationMixin,
    SamplingMixin,
    ModelBase[AverageRegressorConfig],
):
    @property
    def config(self) -> Dict[str, Any]:
        return asdict(self.model_config)

    @property
    def _config_builder(self) -> Type[AverageRegressorConfig]:
        return AverageRegressorConfig

    def _info(self) -> ModelInfo:
        """Initialises the ModelInfo object.

        To be implemented by subclasses.
        """
        return ModelInfo(
            model_description=_DESCRIPTION,
            config_description=_CONFIG_DESCRIPTION,
        )

    def _train(self, train_data: datasets.Dataset, **kwargs: Any) -> Any:
        # validate y features as well
        validate_features(train_data, self.y_features)

        average_fn = _AVERAGE_METHOD_DICT[self.model_config.average_method]
        deviation_fn = _DEVIATION_METHOD_DICT[self.model_config.deviation_method]
        # apply to each variable and save result as the "model"
        self.model = {
            "mu": [average_fn(train_data[y]) for y in self.y_features],
            "sigma": [deviation_fn(train_data[y]) for y in self.y_features],
        }

    def _predict(self, data: datasets.Dataset, **kwargs: Any) -> PredictionResult:
        display_names = self._predict_display_names

        if not len(data):
            return {display_name: [] for display_name in display_names}

        return {
            display_name: np.full(len(data), mu).tolist()
            for display_name, mu in zip(display_names, self.model["mu"])
        }

    def _predict_with_std(
        self,
        data: datasets.Dataset,
        **kwargs: Any,
    ) -> Tuple[PredictionResult, PredictionResult]:
        (
            prediction_display_names,
            prediction_std_display_names,
        ) = self._predict_with_std_display_names

        if not len(data):
            return {display_name: [] for display_name in prediction_display_names}, {
                display_name: [] for display_name in prediction_std_display_names
            }

        return {
            display_name: np.full(len(data), mu).tolist()
            for display_name, mu in zip(prediction_display_names, self.model["mu"])
        }, {
            display_name: np.full(len(data), sigma).tolist()
            for display_name, sigma in zip(
                prediction_std_display_names,
                self.model["sigma"],
            )
        }

    def _predict_with_prediction_interval(
        self,
        data: datasets.Dataset,
        confidence: float,
        **kwargs: Any,
    ) -> Tuple[PredictionResult, PredictionResult]:
        (
            prediction_display_names,
            prediction_interval_display_names,
        ) = self._predict_with_prediction_interval_display_names

        if not len(data):
            return {display_name: [] for display_name in prediction_display_names}, {
                display_name: [] for display_name in prediction_interval_display_names
            }

        # get the result dictionary of means and standard deviations
        prediction_mean_results, prediction_std_results = self._predict_with_std(data)

        # for each column, map the mean and standard deviation to a prediction interval
        prediction_results: PredictionResult = {}
        prediction_prediction_interval_results: PredictionResult = {}
        for prediction_display_name, prediction_interval_display_name, mean, std in zip(
            prediction_display_names,
            prediction_interval_display_names,
            prediction_mean_results.values(),
            prediction_std_results.values(),
        ):
            # compute the prediction interval
            lower_bound, upper_bound = st.norm.interval(
                confidence,
                loc=mean,
                scale=std,
            )

            # where nans occur (ex. from a 0 standard deviation), use the mean instead
            lower_bound = np.where(~np.isnan(lower_bound), lower_bound, mean)
            upper_bound = np.where(~np.isnan(upper_bound), upper_bound, mean)

            prediction_results[prediction_display_name] = mean
            prediction_prediction_interval_results[
                prediction_interval_display_name
            ] = list(zip(lower_bound, upper_bound))

        return prediction_results, prediction_prediction_interval_results

    def _sample(
        self,
        data: datasets.Dataset,
        n_samples: int,
        **kwargs: Any,
    ) -> PredictionResult:
        display_names = self._sample_display_names

        if not len(data):
            return {display_name: [] for display_name in display_names}

        prediction_mean_results, prediction_std_results = self._predict_with_std(data)

        prediction_results: PredictionResult = {}
        for display_name, means, stds in zip(
            display_names,
            prediction_mean_results.values(),
            prediction_std_results.values(),
        ):
            samples = np.random.normal(means, stds, (n_samples, len(means))).T
            prediction_results[display_name] = samples.tolist()

        return prediction_results

    def _calibrate_uncertainty(self, data: datasets.Dataset, **kwargs: Any) -> Any:
        pass
