from collections.abc import Callable
from typing import Any, Literal

import numpy as np
import scipy.stats as st
from pydantic.v1 import dataclasses

import datasets
from molflux.modelzoo.info import ModelInfo
from molflux.modelzoo.model import (
    ModelBase,
    ModelConfig,
    PredictionIntervalMixin,
    SamplingMixin,
    StandardDeviationMixin,
)
from molflux.modelzoo.typing import Features, PredictionResult

_DESCRIPTION = """
A core model that predicts the average (median/mean) of the features.
A Gaussian mixture model approximation can also be used by providing std associated with each feature.
"""

_CONFIG_DESCRIPTION = """
Parameters
----------
average_method: Literal, default = "median". Method by which to summarise the average of the dataset.
    Available options are median or mean. median is preferred since it is robust to outliers.
deviation_method: Literal, default = "mad". Method by which to summarise the deviation of the dataset.
    Available options are std or mad. Note that mad is the median absolute deviation which is a version
    of standard deviation robust to outlier values.
x_std_features: Features, default = None. List of strings providing a subset of the features that
    are standard deviations associated with other features for use in a Gaussian mixture model.
"""

# mapping from method strings to fns
_AVERAGE_METHOD_DICT: dict[str, Callable] = {
    "median": np.nanmedian,
    "mean": np.nanmean,
}
_DEVIATION_METHOD_DICT: dict[str, Callable] = {
    "std": np.nanstd,
    "mad": st.median_abs_deviation,
}


class Config:
    arbitrary_types_allowed = True


@dataclasses.dataclass
class AverageModel:
    average_fn: Callable
    deviation_fn: Callable


@dataclasses.dataclass(config=Config)
class AverageFeaturesRegressorConfig(ModelConfig):
    average_method: Literal["median", "mean"] = "median"
    deviation_method: Literal["std", "mad"] = "mad"
    x_std_features: Features | None = None


class AverageFeaturesRegressor(
    PredictionIntervalMixin,
    StandardDeviationMixin,
    SamplingMixin,
    ModelBase[AverageFeaturesRegressorConfig],
):
    @property
    def _config_builder(self) -> type[AverageFeaturesRegressorConfig]:
        return AverageFeaturesRegressorConfig

    def _info(self) -> ModelInfo:
        """Initialises the ModelInfo object."""
        return ModelInfo(
            model_description=_DESCRIPTION,
            config_description=_CONFIG_DESCRIPTION,
        )

    def _train(self, train_data: datasets.Dataset, **kwargs: Any) -> Any:
        if self.model_config.x_std_features is not None:
            assert (
                len(self.x_features)
                == 2
                * len(
                    self.model_config.x_std_features,
                )
            ), "When providing stds (``x_std_features``), these should match to the means with both of these provided in ``x_features``"
        self.model = AverageModel(
            average_fn=_AVERAGE_METHOD_DICT[self.model_config.average_method],
            deviation_fn=_DEVIATION_METHOD_DICT[self.model_config.deviation_method],
        )

    def _predict(self, data: datasets.Dataset, **kwargs: Any) -> PredictionResult:
        display_names = self._predict_display_names
        if not len(data):
            return {display_name: [] for display_name in display_names}

        data = data.with_format("np")  # easier for processing of missing None values
        assert isinstance(
            self.model,
            AverageModel,
        ), "Ensure ``average_features_regressor`` model has been trained"
        x_features = (
            list(set(self.x_features) - set(self.model_config.x_std_features))
            if self.model_config.x_std_features is not None
            else self.x_features
        )
        params_dict = {
            "mu": [
                self.model.average_fn(list(row.values()))
                for row in data.select_columns(x_features)
            ],
        }

        return {display_name: params_dict["mu"] for display_name in display_names}

    def _predict_with_std(
        self,
        data: datasets.Dataset,
        **kwargs: Any,
    ) -> tuple[PredictionResult, PredictionResult]:
        (
            prediction_display_names,
            prediction_std_display_names,
        ) = self._predict_with_std_display_names

        if not len(data):
            return {display_name: [] for display_name in prediction_display_names}, {
                display_name: [] for display_name in prediction_std_display_names
            }

        data = data.with_format("np")  # easier for processing of missing None values
        assert isinstance(
            self.model,
            AverageModel,
        ), "Ensure ``average_features_regressor`` model has been trained"

        x_features = (
            list(set(self.x_features) - set(self.model_config.x_std_features))
            if self.model_config.x_std_features is not None
            else self.x_features
        )
        filtered_data_x = data.select_columns(x_features)
        params_dict = {
            "mu": [
                self.model.average_fn(list(row.values())) for row in filtered_data_x
            ],
        }
        if self.model_config.x_std_features is not None:
            # Use a Gaussian to approximate a mixture of Gaussians from each of the individual models
            filtered_data_x_std = data.select_columns(self.model_config.x_std_features)
            sigma_sq = []
            for row_x, row_x_std, mu_star in zip(
                filtered_data_x,
                filtered_data_x_std,
                params_dict["mu"],
                strict=False,
            ):
                sigma_sq.append(
                    self.model.average_fn(
                        [
                            x**2 + y**2
                            for x, y in zip(
                                row_x.values(),
                                row_x_std.values(),
                                strict=False,
                            )
                        ],
                    )
                    - mu_star**2,
                )
            params_dict["sigma"] = [np.sqrt(s_sq) for s_sq in sigma_sq]
        else:
            params_dict["sigma"] = [
                self.model.deviation_fn(list(row.values())) for row in filtered_data_x
            ]

        return {
            display_name: params_dict["mu"] for display_name in prediction_display_names
        }, {
            display_name: params_dict["sigma"]
            for display_name in prediction_std_display_names
        }

    def _predict_with_prediction_interval(
        self,
        data: datasets.Dataset,
        confidence: float,
        **kwargs: Any,
    ) -> tuple[PredictionResult, PredictionResult]:
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
            strict=False,
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
            prediction_prediction_interval_results[prediction_interval_display_name] = (
                list(zip(lower_bound, upper_bound, strict=False))
            )

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
            strict=False,
        ):
            samples = np.random.normal(means, stds, (n_samples, len(means))).T
            prediction_results[display_name] = samples.tolist()

        return prediction_results
