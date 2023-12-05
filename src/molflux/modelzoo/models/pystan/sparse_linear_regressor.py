from pathlib import Path
from typing import Any, Optional, Tuple, Type

import numpy as np
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
from molflux.modelzoo.utils import (
    get_concatenated_array,
    pick_features,
    validate_features,
)

try:
    import stan
except ImportError as e:
    from molflux.modelzoo.errors import ExtrasDependencyImportError

    raise ExtrasDependencyImportError("pystan", e) from None


_DESCRIPTION = """
Implementation of bayesian linear regression with a sparsity inducing prior
See this paper: https://projecteuclid.org/journals/electronic-journal-of-statistics/volume-11/issue-2/Sparsity-information-and-regularization-in-the-horseshoe-and-other-shrinkage/10.1214/17-EJS1337SI.full
"""

_CONFIG_DESCRIPTION = """
Parameters
----------
num_chains : int, default=4  Number of MCMC chains to run. These can be run in parallel allowing faster sampling,
    and having multiple chains is also useful in diagnosing convergence.
num_samples: int, default=1000 Number of MCMC samples produced per chain. More samples allows for more accurate
    estimation of expectations, and will help ensure convergence, but will take longer to compute.
random_seed: Optional[int], default=None Random seed to be used during sampling for reproducibility
"""


class BayesLinearRegressorHorseshoePrior:
    def __init__(self) -> None:
        path_to_stan_code = (
            Path(__file__).parent / "BayesLinearRegressorHorseshoePrior.stan"
        )
        self.model_code = path_to_stan_code.read_text()
        self.fit = None  # to store posterior samples
        self.beta = None  # to store average coefficients, beta


class Config:
    arbitrary_types_allowed = True
    extra = "forbid"


@dataclass(config=Config)
class SparseLinearRegressorConfig(ModelConfig):
    num_chains: int = 4
    num_samples: int = 1000
    random_seed: Optional[int] = None

    def __post_init_post_parse__(self) -> None:
        if self.y_features and len(self.y_features) != 1:
            raise NotImplementedError(
                f"This model architecture only supports single task regression for now: got {self.y_features}",
            )


class SparseLinearRegressor(
    UncertaintyCalibrationMixin,
    PredictionIntervalMixin,
    StandardDeviationMixin,
    SamplingMixin,
    ModelBase[SparseLinearRegressorConfig],
):
    @property
    def _config_builder(self) -> Type[SparseLinearRegressorConfig]:
        return SparseLinearRegressorConfig

    def _info(self) -> ModelInfo:
        return ModelInfo(
            model_description=_DESCRIPTION,
            config_description=_CONFIG_DESCRIPTION,
        )

    def _instantiate_model(self) -> BayesLinearRegressorHorseshoePrior:
        return BayesLinearRegressorHorseshoePrior()

    def _train(self, train_data: datasets.Dataset, **kwargs: Any) -> Any:
        validate_features(train_data, self.y_features)

        x_data = pick_features(train_data, self.x_features)
        X = get_concatenated_array(x_data, self.x_features)

        y_data = pick_features(train_data, self.y_features)
        y = get_concatenated_array(y_data, self.y_features)
        y = y.ravel()

        n, p = X.shape
        data = {"n": n, "p": p, "X": X, "y": y}
        self.model = self._instantiate_model()
        posterior = stan.build(
            program_code=self.model.model_code,
            data=data,
            random_seed=self.model_config.random_seed,
        )
        fit = posterior.sample(
            num_chains=self.model_config.num_chains,
            num_samples=self.model_config.num_samples,
        )
        self.model.fit = fit
        self.model.beta = np.mean(fit["beta"], axis=1)

    def _predict(self, data: datasets.Dataset, **kwargs: Any) -> PredictionResult:
        display_names = self._predict_display_names

        if not len(data):
            return {display_name: [] for display_name in display_names}

        X = get_concatenated_array(data, self.x_features)
        ypred = np.dot(X, self.model.beta)

        return {display_name: ypred.tolist() for display_name in display_names}

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
        X = get_concatenated_array(data, self.x_features)
        ypred = np.dot(X, self.model.beta)
        beta_samples = self.model.fit["beta"]
        sigma_samples = self.model.fit["sigma"]
        mu = np.matmul(X, beta_samples)
        pred = mu + np.random.randn(*sigma_samples.shape) * sigma_samples
        pred_std = np.std(pred, axis=1)

        return {
            display_name: ypred.tolist() for display_name in prediction_display_names
        }, {
            display_name: pred_std.tolist()
            for display_name in prediction_std_display_names
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

        X = get_concatenated_array(data, self.x_features)
        ypred = np.dot(X, self.model.beta)
        beta_samples = self.model.fit["beta"]
        sigma_samples = self.model.fit["sigma"]
        mu = np.matmul(X, beta_samples)
        pred = mu + np.random.randn(*sigma_samples.shape) * sigma_samples
        lower_bound = np.quantile(pred, (1 - confidence) / 2.0, axis=1)
        upper_bound = np.quantile(pred, 1 - (1 - confidence) / 2.0, axis=1)
        # where nans occur (ex. from a 0 standard deviation), use the mean instead
        lower_bound = np.where(~np.isnan(lower_bound), lower_bound, ypred)
        upper_bound = np.where(~np.isnan(upper_bound), upper_bound, ypred)

        return {
            display_name: ypred.tolist() for display_name in prediction_display_names
        }, {
            display_name: list(zip(lower_bound.tolist(), upper_bound.tolist()))
            for display_name in prediction_interval_display_names
        }

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
