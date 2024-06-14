from typing import Any, Optional, Type

from pydantic.v1 import dataclasses
from scipy import stats

import datasets
from molflux.modelzoo.info import ModelInfo
from molflux.modelzoo.model import (
    ClassificationMixin,
    ModelBase,
    ModelConfig,
)
from molflux.modelzoo.models.pystan.utils import StanWrapper
from molflux.modelzoo.typing import Classes, PredictionResult
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
Implementation of bayesian ordinal classifier. Similar to a logistic regression model
See Williams et al 2019 Predicting Drug-Induced Liver Injury with Bayesian Machine Learning
"""

_CONFIG_DESCRIPTION = """
Parameters
----------
num_classes: int, default=3 Number of discrete ordinal classes (eg. equivalent to low, medium high)
num_chains: int, default=4 Number of MCMC chains to run. These can be run in parallel allowing faster sampling,
    and having multiple chains is also useful in diagnosing convergence.
num_warmup: int, default=1000 Number of warmup/burnin samples to run to help ensure sampling
    from stationary distribution.
num_samples: int, default=1000 Number of MCMC samples produced per chain. More samples allows for more accurate
    estimation of expectations, and will help ensure convergence, but will take longer to compute.
random_seed: Optional[int], default=None Random seed to be used during sampling for reproducibility.
sigma_prior: float, default=1.0 Scale parameter for prior on sigma.
mu_prior: float, default=2.0 Scale parameter for prior on mu.
"""


class Config:
    arbitrary_types_allowed = True


@dataclasses.dataclass(config=Config)
class OrdinalClassifierConfig(ModelConfig):
    num_classes: int = 3
    num_chains: int = 4
    num_warmup: int = 1000
    num_samples: int = 1000
    random_seed: Optional[int] = None
    sigma_prior: float = 1.0
    mu_prior: float = 2.0

    def __post_init_post_parse__(self) -> None:
        if self.y_features and len(self.y_features) != 1:
            raise NotImplementedError(
                f"This model architecture only supports single task regression for now: got {self.y_features}",
            )


class OrdinalClassifier(
    ClassificationMixin,
    ModelBase[OrdinalClassifierConfig],
):
    @property
    def _config_builder(self) -> Type[OrdinalClassifierConfig]:
        return OrdinalClassifierConfig

    def _info(self) -> ModelInfo:
        return ModelInfo(
            model_description=_DESCRIPTION,
            config_description=_CONFIG_DESCRIPTION,
        )

    def _instantiate_model(self) -> StanWrapper:
        return StanWrapper("OrdinalClassifier.stan")

    @property
    def classes(self) -> Classes:
        classes = [range(self.model_config.num_classes)]
        return {
            task: list(class_int) for task, class_int in zip(self.y_features, classes)
        }

    def _train(self, train_data: datasets.Dataset, **kwargs: Any) -> Any:
        validate_features(train_data, self.y_features)

        x_data = pick_features(train_data, self.x_features)
        X = get_concatenated_array(x_data, self.x_features)

        y_data = pick_features(train_data, self.y_features)
        y = get_concatenated_array(y_data, self.y_features)
        y = y.ravel()

        N, p = X.shape
        stan_data = {
            "n_classes": self.model_config.num_classes,
            "N": N,
            "P": p,
            "X": X,
            "sigma_prior": self.model_config.sigma_prior,
            "mu_prior": self.model_config.mu_prior,
            "y": y,
        }

        self.model = self._instantiate_model()
        posterior = stan.build(
            program_code=self.model.model_code,
            data=stan_data,
            random_seed=self.model_config.random_seed,
        )
        fit = posterior.sample(
            num_chains=self.model_config.num_chains,
            num_warmup=self.model_config.num_warmup,
            num_samples=self.model_config.num_samples,
        )
        self.model.fit = fit

    def _predict(self, data: datasets.Dataset, **kwargs: Any) -> PredictionResult:
        validate_features(data, self.x_features)
        display_names = self._predict_display_names

        if not len(data):
            return {display_name: [] for display_name in display_names}

        x_data = pick_features(data, self.x_features)
        X_test = get_concatenated_array(x_data, self.x_features)
        N_pred, p = X_test.shape

        fit_df = self.model.fit.to_frame()
        cutpoint_cols = [col for col in fit_df.columns if col.startswith("cutpoints")]
        cutpoints = fit_df[cutpoint_cols].to_numpy()
        beta_cols = [col for col in fit_df.columns if col.startswith("beta")]
        beta = fit_df[beta_cols].to_numpy()
        N_samples = self.model_config.num_samples * self.model_config.num_chains
        stan_data = {
            "n_classes": self.model_config.num_classes,
            "N_samples": N_samples,
            "N_pred": N_pred,
            "X_pred": X_test,
            "P": p,
            "cutpoints": cutpoints,
            "beta": beta,
        }

        prediction_model = StanWrapper("OrdinalClassifierPredictions.stan")
        posterior = stan.build(
            program_code=prediction_model.model_code,
            data=stan_data,
            random_seed=self.model_config.random_seed,
        )
        fit = posterior.fixed_param(
            num_chains=1,
            num_samples=1,
        )
        post_pred_df = fit.to_frame()
        predictions = [
            int(
                stats.mode(
                    [
                        post_pred_df[f"ypred.{i}.{j}"][0]
                        for i in range(1, N_samples + 1)
                    ],
                    keepdims=False,
                ).mode,
            )
            for j in range(1, N_pred + 1)
        ]
        return {display_name: predictions for display_name in display_names}

    def _predict_proba(self, data: datasets.Dataset, **kwargs: Any) -> PredictionResult:
        validate_features(data, self.x_features)
        display_names = self._predict_proba_display_names

        if not len(data):
            return {display_name: [] for display_name in display_names}

        x_data = pick_features(data, self.x_features)
        X_test = get_concatenated_array(x_data, self.x_features)
        N_pred, p = X_test.shape

        fit_df = self.model.fit.to_frame()
        cutpoint_cols = [col for col in fit_df.columns if col.startswith("cutpoints")]
        cutpoints = fit_df[cutpoint_cols].to_numpy()
        beta_cols = [col for col in fit_df.columns if col.startswith("beta")]
        beta = fit_df[beta_cols].to_numpy()
        N_samples = self.model_config.num_samples * self.model_config.num_chains
        stan_data = {
            "n_classes": self.model_config.num_classes,
            "N_samples": N_samples,
            "N_pred": N_pred,
            "X_pred": X_test,
            "P": p,
            "cutpoints": cutpoints,
            "beta": beta,
        }

        prediction_model = StanWrapper("OrdinalClassifierPredictions.stan")
        posterior = stan.build(
            program_code=prediction_model.model_code,
            data=stan_data,
            random_seed=self.model_config.random_seed,
        )
        fit = posterior.fixed_param(
            num_chains=1,
            num_samples=1,
        )
        post_pred_df = fit.to_frame()

        all_predictions = [
            [post_pred_df[f"ypred.{i}.{j}"][0] for i in range(1, N_samples + 1)]
            for j in range(1, N_pred + 1)
        ]

        results = {}
        for task, display_name in zip(
            self.y_features,
            display_names,
        ):
            prob_preds = [
                sum(pred == ii for pred in preds) / len(preds)
                for preds in all_predictions
                for ii in self.classes[task]
            ]
            results[display_name] = prob_preds

        # (n_tasks, n_samples, n_classes)
        return results
