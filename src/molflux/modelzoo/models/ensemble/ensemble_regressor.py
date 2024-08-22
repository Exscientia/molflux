import os
from copy import copy
from typing import Any

import numpy as np
from pydantic.v1 import dataclasses

from datasets import Dataset
from molflux.modelzoo import load_from_store, save_to_store
from molflux.modelzoo.errors import NotTrainedError
from molflux.modelzoo.info import ModelInfo
from molflux.modelzoo.load import load_from_dict
from molflux.modelzoo.model import (
    ModelBase,
    ModelConfig,
    PredictionIntervalMixin,
    SamplingMixin,
    StandardDeviationMixin,
)
from molflux.modelzoo.models.core.average_features_regressor import (
    AverageFeaturesRegressor,
)
from molflux.modelzoo.models.ensemble._combo.utils import (
    check_parameter,
    get_split_indices,
    list_diff,
)
from molflux.modelzoo.protocols import (
    supports_prediction_interval,
    supports_std,
)
from molflux.modelzoo.typing import PredictionResult
from molflux.modelzoo.utils import format_wrapped_model_tag, pick_features

try:
    from molflux.modelzoo.models.sklearn.linear_regressor import LinearRegressor
except ImportError as e:
    from molflux.modelzoo.errors import ExtrasDependencyImportError

    raise ExtrasDependencyImportError("sklearn", e) from None

_DESCRIPTION = """
Implementation of an ensemble regression model.

An ensemble of models is a collection of models trained separately on the same data that
when combined together may offer better performance than any of the individual models.
Ensemble models often perform well on competition leaderboards, even when ensembling
lots of models that individually may perform poorly.

In this implementation of meta-ensembling (also known as stacking), each of the
base regression models makes a prediction and a meta-regressor is trained
on the predictions of the individual models, optionally including the original
data as additional features.
"""

_CONFIG_DESCRIPTION = """
Parameters
----------
base_estimators : list
    A list of modelzoo config dicts to define the models used for training
    the base estimators. An error is raised if the base estimators are not
    provided.
meta_estimator : dict, modelzoo model config, optional (default=None)
    The modelzoo model config for a meta regressor to make the final prediction.
    In the default case, the meta_estimator used depends on ``keep_original``.
    Either a linear regressor is used when the original features are kept, or
    when ``keep_original=False`` then the predictions from the base estimators
    are averaged via the ``average_features_regressor``.
n_folds : int, optional (default=2)
    The number of splits of the training sample.
keep_original : bool, optional (default=False)
    If True, keep the original features for training and predicting.
shuffle_data : bool, optional (default=False)
    If True, shuffle the input data.
random_state : int, RandomState or None, optional (default=None)
    If int, random_state is the seed used by the random
    number generator; If RandomState instance, random_state is the random
    number generator; If None, the random number generator is the
    RandomState instance used by `np.random`.
"""


class Config:
    arbitrary_types_allowed = True


@dataclasses.dataclass(config=Config)
class EnsembleRegressorConfig(ModelConfig):
    base_estimators: list[dict[str, Any]] | None = None
    meta_estimator: dict[str, Any] | None = None  # config for modelzoo regression model
    n_folds: int = 2
    keep_original: bool = False
    shuffle_data: bool = False
    random_state: int | None = None


class EnsembleRegressor(
    PredictionIntervalMixin,
    StandardDeviationMixin,
    SamplingMixin,
    ModelBase[EnsembleRegressorConfig],
):
    """
    Meta ensembling, also known as stacking
    See https://datasciblog.github.io/2016/12/27/a-kagglers-guide-to-model-stacking-in-practice/ for more information.
    See also the `combo` library eg. https://github.com/yzhao062/combo/blob/master/combo/models/classifier_stacking.py
    """

    def __init__(self, tag: str | None = None, **config_kwargs: Any) -> None:
        super().__init__(tag=tag, **config_kwargs)

        config = self.config
        if len(config["base_estimators"]) < 2:
            raise ValueError("At least 2 estimators are required")
        self.base_estimators = [
            load_from_dict(cfg) for cfg in config["base_estimators"]
        ]
        self.n_base_estimators_ = len(self.base_estimators)
        if tag is None:
            self.info.tag = format_wrapped_model_tag(
                self.info.tag,
                [estimator.tag for estimator in self.base_estimators],
            )

        # validate input parameters
        if not isinstance(config["n_folds"], int):
            raise ValueError("n_folds must be an integer variable")
        check_parameter(
            config["n_folds"],
            low=2,
            include_left=True,
            param_name="n_folds",
        )
        self.n_folds = config["n_folds"]

        self.all_models_support_std = all(
            supports_std(model) for model in self.base_estimators
        )
        if config["meta_estimator"] is not None:
            self.meta_estimator = load_from_dict(config["meta_estimator"])
        elif config["keep_original"]:
            meta_estimator_colnames = [
                reg.tag for reg in self.base_estimators
            ] + self.x_features
            self.meta_estimator = LinearRegressor(
                x_features=meta_estimator_colnames,
                y_features=self.y_features,
            )
        elif not config["keep_original"]:
            meta_estimator_colnames = [reg.tag for reg in self.base_estimators]
            meta_estimator_std_colnames = [
                f"{reg.tag}::std" for reg in self.base_estimators
            ]
            # If all base estimators provide uncertainty, use this via a Gaussian mixture.
            # O/W fall back to using std of the estimates themselves
            self.meta_estimator = AverageFeaturesRegressor(
                x_features=meta_estimator_colnames,
                y_features=self.y_features,
                x_std_features=meta_estimator_std_colnames
                if self.all_models_support_std
                else None,
            )

        # set flags
        self.keep_original = config["keep_original"]
        self.shuffle_data = config["shuffle_data"]
        self.random_state = config["random_state"]

    def _config(self) -> EnsembleRegressorConfig:
        return EnsembleRegressorConfig()

    @property
    def _config_builder(self) -> type[EnsembleRegressorConfig]:
        return EnsembleRegressorConfig

    def _info(self) -> ModelInfo:
        return ModelInfo(
            model_description=_DESCRIPTION,
            config_description=_CONFIG_DESCRIPTION,
        )

    def _train(self, train_data: Dataset, **kwargs: Any) -> Any:
        """
        Fit ensemble regressor. Iterate over folds of cross validation set to fit individual ensemble models.
        Fit meta-estimator based on these. Then refit individual models on full data.
        """
        n_samples = train_data.shape[0]

        # initialize matrix for storing newly generated features
        new_features = np.zeros(
            [n_samples, (1 + self.all_models_support_std) * self.n_base_estimators_],
        )

        # build CV datasets
        index_lists = get_split_indices(
            dataset=train_data,
            n_folds=self.n_folds,
        )
        if self.shuffle_data:
            train_data = train_data.shuffle(seed=self.random_state)

        # iterate over all base estimators
        for i, reg in enumerate(self.base_estimators):
            # iterate over all folds
            for j in range(self.n_folds):
                # build train and test index
                full_idx = list(range(n_samples))
                test_idx = index_lists[j]
                train_idx = list_diff(full_idx, test_idx)
                train_fold = train_data.select(train_idx)
                test_fold = train_data.select(test_idx)

                # train the regressor
                reg.train(train_data=train_fold)

                # generate the new features on the pseudo test set
                reg_tag_name = f"{reg.tag}::{reg.y_features[0]}"
                if self.all_models_support_std:
                    assert supports_std(reg)
                    preds, stds = reg.predict_with_std(test_fold)
                    (
                        new_features[test_idx, i],
                        new_features[test_idx, i + self.n_base_estimators_],
                    ) = (preds[reg_tag_name], stds[f"{reg_tag_name}::std"])
                else:
                    new_features[test_idx, i] = reg.predict(test_fold)[reg_tag_name]

        # build the new dataset for training
        train_comb = copy(train_data)
        meta_estimator_colnames = [clf.tag for clf in self.base_estimators]
        for i_col, colname in enumerate(meta_estimator_colnames):
            train_comb = train_comb.add_column(
                colname,
                new_features[:, i_col],
            ).flatten_indices()
            if self.all_models_support_std:
                train_comb = train_comb.add_column(
                    f"{colname}::std",
                    new_features[:, self.n_base_estimators_ + i_col],
                ).flatten_indices()

        if self.keep_original:
            meta_estimator_colnames += self.x_features
        assert set(meta_estimator_colnames).issubset(
            set(self.meta_estimator.x_features),
        )
        self.meta_estimator.train(train_data=train_comb)

        # train all base classifiers on the full train dataset
        # iterate over all base estimators
        for reg in self.base_estimators:
            reg.train(train_data=train_data)

        return

    def _process_data(self, dataset: Dataset, with_std: bool = False) -> Any:
        """Internal class for `predict`
        Parameters
        ----------
        dataset: Dataset (n_samples, n_features)
            The input samples.
        with_std: bool
            Use predict_with_std or just predict.
        Returns
        -------
        data_new_comb : Dataset
            The processed dataset
        """
        n_samples = dataset.shape[0]

        # initialize matrix for storing newly generated features
        new_features = np.zeros([n_samples, (1 + with_std) * self.n_base_estimators_])

        # build the new features for unknown samples
        # iterate over all base classifiers
        for i, reg in enumerate(self.base_estimators):
            # generate the new features on the test set
            reg_tag_name = f"{reg.tag}::{reg.y_features[0]}"
            if with_std:
                assert supports_std(reg)
                preds, stds = reg.predict_with_std(dataset)
                new_features[:, i], new_features[:, i + self.n_base_estimators_] = (
                    preds[reg_tag_name],
                    stds[f"{reg_tag_name}::std"],
                )
            else:
                new_features[:, i] = reg.predict(dataset)[reg_tag_name]

        # build the new dataset for unknown samples
        new_colnames = [reg.tag for reg in self.base_estimators]
        data_new_comb = copy(dataset)
        for i_col, colname in enumerate(new_colnames):
            data_new_comb = data_new_comb.add_column(
                colname,
                new_features[:, i_col],
            ).flatten_indices()
            if with_std:
                data_new_comb = data_new_comb.add_column(
                    f"{colname}::std",
                    new_features[:, self.n_base_estimators_ + i_col],
                ).flatten_indices()

        return data_new_comb

    def _predict(self, data: Dataset, **kwargs: Any) -> PredictionResult:
        """Make regression predictions for the provided data.
        Parameters
        ----------
        data: Dataset
        Returns
        -------
        labels : numpy array of shape (n_samples,)
            Predictions for each data sample.
        """

        display_names = self._predict_display_names

        if not len(data):
            return {display_name: [] for display_name in display_names}

        x_data = pick_features(data, self.x_features)
        X_new_comb = self._process_data(x_data)
        original_prediction_results = self.meta_estimator.predict(X_new_comb)
        return dict(
            zip(
                display_names,
                original_prediction_results.values(),
                strict=False,
            ),
        )

    def _predict_with_std(
        self,
        data: Dataset,
        **kwargs: Any,
    ) -> tuple[PredictionResult, PredictionResult]:
        """
        Make regression and std predictions for the provided data.
        """
        assert supports_std(
            self.meta_estimator,
        ), "meta_estimator should support uncertainty via prediction with standard deviation"

        (
            prediction_display_names,
            prediction_std_display_names,
        ) = self._predict_with_std_display_names

        if not len(data):
            return {display_name: [] for display_name in prediction_display_names}, {
                display_name: [] for display_name in prediction_std_display_names
            }

        x_data = pick_features(data, self.x_features)
        X_new_comb = self._process_data(x_data, with_std=self.all_models_support_std)
        (
            original_prediction_results,
            original_prediction_stds,
        ) = self.meta_estimator.predict_with_std(X_new_comb)

        return dict(
            zip(
                prediction_display_names,
                original_prediction_results.values(),
                strict=False,
            ),
        ), dict(
            zip(
                prediction_std_display_names,
                original_prediction_stds.values(),
                strict=False,
            ),
        )

    def _predict_with_prediction_interval(
        self,
        data: Dataset,
        confidence: float,
        **kwargs: Any,
    ) -> tuple[PredictionResult, PredictionResult]:
        assert supports_prediction_interval(
            self.meta_estimator,
        ), "meta_estimator should support uncertainty via prediction with prediction interval"

        (
            prediction_display_names,
            prediction_interval_display_names,
        ) = self._predict_with_prediction_interval_display_names

        if not len(data):
            return {display_name: [] for display_name in prediction_display_names}, {
                display_name: [] for display_name in prediction_interval_display_names
            }

        x_data = pick_features(data, self.x_features)
        X_new_comb = self._process_data(x_data, with_std=self.all_models_support_std)
        (
            original_prediction_results,
            original_prediction_intervals,
        ) = self.meta_estimator.predict_with_prediction_interval(
            X_new_comb,
            confidence=confidence,
        )
        return dict(
            zip(
                prediction_display_names,
                original_prediction_results.values(),
                strict=False,
            ),
        ), dict(
            zip(
                prediction_interval_display_names,
                original_prediction_intervals.values(),
                strict=False,
            ),
        )

    def _sample(
        self,
        data: Dataset,
        n_samples: int,
        **kwargs: Any,
    ) -> PredictionResult:
        assert supports_std(
            self.meta_estimator,
        ), "meta_estimator should support uncertainty via predict with std"

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

    def as_dir(self, directory: str) -> None:
        """
        Serialises a pre-trained ensemble model in a directory.

        As ensembles have a meta estimator and a list of base estimators, each one is
        saved to a subdirectory, in the format:
        `directory/
            meta_estimator/
                ...
            base_estimators/
                0/
                    ...
                1/
                    ...
                ...
        """

        if self.meta_estimator is None:
            raise NotTrainedError

        if self.base_estimators is None:
            raise NotTrainedError

        meta_estimator_path = os.path.join(directory, "meta_estimator")
        base_estimators_base_path = os.path.join(directory, "base_estimators")

        save_to_store(meta_estimator_path, self.meta_estimator)

        for model_index in range(len(self.base_estimators)):
            base_estimator_path = os.path.join(
                base_estimators_base_path,
                str(model_index),
            )
            save_to_store(base_estimator_path, self.base_estimators[model_index])

    def from_dir(self, directory: str) -> None:
        """Deserialises the backend model object stored in a given directory.

        As ensembles have a meta estimator and a list of base estimators, each one is
        saved to a subdirectory, in the format:
        `directory/
            meta_estimator/
                ...
            base_estimators/
                0/
                    ...
                1/
                    ...
                ...
        """

        # The expected model binary
        meta_estimator_path = os.path.join(directory, "meta_estimator")
        base_estimators_base_path = os.path.join(directory, "base_estimators")

        self.meta_estimator = load_from_store(meta_estimator_path)

        base_estimators = []
        for model_index in range(len(self.base_estimators)):
            base_estimator_path = os.path.join(
                base_estimators_base_path,
                str(model_index),
            )
            base_estimators.append(load_from_store(base_estimator_path))

        self.base_estimators = base_estimators
