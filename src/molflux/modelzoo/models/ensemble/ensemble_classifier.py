import os
from copy import copy
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Type

import numpy as np
from pydantic.dataclasses import dataclass

from datasets import Dataset
from molflux.modelzoo import load_from_store, save_to_store
from molflux.modelzoo.errors import NotTrainedError
from molflux.modelzoo.info import ModelInfo
from molflux.modelzoo.load import load_from_dict
from molflux.modelzoo.model import ClassificationMixin, ModelBase, ModelConfig
from molflux.modelzoo.models.ensemble._combo.utils import (
    check_parameter,
    get_split_indices,
    list_diff,
)
from molflux.modelzoo.protocols import SupportsClassification
from molflux.modelzoo.typing import Classes, PredictionResult
from molflux.modelzoo.utils import pick_features

try:
    from molflux.modelzoo.models.sklearn.logistic_regressor import LogisticRegressor
except ImportError as e:
    from molflux.modelzoo.errors import ExtrasDependencyImportError

    raise ExtrasDependencyImportError("sklearn", e) from None


_DESCRIPTION = """
Implementation of an ensemble classification model.

An ensemble of models is a collection of models trained separately on the same data that
when combined together may offer better performance than any of the individual models.
Ensemble models often perform well on competition leaderboards, even when ensembling
lots of models that individually may perform poorly.

In this implementation of meta-ensembling (also known as stacking), each of the
base classification models makes a prediction and a meta-classifier is trained
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
meta_estimator : dict, modelzoo model config, optional (default=logistic regression)
    The config for a modelzoo meta classifier to make the final prediction.
n_folds : int, optional (default=2)
    The number of splits of the training sample.
keep_original : bool, optional (default=False)
    If True, keep the original features for training and predicting.
use_proba: bool, optional (default=False)
    If True, use the probability prediction as the new features.
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
    extra = "forbid"


@dataclass(config=Config)
class EnsembleClassifierConfig(ModelConfig):
    base_estimators: Optional[List[Dict[str, Any]]] = None
    meta_estimator: Optional[
        Dict[str, Any]
    ] = None  # config for a a modelzoo classification model
    n_folds: int = 2
    keep_original: bool = False
    use_proba: bool = False
    shuffle_data: bool = False
    random_state: Optional[int] = None


class EnsembleClassifier(ClassificationMixin, ModelBase[EnsembleClassifierConfig]):
    """
    Meta ensembling, also known as stacking
    See https://datasciblog.github.io/2016/12/27/a-kagglers-guide-to-model-stacking-in-practice/ for more information.
    See also the `combo` library eg. https://github.com/yzhao062/combo/blob/master/combo/models/classifier_stacking.py
    """

    def __init__(self, tag: Optional[str] = None, **config_kwargs: Any) -> None:
        super().__init__(tag=tag, **config_kwargs)

        config = self.config
        if len(config["base_estimators"]) < 2:
            raise ValueError("At least 2 estimators are required")
        self.base_estimators = [
            load_from_dict(cfg) for cfg in config["base_estimators"]
        ]
        self.n_base_estimators_ = len(self.base_estimators)

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

        if config["meta_estimator"] is not None:
            self.meta_estimator = load_from_dict(config["meta_estimator"])
        elif config["keep_original"]:
            meta_estimator_colnames = [
                clf.tag for clf in self.base_estimators
            ] + self.x_features
            self.meta_estimator = LogisticRegressor(
                x_features=meta_estimator_colnames,
                y_features=self.y_features,
            )
        elif not config["keep_original"]:
            meta_estimator_colnames = [clf.tag for clf in self.base_estimators]
            self.meta_estimator = LogisticRegressor(
                x_features=meta_estimator_colnames,
                y_features=self.y_features,
            )
        if not isinstance(self.meta_estimator, SupportsClassification):
            raise ValueError("Meta estimator should support classification")

        # set flags
        self.keep_original = config["keep_original"]
        self.shuffle_data = config["shuffle_data"]
        self.random_state = config["random_state"]
        self.use_proba = config["use_proba"]

    def _config(self) -> EnsembleClassifierConfig:
        return EnsembleClassifierConfig()

    @property
    def config(self) -> Dict[str, Any]:
        return asdict(self.model_config)

    @property
    def _config_builder(self) -> Type[EnsembleClassifierConfig]:
        return EnsembleClassifierConfig

    def _info(self) -> ModelInfo:
        return ModelInfo(
            model_description=_DESCRIPTION,
            config_description=_CONFIG_DESCRIPTION,
        )

    def _train(self, train_data: Dataset, **kwargs: Any) -> Any:
        """
        Fit ensemble classifier. Iterate over folds of cross validation set to fit individual ensemble models.
        Fit meta-estimator based on these. Then refit individual models on full data.
        """
        n_samples = train_data.shape[0]

        # initialize matrix for storing newly generated features
        new_features = np.zeros([n_samples, self.n_base_estimators_])

        # build CV datasets
        index_lists = get_split_indices(
            dataset=train_data,
            n_folds=self.n_folds,
        )
        if self.shuffle_data:
            train_data = train_data.shuffle(seed=self.random_state)

        # iterate over all base estimators
        for i, clf in enumerate(self.base_estimators):
            if not isinstance(clf, SupportsClassification):
                raise ValueError("Base estimators should support classification")
            # iterate over all folds
            for j in range(self.n_folds):
                # build train and test index
                full_idx = list(range(n_samples))
                test_idx = index_lists[j]
                train_idx = list_diff(full_idx, test_idx)
                train_fold = train_data.select(train_idx)
                test_fold = train_data.select(test_idx)

                # train the classifier
                clf.train(train_data=train_fold)

                # generate the new features on the pseudo test set
                clf_tag_name = f"{clf.tag}::{clf.y_features[0]}"
                if self.use_proba:
                    new_features[test_idx, i] = clf.predict_proba(test_fold)[
                        clf_tag_name + "::probabilities"
                    ]
                else:
                    new_features[test_idx, i] = clf.predict(test_fold)[clf_tag_name]

        # build the new dataset for training
        train_comb = copy(train_data)
        meta_estimator_colnames = [clf.tag for clf in self.base_estimators]
        for i_col, colname in enumerate(meta_estimator_colnames):
            train_comb = train_comb.add_column(
                colname,
                new_features[:, i_col],
            ).flatten_indices()
        if self.keep_original:
            meta_estimator_colnames += self.x_features
        assert self.meta_estimator.x_features == meta_estimator_colnames
        self.meta_estimator.train(train_data=train_comb)

        # train all base classifiers on the full train dataset
        # iterate over all base estimators
        for clf in self.base_estimators:
            clf.train(train_data=train_data)

    def _process_data(self, dataset: Dataset) -> Any:
        """Internal class for `predict`
        Parameters
        ----------
        dataset: Dataset (n_samples, n_features)
            The input samples.
        Returns
        -------
        data_new_comb : Dataset
            The processed dataset
        """
        n_samples = dataset.shape[0]

        # initialize matrix for storing newly generated features
        new_features = np.zeros([n_samples, self.n_base_estimators_])

        # build the new features for unknown samples
        # iterate over all base classifiers
        for i, clf in enumerate(self.base_estimators):
            clf_tag_name = f"{clf.tag}::{clf.y_features[0]}"
            if self.use_proba:
                new_features[:, i] = clf.predict_proba(dataset)[  # type: ignore[attr-defined]
                    clf_tag_name + "::probabilities"
                ]
            else:
                new_features[:, i] = clf.predict(dataset)[clf_tag_name]

        # build the new dataset for unknown samples
        new_colnames = [clf.tag for clf in self.base_estimators]
        data_new_comb = copy(dataset)
        for i_col, colname in enumerate(new_colnames):
            data_new_comb = data_new_comb.add_column(
                colname,
                new_features[:, i_col],
            ).flatten_indices()

        return data_new_comb

    def _predict(self, data: Dataset, **kwargs: Any) -> PredictionResult:
        """Predict class labels for the provided data.
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
            ),
        )

    @property
    def classes(self) -> Classes:
        if not hasattr(self.model, "classes_"):
            return {}

        sklearn_classes = self.model.classes_
        # this could be of several shapes / types
        # singletask: (n_classes,)
        # multitask: (n_tasks, n_classes)

        is_multitask = np.ndim(sklearn_classes) == 2
        classes = sklearn_classes if is_multitask else [sklearn_classes]

        return {
            task: classes.tolist() for task, classes in zip(self.y_features, classes)
        }

    def _predict_proba(self, data: Dataset, **kwargs: Any) -> PredictionResult:
        """Return probability estimates for the test data X.
        Parameters
        ----------
        data: Dataset
        Returns
        -------
        p : numpy array of shape (n_samples,)
            The class probabilities of the input samples.
            Classes are ordered by lexicographic order.
        """
        display_names = self._predict_proba_display_names

        if not len(data):
            return {display_name: [] for display_name in display_names}

        x_data = pick_features(data, self.x_features)
        X_new_comb = self._process_data(x_data)
        original_prediction_results = self.meta_estimator.predict_proba(X_new_comb)  # type: ignore[attr-defined]
        return dict(
            zip(
                display_names,
                original_prediction_results.values(),
            ),
        )

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
