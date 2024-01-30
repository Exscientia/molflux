import warnings
from dataclasses import asdict
from typing import Any, Dict, Final, Literal, Optional, Tuple, Type, Union

import numpy as np
from pydantic.dataclasses import dataclass

import datasets
from molflux.modelzoo.info import ModelInfo
from molflux.modelzoo.load import load_from_dict
from molflux.modelzoo.model import (
    ModelBase,
    ModelConfig,
    PredictionIntervalMixin,
    UncertaintyCalibrationMixin,
)
from molflux.modelzoo.models.sklearn import SKLearnModelBase
from molflux.modelzoo.protocols import Estimator
from molflux.modelzoo.typing import PredictionResult
from molflux.modelzoo.utils import (
    get_concatenated_array,
    pick_features,
    validate_features,
)

try:
    from mapie.conformity_scores import AbsoluteConformityScore
    from mapie.regression import MapieRegressor as MapieMapieRegressor
    from sklearn.base import RegressorMixin
    from sklearn.model_selection import BaseCrossValidator

except ImportError as e:
    from molflux.modelzoo.errors import ExtrasDependencyImportError

    raise ExtrasDependencyImportError("mapie", e) from None


_DESCRIPTION = """
Prediction interval with out-of-fold conformity scores.

This class implements the jackknife+ strategy and its variations
for estimating prediction intervals on single-output data. The
idea is to evaluate out-of-fold conformity scores (signed residuals,
absolute residuals, residuals normalized by the predicted mean...)
on hold-out validation sets and to deduce valid prediction intervals
with strong theoretical guarantees.
"""

_CONFIG_DESCRIPTION = """
Parameters
----------
estimator : Union[Estimator,RegressorMixin,None,Dict]
    Any modelzoo estimator or sklearn regressor backed by a scikit-learn API
    (i.e. with fit and predict methods), by default ``None``.
    If ``None``, estimator defaults to a linear regressor instance.
    If a dictionary, it is assumed to me a modelzoo model config for an scikit-learn backed regressor.

method: str, optional
    Method to choose for prediction interval estimates.
    Choose among:

    - "naive", based on training set conformity scores,
    - "base", based on validation sets conformity scores,
    - "plus", based on validation conformity scores and
      testing predictions,
    - "minmax", based on validation conformity scores and
      testing predictions (min/max among cross-validation clones).

    By default "plus".

cv: Optional[Union[int, str, BaseCrossValidator]]
    The cross-validation strategy for computing conformity scores.
    It directly drives the distinction between jackknife and cv variants.
    Choose among:

    - ``None``, to use the default 5-fold cross-validation
    - integer, to specify the number of folds.
      If equal to -1, equivalent to
      ``sklearn.model_selection.LeaveOneOut()``.
    - CV splitter: any ``sklearn.model_selection.BaseCrossValidator``
      Main variants are:
      - ``sklearn.model_selection.LeaveOneOut`` (jackknife),
      - ``sklearn.model_selection.KFold`` (cross-validation),
      - ``subsample.Subsample`` object (bootstrap).
    - ``"prefit"``, assumes that ``estimator`` has been fitted already,
      and the ``method`` parameter is ignored.
      All data provided in the ``fit`` method is then used
      for computing conformity scores only.
      At prediction time, quantiles of these conformity scores are used
      to provide a prediction interval with fixed width.
      The user has to take care manually that data for model fitting and
      conformity scores estimate are disjoint.

    By default ``None``.

n_jobs: Optional[int]
    Number of jobs for parallel processing using joblib
    via the "locky" backend.
    If ``-1`` all CPUs are used.
    If ``1`` is given, no parallel computing code is used at all,
    which is useful for debugging.
    For n_jobs below ``-1``, ``(n_cpus + 1 - n_jobs)`` are used.
    None is a marker for `unset` that will be interpreted as ``n_jobs=1``
    (sequential execution).

    By default ``None``.

agg_function : str
    Determines how to aggregate predictions from perturbed models, both at
    training and prediction time.

    If ``None``, it is ignored except if cv class is ``Subsample``,
    in which case an error is raised.
    If "mean" or "median", returns the mean or median of the predictions
    computed from the out-of-folds models.
    Note: if you plan to set the ``ensemble`` argument to ``True`` in the
    ``predict`` method, you have to specify an aggregation function.
    Otherwise an error would be raised.

    The Jackknife+ interval can be interpreted as an interval around the
    median prediction, and is guaranteed to lie inside the interval,
    unlike the single estimator predictions.

    When the cross-validation strategy is Subsample (i.e. for the
    Jackknife+-after-Bootstrap method), this function is also used to
    aggregate the training set in-sample predictions.

    If cv is ``"prefit"``, ``agg_function`` is ignored.

    By default "mean".

verbose : int, optional
    The verbosity level, used with joblib for multiprocessing.
    The frequency of the messages increases with the verbosity level.
    If it more than ``10``, all iterations are reported.
    Above ``50``, the output is sent to stdout.

    By default ``0``.
"""

Method = Literal["naive", "base", "plus", "minmax"]
AggFunction = Literal["mean", "median"]


_UNLINKED_FLAG: Final = "UNLINKED"


def _as_mapie_estimator(model: Any) -> Any:
    """Converts a generic model into an estimator suitable for mapie.MapieRegressor.

    This allows us to expand the existing mapie interface to accept other inputs,
    such as molflux.modelzoo estimators.

    Returns:
        An estimator that is compatible with the interface expected by mapie.
    """
    if model == _UNLINKED_FLAG:
        raise ValueError(
            "The input estimator has been unlinked. This is likely to have happened because you are reusing a pre-trained MAPIE model. Please train a new model from scratch instead.",
        )

    # Pass-through the native types accepted by mapie.MapieRegressor
    if model is None or isinstance(model, RegressorMixin):
        return model

    # a dictionary is assumed to be a modelzoo config
    if isinstance(model, dict):
        try:
            model = load_from_dict(model)
        except Exception as e:
            raise ValueError(
                f"Could not generate a modelzoo model from the estimator config {model}",
            ) from e

    # Allow to pass generic modelzoo architectures as well
    # TODO(avianello): for the time being we will only allow modezloo models
    #  explicitly backed by a sklearn model. In the future we could expand by
    #  creating an adapter class to turn any modelzoo estimator into the
    #  interface accepted by mapie
    if isinstance(model, Estimator):
        if isinstance(model, SKLearnModelBase):
            # warning: this is not a method provided by the Estimator Protocol
            # it needs to be called because the model.model attribute is None
            # before fitting a modelzoo model
            if model.model is None:
                return model._instantiate_model()
            return model.model
        raise NotImplementedError(
            f"Unsupported estimator architecture: {model.__class__!r}",
        )

    raise TypeError(f"Invalid estimator type for MapieRegressor: {type(model)!r}")


class Config:
    arbitrary_types_allowed = True
    extra = "forbid"


@dataclass(config=Config)
class MapieRegressorConfig(ModelConfig):
    estimator: Union[Estimator, RegressorMixin, None, str, Dict] = None
    method: Method = "plus"
    cv: Optional[Union[int, str, BaseCrossValidator]] = None
    n_jobs: Optional[int] = None
    agg_function: Optional[AggFunction] = "mean"
    verbose: int = 0

    def __post_init_post_parse__(self) -> None:
        if self.y_features and len(self.y_features) != 1:
            raise NotImplementedError(
                f"This model architecture only supports single task regression for now: got {self.y_features}",
            )

        if self.estimator == _UNLINKED_FLAG:
            warnings.warn(
                "Model loaded with an unlinked input estimator. This is expected if loading a pre-trained MAPIE model. Note that the resulting model can only be used for prediction and cannot be retrained.",
                UserWarning,
                stacklevel=1,
            )


class MapieRegressor(
    UncertaintyCalibrationMixin,
    PredictionIntervalMixin,
    ModelBase[MapieRegressorConfig],
):
    @property
    def _config_builder(self) -> Type[MapieRegressorConfig]:
        return MapieRegressorConfig

    @property
    def config(self) -> Dict[str, Any]:
        # handle unserializable config fields
        config = asdict(self.model_config)
        config["estimator"] = _UNLINKED_FLAG
        return config

    def _info(self) -> ModelInfo:
        return ModelInfo(
            model_description=_DESCRIPTION,
            config_description=_CONFIG_DESCRIPTION,
        )

    def _instantiate_model(self) -> MapieMapieRegressor:
        config = self.model_config
        # the conformity score is created on the fly
        conformity_score = AbsoluteConformityScore()
        # avoids automatic checks that sometimes produce errors depending on
        # the dataset and machine due to machine decimal errors
        # when using the conformal methods provided by mapie, there are
        # mathematical guarantees that such checks are redundant
        conformity_score.consistency_check = False
        mapie_regressor = MapieMapieRegressor(
            estimator=_as_mapie_estimator(config.estimator),
            method=config.method,
            cv=config.cv,
            n_jobs=config.n_jobs,
            agg_function=config.agg_function,
            verbose=config.verbose,
            conformity_score=conformity_score,
        )
        return mapie_regressor

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

        # instantiate model
        self.model = self._instantiate_model()

        # train
        self.model.fit(X, y, **kwargs)

    def _predict(
        self,
        data: datasets.Dataset,
        use_ensemble_predictions: bool = True,
        **kwargs: Any,
    ) -> PredictionResult:
        # the conformal predictions require an alpha, which is 1-confidence, between 1/n and (n-1)/n
        # here, we'll 1/2 which respects that for any n>=2
        default_confidence = 0.5

        prediction_result, _ = self._predict_with_prediction_interval(
            data,
            confidence=default_confidence,
            use_ensemble_predictions=use_ensemble_predictions,
            **kwargs,
        )
        return prediction_result

    def _calibrate_uncertainty(self, data: datasets.Dataset, **kwargs: Any) -> Any:
        return self._train(train_data=data, **kwargs)

    def _predict_with_prediction_interval(
        self,
        data: datasets.Dataset,
        confidence: float,
        use_ensemble_predictions: bool = True,
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

        # Evaluate prediction on testing set
        y_predict, y_pis = self.model.predict(
            X,
            ensemble=use_ensemble_predictions,
            alpha=1 - confidence,
        )
        lower_bound, upper_bound = y_pis[:, 0, 0], y_pis[:, 1, 0]

        # Check that predictions match expected features
        y_features = self.y_features
        if y_predict.ndim == 1:
            if len(y_features) != 1:
                raise ValueError(
                    f"Predictions do not have expected shape: {np.shape(y_predict)!r}",
                )
        elif y_predict.shape[1] != len(y_features):
            raise ValueError(
                f"Predictions do not have expected shape: {np.shape(y_predict)!r}",
            )

        # this model architecture only supports single task models for now
        return {
            display_name: y_predict.tolist()
            for display_name in prediction_display_names
        }, {
            display_name: list(zip(lower_bound, upper_bound))
            for display_name in prediction_interval_display_names
        }
