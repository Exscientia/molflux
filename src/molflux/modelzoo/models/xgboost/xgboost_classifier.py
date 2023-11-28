from typing import Callable, Dict, List, Optional, Tuple, Type, Union

import numpy as np
from numpy.random import RandomState
from pydantic.dataclasses import dataclass

from molflux.modelzoo.info import ModelInfo
from molflux.modelzoo.model import ModelConfig
from molflux.modelzoo.models.sklearn import (
    SKLearnClassificationMixin,
    SKLearnModelBase,
)

try:
    from xgboost.sklearn import TrainingCallback, _SklObjective
    from xgboost.sklearn import XGBClassifier as XGBC
except ImportError as e:
    from molflux.modelzoo.errors import ExtrasDependencyImportError

    raise ExtrasDependencyImportError("xgboost", e) from None


_DESCRIPTION = """
Implementation of the scikit-learn API for XGBoost classification.

Extreme Gradient Boosting, or XGBoost for short is an efficient
open-source implementation of the gradient boosting algorithm.

Gradient boosting refers to a class of ensemble machine learning
algorithms that can be used for classification or regression predictive
modeling problems.

Ensembles are constructed from decision tree models. Trees are added one
at a time to the ensemble and fit to correct the prediction errors made
by prior models. This is a type of ensemble machine learning model referred
to as boosting.

Note:
    This Classifier expects already encoded classes for the labels. For a training
    dataset with 1 task and N labels, it assumes that all values [0, 1, 2... N-1] are
    in the label columns. No other formats are permitted ([1...N], strings, chars etc.)
"""

_CONFIG_DESCRIPTION = """
Parameters
----------
n_estimators: int = 100
    Number of estimators, equivalent to the number of boosting rounds.
max_depth :  Optional[int]
    Maximum tree depth for base learners.
max_leaves : Optional[int]
    Maximum number of leaves; 0 indicates no limit.
max_bin : Optional[int]
    If using histogram-based algorithm, maximum number of bins per feature
grow_policy :
    Tree growing policy. 0: favor splitting at nodes closest to the node, i.e. grow
    depth-wise. 1: favor splitting at nodes with highest loss change.
learning_rate : Optional[float]
    Boosting learning rate (xgb's "eta")
verbosity : Optional[int]
    The degree of verbosity. Valid values are 0 (silent) - 3 (debug).
objective : {_SklObjective}
    Specify the learning task and the corresponding learning objective or
    a custom objective function to be used (see note below).
booster: Optional[str]
    Specify which booster to use: gbtree, gblinear or dart.
tree_method: Optional[str]
    Specify which tree method to use.  Default to auto.  If this parameter is set to
    default, XGBoost will choose the most conservative option available.  It's
    recommended to study this option from the parameters document
n_jobs : Optional[int]
    Number of parallel threads used to run xgboost.  When used with other Scikit-Learn
    algorithms like grid search, you may choose which algorithm to parallelize and
    balance the threads.  Creating thread contention will significantly slow down both
    algorithms.
gamma : Optional[float]
    (min_split_loss) Minimum loss reduction required to make a further partition on a
    leaf node of the tree.
min_child_weight : Optional[float]
    Minimum sum of instance weight(hessian) needed in a child.
max_delta_step : Optional[float]
    Maximum delta step we allow each tree's weight estimation to be.
subsample : Optional[float]
    Subsample ratio of the training instance.
sampling_method :
    Sampling method. Used only by `gpu_hist` tree method.
      - `uniform`: select random training instances uniformly.
      - `gradient_based` select random training instances with higher probability when
        the gradient and hessian are larger. (cf. CatBoost)
colsample_bytree : Optional[float]
    Subsample ratio of columns when constructing each tree.
colsample_bylevel : Optional[float]
    Subsample ratio of columns for each level.
colsample_bynode : Optional[float]
    Subsample ratio of columns for each split.
reg_alpha : Optional[float]
    L1 regularization term on weights (xgb's alpha).
reg_lambda : Optional[float]
    L2 regularization term on weights (xgb's lambda).
scale_pos_weight : Optional[float]
    Balancing of positive and negative weights.
base_score : Optional[float]
    The initial prediction score of all instances, global bias.
random_state : Optional[Union[numpy.random.RandomState, int]]
    Random number seed.
    .. note::
       Using gblinear booster with shotgun updater is nondeterministic as
       it uses Hogwild algorithm.
missing : Optional[float], default None
    Value in the data which needs to be present as a missing value.
num_parallel_tree: Optional[int]
    Used for boosting random forest.
monotone_constraints : Optional[Union[Dict[str, int], str]]
    Constraint of variable monotonicity.
interaction_constraints : Optional[Union[str, List[Tuple[str]]]]
    Constraints for interaction representing permitted interactions.  The
    constraints must be specified in the form of a nested list, e.g. ``[[0, 1], [2,
    3, 4]]``, where each inner list is a group of indices of features that are
    allowed to interact with each other.
importance_type: Optional[str]
    The feature importance type for the feature_importances\\_ property:
    * For tree model, it's either "gain", "weight", "cover", "total_gain" or
      "total_cover".
    * For linear model, only "weight" is defined and it's the normalized coefficients
      without bias.
gpu_id : Optional[int]
    Device ordinal.
validate_parameters : Optional[bool]
    Give warnings for unknown parameter.
predictor : Optional[str]
    Force XGBoost to use specific predictor, available choices are [cpu_predictor,
    gpu_predictor].
enable_categorical : bool
    .. note:: This parameter is experimental
    Experimental support for categorical data.  When enabled, cudf/pandas.DataFrame
    should be used to specify categorical data type.  Also, JSON/UBJSON
    serialization format is required.
feature_types : FeatureTypes
    Used for specifying feature types without constructing a dataframe.
max_cat_to_onehot : Optional[int]
    .. note:: This parameter is experimental
    A threshold for deciding whether XGBoost should use one-hot encoding based split
    for categorical data.  When number of categories is lesser than the threshold
    then one-hot encoding is chosen, otherwise the categories will be partitioned
    into children nodes.  Only relevant for regression and binary classification.
eval_metric : Optional[Union[str, List[str], Callable]]
    Metric used for monitoring the training result and early stopping.  It can be a
    string or list of strings as names of predefined metric in XGBoost (See
    doc/parameter.rst), one of the metrics in :py:mod:`sklearn.metrics`, or any other
    user defined metric that looks like `sklearn.metrics`.
    If custom objective is also provided, then custom metric should implement the
    corresponding reverse link function.
    Unlike the `scoring` parameter commonly used in scikit-learn, when a callable
    object is provided, it's assumed to be a cost function and by default XGBoost will
    minimize the result during early stopping.
    .. note::
         This parameter replaces `eval_metric` in :py:meth:`fit` method.  The old one
         receives un-transformed prediction regardless of whether custom objective is
         being used.
    .. code-block:: python
        from sklearn.datasets import load_diabetes
        from sklearn.metrics import mean_absolute_error
        X, y = load_diabetes(return_X_y=True)
        reg = xgb.XGBRegressor(
            tree_method="hist",
            eval_metric=mean_absolute_error,
        )
        reg.fit(X, y, eval_set=[(X, y)])
early_stopping_rounds : Optional[int]
    Activates early stopping. Validation metric needs to improve at least once in
    every **early_stopping_rounds** round(s) to continue training.  Requires at least
    one item in **eval_set** in :py:meth:`fit`.
    The method returns the model from the last iteration (not the best one).  If
    there's more than one item in **eval_set**, the last entry will be used for early
    stopping.  If there's more than one metric in **eval_metric**, the last metric
    will be used for early stopping.
    If early stopping occurs, the model will have three additional fields:
    :py:attr:`best_score`, :py:attr:`best_iteration` and
    :py:attr:`best_ntree_limit`.
    .. note::
        This parameter replaces `early_stopping_rounds` in :py:meth:`fit` method.
callbacks : Optional[List[TrainingCallback]]
    List of callback functions that are applied at end of each iteration.
    .. note::
       States in callback are not preserved during training, which means callback
       objects can not be reused for multiple training sessions without
       reinitialization or deepcopy.
    .. code-block:: python
        for params in parameters_grid:
            # be sure to (re)initialize the callbacks before each run
            callbacks = [xgb.callback.LearningRateScheduler(custom_rates)]
            xgboost.train(params, Xy, callbacks=callbacks)
kwargs : dict, optional
    Keyword arguments for XGBoost Booster object.  Full documentation of parameters
    can be found :doc:`here </parameter>`.
    Attempting to set a parameter via the constructor args and \\*\\*kwargs
    dict simultaneously will result in a TypeError.
    .. note:: \\*\\*kwargs unsupported by scikit-learn
        \\*\\*kwargs is unsupported by scikit-learn.  We do not guarantee
        that parameters passed via this argument will interact properly
        with scikit-learn.
"""


class Config:
    arbitrary_types_allowed = True
    extra = "forbid"


@dataclass(config=Config)
class XGBClassifierConfig(ModelConfig):
    n_estimators: int = 100
    max_depth: Optional[int] = None
    max_leaves: Optional[int] = None
    max_bin: Optional[int] = None
    grow_policy: Optional[int] = None
    learning_rate: Optional[float] = None
    verbosity: Optional[int] = None
    objective: _SklObjective = None
    booster: Optional[str] = None
    tree_method: Optional[str] = None
    n_jobs: Optional[int] = None
    gamma: Optional[float] = None
    min_child_weight: Optional[float] = None
    max_delta_step: Optional[float] = None
    subsample: Optional[float] = None
    sampling_method: Optional[str] = None
    colsample_bytree: Optional[float] = None
    colsample_bylevel: Optional[float] = None
    colsample_bynode: Optional[float] = None
    reg_alpha: Optional[float] = None
    reg_lambda: Optional[float] = None
    scale_pos_weight: Optional[float] = None
    base_score: Optional[float] = None
    random_state: Optional[Union[int, RandomState]] = None
    missing: Optional[float] = None
    num_parallel_tree: Optional[int] = None
    monotone_constraints: Optional[Union[Dict[str, int], str]] = None
    interaction_constraints: Optional[Union[str, List[Tuple[str]]]] = None
    importance_type: Optional[str] = None
    gpu_id: Optional[int] = None
    validate_parameters: Optional[bool] = None
    predictor: Optional[str] = None
    enable_categorical: bool = False
    max_cat_to_onehot: Optional[int] = None
    eval_metric: Optional[Union[str, List[str], Callable]] = None
    early_stopping_rounds: Optional[int] = None
    callbacks: Optional[List[TrainingCallback]] = None
    kwargs: Optional[dict] = None


class XGBoostClassifier(
    SKLearnClassificationMixin,
    SKLearnModelBase[XGBClassifierConfig],
):
    @property
    def _config_builder(self) -> Type[XGBClassifierConfig]:
        return XGBClassifierConfig

    def _info(self) -> ModelInfo:
        return ModelInfo(
            model_description=_DESCRIPTION,
            config_description=_CONFIG_DESCRIPTION,
        )

    def _instantiate_model(self) -> XGBC:
        config = self.model_config

        # Converting None to np.nan, as required by XGBoost
        missing = np.nan if not config.missing else config.missing

        return XGBC(
            n_estimators=config.n_estimators,
            max_depth=config.max_depth,
            max_leaves=config.max_leaves,
            max_bin=config.max_bin,
            grow_policy=config.grow_policy,
            learning_rate=config.learning_rate,
            verbosity=config.verbosity,
            objective=config.objective,
            booster=config.booster,
            tree_method=config.tree_method,
            n_jobs=config.n_jobs,
            gamma=config.gamma,
            min_child_weight=config.min_child_weight,
            max_delta_step=config.max_delta_step,
            subsample=config.subsample,
            sampling_method=config.sampling_method,
            colsample_bytree=config.colsample_bytree,
            colsample_bylevel=config.colsample_bylevel,
            colsample_bynode=config.colsample_bynode,
            reg_alpha=config.reg_alpha,
            reg_lambda=config.reg_lambda,
            scale_pos_weight=config.scale_pos_weight,
            base_score=config.base_score,
            random_state=config.random_state,
            missing=missing,
            num_parallel_tree=config.num_parallel_tree,
            monotone_constraints=config.monotone_constraints,
            interaction_constraints=config.interaction_constraints,
            importance_type=config.importance_type,
            gpu_id=config.gpu_id,
            validate_parameters=config.validate_parameters,
            predictor=config.predictor,
            enable_categorical=config.enable_categorical,
            max_cat_to_onehot=config.max_cat_to_onehot,
            eval_metric=config.eval_metric,
            early_stopping_rounds=config.early_stopping_rounds,
            callbacks=config.callbacks,
            kwargs=config.kwargs,
        )
