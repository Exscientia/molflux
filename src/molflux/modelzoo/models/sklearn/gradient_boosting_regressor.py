from typing import Literal, Optional, Type, Union

from pydantic.dataclasses import dataclass

from molflux.modelzoo.info import ModelInfo
from molflux.modelzoo.model import ModelConfig
from molflux.modelzoo.models.sklearn import SKLearnModelBase

try:
    from sklearn.ensemble import (
        GradientBoostingRegressor as SKGradientBoostingRegressor,
    )
except ImportError as e:
    from molflux.modelzoo.errors import ExtrasDependencyImportError

    raise ExtrasDependencyImportError("sklearn", e) from None


_DESCRIPTION = """
This is an sklearn gradient boosting regressor model.

This estimator builds an additive model in a forward stage-wise fashion; it
allows for the optimization of arbitrary differentiable loss functions. In
each stage a regression tree is fit on the negative gradient of the given
loss function.
"""

_CONFIG_DESCRIPTION = """
Parameters
----------
loss : {'squared_error', 'absolute_error', 'huber', 'quantile'}, default='squared_error'
    Loss function to be optimized. 'squared_error' refers to the squared
    error for regression. 'absolute_error' refers to the absolute error of
    regression and is a robust loss function. 'huber' is a
    combination of the two. 'quantile' allows quantile regression (use
    `alpha` to specify the quantile).
learning_rate : float, default=0.1
    Learning rate shrinks the contribution of each tree by `learning_rate`.
    There is a trade-off between learning_rate and n_estimators.
    Values must be in the range `[0.0, inf)`.
n_estimators : int, default=100
    The number of boosting stages to perform. Gradient boosting
    is fairly robust to over-fitting so a large number usually
    results in better performance.
    Values must be in the range `[1, inf)`.
subsample : float, default=1.0
    The fraction of samples to be used for fitting the individual base
    learners. If smaller than 1.0 this results in Stochastic Gradient
    Boosting. `subsample` interacts with the parameter `n_estimators`.
    Choosing `subsample < 1.0` leads to a reduction of variance
    and an increase in bias.
    Values must be in the range `(0.0, 1.0]`.
criterion : {'friedman_mse', 'squared_error'}, default='friedman_mse'
    The function to measure the quality of a split. Supported criteria are
    "friedman_mse" for the mean squared error with improvement score by
    Friedman, "squared_error" for mean squared error. The default value of
    "friedman_mse" is generally the best as it can provide a better
    approximation in some cases.
min_samples_split : int or float, default=2
    The minimum number of samples required to split an internal node:
    - If int, values must be in the range `[2, inf)`.
    - If float, values must be in the range `(0.0, 1.0]` and `min_samples_split`
      will be `ceil(min_samples_split * n_samples)`.
min_samples_leaf : int or float, default=1
    The minimum number of samples required to be at a leaf node.
    A split point at any depth will only be considered if it leaves at
    least ``min_samples_leaf`` training samples in each of the left and
    right branches.  This may have the effect of smoothing the model,
    especially in regression.
    - If int, values must be in the range `[1, inf)`.
    - If float, values must be in the range `(0.0, 1.0)` and `min_samples_leaf`
      will be `ceil(min_samples_leaf * n_samples)`.
min_weight_fraction_leaf : float, default=0.0
    The minimum weighted fraction of the sum total of weights (of all
    the input samples) required to be at a leaf node. Samples have
    equal weight when sample_weight is not provided.
    Values must be in the range `[0.0, 0.5]`.
max_depth : int or None, default=3
    Maximum depth of the individual regression estimators. The maximum
    depth limits the number of nodes in the tree. Tune this parameter
    for best performance; the best value depends on the interaction
    of the input variables. If None, then nodes are expanded until
    all leaves are pure or until all leaves contain less than
    min_samples_split samples.
    If int, values must be in the range `[1, inf)`.
min_impurity_decrease : float, default=0.0
    A node will be split if this split induces a decrease of the impurity
    greater than or equal to this value.
    Values must be in the range `[0.0, inf)`.
    The weighted impurity decrease equation is the following::
        N_t / N * (impurity - N_t_R / N_t * right_impurity
                            - N_t_L / N_t * left_impurity)
    where ``N`` is the total number of samples, ``N_t`` is the number of
    samples at the current node, ``N_t_L`` is the number of samples in the
    left child, and ``N_t_R`` is the number of samples in the right child.
    ``N``, ``N_t``, ``N_t_R`` and ``N_t_L`` all refer to the weighted sum,
    if ``sample_weight`` is passed.
init : None or 'zero', default=None
    An estimator object that is used to compute the initial predictions.
    If 'zero', the initial raw predictions are set to zero. By default a
    ``DummyEstimator`` is used, predicting either the average target value
    (for loss='squared_error'), or a quantile for the other losses.
random_state : int, RandomState instance or None, default=None
    Controls the random seed given to each Tree estimator at each
    boosting iteration.
    In addition, it controls the random permutation of the features at
    each split (see Notes for more details).
    It also controls the random splitting of the training data to obtain a
    validation set if `n_iter_no_change` is not None.
    Pass an int for reproducible output across multiple function calls.
max_features : {'auto', 'sqrt', 'log2'}, int or float, default=None
    The number of features to consider when looking for the best split:
    - If int, values must be in the range `[1, inf)`.
    - If float, values must be in the range `(0.0, 1.0]` and the features
      considered at each split will be `max(1, int(max_features * n_features_in_))`.
    - If "auto", then `max_features=n_features`.
    - If "sqrt", then `max_features=sqrt(n_features)`.
    - If "log2", then `max_features=log2(n_features)`.
    - If None, then `max_features=n_features`.
    Choosing `max_features < n_features` leads to a reduction of variance
    and an increase in bias.
    Note: the search for a split does not stop until at least one
    valid partition of the node samples is found, even if it requires to
    effectively inspect more than ``max_features`` features.
alpha : float, default=0.9
    The alpha-quantile of the huber loss function and the quantile
    loss function. Only if ``loss='huber'`` or ``loss='quantile'``.
    Values must be in the range `(0.0, 1.0)`.
verbose : int, default=0
    Enable verbose output. If 1 then it prints progress and performance
    once in a while (the more trees the lower the frequency). If greater
    than 1 then it prints progress and performance for every tree.
    Values must be in the range `[0, inf)`.
max_leaf_nodes : int, default=None
    Grow trees with ``max_leaf_nodes`` in best-first fashion.
    Best nodes are defined as relative reduction in impurity.
    Values must be in the range `[2, inf)`.
    If None, then unlimited number of leaf nodes.
warm_start : bool, default=False
    When set to ``True``, reuse the solution of the previous call to fit
    and add more estimators to the ensemble, otherwise, just erase the
    previous solution.
validation_fraction : float, default=0.1
    The proportion of training data to set aside as validation set for
    early stopping. Values must be in the range `(0.0, 1.0)`.
    Only used if ``n_iter_no_change`` is set to an integer.
n_iter_no_change : int, default=None
    ``n_iter_no_change`` is used to decide if early stopping will be used
    to terminate training when validation score is not improving. By
    default it is set to None to disable early stopping. If set to a
    number, it will set aside ``validation_fraction`` size of the training
    data as validation and terminate training when validation score is not
    improving in all of the previous ``n_iter_no_change`` numbers of
    iterations.
    Values must be in the range `[1, inf)`.
tol : float, default=1e-4
    Tolerance for the early stopping. When the loss is not improving
    by at least tol for ``n_iter_no_change`` iterations (if set to a
    number), the training stops.
    Values must be in the range `[0.0, inf)`.
ccp_alpha : non-negative float, default=0.0
    Complexity parameter used for Minimal Cost-Complexity Pruning. The
    subtree with the largest cost complexity that is smaller than
    ``ccp_alpha`` will be chosen. By default, no pruning is performed.
    Values must be in the range `[0.0, inf)`.
"""

Loss = Literal["squared_error", "absolute_error", "huber", "quantile"]
Criterion = Literal["friedman_mse", "squared_error"]
Init = Literal["zero"]
MaxFeatures = Literal["auto", "sqrt", "log2"]


class Config:
    extra = "forbid"
    arbitrary_types_allowed = True
    smart_union = True


@dataclass(config=Config)
class GradientBoostingRegressorConfig(ModelConfig):
    loss: Loss = "squared_error"
    learning_rate: float = 0.1
    n_estimators: int = 100
    subsample: float = 1.0
    criterion: Criterion = "friedman_mse"
    min_samples_split: Union[int, float] = 2
    min_samples_leaf: Union[int, float] = 1
    min_weight_fraction_leaf: float = 0.0
    max_depth: Optional[int] = 3
    min_impurity_decrease: float = 0.0
    init: Optional[Init] = None
    random_state: Optional[int] = None
    max_features: Optional[Union[int, float, MaxFeatures]] = None
    alpha: float = 0.9
    verbose: int = 0
    max_leaf_nodes: Optional[int] = None
    warm_start: bool = False
    validation_fraction: float = 0.1
    n_iter_no_change: Optional[int] = None
    tol: float = 1e-4
    ccp_alpha: float = 0.0

    def __post_init_post_parse__(self) -> None:
        if self.ccp_alpha < 0:
            raise ValueError("Complexity parameter `ccp_alpha` cannot be negative.")


class GradientBoostingRegressor(SKLearnModelBase[GradientBoostingRegressorConfig]):
    @property
    def _config_builder(self) -> Type[GradientBoostingRegressorConfig]:
        return GradientBoostingRegressorConfig

    def _info(self) -> ModelInfo:
        return ModelInfo(
            model_description=_DESCRIPTION,
            config_description=_CONFIG_DESCRIPTION,
        )

    def _instantiate_model(self) -> SKGradientBoostingRegressor:
        config = self.model_config
        return SKGradientBoostingRegressor(
            loss=config.loss,
            learning_rate=config.learning_rate,
            n_estimators=config.n_estimators,
            subsample=config.subsample,
            criterion=config.criterion,
            min_samples_split=config.min_samples_split,
            min_samples_leaf=config.min_samples_leaf,
            min_weight_fraction_leaf=config.min_weight_fraction_leaf,
            max_depth=config.max_depth,
            min_impurity_decrease=config.min_impurity_decrease,
            init=config.init,
            random_state=config.random_state,
            max_features=config.max_features,
            alpha=config.alpha,
            verbose=config.verbose,
            max_leaf_nodes=config.max_leaf_nodes,
            warm_start=config.warm_start,
            validation_fraction=config.validation_fraction,
            n_iter_no_change=config.n_iter_no_change,
            tol=config.tol,
            ccp_alpha=config.ccp_alpha,
        )
