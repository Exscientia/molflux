from typing import Literal

from pydantic.v1 import dataclasses

from molflux.modelzoo.info import ModelInfo
from molflux.modelzoo.model import ModelConfig
from molflux.modelzoo.models.sklearn import SKLearnModelBase

try:
    from sklearn.ensemble import ExtraTreesRegressor as SKExtraTreesRegressor
except ImportError as e:
    from molflux.modelzoo.errors import ExtrasDependencyImportError

    raise ExtrasDependencyImportError("sklearn", e) from None


_DESCRIPTION = """
This is an sklearn extra trees regressor model.

The extra trees regressor is a meta estimator that fits a number of randomized decision
trees (a.k.a. extra-trees) on various sub-samples of the dataset and uses averaging to
improve the predictive accuracy and control over-fitting.
"""

_CONFIG_DESCRIPTION = """
Parameters
----------
n_estimators : int, default=100
    The number of trees in the forest.
criterion : {"squared_error", "absolute_error", "friedman_mse", "poisson"}, \
        default="squared_error"
    The function to measure the quality of a split. Supported criteria
    are "squared_error" for the mean squared error, which is equal to
    variance reduction as feature selection criterion and minimizes the L2
    loss using the mean of each terminal node, "friedman_mse", which uses
    mean squared error with Friedman's improvement score for potential
    splits, "absolute_error" for the mean absolute error, which minimizes
    the L1 loss using the median of each terminal node, and "poisson" which
    uses reduction in Poisson deviance to find splits.
    Training using "absolute_error" is significantly slower
    than when using "squared_error".
max_depth : int, default=None
    The maximum depth of the tree. If None, then nodes are expanded until
    all leaves are pure or until all leaves contain less than
    min_samples_split samples.
min_samples_split : int or float, default=2
    The minimum number of samples required to split an internal node:
    - If int, then consider `min_samples_split` as the minimum number.
    - If float, then `min_samples_split` is a fraction and
      `ceil(min_samples_split * n_samples)` are the minimum
      number of samples for each split.
min_samples_leaf : int or float, default=1
    The minimum number of samples required to be at a leaf node.
    A split point at any depth will only be considered if it leaves at
    least ``min_samples_leaf`` training samples in each of the left and
    right branches.  This may have the effect of smoothing the model,
    especially in regression.
    - If int, then consider `min_samples_leaf` as the minimum number.
    - If float, then `min_samples_leaf` is a fraction and
      `ceil(min_samples_leaf * n_samples)` are the minimum
      number of samples for each node.
min_weight_fraction_leaf : float, default=0.0
    The minimum weighted fraction of the sum total of weights (of all
    the input samples) required to be at a leaf node. Samples have
    equal weight when sample_weight is not provided.
max_features : {"sqrt", "log2", None}, int or float, default=1.0
    The number of features to consider when looking for the best split:
    - If int, then consider `max_features` features at each split.
    - If float, then `max_features` is a fraction and
      `max(1, int(max_features * n_features_in_))` features are considered at each
      split.
    - If "sqrt", then `max_features=sqrt(n_features)`.
    - If "log2", then `max_features=log2(n_features)`.
    - If None or 1.0, then `max_features=n_features`.
    .. note::
        The default of 1.0 is equivalent to bagged trees and more
        randomness can be achieved by setting smaller values, e.g. 0.3.
    Note: the search for a split does not stop until at least one
    valid partition of the node samples is found, even if it requires to
    effectively inspect more than ``max_features`` features.
max_leaf_nodes : int, default=None
    Grow trees with ``max_leaf_nodes`` in best-first fashion.
    Best nodes are defined as relative reduction in impurity.
    If None then unlimited number of leaf nodes.
min_impurity_decrease : float, default=0.0
    A node will be split if this split induces a decrease of the impurity
    greater than or equal to this value.
    The weighted impurity decrease equation is the following::
        N_t / N * (impurity - N_t_R / N_t * right_impurity
                            - N_t_L / N_t * left_impurity)
    where ``N`` is the total number of samples, ``N_t`` is the number of
    samples at the current node, ``N_t_L`` is the number of samples in the
    left child, and ``N_t_R`` is the number of samples in the right child.
    ``N``, ``N_t``, ``N_t_R`` and ``N_t_L`` all refer to the weighted sum,
    if ``sample_weight`` is passed.
bootstrap : bool, default=False
    Whether bootstrap samples are used when building trees. If False, the
    whole dataset is used to build each tree.
oob_score : bool or callable, default=False
    Whether to use out-of-bag samples to estimate the generalization score.
    By default, :func:`~sklearn.metrics.r2_score` is used.
    Provide a callable with signature `metric(y_true, y_pred)` to use a
    custom metric. Only available if `bootstrap=True`.
n_jobs : int, default=None
    The number of jobs to run in parallel. :meth:`fit`, :meth:`predict`,
    :meth:`decision_path` and :meth:`apply` are all parallelized over the
    trees. ``None`` means 1 unless in a :obj:`joblib.parallel_backend`
    context. ``-1`` means using all processors.
random_state : int or None, default=None
    Controls 3 sources of randomness:
    - the bootstrapping of the samples used when building trees
      (if ``bootstrap=True``)
    - the sampling of the features to consider when looking for the best
      split at each node (if ``max_features < n_features``)
    - the draw of the splits for each of the `max_features`
verbose : int, default=0
    Controls the verbosity when fitting and predicting.
warm_start : bool, default=False
    When set to ``True``, reuse the solution of the previous call to fit
    and add more estimators to the ensemble, otherwise, just fit a whole
    new forest.
ccp_alpha : non-negative float, default=0.0
    Complexity parameter used for Minimal Cost-Complexity Pruning. The
    subtree with the largest cost complexity that is smaller than
    ``ccp_alpha`` will be chosen. By default, no pruning is performed.
max_samples : int or float, default=None
    If bootstrap is True, the number of samples to draw from X
    to train each base estimator.
    - If None (default), then draw `X.shape[0]` samples.
    - If int, then draw `max_samples` samples.
    - If float, then draw `max_samples * X.shape[0]` samples. Thus,
      `max_samples` should be in the interval `(0.0, 1.0]`.
"""


Criterion = Literal["squared_error", "absolute_error", "friedman_mse", "poisson"]
MaxFeaturesCallable = Literal["sqrt", "log2"]


class Config:
    extra = "forbid"
    arbitrary_types_allowed = True
    smart_union = True


@dataclasses.dataclass(config=Config)
class ExtraTreesRegressorConfig(ModelConfig):
    n_estimators: int = 100
    criterion: Criterion = "squared_error"
    max_depth: int | None = None
    min_samples_split: int | float = 2
    min_samples_leaf: int | float = 1
    min_weight_fraction_leaf: float = 0.0
    max_features: float | MaxFeaturesCallable | None = "sqrt"
    max_leaf_nodes: int | None = None
    min_impurity_decrease: float = 0.0
    bootstrap: bool = False
    oob_score: bool = False
    n_jobs: int | None = None
    random_state: int | None = None
    verbose: int = 0
    warm_start: bool = False
    ccp_alpha: float = 0.0
    max_samples: int | float | None = None

    def __post_init_post_parse__(self) -> None:
        if self.ccp_alpha < 0:
            raise ValueError("Complexity parameter `ccp_alpha` cannot be negative.")


class ExtraTreesRegressor(SKLearnModelBase[ExtraTreesRegressorConfig]):
    @property
    def _config_builder(self) -> type[ExtraTreesRegressorConfig]:
        return ExtraTreesRegressorConfig

    def _info(self) -> ModelInfo:
        return ModelInfo(
            model_description=_DESCRIPTION,
            config_description=_CONFIG_DESCRIPTION,
        )

    def _instantiate_model(self) -> SKExtraTreesRegressor:
        config = self.model_config
        return SKExtraTreesRegressor(
            n_estimators=config.n_estimators,
            criterion=config.criterion,
            max_depth=config.max_depth,
            min_samples_split=config.min_samples_split,
            min_samples_leaf=config.min_samples_leaf,
            min_weight_fraction_leaf=config.min_weight_fraction_leaf,
            max_features=config.max_features,
            max_leaf_nodes=config.max_leaf_nodes,
            min_impurity_decrease=config.min_impurity_decrease,
            bootstrap=config.bootstrap,
            oob_score=config.oob_score,
            n_jobs=config.n_jobs,
            random_state=config.random_state,
            verbose=config.verbose,
            warm_start=config.warm_start,
            ccp_alpha=config.ccp_alpha,
            max_samples=config.max_samples,
        )
