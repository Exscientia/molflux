from typing import Any, Literal, Optional, Type, Union

from numpy.random import RandomState
from pydantic.dataclasses import dataclass

import datasets
from molflux.modelzoo.info import ModelInfo
from molflux.modelzoo.model import ModelConfig
from molflux.modelzoo.models.sklearn import (
    SKLearnClassificationMixin,
    SKLearnModelBase,
)
from molflux.modelzoo.typing import PredictionResult

try:
    from sklearn.svm import SVC
except ImportError as e:
    from molflux.modelzoo.errors import ExtrasDependencyImportError

    raise ExtrasDependencyImportError("sklearn", e) from None


_DESCRIPTION = """
This is an sklearn support vector classifier model.

The objective of a support vector machine algorithm is to find a hyperplane in an
n-dimensional space that best separates classes of the data. The data points on
either side of the hyperplane that are closest to the hyperplane are called Support
Vectors. These influence the position and orientation of the hyperplane and thus help
build the SVM.
"""

_CONFIG_DESCRIPTION = """
Parameters
----------
C : float, default=1.0
    Regularization parameter. The strength of the regularization is
    inversely proportional to C. Must be strictly positive. The penalty
    is a squared l2 penalty.
kernel : {'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'} or callable,  \
    default='rbf'
    Specifies the kernel type to be used in the algorithm.
    If none is given, 'rbf' will be used. If a callable is given it is
    used to pre-compute the kernel matrix from data matrices; that matrix
    should be an array of shape ``(n_samples, n_samples)``.
degree : int, default=3
    Degree of the polynomial kernel function ('poly').
    Ignored by all other kernels.
gamma : {'scale', 'auto'} or float, default='scale'
    Kernel coefficient for 'rbf', 'poly' and 'sigmoid'.
    - if ``gamma='scale'`` (default) is passed then it uses
      1 / (n_features * X.var()) as value of gamma,
    - if 'auto', uses 1 / n_features.
coef0 : float, default=0.0
    Independent term in kernel function.
    It is only significant in 'poly' and 'sigmoid'.
shrinking : bool, default=True
    Whether to use the shrinking heuristic.
probability : bool, default=False
    Whether to enable probability estimates. This must be enabled prior
    to calling `fit`, will slow down that method as it internally uses
    5-fold cross-validation, and `predict_proba` may be inconsistent with
    `predict`.
tol : float, default=1e-3
    Tolerance for stopping criterion.
cache_size : float, default=200
    Specify the size of the kernel cache (in MB).
class_weight : dict or 'balanced', default=None
    Set the parameter C of class i to class_weight[i]*C for
    SVC. If not given, all classes are supposed to have
    weight one.
    The "balanced" mode uses the values of y to automatically adjust
    weights inversely proportional to class frequencies in the input data
    as ``n_samples / (n_classes * np.bincount(y))``.
verbose : bool, default=False
    Enable verbose output. Note that this setting takes advantage of a
    per-process runtime setting in libsvm that, if enabled, may not work
    properly in a multithreaded context.
max_iter : int, default=-1
    Hard limit on iterations within solver, or -1 for no limit.
decision_function_shape : {'ovo', 'ovr'}, default='ovr'
    Whether to return a one-vs-rest ('ovr') decision function of shape
    (n_samples, n_classes) as all other classifiers, or the original
    one-vs-one ('ovo') decision function of libsvm which has shape
    (n_samples, n_classes * (n_classes - 1) / 2). However, note that
    internally, one-vs-one ('ovo') is always used as a multi-class strategy
    to train models; an ovr matrix is only constructed from the ovo matrix.
    The parameter is ignored for binary classification.
break_ties : bool, default=False
    If true, ``decision_function_shape='ovr'``, and number of classes > 2,
    :term:`predict` will break ties according to the confidence values of
    :term:`decision_function`; otherwise the first class among the tied
    classes is returned. Please note that breaking ties comes at a
    relatively high computational cost compared to a simple predict.
random_state : int, RandomState instance or None, default=None
    Controls the pseudo random number generation for shuffling the data for
    probability estimates. Ignored when `probability` is False.
    Pass an int for reproducible output across multiple function calls.
"""

Kernel = Literal["linear", "poly", "rbf", "sigmoid", "precomputed"]
Gamma = Union[Literal["scale", "auto"], float]
ClassWeight = Optional[Union[dict, Literal["balanced"]]]
DecisionFunctionShape = Literal["ovo", "ovr"]


class Config:
    arbitrary_types_allowed = True
    extra = "forbid"


@dataclass(config=Config)
class SupportVectorClassifierConfig(ModelConfig):
    C: float = 1
    kernel: Kernel = "rbf"
    degree: int = 3
    gamma: Gamma = "scale"
    coef0: float = 0
    shrinking: bool = True
    probability: bool = False
    tol: float = 1e-3
    cache_size: float = 200
    class_weight: ClassWeight = "balanced"
    verbose: bool = False
    max_iter: int = -1
    decision_function_shape: DecisionFunctionShape = "ovr"
    break_ties: bool = False
    random_state: Optional[Union[int, RandomState]] = None


class SupportVectorClassifier(
    SKLearnClassificationMixin,
    SKLearnModelBase[SupportVectorClassifierConfig],
):
    @property
    def _config_builder(self) -> Type[SupportVectorClassifierConfig]:
        return SupportVectorClassifierConfig

    def _info(self) -> ModelInfo:
        return ModelInfo(
            model_description=_DESCRIPTION,
            config_description=_CONFIG_DESCRIPTION,
        )

    def _instantiate_model(self) -> SVC:
        config = self.model_config
        return SVC(
            C=config.C,
            kernel=config.kernel,
            degree=config.degree,
            gamma=config.gamma,
            coef0=config.coef0,
            shrinking=config.shrinking,
            probability=config.probability,
            tol=config.tol,
            cache_size=config.cache_size,
            class_weight=config.class_weight,
            verbose=config.verbose,
            max_iter=config.max_iter,
            decision_function_shape=config.decision_function_shape,
            break_ties=config.break_ties,
            random_state=config.random_state,
        )

    def _predict_proba(self, data: datasets.Dataset, **kwargs: Any) -> PredictionResult:
        if not self.model_config.probability:
            raise RuntimeError(
                "Cannot use the predict_proba for this Support Vector Classifier, as the model config has"
                "probability=False. If you want to use predict_proba function, make sure to set probability=True in the model config before training the model.",
            )
        else:
            return super()._predict_proba(data, **kwargs)
