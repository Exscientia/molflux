from typing import Literal, Type, Union

from pydantic.dataclasses import dataclass

from molflux.modelzoo.info import ModelInfo
from molflux.modelzoo.model import ModelConfig
from molflux.modelzoo.models.sklearn import SKLearnModelBase

try:
    from sklearn.svm import SVR
except ImportError as e:
    from molflux.modelzoo.errors import ExtrasDependencyImportError

    raise ExtrasDependencyImportError("sklearn", e) from None


_DESCRIPTION = """
This is an sklearn support vector regressor model.

The objective of a support vector machine algorithm is to find a hyperplane in an
n-dimensional space that best separates the data. The data points on
either side of the hyperplane that are closest to the hyperplane are called Support
Vectors. These influence the position and orientation of the hyperplane and thus help
build the SVM.
"""

_CONFIG_DESCRIPTION = """
Parameters
----------
kernel : {'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'} or callable,  \
        default='rbf'
    Specifies the kernel type to be used in the algorithm.
    If none is given, 'rbf' will be used. If a callable is given it is
    used to precompute the kernel matrix.
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
tol : float, default=1e-3
    Tolerance for stopping criterion.
C : float, default=1.0
    Regularization parameter. The strength of the regularization is
    inversely proportional to C. Must be strictly positive.
    The penalty is a squared l2 penalty.
epsilon : float, default=0.1
     Epsilon in the epsilon-SVR model. It specifies the epsilon-tube
     within which no penalty is associated in the training loss function
     with points predicted within a distance epsilon from the actual
     value.
shrinking : bool, default=True
    Whether to use the shrinking heuristic.
cache_size : float, default=200
    Specify the size of the kernel cache (in MB).
verbose : bool, default=False
    Enable verbose output. Note that this setting takes advantage of a
    per-process runtime setting in libsvm that, if enabled, may not work
    properly in a multithreaded context.
max_iter : int, default=-1
    Hard limit on iterations within solver, or -1 for no limit.
"""

Kernel = Literal["linear", "poly", "rbf", "sigmoid", "precomputed"]
Gamma = Union[Literal["scale", "auto"], float]


class Config:
    arbitrary_types_allowed = True
    extra = "forbid"


@dataclass(config=Config)
class SupportVectorRegressorConfig(ModelConfig):
    kernel: Kernel = "rbf"
    degree: int = 3
    gamma: Gamma = "scale"
    coef0: float = 0
    tol: float = 1e-3
    C: float = 1
    epsilon: float = 0
    shrinking: bool = True
    cache_size: float = 200
    verbose: bool = False
    max_iter: int = -1


class SupportVectorRegressor(SKLearnModelBase[SupportVectorRegressorConfig]):
    @property
    def _config_builder(self) -> Type[SupportVectorRegressorConfig]:
        return SupportVectorRegressorConfig

    def _info(self) -> ModelInfo:
        return ModelInfo(
            model_description=_DESCRIPTION,
            config_description=_CONFIG_DESCRIPTION,
        )

    def _instantiate_model(self) -> SVR:
        config = self.model_config
        return SVR(
            kernel=config.kernel,
            degree=config.degree,
            gamma=config.gamma,
            coef0=config.coef0,
            tol=config.tol,
            C=config.C,
            epsilon=config.epsilon,
            shrinking=config.shrinking,
            cache_size=config.cache_size,
            verbose=config.verbose,
            max_iter=config.max_iter,
        )
