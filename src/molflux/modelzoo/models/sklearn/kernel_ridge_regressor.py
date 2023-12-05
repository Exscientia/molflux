from typing import Dict, List, Optional, Type, Union

from pydantic.dataclasses import dataclass

from molflux.modelzoo.info import ModelInfo
from molflux.modelzoo.model import ModelConfig
from molflux.modelzoo.models.sklearn import SKLearnModelBase

try:
    from sklearn.kernel_ridge import KernelRidge as KR
except ImportError as e:
    from molflux.modelzoo.errors import ExtrasDependencyImportError

    raise ExtrasDependencyImportError("sklearn", e) from None


_DESCRIPTION = """
Kernel ridge regression (KRR) combines ridge regression (linear least
squares with l2-norm regularization) with the kernel trick. It thus
learns a linear function in the space induced by the respective kernel and
the data. For non-linear kernels, this corresponds to a non-linear
function in the original space.

The form of the model learned by KRR is identical to support vector
regression (SVR). However, different loss functions are used: KRR uses
squared error loss while support vector regression uses epsilon-insensitive
loss, both combined with l2 regularization. In contrast to SVR, fitting a
KRR model can be done in closed-form and is typically faster for
medium-sized datasets. On the other hand, the learned model is non-sparse
and thus slower than SVR, which learns a sparse model for epsilon > 0, at
prediction-time.
"""

_CONFIG_DESCRIPTION = """
Parameters
----------
alpha : float or array-like of shape (n_targets,), default=1.0
    Regularization strength; must be a positive float. Regularization
    improves the conditioning of the problem and reduces the variance of
    the estimates. Larger values specify stronger regularization.
    Alpha corresponds to ``1 / (2C)`` in other linear models such as
    :class:`~sklearn.linear_model.LogisticRegression` or
    :class:`~sklearn.svm.LinearSVC`. If an array is passed, penalties are
    assumed to be specific to the targets. Hence they must correspond in
    number.
kernel : str or callable, default="linear"
    Kernel mapping used internally. This parameter is directly passed to
    :class:`~sklearn.metrics.pairwise.pairwise_kernel`.
    If `kernel` is a string, it must be one of the metrics
    in `pairwise.PAIRWISE_KERNEL_FUNCTIONS` or "precomputed".
    If `kernel` is "precomputed", X is assumed to be a kernel matrix.
    Alternatively, if `kernel` is a callable function, it is called on
    each pair of instances (rows) and the resulting value recorded. The
    callable should take two rows from X as input and return the
    corresponding kernel value as a single number. This means that
    callables from :mod:`sklearn.metrics.pairwise` are not allowed, as
    they operate on matrices, not single samples. Use the string
    identifying the kernel instead.
gamma : float, default=None
    Gamma parameter for the RBF, laplacian, polynomial, exponential chi2
    and sigmoid kernels. Interpretation of the default value is left to
    the kernel; see the documentation for sklearn.metrics.pairwise.
    Ignored by other kernels.
degree : float, default=3
    Degree of the polynomial kernel. Ignored by other kernels.
coef0 : float, default=1
    Zero coefficient for polynomial and sigmoid kernels.
    Ignored by other kernels.
kernel_params : mapping of str to any, default=None
    Additional parameters (keyword arguments) for kernel function passed
    as callable object.
"""


class Config:
    arbitrary_types_allowed = True
    extra = "forbid"


@dataclass(config=Config)
class KernelRidgeConfig(ModelConfig):
    alpha: Union[float, List[float]] = 1.0
    kernel: str = "linear"
    gamma: Optional[float] = None
    degree: int = 3
    coef0: float = 1.0
    kernel_params: Optional[Dict] = None


class KernelRidgeRegressor(SKLearnModelBase[KernelRidgeConfig]):
    @property
    def _config_builder(self) -> Type[KernelRidgeConfig]:
        return KernelRidgeConfig

    def _info(self) -> ModelInfo:
        return ModelInfo(
            model_description=_DESCRIPTION,
            config_description=_CONFIG_DESCRIPTION,
        )

    def _instantiate_model(self) -> KR:
        config = self.model_config
        return KR(
            alpha=config.alpha,
            kernel=config.kernel,
            gamma=config.gamma,
            degree=config.degree,
            coef0=config.coef0,
            kernel_params=config.kernel_params,
        )
