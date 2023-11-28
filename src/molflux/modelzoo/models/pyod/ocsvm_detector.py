from dataclasses import asdict
from typing import Any, Callable, Dict, Literal, Type, Union

from pydantic.dataclasses import dataclass

from molflux.modelzoo.info import ModelInfo
from molflux.modelzoo.models.pyod import (
    PyODClassificationMixin,
    PyODModelBase,
    PyODModelConfig,
)

try:
    from pyod.models.ocsvm import OCSVM
except ImportError as e:
    from molflux.modelzoo.errors import ExtrasDependencyImportError

    raise ExtrasDependencyImportError("pyod", e) from None


_DESCRIPTION = """
Wrapper of scikit-learn one-class SVM Class with more functionalities.
Unsupervised Outlier Detection.

Estimate the support of a high-dimensional distribution.

The implementation is based on libsvm.
See http://scikit-learn.org/stable/modules/svm.html#svm-outlier-detection
and :cite:`scholkopf2001estimating`.
"""

_CONFIG_DESCRIPTION = """
Parameters
----------
kernel : string, optional (default='rbf')
     Specifies the kernel type to be used in the algorithm.
     It must be one of 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed' or
     a callable.
     If none is given, 'rbf' will be used. If a callable is given it is
     used to precompute the kernel matrix.

nu : float, optional
    An upper bound on the fraction of training
    errors and a lower bound of the fraction of support
    vectors. Should be in the interval (0, 1]. By default 0.5
    will be taken.

degree : int, optional (default=3)
    Degree of the polynomial kernel function ('poly').
    Ignored by all other kernels.

gamma : float, optional (default='auto')
    Kernel coefficient for 'rbf', 'poly' and 'sigmoid'.
    If gamma is 'auto' then 1/n_features will be used instead.

coef0 : float, optional (default=0.0)
    Independent term in kernel function.
    It is only significant in 'poly' and 'sigmoid'.

tol : float, optional
    Tolerance for stopping criterion.

shrinking : bool, optional
    Whether to use the shrinking heuristic.

cache_size : float, optional
    Specify the size of the kernel cache (in MB).

verbose : bool, default: False
    Enable verbose output. Note that this setting takes advantage of a
    per-process runtime setting in libsvm that, if enabled, may not work
    properly in a multithreaded context.

max_iter : int, optional (default=-1)
    Hard limit on iterations within solver, or -1 for no limit.

contamination : float in (0., 0.5), optional (default=0.1)
    The amount of contamination of the data set, i.e.
    the proportion of outliers in the data set. Used when fitting to
    define the threshold on the decision function.
"""

Auto = Literal["auto"]
Kernel = Literal["linear", "poly", "rbf", "sigmoid", "precomputed"]


class Config:
    arbitrary_types_allowed = True
    extra = "forbid"


@dataclass(config=Config)
class OCSVMDetectorConfig(PyODModelConfig):
    kernel: Union[Kernel, Callable] = "rbf"
    nu: float = 0.5
    degree: int = 3
    gamma: Union[float, Auto] = "auto"
    coef0: float = 0
    tol: float = 1e-3
    shrinking: bool = True
    cache_size: float = 200
    verbose: bool = False
    max_iter: int = -1
    contamination: float = 0.1


class OCSVMDetector(PyODClassificationMixin, PyODModelBase[OCSVMDetectorConfig]):
    @property
    def config(self) -> Dict[str, Any]:
        return asdict(self.model_config)

    @property
    def _config_builder(self) -> Type[OCSVMDetectorConfig]:
        return OCSVMDetectorConfig

    def _info(self) -> ModelInfo:
        return ModelInfo(
            model_description=_DESCRIPTION,
            config_description=_CONFIG_DESCRIPTION,
        )

    def _instantiate_model(self) -> OCSVM:
        config = self.model_config
        return OCSVM(
            kernel=config.kernel,
            nu=config.nu,
            degree=config.degree,
            gamma=config.gamma,
            coef0=config.coef0,
            tol=config.tol,
            shrinking=config.shrinking,
            cache_size=config.cache_size,
            verbose=config.verbose,
            max_iter=config.max_iter,
            contamination=config.contamination,
        )
