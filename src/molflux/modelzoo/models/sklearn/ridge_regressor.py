from typing import List, Literal, Optional, Type, Union

import numpy.random
from pydantic.dataclasses import dataclass

from molflux.modelzoo.info import ModelInfo
from molflux.modelzoo.model import ModelConfig
from molflux.modelzoo.models.sklearn import SKLearnModelBase

try:
    from sklearn.linear_model import Ridge as SKLearnRidge
except ImportError as e:
    from molflux.modelzoo.errors import ExtrasDependencyImportError

    raise ExtrasDependencyImportError("sklearn", e) from e


_DESCRIPTION = """
This is an sklearn linear least squares regression model with l2 regularization.

Minimizes the objective function::

||y - Xw||^2_2 + alpha * ||w||^2_2

This model solves a regression model where the loss function is
the linear least squares function and regularization is given by
the l2-norm. Also known as Ridge Regression or Tikhonov regularization.
This estimator has built-in support for multi-variate regression
(i.e., when y is a 2d-array of shape (n_samples, n_targets)).
"""

_CONFIG_DESCRIPTION = """
Parameters
----------
alpha : {float, ndarray of shape (n_targets,)}, default=1.0
    Constant that multiplies the L2 term, controlling regularization
    strength. `alpha` must be a non-negative float i.e. in `[0, inf)`.
    When `alpha = 0`, the objective is equivalent to ordinary least
    squares, solved by the :class:`LinearRegression` object. For numerical
    reasons, using `alpha = 0` with the `Ridge` object is not advised.
    Instead, you should use the :class:`LinearRegression` object.
    If an array is passed, penalties are assumed to be specific to the
    targets. Hence they must correspond in number.
fit_intercept : bool, default=True
    Whether to fit the intercept for this model. If set
    to false, no intercept will be used in calculations
    (i.e. ``X`` and ``y`` are expected to be centered).
copy_X : bool, default=True
    If True, X will be copied; else, it may be overwritten.
max_iter : int, default=None
    Maximum number of iterations for conjugate gradient solver.
    For 'sparse_cg' and 'lsqr' solvers, the default value is determined
    by scipy.sparse.linalg. For 'sag' solver, the default value is 1000.
    For 'lbfgs' solver, the default value is 15000.
tol : float, default=1e-3
    Precision of the solution.
solver : {'auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', \
        'sag', 'saga', 'lbfgs'}, default='auto'
    Solver to use in the computational routines:
    - 'auto' chooses the solver automatically based on the type of data.
    - 'svd' uses a Singular Value Decomposition of X to compute the Ridge
      coefficients. It is the most stable solver, in particular more stable
      for singular matrices than 'cholesky' at the cost of being slower.
    - 'cholesky' uses the standard scipy.linalg.solve function to
      obtain a closed-form solution.
    - 'sparse_cg' uses the conjugate gradient solver as found in
      scipy.sparse.linalg.cg. As an iterative algorithm, this solver is
      more appropriate than 'cholesky' for large-scale data
      (possibility to set `tol` and `max_iter`).
    - 'lsqr' uses the dedicated regularized least-squares routine
      scipy.sparse.linalg.lsqr. It is the fastest and uses an iterative
      procedure.
    - 'sag' uses a Stochastic Average Gradient descent, and 'saga' uses
      its improved, unbiased version named SAGA. Both methods also use an
      iterative procedure, and are often faster than other solvers when
      both n_samples and n_features are large. Note that 'sag' and
      'saga' fast convergence is only guaranteed on features with
      approximately the same scale. You can preprocess the data with a
      scaler from sklearn.preprocessing.
    - 'lbfgs' uses L-BFGS-B algorithm implemented in
      `scipy.optimize.minimize`. It can be used only when `positive`
      is True.
    All solvers except 'svd' support both dense and sparse data. However, only
    'lsqr', 'sag', 'sparse_cg', and 'lbfgs' support sparse input when
    `fit_intercept` is True.
positive : bool, default=False
    When set to ``True``, forces the coefficients to be positive.
    Only 'lbfgs' solver is supported in this case.
random_state : int, RandomState instance, default=None
    Used when ``solver`` == 'sag' or 'saga' to shuffle the data.
"""

RidgeRegressorSolver = Literal[
    "auto",
    "svd",
    "cholesky",
    "lsqr",
    "sparse_cg",
    "sag",
    "saga",
    "lbfgs",
]


class Config:
    arbitrary_types_allowed = True
    extra = "forbid"


@dataclass(config=Config)
class RidgeRegressorConfig(ModelConfig):
    alpha: Union[float, List[float]] = 1.0
    fit_intercept: bool = True
    copy_X: bool = True
    max_iter: Optional[int] = None
    tol: float = 1e-3
    solver: RidgeRegressorSolver = "auto"
    positive: bool = False
    random_state: Union[None, int, numpy.random.RandomState] = None


class RidgeRegressor(SKLearnModelBase[RidgeRegressorConfig]):
    @property
    def _config_builder(self) -> Type[RidgeRegressorConfig]:
        return RidgeRegressorConfig

    def _info(self) -> ModelInfo:
        return ModelInfo(
            model_description=_DESCRIPTION,
            config_description=_CONFIG_DESCRIPTION,
        )

    def _instantiate_model(self) -> SKLearnRidge:
        config = self.model_config
        return SKLearnRidge(
            alpha=config.alpha,
            fit_intercept=config.fit_intercept,
            copy_X=config.copy_X,
            max_iter=config.max_iter,
            tol=config.tol,
            solver=config.solver,
            positive=config.positive,
            random_state=config.random_state,
        )
