from typing import Type

from pydantic.dataclasses import dataclass

from molflux.modelzoo.info import ModelInfo
from molflux.modelzoo.model import ModelConfig
from molflux.modelzoo.models.sklearn import SKLearnModelBase

try:
    from sklearn.cross_decomposition import PLSRegression as SKPLSRegression
except ImportError as e:
    from molflux.modelzoo.errors import ExtrasDependencyImportError

    raise ExtrasDependencyImportError("sklearn", e) from None


_DESCRIPTION = """
Partial Least Squares regressor.

PLSRegression is also known as PLS1 (single targets) and PLS2 (multiple targets).

Much like Lasso, PLSRegression is a form of regularized linear regression where
the number of components controls the strength of the regularization.
"""

_CONFIG_DESCRIPTION = """
Parameters
----------
n_components : int, default=2
    Number of components to keep. Should be in `[1, min(n_samples,
    n_features, n_targets)]`.
scale : bool, default=True
    Whether to scale the input data.
max_iter : int, default=500
    The maximum number of iterations of the power method.
tol : float, default=1e-06
    The tolerance used as convergence criteria in the power method: the
    algorithm stops whenever the squared norm of `u_i - u_{i-1}` is less
    than `tol`, where `u` corresponds to the left singular vector.
inplace : bool, default=False
    Whether to copy the training data before applying centering,
    and potentially scaling. If `True`, these operations will be done
    inplace, modifying both arrays.
"""


class Config:
    arbitrary_types_allowed = True
    extra = "forbid"


@dataclass(config=Config)
class PLSRegressorConfig(ModelConfig):
    n_components: int = 2
    scale: bool = True
    max_iter: int = 500
    tol: float = 1e-06
    inplace: bool = False


class PLSRegressor(SKLearnModelBase[PLSRegressorConfig]):
    @property
    def _config_builder(self) -> Type[PLSRegressorConfig]:
        return PLSRegressorConfig

    def _info(self) -> ModelInfo:
        return ModelInfo(
            model_description=_DESCRIPTION,
            config_description=_CONFIG_DESCRIPTION,
        )

    def _instantiate_model(self) -> SKPLSRegression:
        config = self.model_config
        return SKPLSRegression(
            n_components=config.n_components,
            scale=config.scale,
            max_iter=config.max_iter,
            tol=config.tol,
            copy=not config.inplace,
        )
