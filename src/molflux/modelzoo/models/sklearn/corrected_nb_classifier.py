from typing import Iterable, Optional, Type

from pydantic.dataclasses import dataclass

from molflux.modelzoo.info import ModelInfo
from molflux.modelzoo.model import ModelConfig
from molflux.modelzoo.models.sklearn import (
    SKLearnClassificationMixin,
    SKLearnModelBase,
)

try:
    from molflux.modelzoo.models.sklearn.sklearn_discrete_nb.bayes import (  # type: ignore
        CorrectedNB as SKCorrectedNB,
    )
except ImportError as e:
    from molflux.modelzoo.errors import ExtrasDependencyImportError

    raise ExtrasDependencyImportError("sklearn", e) from None


_DESCRIPTION = """
A Naive Bayes classifier with corrected probabilities
based on the paper https://pubs.acs.org/doi/10.1021/jm0303195.
"""

_CONFIG_DESCRIPTION = """
Parameters
----------
alpha : float or array-like of shape (n_features,), default=1.0
    Additive (Laplace/Lidstone) smoothing parameter
    (set alpha=0 for no smoothing).

fit_prior : bool, default=True
    Whether to learn class prior probabilities or not.
    If false, a uniform prior will be used.

class_prior : array-like of shape (n_classes,), default=None
    Prior probabilities of the classes. If specified, the priors are not
    adjusted according to the data.
"""


class Config:
    arbitrary_types_allowed = True
    extra = "forbid"


@dataclass(config=Config)
class CorrectedNBConfig(ModelConfig):
    alpha: float = 1
    fit_prior: bool = True
    class_prior: Optional[Iterable] = None


class CorrectedNBClassifier(
    SKLearnClassificationMixin,
    SKLearnModelBase[CorrectedNBConfig],
):
    @property
    def _config_builder(self) -> Type[CorrectedNBConfig]:
        return CorrectedNBConfig

    def _info(self) -> ModelInfo:
        return ModelInfo(
            model_description=_DESCRIPTION,
            config_description=_CONFIG_DESCRIPTION,
        )

    def _instantiate_model(self) -> SKCorrectedNB:
        config = self.model_config
        return SKCorrectedNB(
            alpha=config.alpha,
            fit_prior=config.fit_prior,
            class_prior=config.class_prior,
        )
