from typing import Iterable, Optional, Type

from pydantic.dataclasses import dataclass

from molflux.modelzoo.info import ModelInfo
from molflux.modelzoo.model import ModelConfig
from molflux.modelzoo.models.sklearn import (
    SKLearnClassificationMixin,
    SKLearnModelBase,
)

try:
    from sklearn.naive_bayes import BernoulliNB as SKBernoulliNB
except ImportError as e:
    from molflux.modelzoo.errors import ExtrasDependencyImportError

    raise ExtrasDependencyImportError("sklearn", e) from None


_DESCRIPTION = """
This is an sklearn naive bayes classifier.

BernoulliNB implements the naive Bayes training and classification algorithms for data
that is distributed according to multivariate Bernoulli distributions; i.e., there may
be multiple features but each one is assumed to be a binary-valued (Bernoulli, boolean)
variable. Therefore, this class requires samples to be represented as binary-valued
feature vectors; if handed any other kind of data, a BernoulliNB instance may
binarize its input (depending on the binarize parameter).
"""

_CONFIG_DESCRIPTION = """
Parameters
----------
alpha : float, default=1.0
    Additive (Laplace/Lidstone) smoothing parameter
    (0 for no smoothing).
binarize : float or None, default=0.0
    Threshold for binarizing (mapping to booleans) of sample features.
    If None, input is presumed to already consist of binary vectors.
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
class BernoulliNBConfig(ModelConfig):
    alpha: float = 1
    binarize: Optional[float] = 0
    fit_prior: bool = True
    class_prior: Optional[Iterable] = None


class BernoulliNBClassifier(
    SKLearnClassificationMixin,
    SKLearnModelBase[BernoulliNBConfig],
):
    @property
    def _config_builder(self) -> Type[BernoulliNBConfig]:
        return BernoulliNBConfig

    def _info(self) -> ModelInfo:
        return ModelInfo(
            model_description=_DESCRIPTION,
            config_description=_CONFIG_DESCRIPTION,
        )

    def _instantiate_model(self) -> SKBernoulliNB:
        config = self.model_config
        return SKBernoulliNB(
            alpha=config.alpha,
            binarize=config.binarize,
            fit_prior=config.fit_prior,
            class_prior=config.class_prior,
        )
