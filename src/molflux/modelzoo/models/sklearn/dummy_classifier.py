from typing import List, Literal, Optional, Type, Union

from pydantic.dataclasses import dataclass

from molflux.modelzoo.info import ModelInfo
from molflux.modelzoo.model import ModelConfig
from molflux.modelzoo.models.sklearn import (
    SKLearnClassificationMixin,
    SKLearnModelBase,
)

try:
    from sklearn.dummy import DummyClassifier as SKDummyClassifier
except ImportError as e:
    from molflux.modelzoo.errors import ExtrasDependencyImportError

    raise ExtrasDependencyImportError("sklearn", e) from None


_DESCRIPTION = """
This is an sklearn dummy classifier model.

This classifier serves as a simple baseline to compare against other more complex classifiers. The specific behavior of
the baseline is selected with the strategy parameter. All strategies make predictions that ignore the input x feature
values provided at training and prediction time. The predictions, however, typically depend on values observed in the
y features provided during training.

Note that the “stratified” and “uniform” strategies lead to non-deterministic predictions that can be rendered
deterministic by setting the random_state parameter if needed. The other strategies are naturally deterministic and,
once trained, always return the same constant prediction for any value of input x features.
"""

_CONFIG_DESCRIPTION = """
Parameters
----------
strategy : {"most_frequent", "prior", "stratified", "uniform", "constant"}, default="prior"
    Strategy to use to generate predictions.

    * "most_frequent": the `predict` method always returns the most
      frequent class label in the observed y features passed during training.
      The `predict_proba` method returns the matching one-hot encoded
      vector.
    * "prior": the `predict` method always returns the most frequent
      class label in the observed y features passed during training (like
      "most_frequent"). ``predict_proba`` always returns the empirical
      class distribution of `y` also known as the empirical class prior
      distribution.
    * "stratified": the `predict_proba` method randomly samples one-hot
      vectors from a multinomial distribution parametrized by the empirical
      class prior probabilities.
      The `predict` method returns the class label which got probability
      one in the one-hot vector of `predict_proba`.
      Each sampled row of both methods is therefore independent and
      identically distributed.
    * "uniform": generates predictions uniformly at random from the list
      of unique classes observed in `y`, i.e. each class has equal
      probability.
    * "constant": always predicts a constant label that is provided by
      the user. This is useful for metrics that evaluate a non-majority
      class.

random_state : int or None, default=None
    Controls the randomness to generate the predictions when
    ``strategy='stratified'`` or ``strategy='uniform'``.
    Pass an int for reproducible output across multiple function calls.

constant : int or str or array-like of shape (n_outputs,), default=None
    The explicit constant as predicted by the "constant" strategy. This
    parameter is useful only for the "constant" strategy.
"""

PredictionsStrategy = Literal[
    "most_frequent",
    "prior",
    "stratified",
    "uniform",
    "constant",
]


class Config:
    extra = "forbid"
    arbitrary_types_allowed = True
    smart_union = True


@dataclass(config=Config)
class DummyClassifierConfig(ModelConfig):
    strategy: PredictionsStrategy = "prior"
    random_state: Optional[int] = None
    constant: Union[int, str, List[Union[int, str]], None] = None


class DummyClassifier(
    SKLearnClassificationMixin,
    SKLearnModelBase[DummyClassifierConfig],
):
    @property
    def _config_builder(self) -> Type[DummyClassifierConfig]:
        return DummyClassifierConfig

    def _info(self) -> ModelInfo:
        return ModelInfo(
            model_description=_DESCRIPTION,
            config_description=_CONFIG_DESCRIPTION,
        )

    def _instantiate_model(self) -> SKDummyClassifier:
        config = self.model_config
        return SKDummyClassifier(
            strategy=config.strategy,
            random_state=config.random_state,
            constant=config.constant,
        )
