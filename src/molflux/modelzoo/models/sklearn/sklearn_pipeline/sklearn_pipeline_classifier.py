from dataclasses import field
from typing import Type

from pydantic.dataclasses import dataclass

from molflux.modelzoo.info import ModelInfo
from molflux.modelzoo.model import ModelConfig
from molflux.modelzoo.models.sklearn import (
    SKLearnClassificationMixin,
    SKLearnModelBase,
)
from molflux.modelzoo.models.sklearn.sklearn_pipeline._utils import (
    StepConfigsT,
    build_pipeline,
)

try:
    from sklearn.base import is_classifier
    from sklearn.pipeline import Pipeline

except ImportError as e:
    from molflux.modelzoo.errors import ExtrasDependencyImportError

    raise ExtrasDependencyImportError("sklearn", e) from e


_DESCRIPTION = """
This is an implementation of an SKLearn Pipeline Classifier. It takes in a list of dictionaries,
each defining a step in the pipeline. This has to conform to the official user guide referenced below. The last
step must be an estimator classifier.

By default, this returns a Pipeline with a StandardScaler before an SKLearn Random Forest Classifier.

Reference: https://scikit-learn.org/stable/modules/compose.html#pipeline
"""

_CONFIG_DESCRIPTION = """
Parameters
----------
step_configs: Optional[list] = None
    List defining each step of an SKLearn Pipeline. The format of the list must follow
    this format:
    [
        {
            "class_name": "sklearn.some_module.ClassName",
            "hyperparameters": {
                "first_hyperparam_for_ClassName": 500,
                "second_hp_for_ClassName": "auto",
                "learning_rate": 0.1,
                ...
            },
        },
        {
            "class_name": ...
            "hyperparameters": ...
        },
        ...
    ]

    It is a list of dictionaries, with a class name and hyperparameters for each step. The last one
    is expected to be a classifier.
"""


DEFAULT_CONFIG: StepConfigsT = [
    {
        "class_name": "sklearn.preprocessing.StandardScaler",
        "hyperparameters": None,
    },
    {
        "class_name": "sklearn.ensemble.RandomForestClassifier",
        "hyperparameters": {
            "n_estimators": 500,
            "random_state": 0,
        },
    },
]


class Config:
    arbitrary_types_allowed = True
    extra = "forbid"


@dataclass(config=Config)
class SklearnPipelineClassifierConfig(ModelConfig):
    step_configs: StepConfigsT = field(
        default_factory=lambda: DEFAULT_CONFIG.copy(),
    )


class SklearnPipelineClassifier(
    SKLearnClassificationMixin,
    SKLearnModelBase[SklearnPipelineClassifierConfig],
):
    @property
    def _config_builder(self) -> Type[SklearnPipelineClassifierConfig]:
        return SklearnPipelineClassifierConfig

    def _info(self) -> ModelInfo:
        return ModelInfo(
            model_description=_DESCRIPTION,
            config_description=_CONFIG_DESCRIPTION,
        )

    def _instantiate_model(self) -> Pipeline:
        config = self.model_config
        pipeline = build_pipeline(config.step_configs)

        # assert that the last estimator of this pipeline is a classifier
        pipeline_estimator = pipeline[-1]

        if not is_classifier(pipeline_estimator):
            raise RuntimeError(
                f"The last step of the pipeline is expected to be a classifier. Found instead "
                f"{pipeline_estimator}",
            )

        return pipeline
