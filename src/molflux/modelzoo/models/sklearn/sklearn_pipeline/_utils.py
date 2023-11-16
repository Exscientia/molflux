"""
This file implements the functions mapping json-serializable configs of SKLearn classes to actual python instances.
"""

import sys
from importlib import import_module
from importlib.util import find_spec
from typing import Any, Dict, List, Optional

# https://docs.pydantic.dev/latest/usage/types/#typeddict
if sys.version_info < (3, 9, 2):
    from typing_extensions import TypedDict
else:
    from typing import TypedDict

try:
    from sklearn.pipeline import Pipeline, make_pipeline

except ImportError as e:
    from molflux.modelzoo.errors import ExtrasDependencyImportError

    raise ExtrasDependencyImportError("sklearn", e) from e


def _load_class(qualified_class: str) -> Any:
    """
    Returns a class from a string definition of a class.
    Args:
        qualified_class (str): String representation of the class path. For example,
        `sklearn.ensemble.RandomForest`.

    Returns:
        Class of the specified string definition.

    """
    *module_name_comps, class_name = qualified_class.split(".")

    # assert the class has a prefix module
    assert module_name_comps

    module_name = ".".join(module_name_comps)
    if find_spec(module_name) is None:
        raise ValueError(f"failed to find module - {module_name}")

    module = import_module(module_name)
    if not hasattr(module, class_name):
        raise ValueError(
            f"failed to find class = {class_name} in module = {module_name}",
        )

    ret = getattr(module, class_name)

    return ret


class StepConfig(TypedDict):
    """Type definition of the dictionary structure for a single pipeline step in an SKLearnPipeline config"""

    class_name: str
    hyperparameters: Optional[Dict[str, Any]]


# type definition of the config expected by SKLearnPipeline models
StepConfigsT = List[StepConfig]


def _get_step_object(step_config: StepConfig) -> Any:
    """
    Return an instance of a class from a given config.

    Args:
        step_config (StepConfig): Dictionary with 2 keys, "class_name" and "hyperparameters".
        This should follow the format:
        {
            "class_name": "sklearn.some_module.ClassName",
            "hyperparameters": {
                "first_hp_for_ClassName": 500,
                "second_hp_for_ClassName": "auto",
                "learning_rate": 0.1,
                ...
            },
        }

    Returns:
        An object equivalent to ClassName(**hyperparameters)

    """
    try:
        # get the class from the specified import path
        class_name = step_config.get("class_name")
        if class_name is None:
            raise ValueError("class_name parameter must be specified")
        model_class = _load_class(class_name)

        # get the hyperparameters for the class
        hyperparams = step_config.get("hyperparameters", {})
        if hyperparams is None:
            hyperparams = {}

        # instantiate an object of the class with the hyperparameters
        return model_class(**hyperparams)
    except Exception as e:
        raise RuntimeError(f"Could not generate step for {step_config=}") from e


def build_pipeline(step_configs: StepConfigsT) -> Pipeline:
    """
    Construct an SKLearn pipeline from a list of step configs.

    Args:
        step_configs (StepConfigsT): List of steps to be formatted into an SKLearn Pipeline.
        This should follow the format:
            [
                {
                    "class_name": "sklearn.some_module.ClassName",
                    "hyperparameters": {
                        "first_hp_for_ClassName": 500,
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

    Returns: Pipeline
    """
    step_objects = []

    # for each step in the pipeline config, instantiate the corresponding object
    for step_config in step_configs:
        step_object = _get_step_object(step_config)
        step_objects.append(step_object)

    # make the pipeline from each step
    return make_pipeline(*step_objects)
