from os import PathLike as OSPathLike
from typing import Any, Dict, Literal, Union

import datasets

AllowedInferenceMethods = Literal[
    "predict",
    "predict_proba",
    "predict_with_prediction_interval",
    "predict_with_std",
    "sample",
]
PathLike = Union[str, OSPathLike]
Dataset = datasets.Dataset
TasksScores = Dict[str, Dict[str, Any]]
FoldScores = Dict[str, TasksScores]
