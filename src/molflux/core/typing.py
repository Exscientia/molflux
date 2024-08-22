from os import PathLike as OSPathLike
from typing import Any, Literal, Union

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
DatasetType = Union[datasets.Dataset, datasets.DatasetDict]
TasksScores = dict[str, dict[str, Any]]
FoldScores = dict[str, TasksScores]
