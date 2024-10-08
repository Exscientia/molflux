"""Prediction interval width for models with interval predictions"""

import logging
from typing import Any

import evaluate
import numpy as np
import scipy

import datasets
from molflux.metrics.bases import UncertaintyMetric
from molflux.metrics.typing import ArrayLike, MetricResult

logger = logging.getLogger(__name__)

_DESCRIPTION = """
Prediction Interval Width function.

Predictions for models with uncertainty may provide lower and upper bounds for
predictions at a certain confidence level. Assessing the width of these intervals
across a dataset can help assess how sharp these predictions are.
"""

_KWARGS_DESCRIPTION = """
Args:
    predictions: array of mean predictions.
    references: Ground truth (correct) target values.
    lower_bound: array of lower bounds of predictions.
    upper_bound: array of upper bounds of predictions.

Returns:
    prediction_interval_width: The value of the prediction interval width.

Examples:
    >>> from molflux.metrics import load_metric
    >>> m = load_metric("prediction_interval_width")
    >>> ref = [0, 1, 2, 3, 4]
    >>> pred = [0, 2, 2, 3, 5]
    >>> lower_bound = [0.1, 0.1, 0.2, 2, 3]
    >>> upper_bound = [1, 1.5, 1.8, 2.8, 5.7]
    >>> prediction_intervals = list(zip(lower_bound, upper_bound))
    >>> m.compute(predictions=pred, references=ref, prediction_intervals=prediction_intervals)
    {'prediction_interval_width': 1.48}
"""

_CITATION = """\
None
"""


@evaluate.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class PredictionIntervalWidth(UncertaintyMetric):
    def _info(self) -> evaluate.MetricInfo:
        return evaluate.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            features=datasets.Features(
                {
                    "predictions": datasets.Value("float64"),
                    "references": datasets.Value("float64"),
                },
            ),
        )

    def _score(
        self,
        *,
        predictions: ArrayLike,
        references: ArrayLike,
        standard_deviations: ArrayLike | None = None,
        prediction_intervals: ArrayLike | None = None,
        confidence: float = 0.9,
        **kwargs: Any,
    ) -> MetricResult:
        if standard_deviations is not None and prediction_intervals is not None:
            raise ValueError(
                "Both standard deviations and prediction intervals given. Please provide only one.",
            )

        if standard_deviations is not None:
            lower_bound, upper_bound = scipy.stats.norm.interval(
                confidence,
                loc=predictions,
                scale=standard_deviations,
            )
        elif prediction_intervals is not None:
            lower_bound, upper_bound = zip(*prediction_intervals, strict=False)
        else:
            raise ValueError(
                "Either standard_deviations or prediction_intervals must be supplied.",
            )

        # ensure the predictions are a numpy array
        lower_bound = np.array(lower_bound)
        upper_bound = np.array(upper_bound)
        if (upper_bound < lower_bound).any():
            raise ValueError("Please ensure upper bound is greater than lower bound.")
        width = np.mean(upper_bound - lower_bound)

        return {self.tag: width}
