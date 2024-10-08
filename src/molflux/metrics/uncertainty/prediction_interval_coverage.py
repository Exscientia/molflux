"""Uncertainty measure: Computes coverage of a prediction interval."""

import logging
from typing import Any

import evaluate
import numpy as np
import scipy

import datasets
from molflux.metrics.bases import UncertaintyMetric
from molflux.metrics.typing import ArrayLike, MetricResult

logger = logging.getLogger(__name__)

_CITATION = """\
@article{barber2021predictive,
  title={Predictive inference with the jackknife+},
  author={Barber, Rina Foygel and Candes, Emmanuel J and Ramdas, Aaditya and Tibshirani, Ryan J},
  journal={The Annals of Statistics},
  volume={49},
  number={1},
  pages={486--507},
  year={2021},
  publisher={Institute of Mathematical Statistics}
}
"""

_DESCRIPTION = """\
Computes coverage of a prediction interval.
This is obtained by estimating the fraction of true labels that lie within the prediction intervals.
"""

_KWARGS_DESCRIPTION = """
Args:
    predictions: Estimated targets as returned by a classifier/regressor.
    references: Ground truth (correct) target values.
    prediction_intervals: Prediction intervals of lower and upper bounds.

Returns:
    prediction_interval_coverage: the coverage indicating the fraction of ground truth values that lie in the prediction interval

Examples:
    >>> from molflux.metrics import load_metric
    >>> m = load_metric("prediction_interval_coverage")
    >>> ref = [0, 1, 2, 3, 4]
    >>> pred = [0, 2, 2, 3, 5]
    >>> lower_bound = [0.1, 0.1, 0.2, 2, 3]
    >>> upper_bound = [1, 1.5, 1.8, 2.8, 5.7]
    >>> prediction_intervals = list(zip(lower_bound, upper_bound))
    >>> m.compute(references=ref, predictions=pred, prediction_intervals=prediction_intervals)
    {'prediction_interval_coverage': 0.4}
"""


@evaluate.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class PredictionIntervalCoverage(UncertaintyMetric):
    def _info(self) -> evaluate.MetricInfo:
        return evaluate.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            features=datasets.Features(
                {
                    "predictions": datasets.Value("float"),
                    "references": datasets.Value("float"),
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

        lower_bound, upper_bound = np.array(lower_bound), np.array(upper_bound)
        score = np.mean((lower_bound <= references) & (upper_bound >= references))
        return {self.tag: float(score)}
