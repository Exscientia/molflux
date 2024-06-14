"""Calibration gap."""

import logging
from typing import Any, Optional

import evaluate
import numpy as np
from scipy.stats import norm

import datasets
from molflux.metrics.bases import PredictionIntervalMetric
from molflux.metrics.typing import ArrayLike, MetricResult
from molflux.metrics.uncertainty.utils import _estimate_standard_deviation

logger = logging.getLogger(__name__)


_DESCRIPTION = """
The calibration gap is the unsigned area between a models calibration curve and a perfectly calibrated model (y=x).
"""

_KWARGS_DESCRIPTION = """
Args:
    predictions: array of mean predictions.
    references: Ground truth (correct) target values.
    standard_deviations: Standard deviation predicted from a model with uncertainty.
        Either standard deviations or prediction_intervals should be provided.
    prediction_intervals: Prediction intervals of lower and upper bounds.
        Either standard deviations or prediction_intervals should be provided.
    confidence (optional): float indicating what proportion of data lies between
        lower and upper bounds. Defaults to 0.9.
    num_thresholds (optional): int giving the number of thresholds at which to
        assess the calibration error. Defaults to 1000.

Returns:
    calibration_gap: unsigned area between the calibration curve and y=x.

Examples:
    >>> from molflux.metrics import load_metric
    >>> m = load_metric("calibration_gap")
    >>> ref = [0, 1, 2, 3, 4]
    >>> pred = [0.5, 0.8, 1.2, 2.4, 4.5]
    >>> lower_bound = [0.1, 0.1, 0.2, 2, 3]
    >>> upper_bound = [1, 1.5, 1.8, 2.8, 5.7]
    >>> prediction_intervals = list(zip(lower_bound, upper_bound))
    >>> m.compute(predictions=pred, references=ref, prediction_intervals=prediction_intervals)
    {'calibration_gap': 0.1314763799612285}
"""

_CITATION = """
@inproceedings{kuleshov2018accurate,
  title={Accurate uncertainties for deep learning using calibrated regression},
  author={Kuleshov, Volodymyr and Fenner, Nathan and Ermon, Stefano},
  booktitle={International conference on machine learning},
  pages={2796--2804},
  year={2018},
  organization={PMLR}
}
"""


@evaluate.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class CalibrationGap(PredictionIntervalMetric):
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
        standard_deviations: Optional[ArrayLike] = None,
        prediction_intervals: Optional[ArrayLike] = None,
        confidence: float = 0.9,
        num_thresholds: int = 100,
        **kwargs: Any,
    ) -> MetricResult:
        if standard_deviations is None:
            if prediction_intervals is None:
                raise ValueError(
                    "Please provide either standard deviation, or prediction intervals",
                )
            lower_bound, upper_bound = zip(*prediction_intervals)
            standard_deviations = _estimate_standard_deviation(
                lower_bound,
                upper_bound,
                confidence,
            )

        assert len(predictions) == len(references)
        pred_cdf = norm.cdf(references, loc=predictions, scale=standard_deviations)
        # expected quantile
        rho_s = np.linspace(0, 1, num_thresholds)
        # empirical frequency of predicted values falling in the rho_s quantile
        p_s = np.array([np.mean(pred_cdf < rho) for rho in rho_s])
        area = np.trapz(np.abs(rho_s - p_s), x=rho_s)
        return {self.tag: area}
