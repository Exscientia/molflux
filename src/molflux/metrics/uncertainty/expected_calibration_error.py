"""Expected calibration error """

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
Expected calibration error
The expected calibration error (or ECE) for regression data measures the average difference
between the observed confidence level and the expected confidence level within each bin
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
        assess the calibration error 20. Defaults to 20.

Returns:
    expected_calibration_error: The expected calibration error (ECE) value.

Examples:
    >>> from molflux.metrics import load_metric
    >>> m = load_metric("expected_calibration_error")
    >>> ref = [0, 1, 2, 3, 4]
    >>> pred = [0.5, 0.8, 1.2, 2.4, 4.5]
    >>> lower_bound = [0.1, 0.1, 0.2, 2, 3]
    >>> upper_bound = [1, 1.5, 1.8, 2.8, 5.7]
    >>> prediction_intervals = list(zip(lower_bound, upper_bound))
    >>> m.compute(predictions=pred, references=ref, prediction_intervals=prediction_intervals)
    {'expected_calibration_error': 0.02675...}
"""

_CITATION = """\
Naeini, Mahdi Pakdaman, Gregory Cooper, and Milos Hauskrecht.
"Obtaining well calibrated probabilities using bayesian binning."
Twenty-Ninth AAAI Conference on Artificial Intelligence. 2015.
"""


@evaluate.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class ExpectedCalibrationError(PredictionIntervalMetric):
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
            reference_urls=["https://arxiv.org/pdf/2306.10060.pdf"],
        )

    def _score(
        self,
        *,
        predictions: ArrayLike,
        references: ArrayLike,
        standard_deviations: Optional[ArrayLike] = None,
        prediction_intervals: Optional[ArrayLike] = None,
        confidence: float = 0.9,
        num_thresholds: int = 20,
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
        rho_s = [s / num_thresholds for s in range(num_thresholds)]
        # empirical frequency of predicted values falling in the rho_s quantile
        p_s = [np.mean(pred_cdf < rho) for rho in rho_s]
        squared_differences = [(rho - p) ** 2 for rho, p in zip(rho_s, p_s)]
        score = np.mean(squared_differences)
        return {self.tag: score}
