"""Gaussain NLL (Gaussian Negative Log Likelihood) regression score function."""

import logging
from typing import Any, Optional

import evaluate
import numpy as np

import datasets
from molflux.metrics.bases import PredictionIntervalMetric
from molflux.metrics.typing import ArrayLike, MetricResult
from molflux.metrics.uncertainty.utils import _estimate_standard_deviation

logger = logging.getLogger(__name__)


_DESCRIPTION = """
Gaussian Negative Log Likelihood function.

The predictions are treated as samples from Gaussian distributions with a given
expectation and variance. The probability density function of the Gaussian is computed
at the reference point. A minus log is applied, and values for multiple data points
are added together.

"""

_KWARGS_DESCRIPTION = """
Args:
    predictions: array of mean predictions.
    references: Ground truth (correct) target values.
    prediction_intervals: Prediction intervals of lower and upper bounds.
    confidence (optional): float indicating what proportion of data lies between
        lower and upper bounds. Defaults to 0.9.
    eps (optional): Minimum value for standard deviations to avoid numeric errors.
        Defaults to 1e-6.

Returns:
    gaussian_nll: The Gaussian NLL value.

Examples:
    >>> from molflux.metrics import load_metric
    >>> m = load_metric("gaussian_nll")
    >>> ref = [0, 1, 2, 3, 4]
    >>> pred = [0.5, 0.8, 1.2, 2.4, 4.5]
    >>> lower_bound = [0.1, 0.1, 0.2, 2, 3]
    >>> upper_bound = [1, 1.5, 1.8, 2.8, 5.7]
    >>> prediction_intervals = list(zip(lower_bound, upper_bound))
    >>> m.compute(predictions=pred, references=ref, prediction_intervals=prediction_intervals)
    {'gaussian_nll': 1.2949031482515838}
"""

_CITATION = """\
None
"""


@evaluate.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class GaussianNLL(PredictionIntervalMetric):
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
            reference_urls=["https://en.wikipedia.org/wiki/Likelihood_function"],
        )

    def _score(
        self,
        *,
        predictions: ArrayLike,
        references: ArrayLike,
        prediction_intervals: Optional[ArrayLike] = None,
        confidence: float = 0.9,
        eps: float = 1e-6,
        **kwargs: Any,
    ) -> MetricResult:
        if prediction_intervals is None:
            raise ValueError(
                "Please provide prediction intervals in the form of lower and upper bounds.",
            )
        mu = np.array(predictions)
        lower_bound, upper_bound = zip(*prediction_intervals)
        lower_bound = np.array(lower_bound)  # type: ignore[assignment]
        upper_bound = np.array(upper_bound)  # type: ignore[assignment]
        if (upper_bound < lower_bound).any():  # type: ignore[attr-defined]
            raise ValueError("Please ensure upper bound is greater than lower bound.")

        sigma = _estimate_standard_deviation(lower_bound, upper_bound, confidence)
        sigma = np.clip(sigma, a_min=eps, a_max=None)

        gaussian_log_likelihood = (
            1
            / 2
            * np.mean(
                np.log(2 * np.pi * (sigma**2))
                + ((references - mu) ** 2) / (sigma**2),
            )
        )

        return {self.tag: gaussian_log_likelihood}
