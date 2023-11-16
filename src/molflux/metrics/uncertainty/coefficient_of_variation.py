"""Uncertainty measure: Computes coefficient of variation."""

import logging
from typing import Any, Optional

import evaluate
import numpy as np

import datasets
from molflux.metrics.bases import PredictionIntervalMetric
from molflux.metrics.typing import ArrayLike, MetricResult

logger = logging.getLogger(__name__)

_CITATION = """\
@article{busk2021calibrated,
  title={Calibrated uncertainty for molecular property prediction using ensembles of message passing neural networks},
  author={Busk, Jonas and J{\\o}rgensen, Peter Bj{\\o}rn and Bhowmik, Arghya and Schmidt, Mikkel N and Winther, Ole and Vegge, Tejs},
  journal={Machine Learning: Science and Technology},
  volume={3},
  number={1},
  pages={015012},
  year={2021},
  publisher={IOP Publishing}
}
"""

_DESCRIPTION = """\
Computes coefficient of variation for a prediction interval across a dataset.
"""

_KWARGS_DESCRIPTION = """
Args:
    predictions: Estimated targets as returned by a classifier/regressor.
    references: Ground truth (correct) target values.
    prediction_intervals: Prediction intervals of lower and upper bounds.

Returns:
    coefficient_of_variation: coefficient of variation (CV) indicating the heteroscedasticity or homoscedasticity of predictions across a dataset.
    High CV indicates large dispersion (high dependence of uncertainty estimates), whereas low CV indicates constant, uninformative uncertainty estimates.

Examples:
    >>> from molflux.metrics import load_metric
    >>> m = load_metric("coefficient_of_variation")
    >>> ref = [0, 1, 2, 3, 4]
    >>> pred = [0, 2, 2, 3, 5]
    >>> lower_bound = [0.1, 0.1, 0.2, 2, 3]
    >>> upper_bound = [1, 1.5, 1.8, 2.8, 5.7]
    >>> prediction_intervals = list(zip(lower_bound, upper_bound))
    >>> m.compute(references=ref, predictions=pred, prediction_intervals=prediction_intervals)
    {'coefficient_of_variation': 0.459...}
"""


@evaluate.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class CoefficientOfVariation(PredictionIntervalMetric):
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
        prediction_intervals: Optional[ArrayLike] = None,
        **kwargs: Any,
    ) -> MetricResult:
        if prediction_intervals is None:
            raise ValueError(
                "Please provide prediction intervals in the form of lower and upper bounds.",
            )
        references = np.array(references)
        lower_bound, upper_bound = zip(*prediction_intervals)
        lower_bound = np.array(lower_bound)  # type: ignore[assignment]
        upper_bound = np.array(upper_bound)  # type: ignore[assignment]
        if (upper_bound < lower_bound).any():  # type: ignore[attr-defined]
            raise ValueError("Please ensure upper bound is greater than lower bound.")
        # Definition in equation 10 of reference in _CITATION relates to standard deviation, sigma.
        # Making a Gaussian assumption, then sigma is related to the prediction interval width by rescaling by a factor of scipy.norm.ppf(1-alpha/2).
        # However, this rescaling factor is in both numerator and denominator of equation 10, so drops out.
        # We can therefore work with the prediction interval width directly, which should generalise better to interval predictions that don't rely on Gaussian assumptions
        sigma = upper_bound - lower_bound  # type: ignore[operator]
        sigma_bar = np.mean(sigma)  # average prediction interval width
        if sigma_bar == 0:
            score = 0  # avoid division by zero in case of interval of width 0
        else:
            score = np.sqrt(np.mean((sigma - sigma_bar) ** 2)) / sigma_bar
        return {self.tag: score}
