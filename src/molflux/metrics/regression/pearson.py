"""Pearson correlation coefficient score function."""

import logging
from typing import Any, Literal

import evaluate
import scipy.stats

import datasets
from molflux.metrics.bases import HFMetric
from molflux.metrics.typing import ArrayLike, MetricResult

logger = logging.getLogger(__name__)

_DESCRIPTION = """
Pearson correlation coefficient and p-value for testing non-correlation.

The Pearson correlation coefficient [1] measures the linear relationship between two datasets.
Like other correlation coefficients, this one varies between -1 and +1 with 0 implying no correlation.
Correlations of -1 or +1 imply an exact linear relationship. Positive correlations imply that as
x increases, so does y. Negative correlations imply that as x increases, y decreases.

This function also performs a test of the null hypothesis that the distributions underlying the samples
are uncorrelated and normally distributed. (See Kowalski [3] for a discussion of the effects of non-normality
of the input on the distribution of the correlation coefficient.) The p-value roughly indicates the probability
of an uncorrelated system producing datasets that have a Pearson correlation at least as extreme as the one
computed from these datasets.
"""

_KWARGS_DESCRIPTION = """
Args:
    predictions: Estimated target values.
    references: Ground truth (correct) target values.
    alternative (optional): Defines the alternative hypothesis. Default is 'two-sided'.
        The following options are available:

        * 'two-sided': the correlation is nonzero
        * 'less': the correlation is negative (less than zero)
        * 'greater':  the correlation is positive (greater than zero)

Returns:
    The Pearson correlation coefficient.
    The p-value associated with the chosen alternative.

Examples:
    >>> from molflux.metrics import load_metric
    >>> metric = load_metric("pearson")
    >>> predictions = [1, 2, 3, 4, 5]
    >>> references = [5, 6, 7, 8, 7]
    >>> metric.compute(predictions=predictions, references=references)
    {'pearson::correlation': 0.83205..., 'pearson::p_value': 0.0805...}
"""

_CITATION = """
“Pearson correlation coefficient”, Wikipedia, https://en.wikipedia.org/wiki/Pearson_correlation_coefficient
"""

Alternative = Literal["two-sided", "less", "greater"]


@evaluate.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class Pearson(HFMetric):
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
            reference_urls=[
                "https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pearsonr.html",
            ],
        )

    def _score(
        self,
        *,
        predictions: ArrayLike,
        references: ArrayLike,
        alternative: Alternative = "two-sided",
        **kwargs: Any,
    ) -> MetricResult:
        correlation, p_value = scipy.stats.pearsonr(
            x=predictions,
            y=references,
            alternative=alternative,
        )
        return {
            f"{self.tag}::correlation": correlation,
            f"{self.tag}::p_value": p_value,
        }
