"""Spearman correlation coefficient score function."""

import logging
from typing import Any, Literal, Optional

import evaluate
import scipy.stats

import datasets
from molflux.metrics.bases import HFMetric
from molflux.metrics.typing import ArrayLike, MetricResult

logger = logging.getLogger(__name__)

_DESCRIPTION = """
Calculate a Spearman correlation coefficient with associated p-value.

The Spearman rank-order correlation coefficient is a nonparametric measure of
the monotonicity of the relationship between two datasets. Unlike the Pearson
correlation, the Spearman correlation does not assume that both datasets are
normally distributed. Like other correlation coefficients, this one varies
between -1 and +1 with 0 implying no correlation. Correlations of -1 or +1 imply
an exact monotonic relationship. Positive correlations imply that as x increases,
so does y. Negative correlations imply that as x increases, y decreases.

The p-value roughly indicates the probability of an uncorrelated system producing
datasets that have a Spearman correlation at least as extreme as the one computed
from these datasets. The p-values are not entirely reliable but are probably
reasonable for datasets larger than 500 or so.
"""

_KWARGS_DESCRIPTION = """
Args:
    predictions: Estimated target values.
    references: Ground truth (correct) target values.
    axis (optional): The axis to evaluate the correlation against.
        If axis=0 (default), then each column represents a variable, with
        observations in the rows. If axis=1, the relationship is transposed:
        each row represents a variable, while the columns contain observations.
        If axis=None, then both arrays will be raveled.
    nan_policy (optional): Defines aggregating of multiple output scores.
        The following options are available (default is 'propagate'):

        * 'propagate': returns nan
        * 'raise': throws an error
        * 'omit': performs the calculations ignoring nan values

    alternative (optional): Defines the alternative hypothesis. Default is 'two-sided'.
        The following options are available:

        * 'two-sided': the correlation is nonzero
        * 'less': the correlation is negative (less than zero)
        * 'greater':  the correlation is positive (greater than zero)

Returns:
    The Spearman correlation matrix or correlation coefficient (if only 2
    variables are given as parameters. Correlation matrix is square with
    length equal to total number of variables (columns or rows) in ``predictions``
    and ``references`` combined.

    The p-value for a hypothesis test whose null hypotheisis
    is that two sets of data are uncorrelated. See `alternative` above
    for alternative hypotheses. `pvalue` has the same
    shape as `correlation`.

Examples:
    >>> from molflux.metrics import load_metric
    >>> metric = load_metric("spearman")
    >>> predictions = [1, 2, 3, 4, 5]
    >>> references = [5, 6, 7, 8, 7]
    >>> metric.compute(predictions=predictions, references=references)
    {'spearman::correlation': 0.82078..., 'spearman::p_value': 0.08858...}
"""

_CITATION = """\
Zwillinger, D. and Kokoska, S. (2000). CRC Standard Probability and Statistics
Tables and Formulae. Chapman & Hall: New York. 2000. Section 14.7
"""

NanPolicy = Literal["propagate", "raise", "omit"]
Alternative = Literal["two-sided", "less", "greater"]


@evaluate.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class Spearman(HFMetric):
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
                "https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.spearmanr.html",
            ],
        )

    def _score(
        self,
        *,
        predictions: ArrayLike,
        references: ArrayLike,
        axis: Optional[int] = 0,
        nan_policy: NanPolicy = "propagate",
        alternative: Alternative = "two-sided",
        **kwargs: Any,
    ) -> MetricResult:
        correlation, p_value = scipy.stats.spearmanr(
            a=predictions,
            b=references,
            axis=axis,
            nan_policy=nan_policy,
            alternative=alternative,
        )
        return {
            f"{self.tag}::correlation": correlation,
            f"{self.tag}::p_value": p_value,
        }
